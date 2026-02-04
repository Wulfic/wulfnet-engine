// =============================================================================
// WulfNet Engine - Rigid Body World Implementation
// =============================================================================

#include "RigidBodyWorld.h"
#include "core/math/MathUtils.h"

namespace WulfNet {

RigidBodyWorld::RigidBodyWorld(const RigidBodyWorldConfig& config)
    : m_gravity(config.gravity)
{
    if (config.broadphaseBackend == RigidBodyWorldConfig::BroadphaseBackend::GpuSpatialHash) {
        m_broadphase = std::make_unique<GpuSpatialHashBroadphase>(config.broadphaseCellSize);
        WULFNET_LOG_INFO("RigidBodyWorld", "Initialized (gravity=(%.2f,%.2f,%.2f), cellSize=%.2f, broadphase=GpuSpatialHash)",
            m_gravity.x, m_gravity.y, m_gravity.z, config.broadphaseCellSize);
    } else {
        m_broadphase = std::make_unique<SpatialHashBroadphase>(config.broadphaseCellSize);
        WULFNET_LOG_INFO("RigidBodyWorld", "Initialized (gravity=(%.2f,%.2f,%.2f), cellSize=%.2f, broadphase=SpatialHashCPU)",
            m_gravity.x, m_gravity.y, m_gravity.z, config.broadphaseCellSize);
    }
}

RigidBodyHandle RigidBodyWorld::createBody(const RigidBodyConfig& config) {
    u32 id = allocateId();
    BodyEntry& entry = m_bodies[id - 1];

    entry.body = std::make_unique<RigidBody>(config);
    entry.layer = config.collisionLayer;
    entry.active = true;

    if (!config.shape) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Body %u created without collision shape", id);
    }

    entry.proxyId = m_broadphase->createProxy(entry.body->getWorldAABB(), entry.layer, entry.body.get());
    m_proxyToBodyId[entry.proxyId] = id;
    ++m_bodyCount;

    WULFNET_LOG_TRACE("RigidBodyWorld", "Created body %u (proxy %u)", id, entry.proxyId);
    return RigidBodyHandle{ id };
}

void RigidBodyWorld::destroyBody(RigidBodyHandle handle) {
    if (!handle.isValid()) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy invalid body handle");
        return;
    }

    BodyEntry* entry = getEntry(handle.value);
    if (!entry || !entry->active) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy non-existent body {}", handle.value);
        return;
    }

    m_broadphase->destroyProxy(entry->proxyId);
    m_proxyToBodyId.erase(entry->proxyId);
    entry->body.reset();
    entry->active = false;
    entry->proxyId = 0;
    entry->layer = CollisionLayer::Default;
    --m_bodyCount;

    freeId(handle.value);
    WULFNET_LOG_TRACE("RigidBodyWorld", "Destroyed body {}", handle.value);
}

RigidBody* RigidBodyWorld::getBody(RigidBodyHandle handle) {
    BodyEntry* entry = getEntry(handle.value);
    return (entry && entry->active) ? entry->body.get() : nullptr;
}

const RigidBody* RigidBodyWorld::getBody(RigidBodyHandle handle) const {
    const BodyEntry* entry = getEntry(handle.value);
    return (entry && entry->active) ? entry->body.get() : nullptr;
}

void RigidBodyWorld::step(f32 dt) {
    if (dt <= 0.0f) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Step called with non-positive dt: {:.6f}", dt);
        return;
    }

    // Integrate bodies
    for (BodyEntry& entry : m_bodies) {
        if (!entry.active || !entry.body) continue;

        RigidBody& body = *entry.body;
        if (body.isDynamic() && body.usesGravity() && body.isAwake()) {
            body.applyForce(m_gravity * body.getMass());
        }

        body.integrate(dt);
        
        // Only update broadphase for awake bodies (optimization)
        if (body.isAwake() || body.isKinematic()) {
            updateBroadphaseProxy(entry);
        }
    }

    // Broadphase overlap pairs
    m_broadphase->findOverlappingPairs(m_pairs);

    m_contacts.clear();
    m_contacts.reserve(m_pairs.size());

    for (const CollisionPair& pair : m_pairs) {
        auto itA = m_proxyToBodyId.find(pair.bodyIdA);
        auto itB = m_proxyToBodyId.find(pair.bodyIdB);
        if (itA == m_proxyToBodyId.end() || itB == m_proxyToBodyId.end()) {
            continue;
        }

        BodyEntry* entryA = getEntry(itA->second);
        BodyEntry* entryB = getEntry(itB->second);
        if (!entryA || !entryB || !entryA->active || !entryB->active) {
            continue;
        }

        // Wake propagation: if one body is awake and dynamic, wake the other
        RigidBody& bodyA = *entryA->body;
        RigidBody& bodyB = *entryB->body;
        
        bool aIsMoving = bodyA.isAwake() && (bodyA.isDynamic() || bodyA.isKinematic());
        bool bIsMoving = bodyB.isAwake() && (bodyB.isDynamic() || bodyB.isKinematic());
        
        if (aIsMoving && !bodyB.isAwake() && bodyB.isDynamic()) {
            bodyB.wakeUp();
        }
        if (bIsMoving && !bodyA.isAwake() && bodyA.isDynamic()) {
            bodyA.wakeUp();
        }

        ContactManifold manifold;
        if (m_narrowphase.generateContacts(bodyA, bodyB, manifold)) {
            manifold.bodyIdA = itA->second;
            manifold.bodyIdB = itB->second;

            u32 aId = manifold.bodyIdA;
            u32 bId = manifold.bodyIdB;
            if (aId > bId) std::swap(aId, bId);
            u64 key = (static_cast<u64>(aId) << 32) | bId;

            auto cacheIt = m_manifoldCache.find(key);
            if (cacheIt != m_manifoldCache.end() && manifold.contactCount > 0) {
                const ContactManifold& cached = cacheIt->second;
                if (cached.contactCount > 0) {
                    constexpr f32 kMatchDistance = 0.05f; // 5cm match radius
                    constexpr f32 kMatchDistanceSq = kMatchDistance * kMatchDistance;
                    constexpr f32 kNormalDotThreshold = 0.9f;

                    bool used[ContactManifold::MAX_CONTACTS] = { false, false, false, false };

                    for (u32 i = 0; i < manifold.contactCount; ++i) {
                        ContactPoint& current = manifold.contacts[i];
                        int bestIdx = -1;
                        f32 bestScore = kMatchDistanceSq * 2.0f;

                        for (u32 j = 0; j < cached.contactCount; ++j) {
                            if (used[j]) continue;

                            const ContactPoint& prev = cached.contacts[j];
                            if (current.normal.dot(prev.normal) < kNormalDotThreshold) continue;

                            f32 distA = (current.positionWorldA - prev.positionWorldA).lengthSq();
                            f32 distB = (current.positionWorldB - prev.positionWorldB).lengthSq();
                            if (distA > kMatchDistanceSq || distB > kMatchDistanceSq) continue;

                            f32 score = distA + distB;
                            if (score < bestScore) {
                                bestScore = score;
                                bestIdx = static_cast<int>(j);
                            }
                        }

                        if (bestIdx >= 0) {
                            const ContactPoint& prev = cached.contacts[bestIdx];
                            current.normalImpulse = prev.normalImpulse;
                            current.tangentImpulse = prev.tangentImpulse;
                            current.tangent = prev.tangent;
                            used[bestIdx] = true;
                        }
                    }
                }
            }

            m_contacts.push_back(manifold);
        }
    }

    m_contactSolver.solve(m_contacts,
        [this](u32 id) { return getBody(RigidBodyHandle{ id }); },
        dt);

    if (m_distanceJointCount > 0) {
        std::vector<DistanceJoint*> activeJoints;
        activeJoints.reserve(m_distanceJointCount);
        for (DistanceJointEntry& entry : m_distanceJoints) {
            if (!entry.active) continue;
            activeJoints.push_back(&entry.joint);
        }

        m_jointSolver.solveDistanceJoints(activeJoints,
            [this](u32 id) { return getBody(RigidBodyHandle{ id }); },
            dt);
    }

    if (m_ballJointCount > 0) {
        std::vector<BallJoint*> activeJoints;
        activeJoints.reserve(m_ballJointCount);
        for (BallJointEntry& entry : m_ballJoints) {
            if (!entry.active) continue;
            activeJoints.push_back(&entry.joint);
        }

        m_jointSolver.solveBallJoints(activeJoints,
            [this](u32 id) { return getBody(RigidBodyHandle{ id }); },
            dt);
    }

    if (m_fixedJointCount > 0) {
        std::vector<FixedJoint*> activeJoints;
        activeJoints.reserve(m_fixedJointCount);
        for (FixedJointEntry& entry : m_fixedJoints) {
            if (!entry.active) continue;
            activeJoints.push_back(&entry.joint);
        }

        m_jointSolver.solveFixedJoints(activeJoints,
            [this](u32 id) { return getBody(RigidBodyHandle{ id }); },
            dt);
    }

    if (m_hingeJointCount > 0) {
        std::vector<HingeJoint*> activeJoints;
        activeJoints.reserve(m_hingeJointCount);
        for (HingeJointEntry& entry : m_hingeJoints) {
            if (!entry.active) continue;
            activeJoints.push_back(&entry.joint);
        }

        m_jointSolver.solveHingeJoints(activeJoints,
            [this](u32 id) { return getBody(RigidBodyHandle{ id }); },
            dt);
    }

    m_manifoldCache.clear();
    m_manifoldCache.reserve(m_contacts.size());
    for (const ContactManifold& manifold : m_contacts) {
        u32 aId = manifold.bodyIdA;
        u32 bId = manifold.bodyIdB;
        if (aId > bId) std::swap(aId, bId);
        u64 key = (static_cast<u64>(aId) << 32) | bId;
        m_manifoldCache[key] = manifold;
    }

    WULFNET_LOG_TRACE("RigidBodyWorld", "Step complete: bodies={}/{} awake, pairs={}, contacts={}",
        getAwakeBodyCount(), m_bodyCount, m_pairs.size(), m_contacts.size());
}

DistanceJointHandle RigidBodyWorld::createDistanceJoint(const DistanceJointConfig& config) {
    if (config.bodyIdA == 0 || config.bodyIdB == 0 || config.bodyIdA == config.bodyIdB) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Invalid distance joint body IDs: {} and {}", config.bodyIdA, config.bodyIdB);
        return DistanceJointHandle::invalid();
    }

    u32 id = allocateJointId();
    DistanceJointEntry& entry = m_distanceJoints[id - 1];

    entry.joint.id = id;
    entry.joint.bodyIdA = config.bodyIdA;
    entry.joint.bodyIdB = config.bodyIdB;
    entry.joint.localAnchorA = config.localAnchorA;
    entry.joint.localAnchorB = config.localAnchorB;
    entry.joint.stiffness = Math::clamp(config.stiffness, 0.0f, 1.0f);
    entry.joint.damping = Math::clamp(config.damping, 0.0f, 1.0f);

    f32 restLength = config.restLength;
    if (restLength < 0.0f) {
        RigidBody* bodyA = getBody(RigidBodyHandle{ config.bodyIdA });
        RigidBody* bodyB = getBody(RigidBodyHandle{ config.bodyIdB });
        if (bodyA && bodyB) {
            Vec3 worldA = bodyA->getTransform().transformPoint(config.localAnchorA);
            Vec3 worldB = bodyB->getTransform().transformPoint(config.localAnchorB);
            restLength = (worldB - worldA).length();
        } else {
            restLength = 0.0f;
        }
    }
    entry.joint.restLength = restLength;
    entry.joint.accumulatedImpulse = 0.0f;
    entry.active = true;

    ++m_distanceJointCount;
    WULFNET_LOG_TRACE("RigidBodyWorld", "Created distance joint {} between bodies {} and {}", id, config.bodyIdA, config.bodyIdB);
    return DistanceJointHandle{ id };
}

void RigidBodyWorld::destroyDistanceJoint(DistanceJointHandle handle) {
    if (!handle.isValid()) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy invalid distance joint handle");
        return;
    }

    u32 id = handle.value;
    if (id == 0 || id > m_distanceJoints.size()) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy non-existent distance joint {}", id);
        return;
    }

    DistanceJointEntry& entry = m_distanceJoints[id - 1];
    if (!entry.active) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy inactive distance joint {}", id);
        return;
    }

    entry.active = false;
    entry.joint = DistanceJoint{};
    freeJointId(id);
    --m_distanceJointCount;

    WULFNET_LOG_TRACE("RigidBodyWorld", "Destroyed distance joint {}", id);
}

BallJointHandle RigidBodyWorld::createBallJoint(const BallJointConfig& config) {
    if (config.bodyIdA == 0 || config.bodyIdB == 0 || config.bodyIdA == config.bodyIdB) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Invalid ball joint body IDs: {} and {}", config.bodyIdA, config.bodyIdB);
        return BallJointHandle::invalid();
    }

    u32 id = allocateBallJointId();
    BallJointEntry& entry = m_ballJoints[id - 1];

    entry.joint.id = id;
    entry.joint.bodyIdA = config.bodyIdA;
    entry.joint.bodyIdB = config.bodyIdB;
    entry.joint.localAnchorA = config.localAnchorA;
    entry.joint.localAnchorB = config.localAnchorB;
    entry.joint.stiffness = Math::clamp(config.stiffness, 0.0f, 1.0f);
    entry.joint.damping = Math::clamp(config.damping, 0.0f, 1.0f);
    entry.joint.accumulatedImpulse = Vec3::zero();
    entry.active = true;

    ++m_ballJointCount;
    WULFNET_LOG_TRACE("RigidBodyWorld", "Created ball joint {} between bodies {} and {}", id, config.bodyIdA, config.bodyIdB);
    return BallJointHandle{ id };
}

void RigidBodyWorld::destroyBallJoint(BallJointHandle handle) {
    if (!handle.isValid()) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy invalid ball joint handle");
        return;
    }

    u32 id = handle.value;
    if (id == 0 || id > m_ballJoints.size()) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy non-existent ball joint {}", id);
        return;
    }

    BallJointEntry& entry = m_ballJoints[id - 1];
    if (!entry.active) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy inactive ball joint {}", id);
        return;
    }

    entry.active = false;
    entry.joint = BallJoint{};
    freeBallJointId(id);
    --m_ballJointCount;

    WULFNET_LOG_TRACE("RigidBodyWorld", "Destroyed ball joint {}", id);
}

FixedJointHandle RigidBodyWorld::createFixedJoint(const FixedJointConfig& config) {
    if (config.bodyIdA == 0 || config.bodyIdB == 0 || config.bodyIdA == config.bodyIdB) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Invalid fixed joint body IDs: {} and {}", config.bodyIdA, config.bodyIdB);
        return FixedJointHandle::invalid();
    }

    u32 id = allocateFixedJointId();
    FixedJointEntry& entry = m_fixedJoints[id - 1];

    entry.joint.id = id;
    entry.joint.bodyIdA = config.bodyIdA;
    entry.joint.bodyIdB = config.bodyIdB;
    entry.joint.localAnchorA = config.localAnchorA;
    entry.joint.localAnchorB = config.localAnchorB;
    entry.joint.stiffness = Math::clamp(config.stiffness, 0.0f, 1.0f);
    entry.joint.damping = Math::clamp(config.damping, 0.0f, 1.0f);
    entry.joint.accumulatedLinearImpulse = Vec3::zero();
    entry.joint.accumulatedAngularImpulse = Vec3::zero();

    Quat restRel = config.restRelativeRotation;
    if (config.computeRestRotation) {
        RigidBody* bodyA = getBody(RigidBodyHandle{ config.bodyIdA });
        RigidBody* bodyB = getBody(RigidBodyHandle{ config.bodyIdB });
        if (bodyA && bodyB) {
            restRel = bodyA->getOrientation().conjugate() * bodyB->getOrientation();
        }
    }
    entry.joint.restRelativeRotation = restRel;
    entry.active = true;

    ++m_fixedJointCount;
    WULFNET_LOG_TRACE("RigidBodyWorld", "Created fixed joint {} between bodies {} and {}", id, config.bodyIdA, config.bodyIdB);
    return FixedJointHandle{ id };
}

void RigidBodyWorld::destroyFixedJoint(FixedJointHandle handle) {
    if (!handle.isValid()) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy invalid fixed joint handle");
        return;
    }

    u32 id = handle.value;
    if (id == 0 || id > m_fixedJoints.size()) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy non-existent fixed joint {}", id);
        return;
    }

    FixedJointEntry& entry = m_fixedJoints[id - 1];
    if (!entry.active) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy inactive fixed joint {}", id);
        return;
    }

    entry.active = false;
    entry.joint = FixedJoint{};
    freeFixedJointId(id);
    --m_fixedJointCount;

    WULFNET_LOG_TRACE("RigidBodyWorld", "Destroyed fixed joint {}", id);
}

HingeJointHandle RigidBodyWorld::createHingeJoint(const HingeJointConfig& config) {
    if (config.bodyIdA == 0 || config.bodyIdB == 0 || config.bodyIdA == config.bodyIdB) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Invalid hinge joint body IDs: {} and {}", config.bodyIdA, config.bodyIdB);
        return HingeJointHandle::invalid();
    }

    u32 id = allocateHingeJointId();
    HingeJointEntry& entry = m_hingeJoints[id - 1];

    entry.joint.id = id;
    entry.joint.bodyIdA = config.bodyIdA;
    entry.joint.bodyIdB = config.bodyIdB;
    entry.joint.localAnchorA = config.localAnchorA;
    entry.joint.localAnchorB = config.localAnchorB;
    entry.joint.localAxisA = config.localAxisA.normalized();
    entry.joint.localAxisB = config.localAxisB.normalized();
    entry.joint.stiffness = Math::clamp(config.stiffness, 0.0f, 1.0f);
    entry.joint.damping = Math::clamp(config.damping, 0.0f, 1.0f);
    entry.joint.accumulatedLinearImpulse = Vec3::zero();
    entry.joint.accumulatedAngularImpulse = Vec3::zero();
    entry.active = true;

    ++m_hingeJointCount;
    WULFNET_LOG_TRACE("RigidBodyWorld", "Created hinge joint {} between bodies {} and {}", id, config.bodyIdA, config.bodyIdB);
    return HingeJointHandle{ id };
}

void RigidBodyWorld::destroyHingeJoint(HingeJointHandle handle) {
    if (!handle.isValid()) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy invalid hinge joint handle");
        return;
    }

    u32 id = handle.value;
    if (id == 0 || id > m_hingeJoints.size()) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy non-existent hinge joint {}", id);
        return;
    }

    HingeJointEntry& entry = m_hingeJoints[id - 1];
    if (!entry.active) {
        WULFNET_LOG_WARN("RigidBodyWorld", "Attempted to destroy inactive hinge joint {}", id);
        return;
    }

    entry.active = false;
    entry.joint = HingeJoint{};
    freeHingeJointId(id);
    --m_hingeJointCount;

    WULFNET_LOG_TRACE("RigidBodyWorld", "Destroyed hinge joint {}", id);
}

u32 RigidBodyWorld::allocateId() {
    if (!m_freeIds.empty()) {
        u32 id = m_freeIds.back();
        m_freeIds.pop_back();
        return id;
    }

    m_bodies.emplace_back();
    return static_cast<u32>(m_bodies.size());
}

void RigidBodyWorld::freeId(u32 id) {
    if (id == 0) return;
    m_freeIds.push_back(id);
}

u32 RigidBodyWorld::allocateJointId() {
    if (!m_freeJointIds.empty()) {
        u32 id = m_freeJointIds.back();
        m_freeJointIds.pop_back();
        return id;
    }

    m_distanceJoints.emplace_back();
    return static_cast<u32>(m_distanceJoints.size());
}

void RigidBodyWorld::freeJointId(u32 id) {
    if (id == 0) return;
    m_freeJointIds.push_back(id);
}

u32 RigidBodyWorld::allocateBallJointId() {
    if (!m_freeBallJointIds.empty()) {
        u32 id = m_freeBallJointIds.back();
        m_freeBallJointIds.pop_back();
        return id;
    }

    m_ballJoints.emplace_back();
    return static_cast<u32>(m_ballJoints.size());
}

void RigidBodyWorld::freeBallJointId(u32 id) {
    if (id == 0) return;
    m_freeBallJointIds.push_back(id);
}

u32 RigidBodyWorld::allocateFixedJointId() {
    if (!m_freeFixedJointIds.empty()) {
        u32 id = m_freeFixedJointIds.back();
        m_freeFixedJointIds.pop_back();
        return id;
    }

    m_fixedJoints.emplace_back();
    return static_cast<u32>(m_fixedJoints.size());
}

void RigidBodyWorld::freeFixedJointId(u32 id) {
    if (id == 0) return;
    m_freeFixedJointIds.push_back(id);
}

u32 RigidBodyWorld::allocateHingeJointId() {
    if (!m_freeHingeJointIds.empty()) {
        u32 id = m_freeHingeJointIds.back();
        m_freeHingeJointIds.pop_back();
        return id;
    }

    m_hingeJoints.emplace_back();
    return static_cast<u32>(m_hingeJoints.size());
}

void RigidBodyWorld::freeHingeJointId(u32 id) {
    if (id == 0) return;
    m_freeHingeJointIds.push_back(id);
}

RigidBodyWorld::BodyEntry* RigidBodyWorld::getEntry(u32 id) {
    if (id == 0 || id > m_bodies.size()) return nullptr;
    return &m_bodies[id - 1];
}

const RigidBodyWorld::BodyEntry* RigidBodyWorld::getEntry(u32 id) const {
    if (id == 0 || id > m_bodies.size()) return nullptr;
    return &m_bodies[id - 1];
}

void RigidBodyWorld::updateBroadphaseProxy(BodyEntry& entry) {
    if (!entry.body) return;

    const AABB& aabb = entry.body->getWorldAABB();
    if (entry.proxyId == 0) {
        entry.proxyId = m_broadphase->createProxy(aabb, entry.layer, entry.body.get());
        m_proxyToBodyId[entry.proxyId] = static_cast<u32>(&entry - m_bodies.data()) + 1;
        return;
    }

    m_broadphase->updateProxy(entry.proxyId, aabb);
}

size_t RigidBodyWorld::getAwakeBodyCount() const {
    size_t count = 0;
    for (const BodyEntry& entry : m_bodies) {
        if (entry.active && entry.body && entry.body->isAwake()) {
            ++count;
        }
    }
    return count;
}

} // namespace WulfNet
