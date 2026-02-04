// =============================================================================
// WulfNet Engine - Rigid Body World
// =============================================================================
// Manages rigid bodies, integrates motion, and drives broadphase queries.
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/Log.h"
#include "core/Assert.h"
#include "core/math/Vec3.h"
#include "physics/dynamics/RigidBody.h"
#include "physics/collision/Broadphase.h"
#include "physics/collision/Narrowphase.h"
#include "physics/constraints/ContactSolver.h"
#include "physics/constraints/Joint.h"
#include <vector>
#include <memory>
#include <unordered_map>

namespace WulfNet {

// =============================================================================
// World Configuration
// =============================================================================

struct RigidBodyWorldConfig {
    Vec3 gravity = Vec3(0.0f, -9.81f, 0.0f);
    f32 broadphaseCellSize = 10.0f;
    enum class BroadphaseBackend : u8 {
        SpatialHashCPU = 0,
        GpuSpatialHash = 1
    } broadphaseBackend = BroadphaseBackend::SpatialHashCPU;
};

// =============================================================================
// Rigid Body World
// =============================================================================

class RigidBodyWorld : public NonCopyable {
public:
    explicit RigidBodyWorld(const RigidBodyWorldConfig& config = {});
    ~RigidBodyWorld() = default;

    // Body lifecycle
    RigidBodyHandle createBody(const RigidBodyConfig& config);
    void destroyBody(RigidBodyHandle handle);

    // Access
    RigidBody* getBody(RigidBodyHandle handle);
    const RigidBody* getBody(RigidBodyHandle handle) const;

    // Simulation
    void step(f32 dt);

    // Distance joints
    DistanceJointHandle createDistanceJoint(const DistanceJointConfig& config);
    void destroyDistanceJoint(DistanceJointHandle handle);

    // Ball joints
    BallJointHandle createBallJoint(const BallJointConfig& config);
    void destroyBallJoint(BallJointHandle handle);

    // Fixed joints
    FixedJointHandle createFixedJoint(const FixedJointConfig& config);
    void destroyFixedJoint(FixedJointHandle handle);

    // Hinge joints
    HingeJointHandle createHingeJoint(const HingeJointConfig& config);
    void destroyHingeJoint(HingeJointHandle handle);

    // Settings
    void setGravity(const Vec3& gravity) { m_gravity = gravity; }
    const Vec3& getGravity() const { return m_gravity; }

    // Stats
    size_t getBodyCount() const { return m_bodyCount; }
    size_t getAwakeBodyCount() const;
    const std::vector<CollisionPair>& getBroadphasePairs() const { return m_pairs; }
    const std::vector<ContactManifold>& getContactManifolds() const { return m_contacts; }

private:
    struct BodyEntry {
        std::unique_ptr<RigidBody> body;
        u32 proxyId = 0;
        CollisionLayer layer = CollisionLayer::Default;
        bool active = false;
    };

    struct DistanceJointEntry {
        DistanceJoint joint;
        bool active = false;
    };

    struct BallJointEntry {
        BallJoint joint;
        bool active = false;
    };

    struct FixedJointEntry {
        FixedJoint joint;
        bool active = false;
    };

    struct HingeJointEntry {
        HingeJoint joint;
        bool active = false;
    };

    u32 allocateId();
    void freeId(u32 id);

    u32 allocateJointId();
    void freeJointId(u32 id);

    u32 allocateBallJointId();
    void freeBallJointId(u32 id);

    u32 allocateFixedJointId();
    void freeFixedJointId(u32 id);

    u32 allocateHingeJointId();
    void freeHingeJointId(u32 id);

    BodyEntry* getEntry(u32 id);
    const BodyEntry* getEntry(u32 id) const;

    void updateBroadphaseProxy(BodyEntry& entry);

    Vec3 m_gravity;
    std::unique_ptr<IBroadphase> m_broadphase;
    Narrowphase m_narrowphase;
    ContactSolverPGS m_contactSolver;
    JointSolverPGS m_jointSolver;

    std::vector<BodyEntry> m_bodies;
    std::vector<u32> m_freeIds;
    std::unordered_map<u32, u32> m_proxyToBodyId;
    std::vector<CollisionPair> m_pairs;
    std::vector<ContactManifold> m_contacts;
    std::unordered_map<u64, ContactManifold> m_manifoldCache;
    size_t m_bodyCount = 0;

    std::vector<DistanceJointEntry> m_distanceJoints;
    std::vector<u32> m_freeJointIds;
    size_t m_distanceJointCount = 0;

    std::vector<BallJointEntry> m_ballJoints;
    std::vector<u32> m_freeBallJointIds;
    size_t m_ballJointCount = 0;

    std::vector<FixedJointEntry> m_fixedJoints;
    std::vector<u32> m_freeFixedJointIds;
    size_t m_fixedJointCount = 0;

    std::vector<HingeJointEntry> m_hingeJoints;
    std::vector<u32> m_freeHingeJointIds;
    size_t m_hingeJointCount = 0;
};

} // namespace WulfNet
