// =============================================================================
// WulfNet Engine - PhysicsWorld.cpp
// =============================================================================

#include "PhysicsWorld.h"
#include "../../Core/Logging/Logger.h"
#include "../../Core/Profiling/Profiler.h"

#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>

#include <thread>
#include <cstdarg>

namespace WulfNet {

// =============================================================================
// Jolt Trace/Assert Integration
// =============================================================================

static void JoltTraceImpl(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    WULFNET_TRACE("Jolt", buffer);
}

#ifdef JPH_ENABLE_ASSERTS
static bool JoltAssertFailedImpl(const char* expression, const char* message,
                                  const char* file, unsigned int line) {
    std::string msg = std::string(expression);
    if (message) {
        msg += " - ";
        msg += message;
    }
    WULFNET_ERROR("Jolt", msg + " at " + file + ":" + std::to_string(line));
    return true; // Break into debugger
}
#endif

// =============================================================================
// Default Layer Filters
// =============================================================================

class DefaultObjectLayerPairFilter : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override {
        switch (inObject1) {
            case Layers::NON_MOVING:
                return inObject2 == Layers::MOVING || inObject2 == Layers::DEBRIS;
            case Layers::MOVING:
                return inObject2 != Layers::SENSOR;
            case Layers::DEBRIS:
                return inObject2 == Layers::NON_MOVING || inObject2 == Layers::MOVING;
            case Layers::SENSOR:
                return inObject2 == Layers::MOVING;
            default:
                return false;
        }
    }
};

class DefaultBroadPhaseLayerInterface : public JPH::BroadPhaseLayerInterface {
public:
    DefaultBroadPhaseLayerInterface() {
        m_objectToBroadPhase[Layers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
        m_objectToBroadPhase[Layers::MOVING] = BroadPhaseLayers::MOVING;
        m_objectToBroadPhase[Layers::DEBRIS] = BroadPhaseLayers::MOVING;
        m_objectToBroadPhase[Layers::SENSOR] = BroadPhaseLayers::MOVING;
    }

    unsigned int GetNumBroadPhaseLayers() const override {
        return BroadPhaseLayers::NUM_LAYERS;
    }

    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override {
        JPH_ASSERT(inLayer < Layers::NUM_LAYERS);
        return m_objectToBroadPhase[inLayer];
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override {
        switch ((JPH::BroadPhaseLayer::Type)inLayer) {
            case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING: return "NON_MOVING";
            case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::MOVING: return "MOVING";
            default: return "INVALID";
        }
    }
#endif

private:
    JPH::BroadPhaseLayer m_objectToBroadPhase[Layers::NUM_LAYERS];
};

class DefaultObjectVsBroadPhaseLayerFilter : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override {
        switch (inLayer1) {
            case Layers::NON_MOVING:
                return inLayer2 == BroadPhaseLayers::MOVING;
            case Layers::MOVING:
            case Layers::DEBRIS:
            case Layers::SENSOR:
                return true;
            default:
                return false;
        }
    }
};

// =============================================================================
// Contact Listener Implementation
// =============================================================================

class PhysicsWorld::ContactListenerImpl : public JPH::ContactListener {
public:
    ContactListenerImpl(PhysicsWorld& world) : m_world(world) {}

    JPH::ValidateResult OnContactValidate(
        const JPH::Body& body1, const JPH::Body& body2,
        JPH::RVec3Arg baseOffset, const JPH::CollideShapeResult& collisionResult) override
    {
        return JPH::ValidateResult::AcceptAllContactsForThisBodyPair;
    }

    void OnContactAdded(
        const JPH::Body& body1, const JPH::Body& body2,
        const JPH::ContactManifold& manifold, JPH::ContactSettings& settings) override
    {
        if (m_world.m_onContactAdded) {
            ContactEvent event;
            event.body1 = body1.GetID();
            event.body2 = body2.GetID();
            event.contactPoint = manifold.mBaseOffset + manifold.GetWorldSpaceContactPointOn1(0);
            event.contactNormal = manifold.mWorldSpaceNormal;
            event.penetrationDepth = manifold.mPenetrationDepth;
            m_world.m_onContactAdded(event);
        }
    }

    void OnContactPersisted(
        const JPH::Body& body1, const JPH::Body& body2,
        const JPH::ContactManifold& manifold, JPH::ContactSettings& settings) override
    {
        // Can add persisted callback if needed
    }

    void OnContactRemoved(const JPH::SubShapeIDPair& subShapePair) override {
        if (m_world.m_onContactRemoved) {
            m_world.m_onContactRemoved(subShapePair.GetBody1ID(), subShapePair.GetBody2ID());
        }
    }

private:
    PhysicsWorld& m_world;
};

// =============================================================================
// Body Activation Listener Implementation
// =============================================================================

class PhysicsWorld::BodyActivationListenerImpl : public JPH::BodyActivationListener {
public:
    BodyActivationListenerImpl(PhysicsWorld& world) : m_world(world) {}

    void OnBodyActivated(const JPH::BodyID& bodyID, uint64_t userData) override {
        if (m_world.m_onBodyActivated) {
            m_world.m_onBodyActivated(bodyID);
        }
    }

    void OnBodyDeactivated(const JPH::BodyID& bodyID, uint64_t userData) override {
        if (m_world.m_onBodyDeactivated) {
            m_world.m_onBodyDeactivated(bodyID);
        }
    }

private:
    PhysicsWorld& m_world;
};

// =============================================================================
// PhysicsWorld Implementation
// =============================================================================

PhysicsWorld::PhysicsWorld() = default;

PhysicsWorld::~PhysicsWorld() {
    Shutdown();
}

PhysicsWorld::PhysicsWorld(PhysicsWorld&&) noexcept = default;
PhysicsWorld& PhysicsWorld::operator=(PhysicsWorld&&) noexcept = default;

void PhysicsWorld::RegisterJoltTypes() {
    // Only register once
    static bool registered = false;
    if (!registered) {
        JPH::RegisterDefaultAllocator();
        JPH::Trace = JoltTraceImpl;
        JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = JoltAssertFailedImpl;)

        if (!JPH::Factory::sInstance) {
            JPH::Factory::sInstance = new JPH::Factory();
        }
        JPH::RegisterTypes();
        registered = true;

        WULFNET_INFO("Physics", "Jolt Physics types registered");
    }
}

void PhysicsWorld::CreateLayerInterfaces() {
    m_broadPhaseLayerInterface = std::make_unique<DefaultBroadPhaseLayerInterface>();
    m_objectVsBroadPhaseLayerFilter = std::make_unique<DefaultObjectVsBroadPhaseLayerFilter>();
    m_objectLayerPairFilter = std::make_unique<DefaultObjectLayerPairFilter>();
}

bool PhysicsWorld::Initialize(const PhysicsWorldSettings& settings) {
    WULFNET_ZONE_NAMED("PhysicsWorld::Initialize");

    if (m_initialized) {
        WULFNET_WARNING("Physics", "PhysicsWorld already initialized");
        return false;
    }

    m_settings = settings;

    WULFNET_INFO("Physics", "Initializing PhysicsWorld...");
    WULFNET_DEBUG("Physics", "  Max Bodies: " + std::to_string(settings.maxBodies));
    WULFNET_DEBUG("Physics", "  Max Body Pairs: " + std::to_string(settings.maxBodyPairs));
    WULFNET_DEBUG("Physics", "  Max Contact Constraints: " + std::to_string(settings.maxContactConstraints));

    // Register Jolt types (only happens once)
    RegisterJoltTypes();

    // Create temp allocator
    m_tempAllocator = std::make_unique<JPH::TempAllocatorImpl>(settings.tempAllocatorSize);
    WULFNET_DEBUG("Physics", "  Temp Allocator: " + std::to_string(settings.tempAllocatorSize / (1024 * 1024)) + " MB");

    // Create job system
    uint32_t numThreads = settings.numThreads;
    if (numThreads == 0) {
        numThreads = std::max(1u, std::thread::hardware_concurrency() - 1);
    }
    m_jobSystem = std::make_unique<JPH::JobSystemThreadPool>(
        JPH::cMaxPhysicsJobs,
        JPH::cMaxPhysicsBarriers,
        static_cast<int>(numThreads)
    );
    WULFNET_DEBUG("Physics", "  Job System Threads: " + std::to_string(numThreads));

    // Create layer interfaces
    CreateLayerInterfaces();

    // Create physics system
    m_physicsSystem = std::make_unique<JPH::PhysicsSystem>();
    m_physicsSystem->Init(
        settings.maxBodies,
        settings.numBodyMutexes,
        settings.maxBodyPairs,
        settings.maxContactConstraints,
        *m_broadPhaseLayerInterface,
        *m_objectVsBroadPhaseLayerFilter,
        *m_objectLayerPairFilter
    );

    // Set gravity
    m_physicsSystem->SetGravity(settings.gravity);

    // Create and register listeners
    m_contactListener = std::make_unique<ContactListenerImpl>(*this);
    m_bodyActivationListener = std::make_unique<BodyActivationListenerImpl>(*this);
    m_physicsSystem->SetContactListener(m_contactListener.get());
    m_physicsSystem->SetBodyActivationListener(m_bodyActivationListener.get());

    m_initialized = true;
    WULFNET_INFO("Physics", "PhysicsWorld initialized successfully");

    return true;
}

void PhysicsWorld::Shutdown() {
    if (!m_initialized) {
        return;
    }

    WULFNET_INFO("Physics", "Shutting down PhysicsWorld...");

    // Clear listeners first
    if (m_physicsSystem) {
        m_physicsSystem->SetContactListener(nullptr);
        m_physicsSystem->SetBodyActivationListener(nullptr);
    }

    // Destroy in reverse order
    m_bodyActivationListener.reset();
    m_contactListener.reset();
    m_physicsSystem.reset();
    m_jobSystem.reset();
    m_tempAllocator.reset();
    m_objectLayerPairFilter.reset();
    m_objectVsBroadPhaseLayerFilter.reset();
    m_broadPhaseLayerInterface.reset();

    m_initialized = false;
    WULFNET_INFO("Physics", "PhysicsWorld shutdown complete");
}

JPH::EPhysicsUpdateError PhysicsWorld::Step(float deltaTime) {
    WULFNET_ZONE_NAMED("PhysicsWorld::Step");

    if (!m_initialized) {
        WULFNET_ERROR("Physics", "Cannot step uninitialized PhysicsWorld");
        return JPH::EPhysicsUpdateError::ManifoldCacheFull;
    }

    ManualTimer timer;
    timer.Start();

    // Step Jolt physics
    JPH::EPhysicsUpdateError error = m_physicsSystem->Update(
        deltaTime,
        m_settings.collisionSteps,
        m_tempAllocator.get(),
        m_jobSystem.get()
    );

    // Update statistics
    m_statistics.lastStepTimeMs = static_cast<float>(timer.ElapsedMilliseconds());
    m_statistics.numActiveBodies = m_physicsSystem->GetNumActiveBodies(JPH::EBodyType::RigidBody);
    m_statistics.numBodies = m_physicsSystem->GetNumBodies();

    if (error != JPH::EPhysicsUpdateError::None) {
        WULFNET_WARNING("Physics", "Physics update error: " + std::to_string(static_cast<int>(error)));
    }

    return error;
}

void PhysicsWorld::OptimizeBroadPhase() {
    WULFNET_ZONE_NAMED("PhysicsWorld::OptimizeBroadPhase");

    if (m_initialized) {
        m_physicsSystem->OptimizeBroadPhase();
        WULFNET_DEBUG("Physics", "Broadphase optimized");
    }
}

JPH::BodyInterface& PhysicsWorld::GetBodyInterface() {
    return m_physicsSystem->GetBodyInterface();
}

const JPH::BodyInterface& PhysicsWorld::GetBodyInterface() const {
    return m_physicsSystem->GetBodyInterface();
}

JPH::BodyInterface& PhysicsWorld::GetBodyInterfaceNoLock() {
    return m_physicsSystem->GetBodyInterfaceNoLock();
}

const JPH::BodyInterface& PhysicsWorld::GetBodyInterfaceNoLock() const {
    return m_physicsSystem->GetBodyInterfaceNoLock();
}

uint32_t PhysicsWorld::GetNumActiveBodies() const {
    return m_initialized ? m_physicsSystem->GetNumActiveBodies(JPH::EBodyType::RigidBody) : 0;
}

uint32_t PhysicsWorld::GetNumBodies() const {
    return m_initialized ? m_physicsSystem->GetNumBodies() : 0;
}

const JPH::BroadPhaseQuery& PhysicsWorld::GetBroadPhaseQuery() const {
    return m_physicsSystem->GetBroadPhaseQuery();
}

const JPH::NarrowPhaseQuery& PhysicsWorld::GetNarrowPhaseQuery() const {
    return m_physicsSystem->GetNarrowPhaseQuery();
}

void PhysicsWorld::AddConstraint(JPH::Constraint* constraint) {
    if (m_initialized && constraint) {
        m_physicsSystem->AddConstraint(constraint);
    }
}

void PhysicsWorld::RemoveConstraint(JPH::Constraint* constraint) {
    if (m_initialized && constraint) {
        m_physicsSystem->RemoveConstraint(constraint);
    }
}

void PhysicsWorld::SetGravity(const JPH::Vec3& gravity) {
    if (m_initialized) {
        m_physicsSystem->SetGravity(gravity);
    }
}

JPH::Vec3 PhysicsWorld::GetGravity() const {
    return m_initialized ? m_physicsSystem->GetGravity() : JPH::Vec3::sZero();
}

void PhysicsWorld::SetPhysicsSettings(const JPH::PhysicsSettings& settings) {
    if (m_initialized) {
        m_physicsSystem->SetPhysicsSettings(settings);
    }
}

const JPH::PhysicsSettings& PhysicsWorld::GetPhysicsSettings() const {
    return m_physicsSystem->GetPhysicsSettings();
}

void PhysicsWorld::SetContactAddedCallback(ContactAddedCallback callback) {
    m_onContactAdded = std::move(callback);
}

void PhysicsWorld::SetContactRemovedCallback(ContactRemovedCallback callback) {
    m_onContactRemoved = std::move(callback);
}

void PhysicsWorld::SetBodyActivatedCallback(BodyActivatedCallback callback) {
    m_onBodyActivated = std::move(callback);
}

void PhysicsWorld::SetBodyDeactivatedCallback(BodyDeactivatedCallback callback) {
    m_onBodyDeactivated = std::move(callback);
}

} // namespace WulfNet
