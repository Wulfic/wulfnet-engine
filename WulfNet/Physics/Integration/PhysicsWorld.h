// =============================================================================
// WulfNet Engine - PhysicsWorld.h
// =============================================================================
// High-level wrapper around Jolt's PhysicsSystem that provides a clean interface
// and integrates WulfNet's extended physics systems.
// =============================================================================

#pragma once

#include <memory>
#include <vector>
#include <functional>

// Jolt includes
#include <Jolt/Jolt.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceMask.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterMask.h>

#include "API.h"

namespace WulfNet {

// =============================================================================
// Forward Declarations
// =============================================================================

class MPMSystem;
class FluidSystem;
struct FluidSystemConfig;
class GaseousSystem;
struct GaseousSystemConfig;
class DestructionSystem;
struct DestructionConfig;
class TerrainDeformation;
struct TerrainDeformConfig;
class MPMRigidCoupling;
struct MPMCouplingConfig;

// =============================================================================
// Physics World Settings
// =============================================================================

struct PhysicsWorldSettings {
    // Body limits
    uint32_t maxBodies = 65536;
    uint32_t maxBodyPairs = 65536;
    uint32_t maxContactConstraints = 10240;
    uint32_t numBodyMutexes = 0; // 0 = auto-detect

    // Threading
    uint32_t numThreads = 0; // 0 = auto-detect (hardware_concurrency - 1)

    // Temp allocator size (bytes)
    uint32_t tempAllocatorSize = 32 * 1024 * 1024; // 32 MB

    // Physics settings
    JPH::Vec3 gravity = JPH::Vec3(0.0f, -9.81f, 0.0f);
    int collisionSteps = 1;

    // Extended physics (WulfNet)
    bool enableFluidPhysics = false;
    bool enableMPMPhysics = false;
    bool enableGaseousPhysics = false;
    bool enableDestructionPhysics = false;

    /// Validate configuration and return true if valid.
    bool Validate() const {
        if (maxBodies == 0) return false;
        if (maxBodyPairs == 0) return false;
        if (maxContactConstraints == 0) return false;
        if (tempAllocatorSize < 1024 * 1024) return false; // Minimum 1 MB
        if (collisionSteps <= 0) return false;
        return true;
    }
};

// =============================================================================
// Object Layers (Default Configuration)
// =============================================================================

namespace Layers {
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING = 1;
    static constexpr JPH::ObjectLayer DEBRIS = 2;
    static constexpr JPH::ObjectLayer SENSOR = 3;
    static constexpr JPH::ObjectLayer NUM_LAYERS = 4;
}

namespace BroadPhaseLayers {
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr uint32_t NUM_LAYERS = 2;
}

// =============================================================================
// Physics Event Types
// =============================================================================

struct ContactEvent {
    JPH::BodyID body1;
    JPH::BodyID body2;
    JPH::Vec3 contactPoint;
    JPH::Vec3 contactNormal;
    float penetrationDepth;
};

using ContactAddedCallback = std::function<void(const ContactEvent&)>;
using ContactRemovedCallback = std::function<void(JPH::BodyID, JPH::BodyID)>;
using BodyActivatedCallback = std::function<void(JPH::BodyID)>;
using BodyDeactivatedCallback = std::function<void(JPH::BodyID)>;

// =============================================================================
// Physics World Class
// =============================================================================

class WULFNET_API PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    // Non-copyable, moveable
    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;
    PhysicsWorld(PhysicsWorld&&) noexcept;
    PhysicsWorld& operator=(PhysicsWorld&&) noexcept;

    // ==========================================================================
    // Initialization
    // ==========================================================================

    /// Initialize the physics world with the given settings
    bool Initialize(const PhysicsWorldSettings& settings = PhysicsWorldSettings());

    /// Shutdown and cleanup all physics systems
    void Shutdown();

    /// Check if the world is initialized
    bool IsInitialized() const { return m_initialized; }

    // ==========================================================================
    // Simulation
    // ==========================================================================

    /// Step the simulation by deltaTime seconds
    /// @param deltaTime Time step in seconds (typically 1/60)
    /// @return Error code from Jolt (EPhysicsUpdateError::None on success)
    JPH::EPhysicsUpdateError Step(float deltaTime);

    /// Optimize the broadphase (call after adding many bodies at once)
    void OptimizeBroadPhase();

    // ==========================================================================
    // Body Management
    // ==========================================================================

    /// Get the Jolt body interface for creating/modifying bodies
    JPH::BodyInterface& GetBodyInterface();
    const JPH::BodyInterface& GetBodyInterface() const;

    /// Get the non-locking body interface (use with care!)
    JPH::BodyInterface& GetBodyInterfaceNoLock();
    const JPH::BodyInterface& GetBodyInterfaceNoLock() const;

    /// Get the number of active bodies
    uint32_t GetNumActiveBodies() const;

    /// Get the number of bodies
    uint32_t GetNumBodies() const;

    // ==========================================================================
    // Queries
    // ==========================================================================

    /// Get the broadphase query interface
    const JPH::BroadPhaseQuery& GetBroadPhaseQuery() const;

    /// Get the narrowphase query interface
    const JPH::NarrowPhaseQuery& GetNarrowPhaseQuery() const;

    // ==========================================================================
    // Constraints
    // ==========================================================================

    /// Add a constraint to the world
    void AddConstraint(JPH::Constraint* constraint);

    /// Remove a constraint from the world
    void RemoveConstraint(JPH::Constraint* constraint);

    // ==========================================================================
    // Settings
    // ==========================================================================

    /// Set gravity
    void SetGravity(const JPH::Vec3& gravity);
    JPH::Vec3 GetGravity() const;

    /// Access physics settings
    void SetPhysicsSettings(const JPH::PhysicsSettings& settings);
    const JPH::PhysicsSettings& GetPhysicsSettings() const;

    // ==========================================================================
    // Event Callbacks
    // ==========================================================================

    void SetContactAddedCallback(ContactAddedCallback callback);
    void SetContactRemovedCallback(ContactRemovedCallback callback);
    void SetBodyActivatedCallback(BodyActivatedCallback callback);
    void SetBodyDeactivatedCallback(BodyDeactivatedCallback callback);

    // ==========================================================================
    // Jolt Access (for advanced use)
    // ==========================================================================

    /// Direct access to Jolt's PhysicsSystem
    JPH::PhysicsSystem& GetJoltPhysics() { return *m_physicsSystem; }
    const JPH::PhysicsSystem& GetJoltPhysics() const { return *m_physicsSystem; }

    /// Access to Jolt's job system
    JPH::JobSystem& GetJobSystem() { return *m_jobSystem; }

    // ==========================================================================
    // Statistics
    // ==========================================================================

    struct Statistics {
        float lastStepTimeMs = 0.0f;
        uint32_t numActiveBodies = 0;
        uint32_t numBodies = 0;
        uint32_t numConstraints = 0;
        uint32_t numContacts = 0;
    };

    const Statistics& GetStatistics() const { return m_statistics; }

    // ==========================================================================
    // Extended Physics Systems (WulfNet)
    // ==========================================================================

    /// Get the fluid simulation system. Only valid if enableFluidPhysics was true.
    FluidSystem* GetFluidSystem() { return m_fluidSystem.get(); }
    const FluidSystem* GetFluidSystem() const { return m_fluidSystem.get(); }

    /// Get the gaseous simulation system. Only valid if enableGaseousPhysics was true.
    GaseousSystem* GetGaseousSystem() { return m_gaseousSystem.get(); }
    const GaseousSystem* GetGaseousSystem() const { return m_gaseousSystem.get(); }

    /// Get the destruction system. Only valid if enableDestructionPhysics was true.
    DestructionSystem* GetDestructionSystem() { return m_destructionSystem.get(); }
    const DestructionSystem* GetDestructionSystem() const { return m_destructionSystem.get(); }

    /// Get the terrain deformation system (always available if physics is initialized).
    TerrainDeformation* GetTerrainDeformation() { return m_terrainDeformation.get(); }
    const TerrainDeformation* GetTerrainDeformation() const { return m_terrainDeformation.get(); }

    /// Get the MPM-rigid coupling system. Only valid if enableMPMPhysics was true.
    MPMRigidCoupling* GetMPMCoupling() { return m_mpmCoupling.get(); }
    const MPMRigidCoupling* GetMPMCoupling() const { return m_mpmCoupling.get(); }

private:
    // Internal initialization
    void RegisterJoltTypes();
    void CreateLayerInterfaces();
    void InitExtendedSystems();
    void ShutdownExtendedSystems();
    void StepExtendedSystems(float deltaTime);

    // Jolt core systems
    std::unique_ptr<JPH::TempAllocator> m_tempAllocator;
    std::unique_ptr<JPH::JobSystemThreadPool> m_jobSystem;
    std::unique_ptr<JPH::PhysicsSystem> m_physicsSystem;

    // Layer interfaces (must outlive PhysicsSystem)
    std::unique_ptr<JPH::BroadPhaseLayerInterface> m_broadPhaseLayerInterface;
    std::unique_ptr<JPH::ObjectVsBroadPhaseLayerFilter> m_objectVsBroadPhaseLayerFilter;
    std::unique_ptr<JPH::ObjectLayerPairFilter> m_objectLayerPairFilter;

    // Internal listener implementations
    class ContactListenerImpl;
    class BodyActivationListenerImpl;
    std::unique_ptr<ContactListenerImpl> m_contactListener;
    std::unique_ptr<BodyActivationListenerImpl> m_bodyActivationListener;

    // User callbacks
    ContactAddedCallback m_onContactAdded;
    ContactRemovedCallback m_onContactRemoved;
    BodyActivatedCallback m_onBodyActivated;
    BodyDeactivatedCallback m_onBodyDeactivated;

    // Settings
    PhysicsWorldSettings m_settings;
    bool m_initialized = false;

    // Statistics
    Statistics m_statistics;

    // Extended physics systems
    std::unique_ptr<FluidSystem> m_fluidSystem;
    std::unique_ptr<GaseousSystem> m_gaseousSystem;
    std::unique_ptr<DestructionSystem> m_destructionSystem;
    std::unique_ptr<TerrainDeformation> m_terrainDeformation;
    std::unique_ptr<MPMRigidCoupling> m_mpmCoupling;
};

} // namespace WulfNet
