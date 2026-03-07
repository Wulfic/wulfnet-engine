// =============================================================================
// WulfNet Engine - MPM ↔ Rigid Body Coupling
// =============================================================================
// Bidirectional force exchange between MPM continuum particles and Jolt Physics
// rigid bodies. Implements:
//   - Particle → Body forces (pressure, drag, contact)
//   - Body → Particle velocity fields (solid velocity boundary)
//   - Signed distance function body representations
//   - Multi-body coupling with spatial hashing
//
// References:
//   - Stomakhin et al. 2013 "Augmented MPM for Phase-Change & Varied Materials"
//   - Hu et al. 2018 "A Moving Least Squares Material Point Method with
//     Displacement Discontinuity and Two-Way Rigid Body Coupling"
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

// Jolt includes
#include <WulfNet/Jolt/Jolt.h>
#include <WulfNet/Jolt/Physics/Body/BodyID.h>

namespace JPH {
    class PhysicsSystem;
    class BodyInterface;
    class BroadPhaseQuery;
    class NarrowPhaseQuery;
}

namespace WulfNet {

// Forward declarations
struct MPMParticle;
struct MPMMaterialParams;
class PhysicsWorld;

// =============================================================================
// Coupled Rigid Body — Tracks a Jolt body for MPM interaction
// =============================================================================

struct CoupledRigidBody {
    JPH::BodyID bodyId;

    // Cached body state (updated each coupling step)
    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;       // Center of mass
    float velX = 0.0f, velY = 0.0f, velZ = 0.0f;        // Linear velocity
    float angVelX = 0.0f, angVelY = 0.0f, angVelZ = 0.0f; // Angular velocity
    float quatX = 0.0f, quatY = 0.0f, quatZ = 0.0f, quatW = 1.0f; // Orientation

    // Shape approximation for SDF
    enum class ShapeType : uint32_t {
        Sphere = 0,
        Box = 1,
        Capsule = 2,
        Custom = 255
    };
    ShapeType shapeType = ShapeType::Sphere;
    float halfExtentX = 0.5f, halfExtentY = 0.5f, halfExtentZ = 0.5f;
    float radius = 0.5f;
    float height = 1.0f;    // For capsule

    // Physical properties
    float mass = 1.0f;
    float invMass = 1.0f;
    float friction = 0.3f;
    float restitution = 0.2f;
    bool isStatic = false;

    // Accumulated forces from MPM coupling (applied at end of step)
    float accForceX = 0.0f, accForceY = 0.0f, accForceZ = 0.0f;
    float accTorqueX = 0.0f, accTorqueY = 0.0f, accTorqueZ = 0.0f;
    uint32_t contactCount = 0;

    // Coupling parameters
    float couplingStrength = 1.0f;   // 0 = no coupling, 1 = full coupling
    bool enabled = true;

    void ClearAccumulators() {
        accForceX = accForceY = accForceZ = 0.0f;
        accTorqueX = accTorqueY = accTorqueZ = 0.0f;
        contactCount = 0;
    }
};

// =============================================================================
// Coupling Configuration
// =============================================================================

struct MPMCouplingConfig {
    // Coupling method
    float penaltyStiffness = 1.0e4f;    // Penalty force stiffness (N/m)
    float dampingCoefficient = 100.0f;    // Velocity damping (N·s/m)
    float frictionCoefficient = 0.3f;     // Coulomb friction

    // Interaction radius
    float interactionRadius = 0.05f;     // Distance threshold for coupling (m)
    float smoothingRadius = 0.1f;        // Kernel support radius (m)

    // Safety
    float maxCouplingForce = 1.0e6f;     // Cap per-particle force magnitude (N)
    float maxBodyForce = 1.0e8f;         // Cap total force on a single body (N)

    // Performance
    uint32_t maxParticlesPerBody = 10000; // Limit particles interacting per body
    bool useSpatialHash = true;           // Enable spatial hashing for lookups

    // Grid cell size for spatial hashing
    float hashCellSize = 0.2f;

    // Enable specific coupling directions
    bool enableParticleToBody = true;   // MPM → Jolt forces
    bool enableBodyToParticle = true;   // Jolt → MPM boundary conditions
    bool enableFriction = true;
};

// =============================================================================
// Coupling Statistics
// =============================================================================

struct MPMCouplingStats {
    uint32_t activeBodies = 0;
    uint32_t particleBodyContacts = 0;
    float maxForceApplied = 0.0f;
    float totalForceApplied = 0.0f;
    float couplingTimeMs = 0.0f;
    uint32_t spatialHashCollisions = 0;
};

// =============================================================================
// MPM Rigid Body Coupling System
// =============================================================================

class MPMRigidCoupling {
public:
    MPMRigidCoupling();
    ~MPMRigidCoupling();

    // =========================================================================
    // Initialization
    // =========================================================================

    bool Initialize(const MPMCouplingConfig& config);
    void Shutdown();
    bool IsInitialized() const { return m_initialized; }

    // =========================================================================
    // Body Registration
    // =========================================================================

    /// Register a Jolt rigid body for coupling with MPM particles
    /// @return Handle index for the coupled body
    uint32_t AddCoupledBody(JPH::BodyID bodyId,
                            CoupledRigidBody::ShapeType shape,
                            float radius = 0.5f,
                            float halfExtentX = 0.5f,
                            float halfExtentY = 0.5f,
                            float halfExtentZ = 0.5f);

    /// Remove a coupled body
    void RemoveCoupledBody(uint32_t handle);

    /// Get a coupled body by handle
    CoupledRigidBody* GetCoupledBody(uint32_t handle);
    const CoupledRigidBody* GetCoupledBody(uint32_t handle) const;

    /// Get number of coupled bodies
    uint32_t GetCoupledBodyCount() const { return static_cast<uint32_t>(m_bodies.size()); }

    // =========================================================================
    // Coupling Computation
    // =========================================================================

    /// Main coupling step: compute bidirectional forces between MPM particles
    /// and registered Jolt rigid bodies.
    /// Call this AFTER MPM P2G/G2P step and BEFORE applying results.
    ///
    /// @param particles      Array of MPM particles
    /// @param particleCount  Number of active particles
    /// @param params         Material parameters (for stress/stiffness info)
    /// @param joltPhysics    Jolt physics system (for body state queries)
    /// @param dt             Time step
    void ComputeCoupling(MPMParticle* particles,
                         uint32_t particleCount,
                         const MPMMaterialParams& params,
                         JPH::PhysicsSystem& joltPhysics,
                         float dt);

    /// Apply accumulated forces to Jolt bodies
    /// Call AFTER ComputeCoupling, during the Jolt step preparation
    void ApplyForcesToBodies(JPH::PhysicsSystem& joltPhysics);

    // =========================================================================
    // SDF Queries
    // =========================================================================

    /// Compute signed distance from a point to the nearest coupled body
    /// @param x, y, z  World position
    /// @param bodyIndex Output: index of nearest body (-1 if none)
    /// @param nx, ny, nz Output: surface normal (outward)
    /// @return Signed distance (negative = inside body)
    float QuerySDF(float x, float y, float z,
                   int32_t& bodyIndex,
                   float& nx, float& ny, float& nz) const;

    /// Compute the velocity of a body's surface at a given world point
    /// (linear + angular contribution)
    void GetBodySurfaceVelocity(uint32_t bodyIdx,
                                float px, float py, float pz,
                                float& vx, float& vy, float& vz) const;

    // =========================================================================
    // Access
    // =========================================================================

    const MPMCouplingConfig& GetConfig() const { return m_config; }
    void SetConfig(const MPMCouplingConfig& config) { m_config = config; }
    const MPMCouplingStats& GetStats() const { return m_stats; }

private:
    // Internal methods
    void SyncBodyStates(JPH::PhysicsSystem& joltPhysics);
    void ClearAccumulators();

    float ComputeBodySDF(const CoupledRigidBody& body,
                         float px, float py, float pz,
                         float& nx, float& ny, float& nz) const;

    void TransformToBodyLocal(const CoupledRigidBody& body,
                              float wx, float wy, float wz,
                              float& lx, float& ly, float& lz) const;

    void TransformNormalToWorld(const CoupledRigidBody& body,
                               float lnx, float lny, float lnz,
                               float& wnx, float& wny, float& wnz) const;

    void AccumulateForceOnBody(CoupledRigidBody& body,
                               float fx, float fy, float fz,
                               float px, float py, float pz);

    // Data
    MPMCouplingConfig m_config;
    bool m_initialized = false;

    std::vector<CoupledRigidBody> m_bodies;

    // Statistics
    MPMCouplingStats m_stats;
};

} // namespace WulfNet
