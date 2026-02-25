// =============================================================================
// WulfNet Engine - Destruction Physics System
// =============================================================================
// Pre-fractured Voronoi destruction for rigid bodies. Integrates with Jolt
// Physics by converting intact bodies into fragment bodies upon fracture.
//
// Pipeline:
//   1. Pre-fracture: Generate Voronoi cells for a destructible body
//   2. Runtime: Evaluate stress/impulse at contact points
//   3. Fracture: If threshold exceeded, spawn fragment bodies in Jolt
//   4. Secondary: Fragments can fracture again (recursive)
//
// References:
//   - Müller et al. 2013 "Real Time Dynamic Fracture with Volumetric
//     Approximate Convex Decompositions"
//   - Voronoi tessellation for pre-computed fracture patterns
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

// Jolt includes
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>

namespace JPH {
    class PhysicsSystem;
    class BodyInterface;
}

namespace WulfNet {

// =============================================================================
// Fracture Pattern — Pre-computed Voronoi decomposition
// =============================================================================

struct VoronoiCell {
    // Center site (local space relative to body center)
    float centerX = 0.0f, centerY = 0.0f, centerZ = 0.0f;

    // Approximate convex hull as AABB (simplified for CPU performance)
    float minX = 0.0f, minY = 0.0f, minZ = 0.0f;
    float maxX = 0.0f, maxY = 0.0f, maxZ = 0.0f;

    // Physical properties
    float volume = 0.0f;       // Volume of this cell
    float mass = 0.0f;         // Mass = volume * density

    // Fragment state
    bool detached = false;     // Has this cell been fractured off?
};

struct FracturePattern {
    std::vector<VoronoiCell> cells;

    // The generating body's reference shape
    float boundMinX = 0.0f, boundMinY = 0.0f, boundMinZ = 0.0f;
    float boundMaxX = 0.0f, boundMaxY = 0.0f, boundMaxZ = 0.0f;

    float totalVolume = 0.0f;
    float density = 1000.0f;    // kg/m³

    uint32_t GetCellCount() const {
        return static_cast<uint32_t>(cells.size());
    }
};

// =============================================================================
// Destructible Body — Tracks a Jolt body that can be fractured
// =============================================================================

struct DestructibleBody {
    JPH::BodyID intactBodyId;          // Original body in Jolt

    // Fracture threshold
    float fractureThreshold = 1000.0f; // Impulse magnitude to trigger fracture (N·s)
    float stressThreshold = 1.0e6f;    // Stress to trigger fracture (Pa)

    // Pre-computed fracture pattern
    FracturePattern pattern;

    // State
    bool fractured = false;            // Has been fractured
    bool enabled = true;

    // Fragment tracking
    std::vector<JPH::BodyID> fragmentBodyIds;

    // Properties
    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;  // Cached position
    float quatX = 0.0f, quatY = 0.0f, quatZ = 0.0f, quatW = 1.0f;
    float mass = 1.0f;

    // Recursion
    uint32_t fractureLevel = 0;        // How many times re-fractured
    uint32_t maxFractureLevel = 2;     // Maximum recursive fractures
};

// =============================================================================
// Fracture Event — Emitted when a body fractures
// =============================================================================

struct FractureEvent {
    uint32_t destructibleIndex;        // Index into m_destructibles
    JPH::BodyID originalBodyId;        // The body that fractured
    float impactX, impactY, impactZ;   // World-space impact point
    float impulse;                     // Impact impulse magnitude
    uint32_t fragmentCount;            // Number of fragments generated
};

using FractureCallback = std::function<void(const FractureEvent&)>;

// =============================================================================
// Destruction System Configuration
// =============================================================================

struct DestructionConfig {
    // Voronoi generation
    uint32_t defaultCellCount = 8;      // Default Voronoi cells per body
    uint32_t maxCellCount = 64;         // Maximum cells per body

    // Fracture behavior
    float minFragmentMass = 0.1f;       // Minimum fragment mass (kg)
    float fragmentEjectionSpeed = 2.0f; // Speed added to fragments (m/s)
    float fragmentAngularSpeed = 5.0f;  // Random angular velocity (rad/s)

    // Performance limits
    uint32_t maxFragmentsPerFrame = 100;
    uint32_t maxTotalFragments = 5000;

    // Fragment cleanup
    float fragmentLifetime = 30.0f;     // Seconds before fragments fade
    bool enableSecondaryFracture = false;

    // Thresholds
    float globalImpulseScale = 1.0f;    // Scale applied to all impulse checks
};

// =============================================================================
// Destruction Statistics
// =============================================================================

struct DestructionStats {
    uint32_t totalDestructibles = 0;
    uint32_t fracturedBodies = 0;
    uint32_t activeFragments = 0;
    uint32_t totalFragmentsGenerated = 0;
    uint32_t fracturesThisFrame = 0;
    float evaluationTimeMs = 0.0f;
    float fractureTimeMs = 0.0f;
};

// =============================================================================
// Destruction System
// =============================================================================

class DestructionSystem {
public:
    DestructionSystem();
    ~DestructionSystem();

    // Non-copyable
    DestructionSystem(const DestructionSystem&) = delete;
    DestructionSystem& operator=(const DestructionSystem&) = delete;

    // =========================================================================
    // Initialization
    // =========================================================================

    bool Initialize(const DestructionConfig& config);
    void Shutdown();
    bool IsInitialized() const { return m_initialized; }

    // =========================================================================
    // Destructible Body Registration
    // =========================================================================

    /// Register a Jolt body as destructible
    /// @param bodyId The Jolt body to make destructible
    /// @param threshold Impulse threshold for fracture
    /// @param cellCount Number of Voronoi cells in fracture pattern
    /// @return Handle index for the destructible
    uint32_t AddDestructible(JPH::BodyID bodyId,
                             float threshold = 1000.0f,
                             uint32_t cellCount = 0);

    /// Get a destructible by handle
    DestructibleBody* GetDestructible(uint32_t handle);
    const DestructibleBody* GetDestructible(uint32_t handle) const;

    /// Remove a destructible
    void RemoveDestructible(uint32_t handle);

    uint32_t GetDestructibleCount() const {
        return static_cast<uint32_t>(m_destructibles.size());
    }

    // =========================================================================
    // Fracture Pattern Generation
    // =========================================================================

    /// Generate a Voronoi fracture pattern for a box-shaped body
    /// @param halfExtX/Y/Z Half extents of the body's bounding box
    /// @param cellCount Number of Voronoi cells
    /// @param density Material density (kg/m³)
    static FracturePattern GenerateBoxPattern(
        float halfExtX, float halfExtY, float halfExtZ,
        uint32_t cellCount, float density = 1000.0f);

    /// Generate a Voronoi fracture pattern for a sphere-shaped body
    static FracturePattern GenerateSpherePattern(
        float radius, uint32_t cellCount, float density = 1000.0f);

    // =========================================================================
    // Runtime Evaluation
    // =========================================================================

    /// Evaluate an impact on a destructible body
    /// @return true if body was fractured by this impact
    bool EvaluateImpact(uint32_t handle,
                        float impactX, float impactY, float impactZ,
                        float impulse);

    /// Step: sync body states, evaluate pending fractures
    void Step(float deltaTime, JPH::PhysicsSystem* joltPhysics = nullptr);

    // =========================================================================
    // Fracture Execution
    // =========================================================================

    /// Manually trigger fracture of a destructible body
    /// @return Number of fragments generated
    uint32_t Fracture(uint32_t handle,
                      float impactX, float impactY, float impactZ);

    // =========================================================================
    // Callback
    // =========================================================================

    void SetFractureCallback(FractureCallback callback) {
        m_fractureCallback = std::move(callback);
    }

    // =========================================================================
    // Access
    // =========================================================================

    const DestructionConfig& GetConfig() const { return m_config; }
    void SetConfig(const DestructionConfig& config) { m_config = config; }
    const DestructionStats& GetStats() const { return m_stats; }

    /// Get all active fragment body IDs (for rendering/cleanup)
    const std::vector<JPH::BodyID>& GetActiveFragments() const {
        return m_allFragments;
    }

private:
    // Voronoi helpers
    void GenerateVoronoiSites(float minX, float minY, float minZ,
                              float maxX, float maxY, float maxZ,
                              uint32_t count,
                              std::vector<VoronoiCell>& cells);

    void ComputeVoronoiVolumes(FracturePattern& pattern);

    // Simple deterministic RNG for reproducible fracture patterns
    uint32_t m_rngState = 12345;
    float RandomFloat();    // [0, 1)
    float RandomRange(float minVal, float maxVal);

    // Data
    DestructionConfig m_config;
    bool m_initialized = false;

    std::vector<DestructibleBody> m_destructibles;
    std::vector<JPH::BodyID> m_allFragments;

    FractureCallback m_fractureCallback;

    DestructionStats m_stats;
};

} // namespace WulfNet
