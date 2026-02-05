// =============================================================================
// WulfNet Engine - MPM Fluid System (CPU Reference + GPU Compute)
// =============================================================================
// Material Point Method (MPM) with APIC transfers for fluid simulation.
// Supports single-GPU game engine optimization with CPU fallback.
// =============================================================================

#pragma once

#include "FluidParticle.h"
#include "FluidGrid.h"
#include <vector>
#include <memory>
#include <functional>

namespace WulfNet {

// Forward declarations
class VulkanContext;
class GPUBuffer;
class ComputePipeline;

// =============================================================================
// Fluid System Configuration
// =============================================================================

struct FluidSystemConfig {
    // Grid settings
    uint32_t gridResolutionX = 128;
    uint32_t gridResolutionY = 64;
    uint32_t gridResolutionZ = 128;
    float cellSize = 0.05f;             // 5cm cells

    // Simulation bounds
    float boundsMinX = 0.0f, boundsMinY = 0.0f, boundsMinZ = 0.0f;
    float boundsMaxX = 6.4f, boundsMaxY = 3.2f, boundsMaxZ = 6.4f;

    // Particle settings
    uint32_t maxParticles = 1000000;    // 1M particles max
    uint32_t particlesPerCell = 8;       // Target particles per cell

    // Physics
    float gravity = -9.81f;
    float flipRatio = 0.95f;            // FLIP/PIC blend (0 = PIC, 1 = FLIP)
    uint32_t pressureIterations = 50;   // Jacobi iterations

    // Optimization
    bool useGPU = true;
    bool enableSleeping = true;
    float sleepThreshold = 0.001f;      // Velocity threshold
    bool enableSpatialHash = true;

    // Quality
    uint32_t substeps = 2;
    float maxTimestep = 0.016f;         // 60fps max
};

// =============================================================================
// Fluid Emitter (spawn particles)
// =============================================================================

enum class EmitterType {
    Point,      // Single point
    Sphere,     // Spherical volume
    Box,        // Box volume
    Plane,      // Infinite plane (for rivers)
    Mesh        // From mesh surface
};

struct FluidEmitter {
    EmitterType type = EmitterType::Point;

    // Position and orientation
    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
    float dirX = 0.0f, dirY = -1.0f, dirZ = 0.0f;

    // Size
    float radius = 0.5f;        // For sphere
    float sizeX = 1.0f, sizeY = 1.0f, sizeZ = 1.0f;  // For box

    // Emission
    float emissionRate = 1000.0f;   // Particles per second
    float initialSpeed = 2.0f;
    float speedVariance = 0.1f;

    // Material
    uint32_t materialId = 0;

    // State
    bool enabled = true;
    float accumulatedTime = 0.0f;
    uint32_t particlesEmitted = 0;
};

// =============================================================================
// Fluid Collider (for boundary handling)
// =============================================================================

enum class ColliderType {
    Plane,
    Sphere,
    Box,
    Capsule,
    Mesh,
    HeightField
};

struct FluidCollider {
    ColliderType type = ColliderType::Box;

    // Transform
    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
    float rotX = 0.0f, rotY = 0.0f, rotZ = 0.0f, rotW = 1.0f;
    float scaleX = 1.0f, scaleY = 1.0f, scaleZ = 1.0f;

    // Shape parameters
    float radius = 0.5f;
    float height = 1.0f;

    // Friction/bounce
    float friction = 0.3f;
    float restitution = 0.1f;

    // Velocity (for moving objects)
    float velX = 0.0f, velY = 0.0f, velZ = 0.0f;
    float angVelX = 0.0f, angVelY = 0.0f, angVelZ = 0.0f;

    // Flags
    bool enabled = true;
    bool isTwoSided = false;    // For planes
};

// =============================================================================
// Buoyancy Object (for floating/sinking)
// =============================================================================

struct BuoyancyObject {
    uint32_t bodyId;            // Jolt physics body ID
    float density = 500.0f;     // Object density (kg/m³)
    float volume = 1.0f;        // Object volume (m³)
    float dragCoefficient = 0.5f;
    float submergedVolume = 0.0f;   // Computed each frame

    // Callbacks for Jolt integration
    // Force to apply will be computed by system
    float forceX = 0.0f, forceY = 0.0f, forceZ = 0.0f;
    float torqueX = 0.0f, torqueY = 0.0f, torqueZ = 0.0f;
};

// =============================================================================
// Fluid Statistics
// =============================================================================

struct FluidStats {
    uint32_t activeParticles = 0;
    uint32_t sleepingParticles = 0;
    uint32_t surfaceParticles = 0;
    uint32_t sprayParticles = 0;

    float averageVelocity = 0.0f;
    float maxVelocity = 0.0f;
    float totalKineticEnergy = 0.0f;
    float totalPotentialEnergy = 0.0f;

    float p2gTimeMs = 0.0f;     // Particle to grid
    float gridSolveTimeMs = 0.0f;
    float g2pTimeMs = 0.0f;     // Grid to particle
    float collisionTimeMs = 0.0f;
    float totalTimeMs = 0.0f;

    uint32_t gridCellsUsed = 0;
    float averageDensity = 0.0f;
};

// =============================================================================
// MPM Fluid System
// =============================================================================

class FluidSystem {
public:
    FluidSystem();
    ~FluidSystem();

    // Initialization
    bool Initialize(const FluidSystemConfig& config);
    void Shutdown();
    bool IsInitialized() const { return m_initialized; }

    // GPU setup (optional)
    bool InitializeGPU(VulkanContext* context);
    bool IsGPUEnabled() const { return m_gpuEnabled; }

    // Materials
    uint32_t AddMaterial(const FluidMaterial& material);
    FluidMaterial* GetMaterial(uint32_t id);
    size_t GetMaterialCount() const { return m_materials.size(); }

    // Emitters
    uint32_t AddEmitter(const FluidEmitter& emitter);
    FluidEmitter* GetEmitter(uint32_t id);
    void RemoveEmitter(uint32_t id);
    size_t GetEmitterCount() const { return m_emitters.size(); }

    // Colliders
    uint32_t AddCollider(const FluidCollider& collider);
    FluidCollider* GetCollider(uint32_t id);
    void RemoveCollider(uint32_t id);
    size_t GetColliderCount() const { return m_colliders.size(); }

    // Buoyancy objects
    uint32_t AddBuoyancyObject(const BuoyancyObject& obj);
    BuoyancyObject* GetBuoyancyObject(uint32_t id);
    void RemoveBuoyancyObject(uint32_t id);

    // Particles (for initial fill)
    void AddParticle(float x, float y, float z, uint32_t materialId = 0);
    void AddParticleBox(float minX, float minY, float minZ,
                        float maxX, float maxY, float maxZ,
                        uint32_t materialId = 0);
    void AddParticleSphere(float cx, float cy, float cz, float radius,
                           uint32_t materialId = 0);
    void ClearParticles();

    // Simulation
    void Step(float deltaTime);
    void Reset();

    // Access
    const FluidParticle* GetParticles() const { return m_particles.data(); }
    uint32_t GetParticleCount() const { return m_activeParticles; }
    uint32_t GetMaxParticles() const { return m_config.maxParticles; }
    const FluidStats& GetStats() const { return m_stats; }
    const FluidSystemConfig& GetConfig() const { return m_config; }

    // Grid access (for debugging/visualization)
    const FluidGrid* GetGrid() const { return m_grid.get(); }

private:
    // Internal simulation steps (CPU reference)
    void EmitParticles(float deltaTime);
    void ParticleToGrid();          // P2G transfer
    void GridForces(float deltaTime);
    void PressureSolve();
    void GridToParticle(float deltaTime);  // G2P transfer
    void ParticleCollisions();
    void UpdateSleeping();
    void ComputeBuoyancy();
    void UpdateStats();

    // GPU dispatch
    void StepGPU(float deltaTime);
    void SyncParticlesToGPU();
    void SyncParticlesFromGPU();

    // Data
    FluidSystemConfig m_config;
    bool m_initialized = false;
    bool m_gpuEnabled = false;

    // Particles
    std::vector<FluidParticle> m_particles;
    uint32_t m_activeParticles = 0;

    // Grid
    std::unique_ptr<FluidGrid> m_grid;

    // Materials
    std::vector<FluidMaterial> m_materials;

    // Emitters and colliders
    std::vector<FluidEmitter> m_emitters;
    std::vector<FluidCollider> m_colliders;
    std::vector<BuoyancyObject> m_buoyancyObjects;

    // Statistics
    FluidStats m_stats;

    // GPU resources (forward declared, not used yet)
    VulkanContext* m_vulkanContext = nullptr;
    // TODO: GPU buffers and pipelines will be added when GPU implementation is ready
    // std::unique_ptr<GPUBuffer> m_particleBuffer;
    // std::unique_ptr<GPUBuffer> m_gridBuffer;
    // std::unique_ptr<ComputePipeline> m_p2gPipeline;
    // etc.
};

} // namespace WulfNet
