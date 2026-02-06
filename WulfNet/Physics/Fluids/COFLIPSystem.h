// =============================================================================
// WulfNet Engine - CO-FLIP (Coadjoint Orbit FLIP) Fluid Simulation
// Based on "Fluid Implicit Particles on Coadjoint Orbits" (SIGGRAPH Asia 2024)
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <functional>

// Forward declaration for Jolt compute system (must be before WulfNet namespace)
namespace JPH {
    class ComputeSystem;
}

namespace WulfNet {

// Forward declarations
class VulkanContext;
class VulkanFluidCompute;
struct FluidSimParams;

// =============================================================================
// CO-FLIP Particle - Optimized for GPU (64 bytes aligned)
// =============================================================================
struct alignas(16) COFLIPParticle {
    // Position (12 bytes)
    float x, y, z;

    // Velocity (12 bytes)
    float vx, vy, vz;

    // Vorticity/circulation (12 bytes) - preserved by CO-FLIP
    float wx, wy, wz;

    // Mass and volume (8 bytes)
    float mass;
    float volume;

    // Material and flags (8 bytes)
    uint32_t materialId;
    uint32_t flags;

    // Padding to 64 bytes (12 bytes)
    float _pad[3];
};
static_assert(sizeof(COFLIPParticle) == 64, "COFLIPParticle must be 64 bytes for GPU alignment");

// =============================================================================
// MAC Grid Cell for CO-FLIP - Staggered velocity storage
// =============================================================================
struct alignas(16) COFLIPCell {
    // Face velocities (staggered MAC grid)
    float u, v, w;          // Velocity at face centers

    // Pressure and divergence
    float pressure;
    float divergence;

    // Weights for transfer
    float weightU, weightV, weightW;

    // Cell type (0=air, 1=fluid, 2=solid)
    uint32_t type;

    // Padding
    float _pad[2];
};
static_assert(sizeof(COFLIPCell) == 48, "COFLIPCell should be 48 bytes");

// =============================================================================
// CO-FLIP Configuration
// =============================================================================
struct COFLIPConfig {
    // Grid dimensions (keep LOW for CO-FLIP - it works well at low res!)
    uint32_t gridSizeX = 64;
    uint32_t gridSizeY = 64;
    uint32_t gridSizeZ = 64;

    // Cell size in world units
    float cellSize = 0.1f;

    // Time step
    float dt = 1.0f / 60.0f;

    // Gravity
    float gravityX = 0.0f;
    float gravityY = -9.81f;
    float gravityZ = 0.0f;

    // FLIP/PIC blend (1.0 = pure FLIP, 0.0 = pure PIC)
    // CO-FLIP typically uses high FLIP ratio due to energy preservation
    float flipRatio = 0.99f;

    // Pressure solver iterations
    uint32_t pressureIterations = 50;

    // Surface tension coefficient
    float surfaceTension = 0.0728f;  // Water at 20°C

    // Viscosity (dynamic)
    float viscosity = 0.001f;  // Water

    // Density
    float restDensity = 1000.0f;  // Water kg/m³

    // Enable GPU acceleration
    bool useGPU = true;

    // Particles per cell (for initialization)
    uint32_t particlesPerCell = 8;
};

// =============================================================================
// CO-FLIP Statistics
// =============================================================================
struct COFLIPStats {
    uint32_t activeParticles = 0;
    uint32_t fluidCells = 0;
    float totalEnergy = 0.0f;       // Kinetic + potential (should be conserved!)
    float totalCirculation = 0.0f;  // Vorticity integral (should be conserved!)
    float maxVelocity = 0.0f;
    float p2gTimeMs = 0.0f;
    float pressureTimeMs = 0.0f;
    float g2pTimeMs = 0.0f;
    float totalTimeMs = 0.0f;
};

// =============================================================================
// GPU Buffer Handle (opaque, managed by Vulkan backend)
// =============================================================================
struct GPUBufferHandle {
    uint64_t handle = 0;
    size_t size = 0;
    bool valid() const { return handle != 0; }
};

// =============================================================================
// CO-FLIP System - Main fluid simulation class
// =============================================================================

class COFLIPSystem {
public:
    COFLIPSystem();
    ~COFLIPSystem();

    // Initialization (WulfNet VulkanContext path)
    bool Initialize(const COFLIPConfig& config, VulkanContext* vulkan = nullptr);

    // Initialization (Jolt ComputeSystem path - for Samples app integration)
    bool InitializeFromJolt(const COFLIPConfig& config, ::JPH::ComputeSystem* joltCompute);

    void Shutdown();
    bool IsInitialized() const { return m_initialized; }

    // Simulation
    void Step(float dt);
    void Reset();

    // Particle management
    uint32_t AddParticle(float x, float y, float z, float vx = 0, float vy = 0, float vz = 0);
    void AddParticleBox(float minX, float minY, float minZ, float maxX, float maxY, float maxZ);
    void AddParticleSphere(float cx, float cy, float cz, float radius);

    // Emitters
    void AddEmitter(float x, float y, float z, float dirX, float dirY, float dirZ, float rate, float speed);

    // Solid obstacles
    void AddSolidBox(float minX, float minY, float minZ, float maxX, float maxY, float maxZ);
    void AddSolidSphere(float cx, float cy, float cz, float radius);

    // Accessors
    const std::vector<COFLIPParticle>& GetParticles() const { return m_particles; }
    std::vector<COFLIPParticle>& GetParticles() { return m_particles; }
    uint32_t GetActiveParticleCount() const { return m_activeParticles; }
    const COFLIPStats& GetStats() const { return m_stats; }
    const COFLIPConfig& GetConfig() const { return m_config; }

    // GPU buffer access for rendering
    GPUBufferHandle GetParticleBuffer() const { return m_particleBufferHandle; }
    GPUBufferHandle GetGridBuffer() const { return m_gridBufferHandle; }

    // Sync particles between CPU and GPU
    void SyncParticlesToGPU();
    void SyncParticlesFromGPU();

private:
    // Core simulation steps (CPU fallback)
    void ParticleToGrid_CPU();
    void ApplyExternalForces_CPU(float dt);
    void ComputeDivergence_CPU();
    void PressureSolve_CPU();
    void ApplyPressureGradient_CPU();
    void GridToParticle_CPU();

    // GPU simulation steps
    void ParticleToGrid_GPU();
    void ApplyExternalForces_GPU(float dt);
    void PressureSolve_GPU();
    void GridToParticle_GPU();

    // Divergence-free interpolation (key CO-FLIP innovation)
    void InterpolateDivergenceFree(float x, float y, float z, float& vx, float& vy, float& vz) const;
    void InterpolateDivergenceFreeQuadratic(float x, float y, float z, float& vx, float& vy, float& vz) const;
    void InterpolateVelocityGradient(float x, float y, float z, float grad[9]) const;

    // B-spline basis functions for high-order interpolation
    float BSpline(float x) const;
    float BSplineDerivative(float x) const;

    // Grid helpers
    int GridIndex(int i, int j, int k) const;
    void WorldToGrid(float wx, float wy, float wz, float& gx, float& gy, float& gz) const;
    void GridToWorld(float gx, float gy, float gz, float& wx, float& wy, float& wz) const;
    bool InBounds(int i, int j, int k) const;

    // Energy/circulation computation (for monitoring conservation)
    float ComputeKineticEnergy() const;
    float ComputePotentialEnergy() const;
    float ComputeCirculation() const;

    // Update statistics
    void UpdateStats();

private:
    // Configuration
    COFLIPConfig m_config;
    bool m_initialized = false;
    bool m_gpuEnabled = false;

    // Particles
    std::vector<COFLIPParticle> m_particles;
    uint32_t m_activeParticles = 0;

    // Grid (MAC staggered)
    std::vector<COFLIPCell> m_grid;
    uint32_t m_gridTotalCells = 0;

    // Previous velocities for FLIP update
    std::vector<float> m_prevU, m_prevV, m_prevW;

    // Emitter data
    struct Emitter {
        float x, y, z;
        float dirX, dirY, dirZ;
        float rate, speed;
        float accumulator;
    };
    std::vector<Emitter> m_emitters;

    // Solid obstacles (marked in grid)
    std::vector<bool> m_solidCells;

    // Statistics
    COFLIPStats m_stats;

    // GPU resources
    VulkanContext* m_vulkanContext = nullptr;
    std::unique_ptr<VulkanFluidCompute> m_gpuCompute;
    GPUBufferHandle m_particleBufferHandle;
    GPUBufferHandle m_gridBufferHandle;
    GPUBufferHandle m_uniformBufferHandle;

    // GPU pipeline handles (will be initialized when GPU is available)
    uint64_t m_p2gPipeline = 0;
    uint64_t m_divergencePipeline = 0;
    uint64_t m_pressurePipeline = 0;
    uint64_t m_g2pPipeline = 0;
};

} // namespace WulfNet
