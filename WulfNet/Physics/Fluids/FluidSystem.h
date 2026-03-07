#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Core/JobSystem.h>
#include <vector>
#include <atomic>
#include <mutex>

#include "API.h"

// Forward declare GPU compute class (only needed as pointer)
namespace WulfNet { class SWEComputeGPU; }

namespace WulfNet {

// Shallow Water Equation (SWE) configuration
struct FluidSystemConfig {
    float gridSize = 1.0f;       // Size of each grid cell (meters)
    uint32_t width = 1024;       // Grid width
    uint32_t height = 1024;      // Grid height
    float gravity = 9.81f;       // Gravity acceleration
    float dtMax = 0.016f;        // Max internal timestep for stability (CFL condition)
    float density = 1000.0f;     // Water density (kg/m^3)
    float dragCoefficient = 1.05f; // Drag coefficient for floating objects
    float fluxDamping = 0.5f;     // Flux damping rate (per second). Higher = faster energy dissipation.
    float originX = 0.0f;         // World X coordinate of grid cell (0,0)
    float originZ = 0.0f;         // World Z coordinate of grid cell (0,0)
};

// Backward-compatible alias
using WaterSystemV3Config = FluidSystemConfig;

// Indirect Dispatch mapping structs for Sparse Execution Optimization
struct MacroTileID {
    uint32_t groupX;
    uint32_t groupY;
};

// Standard D3D12/Vulkan Indirect Dispatch alignment
struct alignas(16) DispatchIndirectArgs {
    uint32_t threadGroupCountX;
    uint32_t threadGroupCountY;
    uint32_t threadGroupCountZ;
};

// Represents a 2.5D Water Grid using SoA (Structure of Arrays) layout for optimal processing
struct WaterStateSOA {
    std::vector<float> terrainHeight;  // h: Static terrain height map
    std::vector<float> waterDepth;     // d: Current water depth per cell

    // OPTIMIZATION: Packed 128-bit float4 for perfect GPU memory coalescing
    // Replaces 4 distinct buffer reads with a single fast VRAM cache instruction.
    struct alignas(16) Float4 { float L, R, T, B; };
    std::vector<Float4> flux;

    void Resize(uint32_t size) {
        terrainHeight.assign(size, 0.0f);
        waterDepth.assign(size, 0.0f);
        flux.assign(size, {0.0f, 0.0f, 0.0f, 0.0f});
    }

    // Test Helper: Total Volume
    double CalculateTotalVolume(float cellArea) const;
};

// Jolt Physics Buoyancy Extension - interacts with the SWE Grid
class WULFNET_API FluidSystem {
public:
    FluidSystem(const FluidSystemConfig& config, JPH::PhysicsSystem* physicsSystem);
    ~FluidSystem();

    // GPU abstraction interfaces (impl tied to WulfNet::Compute)
    void InitializeGPUBuffers(); // Creates Vulkan SSBOs/DX12 UAVs
    void DispatchCompute(float deltaTime); // Dispatches HLSL/GLSL compute shaders

    // CPU fallback & mathematically exact simulation for Unit Tests
    void StepSimulationCPU(float deltaTime);

    // Binds Jolt bodies to parallel buoyancy evaluation
    void ApplyBuoyancyForces(JPH::JobSystem* jobSystem);

    WaterStateSOA& GetCPUState() { return m_state; }
    const WaterStateSOA& GetCPUState() const { return m_state; }
    void RequestAsyncReadback(); // Request GPU to transfer water bounding box -> CPU

    // Sparse Execution (Macro-Tiles) logic
    void BuildSparseActiveTilesCPU(); // CPU equivalent of the tile classifier shader pass

    // Modifiers
    void AddWater(uint32_t x, uint32_t y, float volume);
    void RemoveWater(uint32_t x, uint32_t y, float volume);

    const FluidSystemConfig& GetConfig() const { return m_config; }

    // Parallel Job Processing batching
    struct BuoyancyJobContext {
        std::vector<JPH::BodyID> interactingBodies;
        std::atomic<size_t> nextBodyIndex{0};
    };

    BuoyancyJobContext& GetJobContextForTesting() { return m_jobContext; }

private:
    float SampleWaterSurfaceHeight(float worldX, float worldZ) const;

    FluidSystemConfig m_config;
    JPH::PhysicsSystem* m_joltSystem;

    WaterStateSOA m_state; // CPU side state (or mapped readback state)

    std::vector<MacroTileID> m_activeTiles; // List of active 8x8 regions

    // Pre-built index vectors for parallel dispatch (avoid alloc per frame)
    std::vector<uint32_t> m_rowIndices;   // [0..H-1] for row-parallel passes
    std::vector<uint32_t> m_cellIndices;  // [0..W*H-1] for cell-parallel passes

    BuoyancyJobContext m_jobContext;
    std::mutex m_bodyMutex;

    // GPU compute backend (optional, only active when Vulkan is available)
    std::unique_ptr<SWEComputeGPU> m_gpuCompute;
    bool m_gpuInitialized = false;
};

// Backward-compatible alias
using WaterSystemV3 = FluidSystem;

} // namespace WulfNet
