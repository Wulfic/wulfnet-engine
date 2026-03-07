// =============================================================================
// WulfNet Engine - SWE GPU Compute Acceleration
// =============================================================================
// GPU compute shader integration for the Shallow Water Equations (SWE) solver.
// Offloads the 4 SWE phases (velocity, outflow, gather, boundary) to GPU
// compute shaders for massive parallelism on 2D water grids.
//
// Follows the same patterns as VulkanFluidCompute for CO-FLIP simulation.
// =============================================================================

#pragma once

#include "WulfNet/Compute/Vulkan/VulkanContext.h"
#include "WulfNet/Compute/Memory/ComputeBuffer.h"
#include "WulfNet/Compute/Shaders/ComputePipeline.h"
#include <memory>
#include <string>
#include <cstdint>

namespace WulfNet {

// =============================================================================
// SWE GPU Push Constants (matches shader layout — 32 bytes, padded to 16-align)
// =============================================================================
struct alignas(16) SWESimParams {
    uint32_t gridSizeX;
    uint32_t gridSizeZ;
    float    gravity_over_cs;
    float    damping;

    float    viscosity;
    float    dt;
    float    _pad0;
    float    _pad1;
};
static_assert(sizeof(SWESimParams) == 32, "SWESimParams must be 32 bytes for push constants");

// =============================================================================
// SWE Compute GPU — Manages pipelines, buffers, and dispatch for SWE solver
// =============================================================================
class SWEComputeGPU {
public:
    SWEComputeGPU();
    ~SWEComputeGPU();

    // Non-copyable
    SWEComputeGPU(const SWEComputeGPU&) = delete;
    SWEComputeGPU& operator=(const SWEComputeGPU&) = delete;

    // =========================================================================
    // Initialization
    // =========================================================================

    /// Initialize GPU compute for SWE with given grid dimensions.
    /// @param gridSizeX  Number of cells in X
    /// @param gridSizeZ  Number of cells in Z
    /// @param shaderPath Directory containing compiled .spv shaders
    /// @return true if all shaders loaded and buffers created
    bool Initialize(uint32_t gridSizeX, uint32_t gridSizeZ,
                    const std::string& shaderPath = "Assets/Shaders/Compute");

    void Shutdown();

    bool IsInitialized() const { return m_initialized; }

    // =========================================================================
    // Data Transfer
    // =========================================================================

    /// Upload the full grid from CPU to GPU. Data layout: vec4 per cell
    /// (waterHeight, terrainHeight, vx, vz).
    bool UploadGrid(const float* gridData, uint32_t cellCount);

    /// Download the full grid from GPU back to CPU.
    bool DownloadGrid(float* gridData, uint32_t cellCount);

    /// Kick off an async GPU→CPU grid readback (non-blocking).
    bool DownloadGridAsync();

    /// Non-blocking check: has the last async readback completed?
    bool IsReadbackReady() const;

    /// Copy downloaded data into outData. Blocks if readback hasn't finished yet.
    bool GetReadbackData(std::vector<float>& outData);

    // =========================================================================
    // Compute Dispatch — Single Full SWE Step
    // =========================================================================

    /// Execute one full SWE timestep on the GPU (all 4 phases batched).
    /// Internally: snapshot → velocity → outflow → gather → boundary
    void StepSWE(const SWESimParams& params);

    /// Execute one full SWE timestep using individual DispatchAndWait calls
    /// (slower but useful for debugging).
    void StepSWEUnbatched(const SWESimParams& params);

    // =========================================================================
    // Stats
    // =========================================================================

    uint32_t GetGridSizeX() const { return m_gridSizeX; }
    uint32_t GetGridSizeZ() const { return m_gridSizeZ; }
    uint32_t GetTotalCells() const { return m_totalCells; }

private:
    bool LoadShaders(const std::string& shaderPath);
    bool CreateBuffers();

    // Batched dispatch recording helpers
    void RecordVelocity(void* cmdBuffer, const SWESimParams& params);
    void RecordOutflow(void* cmdBuffer, const SWESimParams& params);
    void RecordGather(void* cmdBuffer, const SWESimParams& params);
    void RecordBoundary(void* cmdBuffer, const SWESimParams& params);
    void RecordMemoryBarrier(void* cmdBuffer);
    void RecordSnapshotCopy(void* cmdBuffer);

    bool m_initialized = false;

    uint32_t m_gridSizeX = 0;
    uint32_t m_gridSizeZ = 0;
    uint32_t m_totalCells = 0;

    // Workgroup dispatch sizes (precomputed)
    uint32_t m_dispatchX = 0;
    uint32_t m_dispatchZ = 0;

    // GPU Buffers — WaterCell stored as vec4 (float x 4)
    std::unique_ptr<ComputeBuffer<float>> m_gridBuffer;     // Live grid (read/write)
    std::unique_ptr<ComputeBuffer<float>> m_snapshotBuffer; // Per-step snapshot (read-only during phases)
    std::unique_ptr<ComputeBuffer<float>> m_outflowBuffer;  // 4 floats per cell (L,R,B,F)

    // Compute Pipelines
    std::unique_ptr<ComputePipeline> m_velocityPipeline;
    std::unique_ptr<ComputePipeline> m_outflowPipeline;
    std::unique_ptr<ComputePipeline> m_gatherPipeline;
    std::unique_ptr<ComputePipeline> m_boundaryPipeline;
};

} // namespace WulfNet
