// =============================================================================
// WulfNet Engine - Vulkan Fluid Compute Integration
// =============================================================================
// GPU compute shader integration for CO-FLIP fluid simulation.
// Manages compute pipelines, GPU buffers, and dispatches for all fluid stages.
// =============================================================================

#pragma once

#include "WulfNet/Compute/Vulkan/VulkanContext.h"
#include "WulfNet/Compute/Memory/ComputeBuffer.h"
#include "WulfNet/Compute/Shaders/ComputePipeline.h"
#include "WulfNet/Physics/Fluids/COFLIPSystem.h"
#include <memory>
#include <string>

namespace WulfNet {

// =============================================================================
// GPU Simulation Parameters (matches shader push constants)
// =============================================================================
struct alignas(16) FluidSimParams {
    // Grid dimensions
    uint32_t gridSizeX;
    uint32_t gridSizeY;
    uint32_t gridSizeZ;
    uint32_t particleCount;

    // Cell size and inverse
    float cellSize;
    float invCellSize;
    float dt;
    float flipRatio;

    // Gravity
    float gravityX;
    float gravityY;
    float gravityZ;
    float restDensity;

    // Pressure solver
    uint32_t pressureIterations;
    float sorOmega;  // SOR relaxation factor (typically 1.5-1.9)
    float surfaceTension;
    float viscosity;
};
static_assert(sizeof(FluidSimParams) == 64, "FluidSimParams must be 64 bytes for push constants");

// =============================================================================
// Vulkan Fluid Compute
// =============================================================================
class VulkanFluidCompute {
public:
    VulkanFluidCompute();
    ~VulkanFluidCompute();

    // Non-copyable
    VulkanFluidCompute(const VulkanFluidCompute&) = delete;
    VulkanFluidCompute& operator=(const VulkanFluidCompute&) = delete;

    // ==========================================================================
    // Initialization
    // ==========================================================================

    /// Initialize GPU compute resources
    /// @param vulkan Vulkan context for GPU operations
    /// @param config CO-FLIP configuration for buffer sizing
    /// @param shaderPath Path to compiled SPIR-V shaders
    /// @return true if successful
    bool Initialize(VulkanContext* vulkan, const COFLIPConfig& config,
                    const std::string& shaderPath = "Assets/Shaders/Compute");

    /// Release all GPU resources
    void Shutdown();

    /// Check if GPU compute is available
    bool IsInitialized() const { return m_initialized; }

    // ==========================================================================
    // Buffer Management
    // ==========================================================================

    /// Upload particles from CPU to GPU
    bool UploadParticles(const std::vector<COFLIPParticle>& particles, uint32_t count);

    /// Download particles from GPU to CPU
    bool DownloadParticles(std::vector<COFLIPParticle>& particles, uint32_t count);

    /// Upload grid from CPU to GPU
    bool UploadGrid(const std::vector<COFLIPCell>& grid);

    /// Download grid from GPU to CPU
    bool DownloadGrid(std::vector<COFLIPCell>& grid);

    // ==========================================================================
    // Compute Dispatches
    // ==========================================================================

    /// Particle-to-Grid transfer with atomic accumulation
    void DispatchP2G(const FluidSimParams& params);

    /// Normalize grid velocities by accumulated weights
    void DispatchNormalize(const FluidSimParams& params);

    /// Apply external forces (gravity, etc.)
    void DispatchForces(const FluidSimParams& params);

    /// Compute velocity divergence
    void DispatchDivergence(const FluidSimParams& params);

    /// Jacobi pressure solve iteration
    void DispatchPressure(const FluidSimParams& params, uint32_t iterations);

    /// Apply pressure gradient for divergence-free velocity
    void DispatchGradient(const FluidSimParams& params);

    /// Grid-to-Particle transfer with FLIP/PIC blend
    void DispatchG2P(const FluidSimParams& params);

    /// Full simulation step (all stages) - BATCHED for performance
    void DispatchFullStep(const FluidSimParams& params);

    /// Full simulation step using single command buffer (maximum performance)
    /// This eliminates per-dispatch synchronization overhead
    void DispatchFullStepBatched(const FluidSimParams& params);

    /// Full simulation step with particle sorting for optimal cache performance
    void DispatchFullStepSorted(const FluidSimParams& params);

    /// Async compute - begin simulation step (returns immediately)
    /// Call WaitForSimulation() before accessing results
    void BeginAsyncSimulation(const FluidSimParams& params);

    /// Wait for async simulation to complete
    void WaitForSimulation();

    /// Check if async simulation is in progress
    bool IsSimulationInProgress() const { return m_asyncInProgress; }

    /// Sort particles by cell for improved cache coherence
    void SortParticlesByCell(const FluidSimParams& params);

    // ==========================================================================
    // GPU Buffer Handles (for external rendering)
    // ==========================================================================

    /// Get particle buffer for rendering
    VkBuffer GetParticleBuffer() const;
    size_t GetParticleBufferSize() const;

    /// Get grid buffer for visualization
    VkBuffer GetGridBuffer() const;

private:
    bool LoadShaders(const std::string& shaderPath);
    bool CreateBuffers(const COFLIPConfig& config);
    void InsertMemoryBarrier();

    // Batched dispatch helpers
    void RecordP2G(VkCommandBuffer cmd, const FluidSimParams& params);
    void RecordNormalize(VkCommandBuffer cmd, const FluidSimParams& params);
    void RecordForces(VkCommandBuffer cmd, const FluidSimParams& params);
    void RecordDivergence(VkCommandBuffer cmd, const FluidSimParams& params);
    void RecordPressure(VkCommandBuffer cmd, const FluidSimParams& params, uint32_t iterations);
    void RecordGradient(VkCommandBuffer cmd, const FluidSimParams& params);
    void RecordG2P(VkCommandBuffer cmd, const FluidSimParams& params);
    void RecordMemoryBarrier(VkCommandBuffer cmd);

    // Particle sorting helpers
    void RecordCellIndexCompute(VkCommandBuffer cmd, const FluidSimParams& params);
    void RecordRadixSort(VkCommandBuffer cmd, const FluidSimParams& params);
    void RecordParticleReorder(VkCommandBuffer cmd, const FluidSimParams& params);

    bool m_initialized = false;
    VulkanContext* m_vulkan = nullptr;

    // Async compute state
    bool m_asyncInProgress = false;
    VkFence m_asyncFence = VK_NULL_HANDLE;
    VkCommandBuffer m_asyncCmdBuffer = VK_NULL_HANDLE;

    // Compute pipelines
    std::unique_ptr<ComputePipeline> m_p2gPipeline;
    std::unique_ptr<ComputePipeline> m_normalizePipeline;
    std::unique_ptr<ComputePipeline> m_forcesPipeline;
    std::unique_ptr<ComputePipeline> m_divergencePipeline;
    std::unique_ptr<ComputePipeline> m_pressurePipeline;
    std::unique_ptr<ComputePipeline> m_gradientPipeline;
    std::unique_ptr<ComputePipeline> m_g2pPipeline;

    // Particle sorting pipelines
    std::unique_ptr<ComputePipeline> m_cellIndexPipeline;
    std::unique_ptr<ComputePipeline> m_radixSortPipeline;
    std::unique_ptr<ComputePipeline> m_reorderPipeline;

    // GPU buffers
    std::unique_ptr<ComputeBuffer<COFLIPParticle>> m_particleBuffer;
    std::unique_ptr<ComputeBuffer<COFLIPCell>> m_gridBuffer;
    std::unique_ptr<ComputeBuffer<float>> m_prevVelocityBuffer;  // Previous velocities for FLIP

    // Sorting buffers
    std::unique_ptr<ComputeBuffer<uint32_t>> m_cellIndexBuffer;
    std::unique_ptr<ComputeBuffer<uint32_t>> m_particleIndexBuffer;
    std::unique_ptr<ComputeBuffer<uint32_t>> m_tempCellIndexBuffer;
    std::unique_ptr<ComputeBuffer<uint32_t>> m_tempParticleIndexBuffer;
    std::unique_ptr<ComputeBuffer<uint32_t>> m_histogramBuffer;
    std::unique_ptr<ComputeBuffer<COFLIPParticle>> m_sortedParticleBuffer;

    // Buffer sizes
    uint32_t m_maxParticles = 0;
    uint32_t m_gridTotalCells = 0;
    uint32_t m_gridSizeX = 0, m_gridSizeY = 0, m_gridSizeZ = 0;
};

} // namespace WulfNet
