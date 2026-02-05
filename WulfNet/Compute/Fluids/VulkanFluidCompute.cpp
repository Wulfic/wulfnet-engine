// =============================================================================
// WulfNet Engine - Vulkan Fluid Compute Integration
// =============================================================================
// GPU compute shader integration for CO-FLIP fluid simulation.
// =============================================================================

#include "VulkanFluidCompute.h"
#include <fstream>
#include <filesystem>
#include <iostream>  // For fallback logging

// Vulkan function pointer extern (from VulkanContext)
extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_External;

namespace WulfNet {

// Simple logging fallback (Logger macros require category)
#define FLUID_LOG_INFO(msg)  std::cout << "[FluidCompute] " << msg << std::endl
#define FLUID_LOG_WARN(msg)  std::cout << "[FluidCompute WARN] " << msg << std::endl
#define FLUID_LOG_ERROR(msg) std::cerr << "[FluidCompute ERROR] " << msg << std::endl

// =============================================================================
// Local Vulkan Function Pointers for Batched Dispatch
// =============================================================================
static PFN_vkAllocateCommandBuffers     s_vkAllocateCommandBuffers = nullptr;
static PFN_vkFreeCommandBuffers         s_vkFreeCommandBuffers = nullptr;
static PFN_vkBeginCommandBuffer         s_vkBeginCommandBuffer = nullptr;
static PFN_vkEndCommandBuffer           s_vkEndCommandBuffer = nullptr;
static PFN_vkCmdBindPipeline            s_vkCmdBindPipeline = nullptr;
static PFN_vkCmdBindDescriptorSets      s_vkCmdBindDescriptorSets = nullptr;
static PFN_vkCmdPushConstants           s_vkCmdPushConstants = nullptr;
static PFN_vkCmdDispatch                s_vkCmdDispatch = nullptr;
static PFN_vkCmdPipelineBarrier         s_vkCmdPipelineBarrier = nullptr;
static PFN_vkQueueSubmit                s_vkQueueSubmit = nullptr;
static PFN_vkQueueWaitIdle              s_vkQueueWaitIdle = nullptr;
static bool s_fluidFunctionsLoaded = false;

static bool LoadFluidVkFunctions(VkInstance instance) {
    if (s_fluidFunctionsLoaded) return true;
    if (!vkGetInstanceProcAddr_External) return false;

    #define LOAD_VK_FUNC(name) \
        s_##name = reinterpret_cast<PFN_##name>( \
            vkGetInstanceProcAddr_External(instance, #name))

    LOAD_VK_FUNC(vkAllocateCommandBuffers);
    LOAD_VK_FUNC(vkFreeCommandBuffers);
    LOAD_VK_FUNC(vkBeginCommandBuffer);
    LOAD_VK_FUNC(vkEndCommandBuffer);
    LOAD_VK_FUNC(vkCmdBindPipeline);
    LOAD_VK_FUNC(vkCmdBindDescriptorSets);
    LOAD_VK_FUNC(vkCmdPushConstants);
    LOAD_VK_FUNC(vkCmdDispatch);
    LOAD_VK_FUNC(vkCmdPipelineBarrier);
    LOAD_VK_FUNC(vkQueueSubmit);
    LOAD_VK_FUNC(vkQueueWaitIdle);

    #undef LOAD_VK_FUNC

    s_fluidFunctionsLoaded = true;
    return true;
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

VulkanFluidCompute::VulkanFluidCompute() = default;

VulkanFluidCompute::~VulkanFluidCompute() {
    Shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool VulkanFluidCompute::Initialize(VulkanContext* vulkan, const COFLIPConfig& config,
                                     const std::string& shaderPath) {
    if (m_initialized) {
        FLUID_LOG_WARN("VulkanFluidCompute already initialized");
        return true;
    }

    if (!vulkan || !vulkan->IsValid()) {
        FLUID_LOG_ERROR("VulkanFluidCompute: Invalid Vulkan context");
        return false;
    }

    // Load Vulkan function pointers for batched dispatch
    if (!LoadFluidVkFunctions(vulkan->GetInstance())) {
        FLUID_LOG_ERROR("VulkanFluidCompute: Failed to load Vulkan functions");
        return false;
    }

    m_vulkan = vulkan;
    m_gridSizeX = config.gridSizeX;
    m_gridSizeY = config.gridSizeY;
    m_gridSizeZ = config.gridSizeZ;
    m_gridTotalCells = m_gridSizeX * m_gridSizeY * m_gridSizeZ;

    // Estimate max particles (8 per cell is typical)
    m_maxParticles = m_gridTotalCells * config.particlesPerCell;

    FLUID_LOG_INFO("VulkanFluidCompute: Initializing GPU compute");
    FLUID_LOG_INFO("  Grid: {}x{}x{} = {} cells", m_gridSizeX, m_gridSizeY, m_gridSizeZ, m_gridTotalCells);
    FLUID_LOG_INFO("  Max particles: {}", m_maxParticles);

    // Create GPU buffers
    if (!CreateBuffers(config)) {
        FLUID_LOG_ERROR("VulkanFluidCompute: Failed to create GPU buffers");
        return false;
    }

    // Load and create compute pipelines
    if (!LoadShaders(shaderPath)) {
        FLUID_LOG_ERROR("VulkanFluidCompute: Failed to load shaders");
        return false;
    }

    m_initialized = true;
    FLUID_LOG_INFO("VulkanFluidCompute: GPU compute ready");
    return true;
}

void VulkanFluidCompute::Shutdown() {
    if (!m_initialized) return;

    FLUID_LOG_INFO("VulkanFluidCompute: Shutting down");

    // Wait for any pending GPU work
    if (m_vulkan && m_vulkan->IsValid()) {
        m_vulkan->WaitIdle();
    }

    // Destroy pipelines
    m_p2gPipeline.reset();
    m_normalizePipeline.reset();
    m_forcesPipeline.reset();
    m_divergencePipeline.reset();
    m_pressurePipeline.reset();
    m_gradientPipeline.reset();
    m_g2pPipeline.reset();

    // Release buffers
    m_particleBuffer.reset();
    m_gridBuffer.reset();
    m_prevVelocityBuffer.reset();

    m_vulkan = nullptr;
    m_initialized = false;
}

// =============================================================================
// Buffer Creation
// =============================================================================

bool VulkanFluidCompute::CreateBuffers(const COFLIPConfig& config) {
    try {
        // Particle buffer (device local, transfer enabled)
        m_particleBuffer = std::make_unique<ComputeBuffer<COFLIPParticle>>(
            m_maxParticles,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        // Grid buffer
        m_gridBuffer = std::make_unique<ComputeBuffer<COFLIPCell>>(
            m_gridTotalCells,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        // Previous velocity buffer (3 floats per cell face = 3 * gridSize)
        // Actually we need previous U, V, W stored separately for FLIP
        size_t totalFaces = static_cast<size_t>(m_gridSizeX + 1) * m_gridSizeY * m_gridSizeZ +
                            m_gridSizeX * static_cast<size_t>(m_gridSizeY + 1) * m_gridSizeZ +
                            m_gridSizeX * m_gridSizeY * static_cast<size_t>(m_gridSizeZ + 1);
        m_prevVelocityBuffer = std::make_unique<ComputeBuffer<float>>(
            totalFaces,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        FLUID_LOG_INFO("  Particle buffer: {} MB", m_particleBuffer->GetSizeBytes() / (1024 * 1024));
        FLUID_LOG_INFO("  Grid buffer: {} MB", m_gridBuffer->GetSizeBytes() / (1024 * 1024));
        FLUID_LOG_INFO("  Prev velocity buffer: {} MB", m_prevVelocityBuffer->GetSizeBytes() / (1024 * 1024));

        // Sorting buffers for cache-coherent particle access
        m_cellIndexBuffer = std::make_unique<ComputeBuffer<uint32_t>>(
            m_maxParticles,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        m_particleIndexBuffer = std::make_unique<ComputeBuffer<uint32_t>>(
            m_maxParticles,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        m_tempCellIndexBuffer = std::make_unique<ComputeBuffer<uint32_t>>(
            m_maxParticles,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        m_tempParticleIndexBuffer = std::make_unique<ComputeBuffer<uint32_t>>(
            m_maxParticles,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        // Radix sort histogram: 256 bins per workgroup, max ~1000 workgroups
        m_histogramBuffer = std::make_unique<ComputeBuffer<uint32_t>>(
            256 * 1024, // 256 radix bins * 1024 workgroups
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        // Sorted particle buffer (double buffer for sorting)
        m_sortedParticleBuffer = std::make_unique<ComputeBuffer<COFLIPParticle>>(
            m_maxParticles,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        FLUID_LOG_INFO("  Sorting buffers: {} MB total",
            (m_cellIndexBuffer->GetSizeBytes() + m_particleIndexBuffer->GetSizeBytes() +
             m_tempCellIndexBuffer->GetSizeBytes() + m_tempParticleIndexBuffer->GetSizeBytes() +
             m_histogramBuffer->GetSizeBytes() + m_sortedParticleBuffer->GetSizeBytes()) / (1024 * 1024));

        return m_particleBuffer->IsValid() && m_gridBuffer->IsValid() && m_prevVelocityBuffer->IsValid() &&
               m_cellIndexBuffer->IsValid() && m_sortedParticleBuffer->IsValid();
    }
    catch (const std::exception& e) {
        FLUID_LOG_ERROR("  Buffer creation failed: {}", e.what());
        return false;
    }
}

// =============================================================================
// Shader Loading
// =============================================================================

bool VulkanFluidCompute::LoadShaders(const std::string& shaderPath) {
    // Define shader bindings (matches our GLSL compute shaders)
    // Binding 0: Particles (storage buffer)
    // Binding 1: Grid (storage buffer)
    // Binding 2: Previous velocities (storage buffer)
    std::vector<ShaderBinding> fluidBindings = {
        {0, ShaderBindingType::StorageBuffer, "particles"},
        {1, ShaderBindingType::StorageBuffer, "grid"},
        {2, ShaderBindingType::StorageBuffer, "prevVelocity"}
    };

    // Push constant range for simulation parameters
    PushConstantRange pushRange{0, sizeof(FluidSimParams)};

    auto loadPipeline = [&](const std::string& name, uint32_t localX = 256, uint32_t localY = 1, uint32_t localZ = 1)
        -> std::unique_ptr<ComputePipeline>
    {
        std::string path = shaderPath + "/" + name + ".spv";

        // Check if file exists
        if (!std::filesystem::exists(path)) {
            FLUID_LOG_ERROR("  Shader not found: {}", path);
            return nullptr;
        }

        auto pipeline = std::make_unique<ComputePipeline>();
        if (!pipeline->CreateFromFile(path, fluidBindings, pushRange)) {
            FLUID_LOG_ERROR("  Failed to create pipeline: {}", name);
            return nullptr;
        }

        FLUID_LOG_INFO("  Loaded shader: {}", name);
        return pipeline;
    };

    // Load all CO-FLIP shaders
    m_p2gPipeline = loadPipeline("coflip_p2g", 256, 1, 1);
    m_normalizePipeline = loadPipeline("coflip_normalize", 8, 8, 8);
    m_forcesPipeline = loadPipeline("coflip_forces", 8, 8, 8);
    m_divergencePipeline = loadPipeline("coflip_divergence", 8, 8, 8);
    m_pressurePipeline = loadPipeline("coflip_pressure", 8, 8, 8);
    m_gradientPipeline = loadPipeline("coflip_gradient", 8, 8, 8);
    m_g2pPipeline = loadPipeline("coflip_g2p", 256, 1, 1);

    // Check all pipelines loaded successfully
    bool allLoaded = m_p2gPipeline && m_normalizePipeline && m_forcesPipeline &&
                     m_divergencePipeline && m_pressurePipeline && m_gradientPipeline &&
                     m_g2pPipeline;

    if (!allLoaded) {
        FLUID_LOG_WARN("  Some shaders failed to load - GPU compute will be disabled");
        return false;
    }

    // Load sorting shaders (optional - sorted dispatch won't work without these)
    std::vector<ShaderBinding> sortBindings = {
        {0, ShaderBindingType::StorageBuffer, "particles"},
        {1, ShaderBindingType::StorageBuffer, "cellIndices"},
        {2, ShaderBindingType::StorageBuffer, "particleIndices"}
    };

    std::vector<ShaderBinding> radixBindings = {
        {0, ShaderBindingType::StorageBuffer, "cellIndices"},
        {1, ShaderBindingType::StorageBuffer, "particleIndices"},
        {2, ShaderBindingType::StorageBuffer, "tempCellIndices"},
        {3, ShaderBindingType::StorageBuffer, "tempParticleIndices"},
        {4, ShaderBindingType::StorageBuffer, "histogram"}
    };

    std::vector<ShaderBinding> reorderBindings = {
        {0, ShaderBindingType::StorageBuffer, "srcParticles"},
        {1, ShaderBindingType::StorageBuffer, "dstParticles"},
        {2, ShaderBindingType::StorageBuffer, "particleIndices"}
    };

    // Lambda for loading sorting pipelines with custom bindings
    auto loadSortPipeline = [&](const std::string& name, const std::vector<ShaderBinding>& bindings,
                                uint32_t localX = 256) -> std::unique_ptr<ComputePipeline>
    {
        std::string path = shaderPath + "/" + name + ".spv";
        if (!std::filesystem::exists(path)) {
            FLUID_LOG_DEBUG("  Optional shader not found: {}", path);
            return nullptr;
        }

        auto pipeline = std::make_unique<ComputePipeline>();
        if (!pipeline->CreateFromFile(path, bindings, pushRange)) {
            FLUID_LOG_DEBUG("  Failed to create optional pipeline: {}", name);
            return nullptr;
        }

        FLUID_LOG_INFO("  Loaded optional shader: {}", name);
        return pipeline;
    };

    m_cellIndexPipeline = loadSortPipeline("coflip_cell_index", sortBindings);
    m_radixSortPipeline = loadSortPipeline("coflip_radix_sort", radixBindings);
    m_reorderPipeline = loadSortPipeline("coflip_reorder", reorderBindings);

    if (m_cellIndexPipeline && m_radixSortPipeline && m_reorderPipeline) {
        FLUID_LOG_INFO("  Particle sorting enabled");
    } else {
        FLUID_LOG_INFO("  Particle sorting disabled (optional shaders not found)");
    }

    return true;
}

// =============================================================================
// Buffer Upload/Download
// =============================================================================

bool VulkanFluidCompute::UploadParticles(const std::vector<COFLIPParticle>& particles, uint32_t count) {
    if (!m_initialized || !m_particleBuffer) return false;
    if (count == 0) return true;
    if (count > m_maxParticles) {
        FLUID_LOG_WARN("Particle count {} exceeds max {}, clamping", count, m_maxParticles);
        count = m_maxParticles;
    }
    return m_particleBuffer->Upload(particles.data(), count, 0);
}

bool VulkanFluidCompute::DownloadParticles(std::vector<COFLIPParticle>& particles, uint32_t count) {
    if (!m_initialized || !m_particleBuffer) return false;
    if (count == 0) return true;
    if (particles.size() < count) {
        particles.resize(count);
    }
    return m_particleBuffer->Download(particles.data(), count, 0);
}

bool VulkanFluidCompute::UploadGrid(const std::vector<COFLIPCell>& grid) {
    if (!m_initialized || !m_gridBuffer) return false;
    if (grid.size() != m_gridTotalCells) {
        FLUID_LOG_ERROR("Grid size mismatch: {} vs {}", grid.size(), m_gridTotalCells);
        return false;
    }
    return m_gridBuffer->Upload(grid.data(), m_gridTotalCells, 0);
}

bool VulkanFluidCompute::DownloadGrid(std::vector<COFLIPCell>& grid) {
    if (!m_initialized || !m_gridBuffer) return false;
    if (grid.size() != m_gridTotalCells) {
        grid.resize(m_gridTotalCells);
    }
    return m_gridBuffer->Download(grid.data(), m_gridTotalCells, 0);
}

// =============================================================================
// Compute Dispatches
// =============================================================================

void VulkanFluidCompute::InsertMemoryBarrier() {
    // Memory barrier is handled internally by Vulkan command buffer submission
    // When using DispatchAndWait, barriers are implicit
    // For async dispatch, we'd need to insert VK_ACCESS_SHADER_WRITE_BIT -> VK_ACCESS_SHADER_READ_BIT
}

void VulkanFluidCompute::DispatchP2G(const FluidSimParams& params) {
    if (!m_p2gPipeline || !m_p2gPipeline->IsValid()) return;

    // Bind buffers
    m_p2gPipeline->BindBuffer(0, *m_particleBuffer);
    m_p2gPipeline->BindBuffer(1, *m_gridBuffer);
    m_p2gPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_p2gPipeline->UpdateBindings();

    // Set push constants
    m_p2gPipeline->SetPushConstants(params);

    // Dispatch one thread per particle
    uint32_t groupCount = m_p2gPipeline->CalculateGroupCount(params.particleCount);
    m_p2gPipeline->DispatchAndWait(groupCount, 1, 1);
}

void VulkanFluidCompute::DispatchNormalize(const FluidSimParams& params) {
    if (!m_normalizePipeline || !m_normalizePipeline->IsValid()) return;

    m_normalizePipeline->BindBuffer(0, *m_particleBuffer);
    m_normalizePipeline->BindBuffer(1, *m_gridBuffer);
    m_normalizePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_normalizePipeline->UpdateBindings();

    m_normalizePipeline->SetPushConstants(params);

    // Dispatch over grid cells (8x8x8 workgroups)
    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    m_normalizePipeline->DispatchAndWait(gx, gy, gz);
}

void VulkanFluidCompute::DispatchForces(const FluidSimParams& params) {
    if (!m_forcesPipeline || !m_forcesPipeline->IsValid()) return;

    m_forcesPipeline->BindBuffer(0, *m_particleBuffer);
    m_forcesPipeline->BindBuffer(1, *m_gridBuffer);
    m_forcesPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_forcesPipeline->UpdateBindings();

    m_forcesPipeline->SetPushConstants(params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    m_forcesPipeline->DispatchAndWait(gx, gy, gz);
}

void VulkanFluidCompute::DispatchDivergence(const FluidSimParams& params) {
    if (!m_divergencePipeline || !m_divergencePipeline->IsValid()) return;

    m_divergencePipeline->BindBuffer(0, *m_particleBuffer);
    m_divergencePipeline->BindBuffer(1, *m_gridBuffer);
    m_divergencePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_divergencePipeline->UpdateBindings();

    m_divergencePipeline->SetPushConstants(params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    m_divergencePipeline->DispatchAndWait(gx, gy, gz);
}

void VulkanFluidCompute::DispatchPressure(const FluidSimParams& params, uint32_t iterations) {
    if (!m_pressurePipeline || !m_pressurePipeline->IsValid()) return;

    m_pressurePipeline->BindBuffer(0, *m_particleBuffer);
    m_pressurePipeline->BindBuffer(1, *m_gridBuffer);
    m_pressurePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_pressurePipeline->UpdateBindings();

    m_pressurePipeline->SetPushConstants(params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;

    // Run multiple Jacobi iterations
    for (uint32_t i = 0; i < iterations; ++i) {
        m_pressurePipeline->DispatchAndWait(gx, gy, gz);
    }
}

void VulkanFluidCompute::DispatchGradient(const FluidSimParams& params) {
    if (!m_gradientPipeline || !m_gradientPipeline->IsValid()) return;

    m_gradientPipeline->BindBuffer(0, *m_particleBuffer);
    m_gradientPipeline->BindBuffer(1, *m_gridBuffer);
    m_gradientPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_gradientPipeline->UpdateBindings();

    m_gradientPipeline->SetPushConstants(params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    m_gradientPipeline->DispatchAndWait(gx, gy, gz);
}

void VulkanFluidCompute::DispatchG2P(const FluidSimParams& params) {
    if (!m_g2pPipeline || !m_g2pPipeline->IsValid()) return;

    m_g2pPipeline->BindBuffer(0, *m_particleBuffer);
    m_g2pPipeline->BindBuffer(1, *m_gridBuffer);
    m_g2pPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_g2pPipeline->UpdateBindings();

    m_g2pPipeline->SetPushConstants(params);

    uint32_t groupCount = m_g2pPipeline->CalculateGroupCount(params.particleCount);
    m_g2pPipeline->DispatchAndWait(groupCount, 1, 1);
}

void VulkanFluidCompute::DispatchFullStep(const FluidSimParams& params) {
    // Full CO-FLIP simulation step on GPU
    // 1. Particle to Grid (P2G)
    DispatchP2G(params);

    // 2. Normalize grid velocities
    DispatchNormalize(params);

    // 3. Apply external forces
    DispatchForces(params);

    // 4. Compute divergence
    DispatchDivergence(params);

    // 5. Pressure solve (multiple iterations)
    DispatchPressure(params, params.pressureIterations);

    // 6. Apply pressure gradient
    DispatchGradient(params);

    // 7. Grid to Particle (G2P) with FLIP/PIC blend
    DispatchG2P(params);
}

// =============================================================================
// Batched Dispatch Implementation (Maximum Performance)
// =============================================================================
// Records all simulation stages into a single command buffer with proper
// memory barriers, eliminating the massive overhead of per-dispatch sync.
// =============================================================================

void VulkanFluidCompute::RecordMemoryBarrier(VkCommandBuffer cmd) {
    // Memory barrier to ensure all writes complete before next stage reads
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    s_vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr
    );
}

void VulkanFluidCompute::RecordP2G(VkCommandBuffer cmd, const FluidSimParams& params) {
    if (!m_p2gPipeline || !m_p2gPipeline->IsValid()) return;

    s_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_p2gPipeline->GetVkPipeline());
    s_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_p2gPipeline->GetVkPipelineLayout(), 0, 1,
                              &m_p2gPipeline->GetVkDescriptorSet(), 0, nullptr);
    s_vkCmdPushConstants(cmd, m_p2gPipeline->GetVkPipelineLayout(),
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    uint32_t groupCount = m_p2gPipeline->CalculateGroupCount(params.particleCount);
    s_vkCmdDispatch(cmd, groupCount, 1, 1);
}

void VulkanFluidCompute::RecordNormalize(VkCommandBuffer cmd, const FluidSimParams& params) {
    if (!m_normalizePipeline || !m_normalizePipeline->IsValid()) return;

    s_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_normalizePipeline->GetVkPipeline());
    s_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_normalizePipeline->GetVkPipelineLayout(), 0, 1,
                              &m_normalizePipeline->GetVkDescriptorSet(), 0, nullptr);
    s_vkCmdPushConstants(cmd, m_normalizePipeline->GetVkPipelineLayout(),
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    s_vkCmdDispatch(cmd, gx, gy, gz);
}

void VulkanFluidCompute::RecordForces(VkCommandBuffer cmd, const FluidSimParams& params) {
    if (!m_forcesPipeline || !m_forcesPipeline->IsValid()) return;

    s_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_forcesPipeline->GetVkPipeline());
    s_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_forcesPipeline->GetVkPipelineLayout(), 0, 1,
                              &m_forcesPipeline->GetVkDescriptorSet(), 0, nullptr);
    s_vkCmdPushConstants(cmd, m_forcesPipeline->GetVkPipelineLayout(),
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    s_vkCmdDispatch(cmd, gx, gy, gz);
}

void VulkanFluidCompute::RecordDivergence(VkCommandBuffer cmd, const FluidSimParams& params) {
    if (!m_divergencePipeline || !m_divergencePipeline->IsValid()) return;

    s_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_divergencePipeline->GetVkPipeline());
    s_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_divergencePipeline->GetVkPipelineLayout(), 0, 1,
                              &m_divergencePipeline->GetVkDescriptorSet(), 0, nullptr);
    s_vkCmdPushConstants(cmd, m_divergencePipeline->GetVkPipelineLayout(),
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    s_vkCmdDispatch(cmd, gx, gy, gz);
}

void VulkanFluidCompute::RecordPressure(VkCommandBuffer cmd, const FluidSimParams& params, uint32_t iterations) {
    if (!m_pressurePipeline || !m_pressurePipeline->IsValid()) return;

    s_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pressurePipeline->GetVkPipeline());
    s_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_pressurePipeline->GetVkPipelineLayout(), 0, 1,
                              &m_pressurePipeline->GetVkDescriptorSet(), 0, nullptr);
    s_vkCmdPushConstants(cmd, m_pressurePipeline->GetVkPipelineLayout(),
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;

    for (uint32_t i = 0; i < iterations; ++i) {
        s_vkCmdDispatch(cmd, gx, gy, gz);
        // Barrier between pressure iterations
        if (i < iterations - 1) {
            RecordMemoryBarrier(cmd);
        }
    }
}

void VulkanFluidCompute::RecordGradient(VkCommandBuffer cmd, const FluidSimParams& params) {
    if (!m_gradientPipeline || !m_gradientPipeline->IsValid()) return;

    s_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_gradientPipeline->GetVkPipeline());
    s_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_gradientPipeline->GetVkPipelineLayout(), 0, 1,
                              &m_gradientPipeline->GetVkDescriptorSet(), 0, nullptr);
    s_vkCmdPushConstants(cmd, m_gradientPipeline->GetVkPipelineLayout(),
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    s_vkCmdDispatch(cmd, gx, gy, gz);
}

void VulkanFluidCompute::RecordG2P(VkCommandBuffer cmd, const FluidSimParams& params) {
    if (!m_g2pPipeline || !m_g2pPipeline->IsValid()) return;

    s_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_g2pPipeline->GetVkPipeline());
    s_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_g2pPipeline->GetVkPipelineLayout(), 0, 1,
                              &m_g2pPipeline->GetVkDescriptorSet(), 0, nullptr);
    s_vkCmdPushConstants(cmd, m_g2pPipeline->GetVkPipelineLayout(),
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    uint32_t groupCount = m_g2pPipeline->CalculateGroupCount(params.particleCount);
    s_vkCmdDispatch(cmd, groupCount, 1, 1);
}

void VulkanFluidCompute::DispatchFullStepBatched(const FluidSimParams& params) {
    if (!m_initialized || !m_vulkan) return;

    // Ensure bindings are up to date
    m_p2gPipeline->BindBuffer(0, *m_particleBuffer);
    m_p2gPipeline->BindBuffer(1, *m_gridBuffer);
    m_p2gPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_p2gPipeline->UpdateBindings();

    m_normalizePipeline->BindBuffer(0, *m_particleBuffer);
    m_normalizePipeline->BindBuffer(1, *m_gridBuffer);
    m_normalizePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_normalizePipeline->UpdateBindings();

    m_forcesPipeline->BindBuffer(0, *m_particleBuffer);
    m_forcesPipeline->BindBuffer(1, *m_gridBuffer);
    m_forcesPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_forcesPipeline->UpdateBindings();

    m_divergencePipeline->BindBuffer(0, *m_particleBuffer);
    m_divergencePipeline->BindBuffer(1, *m_gridBuffer);
    m_divergencePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_divergencePipeline->UpdateBindings();

    m_pressurePipeline->BindBuffer(0, *m_particleBuffer);
    m_pressurePipeline->BindBuffer(1, *m_gridBuffer);
    m_pressurePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_pressurePipeline->UpdateBindings();

    m_gradientPipeline->BindBuffer(0, *m_particleBuffer);
    m_gradientPipeline->BindBuffer(1, *m_gridBuffer);
    m_gradientPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_gradientPipeline->UpdateBindings();

    m_g2pPipeline->BindBuffer(0, *m_particleBuffer);
    m_g2pPipeline->BindBuffer(1, *m_gridBuffer);
    m_g2pPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_g2pPipeline->UpdateBindings();

    // Allocate a single command buffer for the entire simulation step
    VkDevice device = m_vulkan->GetDevice();
    VkCommandPool cmdPool = m_vulkan->GetComputeCommandPool();

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer;
    if (s_vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
        FLUID_LOG_ERROR("Failed to allocate batched command buffer");
        return;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    s_vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    // =========================================================================
    // Record all simulation stages with barriers between dependent stages
    // =========================================================================

    // Stage 1: P2G - transfer particle velocities to grid
    RecordP2G(cmdBuffer, params);
    RecordMemoryBarrier(cmdBuffer);

    // Stage 2: Normalize grid velocities
    RecordNormalize(cmdBuffer, params);
    RecordMemoryBarrier(cmdBuffer);

    // Stage 3: Apply external forces (gravity)
    RecordForces(cmdBuffer, params);
    RecordMemoryBarrier(cmdBuffer);

    // Stage 4: Compute divergence
    RecordDivergence(cmdBuffer, params);
    RecordMemoryBarrier(cmdBuffer);

    // Stage 5: Pressure solve (iterative)
    RecordPressure(cmdBuffer, params, params.pressureIterations);
    RecordMemoryBarrier(cmdBuffer);

    // Stage 6: Apply pressure gradient
    RecordGradient(cmdBuffer, params);
    RecordMemoryBarrier(cmdBuffer);

    // Stage 7: G2P - transfer grid velocities back to particles
    RecordG2P(cmdBuffer, params);

    s_vkEndCommandBuffer(cmdBuffer);

    // Submit and wait ONCE for the entire simulation step
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    VkQueue queue = m_vulkan->GetComputeQueue();
    s_vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    s_vkQueueWaitIdle(queue);  // Only ONE wait for entire simulation!

    s_vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

// =============================================================================
// Buffer Access
// =============================================================================

VkBuffer VulkanFluidCompute::GetParticleBuffer() const {
    return m_particleBuffer ? m_particleBuffer->GetVkBuffer() : nullptr;
}

size_t VulkanFluidCompute::GetParticleBufferSize() const {
    return m_particleBuffer ? m_particleBuffer->GetSizeBytes() : 0;
}

VkBuffer VulkanFluidCompute::GetGridBuffer() const {
    return m_gridBuffer ? m_gridBuffer->GetVkBuffer() : nullptr;
}

// =============================================================================
// Async Compute Implementation
// =============================================================================

// Additional Vulkan function pointers for async
static PFN_vkCreateFence s_vkCreateFence = nullptr;
static PFN_vkDestroyFence s_vkDestroyFence = nullptr;
static PFN_vkWaitForFences s_vkWaitForFences = nullptr;
static PFN_vkResetFences s_vkResetFences = nullptr;

void VulkanFluidCompute::BeginAsyncSimulation(const FluidSimParams& params) {
    if (!m_initialized || !m_vulkan || m_asyncInProgress) return;

    // Load fence functions if needed
    if (!s_vkCreateFence && vkGetInstanceProcAddr_External) {
        VkInstance instance = m_vulkan->GetInstance();
        s_vkCreateFence = reinterpret_cast<PFN_vkCreateFence>(
            vkGetInstanceProcAddr_External(instance, "vkCreateFence"));
        s_vkDestroyFence = reinterpret_cast<PFN_vkDestroyFence>(
            vkGetInstanceProcAddr_External(instance, "vkDestroyFence"));
        s_vkWaitForFences = reinterpret_cast<PFN_vkWaitForFences>(
            vkGetInstanceProcAddr_External(instance, "vkWaitForFences"));
        s_vkResetFences = reinterpret_cast<PFN_vkResetFences>(
            vkGetInstanceProcAddr_External(instance, "vkResetFences"));
    }

    VkDevice device = m_vulkan->GetDevice();
    VkCommandPool cmdPool = m_vulkan->GetComputeCommandPool();

    // Create fence if needed
    if (m_asyncFence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = 0;
        s_vkCreateFence(device, &fenceInfo, nullptr, &m_asyncFence);
    } else {
        s_vkResetFences(device, 1, &m_asyncFence);
    }

    // Ensure bindings are up to date
    m_p2gPipeline->BindBuffer(0, *m_particleBuffer);
    m_p2gPipeline->BindBuffer(1, *m_gridBuffer);
    m_p2gPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_p2gPipeline->UpdateBindings();

    m_normalizePipeline->BindBuffer(0, *m_particleBuffer);
    m_normalizePipeline->BindBuffer(1, *m_gridBuffer);
    m_normalizePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_normalizePipeline->UpdateBindings();

    m_forcesPipeline->BindBuffer(0, *m_particleBuffer);
    m_forcesPipeline->BindBuffer(1, *m_gridBuffer);
    m_forcesPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_forcesPipeline->UpdateBindings();

    m_divergencePipeline->BindBuffer(0, *m_particleBuffer);
    m_divergencePipeline->BindBuffer(1, *m_gridBuffer);
    m_divergencePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_divergencePipeline->UpdateBindings();

    m_pressurePipeline->BindBuffer(0, *m_particleBuffer);
    m_pressurePipeline->BindBuffer(1, *m_gridBuffer);
    m_pressurePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_pressurePipeline->UpdateBindings();

    m_gradientPipeline->BindBuffer(0, *m_particleBuffer);
    m_gradientPipeline->BindBuffer(1, *m_gridBuffer);
    m_gradientPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_gradientPipeline->UpdateBindings();

    m_g2pPipeline->BindBuffer(0, *m_particleBuffer);
    m_g2pPipeline->BindBuffer(1, *m_gridBuffer);
    m_g2pPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_g2pPipeline->UpdateBindings();

    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    s_vkAllocateCommandBuffers(device, &allocInfo, &m_asyncCmdBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    s_vkBeginCommandBuffer(m_asyncCmdBuffer, &beginInfo);

    // Record all simulation stages
    RecordP2G(m_asyncCmdBuffer, params);
    RecordMemoryBarrier(m_asyncCmdBuffer);
    RecordNormalize(m_asyncCmdBuffer, params);
    RecordMemoryBarrier(m_asyncCmdBuffer);
    RecordForces(m_asyncCmdBuffer, params);
    RecordMemoryBarrier(m_asyncCmdBuffer);
    RecordDivergence(m_asyncCmdBuffer, params);
    RecordMemoryBarrier(m_asyncCmdBuffer);
    RecordPressure(m_asyncCmdBuffer, params, params.pressureIterations);
    RecordMemoryBarrier(m_asyncCmdBuffer);
    RecordGradient(m_asyncCmdBuffer, params);
    RecordMemoryBarrier(m_asyncCmdBuffer);
    RecordG2P(m_asyncCmdBuffer, params);

    s_vkEndCommandBuffer(m_asyncCmdBuffer);

    // Submit with fence (non-blocking)
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_asyncCmdBuffer;

    VkQueue queue = m_vulkan->GetComputeQueue();
    s_vkQueueSubmit(queue, 1, &submitInfo, m_asyncFence);

    m_asyncInProgress = true;
}

void VulkanFluidCompute::WaitForSimulation() {
    if (!m_asyncInProgress || !m_vulkan) return;

    VkDevice device = m_vulkan->GetDevice();

    // Wait for the fence
    s_vkWaitForFences(device, 1, &m_asyncFence, VK_TRUE, UINT64_MAX);

    // Free command buffer
    VkCommandPool cmdPool = m_vulkan->GetComputeCommandPool();
    s_vkFreeCommandBuffers(device, cmdPool, 1, &m_asyncCmdBuffer);
    m_asyncCmdBuffer = VK_NULL_HANDLE;

    m_asyncInProgress = false;
}

// =============================================================================
// Particle Sorting Implementation
// =============================================================================

void VulkanFluidCompute::RecordCellIndexCompute(VkCommandBuffer cmd, const FluidSimParams& params) {
    if (!m_cellIndexPipeline || !m_cellIndexPipeline->IsValid()) return;

    s_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cellIndexPipeline->GetVkPipeline());
    s_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_cellIndexPipeline->GetVkPipelineLayout(), 0, 1,
                              &m_cellIndexPipeline->GetVkDescriptorSet(), 0, nullptr);
    s_vkCmdPushConstants(cmd, m_cellIndexPipeline->GetVkPipelineLayout(),
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    uint32_t groupCount = (params.particleCount + 255) / 256;
    s_vkCmdDispatch(cmd, groupCount, 1, 1);
}

void VulkanFluidCompute::RecordRadixSort(VkCommandBuffer cmd, const FluidSimParams& params) {
    if (!m_radixSortPipeline || !m_radixSortPipeline->IsValid()) return;

    // Simplified radix sort - in production would do full 4-pass radix sort
    // For now, just compute cell indices which still helps with locality

    s_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_radixSortPipeline->GetVkPipeline());
    s_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_radixSortPipeline->GetVkPipelineLayout(), 0, 1,
                              &m_radixSortPipeline->GetVkDescriptorSet(), 0, nullptr);

    struct SortParams {
        uint32_t particleCount;
        uint32_t bitOffset;
        uint32_t numWorkgroups;
        uint32_t pass;
    } sortParams;

    sortParams.particleCount = params.particleCount;
    sortParams.numWorkgroups = (params.particleCount + 255) / 256;

    // 4 passes of 8-bit radix sort (32-bit keys)
    for (uint32_t bitOffset = 0; bitOffset < 32; bitOffset += 8) {
        sortParams.bitOffset = bitOffset;

        // Pass 0: Histogram
        sortParams.pass = 0;
        s_vkCmdPushConstants(cmd, m_radixSortPipeline->GetVkPipelineLayout(),
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sortParams), &sortParams);
        s_vkCmdDispatch(cmd, sortParams.numWorkgroups, 1, 1);
        RecordMemoryBarrier(cmd);

        // Pass 1: Scatter
        sortParams.pass = 1;
        s_vkCmdPushConstants(cmd, m_radixSortPipeline->GetVkPipelineLayout(),
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sortParams), &sortParams);
        s_vkCmdDispatch(cmd, sortParams.numWorkgroups, 1, 1);
        RecordMemoryBarrier(cmd);
    }
}

void VulkanFluidCompute::RecordParticleReorder(VkCommandBuffer cmd, const FluidSimParams& params) {
    if (!m_reorderPipeline || !m_reorderPipeline->IsValid()) return;

    s_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_reorderPipeline->GetVkPipeline());
    s_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_reorderPipeline->GetVkPipelineLayout(), 0, 1,
                              &m_reorderPipeline->GetVkDescriptorSet(), 0, nullptr);

    struct ReorderParams {
        uint32_t particleCount;
    } reorderParams;
    reorderParams.particleCount = params.particleCount;

    s_vkCmdPushConstants(cmd, m_reorderPipeline->GetVkPipelineLayout(),
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(reorderParams), &reorderParams);

    uint32_t groupCount = (params.particleCount + 255) / 256;
    s_vkCmdDispatch(cmd, groupCount, 1, 1);
}

void VulkanFluidCompute::SortParticlesByCell(const FluidSimParams& params) {
    if (!m_initialized || !m_vulkan) return;
    if (!m_cellIndexPipeline || !m_cellIndexBuffer) return;

    // Update bindings for sorting pipelines
    m_cellIndexPipeline->BindBuffer(0, *m_particleBuffer);
    m_cellIndexPipeline->BindBuffer(1, *m_cellIndexBuffer);
    m_cellIndexPipeline->BindBuffer(2, *m_particleIndexBuffer);
    m_cellIndexPipeline->UpdateBindings();

    if (m_radixSortPipeline) {
        m_radixSortPipeline->BindBuffer(0, *m_cellIndexBuffer);
        m_radixSortPipeline->BindBuffer(1, *m_particleIndexBuffer);
        m_radixSortPipeline->BindBuffer(2, *m_tempCellIndexBuffer);
        m_radixSortPipeline->BindBuffer(3, *m_tempParticleIndexBuffer);
        m_radixSortPipeline->BindBuffer(4, *m_histogramBuffer);
        m_radixSortPipeline->UpdateBindings();
    }

    if (m_reorderPipeline && m_sortedParticleBuffer) {
        m_reorderPipeline->BindBuffer(0, *m_particleBuffer);
        m_reorderPipeline->BindBuffer(1, *m_sortedParticleBuffer);
        m_reorderPipeline->BindBuffer(2, *m_particleIndexBuffer);
        m_reorderPipeline->UpdateBindings();
    }

    VkDevice device = m_vulkan->GetDevice();
    VkCommandPool cmdPool = m_vulkan->GetComputeCommandPool();

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer;
    s_vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    s_vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    // Step 1: Compute cell indices
    RecordCellIndexCompute(cmdBuffer, params);
    RecordMemoryBarrier(cmdBuffer);

    // Step 2: Radix sort by cell index
    RecordRadixSort(cmdBuffer, params);
    RecordMemoryBarrier(cmdBuffer);

    // Step 3: Reorder particles
    RecordParticleReorder(cmdBuffer, params);

    s_vkEndCommandBuffer(cmdBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    VkQueue queue = m_vulkan->GetComputeQueue();
    s_vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    s_vkQueueWaitIdle(queue);

    s_vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);

    // Swap particle buffers so sorted particles become the active buffer
    std::swap(m_particleBuffer, m_sortedParticleBuffer);
}

void VulkanFluidCompute::DispatchFullStepSorted(const FluidSimParams& params) {
    // Sort particles by cell for optimal cache performance
    SortParticlesByCell(params);

    // Run simulation with sorted particles
    DispatchFullStepBatched(params);
}

} // namespace WulfNet
