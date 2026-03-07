// =============================================================================
// WulfNet Engine - SWE GPU Compute Implementation
// =============================================================================

#include "SWEComputeGPU.h"
#include "WulfNet/Core/Logging/Logger.h"

#ifdef WULFNET_PLATFORM_WINDOWS
    #define VK_USE_PLATFORM_WIN32_KHR
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <filesystem>
#include <cstring>
#include <iostream>

namespace WulfNet {

// =============================================================================
// Vulkan function pointers for batched dispatch
// =============================================================================
extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_External;

static PFN_vkAllocateCommandBuffers  s_swevkAllocateCommandBuffers = nullptr;
static PFN_vkFreeCommandBuffers      s_swevkFreeCommandBuffers = nullptr;
static PFN_vkBeginCommandBuffer      s_swevkBeginCommandBuffer = nullptr;
static PFN_vkEndCommandBuffer        s_swevkEndCommandBuffer = nullptr;
static PFN_vkCmdBindPipeline         s_swevkCmdBindPipeline = nullptr;
static PFN_vkCmdBindDescriptorSets   s_swevkCmdBindDescriptorSets = nullptr;
static PFN_vkCmdPushConstants        s_swevkCmdPushConstants = nullptr;
static PFN_vkCmdDispatch             s_swevkCmdDispatch = nullptr;
static PFN_vkCmdPipelineBarrier      s_swevkCmdPipelineBarrier = nullptr;
static PFN_vkCmdCopyBuffer           s_swevkCmdCopyBuffer = nullptr;
static PFN_vkQueueSubmit             s_swevkQueueSubmit = nullptr;
static PFN_vkQueueWaitIdle           s_swevkQueueWaitIdle = nullptr;
static bool s_sweFunctionsLoaded = false;

static bool LoadSWEVkFunctions() {
    if (s_sweFunctionsLoaded) return true;
    if (!IsVulkanContextInitialized()) return false;

    VkInstance instance = GetVulkanContext().GetInstance();
    auto getProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetVulkanInstanceProcAddr());
    if (!getProc) getProc = vkGetInstanceProcAddr_External;
    if (!getProc) return false;

    #define LOAD_SWE_VK(name) \
        s_swe##name = reinterpret_cast<PFN_##name>(getProc(instance, #name))

    LOAD_SWE_VK(vkAllocateCommandBuffers);
    LOAD_SWE_VK(vkFreeCommandBuffers);
    LOAD_SWE_VK(vkBeginCommandBuffer);
    LOAD_SWE_VK(vkEndCommandBuffer);
    LOAD_SWE_VK(vkCmdBindPipeline);
    LOAD_SWE_VK(vkCmdBindDescriptorSets);
    LOAD_SWE_VK(vkCmdPushConstants);
    LOAD_SWE_VK(vkCmdDispatch);
    LOAD_SWE_VK(vkCmdPipelineBarrier);
    LOAD_SWE_VK(vkCmdCopyBuffer);
    LOAD_SWE_VK(vkQueueSubmit);
    LOAD_SWE_VK(vkQueueWaitIdle);

    #undef LOAD_SWE_VK

    s_sweFunctionsLoaded = true;
    return true;
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

SWEComputeGPU::SWEComputeGPU() = default;

SWEComputeGPU::~SWEComputeGPU() {
    Shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool SWEComputeGPU::Initialize(uint32_t gridSizeX, uint32_t gridSizeZ,
                                const std::string& shaderPath) {
    if (m_initialized) {
        WULFNET_WARNING("SWECompute", "Already initialized");
        return true;
    }

    if (!IsVulkanContextInitialized()) {
        WULFNET_ERROR("SWECompute", "VulkanContext not initialized — GPU SWE unavailable");
        return false;
    }

    if (!LoadSWEVkFunctions()) {
        WULFNET_ERROR("SWECompute", "Failed to load Vulkan functions");
        return false;
    }

    m_gridSizeX = gridSizeX;
    m_gridSizeZ = gridSizeZ;
    m_totalCells = gridSizeX * gridSizeZ;

    // Workgroup dispatch dimensions (16x16 per workgroup)
    m_dispatchX = (gridSizeX + 15) / 16;
    m_dispatchZ = (gridSizeZ + 15) / 16;

    WULFNET_INFO("SWECompute", "Initializing GPU SWE compute");
    WULFNET_INFO("SWECompute", "  Grid: " + std::to_string(gridSizeX) + "x" +
                 std::to_string(gridSizeZ) + " = " + std::to_string(m_totalCells) + " cells");
    WULFNET_INFO("SWECompute", "  Dispatch: " + std::to_string(m_dispatchX) + "x" +
                 std::to_string(m_dispatchZ) + " workgroups");

    if (!CreateBuffers()) {
        WULFNET_ERROR("SWECompute", "Failed to create GPU buffers");
        return false;
    }

    if (!LoadShaders(shaderPath)) {
        WULFNET_ERROR("SWECompute", "Failed to load compute shaders");
        return false;
    }

    m_initialized = true;
    WULFNET_INFO("SWECompute", "GPU SWE compute ready");
    return true;
}

void SWEComputeGPU::Shutdown() {
    if (!m_initialized) return;

    WULFNET_INFO("SWECompute", "Shutting down GPU SWE compute");

    if (IsVulkanContextInitialized()) {
        GetVulkanContext().WaitIdle();
    }

    m_velocityPipeline.reset();
    m_outflowPipeline.reset();
    m_gatherPipeline.reset();
    m_boundaryPipeline.reset();

    m_gridBuffer.reset();
    m_snapshotBuffer.reset();
    m_outflowBuffer.reset();

    m_initialized = false;
}

// =============================================================================
// Buffer Creation
// =============================================================================

bool SWEComputeGPU::CreateBuffers() {
    try {
        // Each WaterCell = 4 floats (vec4). Grid buffer = totalCells * 4 floats.
        const size_t gridFloats = static_cast<size_t>(m_totalCells) * 4;

        m_gridBuffer = std::make_unique<ComputeBuffer<float>>(
            gridFloats, GPUBufferUsage::ComputeStorage, GPUMemoryLocation::DeviceLocal);

        m_snapshotBuffer = std::make_unique<ComputeBuffer<float>>(
            gridFloats, GPUBufferUsage::ComputeStorage, GPUMemoryLocation::DeviceLocal);

        // Outflow = 4 floats per cell (L, R, B, F), stored as vec4
        const size_t outflowFloats = static_cast<size_t>(m_totalCells) * 4;
        m_outflowBuffer = std::make_unique<ComputeBuffer<float>>(
            outflowFloats, GPUBufferUsage::ComputeStorage, GPUMemoryLocation::DeviceLocal);

        size_t totalBytes = (gridFloats * 2 + outflowFloats) * sizeof(float);
        WULFNET_INFO("SWECompute", "  GPU buffers: " + std::to_string(totalBytes / 1024) + " KB total");

        return m_gridBuffer->IsValid() && m_snapshotBuffer->IsValid() && m_outflowBuffer->IsValid();
    }
    catch (const std::exception& e) {
        WULFNET_ERROR("SWECompute", "Buffer creation failed: " + std::string(e.what()));
        return false;
    }
}

// =============================================================================
// Shader Loading
// =============================================================================

bool SWEComputeGPU::LoadShaders(const std::string& shaderPath) {
    PushConstantRange pushRange{0, sizeof(SWESimParams)};

    // --- Velocity shader: reads snapshot (binding 0+1), writes grid ---
    // Shader layout:
    //   binding 0 = GridBuffer (rw)
    //   binding 1 = SnapBuffer (ro)
    std::vector<ShaderBinding> velocityBindings = {
        {0, ShaderBindingType::StorageBuffer, "grid"},
        {1, ShaderBindingType::StorageBuffer, "snapshot"}
    };

    // --- Outflow shader: reads snapshot, writes outflow ---
    //   binding 0 = SnapBuffer (ro)
    //   binding 1 = OutflowBuf (wo)
    std::vector<ShaderBinding> outflowBindings = {
        {0, ShaderBindingType::StorageBuffer, "snapshot"},
        {1, ShaderBindingType::StorageBuffer, "outflow"}
    };

    // --- Gather shader: reads snapshot + outflow, writes grid ---
    //   binding 0 = GridBuffer (rw)
    //   binding 1 = SnapBuffer (ro)
    //   binding 2 = OutflowBuf (ro)
    std::vector<ShaderBinding> gatherBindings = {
        {0, ShaderBindingType::StorageBuffer, "grid"},
        {1, ShaderBindingType::StorageBuffer, "snapshot"},
        {2, ShaderBindingType::StorageBuffer, "outflow"}
    };

    // --- Boundary shader: reads/writes grid ---
    //   binding 0 = GridBuffer (rw)
    std::vector<ShaderBinding> boundaryBindings = {
        {0, ShaderBindingType::StorageBuffer, "grid"}
    };

    auto loadPipeline = [&](const std::string& name,
                            const std::vector<ShaderBinding>& bindings)
        -> std::unique_ptr<ComputePipeline>
    {
        std::string path = shaderPath + "/" + name + ".spv";
        if (!std::filesystem::exists(path)) {
            WULFNET_ERROR("SWECompute", "Shader not found: " + path);
            return nullptr;
        }

        auto pipeline = std::make_unique<ComputePipeline>();
        if (!pipeline->CreateFromFile(path, bindings, pushRange)) {
            WULFNET_ERROR("SWECompute", "Failed to create pipeline: " + name);
            return nullptr;
        }

        WULFNET_INFO("SWECompute", "  Loaded shader: " + name);
        return pipeline;
    };

    m_velocityPipeline = loadPipeline("swe_velocity", velocityBindings);
    m_outflowPipeline  = loadPipeline("swe_outflow", outflowBindings);
    m_gatherPipeline   = loadPipeline("swe_gather", gatherBindings);
    m_boundaryPipeline = loadPipeline("swe_boundary", boundaryBindings);

    bool allLoaded = m_velocityPipeline && m_outflowPipeline &&
                     m_gatherPipeline && m_boundaryPipeline;

    if (!allLoaded) {
        WULFNET_WARNING("SWECompute", "Some shaders failed to load — GPU SWE disabled");
        return false;
    }

    return true;
}

// =============================================================================
// Data Transfer
// =============================================================================

bool SWEComputeGPU::UploadGrid(const float* gridData, uint32_t cellCount) {
    if (!m_initialized || !m_gridBuffer) return false;
    if (cellCount != m_totalCells) {
        WULFNET_ERROR("SWECompute", "Upload cell count mismatch: " +
                     std::to_string(cellCount) + " vs " + std::to_string(m_totalCells));
        return false;
    }
    // Upload all 4 floats per cell
    return m_gridBuffer->Upload(gridData, static_cast<size_t>(cellCount) * 4, 0);
}

bool SWEComputeGPU::DownloadGrid(float* gridData, uint32_t cellCount) {
    if (!m_initialized || !m_gridBuffer) return false;
    if (cellCount != m_totalCells) {
        WULFNET_ERROR("SWECompute", "Download cell count mismatch");
        return false;
    }
    return m_gridBuffer->Download(gridData, static_cast<size_t>(cellCount) * 4, 0);
}

bool SWEComputeGPU::DownloadGridAsync() {
    if (!m_initialized || !m_gridBuffer) return false;
    return m_gridBuffer->DownloadAsync();
}

bool SWEComputeGPU::IsReadbackReady() const {
    if (!m_initialized || !m_gridBuffer) return false;
    return m_gridBuffer->IsDownloadReady();
}

bool SWEComputeGPU::GetReadbackData(std::vector<float>& outData) {
    if (!m_initialized || !m_gridBuffer) return false;
    return m_gridBuffer->GetDownloadedData(outData);
}

// =============================================================================
// Batched Dispatch — Record Commands
// =============================================================================

void SWEComputeGPU::RecordMemoryBarrier(void* cmd) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    s_swevkCmdPipelineBarrier(
        static_cast<VkCommandBuffer>(cmd),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr
    );
}

void SWEComputeGPU::RecordSnapshotCopy(void* cmd) {
    // Copy grid → snapshot (GPU-side memcpy for read-only access during phases)
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = static_cast<VkDeviceSize>(m_totalCells) * 4 * sizeof(float);

    s_swevkCmdCopyBuffer(
        static_cast<VkCommandBuffer>(cmd),
        m_gridBuffer->GetVkBuffer(),
        m_snapshotBuffer->GetVkBuffer(),
        1, &copyRegion
    );

    // Barrier: transfer write → shader read
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    s_swevkCmdPipelineBarrier(
        static_cast<VkCommandBuffer>(cmd),
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr
    );
}

void SWEComputeGPU::RecordVelocity(void* cmd, const SWESimParams& params) {
    if (!m_velocityPipeline || !m_velocityPipeline->IsValid()) return;

    auto cmdBuf = static_cast<VkCommandBuffer>(cmd);
    VkDescriptorSet descSet = m_velocityPipeline->GetVkDescriptorSet();

    s_swevkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_velocityPipeline->GetVkPipeline());
    s_swevkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  m_velocityPipeline->GetVkPipelineLayout(),
                                  0, 1, &descSet, 0, nullptr);
    s_swevkCmdPushConstants(cmdBuf, m_velocityPipeline->GetVkPipelineLayout(),
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    s_swevkCmdDispatch(cmdBuf, m_dispatchX, m_dispatchZ, 1);
}

void SWEComputeGPU::RecordOutflow(void* cmd, const SWESimParams& params) {
    if (!m_outflowPipeline || !m_outflowPipeline->IsValid()) return;

    auto cmdBuf = static_cast<VkCommandBuffer>(cmd);
    VkDescriptorSet descSet = m_outflowPipeline->GetVkDescriptorSet();

    s_swevkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_outflowPipeline->GetVkPipeline());
    s_swevkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  m_outflowPipeline->GetVkPipelineLayout(),
                                  0, 1, &descSet, 0, nullptr);
    s_swevkCmdPushConstants(cmdBuf, m_outflowPipeline->GetVkPipelineLayout(),
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    s_swevkCmdDispatch(cmdBuf, m_dispatchX, m_dispatchZ, 1);
}

void SWEComputeGPU::RecordGather(void* cmd, const SWESimParams& params) {
    if (!m_gatherPipeline || !m_gatherPipeline->IsValid()) return;

    auto cmdBuf = static_cast<VkCommandBuffer>(cmd);
    VkDescriptorSet descSet = m_gatherPipeline->GetVkDescriptorSet();

    s_swevkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_gatherPipeline->GetVkPipeline());
    s_swevkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  m_gatherPipeline->GetVkPipelineLayout(),
                                  0, 1, &descSet, 0, nullptr);
    s_swevkCmdPushConstants(cmdBuf, m_gatherPipeline->GetVkPipelineLayout(),
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    s_swevkCmdDispatch(cmdBuf, m_dispatchX, m_dispatchZ, 1);
}

void SWEComputeGPU::RecordBoundary(void* cmd, const SWESimParams& params) {
    if (!m_boundaryPipeline || !m_boundaryPipeline->IsValid()) return;

    auto cmdBuf = static_cast<VkCommandBuffer>(cmd);
    VkDescriptorSet descSet = m_boundaryPipeline->GetVkDescriptorSet();

    s_swevkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_boundaryPipeline->GetVkPipeline());
    s_swevkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  m_boundaryPipeline->GetVkPipelineLayout(),
                                  0, 1, &descSet, 0, nullptr);
    s_swevkCmdPushConstants(cmdBuf, m_boundaryPipeline->GetVkPipelineLayout(),
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);

    s_swevkCmdDispatch(cmdBuf, m_dispatchX, m_dispatchZ, 1);
}

// =============================================================================
// Full Step — Batched (Maximum Performance)
// =============================================================================

void SWEComputeGPU::StepSWE(const SWESimParams& params) {
    if (!m_initialized) return;

    // Bind all buffers to their pipelines
    // Velocity: binding 0 = grid (rw), binding 1 = snapshot (ro)
    m_velocityPipeline->BindBuffer(0, *m_gridBuffer);
    m_velocityPipeline->BindBuffer(1, *m_snapshotBuffer);
    m_velocityPipeline->UpdateBindings();

    // Outflow: binding 0 = snapshot (ro), binding 1 = outflow (wo)
    m_outflowPipeline->BindBuffer(0, *m_snapshotBuffer);
    m_outflowPipeline->BindBuffer(1, *m_outflowBuffer);
    m_outflowPipeline->UpdateBindings();

    // Gather: binding 0 = grid (rw), binding 1 = snapshot (ro), binding 2 = outflow (ro)
    m_gatherPipeline->BindBuffer(0, *m_gridBuffer);
    m_gatherPipeline->BindBuffer(1, *m_snapshotBuffer);
    m_gatherPipeline->BindBuffer(2, *m_outflowBuffer);
    m_gatherPipeline->UpdateBindings();

    // Boundary: binding 0 = grid (rw)
    m_boundaryPipeline->BindBuffer(0, *m_gridBuffer);
    m_boundaryPipeline->UpdateBindings();

    // Allocate single command buffer for entire step
    VkDevice device = GetVulkanContext().GetDevice();
    VkCommandPool cmdPool = GetVulkanContext().GetComputeCommandPool();

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer;
    if (s_swevkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
        WULFNET_ERROR("SWECompute", "Failed to allocate command buffer");
        return;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    s_swevkBeginCommandBuffer(cmdBuffer, &beginInfo);

    // Stage 0: Copy grid → snapshot (GPU memcpy)
    RecordSnapshotCopy(cmdBuffer);

    // Stage 1: Velocity update (reads snapshot, writes grid vx/vz)
    RecordVelocity(cmdBuffer, params);
    RecordMemoryBarrier(cmdBuffer);

    // Stage 2: Outflow computation (reads snapshot, writes outflow)
    RecordOutflow(cmdBuffer, params);
    RecordMemoryBarrier(cmdBuffer);

    // Stage 3: Gather (reads snapshot + outflow, writes grid waterHeight)
    RecordGather(cmdBuffer, params);
    RecordMemoryBarrier(cmdBuffer);

    // Stage 4: Boundary conditions (reads/writes grid)
    RecordBoundary(cmdBuffer, params);

    s_swevkEndCommandBuffer(cmdBuffer);

    // Submit and wait ONCE for entire step
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    VkQueue queue = GetVulkanContext().GetComputeQueue();
    s_swevkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    s_swevkQueueWaitIdle(queue);

    s_swevkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

// =============================================================================
// Full Step — Unbatched (Debugging)
// =============================================================================

void SWEComputeGPU::StepSWEUnbatched(const SWESimParams& params) {
    if (!m_initialized) return;

    // Copy grid → snapshot on GPU
    GetVulkanContext().SubmitAndWait([&](void* cmd) {
        RecordSnapshotCopy(cmd);
    });

    // Phase 1: Velocity
    m_velocityPipeline->BindBuffer(0, *m_gridBuffer);
    m_velocityPipeline->BindBuffer(1, *m_snapshotBuffer);
    m_velocityPipeline->UpdateBindings();
    m_velocityPipeline->SetPushConstants(params);
    m_velocityPipeline->DispatchAndWait(m_dispatchX, m_dispatchZ, 1);

    // Phase 2: Outflow
    m_outflowPipeline->BindBuffer(0, *m_snapshotBuffer);
    m_outflowPipeline->BindBuffer(1, *m_outflowBuffer);
    m_outflowPipeline->UpdateBindings();
    m_outflowPipeline->SetPushConstants(params);
    m_outflowPipeline->DispatchAndWait(m_dispatchX, m_dispatchZ, 1);

    // Phase 3: Gather
    m_gatherPipeline->BindBuffer(0, *m_gridBuffer);
    m_gatherPipeline->BindBuffer(1, *m_snapshotBuffer);
    m_gatherPipeline->BindBuffer(2, *m_outflowBuffer);
    m_gatherPipeline->UpdateBindings();
    m_gatherPipeline->SetPushConstants(params);
    m_gatherPipeline->DispatchAndWait(m_dispatchX, m_dispatchZ, 1);

    // Phase 4: Boundary
    m_boundaryPipeline->BindBuffer(0, *m_gridBuffer);
    m_boundaryPipeline->UpdateBindings();
    m_boundaryPipeline->SetPushConstants(params);
    m_boundaryPipeline->DispatchAndWait(m_dispatchX, m_dispatchZ, 1);
}

} // namespace WulfNet
