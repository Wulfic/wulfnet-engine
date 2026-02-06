// =============================================================================
// WulfNet Engine - Compute Pipeline Implementation
// =============================================================================

#include "WulfNet/Compute/Shaders/ComputePipeline.h"
#include "WulfNet/Compute/Vulkan/VulkanContext.h"
#include "WulfNet/Core/Logging/Logger.h"
#include "WulfNet/Core/Profiling/Profiler.h"

#ifdef WULFNET_PLATFORM_WINDOWS
    #define VK_USE_PLATFORM_WIN32_KHR
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <fstream>
#include <cstring>

namespace WulfNet {

// =============================================================================
// External Vulkan Function Declarations (loaded in VulkanContext)
// =============================================================================

extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_External;

// Local function pointers for pipeline operations
static PFN_vkCreateShaderModule s_vkCreateShaderModule = nullptr;
static PFN_vkDestroyShaderModule s_vkDestroyShaderModule = nullptr;
static PFN_vkCreateDescriptorSetLayout s_vkCreateDescriptorSetLayout = nullptr;
static PFN_vkDestroyDescriptorSetLayout s_vkDestroyDescriptorSetLayout = nullptr;
static PFN_vkCreatePipelineLayout s_vkCreatePipelineLayout = nullptr;
static PFN_vkDestroyPipelineLayout s_vkDestroyPipelineLayout = nullptr;
static PFN_vkCreateComputePipelines s_vkCreateComputePipelines = nullptr;
static PFN_vkDestroyPipeline s_vkDestroyPipeline = nullptr;
static PFN_vkAllocateDescriptorSets s_vkAllocateDescriptorSets = nullptr;
static PFN_vkFreeDescriptorSets s_vkFreeDescriptorSets = nullptr;
static PFN_vkUpdateDescriptorSets s_vkUpdateDescriptorSets = nullptr;
static PFN_vkAllocateCommandBuffers s_vkAllocateCommandBuffers = nullptr;
static PFN_vkFreeCommandBuffers s_vkFreeCommandBuffers = nullptr;
static PFN_vkBeginCommandBuffer s_vkBeginCommandBuffer = nullptr;
static PFN_vkEndCommandBuffer s_vkEndCommandBuffer = nullptr;
static PFN_vkCmdBindPipeline s_vkCmdBindPipeline = nullptr;
static PFN_vkCmdBindDescriptorSets s_vkCmdBindDescriptorSets = nullptr;
static PFN_vkCmdPushConstants s_vkCmdPushConstants = nullptr;
static PFN_vkCmdDispatch s_vkCmdDispatch = nullptr;
static PFN_vkQueueSubmit s_vkQueueSubmit = nullptr;
static PFN_vkQueueWaitIdle s_vkQueueWaitIdle = nullptr;
static PFN_vkCreateFence s_vkCreateFence = nullptr;
static PFN_vkDestroyFence s_vkDestroyFence = nullptr;
static PFN_vkWaitForFences s_vkWaitForFences = nullptr;

static bool s_pipelineFunctionsLoaded = false;

static bool LoadPipelineFunctions() {
    if (s_pipelineFunctionsLoaded) return true;
    if (!IsVulkanContextInitialized()) return false;

    VkInstance instance = GetVulkanContext().GetInstance();

    // Get vkGetInstanceProcAddr from VulkanContext first
    auto getProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetVulkanInstanceProcAddr());

    // Update the external pointer so other code can use it too
    if (getProc && !vkGetInstanceProcAddr_External) {
        vkGetInstanceProcAddr_External = getProc;
    }

    // Fall back to external if VulkanContext doesn't have it
    if (!getProc) {
        getProc = vkGetInstanceProcAddr_External;
    }

    if (!getProc) {
        WULFNET_ERROR("Compute", "Cannot get vkGetInstanceProcAddr for pipeline functions");
        return false;
    }

    #define LOAD_VK_FUNC(name) \
        s_##name = reinterpret_cast<PFN_##name>(getProc(instance, #name))

    LOAD_VK_FUNC(vkCreateShaderModule);
    LOAD_VK_FUNC(vkDestroyShaderModule);
    LOAD_VK_FUNC(vkCreateDescriptorSetLayout);
    LOAD_VK_FUNC(vkDestroyDescriptorSetLayout);
    LOAD_VK_FUNC(vkCreatePipelineLayout);
    LOAD_VK_FUNC(vkDestroyPipelineLayout);
    LOAD_VK_FUNC(vkCreateComputePipelines);
    LOAD_VK_FUNC(vkDestroyPipeline);
    LOAD_VK_FUNC(vkAllocateDescriptorSets);
    LOAD_VK_FUNC(vkFreeDescriptorSets);
    LOAD_VK_FUNC(vkUpdateDescriptorSets);
    LOAD_VK_FUNC(vkAllocateCommandBuffers);
    LOAD_VK_FUNC(vkFreeCommandBuffers);
    LOAD_VK_FUNC(vkBeginCommandBuffer);
    LOAD_VK_FUNC(vkEndCommandBuffer);
    LOAD_VK_FUNC(vkCmdBindPipeline);
    LOAD_VK_FUNC(vkCmdBindDescriptorSets);
    LOAD_VK_FUNC(vkCmdPushConstants);
    LOAD_VK_FUNC(vkCmdDispatch);
    LOAD_VK_FUNC(vkQueueSubmit);
    LOAD_VK_FUNC(vkQueueWaitIdle);
    LOAD_VK_FUNC(vkCreateFence);
    LOAD_VK_FUNC(vkDestroyFence);
    LOAD_VK_FUNC(vkWaitForFences);

    #undef LOAD_VK_FUNC

    s_pipelineFunctionsLoaded = true;
    return true;
}

// Store the external proc addr
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_External = nullptr;

// =============================================================================
// ComputePipeline Implementation
// =============================================================================

ComputePipeline::ComputePipeline() {
    m_pushConstantData.resize(128, 0); // Max push constant size
}

ComputePipeline::~ComputePipeline() {
    Destroy();
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : m_shaderModule(other.m_shaderModule)
    , m_descriptorSetLayout(other.m_descriptorSetLayout)
    , m_pipelineLayout(other.m_pipelineLayout)
    , m_pipeline(other.m_pipeline)
    , m_descriptorSet(other.m_descriptorSet)
    , m_localSizeX(other.m_localSizeX)
    , m_localSizeY(other.m_localSizeY)
    , m_localSizeZ(other.m_localSizeZ)
    , m_name(std::move(other.m_name))
    , m_pendingBindings(std::move(other.m_pendingBindings))
    , m_pushConstantData(std::move(other.m_pushConstantData))
    , m_pushConstantSize(other.m_pushConstantSize)
{
    other.m_shaderModule = nullptr;
    other.m_descriptorSetLayout = nullptr;
    other.m_pipelineLayout = nullptr;
    other.m_pipeline = nullptr;
    other.m_descriptorSet = nullptr;
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
    if (this != &other) {
        Destroy();

        m_shaderModule = other.m_shaderModule;
        m_descriptorSetLayout = other.m_descriptorSetLayout;
        m_pipelineLayout = other.m_pipelineLayout;
        m_pipeline = other.m_pipeline;
        m_descriptorSet = other.m_descriptorSet;
        m_localSizeX = other.m_localSizeX;
        m_localSizeY = other.m_localSizeY;
        m_localSizeZ = other.m_localSizeZ;
        m_name = std::move(other.m_name);
        m_pendingBindings = std::move(other.m_pendingBindings);
        m_pushConstantData = std::move(other.m_pushConstantData);
        m_pushConstantSize = other.m_pushConstantSize;

        other.m_shaderModule = nullptr;
        other.m_descriptorSetLayout = nullptr;
        other.m_pipelineLayout = nullptr;
        other.m_pipeline = nullptr;
        other.m_descriptorSet = nullptr;
    }
    return *this;
}

bool ComputePipeline::Create(const ComputePipelineDesc& desc) {
    WULFNET_ZONE();

    if (!LoadPipelineFunctions()) {
        WULFNET_ERROR("Compute", "Failed to load pipeline Vulkan functions");
        return false;
    }

    Destroy();

    m_localSizeX = desc.localSizeX;
    m_localSizeY = desc.localSizeY;
    m_localSizeZ = desc.localSizeZ;
    m_name = desc.name;
    m_pushConstantSize = desc.pushConstants.size;

    if (!CreateShaderModule(desc.spirvCode)) {
        WULFNET_ERROR("Compute", "Failed to create shader module for '" + m_name + "'");
        return false;
    }

    if (!CreateDescriptorSetLayout(desc.bindings)) {
        WULFNET_ERROR("Compute", "Failed to create descriptor set layout for '" + m_name + "'");
        Destroy();
        return false;
    }

    if (!CreatePipelineLayout(desc.pushConstants)) {
        WULFNET_ERROR("Compute", "Failed to create pipeline layout for '" + m_name + "'");
        Destroy();
        return false;
    }

    if (!CreatePipeline(desc.entryPoint)) {
        WULFNET_ERROR("Compute", "Failed to create compute pipeline for '" + m_name + "'");
        Destroy();
        return false;
    }

    if (!AllocateDescriptorSet()) {
        WULFNET_ERROR("Compute", "Failed to allocate descriptor set for '" + m_name + "'");
        Destroy();
        return false;
    }

    WULFNET_DEBUG("Compute", "Created compute pipeline '" + m_name + "'");
    return true;
}

bool ComputePipeline::CreateFromFile(const std::string& spirvPath,
                                     const std::vector<ShaderBinding>& bindings,
                                     const PushConstantRange& pushConstants) {
    auto spirv = ShaderUtils::LoadSPIRV(spirvPath);
    if (spirv.empty()) {
        WULFNET_ERROR("Compute", "Failed to load SPIR-V from '" + spirvPath + "'");
        return false;
    }

    ComputePipelineDesc desc;
    desc.spirvCode = std::move(spirv);
    desc.bindings = bindings;
    desc.pushConstants = pushConstants;
    desc.name = spirvPath;

    return Create(desc);
}

void ComputePipeline::Destroy() {
    if (!IsVulkanContextInitialized()) return;

    VkDevice device = GetVulkanContext().GetDevice();

    if (m_descriptorSet && s_vkFreeDescriptorSets) {
        s_vkFreeDescriptorSets(device, GetVulkanContext().GetDescriptorPool(),
                              1, &m_descriptorSet);
        m_descriptorSet = nullptr;
    }

    if (m_pipeline && s_vkDestroyPipeline) {
        s_vkDestroyPipeline(device, m_pipeline, nullptr);
        m_pipeline = nullptr;
    }

    if (m_pipelineLayout && s_vkDestroyPipelineLayout) {
        s_vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = nullptr;
    }

    if (m_descriptorSetLayout && s_vkDestroyDescriptorSetLayout) {
        s_vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
        m_descriptorSetLayout = nullptr;
    }

    if (m_shaderModule && s_vkDestroyShaderModule) {
        s_vkDestroyShaderModule(device, m_shaderModule, nullptr);
        m_shaderModule = nullptr;
    }

    m_pendingBindings.clear();
}

void ComputePipeline::BindBuffer(uint32_t binding, const GPUBufferBase& buffer) {
    BindBuffer(binding, buffer, 0, buffer.GetSizeBytes());
}

void ComputePipeline::BindBuffer(uint32_t binding, const GPUBufferBase& buffer,
                                 size_t offset, size_t range) {
    PendingBinding pb;
    pb.binding = binding;
    pb.buffer = buffer.GetVkBuffer();
    pb.offset = offset;
    pb.range = range;

    // Replace existing binding if present
    for (auto& existing : m_pendingBindings) {
        if (existing.binding == binding) {
            existing = pb;
            return;
        }
    }

    m_pendingBindings.push_back(pb);
}

bool ComputePipeline::UpdateBindings() {
    WULFNET_ZONE();

    if (!IsValid() || m_pendingBindings.empty()) return true;

    VkDevice device = GetVulkanContext().GetDevice();

    std::vector<VkDescriptorBufferInfo> bufferInfos(m_pendingBindings.size());
    std::vector<VkWriteDescriptorSet> writes(m_pendingBindings.size());

    for (size_t i = 0; i < m_pendingBindings.size(); i++) {
        const auto& pb = m_pendingBindings[i];

        bufferInfos[i].buffer = pb.buffer;
        bufferInfos[i].offset = pb.offset;
        bufferInfos[i].range = pb.range;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = nullptr;
        writes[i].dstSet = m_descriptorSet;
        writes[i].dstBinding = pb.binding;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufferInfos[i];
        writes[i].pImageInfo = nullptr;
        writes[i].pTexelBufferView = nullptr;
    }

    s_vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                             writes.data(), 0, nullptr);

    m_pendingBindings.clear();
    return true;
}

void ComputePipeline::SetPushConstantsRaw(const void* data, uint32_t size, uint32_t offset) {
    if (offset + size > m_pushConstantData.size()) {
        WULFNET_ERROR("Compute", "Push constant size exceeds maximum (128 bytes)");
        return;
    }
    std::memcpy(m_pushConstantData.data() + offset, data, size);
}

void ComputePipeline::Dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    WULFNET_ZONE();

    if (!IsValid()) {
        WULFNET_ERROR("Compute", "Cannot dispatch invalid pipeline");
        return;
    }

    // Ensure bindings are updated
    UpdateBindings();

    // Create and record command buffer
    VkDevice device = GetVulkanContext().GetDevice();
    VkCommandPool cmdPool = GetVulkanContext().GetComputeCommandPool();

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer;
    if (s_vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to allocate command buffer");
        return;
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    s_vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    // Bind pipeline
    s_vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

    // Bind descriptor set
    s_vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    // Push constants
    if (m_pushConstantSize > 0) {
        s_vkCmdPushConstants(cmdBuffer, m_pipelineLayout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            m_pushConstantSize, m_pushConstantData.data());
    }

    // Dispatch
    s_vkCmdDispatch(cmdBuffer, groupCountX, groupCountY, groupCountZ);

    s_vkEndCommandBuffer(cmdBuffer);

    // Submit
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    VkQueue queue = GetVulkanContext().GetComputeQueue();
    s_vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);

    // Free command buffer (will be available after queue idle)
    s_vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

void ComputePipeline::DispatchAndWait(uint32_t groupCountX, uint32_t groupCountY,
                                       uint32_t groupCountZ) {
    Dispatch(groupCountX, groupCountY, groupCountZ);
    s_vkQueueWaitIdle(GetVulkanContext().GetComputeQueue());
}

bool ComputePipeline::CreateShaderModule(const std::vector<uint32_t>& spirvCode) {
    if (spirvCode.empty()) {
        WULFNET_ERROR("Compute", "Empty SPIR-V code");
        return false;
    }

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
    createInfo.pCode = spirvCode.data();

    VkDevice device = GetVulkanContext().GetDevice();
    if (s_vkCreateShaderModule(device, &createInfo, nullptr, &m_shaderModule) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool ComputePipeline::CreateDescriptorSetLayout(const std::vector<ShaderBinding>& bindings) {
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

    for (const auto& binding : bindings) {
        VkDescriptorSetLayoutBinding layoutBinding = {};
        layoutBinding.binding = binding.binding;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        layoutBinding.pImmutableSamplers = nullptr;

        switch (binding.type) {
            case ShaderBindingType::StorageBuffer:
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case ShaderBindingType::UniformBuffer:
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case ShaderBindingType::StorageImage:
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            case ShaderBindingType::SampledImage:
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            case ShaderBindingType::Sampler:
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                break;
        }

        layoutBindings.push_back(layoutBinding);
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();

    VkDevice device = GetVulkanContext().GetDevice();
    if (s_vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                      &m_descriptorSetLayout) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool ComputePipeline::CreatePipelineLayout(const PushConstantRange& pushConstants) {
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = pushConstants.offset;
    pcRange.size = pushConstants.size;

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_descriptorSetLayout;

    if (pushConstants.size > 0) {
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pcRange;
    }

    VkDevice device = GetVulkanContext().GetDevice();
    if (s_vkCreatePipelineLayout(device, &layoutInfo, nullptr,
                                 &m_pipelineLayout) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool ComputePipeline::CreatePipeline(const std::string& entryPoint) {
    VkPipelineShaderStageCreateInfo stageInfo = {};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = m_shaderModule;
    stageInfo.pName = entryPoint.c_str();

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = m_pipelineLayout;

    VkDevice device = GetVulkanContext().GetDevice();
    VkPipelineCache cache = GetVulkanContext().GetPipelineCache();

    if (s_vkCreateComputePipelines(device, cache, 1, &pipelineInfo, nullptr,
                                   &m_pipeline) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool ComputePipeline::AllocateDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = GetVulkanContext().GetDescriptorPool();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;

    VkDevice device = GetVulkanContext().GetDevice();
    if (s_vkAllocateDescriptorSets(device, &allocInfo, &m_descriptorSet) != VK_SUCCESS) {
        return false;
    }

    return true;
}

// =============================================================================
// Shader Utilities
// =============================================================================

std::vector<uint32_t> ShaderUtils::LoadSPIRV(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        WULFNET_ERROR("Compute", "Failed to open SPIR-V file: " + path);
        return {};
    }

    size_t fileSize = static_cast<size_t>(file.tellg());

    if (fileSize % sizeof(uint32_t) != 0) {
        WULFNET_ERROR("Compute", "Invalid SPIR-V file size: " + path);
        return {};
    }

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    // Validate SPIR-V magic number
    if (buffer.empty() || buffer[0] != 0x07230203) {
        WULFNET_ERROR("Compute", "Invalid SPIR-V magic number in: " + path);
        return {};
    }

    WULFNET_DEBUG("Compute", "Loaded SPIR-V: " + path + " (" + std::to_string(fileSize) + " bytes)");
    return buffer;
}

std::vector<uint32_t> ShaderUtils::CompileGLSL(const std::string& source,
                                                const std::string& entryPoint,
                                                const std::string& stage) {
    // TODO: Implement runtime GLSL compilation using shaderc or glslang
    // For now, require pre-compiled SPIR-V
    (void)source;
    (void)entryPoint;
    (void)stage;

    WULFNET_WARNING("Compute", "Runtime GLSL compilation not yet implemented");
    return {};
}

std::vector<uint32_t> ShaderUtils::CompileGLSLFile(const std::string& path,
                                                    const std::string& entryPoint) {
    (void)path;
    (void)entryPoint;

    WULFNET_WARNING("Compute", "Runtime GLSL compilation not yet implemented");
    return {};
}

} // namespace WulfNet
