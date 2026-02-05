// =============================================================================
// WulfNet Engine - Compute Pipeline
// =============================================================================
// Manages Vulkan compute pipelines for GPU shader execution.
// =============================================================================

#pragma once

#include "WulfNet/Compute/Vulkan/VulkanContext.h"
#include "WulfNet/Compute/Memory/ComputeBuffer.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

// Forward declare Vulkan types
struct VkShaderModule_T;
struct VkPipeline_T;
struct VkPipelineLayout_T;
struct VkDescriptorSetLayout_T;
struct VkDescriptorSet_T;
typedef VkShaderModule_T* VkShaderModule;
typedef VkPipeline_T* VkPipeline;
typedef VkPipelineLayout_T* VkPipelineLayout;
typedef VkDescriptorSetLayout_T* VkDescriptorSetLayout;
typedef VkDescriptorSet_T* VkDescriptorSet;

namespace WulfNet {

// =============================================================================
// Shader Binding Types
// =============================================================================

enum class ShaderBindingType {
    StorageBuffer,      // Read/write buffer (SSBO)
    UniformBuffer,      // Read-only constants (UBO)
    StorageImage,       // Read/write image
    SampledImage,       // Read-only sampled image
    Sampler             // Texture sampler
};

struct ShaderBinding {
    uint32_t binding;
    ShaderBindingType type;
    std::string name;       // Optional name for debugging
};

// =============================================================================
// Push Constants
// =============================================================================

struct PushConstantRange {
    uint32_t offset = 0;
    uint32_t size = 0;
};

// =============================================================================
// Compute Pipeline Description
// =============================================================================

struct ComputePipelineDesc {
    std::vector<uint32_t> spirvCode;            // SPIR-V bytecode
    std::vector<ShaderBinding> bindings;         // Descriptor bindings
    PushConstantRange pushConstants;             // Push constant range
    std::string entryPoint = "main";             // Shader entry point
    uint32_t localSizeX = 256;                   // Workgroup size X (for validation)
    uint32_t localSizeY = 1;                     // Workgroup size Y
    uint32_t localSizeZ = 1;                     // Workgroup size Z
    std::string name;                            // Pipeline name for debugging
};

// =============================================================================
// Compute Pipeline
// =============================================================================

class ComputePipeline {
public:
    ComputePipeline();
    ~ComputePipeline();

    // Non-copyable, movable
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline& operator=(ComputePipeline&& other) noexcept;

    // ==========================================================================
    // Creation
    // ==========================================================================

    /// Create pipeline from description
    bool Create(const ComputePipelineDesc& desc);

    /// Create pipeline from SPIR-V file
    bool CreateFromFile(const std::string& spirvPath,
                        const std::vector<ShaderBinding>& bindings,
                        const PushConstantRange& pushConstants = {});

    /// Destroy pipeline resources
    void Destroy();

    /// Check if pipeline is valid
    bool IsValid() const { return m_pipeline != nullptr; }

    // ==========================================================================
    // Binding
    // ==========================================================================

    /// Bind a buffer to a descriptor slot
    void BindBuffer(uint32_t binding, const GPUBufferBase& buffer);

    /// Bind a buffer with explicit range
    void BindBuffer(uint32_t binding, const GPUBufferBase& buffer,
                    size_t offset, size_t range);

    /// Update all pending bindings to descriptor set
    bool UpdateBindings();

    // ==========================================================================
    // Push Constants
    // ==========================================================================

    /// Set push constant data
    template<typename T>
    void SetPushConstants(const T& data) {
        static_assert(sizeof(T) <= 128, "Push constants must be <= 128 bytes");
        SetPushConstantsRaw(&data, sizeof(T), 0);
    }

    /// Set push constants with offset
    template<typename T>
    void SetPushConstants(const T& data, uint32_t offset) {
        SetPushConstantsRaw(&data, sizeof(T), offset);
    }

    void SetPushConstantsRaw(const void* data, uint32_t size, uint32_t offset);

    // ==========================================================================
    // Dispatch
    // ==========================================================================

    /// Dispatch compute shader with group counts
    void Dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);

    /// Dispatch and wait for completion
    void DispatchAndWait(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);

    /// Calculate optimal group count for element count
    uint32_t CalculateGroupCount(uint32_t elementCount) const {
        return (elementCount + m_localSizeX - 1) / m_localSizeX;
    }

    // ==========================================================================
    // Getters
    // ==========================================================================

    uint32_t GetLocalSizeX() const { return m_localSizeX; }
    uint32_t GetLocalSizeY() const { return m_localSizeY; }
    uint32_t GetLocalSizeZ() const { return m_localSizeZ; }
    const std::string& GetName() const { return m_name; }

    VkPipeline GetVkPipeline() const { return m_pipeline; }
    VkPipelineLayout GetVkPipelineLayout() const { return m_pipelineLayout; }
    VkDescriptorSet GetVkDescriptorSet() const { return m_descriptorSet; }

private:
    bool CreateShaderModule(const std::vector<uint32_t>& spirvCode);
    bool CreateDescriptorSetLayout(const std::vector<ShaderBinding>& bindings);
    bool CreatePipelineLayout(const PushConstantRange& pushConstants);
    bool CreatePipeline(const std::string& entryPoint);
    bool AllocateDescriptorSet();

    VkShaderModule m_shaderModule = nullptr;
    VkDescriptorSetLayout m_descriptorSetLayout = nullptr;
    VkPipelineLayout m_pipelineLayout = nullptr;
    VkPipeline m_pipeline = nullptr;
    VkDescriptorSet m_descriptorSet = nullptr;

    uint32_t m_localSizeX = 256;
    uint32_t m_localSizeY = 1;
    uint32_t m_localSizeZ = 1;
    std::string m_name;

    // Pending descriptor writes
    struct PendingBinding {
        uint32_t binding;
        VkBuffer buffer;
        size_t offset;
        size_t range;
    };
    std::vector<PendingBinding> m_pendingBindings;

    // Push constant data
    std::vector<uint8_t> m_pushConstantData;
    uint32_t m_pushConstantSize = 0;
};

// =============================================================================
// Shader Utilities
// =============================================================================

namespace ShaderUtils {
    /// Load SPIR-V bytecode from file
    std::vector<uint32_t> LoadSPIRV(const std::string& path);

    /// Compile GLSL/HLSL to SPIR-V (requires glslc/dxc in PATH)
    std::vector<uint32_t> CompileGLSL(const std::string& source,
                                       const std::string& entryPoint = "main",
                                       const std::string& stage = "comp");

    /// Compile from file
    std::vector<uint32_t> CompileGLSLFile(const std::string& path,
                                           const std::string& entryPoint = "main");
}

} // namespace WulfNet
