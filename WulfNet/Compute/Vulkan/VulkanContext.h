// =============================================================================
// WulfNet Engine - Vulkan Compute Context
// =============================================================================
// Manages Vulkan instance, device, and compute queue for GPU compute operations.
// This is a headless compute-only context (no presentation/swapchain).
// =============================================================================

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>

// Forward declare Vulkan types to avoid including vulkan.h in header
// Users who need raw Vulkan access should include VulkanContext.cpp or vulkan.h
struct VkInstance_T;
struct VkPhysicalDevice_T;
struct VkDevice_T;
struct VkQueue_T;
struct VkCommandPool_T;
struct VkDescriptorPool_T;
struct VkPipelineCache_T;
struct VkCommandBuffer_T;
struct VkFence_T;
struct VkDescriptorSet_T;

typedef VkInstance_T* VkInstance;
typedef VkPhysicalDevice_T* VkPhysicalDevice;
typedef VkDevice_T* VkDevice;
typedef VkQueue_T* VkQueue;
typedef VkCommandPool_T* VkCommandPool;
typedef VkDescriptorPool_T* VkDescriptorPool;
typedef VkPipelineCache_T* VkPipelineCache;
typedef VkCommandBuffer_T* VkCommandBuffer;
typedef VkFence_T* VkFence;
typedef VkDescriptorSet_T* VkDescriptorSet;

#define VK_NULL_HANDLE nullptr

namespace WulfNet {

// =============================================================================
// GPU Device Information
// =============================================================================

struct GPUDeviceInfo {
    std::string name;
    uint32_t vendorId = 0;
    uint32_t deviceId = 0;
    uint64_t totalMemory = 0;           // Total device-local memory in bytes
    uint32_t maxComputeWorkGroupSize[3] = {0, 0, 0};
    uint32_t maxComputeWorkGroupCount[3] = {0, 0, 0};
    uint32_t maxComputeSharedMemory = 0;
    uint32_t computeQueueFamilyIndex = 0;
    uint32_t transferQueueFamilyIndex = 0;
    bool supportsAsyncCompute = false;
    bool supportsTimestampQueries = false;
    float timestampPeriod = 0.0f;       // Nanoseconds per timestamp tick

    // Vendor identification
    bool IsNvidia() const { return vendorId == 0x10DE; }
    bool IsAMD() const { return vendorId == 0x1002 || vendorId == 0x1022; }
    bool IsIntel() const { return vendorId == 0x8086; }
};

// =============================================================================
// Vulkan Context Settings
// =============================================================================

struct VulkanContextSettings {
    std::string applicationName = "WulfNet Engine";
    uint32_t applicationVersion = 1;
    bool enableValidation = true;       // Enable Vulkan validation layers (debug)
    bool enableGPUAssisted = false;     // GPU-assisted validation (slower)
    bool preferDiscreteGPU = true;      // Prefer discrete over integrated GPU
    uint32_t preferredDeviceIndex = 0;  // Override device selection
    uint32_t maxDescriptorSets = 1024;
    uint32_t maxUniformBuffers = 256;
    uint32_t maxStorageBuffers = 256;
    uint32_t maxStorageImages = 64;
    uint32_t maxSamplers = 64;
};

// =============================================================================
// Vulkan Context
// =============================================================================

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    // Non-copyable, movable
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&& other) noexcept;
    VulkanContext& operator=(VulkanContext&& other) noexcept;

    // ==========================================================================
    // Initialization
    // ==========================================================================

    /// Initialize Vulkan instance and select a compute-capable device
    bool Initialize(const VulkanContextSettings& settings = {});

    /// Initialize from external Vulkan handles (e.g., from Jolt renderer)
    /// This allows WulfNet compute to share a Vulkan context with another system.
    /// Note: Command pools, descriptor pools, and pipeline cache will be created
    /// by WulfNet, but the instance/device/queue are borrowed (not owned).
    /// @param instance External Vulkan instance
    /// @param physicalDevice External physical device
    /// @param device External logical device
    /// @param computeQueue External compute-capable queue
    /// @param computeQueueFamilyIndex Queue family index for compute
    /// @param settings Settings for descriptor pool sizing, etc.
    /// @return true if initialization succeeded
    bool InitializeFromExternal(VkInstance instance,
                                 VkPhysicalDevice physicalDevice,
                                 VkDevice device,
                                 VkQueue computeQueue,
                                 uint32_t computeQueueFamilyIndex,
                                 const VulkanContextSettings& settings = {});

    /// Shutdown and release all Vulkan resources
    void Shutdown();

    /// Check if context is valid and ready for use
    bool IsValid() const { return m_initialized; }

    // ==========================================================================
    // Device Information
    // ==========================================================================

    /// Get information about the selected GPU
    const GPUDeviceInfo& GetDeviceInfo() const { return m_deviceInfo; }

    /// Get list of all available GPUs
    static std::vector<GPUDeviceInfo> EnumerateDevices();

    // ==========================================================================
    // Vulkan Handles (for advanced users)
    // ==========================================================================

    VkInstance GetInstance() const { return m_instance; }
    VkPhysicalDevice GetPhysicalDevice() const { return m_physicalDevice; }
    VkDevice GetDevice() const { return m_device; }
    VkQueue GetComputeQueue() const { return m_computeQueue; }
    VkQueue GetTransferQueue() const { return m_transferQueue; }
    VkCommandPool GetComputeCommandPool() const { return m_computeCommandPool; }
    VkCommandPool GetTransferCommandPool() const { return m_transferCommandPool; }
    VkDescriptorPool GetDescriptorPool() const { return m_descriptorPool; }
    VkPipelineCache GetPipelineCache() const { return m_pipelineCache; }

    uint32_t GetComputeQueueFamily() const { return m_deviceInfo.computeQueueFamilyIndex; }
    uint32_t GetTransferQueueFamily() const { return m_deviceInfo.transferQueueFamilyIndex; }

    // ==========================================================================
    // Synchronization Helpers
    // ==========================================================================

    /// Wait for all GPU work to complete
    void WaitIdle();

    /// Submit a one-time command buffer and wait for completion
    bool SubmitAndWait(std::function<void(void* cmdBuffer)> recordFunc);

private:
    bool CreateInstance(const VulkanContextSettings& settings);
    bool SelectPhysicalDevice(const VulkanContextSettings& settings);
    bool CreateDevice(const VulkanContextSettings& settings);
    bool CreateCommandPools();
    bool CreateDescriptorPool(const VulkanContextSettings& settings);
    bool CreatePipelineCache();
    void DestroyDebugMessenger();

    bool m_initialized = false;
    bool m_ownsDevice = true;  // false when initialized from external handles

    VkInstance m_instance = nullptr;
    VkPhysicalDevice m_physicalDevice = nullptr;
    VkDevice m_device = nullptr;
    VkQueue m_computeQueue = nullptr;
    VkQueue m_transferQueue = nullptr;
    VkCommandPool m_computeCommandPool = nullptr;
    VkCommandPool m_transferCommandPool = nullptr;
    VkDescriptorPool m_descriptorPool = nullptr;
    VkPipelineCache m_pipelineCache = nullptr;

    // Debug messenger (validation layers)
    void* m_debugMessenger = nullptr;

    GPUDeviceInfo m_deviceInfo;
};

// =============================================================================
// Global Vulkan Context Access
// =============================================================================

/// Get the global Vulkan context (lazy initialized)
VulkanContext& GetVulkanContext();

/// Check if global Vulkan context is initialized
bool IsVulkanContextInitialized();

/// Initialize the global Vulkan context with custom settings
bool InitializeVulkanContext(const VulkanContextSettings& settings = {});

/// Initialize the global Vulkan context from external Vulkan handles
/// This is used when integrating with an external renderer (e.g., Jolt)
bool InitializeVulkanContextFromExternal(VkInstance instance,
                                          VkPhysicalDevice physicalDevice,
                                          VkDevice device,
                                          VkQueue computeQueue,
                                          uint32_t computeQueueFamilyIndex,
                                          const VulkanContextSettings& settings = {});

/// Shutdown the global Vulkan context
void ShutdownVulkanContext();

/// Get the loaded vkGetInstanceProcAddr function pointer
/// Returns nullptr if not loaded yet
typedef void (*PFN_vkVoidFunction)(void);
typedef PFN_vkVoidFunction (*PFN_vkGetInstanceProcAddrType)(VkInstance instance, const char* pName);
PFN_vkGetInstanceProcAddrType GetVulkanInstanceProcAddr();

} // namespace WulfNet
