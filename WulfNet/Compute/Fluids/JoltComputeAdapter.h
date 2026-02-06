// =============================================================================
// WulfNet Engine - Jolt Compute System Adapter
// Bridges Jolt's ComputeSystemVK to WulfNet's fluid compute system
// =============================================================================

#pragma once

#ifdef JPH_USE_VK

#include <vulkan/vulkan.h>

// Forward declarations for Jolt types
namespace JPH {
    class ComputeSystemVK;
}

namespace WulfNet {

/// Adapter that extracts Vulkan resources from Jolt's ComputeSystemVK
/// for use with WulfNet's fluid compute shaders
class JoltComputeAdapter {
public:
    JoltComputeAdapter() = default;
    ~JoltComputeAdapter() = default;

    /// Initialize from Jolt's compute system
    /// @param computeSystem Pointer to Jolt's ComputeSystemVK (or derived class like ComputeSystemVKImpl)
    /// @return true if successful
    bool Initialize(JPH::ComputeSystemVK* computeSystem);

    /// Shutdown and release resources
    void Shutdown();

    /// Check if adapter is valid
    bool IsValid() const { return m_device != VK_NULL_HANDLE; }

    /// Access Vulkan objects
    VkInstance GetInstance() const { return m_instance; }
    VkPhysicalDevice GetPhysicalDevice() const { return m_physicalDevice; }
    VkDevice GetDevice() const { return m_device; }
    VkQueue GetComputeQueue() const { return m_computeQueue; }
    uint32_t GetComputeQueueFamilyIndex() const { return m_computeQueueFamilyIndex; }

    /// Get the Jolt compute system pointer for advanced operations
    JPH::ComputeSystemVK* GetJoltComputeSystem() const { return m_joltCompute; }

private:
    JPH::ComputeSystemVK* m_joltCompute = nullptr;
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    uint32_t m_computeQueueFamilyIndex = 0;
};

} // namespace WulfNet

#endif // JPH_USE_VK
