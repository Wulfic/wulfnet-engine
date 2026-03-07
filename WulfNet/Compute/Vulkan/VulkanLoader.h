// =============================================================================
// WulfNet Engine - Vulkan Dynamic Function Loader
// =============================================================================
// Manages dynamic loading of Vulkan function pointers without requiring
// the Vulkan SDK at compile time. Used by VulkanContext and related modules.
// =============================================================================

#pragma once

// Ensure VK_NO_PROTOTYPES is set before including vulkan.h — all function
// pointers are resolved at runtime via vkGetInstanceProcAddr.
#ifndef VK_NO_PROTOTYPES
    #define VK_NO_PROTOTYPES
#endif

#include <vulkan/vulkan.h>

namespace WulfNet {

// =============================================================================
// Vulkan Function Pointer Table
// =============================================================================

/// Holds dynamically-loaded Vulkan function pointers used throughout the
/// compute pipeline.  Populated by LoadVulkanFunctions / LoadInstanceFunctions
/// / LoadDeviceFunctions.
struct VulkanFunctions {
    // Instance-level functions
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2 = nullptr;
    PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties = nullptr;
    PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkDestroyDevice vkDestroyDevice = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;

    // Debug
    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = nullptr;
    PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = nullptr;

    // Device-level functions
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
    PFN_vkQueueSubmit vkQueueSubmit = nullptr;
    PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
    PFN_vkCreatePipelineCache vkCreatePipelineCache = nullptr;
    PFN_vkDestroyPipelineCache vkDestroyPipelineCache = nullptr;
    PFN_vkCreateFence vkCreateFence = nullptr;
    PFN_vkDestroyFence vkDestroyFence = nullptr;
    PFN_vkWaitForFences vkWaitForFences = nullptr;
    PFN_vkResetFences vkResetFences = nullptr;
    PFN_vkGetFenceStatus vkGetFenceStatus = nullptr;
    PFN_vkCreateSemaphore vkCreateSemaphore = nullptr;
    PFN_vkDestroySemaphore vkDestroySemaphore = nullptr;
    PFN_vkResetCommandBuffer vkResetCommandBuffer = nullptr;

    bool loaded = false;
};

/// Global Vulkan function pointer table
extern VulkanFunctions g_vkFuncs;

/// Handle to the dynamically loaded Vulkan shared library
extern void* g_vulkanLibrary;

/// Load the Vulkan library and resolve base global function pointers.
/// Safe to call multiple times (no-op after first successful load).
bool LoadVulkanFunctions();

/// Resolve instance-level function pointers after VkInstance creation.
void LoadInstanceFunctions(VkInstance instance);

/// Resolve device-level function pointers after VkDevice creation.
void LoadDeviceFunctions(VkInstance instance, VkDevice device);

/// Vulkan debug messenger callback for validation layer output.
VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData);

} // namespace WulfNet
