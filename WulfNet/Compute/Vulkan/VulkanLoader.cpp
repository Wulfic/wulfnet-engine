// =============================================================================
// WulfNet Engine - Vulkan Dynamic Function Loader Implementation
// =============================================================================
// Platform-specific library loading and Vulkan function pointer resolution.
// Extracted from VulkanContext.cpp to keep the loader logic self-contained.
// =============================================================================

#include "WulfNet/Compute/Vulkan/VulkanLoader.h"
#include "WulfNet/Core/Logging/Logger.h"

namespace WulfNet {

// =============================================================================
// Global State
// =============================================================================

VulkanFunctions g_vkFuncs;
void* g_vulkanLibrary = nullptr;

// =============================================================================
// Platform-Specific Library Loading
// =============================================================================

#ifdef WULFNET_PLATFORM_WINDOWS
    #include <Windows.h>
    static void* LoadVulkanLibrary() {
        return LoadLibraryA("vulkan-1.dll");
    }
    static void* GetVulkanProcAddr(void* lib, const char* name) {
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
    }
    static void UnloadVulkanLibrary(void* lib) {
        FreeLibrary(static_cast<HMODULE>(lib));
    }
#else
    #include <dlfcn.h>
    static void* LoadVulkanLibrary() {
        #ifdef WULFNET_PLATFORM_MACOS
            return dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
        #else
            return dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        #endif
    }
    static void* GetVulkanProcAddr(void* lib, const char* name) {
        return dlsym(lib, name);
    }
    static void UnloadVulkanLibrary(void* lib) {
        dlclose(lib);
    }
#endif

// =============================================================================
// Function Loading
// =============================================================================

bool LoadVulkanFunctions() {
    if (g_vkFuncs.loaded) return true;

    g_vulkanLibrary = LoadVulkanLibrary();
    if (!g_vulkanLibrary) {
        WULFNET_ERROR("Compute", "Failed to load Vulkan library");
        return false;
    }

    g_vkFuncs.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetVulkanProcAddr(g_vulkanLibrary, "vkGetInstanceProcAddr"));

    if (!g_vkFuncs.vkGetInstanceProcAddr) {
        WULFNET_ERROR("Compute", "Failed to load vkGetInstanceProcAddr");
        return false;
    }

    // Load global functions
    #define LOAD_VK_FUNC(name) \
        g_vkFuncs.name = reinterpret_cast<PFN_##name>( \
            g_vkFuncs.vkGetInstanceProcAddr(nullptr, #name))

    LOAD_VK_FUNC(vkCreateInstance);
    LOAD_VK_FUNC(vkEnumerateInstanceLayerProperties);
    LOAD_VK_FUNC(vkEnumerateInstanceExtensionProperties);

    #undef LOAD_VK_FUNC

    g_vkFuncs.loaded = true;
    WULFNET_INFO("Compute", "Vulkan library loaded successfully");
    return true;
}

void LoadInstanceFunctions(VkInstance instance) {
    #define LOAD_VK_FUNC(name) \
        g_vkFuncs.name = reinterpret_cast<PFN_##name>( \
            g_vkFuncs.vkGetInstanceProcAddr(instance, #name))

    LOAD_VK_FUNC(vkDestroyInstance);
    LOAD_VK_FUNC(vkEnumeratePhysicalDevices);
    LOAD_VK_FUNC(vkGetPhysicalDeviceProperties);
    LOAD_VK_FUNC(vkGetPhysicalDeviceProperties2);
    LOAD_VK_FUNC(vkGetPhysicalDeviceMemoryProperties);
    LOAD_VK_FUNC(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_VK_FUNC(vkGetPhysicalDeviceFeatures);
    LOAD_VK_FUNC(vkGetPhysicalDeviceFeatures2);
    LOAD_VK_FUNC(vkEnumerateDeviceExtensionProperties);
    LOAD_VK_FUNC(vkCreateDevice);
    LOAD_VK_FUNC(vkDestroyDevice);
    LOAD_VK_FUNC(vkGetDeviceQueue);
    LOAD_VK_FUNC(vkCreateDebugUtilsMessengerEXT);
    LOAD_VK_FUNC(vkDestroyDebugUtilsMessengerEXT);

    #undef LOAD_VK_FUNC
}

void LoadDeviceFunctions(VkInstance instance, VkDevice /*device*/) {
    #define LOAD_VK_FUNC(name) \
        g_vkFuncs.name = reinterpret_cast<PFN_##name>( \
            g_vkFuncs.vkGetInstanceProcAddr(instance, #name))

    LOAD_VK_FUNC(vkDeviceWaitIdle);
    LOAD_VK_FUNC(vkCreateCommandPool);
    LOAD_VK_FUNC(vkDestroyCommandPool);
    LOAD_VK_FUNC(vkAllocateCommandBuffers);
    LOAD_VK_FUNC(vkFreeCommandBuffers);
    LOAD_VK_FUNC(vkBeginCommandBuffer);
    LOAD_VK_FUNC(vkEndCommandBuffer);
    LOAD_VK_FUNC(vkQueueSubmit);
    LOAD_VK_FUNC(vkQueueWaitIdle);
    LOAD_VK_FUNC(vkCreateDescriptorPool);
    LOAD_VK_FUNC(vkDestroyDescriptorPool);
    LOAD_VK_FUNC(vkCreatePipelineCache);
    LOAD_VK_FUNC(vkDestroyPipelineCache);
    LOAD_VK_FUNC(vkCreateFence);
    LOAD_VK_FUNC(vkDestroyFence);
    LOAD_VK_FUNC(vkWaitForFences);
    LOAD_VK_FUNC(vkResetFences);
    LOAD_VK_FUNC(vkGetFenceStatus);
    LOAD_VK_FUNC(vkCreateSemaphore);
    LOAD_VK_FUNC(vkDestroySemaphore);
    LOAD_VK_FUNC(vkResetCommandBuffer);

    #undef LOAD_VK_FUNC
}

// =============================================================================
// Debug Callback
// =============================================================================

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* /*pUserData*/)
{
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        WULFNET_ERROR("Vulkan", pCallbackData->pMessage);
    } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        WULFNET_WARNING("Vulkan", pCallbackData->pMessage);
    } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        WULFNET_DEBUG("Vulkan", pCallbackData->pMessage);
    }
    return VK_FALSE;
}

} // namespace WulfNet
