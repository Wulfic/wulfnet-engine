// =============================================================================
// WulfNet Engine - Vulkan Compute Context Implementation
// =============================================================================

#include "WulfNet/Compute/Vulkan/VulkanContext.h"
#include "WulfNet/Core/Logging/Logger.h"
#include "WulfNet/Core/Profiling/Profiler.h"

// Only include Vulkan in the implementation
#ifdef WULFNET_PLATFORM_WINDOWS
    #define VK_USE_PLATFORM_WIN32_KHR
#elif defined(WULFNET_PLATFORM_LINUX)
    #define VK_USE_PLATFORM_XCB_KHR
#elif defined(WULFNET_PLATFORM_MACOS)
    #define VK_USE_PLATFORM_MACOS_MVK
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>

namespace WulfNet {

// =============================================================================
// Vulkan Function Loader
// =============================================================================

// We use dynamic loading to avoid requiring the Vulkan SDK at compile time
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

    bool loaded = false;
};

static VulkanFunctions g_vkFuncs;
static void* g_vulkanLibrary = nullptr;

// Platform-specific library loading
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

static bool LoadVulkanFunctions() {
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

static void LoadInstanceFunctions(VkInstance instance) {
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

static void LoadDeviceFunctions(VkInstance instance, VkDevice /*device*/) {
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

    #undef LOAD_VK_FUNC
}

// =============================================================================
// Debug Callback
// =============================================================================

static VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
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

// =============================================================================
// VulkanContext Implementation
// =============================================================================

VulkanContext::VulkanContext() = default;

VulkanContext::~VulkanContext() {
    Shutdown();
}

VulkanContext::VulkanContext(VulkanContext&& other) noexcept
    : m_initialized(other.m_initialized)
    , m_instance(other.m_instance)
    , m_physicalDevice(other.m_physicalDevice)
    , m_device(other.m_device)
    , m_computeQueue(other.m_computeQueue)
    , m_transferQueue(other.m_transferQueue)
    , m_computeCommandPool(other.m_computeCommandPool)
    , m_transferCommandPool(other.m_transferCommandPool)
    , m_descriptorPool(other.m_descriptorPool)
    , m_pipelineCache(other.m_pipelineCache)
    , m_debugMessenger(other.m_debugMessenger)
    , m_deviceInfo(std::move(other.m_deviceInfo))
{
    other.m_initialized = false;
    other.m_instance = nullptr;
    other.m_physicalDevice = nullptr;
    other.m_device = nullptr;
    other.m_computeQueue = nullptr;
    other.m_transferQueue = nullptr;
    other.m_computeCommandPool = nullptr;
    other.m_transferCommandPool = nullptr;
    other.m_descriptorPool = nullptr;
    other.m_pipelineCache = nullptr;
    other.m_debugMessenger = nullptr;
}

VulkanContext& VulkanContext::operator=(VulkanContext&& other) noexcept {
    if (this != &other) {
        Shutdown();

        m_initialized = other.m_initialized;
        m_instance = other.m_instance;
        m_physicalDevice = other.m_physicalDevice;
        m_device = other.m_device;
        m_computeQueue = other.m_computeQueue;
        m_transferQueue = other.m_transferQueue;
        m_computeCommandPool = other.m_computeCommandPool;
        m_transferCommandPool = other.m_transferCommandPool;
        m_descriptorPool = other.m_descriptorPool;
        m_pipelineCache = other.m_pipelineCache;
        m_debugMessenger = other.m_debugMessenger;
        m_deviceInfo = std::move(other.m_deviceInfo);

        other.m_initialized = false;
        other.m_instance = nullptr;
        other.m_physicalDevice = nullptr;
        other.m_device = nullptr;
        other.m_computeQueue = nullptr;
        other.m_transferQueue = nullptr;
        other.m_computeCommandPool = nullptr;
        other.m_transferCommandPool = nullptr;
        other.m_descriptorPool = nullptr;
        other.m_pipelineCache = nullptr;
        other.m_debugMessenger = nullptr;
    }
    return *this;
}

bool VulkanContext::Initialize(const VulkanContextSettings& settings) {
    WULFNET_ZONE();

    if (m_initialized) {
        WULFNET_WARNING("Compute", "VulkanContext already initialized");
        return true;
    }

    WULFNET_INFO("Compute", "Initializing Vulkan compute context...");

    if (!LoadVulkanFunctions()) {
        return false;
    }

    if (!CreateInstance(settings)) {
        return false;
    }

    if (!SelectPhysicalDevice(settings)) {
        Shutdown();
        return false;
    }

    if (!CreateDevice(settings)) {
        Shutdown();
        return false;
    }

    if (!CreateCommandPools()) {
        Shutdown();
        return false;
    }

    if (!CreateDescriptorPool(settings)) {
        Shutdown();
        return false;
    }

    if (!CreatePipelineCache()) {
        Shutdown();
        return false;
    }

    m_initialized = true;

    WULFNET_INFO("Compute", "Vulkan compute context initialized successfully");
    WULFNET_INFO("Compute", std::string("  GPU: ") + m_deviceInfo.name);
    WULFNET_INFO("Compute", "  Memory: " + std::to_string(m_deviceInfo.totalMemory / (1024 * 1024)) + " MB");
    WULFNET_INFO("Compute", "  Max Workgroup Size: " +
        std::to_string(m_deviceInfo.maxComputeWorkGroupSize[0]) + "x" +
        std::to_string(m_deviceInfo.maxComputeWorkGroupSize[1]) + "x" +
        std::to_string(m_deviceInfo.maxComputeWorkGroupSize[2]));

    return true;
}

void VulkanContext::Shutdown() {
    if (!m_initialized && !m_instance) return;

    WULFNET_INFO("Compute", "Shutting down Vulkan compute context...");

    if (m_device) {
        g_vkFuncs.vkDeviceWaitIdle(m_device);
    }

    if (m_pipelineCache && g_vkFuncs.vkDestroyPipelineCache) {
        g_vkFuncs.vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
        m_pipelineCache = nullptr;
    }

    if (m_descriptorPool && g_vkFuncs.vkDestroyDescriptorPool) {
        g_vkFuncs.vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_descriptorPool = nullptr;
    }

    if (m_transferCommandPool && g_vkFuncs.vkDestroyCommandPool) {
        g_vkFuncs.vkDestroyCommandPool(m_device, m_transferCommandPool, nullptr);
        m_transferCommandPool = nullptr;
    }

    if (m_computeCommandPool && g_vkFuncs.vkDestroyCommandPool) {
        g_vkFuncs.vkDestroyCommandPool(m_device, m_computeCommandPool, nullptr);
        m_computeCommandPool = nullptr;
    }

    if (m_device && g_vkFuncs.vkDestroyDevice) {
        g_vkFuncs.vkDestroyDevice(m_device, nullptr);
        m_device = nullptr;
    }

    DestroyDebugMessenger();

    if (m_instance && g_vkFuncs.vkDestroyInstance) {
        g_vkFuncs.vkDestroyInstance(m_instance, nullptr);
        m_instance = nullptr;
    }

    m_physicalDevice = nullptr;
    m_computeQueue = nullptr;
    m_transferQueue = nullptr;
    m_initialized = false;

    WULFNET_INFO("Compute", "Vulkan compute context shutdown complete");
}

bool VulkanContext::CreateInstance(const VulkanContextSettings& settings) {
    // Check for validation layer support
    std::vector<const char*> layers;
    std::vector<const char*> extensions;

    if (settings.enableValidation) {
        uint32_t layerCount = 0;
        g_vkFuncs.vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        g_vkFuncs.vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        bool validationSupported = false;
        for (const auto& layer : availableLayers) {
            if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                validationSupported = true;
                break;
            }
        }

        if (validationSupported) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            WULFNET_DEBUG("Compute", "Vulkan validation layers enabled");
        } else {
            WULFNET_WARNING("Compute", "Vulkan validation layers requested but not available");
        }
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = settings.applicationName.c_str();
    appInfo.applicationVersion = settings.applicationVersion;
    appInfo.pEngineName = "WulfNet Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkResult result = g_vkFuncs.vkCreateInstance(&createInfo, nullptr, &m_instance);
    if (result != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to create Vulkan instance: " + std::to_string(static_cast<int>(result)));
        return false;
    }

    LoadInstanceFunctions(m_instance);

    // Create debug messenger
    if (settings.enableValidation && g_vkFuncs.vkCreateDebugUtilsMessengerEXT) {
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = VulkanDebugCallback;

        VkDebugUtilsMessengerEXT messenger;
        if (g_vkFuncs.vkCreateDebugUtilsMessengerEXT(m_instance, &debugCreateInfo, nullptr, &messenger) == VK_SUCCESS) {
            m_debugMessenger = messenger;
        }
    }

    WULFNET_DEBUG("Compute", "Vulkan instance created");
    return true;
}

bool VulkanContext::SelectPhysicalDevice(const VulkanContextSettings& settings) {
    uint32_t deviceCount = 0;
    g_vkFuncs.vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        WULFNET_ERROR("Compute", "No Vulkan-capable GPUs found");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    g_vkFuncs.vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    WULFNET_INFO("Compute", "Found " + std::to_string(deviceCount) + " Vulkan device(s)");

    // Score and select best device
    int bestScore = -1;
    VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
    GPUDeviceInfo bestInfo;

    for (size_t i = 0; i < devices.size(); i++) {
        VkPhysicalDeviceProperties props;
        g_vkFuncs.vkGetPhysicalDeviceProperties(devices[i], &props);

        VkPhysicalDeviceMemoryProperties memProps;
        g_vkFuncs.vkGetPhysicalDeviceMemoryProperties(devices[i], &memProps);

        // Get queue families
        uint32_t queueFamilyCount = 0;
        g_vkFuncs.vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        g_vkFuncs.vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queueFamilyCount, queueFamilies.data());

        // Find compute queue
        uint32_t computeQueueFamily = UINT32_MAX;
        uint32_t transferQueueFamily = UINT32_MAX;

        for (uint32_t j = 0; j < queueFamilyCount; j++) {
            if ((queueFamilies[j].queueFlags & VK_QUEUE_COMPUTE_BIT) && computeQueueFamily == UINT32_MAX) {
                computeQueueFamily = j;
            }
            // Prefer dedicated transfer queue
            if ((queueFamilies[j].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                !(queueFamilies[j].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                transferQueueFamily == UINT32_MAX) {
                transferQueueFamily = j;
            }
        }

        // Fallback: use compute queue for transfer
        if (transferQueueFamily == UINT32_MAX) {
            transferQueueFamily = computeQueueFamily;
        }

        if (computeQueueFamily == UINT32_MAX) {
            WULFNET_DEBUG("Compute", "  Device " + std::to_string(i) + ": " + std::string(props.deviceName) + " - No compute queue, skipping");
            continue;
        }

        // Calculate total device-local memory
        uint64_t totalMemory = 0;
        for (uint32_t j = 0; j < memProps.memoryHeapCount; j++) {
            if (memProps.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                totalMemory += memProps.memoryHeaps[j].size;
            }
        }

        // Build device info
        GPUDeviceInfo info;
        info.name = props.deviceName;
        info.vendorId = props.vendorID;
        info.deviceId = props.deviceID;
        info.totalMemory = totalMemory;
        info.maxComputeWorkGroupSize[0] = props.limits.maxComputeWorkGroupSize[0];
        info.maxComputeWorkGroupSize[1] = props.limits.maxComputeWorkGroupSize[1];
        info.maxComputeWorkGroupSize[2] = props.limits.maxComputeWorkGroupSize[2];
        info.maxComputeWorkGroupCount[0] = props.limits.maxComputeWorkGroupCount[0];
        info.maxComputeWorkGroupCount[1] = props.limits.maxComputeWorkGroupCount[1];
        info.maxComputeWorkGroupCount[2] = props.limits.maxComputeWorkGroupCount[2];
        info.maxComputeSharedMemory = props.limits.maxComputeSharedMemorySize;
        info.computeQueueFamilyIndex = computeQueueFamily;
        info.transferQueueFamilyIndex = transferQueueFamily;
        info.supportsAsyncCompute = (transferQueueFamily != computeQueueFamily);
        info.supportsTimestampQueries = (props.limits.timestampComputeAndGraphics != 0);
        info.timestampPeriod = props.limits.timestampPeriod;

        // Score the device
        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        }
        score += static_cast<int>(totalMemory / (1024 * 1024)); // MB of memory

        // Apply user preference
        if (settings.preferDiscreteGPU && props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score /= 2;
        }

        if (i == settings.preferredDeviceIndex) {
            score += 10000; // Strong preference for user-selected device
        }

        WULFNET_INFO("Compute", "  Device " + std::to_string(i) + ": " + std::string(props.deviceName) + " (Score: " + std::to_string(score) + ")");

        if (score > bestScore) {
            bestScore = score;
            bestDevice = devices[i];
            bestInfo = info;
        }
    }

    if (bestDevice == VK_NULL_HANDLE) {
        WULFNET_ERROR("Compute", "No suitable Vulkan device found");
        return false;
    }

    m_physicalDevice = bestDevice;
    m_deviceInfo = bestInfo;

    WULFNET_INFO("Compute", std::string("Selected GPU: ") + m_deviceInfo.name);
    return true;
}

bool VulkanContext::CreateDevice(const VulkanContextSettings& /*settings*/) {
    // Queue create infos
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo computeQueueInfo = {};
    computeQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    computeQueueInfo.queueFamilyIndex = m_deviceInfo.computeQueueFamilyIndex;
    computeQueueInfo.queueCount = 1;
    computeQueueInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(computeQueueInfo);

    if (m_deviceInfo.transferQueueFamilyIndex != m_deviceInfo.computeQueueFamilyIndex) {
        VkDeviceQueueCreateInfo transferQueueInfo = {};
        transferQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        transferQueueInfo.queueFamilyIndex = m_deviceInfo.transferQueueFamilyIndex;
        transferQueueInfo.queueCount = 1;
        transferQueueInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(transferQueueInfo);
    }

    // Device features
    VkPhysicalDeviceFeatures deviceFeatures = {};

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;

    VkResult result = g_vkFuncs.vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to create Vulkan device: " + std::to_string(static_cast<int>(result)));
        return false;
    }

    LoadDeviceFunctions(m_instance, m_device);

    // Get queues
    g_vkFuncs.vkGetDeviceQueue(m_device, m_deviceInfo.computeQueueFamilyIndex, 0, &m_computeQueue);

    if (m_deviceInfo.transferQueueFamilyIndex != m_deviceInfo.computeQueueFamilyIndex) {
        g_vkFuncs.vkGetDeviceQueue(m_device, m_deviceInfo.transferQueueFamilyIndex, 0, &m_transferQueue);
    } else {
        m_transferQueue = m_computeQueue;
    }

    WULFNET_DEBUG("Compute", "Vulkan device created");
    return true;
}

bool VulkanContext::CreateCommandPools() {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_deviceInfo.computeQueueFamilyIndex;

    VkResult result = g_vkFuncs.vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_computeCommandPool);
    if (result != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to create compute command pool: " + std::to_string(static_cast<int>(result)));
        return false;
    }

    if (m_deviceInfo.transferQueueFamilyIndex != m_deviceInfo.computeQueueFamilyIndex) {
        poolInfo.queueFamilyIndex = m_deviceInfo.transferQueueFamilyIndex;
        result = g_vkFuncs.vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_transferCommandPool);
        if (result != VK_SUCCESS) {
            WULFNET_ERROR("Compute", "Failed to create transfer command pool: " + std::to_string(static_cast<int>(result)));
            return false;
        }
    } else {
        m_transferCommandPool = m_computeCommandPool;
    }

    WULFNET_DEBUG("Compute", "Command pools created");
    return true;
}

bool VulkanContext::CreateDescriptorPool(const VulkanContextSettings& settings) {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, settings.maxUniformBuffers },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, settings.maxStorageBuffers },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, settings.maxStorageImages },
        { VK_DESCRIPTOR_TYPE_SAMPLER, settings.maxSamplers },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, settings.maxSamplers }
    };

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = settings.maxDescriptorSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    VkResult result = g_vkFuncs.vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool);
    if (result != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to create descriptor pool: " + std::to_string(static_cast<int>(result)));
        return false;
    }

    WULFNET_DEBUG("Compute", "Descriptor pool created");
    return true;
}

bool VulkanContext::CreatePipelineCache() {
    VkPipelineCacheCreateInfo cacheInfo = {};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

    VkResult result = g_vkFuncs.vkCreatePipelineCache(m_device, &cacheInfo, nullptr, &m_pipelineCache);
    if (result != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to create pipeline cache: " + std::to_string(static_cast<int>(result)));
        return false;
    }

    WULFNET_DEBUG("Compute", "Pipeline cache created");
    return true;
}

void VulkanContext::DestroyDebugMessenger() {
    if (m_debugMessenger && g_vkFuncs.vkDestroyDebugUtilsMessengerEXT) {
        g_vkFuncs.vkDestroyDebugUtilsMessengerEXT(
            m_instance,
            static_cast<VkDebugUtilsMessengerEXT>(m_debugMessenger),
            nullptr);
        m_debugMessenger = nullptr;
    }
}

void VulkanContext::WaitIdle() {
    if (m_device) {
        g_vkFuncs.vkDeviceWaitIdle(m_device);
    }
}

bool VulkanContext::SubmitAndWait(std::function<void(void* cmdBuffer)> recordFunc) {
    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_computeCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer;
    if (g_vkFuncs.vkAllocateCommandBuffers(m_device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
        return false;
    }

    // Begin recording
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    g_vkFuncs.vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    // Record user commands
    recordFunc(cmdBuffer);

    // End recording
    g_vkFuncs.vkEndCommandBuffer(cmdBuffer);

    // Create fence
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    VkFence fence;
    if (g_vkFuncs.vkCreateFence(m_device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        g_vkFuncs.vkFreeCommandBuffers(m_device, m_computeCommandPool, 1, &cmdBuffer);
        return false;
    }

    // Submit
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    g_vkFuncs.vkQueueSubmit(m_computeQueue, 1, &submitInfo, fence);

    // Wait for completion
    g_vkFuncs.vkWaitForFences(m_device, 1, &fence, VK_TRUE, UINT64_MAX);

    // Cleanup
    g_vkFuncs.vkDestroyFence(m_device, fence, nullptr);
    g_vkFuncs.vkFreeCommandBuffers(m_device, m_computeCommandPool, 1, &cmdBuffer);

    return true;
}

std::vector<GPUDeviceInfo> VulkanContext::EnumerateDevices() {
    std::vector<GPUDeviceInfo> result;

    if (!LoadVulkanFunctions()) {
        return result;
    }

    // Create temporary instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "WulfNet Enumerate";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance tempInstance;
    if (g_vkFuncs.vkCreateInstance(&createInfo, nullptr, &tempInstance) != VK_SUCCESS) {
        return result;
    }

    LoadInstanceFunctions(tempInstance);

    uint32_t deviceCount = 0;
    g_vkFuncs.vkEnumeratePhysicalDevices(tempInstance, &deviceCount, nullptr);

    std::vector<VkPhysicalDevice> devices(deviceCount);
    g_vkFuncs.vkEnumeratePhysicalDevices(tempInstance, &deviceCount, devices.data());

    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        g_vkFuncs.vkGetPhysicalDeviceProperties(device, &props);

        VkPhysicalDeviceMemoryProperties memProps;
        g_vkFuncs.vkGetPhysicalDeviceMemoryProperties(device, &memProps);

        uint64_t totalMemory = 0;
        for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
            if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                totalMemory += memProps.memoryHeaps[i].size;
            }
        }

        GPUDeviceInfo info;
        info.name = props.deviceName;
        info.vendorId = props.vendorID;
        info.deviceId = props.deviceID;
        info.totalMemory = totalMemory;
        info.maxComputeWorkGroupSize[0] = props.limits.maxComputeWorkGroupSize[0];
        info.maxComputeWorkGroupSize[1] = props.limits.maxComputeWorkGroupSize[1];
        info.maxComputeWorkGroupSize[2] = props.limits.maxComputeWorkGroupSize[2];
        info.maxComputeSharedMemory = props.limits.maxComputeSharedMemorySize;

        result.push_back(info);
    }

    g_vkFuncs.vkDestroyInstance(tempInstance, nullptr);

    return result;
}

// =============================================================================
// Global Context
// =============================================================================

static std::unique_ptr<VulkanContext> g_vulkanContext;

VulkanContext& GetVulkanContext() {
    if (!g_vulkanContext) {
        g_vulkanContext = std::make_unique<VulkanContext>();
    }
    return *g_vulkanContext;
}

bool IsVulkanContextInitialized() {
    return g_vulkanContext && g_vulkanContext->IsValid();
}

bool InitializeVulkanContext(const VulkanContextSettings& settings) {
    return GetVulkanContext().Initialize(settings);
}

void ShutdownVulkanContext() {
    if (g_vulkanContext) {
        g_vulkanContext->Shutdown();
        g_vulkanContext.reset();
    }
}

} // namespace WulfNet
