// =============================================================================
// WulfNet Engine - GPU Buffer Implementation
// =============================================================================

#include "WulfNet/Compute/Memory/ComputeBuffer.h"
#include "WulfNet/Compute/Vulkan/VulkanContext.h"
#include "WulfNet/Physics/Fluids/COFLIPSystem.h"  // For template instantiations
#include "WulfNet/Core/Logging/Logger.h"
#include "WulfNet/Core/Profiling/Profiler.h"

#ifdef WULFNET_PLATFORM_WINDOWS
    #define VK_USE_PLATFORM_WIN32_KHR
    // Prevent Windows min/max macros from conflicting with std::min/max
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <cstring>
#include <algorithm>

namespace WulfNet {

// =============================================================================
// External Vulkan Function Declarations
// =============================================================================

// These are loaded in VulkanContext.cpp
extern PFN_vkCreateBuffer vkCreateBuffer;
extern PFN_vkDestroyBuffer vkDestroyBuffer;
extern PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
extern PFN_vkAllocateMemory vkAllocateMemory;
extern PFN_vkFreeMemory vkFreeMemory;
extern PFN_vkBindBufferMemory vkBindBufferMemory;
extern PFN_vkMapMemory vkMapMemory;
extern PFN_vkUnmapMemory vkUnmapMemory;
extern PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;
extern PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges;
extern PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
extern PFN_vkCmdFillBuffer vkCmdFillBuffer;
extern PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties_Local;

// Storage for the function pointers (loaded dynamically)
PFN_vkCreateBuffer vkCreateBuffer = nullptr;
PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
PFN_vkAllocateMemory vkAllocateMemory = nullptr;
PFN_vkFreeMemory vkFreeMemory = nullptr;
PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
PFN_vkMapMemory vkMapMemory = nullptr;
PFN_vkUnmapMemory vkUnmapMemory = nullptr;
PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges = nullptr;
PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges = nullptr;
PFN_vkCmdCopyBuffer vkCmdCopyBuffer = nullptr;
PFN_vkCmdFillBuffer vkCmdFillBuffer = nullptr;
PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties_Local = nullptr;

static bool s_bufferFunctionsLoaded = false;

static bool LoadBufferFunctions() {
    if (s_bufferFunctionsLoaded) return true;
    if (!IsVulkanContextInitialized()) {
        WULFNET_ERROR("Compute", "Cannot load buffer functions - VulkanContext not initialized");
        return false;
    }

    VkInstance instance = GetVulkanContext().GetInstance();

    // Get vkGetInstanceProcAddr from VulkanContext
    auto getProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetVulkanInstanceProcAddr());

    // Fall back to externally loaded one if VulkanContext doesn't have it
    if (!getProc) {
        extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_External;
        getProc = vkGetInstanceProcAddr_External;
    }

    if (!getProc) {
        WULFNET_ERROR("Compute", "Cannot get vkGetInstanceProcAddr for buffer functions");
        return false;
    }

    #define LOAD_VK_FUNC(name) \
        name = reinterpret_cast<PFN_##name>(getProc(instance, #name)); \
        if (!name) { \
            WULFNET_ERROR("Compute", "Failed to load " #name); \
            return false; \
        }

    LOAD_VK_FUNC(vkCreateBuffer);
    LOAD_VK_FUNC(vkDestroyBuffer);
    LOAD_VK_FUNC(vkGetBufferMemoryRequirements);
    LOAD_VK_FUNC(vkAllocateMemory);
    LOAD_VK_FUNC(vkFreeMemory);
    LOAD_VK_FUNC(vkBindBufferMemory);
    LOAD_VK_FUNC(vkMapMemory);
    LOAD_VK_FUNC(vkUnmapMemory);
    LOAD_VK_FUNC(vkFlushMappedMemoryRanges);
    LOAD_VK_FUNC(vkInvalidateMappedMemoryRanges);
    LOAD_VK_FUNC(vkCmdCopyBuffer);
    LOAD_VK_FUNC(vkCmdFillBuffer);
    vkGetPhysicalDeviceMemoryProperties_Local = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        getProc(instance, "vkGetPhysicalDeviceMemoryProperties"));
    if (!vkGetPhysicalDeviceMemoryProperties_Local) {
        WULFNET_ERROR("Compute", "Failed to load vkGetPhysicalDeviceMemoryProperties");
        return false;
    }

    #undef LOAD_VK_FUNC

    WULFNET_INFO("Compute", "Buffer Vulkan functions loaded successfully");
    s_bufferFunctionsLoaded = true;
    return true;
}

// =============================================================================
// ComputeBuffer Implementation
// =============================================================================

template<typename T>
ComputeBuffer<T>::ComputeBuffer(size_t count, GPUBufferUsage usage, GPUMemoryLocation location) {
    Allocate(count, usage, location);
}

template<typename T>
ComputeBuffer<T>::ComputeBuffer(const T* data, size_t count, GPUBufferUsage usage, GPUMemoryLocation location) {
    if (Allocate(count, usage, location)) {
        Upload(data, count);
    }
}

template<typename T>
ComputeBuffer<T>::ComputeBuffer(const std::vector<T>& data, GPUBufferUsage usage, GPUMemoryLocation location) {
    if (Allocate(data.size(), usage, location)) {
        Upload(data);
    }
}

template<typename T>
ComputeBuffer<T>::~ComputeBuffer() {
    Release();
}

template<typename T>
ComputeBuffer<T>::ComputeBuffer(ComputeBuffer&& other) noexcept
    : m_buffer(other.m_buffer)
    , m_memory(other.m_memory)
    , m_stagingBuffer(other.m_stagingBuffer)
    , m_stagingMemory(other.m_stagingMemory)
    , m_count(other.m_count)
    , m_usage(other.m_usage)
    , m_location(other.m_location)
    , m_mappedPtr(other.m_mappedPtr)
{
    other.m_buffer = nullptr;
    other.m_memory = nullptr;
    other.m_stagingBuffer = nullptr;
    other.m_stagingMemory = nullptr;
    other.m_count = 0;
    other.m_mappedPtr = nullptr;
}

template<typename T>
ComputeBuffer<T>& ComputeBuffer<T>::operator=(ComputeBuffer&& other) noexcept {
    if (this != &other) {
        Release();

        m_buffer = other.m_buffer;
        m_memory = other.m_memory;
        m_stagingBuffer = other.m_stagingBuffer;
        m_stagingMemory = other.m_stagingMemory;
        m_count = other.m_count;
        m_usage = other.m_usage;
        m_location = other.m_location;
        m_mappedPtr = other.m_mappedPtr;

        other.m_buffer = nullptr;
        other.m_memory = nullptr;
        other.m_stagingBuffer = nullptr;
        other.m_stagingMemory = nullptr;
        other.m_count = 0;
        other.m_mappedPtr = nullptr;
    }
    return *this;
}

template<typename T>
bool ComputeBuffer<T>::Allocate(size_t count, GPUBufferUsage usage, GPUMemoryLocation location) {
    WULFNET_ZONE();

    // Ensure Vulkan buffer functions are loaded
    if (!LoadBufferFunctions()) {
        WULFNET_ERROR("Compute", "Failed to load Vulkan buffer functions");
        return false;
    }

    if (count == 0) {
        WULFNET_WARNING("Compute", "Attempted to allocate zero-size buffer");
        return false;
    }

    Release();

    m_count = count;
    m_usage = usage;
    m_location = location;

    size_t sizeBytes = count * sizeof(T);

    if (!CreateBuffer(sizeBytes, usage, location)) {
        return false;
    }

    // Create staging buffer for device-local memory
    if (location == GPUMemoryLocation::DeviceLocal) {
        if (!CreateStagingBuffer(sizeBytes)) {
            DestroyBuffer();
            return false;
        }
    }

    WULFNET_DEBUG("Compute", "Allocated buffer: " + std::to_string(count) + " elements, " + std::to_string(sizeBytes) + " bytes");
    return true;
}

template<typename T>
void ComputeBuffer<T>::Release() {
    if (m_mappedPtr) {
        Unmap();
    }
    DestroyStagingBuffer();
    DestroyBuffer();
    m_count = 0;
}

template<typename T>
bool ComputeBuffer<T>::Resize(size_t newCount, bool preserveData) {
    WULFNET_ZONE();

    if (newCount == m_count) return true;
    if (newCount == 0) {
        Release();
        return true;
    }

    if (!IsValid()) {
        return Allocate(newCount, m_usage, m_location);
    }

    // Store old data if preserving
    std::vector<T> oldData;
    if (preserveData && m_count > 0) {
        oldData.resize(m_count);
        if (!Download(oldData)) {
            preserveData = false;  // Failed to download, can't preserve
        }
    }

    // Reallocate
    GPUBufferUsage oldUsage = m_usage;
    GPUMemoryLocation oldLocation = m_location;

    Release();

    if (!Allocate(newCount, oldUsage, oldLocation)) {
        return false;
    }

    // Restore old data
    if (preserveData && !oldData.empty()) {
        size_t copyCount = std::min(oldData.size(), static_cast<size_t>(newCount));
        Upload(oldData.data(), copyCount);
    }

    return true;
}

template<typename T>
bool ComputeBuffer<T>::Upload(const T* data, size_t count, size_t offset) {
    WULFNET_ZONE();

    if (!IsValid()) {
        WULFNET_ERROR("Compute", "Cannot upload to invalid buffer");
        return false;
    }

    if (offset + count > m_count) {
        WULFNET_ERROR("Compute", "Upload exceeds buffer size");
        return false;
    }

    if (!data || count == 0) return true;

    size_t sizeBytes = count * sizeof(T);
    size_t offsetBytes = offset * sizeof(T);

    if (m_location == GPUMemoryLocation::DeviceLocal) {
        // Use staging buffer
        if (!m_stagingBuffer) {
            WULFNET_ERROR("Compute", "No staging buffer for device-local upload");
            return false;
        }

        // Map staging, copy data, unmap
        void* mapped = nullptr;
        if (vkMapMemory(GetVulkanContext().GetDevice(), m_stagingMemory,
                        0, sizeBytes, 0, &mapped) != VK_SUCCESS) {
            return false;
        }

        std::memcpy(mapped, data, sizeBytes);
        vkUnmapMemory(GetVulkanContext().GetDevice(), m_stagingMemory);

        // Copy staging -> device
        GetVulkanContext().SubmitAndWait([&](void* cmdBuffer) {
            VkBufferCopy copyRegion = {};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = offsetBytes;
            copyRegion.size = sizeBytes;
            vkCmdCopyBuffer(static_cast<VkCommandBuffer>(cmdBuffer),
                           m_stagingBuffer, m_buffer, 1, &copyRegion);
        });
    } else {
        // Direct map for host-visible memory
        void* mapped = nullptr;
        if (vkMapMemory(GetVulkanContext().GetDevice(), m_memory,
                        offsetBytes, sizeBytes, 0, &mapped) != VK_SUCCESS) {
            return false;
        }

        std::memcpy(mapped, data, sizeBytes);
        vkUnmapMemory(GetVulkanContext().GetDevice(), m_memory);
    }

    return true;
}

template<typename T>
bool ComputeBuffer<T>::Upload(const std::vector<T>& data, size_t offset) {
    return Upload(data.data(), data.size(), offset);
}

template<typename T>
bool ComputeBuffer<T>::Download(T* data, size_t count, size_t offset) const {
    WULFNET_ZONE();

    if (!IsValid()) {
        WULFNET_ERROR("Compute", "Cannot download from invalid buffer");
        return false;
    }

    if (offset + count > m_count) {
        WULFNET_ERROR("Compute", "Download exceeds buffer size");
        return false;
    }

    if (!data || count == 0) return true;

    size_t sizeBytes = count * sizeof(T);
    size_t offsetBytes = offset * sizeof(T);

    if (m_location == GPUMemoryLocation::DeviceLocal) {
        // Copy device -> staging
        GetVulkanContext().SubmitAndWait([&](void* cmdBuffer) {
            VkBufferCopy copyRegion = {};
            copyRegion.srcOffset = offsetBytes;
            copyRegion.dstOffset = 0;
            copyRegion.size = sizeBytes;
            vkCmdCopyBuffer(static_cast<VkCommandBuffer>(cmdBuffer),
                           m_buffer, m_stagingBuffer, 1, &copyRegion);
        });

        // Map staging, copy data, unmap
        void* mapped = nullptr;
        if (vkMapMemory(GetVulkanContext().GetDevice(), m_stagingMemory,
                        0, sizeBytes, 0, &mapped) != VK_SUCCESS) {
            return false;
        }

        std::memcpy(data, mapped, sizeBytes);
        vkUnmapMemory(GetVulkanContext().GetDevice(), m_stagingMemory);
    } else {
        // Direct map for host-visible memory
        void* mapped = nullptr;
        if (vkMapMemory(GetVulkanContext().GetDevice(), m_memory,
                        offsetBytes, sizeBytes, 0, &mapped) != VK_SUCCESS) {
            return false;
        }

        std::memcpy(data, mapped, sizeBytes);
        vkUnmapMemory(GetVulkanContext().GetDevice(), m_memory);
    }

    return true;
}

template<typename T>
bool ComputeBuffer<T>::Download(std::vector<T>& data) const {
    data.resize(m_count);
    return Download(data.data(), m_count, 0);
}

template<typename T>
bool ComputeBuffer<T>::Clear() {
    WULFNET_ZONE();

    if (!IsValid()) return false;

    GetVulkanContext().SubmitAndWait([&](void* cmdBuffer) {
        vkCmdFillBuffer(static_cast<VkCommandBuffer>(cmdBuffer),
                       m_buffer, 0, VK_WHOLE_SIZE, 0);
    });

    return true;
}

template<typename T>
bool ComputeBuffer<T>::Fill(const T& value) {
    WULFNET_ZONE();

    if (!IsValid()) return false;

    // For simple types that fit in uint32_t, use vkCmdFillBuffer
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        uint32_t fillValue;
        std::memcpy(&fillValue, &value, sizeof(uint32_t));

        GetVulkanContext().SubmitAndWait([&](void* cmdBuffer) {
            vkCmdFillBuffer(static_cast<VkCommandBuffer>(cmdBuffer),
                           m_buffer, 0, VK_WHOLE_SIZE, fillValue);
        });
        return true;
    }

    // For other types, upload a vector filled with the value
    std::vector<T> data(m_count, value);
    return Upload(data);
}

template<typename T>
T* ComputeBuffer<T>::Map() {
    if (!IsValid()) return nullptr;
    if (m_location == GPUMemoryLocation::DeviceLocal) {
        WULFNET_ERROR("Compute", "Cannot map device-local buffer directly");
        return nullptr;
    }
    if (m_mappedPtr) return m_mappedPtr;

    void* mapped = nullptr;
    if (vkMapMemory(GetVulkanContext().GetDevice(), m_memory,
                    0, GetSizeBytes(), 0, &mapped) != VK_SUCCESS) {
        return nullptr;
    }

    m_mappedPtr = static_cast<T*>(mapped);
    return m_mappedPtr;
}

template<typename T>
const T* ComputeBuffer<T>::MapRead() const {
    return const_cast<ComputeBuffer<T>*>(this)->Map();
}

template<typename T>
void ComputeBuffer<T>::Unmap() {
    if (m_mappedPtr && m_memory) {
        vkUnmapMemory(GetVulkanContext().GetDevice(), m_memory);
        m_mappedPtr = nullptr;
    }
}

template<typename T>
bool ComputeBuffer<T>::CreateBuffer(size_t sizeBytes, GPUBufferUsage usage, GPUMemoryLocation location) {
    if (!IsVulkanContextInitialized()) {
        WULFNET_ERROR("Compute", "VulkanContext not initialized");
        return false;
    }

    VkDevice device = GetVulkanContext().GetDevice();

    // Convert usage flags
    VkBufferUsageFlags vkUsage = 0;
    if (HasFlag(usage, GPUBufferUsage::Storage)) vkUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (HasFlag(usage, GPUBufferUsage::Uniform)) vkUsage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (HasFlag(usage, GPUBufferUsage::Vertex)) vkUsage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (HasFlag(usage, GPUBufferUsage::Index)) vkUsage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (HasFlag(usage, GPUBufferUsage::Indirect)) vkUsage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (HasFlag(usage, GPUBufferUsage::TransferSrc)) vkUsage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (HasFlag(usage, GPUBufferUsage::TransferDst)) vkUsage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // Always enable transfer for device-local buffers
    if (location == GPUMemoryLocation::DeviceLocal) {
        vkUsage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeBytes;
    bufferInfo.usage = vkUsage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &m_buffer) != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to create buffer");
        return false;
    }

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, m_buffer, &memRequirements);

    // Determine memory properties
    VkMemoryPropertyFlags memProperties = 0;
    switch (location) {
        case GPUMemoryLocation::DeviceLocal:
            memProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        case GPUMemoryLocation::HostVisible:
            memProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case GPUMemoryLocation::HostCached:
            memProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            break;
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, memProperties);

    if (allocInfo.memoryTypeIndex == UINT32_MAX) {
        WULFNET_ERROR("Compute", "Failed to find suitable memory type");
        vkDestroyBuffer(device, m_buffer, nullptr);
        m_buffer = nullptr;
        return false;
    }

    if (vkAllocateMemory(device, &allocInfo, nullptr, &m_memory) != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to allocate buffer memory");
        vkDestroyBuffer(device, m_buffer, nullptr);
        m_buffer = nullptr;
        return false;
    }

    if (vkBindBufferMemory(device, m_buffer, m_memory, 0) != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to bind buffer memory");
        vkFreeMemory(device, m_memory, nullptr);
        vkDestroyBuffer(device, m_buffer, nullptr);
        m_buffer = nullptr;
        m_memory = nullptr;
        return false;
    }

    return true;
}

template<typename T>
bool ComputeBuffer<T>::CreateStagingBuffer(size_t sizeBytes) {
    VkDevice device = GetVulkanContext().GetDevice();

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &m_stagingBuffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, m_stagingBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (allocInfo.memoryTypeIndex == UINT32_MAX) {
        vkDestroyBuffer(device, m_stagingBuffer, nullptr);
        m_stagingBuffer = nullptr;
        return false;
    }

    if (vkAllocateMemory(device, &allocInfo, nullptr, &m_stagingMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, m_stagingBuffer, nullptr);
        m_stagingBuffer = nullptr;
        return false;
    }

    if (vkBindBufferMemory(device, m_stagingBuffer, m_stagingMemory, 0) != VK_SUCCESS) {
        vkFreeMemory(device, m_stagingMemory, nullptr);
        vkDestroyBuffer(device, m_stagingBuffer, nullptr);
        m_stagingBuffer = nullptr;
        m_stagingMemory = nullptr;
        return false;
    }

    return true;
}

template<typename T>
void ComputeBuffer<T>::DestroyBuffer() {
    if (!IsVulkanContextInitialized()) return;
    VkDevice device = GetVulkanContext().GetDevice();

    if (m_memory) {
        vkFreeMemory(device, m_memory, nullptr);
        m_memory = nullptr;
    }
    if (m_buffer) {
        vkDestroyBuffer(device, m_buffer, nullptr);
        m_buffer = nullptr;
    }
}

template<typename T>
void ComputeBuffer<T>::DestroyStagingBuffer() {
    if (!IsVulkanContextInitialized()) return;
    VkDevice device = GetVulkanContext().GetDevice();

    if (m_stagingMemory) {
        vkFreeMemory(device, m_stagingMemory, nullptr);
        m_stagingMemory = nullptr;
    }
    if (m_stagingBuffer) {
        vkDestroyBuffer(device, m_stagingBuffer, nullptr);
        m_stagingBuffer = nullptr;
    }
}

template<typename T>
uint32_t ComputeBuffer<T>::FindMemoryType(uint32_t typeFilter, uint32_t properties) const {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties_Local(GetVulkanContext().GetPhysicalDevice(), &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return UINT32_MAX;
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template class ComputeBuffer<float>;
template class ComputeBuffer<int32_t>;
template class ComputeBuffer<uint32_t>;
template class ComputeBuffer<Vec4>;
template class ComputeBuffer<ParticleData>;

// CO-FLIP fluid simulation types
template class ComputeBuffer<COFLIPParticle>;
template class ComputeBuffer<COFLIPCell>;

} // namespace WulfNet
