// =============================================================================
// WulfNet Engine - GPU Buffer Management
// =============================================================================
// Template-based GPU buffer abstraction for compute shaders.
// Supports automatic staging, memory type selection, and data transfer.
// =============================================================================

#pragma once

#include "WulfNet/Compute/Vulkan/VulkanContext.h"
#include <cstdint>
#include <vector>
#include <memory>
#include <type_traits>

// Forward declare Vulkan types
struct VkBuffer_T;
struct VkDeviceMemory_T;
typedef VkBuffer_T* VkBuffer;
typedef VkDeviceMemory_T* VkDeviceMemory;

namespace WulfNet {

// =============================================================================
// Buffer Usage Flags
// =============================================================================

enum class GPUBufferUsage : uint32_t {
    None = 0,
    Storage = 1 << 0,           // Shader storage buffer (read/write in compute)
    Uniform = 1 << 1,           // Uniform buffer (read-only constants)
    Vertex = 1 << 2,            // Vertex buffer (for visualization)
    Index = 1 << 3,             // Index buffer (for visualization)
    Indirect = 1 << 4,          // Indirect dispatch/draw commands
    TransferSrc = 1 << 5,       // Can be source of transfer
    TransferDst = 1 << 6,       // Can be destination of transfer

    // Common combinations
    ComputeStorage = Storage | TransferSrc | TransferDst,
    HostVisible = TransferSrc | TransferDst
};

inline GPUBufferUsage operator|(GPUBufferUsage a, GPUBufferUsage b) {
    return static_cast<GPUBufferUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline GPUBufferUsage operator&(GPUBufferUsage a, GPUBufferUsage b) {
    return static_cast<GPUBufferUsage>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline bool HasFlag(GPUBufferUsage value, GPUBufferUsage flag) {
    return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
}

// =============================================================================
// Memory Location
// =============================================================================

enum class GPUMemoryLocation {
    DeviceLocal,        // GPU memory (fastest, requires staging for CPU access)
    HostVisible,        // CPU-visible memory (slower GPU access, no staging needed)
    HostCached          // CPU-cached memory (good for GPU->CPU readback)
};

// =============================================================================
// GPU Buffer Base
// =============================================================================

class GPUBufferBase {
public:
    virtual ~GPUBufferBase() = default;

    /// Get buffer size in bytes
    virtual size_t GetSizeBytes() const = 0;

    /// Get element count
    virtual size_t GetCount() const = 0;

    /// Get element stride in bytes
    virtual size_t GetStride() const = 0;

    /// Check if buffer is valid
    virtual bool IsValid() const = 0;

    /// Get raw Vulkan buffer handle
    virtual VkBuffer GetVkBuffer() const = 0;

    /// Get buffer usage flags
    virtual GPUBufferUsage GetUsage() const = 0;

    /// Get memory location
    virtual GPUMemoryLocation GetMemoryLocation() const = 0;
};

// =============================================================================
// Typed GPU Buffer
// =============================================================================

template<typename T>
class ComputeBuffer : public GPUBufferBase {
    static_assert(std::is_trivially_copyable_v<T>, "ComputeBuffer element type must be trivially copyable");

public:
    ComputeBuffer() = default;

    /// Create a buffer with specified count and usage
    explicit ComputeBuffer(size_t count,
                          GPUBufferUsage usage = GPUBufferUsage::ComputeStorage,
                          GPUMemoryLocation location = GPUMemoryLocation::DeviceLocal);

    /// Create and initialize from data
    ComputeBuffer(const T* data, size_t count,
                  GPUBufferUsage usage = GPUBufferUsage::ComputeStorage,
                  GPUMemoryLocation location = GPUMemoryLocation::DeviceLocal);

    /// Create and initialize from vector
    explicit ComputeBuffer(const std::vector<T>& data,
                          GPUBufferUsage usage = GPUBufferUsage::ComputeStorage,
                          GPUMemoryLocation location = GPUMemoryLocation::DeviceLocal);

    ~ComputeBuffer() override;

    // Non-copyable, movable
    ComputeBuffer(const ComputeBuffer&) = delete;
    ComputeBuffer& operator=(const ComputeBuffer&) = delete;
    ComputeBuffer(ComputeBuffer&& other) noexcept;
    ComputeBuffer& operator=(ComputeBuffer&& other) noexcept;

    // ==========================================================================
    // Buffer Creation
    // ==========================================================================

    /// Allocate buffer without initialization
    bool Allocate(size_t count,
                  GPUBufferUsage usage = GPUBufferUsage::ComputeStorage,
                  GPUMemoryLocation location = GPUMemoryLocation::DeviceLocal);

    /// Release GPU resources
    void Release();

    /// Resize buffer (preserves data if possible)
    bool Resize(size_t newCount, bool preserveData = true);

    // ==========================================================================
    // Data Transfer
    // ==========================================================================

    /// Upload data from CPU to GPU
    bool Upload(const T* data, size_t count, size_t offset = 0);

    /// Upload data from vector
    bool Upload(const std::vector<T>& data, size_t offset = 0);

    /// Download data from GPU to CPU
    bool Download(T* data, size_t count, size_t offset = 0) const;

    /// Download data to vector
    bool Download(std::vector<T>& data) const;

    /// Set all elements to zero
    bool Clear();

    /// Fill with a single value
    bool Fill(const T& value);

    // ==========================================================================
    // Mapping (for host-visible buffers only)
    // ==========================================================================

    /// Map buffer for CPU access (host-visible only)
    T* Map();

    /// Map buffer for read-only CPU access (host-visible only)
    const T* MapRead() const;

    /// Unmap buffer
    void Unmap();

    /// Check if currently mapped
    bool IsMapped() const { return m_mappedPtr != nullptr; }

    // ==========================================================================
    // GPUBufferBase Interface
    // ==========================================================================

    size_t GetSizeBytes() const override { return m_count * sizeof(T); }
    size_t GetCount() const override { return m_count; }
    size_t GetStride() const override { return sizeof(T); }
    bool IsValid() const override { return m_buffer != nullptr; }
    VkBuffer GetVkBuffer() const override { return m_buffer; }
    GPUBufferUsage GetUsage() const override { return m_usage; }
    GPUMemoryLocation GetMemoryLocation() const override { return m_location; }

private:
    bool CreateBuffer(size_t sizeBytes, GPUBufferUsage usage, GPUMemoryLocation location);
    bool CreateStagingBuffer(size_t sizeBytes);
    void DestroyBuffer();
    void DestroyStagingBuffer();
    uint32_t FindMemoryType(uint32_t typeFilter, uint32_t properties) const;

    VkBuffer m_buffer = nullptr;
    VkDeviceMemory m_memory = nullptr;
    VkBuffer m_stagingBuffer = nullptr;
    VkDeviceMemory m_stagingMemory = nullptr;

    size_t m_count = 0;
    GPUBufferUsage m_usage = GPUBufferUsage::None;
    GPUMemoryLocation m_location = GPUMemoryLocation::DeviceLocal;

    mutable T* m_mappedPtr = nullptr;
};

// =============================================================================
// Convenience Type Aliases
// =============================================================================

using FloatBuffer = ComputeBuffer<float>;
using IntBuffer = ComputeBuffer<int32_t>;
using UIntBuffer = ComputeBuffer<uint32_t>;

// Common particle/simulation data types
struct alignas(16) Vec4 {
    float x, y, z, w;
};

struct alignas(16) ParticleData {
    Vec4 position;   // xyz = position, w = mass
    Vec4 velocity;   // xyz = velocity, w = density
};

using Vec4Buffer = ComputeBuffer<Vec4>;
using ParticleBuffer = ComputeBuffer<ParticleData>;

} // namespace WulfNet
