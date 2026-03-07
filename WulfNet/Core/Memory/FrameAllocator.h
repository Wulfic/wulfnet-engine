// =============================================================================
// WulfNet Engine - Frame Allocator
// =============================================================================
// A linear (bump) allocator that resets each frame. Provides O(1) allocation
// with zero per-allocation overhead. Ideal for per-frame transient data:
// temp buffers, active tile lists, scratch arrays, etc.
//
// Usage:
//   FrameAllocator::Get().BeginFrame();   // call once at start of frame
//   float* buf = FrameAllocator::Get().Alloc<float>(1024);  // zero-cost alloc
//   // ... use buf within this frame ...
//   // buf is implicitly freed at next BeginFrame()
//
// Thread safety:
//   NOT thread-safe. Use one allocator per thread, or protect with a mutex.
//   For the common case (main thread only), no synchronization needed.
// =============================================================================

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <new>

namespace WulfNet {

class FrameAllocator {
public:
    /// Get the singleton instance (main-thread allocator).
    static FrameAllocator& Get() {
        static FrameAllocator instance;
        return instance;
    }

    /// Initialize with a given capacity in bytes. Called once at startup.
    /// Subsequent calls with a larger capacity will grow the buffer.
    void Initialize(size_t capacityBytes = DEFAULT_CAPACITY) {
        if (capacityBytes <= m_capacity && m_buffer) return;
        delete[] m_buffer;
        m_buffer = new(std::nothrow) uint8_t[capacityBytes];
        m_capacity = m_buffer ? capacityBytes : 0;
        m_offset = 0;
        m_peakUsage = 0;
    }

    /// Reset the allocator. Call once at the start of each frame.
    /// All previous allocations are invalidated.
    void BeginFrame() {
        m_offset = 0;
    }

    /// Allocate `count` elements of type T with proper alignment.
    /// Returns nullptr if out of space (should not happen if sized correctly).
    template<typename T>
    T* Alloc(size_t count) {
        return reinterpret_cast<T*>(AllocRaw(count * sizeof(T), alignof(T)));
    }

    /// Allocate `count` elements of type T, zero-initialized.
    template<typename T>
    T* AllocZeroed(size_t count) {
        T* ptr = Alloc<T>(count);
        if (ptr) {
            std::memset(ptr, 0, count * sizeof(T));
        }
        return ptr;
    }

    /// Raw allocation with explicit size and alignment.
    void* AllocRaw(size_t sizeBytes, size_t alignment = 16) {
        if (!m_buffer) {
            Initialize();
        }

        // Align the current offset up to the requested alignment
        size_t aligned = (m_offset + alignment - 1) & ~(alignment - 1);

        if (aligned + sizeBytes > m_capacity) {
            // Out of space — this indicates the capacity is too small.
            // In debug builds, this would assert. In release, return nullptr.
            return nullptr;
        }

        void* ptr = m_buffer + aligned;
        m_offset = aligned + sizeBytes;

        if (m_offset > m_peakUsage) {
            m_peakUsage = m_offset;
        }

        return ptr;
    }

    /// Get current usage in bytes (since last BeginFrame).
    size_t GetCurrentUsage() const { return m_offset; }

    /// Get peak usage across all frames (useful for tuning capacity).
    size_t GetPeakUsage() const { return m_peakUsage; }

    /// Get total capacity in bytes.
    size_t GetCapacity() const { return m_capacity; }

    /// Shutdown — free the backing buffer.
    void Shutdown() {
        delete[] m_buffer;
        m_buffer = nullptr;
        m_capacity = 0;
        m_offset = 0;
    }

    ~FrameAllocator() {
        Shutdown();
    }

    // Default capacity: 4 MB — enough for most frame-transient data.
    static constexpr size_t DEFAULT_CAPACITY = 4 * 1024 * 1024;

private:
    FrameAllocator() = default;
    FrameAllocator(const FrameAllocator&) = delete;
    FrameAllocator& operator=(const FrameAllocator&) = delete;

    uint8_t* m_buffer   = nullptr;
    size_t   m_capacity = 0;
    size_t   m_offset   = 0;
    size_t   m_peakUsage = 0;
};

} // namespace WulfNet
