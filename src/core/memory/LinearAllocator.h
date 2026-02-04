// =============================================================================
// WulfNet Engine - Linear Allocator
// =============================================================================
// Fast bump allocator for frame-based temporary allocations
// O(1) allocation, O(1) reset - no individual deallocation
// =============================================================================

#pragma once

#include "Allocator.h"

namespace WulfNet {

class LinearAllocator final : public Allocator {
public:
    explicit LinearAllocator(usize capacity)
        : m_start(nullptr)
        , m_current(nullptr)
        , m_capacity(capacity)
        , m_totalAllocated(0)
        , m_ownsMemory(true)
    {
        m_start = static_cast<byte*>(Memory::systemAlloc(capacity, WULFNET_CACHE_LINE));
        m_current = m_start;
        WULFNET_ASSERT_MSG(m_start != nullptr, "Failed to allocate memory for LinearAllocator");
    }
    
    // Use pre-allocated memory buffer (does not take ownership)
    LinearAllocator(void* buffer, usize capacity)
        : m_start(static_cast<byte*>(buffer))
        , m_current(static_cast<byte*>(buffer))
        , m_capacity(capacity)
        , m_totalAllocated(0)
        , m_ownsMemory(false)
    {
        WULFNET_ASSERT(buffer != nullptr);
        WULFNET_ASSERT(capacity > 0);
    }
    
    ~LinearAllocator() override {
        if (m_ownsMemory && m_start) {
            Memory::systemFree(m_start);
        }
    }
    
    // Move operations
    LinearAllocator(LinearAllocator&& other) noexcept
        : m_start(other.m_start)
        , m_current(other.m_current)
        , m_capacity(other.m_capacity)
        , m_totalAllocated(other.m_totalAllocated)
        , m_ownsMemory(other.m_ownsMemory)
    {
        other.m_start = nullptr;
        other.m_current = nullptr;
        other.m_capacity = 0;
        other.m_totalAllocated = 0;
        other.m_ownsMemory = false;
    }
    
    LinearAllocator& operator=(LinearAllocator&& other) noexcept {
        if (this != &other) {
            if (m_ownsMemory && m_start) {
                Memory::systemFree(m_start);
            }
            m_start = other.m_start;
            m_current = other.m_current;
            m_capacity = other.m_capacity;
            m_totalAllocated = other.m_totalAllocated;
            m_ownsMemory = other.m_ownsMemory;
            
            other.m_start = nullptr;
            other.m_current = nullptr;
            other.m_capacity = 0;
            other.m_totalAllocated = 0;
            other.m_ownsMemory = false;
        }
        return *this;
    }
    
    [[nodiscard]] void* allocate(usize size, usize alignment = alignof(std::max_align_t)) override {
        WULFNET_ASSERT(isPowerOfTwo(alignment));
        
        usize adjustment = calculateAlignmentAdjustment(m_current, alignment);
        
        if (m_current + adjustment + size > m_start + m_capacity) {
            // Out of memory - could log warning here
            return nullptr;
        }
        
        byte* alignedAddress = m_current + adjustment;
        m_current = alignedAddress + size;
        m_totalAllocated = static_cast<usize>(m_current - m_start);
        
        return alignedAddress;
    }
    
    void deallocate([[maybe_unused]] void* ptr) override {
        // Linear allocator doesn't support individual deallocations
        // Use reset() to free all memory at once
    }
    
    void reset() override {
        m_current = m_start;
        m_totalAllocated = 0;
    }
    
    // Reset to a specific marker (for nested scopes)
    void resetTo(void* marker) {
        WULFNET_ASSERT(marker >= m_start && marker <= m_current);
        m_current = static_cast<byte*>(marker);
        m_totalAllocated = static_cast<usize>(m_current - m_start);
    }
    
    // Get current position as marker for later reset
    void* getMarker() const {
        return m_current;
    }
    
    usize getTotalAllocated() const override { return m_totalAllocated; }
    usize getCapacity() const override { return m_capacity; }
    usize getRemainingSpace() const { return m_capacity - m_totalAllocated; }
    
private:
    byte* m_start;
    byte* m_current;
    usize m_capacity;
    usize m_totalAllocated;
    bool m_ownsMemory;
};

// =============================================================================
// Scoped Linear Allocator (RAII reset on scope exit)
// =============================================================================

class ScopedLinearAllocator {
public:
    explicit ScopedLinearAllocator(LinearAllocator& allocator)
        : m_allocator(allocator)
        , m_marker(allocator.getMarker())
    {}
    
    ~ScopedLinearAllocator() {
        m_allocator.resetTo(m_marker);
    }
    
    [[nodiscard]] void* allocate(usize size, usize alignment = alignof(std::max_align_t)) {
        return m_allocator.allocate(size, alignment);
    }
    
    template<typename T, typename... Args>
    [[nodiscard]] T* create(Args&&... args) {
        return m_allocator.create<T>(std::forward<Args>(args)...);
    }
    
private:
    LinearAllocator& m_allocator;
    void* m_marker;
};

} // namespace WulfNet
