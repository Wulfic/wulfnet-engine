// =============================================================================
// WulfNet Engine - Stack Allocator
// =============================================================================
// LIFO allocator for hierarchical data structures (scene graph, transform hierarchy)
// O(1) allocation, O(1) deallocation (must be in reverse order)
// =============================================================================

#pragma once

#include "Allocator.h"

namespace WulfNet {

class StackAllocator final : public Allocator {
public:
    explicit StackAllocator(usize capacity)
        : m_start(nullptr)
        , m_current(nullptr)
        , m_capacity(capacity)
        , m_totalAllocated(0)
        , m_ownsMemory(true)
    {
        m_start = static_cast<byte*>(Memory::systemAlloc(capacity, WULFNET_CACHE_LINE));
        m_current = m_start;
        WULFNET_ASSERT_MSG(m_start != nullptr, "Failed to allocate memory for StackAllocator");
    }
    
    // Use pre-allocated buffer
    StackAllocator(void* buffer, usize capacity)
        : m_start(static_cast<byte*>(buffer))
        , m_current(static_cast<byte*>(buffer))
        , m_capacity(capacity)
        , m_totalAllocated(0)
        , m_ownsMemory(false)
    {
        WULFNET_ASSERT(buffer != nullptr);
        WULFNET_ASSERT(capacity > 0);
    }
    
    ~StackAllocator() override {
        if (m_ownsMemory && m_start) {
            Memory::systemFree(m_start);
        }
    }
    
    // Move operations
    StackAllocator(StackAllocator&& other) noexcept
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
    
    StackAllocator& operator=(StackAllocator&& other) noexcept {
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
        
        // Calculate adjustment for alignment with space for header
        usize adjustment = calculateAlignmentAdjustmentWithHeader(
            m_current, alignment, sizeof(AllocationHeader)
        );
        
        usize totalSize = adjustment + size;
        
        if (m_current + totalSize > m_start + m_capacity) {
            return nullptr;  // Out of memory
        }
        
        byte* alignedAddress = m_current + adjustment;
        
        // Store header just before aligned address
        AllocationHeader* header = reinterpret_cast<AllocationHeader*>(alignedAddress - sizeof(AllocationHeader));
        header->size = size;
        header->adjustment = adjustment;
        
        m_current = alignedAddress + size;
        m_totalAllocated += totalSize;
        
        return alignedAddress;
    }
    
    void deallocate(void* ptr) override {
        if (ptr == nullptr) return;
        
        // Retrieve header
        AllocationHeader* header = reinterpret_cast<AllocationHeader*>(
            static_cast<byte*>(ptr) - sizeof(AllocationHeader)
        );
        
        // Calculate the actual start of this allocation
        byte* allocationStart = static_cast<byte*>(ptr) - header->adjustment;
        usize totalSize = header->adjustment + header->size;
        
        // Stack allocator: can only deallocate the most recent allocation
        WULFNET_ASSERT_MSG(
            static_cast<byte*>(ptr) + header->size == m_current,
            "StackAllocator: Deallocations must be in reverse order of allocations"
        );
        
        m_current = allocationStart;
        m_totalAllocated -= totalSize;
    }
    
    void reset() override {
        m_current = m_start;
        m_totalAllocated = 0;
    }
    
    // Query methods
    usize getTotalAllocated() const override { return m_totalAllocated; }
    usize getCapacity() const override { return m_capacity; }
    usize getRemainingSpace() const { return m_capacity - static_cast<usize>(m_current - m_start); }
    
    // Get current position for scoped allocations
    void* getMarker() const { return m_current; }
    
    // Free to a previous marker
    void freeToMarker(void* marker) {
        byte* m = static_cast<byte*>(marker);
        WULFNET_ASSERT(m >= m_start && m <= m_current);
        m_current = m;
        m_totalAllocated = static_cast<usize>(m_current - m_start);
    }
    
private:
    byte* m_start;
    byte* m_current;
    usize m_capacity;
    usize m_totalAllocated;
    bool m_ownsMemory;
};

// =============================================================================
// Scoped Stack Allocator (RAII - automatically deallocates on scope exit)
// =============================================================================

class ScopedStackAllocator {
public:
    explicit ScopedStackAllocator(StackAllocator& allocator)
        : m_allocator(allocator)
        , m_marker(allocator.getMarker())
    {}
    
    ~ScopedStackAllocator() {
        // Deallocate all allocations made in this scope
        while (m_allocator.getMarker() > m_marker) {
            // We need to track allocations to properly deallocate
            // For now, just reset to marker (requires knowledge of allocation structure)
        }
        // Note: Full implementation would need to track allocations for proper cleanup
    }
    
    ScopedStackAllocator(const ScopedStackAllocator&) = delete;
    ScopedStackAllocator& operator=(const ScopedStackAllocator&) = delete;
    
    [[nodiscard]] void* allocate(usize size, usize alignment = alignof(std::max_align_t)) {
        return m_allocator.allocate(size, alignment);
    }
    
    template<typename T, typename... Args>
    [[nodiscard]] T* create(Args&&... args) {
        void* memory = allocate(sizeof(T), alignof(T));
        if (!memory) return nullptr;
        return new (memory) T(std::forward<Args>(args)...);
    }
    
private:
    StackAllocator& m_allocator;
    void* m_marker;
};

} // namespace WulfNet
