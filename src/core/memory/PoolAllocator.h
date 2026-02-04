// =============================================================================
// WulfNet Engine - Pool Allocator
// =============================================================================
// Fixed-size block allocator for objects of uniform size (entities, particles)
// O(1) allocation and deallocation using a free list
// =============================================================================

#pragma once

#include "Allocator.h"
#include <atomic>

namespace WulfNet {

class PoolAllocator final : public Allocator {
public:
    PoolAllocator(usize blockSize, usize blockCount, usize alignment = alignof(std::max_align_t))
        : m_start(nullptr)
        , m_freeList(nullptr)
        , m_blockSize(0)
        , m_blockCount(blockCount)
        , m_alignment(alignment)
        , m_allocatedCount(0)
        , m_ownsMemory(true)
    {
        WULFNET_ASSERT(blockCount > 0);
        WULFNET_ASSERT(isPowerOfTwo(alignment));
        
        // Block size must be at least pointer size for free list and aligned
        m_blockSize = alignUp(std::max(blockSize, sizeof(void*)), alignment);
        
        usize totalSize = m_blockSize * m_blockCount;
        m_start = static_cast<byte*>(Memory::systemAlloc(totalSize, alignment));
        WULFNET_ASSERT_MSG(m_start != nullptr, "Failed to allocate memory for PoolAllocator");
        
        initializeFreeList();
    }
    
    // Use pre-allocated buffer
    PoolAllocator(void* buffer, usize bufferSize, usize blockSize, usize alignment = alignof(std::max_align_t))
        : m_start(static_cast<byte*>(buffer))
        , m_freeList(nullptr)
        , m_blockSize(0)
        , m_blockCount(0)
        , m_alignment(alignment)
        , m_allocatedCount(0)
        , m_ownsMemory(false)
    {
        WULFNET_ASSERT(buffer != nullptr);
        WULFNET_ASSERT(bufferSize > 0);
        WULFNET_ASSERT(isPowerOfTwo(alignment));
        
        m_blockSize = alignUp(std::max(blockSize, sizeof(void*)), alignment);
        m_blockCount = bufferSize / m_blockSize;
        
        WULFNET_ASSERT(m_blockCount > 0);
        initializeFreeList();
    }
    
    ~PoolAllocator() override {
        if (m_ownsMemory && m_start) {
            Memory::systemFree(m_start);
        }
    }
    
    // Non-copyable but movable
    PoolAllocator(PoolAllocator&& other) noexcept
        : m_start(other.m_start)
        , m_freeList(other.m_freeList)
        , m_blockSize(other.m_blockSize)
        , m_blockCount(other.m_blockCount)
        , m_alignment(other.m_alignment)
        , m_allocatedCount(other.m_allocatedCount)
        , m_ownsMemory(other.m_ownsMemory)
    {
        other.m_start = nullptr;
        other.m_freeList = nullptr;
        other.m_ownsMemory = false;
    }
    
    PoolAllocator& operator=(PoolAllocator&& other) noexcept {
        if (this != &other) {
            if (m_ownsMemory && m_start) {
                Memory::systemFree(m_start);
            }
            
            m_start = other.m_start;
            m_freeList = other.m_freeList;
            m_blockSize = other.m_blockSize;
            m_blockCount = other.m_blockCount;
            m_alignment = other.m_alignment;
            m_allocatedCount = other.m_allocatedCount;
            m_ownsMemory = other.m_ownsMemory;
            
            other.m_start = nullptr;
            other.m_freeList = nullptr;
            other.m_ownsMemory = false;
        }
        return *this;
    }
    
    [[nodiscard]] void* allocate([[maybe_unused]] usize size, 
                                  [[maybe_unused]] usize alignment = alignof(std::max_align_t)) override {
        WULFNET_ASSERT_MSG(size <= m_blockSize, "Requested size exceeds pool block size");
        WULFNET_ASSERT_MSG(alignment <= m_alignment, "Requested alignment exceeds pool alignment");
        
        if (m_freeList == nullptr) {
            // Pool exhausted
            return nullptr;
        }
        
        // Pop from free list
        void* block = m_freeList;
        m_freeList = *static_cast<void**>(m_freeList);
        m_allocatedCount++;
        
        return block;
    }
    
    void deallocate(void* ptr) override {
        if (ptr == nullptr) return;
        
        WULFNET_ASSERT_MSG(ownsPointer(ptr), "Pointer does not belong to this pool");
        
        // Push to free list
        *static_cast<void**>(ptr) = m_freeList;
        m_freeList = ptr;
        m_allocatedCount--;
    }
    
    void reset() override {
        m_allocatedCount = 0;
        initializeFreeList();
    }
    
    // Query methods
    usize getTotalAllocated() const override { return m_allocatedCount * m_blockSize; }
    usize getCapacity() const override { return m_blockCount * m_blockSize; }
    usize getBlockSize() const { return m_blockSize; }
    usize getBlockCount() const { return m_blockCount; }
    usize getAllocatedBlockCount() const { return m_allocatedCount; }
    usize getFreeBlockCount() const { return m_blockCount - m_allocatedCount; }
    bool isFull() const { return m_freeList == nullptr; }
    bool isEmpty() const { return m_allocatedCount == 0; }
    
    // Check if a pointer belongs to this pool
    bool ownsPointer(const void* ptr) const {
        const byte* p = static_cast<const byte*>(ptr);
        return p >= m_start && p < m_start + (m_blockCount * m_blockSize);
    }
    
    // Get block index from pointer
    usize getBlockIndex(const void* ptr) const {
        WULFNET_ASSERT(ownsPointer(ptr));
        return static_cast<usize>(static_cast<const byte*>(ptr) - m_start) / m_blockSize;
    }
    
    // Get pointer from block index
    void* getBlockPointer(usize index) const {
        WULFNET_ASSERT(index < m_blockCount);
        return m_start + (index * m_blockSize);
    }
    
private:
    void initializeFreeList() {
        // Initialize free list - each block points to the next
        m_freeList = m_start;
        
        for (usize i = 0; i < m_blockCount - 1; i++) {
            void* current = m_start + (i * m_blockSize);
            void* next = m_start + ((i + 1) * m_blockSize);
            *static_cast<void**>(current) = next;
        }
        
        // Last block points to null
        void* last = m_start + ((m_blockCount - 1) * m_blockSize);
        *static_cast<void**>(last) = nullptr;
    }
    
    byte* m_start;
    void* m_freeList;
    usize m_blockSize;
    usize m_blockCount;
    usize m_alignment;
    usize m_allocatedCount;
    bool m_ownsMemory;
};

// =============================================================================
// Typed Pool Allocator (Convenience wrapper for specific types)
// =============================================================================

template<typename T>
class TypedPoolAllocator {
public:
    explicit TypedPoolAllocator(usize count)
        : m_pool(sizeof(T), count, alignof(T))
    {}
    
    template<typename... Args>
    [[nodiscard]] T* create(Args&&... args) {
        void* memory = m_pool.allocate(sizeof(T), alignof(T));
        if (!memory) return nullptr;
        return new (memory) T(std::forward<Args>(args)...);
    }
    
    void destroy(T* ptr) {
        if (ptr) {
            ptr->~T();
            m_pool.deallocate(ptr);
        }
    }
    
    void reset() { m_pool.reset(); }
    usize getAllocatedCount() const { return m_pool.getAllocatedBlockCount(); }
    usize getFreeCount() const { return m_pool.getFreeBlockCount(); }
    usize getCapacity() const { return m_pool.getBlockCount(); }
    bool isFull() const { return m_pool.isFull(); }
    
private:
    PoolAllocator m_pool;
};

} // namespace WulfNet
