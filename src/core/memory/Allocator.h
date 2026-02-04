// =============================================================================
// WulfNet Engine - Memory Allocator Interface
// =============================================================================
// Base allocator interface for all memory allocation strategies
// =============================================================================

#pragma once

#include "../Types.h"
#include "../Assert.h"
#include <new>

namespace WulfNet {

// =============================================================================
// Allocator Base Interface
// =============================================================================

class Allocator : public NonCopyable {
public:
    virtual ~Allocator() = default;
    
    // Core allocation interface
    [[nodiscard]] virtual void* allocate(usize size, usize alignment = alignof(std::max_align_t)) = 0;
    virtual void deallocate(void* ptr) = 0;
    
    // Query methods
    virtual usize getTotalAllocated() const = 0;
    virtual usize getCapacity() const = 0;
    
    // Optional: Reset all allocations (for linear/stack allocators)
    virtual void reset() {}
    
    // Typed allocation helpers
    template<typename T, typename... Args>
    [[nodiscard]] T* create(Args&&... args) {
        void* memory = allocate(sizeof(T), alignof(T));
        if (!memory) return nullptr;
        return new (memory) T(std::forward<Args>(args)...);
    }
    
    template<typename T>
    void destroy(T* ptr) {
        if (ptr) {
            ptr->~T();
            deallocate(ptr);
        }
    }
    
    template<typename T>
    [[nodiscard]] T* allocateArray(usize count) {
        return static_cast<T*>(allocate(sizeof(T) * count, alignof(T)));
    }
    
protected:
    Allocator() = default;
};

// =============================================================================
// Allocation Header (for tracking allocations)
// =============================================================================

struct AllocationHeader {
    usize size;
    usize adjustment;  // Padding for alignment
};

// =============================================================================
// Utility Functions
// =============================================================================

// Calculate adjustment needed to align pointer
inline usize calculateAlignmentAdjustment(const void* address, usize alignment) {
    usize addr = reinterpret_cast<usize>(address);
    usize mask = alignment - 1;
    usize misalignment = addr & mask;
    return misalignment == 0 ? 0 : alignment - misalignment;
}

// Calculate adjustment with header space
inline usize calculateAlignmentAdjustmentWithHeader(const void* address, usize alignment, 
                                                     usize headerSize) {
    usize adjustment = calculateAlignmentAdjustment(address, alignment);
    
    if (adjustment < headerSize) {
        headerSize -= adjustment;
        adjustment += alignment * ((headerSize + alignment - 1) / alignment);
    }
    
    return adjustment;
}

// =============================================================================
// Global Memory Operations
// =============================================================================

namespace Memory {
    // Default system allocator
    [[nodiscard]] void* systemAlloc(usize size, usize alignment = alignof(std::max_align_t));
    void systemFree(void* ptr);
    
    // Memory operations
    void copy(void* dst, const void* src, usize size);
    void move(void* dst, const void* src, usize size);
    void set(void* dst, u8 value, usize size);
    void zero(void* dst, usize size);
    
    int compare(const void* a, const void* b, usize size);
}

} // namespace WulfNet
