// =============================================================================
// WulfNet Engine - Memory Allocator Implementation
// =============================================================================

#include "Allocator.h"
#include <cstdlib>
#include <cstring>

namespace WulfNet::Memory {

void* systemAlloc(usize size, usize alignment) {
    #if WULFNET_PLATFORM_WINDOWS
        return _aligned_malloc(size, alignment);
    #else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return nullptr;
        }
        return ptr;
    #endif
}

void systemFree(void* ptr) {
    #if WULFNET_PLATFORM_WINDOWS
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

void copy(void* dst, const void* src, usize size) {
    std::memcpy(dst, src, size);
}

void move(void* dst, const void* src, usize size) {
    std::memmove(dst, src, size);
}

void set(void* dst, u8 value, usize size) {
    std::memset(dst, value, size);
}

void zero(void* dst, usize size) {
    std::memset(dst, 0, size);
}

int compare(const void* a, const void* b, usize size) {
    return std::memcmp(a, b, size);
}

} // namespace WulfNet::Memory
