// =============================================================================
// WulfNet Engine - Job Definition
// =============================================================================
// Job structure and related types for the parallel job system
// =============================================================================

#pragma once

#include "../Types.h"
#include "../Assert.h"
#include "../memory/Memory.h"
#include <atomic>
#include <functional>

namespace WulfNet {

// =============================================================================
// Job Priority Levels
// =============================================================================

enum class JobPriority : u8 {
    Critical = 0,   // Physics simulation, time-critical work
    High = 1,       // Rendering preparation
    Normal = 2,     // General gameplay, audio
    Low = 3,        // Background I/O, streaming
    Count
};

// =============================================================================
// Job Handle
// =============================================================================

struct JobHandle {
    u32 index = 0;
    u32 generation = 0;
    
    bool isValid() const { return generation != 0; }
    bool operator==(const JobHandle& other) const {
        return index == other.index && generation == other.generation;
    }
    bool operator!=(const JobHandle& other) const {
        return !(*this == other);
    }
    
    static JobHandle invalid() { return JobHandle{0, 0}; }
};

// =============================================================================
// Job Function Types
// =============================================================================

// Standard job function: void(void* userData)
using JobFunction = void(*)(void* userData);

// Parallel-for job function: void(u32 itemIndex, void* userData)
using ParallelForFunction = void(*)(u32 itemIndex, void* userData);

// Lambda-based job (heap allocated, use sparingly)
using JobLambda = std::function<void()>;

// =============================================================================
// Job Structure
// =============================================================================

struct alignas(64) Job {
    static constexpr usize PAYLOAD_SIZE = 48;
    
    JobFunction function = nullptr;
    JobHandle parent;                          // Parent job for dependencies
    std::atomic<i32> unfinishedJobs{0};        // Number of unfinished child jobs
    JobPriority priority = JobPriority::Normal;
    u8 padding[3] = {0};
    
    // Inline payload to avoid heap allocation for small data
    alignas(8) u8 payload[PAYLOAD_SIZE];
    
    void setPayload(const void* data, usize size) {
        WULFNET_ASSERT(size <= PAYLOAD_SIZE);
        Memory::copy(payload, data, size);
    }
    
    template<typename T>
    void setPayload(const T& data) {
        static_assert(sizeof(T) <= PAYLOAD_SIZE, "Payload too large");
        static_assert(alignof(T) <= 8, "Payload alignment too strict");
        new (payload) T(data);
    }
    
    template<typename T>
    T& getPayload() {
        return *reinterpret_cast<T*>(payload);
    }
    
    template<typename T>
    const T& getPayload() const {
        return *reinterpret_cast<const T*>(payload);
    }
};

// Ensure cache-line alignment to prevent false sharing
static_assert(sizeof(Job) == 64 || sizeof(Job) == 128, 
    "Job should be cache-line sized");

// =============================================================================
// Job Counter (for synchronization)
// =============================================================================

struct JobCounter {
    std::atomic<i32> value{0};
    
    void increment() { value.fetch_add(1, std::memory_order_relaxed); }
    void decrement() { value.fetch_sub(1, std::memory_order_release); }
    i32 get() const { return value.load(std::memory_order_acquire); }
    bool isZero() const { return get() == 0; }
    
    void wait() const {
        while (!isZero()) {
            // Spin with pause hint
            #if WULFNET_COMPILER_MSVC
                _mm_pause();
            #else
                __builtin_ia32_pause();
            #endif
        }
    }
};

// =============================================================================
// Job Group (for batch submission)
// =============================================================================

struct JobGroup {
    JobHandle* handles = nullptr;
    u32 count = 0;
    std::atomic<u32> completedCount{0};
    
    bool isComplete() const { 
        return completedCount.load(std::memory_order_acquire) >= count; 
    }
};

// =============================================================================
// Parallel For Descriptor
// =============================================================================

struct ParallelForDesc {
    ParallelForFunction function = nullptr;
    void* userData = nullptr;
    u32 itemCount = 0;
    u32 minBatchSize = 1;      // Minimum items per job
    JobPriority priority = JobPriority::Normal;
    JobHandle dependency;       // Optional dependency
};

} // namespace WulfNet
