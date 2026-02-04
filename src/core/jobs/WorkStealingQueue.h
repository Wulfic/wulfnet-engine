// =============================================================================
// WulfNet Engine - Lock-Free Work Stealing Queue
// =============================================================================
// Chase-Lev deque for efficient work stealing between worker threads
// Based on: "Dynamic Circular Work-Stealing Deque" by Chase and Lev
// =============================================================================

#pragma once

#include "../Types.h"
#include "../Assert.h"
#include "../memory/Memory.h"
#include "Job.h"
#include <atomic>
#include <memory>

namespace WulfNet {

// =============================================================================
// Work Stealing Queue (Lock-Free Chase-Lev Deque)
// =============================================================================

class WorkStealingQueue : public NonCopyable {
public:
    static constexpr u32 INITIAL_CAPACITY = 1024;
    static constexpr u32 MAX_CAPACITY = 1024 * 1024;
    
    WorkStealingQueue(u32 capacity = INITIAL_CAPACITY)
        : m_bottom(0)
        , m_top(0)
    {
        WULFNET_ASSERT(isPowerOfTwo(capacity));
        m_buffer.store(new CircularBuffer(capacity), std::memory_order_relaxed);
    }
    
    ~WorkStealingQueue() {
        delete m_buffer.load(std::memory_order_relaxed);
    }
    
    // Move constructor
    WorkStealingQueue(WorkStealingQueue&& other) noexcept
        : m_bottom(other.m_bottom.load(std::memory_order_relaxed))
        , m_top(other.m_top.load(std::memory_order_relaxed))
    {
        m_buffer.store(other.m_buffer.exchange(nullptr, std::memory_order_relaxed), 
                       std::memory_order_relaxed);
    }
    
    // Move assignment
    WorkStealingQueue& operator=(WorkStealingQueue&& other) noexcept {
        if (this != &other) {
            delete m_buffer.load(std::memory_order_relaxed);
            m_buffer.store(other.m_buffer.exchange(nullptr, std::memory_order_relaxed),
                           std::memory_order_relaxed);
            m_bottom.store(other.m_bottom.load(std::memory_order_relaxed), std::memory_order_relaxed);
            m_top.store(other.m_top.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
    
    // =========================================================================
    // Push (called by owner thread only)
    // =========================================================================
    
    void push(Job* job) {
        i64 bottom = m_bottom.load(std::memory_order_relaxed);
        i64 top = m_top.load(std::memory_order_acquire);
        CircularBuffer* buffer = m_buffer.load(std::memory_order_relaxed);
        
        i64 size = bottom - top;
        
        // Check if we need to grow
        if (size >= static_cast<i64>(buffer->capacity) - 1) {
            buffer = grow(buffer, bottom, top);
            m_buffer.store(buffer, std::memory_order_relaxed);
        }
        
        buffer->set(bottom, job);
        std::atomic_thread_fence(std::memory_order_release);
        m_bottom.store(bottom + 1, std::memory_order_relaxed);
    }
    
    // =========================================================================
    // Pop (called by owner thread only)
    // =========================================================================
    
    Job* pop() {
        i64 bottom = m_bottom.load(std::memory_order_relaxed) - 1;
        CircularBuffer* buffer = m_buffer.load(std::memory_order_relaxed);
        
        m_bottom.store(bottom, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        i64 top = m_top.load(std::memory_order_relaxed);
        
        if (top <= bottom) {
            // Non-empty queue
            Job* job = buffer->get(bottom);
            
            if (top == bottom) {
                // Last element - compete with stealers
                if (!m_top.compare_exchange_strong(top, top + 1,
                    std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    // Lost race to stealer
                    job = nullptr;
                }
                m_bottom.store(bottom + 1, std::memory_order_relaxed);
            }
            
            return job;
        } else {
            // Empty queue
            m_bottom.store(bottom + 1, std::memory_order_relaxed);
            return nullptr;
        }
    }
    
    // =========================================================================
    // Steal (called by other threads)
    // =========================================================================
    
    Job* steal() {
        i64 top = m_top.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        i64 bottom = m_bottom.load(std::memory_order_acquire);
        
        if (top < bottom) {
            // Non-empty queue
            CircularBuffer* buffer = m_buffer.load(std::memory_order_consume);
            Job* job = buffer->get(top);
            
            if (!m_top.compare_exchange_strong(top, top + 1,
                std::memory_order_seq_cst, std::memory_order_relaxed)) {
                // Lost race with another stealer or owner
                return nullptr;
            }
            
            return job;
        }
        
        return nullptr;
    }
    
    // =========================================================================
    // Query
    // =========================================================================
    
    usize size() const {
        i64 bottom = m_bottom.load(std::memory_order_relaxed);
        i64 top = m_top.load(std::memory_order_relaxed);
        return static_cast<usize>(bottom >= top ? bottom - top : 0);
    }
    
    bool empty() const {
        return size() == 0;
    }
    
private:
    // =========================================================================
    // Circular Buffer
    // =========================================================================
    
    struct CircularBuffer {
        u32 capacity;
        u32 mask;
        Job** data;
        
        explicit CircularBuffer(u32 cap)
            : capacity(cap)
            , mask(cap - 1)
        {
            data = new Job*[cap];
            Memory::zero(data, sizeof(Job*) * cap);
        }
        
        ~CircularBuffer() {
            delete[] data;
        }
        
        Job* get(i64 index) const {
            return data[index & mask];
        }
        
        void set(i64 index, Job* job) {
            data[index & mask] = job;
        }
        
        CircularBuffer* grow(i64 bottom, i64 top) const {
            u32 newCapacity = capacity * 2;
            WULFNET_ASSERT(newCapacity <= MAX_CAPACITY);
            
            CircularBuffer* newBuffer = new CircularBuffer(newCapacity);
            
            for (i64 i = top; i < bottom; i++) {
                newBuffer->set(i, get(i));
            }
            
            return newBuffer;
        }
    };
    
    CircularBuffer* grow(CircularBuffer* old, i64 bottom, i64 top) {
        CircularBuffer* newBuffer = old->grow(bottom, top);
        // Note: Old buffer will be leaked. In production, use hazard pointers
        // or epoch-based reclamation for safe memory reclamation.
        return newBuffer;
    }
    
    // =========================================================================
    // Data Members
    // =========================================================================
    
    WULFNET_CACHE_ALIGNED std::atomic<i64> m_bottom;
    WULFNET_CACHE_ALIGNED std::atomic<i64> m_top;
    WULFNET_CACHE_ALIGNED std::atomic<CircularBuffer*> m_buffer;
};

} // namespace WulfNet
