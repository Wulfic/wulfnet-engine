// =============================================================================
// WulfNet Engine - Fiber System
// =============================================================================
// Lightweight cooperative threading for job suspension/resumption
// =============================================================================

#pragma once

#include "../Types.h"
#include "../Assert.h"

#if WULFNET_PLATFORM_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
#else
    #include <ucontext.h>
#endif

namespace WulfNet {

// =============================================================================
// Fiber Handle
// =============================================================================

#if WULFNET_PLATFORM_WINDOWS
    using FiberHandle = void*;
#else
    using FiberHandle = ucontext_t*;
#endif

// =============================================================================
// Fiber Entry Function
// =============================================================================

using FiberEntryFunc = void(*)(void* userData);

// =============================================================================
// Fiber Class
// =============================================================================

class Fiber : public NonCopyable {
public:
    static constexpr usize DEFAULT_STACK_SIZE = 64 * 1024;  // 64KB stack
    
    Fiber() : m_handle(nullptr), m_userData(nullptr), m_stackSize(0), m_isThreadFiber(false) {}
    
    ~Fiber() {
        destroy();
    }
    
    // Move operations
    Fiber(Fiber&& other) noexcept
        : m_handle(other.m_handle)
        , m_userData(other.m_userData)
        , m_stackSize(other.m_stackSize)
        , m_isThreadFiber(other.m_isThreadFiber)
    {
        other.m_handle = nullptr;
        other.m_userData = nullptr;
        other.m_isThreadFiber = false;
    }
    
    Fiber& operator=(Fiber&& other) noexcept {
        if (this != &other) {
            destroy();
            m_handle = other.m_handle;
            m_userData = other.m_userData;
            m_stackSize = other.m_stackSize;
            m_isThreadFiber = other.m_isThreadFiber;
            other.m_handle = nullptr;
            other.m_userData = nullptr;
            other.m_isThreadFiber = false;
        }
        return *this;
    }
    
    // =========================================================================
    // Creation / Destruction
    // =========================================================================
    
    bool create(FiberEntryFunc entryPoint, void* userData = nullptr, 
                usize stackSize = DEFAULT_STACK_SIZE) {
        WULFNET_ASSERT(m_handle == nullptr);
        
        m_userData = userData;
        m_stackSize = stackSize;
        
        #if WULFNET_PLATFORM_WINDOWS
            m_handle = CreateFiber(stackSize, 
                reinterpret_cast<LPFIBER_START_ROUTINE>(entryPoint), userData);
            return m_handle != nullptr;
        #else
            m_handle = new ucontext_t;
            if (getcontext(m_handle) != 0) {
                delete m_handle;
                m_handle = nullptr;
                return false;
            }
            
            m_stack = new u8[stackSize];
            m_handle->uc_stack.ss_sp = m_stack;
            m_handle->uc_stack.ss_size = stackSize;
            m_handle->uc_link = nullptr;
            
            makecontext(m_handle, reinterpret_cast<void(*)()>(entryPoint), 1, userData);
            return true;
        #endif
    }
    
    void destroy() {
        if (m_handle) {
            if (!m_isThreadFiber) {
                #if WULFNET_PLATFORM_WINDOWS
                    DeleteFiber(m_handle);
                #else
                    delete[] m_stack;
                    delete m_handle;
                    m_stack = nullptr;
                #endif
            }
            m_handle = nullptr;
        }
    }
    
    // =========================================================================
    // Thread-to-Fiber Conversion
    // =========================================================================
    
    static Fiber convertThreadToFiber(void* userData = nullptr) {
        Fiber fiber;
        
        #if WULFNET_PLATFORM_WINDOWS
            fiber.m_handle = ConvertThreadToFiber(userData);
        #else
            fiber.m_handle = new ucontext_t;
            getcontext(fiber.m_handle);
        #endif
        
        fiber.m_userData = userData;
        fiber.m_isThreadFiber = true;
        return fiber;
    }
    
    static void convertFiberToThread() {
        #if WULFNET_PLATFORM_WINDOWS
            ConvertFiberToThread();
        #endif
        // On POSIX, nothing special needed
    }
    
    // =========================================================================
    // Switching
    // =========================================================================
    
    void switchTo() const {
        WULFNET_ASSERT(m_handle != nullptr);
        
        #if WULFNET_PLATFORM_WINDOWS
            SwitchToFiber(m_handle);
        #else
            // On POSIX, we need the current context
            // This is typically handled by the job system
            WULFNET_ASSERT_MSG(false, "Use switchTo(Fiber& from) on POSIX");
        #endif
    }
    
    void switchTo(Fiber& from) const {
        WULFNET_ASSERT(m_handle != nullptr);
        WULFNET_ASSERT(from.m_handle != nullptr);
        
        #if WULFNET_PLATFORM_WINDOWS
            (void)from;
            SwitchToFiber(m_handle);
        #else
            swapcontext(from.m_handle, m_handle);
        #endif
    }
    
    // =========================================================================
    // Query
    // =========================================================================
    
    bool isValid() const { return m_handle != nullptr; }
    FiberHandle getHandle() const { return m_handle; }
    void* getUserData() const { return m_userData; }
    usize getStackSize() const { return m_stackSize; }
    
    static Fiber* getCurrentFiber() {
        #if WULFNET_PLATFORM_WINDOWS
            return static_cast<Fiber*>(GetFiberData());
        #else
            // On POSIX, this needs to be tracked manually by the job system
            return nullptr;
        #endif
    }
    
private:
    FiberHandle m_handle;
    void* m_userData;
    usize m_stackSize;
    bool m_isThreadFiber = false;
    
    #if !WULFNET_PLATFORM_WINDOWS
        u8* m_stack = nullptr;
    #endif
};

// =============================================================================
// Fiber Pool
// =============================================================================

class FiberPool : public NonCopyable {
public:
    explicit FiberPool(usize fiberCount, FiberEntryFunc entryPoint, 
                       usize stackSize = Fiber::DEFAULT_STACK_SIZE)
        : m_fibers(nullptr)
        , m_freeList(nullptr)
        , m_count(fiberCount)
    {
        m_fibers = new Fiber[fiberCount];
        m_freeIndices = new std::atomic<u32>[fiberCount];
        
        for (usize i = 0; i < fiberCount; i++) {
            m_fibers[i].create(entryPoint, reinterpret_cast<void*>(i), stackSize);
            m_freeIndices[i].store(static_cast<u32>(i + 1), std::memory_order_relaxed);
        }
        
        // Last element points to invalid
        m_freeIndices[fiberCount - 1].store(UINT32_MAX, std::memory_order_relaxed);
        m_freeHead.store(0, std::memory_order_release);
    }
    
    ~FiberPool() {
        delete[] m_freeIndices;
        delete[] m_fibers;
    }
    
    Fiber* acquire() {
        u32 head = m_freeHead.load(std::memory_order_acquire);
        
        while (head != UINT32_MAX) {
            u32 next = m_freeIndices[head].load(std::memory_order_relaxed);
            
            if (m_freeHead.compare_exchange_weak(head, next,
                std::memory_order_release, std::memory_order_acquire)) {
                return &m_fibers[head];
            }
        }
        
        return nullptr;  // Pool exhausted
    }
    
    void release(Fiber* fiber) {
        usize index = fiber - m_fibers;
        WULFNET_ASSERT(index < m_count);
        
        u32 head = m_freeHead.load(std::memory_order_acquire);
        
        do {
            m_freeIndices[index].store(head, std::memory_order_relaxed);
        } while (!m_freeHead.compare_exchange_weak(head, static_cast<u32>(index),
            std::memory_order_release, std::memory_order_acquire));
    }
    
    usize getCount() const { return m_count; }
    
private:
    Fiber* m_fibers;
    std::atomic<u32>* m_freeIndices;
    std::atomic<u32> m_freeHead;
    void* m_freeList;
    usize m_count;
};

} // namespace WulfNet
