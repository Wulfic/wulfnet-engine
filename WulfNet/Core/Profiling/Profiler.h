// =============================================================================
// WulfNet Engine - Profiler.h
// =============================================================================
// Tracy profiler integration with fallback no-op macros when Tracy is disabled.
// =============================================================================

#pragma once

// Check if Tracy is available
#ifdef WULFNET_ENABLE_TRACY
    #include <tracy/Tracy.hpp>

    // Zone macros
    #define WULFNET_ZONE()                   ZoneScoped
    #define WULFNET_ZONE_NAMED(name)         ZoneScopedN(name)
    #define WULFNET_ZONE_COLOR(color)        ZoneScopedC(color)
    #define WULFNET_ZONE_NAMED_COLOR(n, c)   ZoneScopedNC(n, c)

    // Frame markers
    #define WULFNET_FRAME_MARK()             FrameMark
    #define WULFNET_FRAME_MARK_NAMED(name)   FrameMarkNamed(name)

    // Memory tracking
    #define WULFNET_ALLOC(ptr, size)         TracyAlloc(ptr, size)
    #define WULFNET_FREE(ptr)                TracyFree(ptr)

    // Messages
    #define WULFNET_MESSAGE(text, len)       TracyMessage(text, len)
    #define WULFNET_MESSAGE_L(text)          TracyMessageL(text)

    // Plot values
    #define WULFNET_PLOT(name, value)        TracyPlot(name, value)

    // Lock annotations
    #define WULFNET_LOCKABLE(type, name)     TracyLockable(type, name)
    #define WULFNET_SHARED_LOCKABLE(t, name) TracySharedLockable(t, name)

#else
    // No-op fallbacks when Tracy is disabled
    #define WULFNET_ZONE()                   ((void)0)
    #define WULFNET_ZONE_NAMED(name)         ((void)0)
    #define WULFNET_ZONE_COLOR(color)        ((void)0)
    #define WULFNET_ZONE_NAMED_COLOR(n, c)   ((void)0)

    #define WULFNET_FRAME_MARK()             ((void)0)
    #define WULFNET_FRAME_MARK_NAMED(name)   ((void)0)

    #define WULFNET_ALLOC(ptr, size)         ((void)0)
    #define WULFNET_FREE(ptr)                ((void)0)

    #define WULFNET_MESSAGE(text, len)       ((void)0)
    #define WULFNET_MESSAGE_L(text)          ((void)0)

    #define WULFNET_PLOT(name, value)        ((void)0)

    #define WULFNET_LOCKABLE(type, name)     type name
    #define WULFNET_SHARED_LOCKABLE(t, name) t name
#endif

// =============================================================================
// Scoped Timer (always available, logs to our Logger)
// =============================================================================

#include <chrono>
#include <string>

namespace WulfNet {

class ScopedTimer {
public:
    explicit ScopedTimer(const char* name)
        : m_name(name)
        , m_start(std::chrono::high_resolution_clock::now())
    {
    }

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
        LogTime(duration.count());
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    void LogTime(long long microseconds);

    const char* m_name;
    std::chrono::high_resolution_clock::time_point m_start;
};

// Simple timing without logging (just returns duration)
class ManualTimer {
public:
    void Start() {
        m_start = std::chrono::high_resolution_clock::now();
    }

    double ElapsedMilliseconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - m_start).count();
    }

    double ElapsedMicroseconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(now - m_start).count();
    }

    double ElapsedSeconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - m_start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point m_start;
};

} // namespace WulfNet

#define WULFNET_SCOPED_TIMER(name) ::WulfNet::ScopedTimer _timer_##__LINE__(name)
