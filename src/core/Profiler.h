// =============================================================================
// WulfNet Engine - Profiler Integration
// =============================================================================
// Tracy profiler integration and custom profiling utilities
// =============================================================================

#pragma once

#include "Types.h"
#include "Log.h"
#include "math/MathUtils.h"
#include <chrono>
#include <cmath>

// =============================================================================
// Tracy Profiler Integration
// =============================================================================

#if WULFNET_ENABLE_PROFILING && __has_include(<tracy/Tracy.hpp>)
    #include <tracy/Tracy.hpp>
    #define WULFNET_PROFILER_ENABLED 1
#else
    #define WULFNET_PROFILER_ENABLED 0
#endif

// =============================================================================
// Profiling Macros
// =============================================================================

#if WULFNET_PROFILER_ENABLED

    // Frame markers
    #define WULFNET_PROFILE_FRAME()          FrameMark
    #define WULFNET_PROFILE_FRAME_NAMED(name) FrameMarkNamed(name)
    
    // Function/scope profiling
    #define WULFNET_PROFILE_SCOPE()          ZoneScoped
    #define WULFNET_PROFILE_SCOPE_NAMED(name) ZoneScopedN(name)
    #define WULFNET_PROFILE_SCOPE_COLOR(color) ZoneScopedC(color)
    
    // Named zones with dynamic names
    #define WULFNET_PROFILE_ZONE_BEGIN(name) \
        static constexpr tracy::SourceLocationData TracyConcat(__tracy_source_location,__LINE__) { name, __FUNCTION__, __FILE__, (uint32_t)__LINE__, 0 }; \
        tracy::ScopedZone ___tracy_scoped_zone(&TracyConcat(__tracy_source_location,__LINE__))
    #define WULFNET_PROFILE_ZONE_END()
    
    // Memory tracking
    #define WULFNET_PROFILE_ALLOC(ptr, size)   TracyAlloc(ptr, size)
    #define WULFNET_PROFILE_FREE(ptr)          TracyFree(ptr)
    #define WULFNET_PROFILE_ALLOC_NAMED(ptr, size, name) TracyAllocN(ptr, size, name)
    #define WULFNET_PROFILE_FREE_NAMED(ptr, name)        TracyFreeN(ptr, name)
    
    // Value plotting
    #define WULFNET_PROFILE_PLOT(name, value)  TracyPlot(name, value)
    
    // Message logging
    #define WULFNET_PROFILE_MESSAGE(msg)       TracyMessage(msg, strlen(msg))
    #define WULFNET_PROFILE_MESSAGE_COLOR(msg, color) TracyMessageC(msg, strlen(msg), color)
    
    // Lock profiling
    #define WULFNET_PROFILE_LOCKABLE(type, name) TracyLockable(type, name)
    #define WULFNET_PROFILE_SHARED_LOCKABLE(type, name) TracySharedLockable(type, name)
    
    // GPU zones (requires integration with rendering backend)
    #define WULFNET_PROFILE_GPU_ZONE(name)
    #define WULFNET_PROFILE_GPU_COLLECT()

#else

    // Stub macros when profiling is disabled
    #define WULFNET_PROFILE_FRAME()
    #define WULFNET_PROFILE_FRAME_NAMED(name)
    #define WULFNET_PROFILE_SCOPE()
    #define WULFNET_PROFILE_SCOPE_NAMED(name)
    #define WULFNET_PROFILE_SCOPE_COLOR(color)
    #define WULFNET_PROFILE_ZONE_BEGIN(name)
    #define WULFNET_PROFILE_ZONE_END()
    #define WULFNET_PROFILE_ALLOC(ptr, size)
    #define WULFNET_PROFILE_FREE(ptr)
    #define WULFNET_PROFILE_ALLOC_NAMED(ptr, size, name)
    #define WULFNET_PROFILE_FREE_NAMED(ptr, name)
    #define WULFNET_PROFILE_PLOT(name, value)
    #define WULFNET_PROFILE_MESSAGE(msg)
    #define WULFNET_PROFILE_MESSAGE_COLOR(msg, color)
    #define WULFNET_PROFILE_LOCKABLE(type, name) type name
    #define WULFNET_PROFILE_SHARED_LOCKABLE(type, name) type name
    #define WULFNET_PROFILE_GPU_ZONE(name)
    #define WULFNET_PROFILE_GPU_COLLECT()

#endif

// =============================================================================
// Predefined Colors (for Tracy zones)
// =============================================================================

namespace WulfNet::ProfileColors {
    constexpr u32 Physics     = 0xFF5555;  // Red
    constexpr u32 Rendering   = 0x55FF55;  // Green
    constexpr u32 Audio       = 0x5555FF;  // Blue
    constexpr u32 Animation   = 0xFFFF55;  // Yellow
    constexpr u32 AI          = 0xFF55FF;  // Magenta
    constexpr u32 Network     = 0x55FFFF;  // Cyan
    constexpr u32 IO          = 0xFFAA55;  // Orange
    constexpr u32 Memory      = 0xAA55FF;  // Purple
    constexpr u32 JobSystem   = 0x55AAFF;  // Light Blue
    constexpr u32 Scene       = 0xAAFF55;  // Lime
}

// =============================================================================
// Scoped Timer (for custom timing without Tracy)
// =============================================================================

namespace WulfNet {

class ScopedTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    
    explicit ScopedTimer(const char* name, f64* outMs = nullptr)
        : m_name(name)
        , m_start(Clock::now())
        , m_outMs(outMs)
    {}
    
    ~ScopedTimer() {
        auto end = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
        f64 ms = duration.count() / 1000.0;
        
        if (m_outMs) {
            *m_outMs = ms;
        }
        
        WULFNET_LOG_DEBUG("Timer", "%s: %.3f ms", m_name, ms);
    }
    
private:
    const char* m_name;
    TimePoint m_start;
    f64* m_outMs;
};

// =============================================================================
// Statistics Accumulator
// =============================================================================

class StatisticsAccumulator {
public:
    void addSample(f64 value) {
        m_count++;
        m_sum += value;
        m_sumSquares += value * value;
        m_min = Math::min(m_min, value);
        m_max = Math::max(m_max, value);
    }
    
    void reset() {
        m_count = 0;
        m_sum = 0;
        m_sumSquares = 0;
        m_min = Math::LARGE_NUM;
        m_max = -Math::LARGE_NUM;
    }
    
    u64 count() const { return m_count; }
    f64 sum() const { return m_sum; }
    f64 mean() const { return m_count > 0 ? m_sum / m_count : 0; }
    f64 min() const { return m_min; }
    f64 max() const { return m_max; }
    
    f64 variance() const {
        if (m_count < 2) return 0;
        f64 m = mean();
        return (m_sumSquares - m_count * m * m) / (m_count - 1);
    }
    
    f64 stdDev() const {
        return std::sqrt(variance());
    }
    
private:
    u64 m_count = 0;
    f64 m_sum = 0;
    f64 m_sumSquares = 0;
    f64 m_min = Math::LARGE_NUM;
    f64 m_max = -Math::LARGE_NUM;
};

} // namespace WulfNet

// =============================================================================
// Convenience Macro for Timed Scope
// =============================================================================

#define WULFNET_TIMED_SCOPE(name) \
    ::WulfNet::ScopedTimer WULFNET_CONCAT(__timer_, __LINE__)(name)
