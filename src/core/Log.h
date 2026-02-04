// =============================================================================
// WulfNet Engine - Logging System
// =============================================================================
// Thread-safe logging with severity levels and multiple outputs
// =============================================================================

#pragma once

#include "Types.h"
#include <cstdio>
#include <cstdarg>
#include <mutex>
#include <chrono>

namespace WulfNet {

// =============================================================================
// Log Severity Levels
// =============================================================================

enum class LogLevel : u8 {
    Trace = 0,   // Extremely verbose, for debugging only
    Debug,       // Debug information
    Info,        // General information
    Warn,        // Warnings
    Error,       // Errors that don't stop execution
    Fatal,       // Fatal errors that require shutdown
    Off          // Disable logging
};

constexpr const char* logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO ";
        case LogLevel::Warn:  return "WARN ";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Fatal: return "FATAL";
        default:              return "?????";
    }
}

constexpr const char* logLevelColor(LogLevel level) {
    switch (level) {
        case LogLevel::Trace: return "\033[90m";    // Gray
        case LogLevel::Debug: return "\033[36m";    // Cyan
        case LogLevel::Info:  return "\033[32m";    // Green
        case LogLevel::Warn:  return "\033[33m";    // Yellow
        case LogLevel::Error: return "\033[31m";    // Red
        case LogLevel::Fatal: return "\033[35;1m";  // Bright Magenta
        default:              return "\033[0m";     // Reset
    }
}

// =============================================================================
// Logger Class
// =============================================================================

class Logger : public NonCopyable {
public:
    static Logger& instance() {
        static Logger s_instance;
        return s_instance;
    }
    
    void setLevel(LogLevel level) { m_minLevel = level; }
    LogLevel getLevel() const { return m_minLevel; }
    
    void enableColors(bool enable) { m_useColors = enable; }
    void enableTimestamps(bool enable) { m_showTimestamps = enable; }
    
    void setOutput(FILE* file) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_output = file;
    }
    
    void log(LogLevel level, const char* category, const char* file, int line,
             const char* format, ...) {
        if (level < m_minLevel) return;
        
        va_list args;
        va_start(args, format);
        logImpl(level, category, file, line, format, args);
        va_end(args);
    }
    
private:
    Logger() 
        : m_output(stderr)
        , m_minLevel(LogLevel::Info)
        , m_useColors(true)
        , m_showTimestamps(true) 
    {
        m_startTime = std::chrono::steady_clock::now();
        
        #if WULFNET_PLATFORM_WINDOWS
        // Enable ANSI colors on Windows 10+
        enableWindowsAnsiColors();
        #endif
    }
    
    void logImpl(LogLevel level, const char* category, const char* file, int line,
                 const char* format, va_list args) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // Timestamp
        if (m_showTimestamps) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - m_startTime
            ).count();
            std::fprintf(m_output, "[%8lld.%03lld] ", 
                elapsed / 1000, elapsed % 1000);
        }
        
        // Level with color
        if (m_useColors) {
            std::fprintf(m_output, "%s[%s]\033[0m ", 
                logLevelColor(level), logLevelToString(level));
        } else {
            std::fprintf(m_output, "[%s] ", logLevelToString(level));
        }
        
        // Category
        if (category && category[0] != '\0') {
            std::fprintf(m_output, "[%s] ", category);
        }
        
        // Message
        std::vfprintf(m_output, format, args);
        
        // File and line for warnings and above
        if (level >= LogLevel::Warn && file) {
            // Extract just the filename
            const char* filename = file;
            for (const char* p = file; *p; ++p) {
                if (*p == '/' || *p == '\\') filename = p + 1;
            }
            std::fprintf(m_output, " (%s:%d)", filename, line);
        }
        
        std::fprintf(m_output, "\n");
        std::fflush(m_output);
    }
    
    #if WULFNET_PLATFORM_WINDOWS
    void enableWindowsAnsiColors();
    #endif
    
    FILE* m_output;
    std::mutex m_mutex;
    LogLevel m_minLevel;
    bool m_useColors;
    bool m_showTimestamps;
    std::chrono::steady_clock::time_point m_startTime;
};

} // namespace WulfNet

// =============================================================================
// Logging Macros
// =============================================================================

#define WULFNET_LOG(level, category, ...) \
    ::WulfNet::Logger::instance().log(level, category, __FILE__, __LINE__, __VA_ARGS__)

#define WULFNET_LOG_TRACE(category, ...) WULFNET_LOG(::WulfNet::LogLevel::Trace, category, __VA_ARGS__)
#define WULFNET_LOG_DEBUG(category, ...) WULFNET_LOG(::WulfNet::LogLevel::Debug, category, __VA_ARGS__)
#define WULFNET_LOG_INFO(category, ...)  WULFNET_LOG(::WulfNet::LogLevel::Info, category, __VA_ARGS__)
#define WULFNET_LOG_WARN(category, ...)  WULFNET_LOG(::WulfNet::LogLevel::Warn, category, __VA_ARGS__)
#define WULFNET_LOG_ERROR(category, ...) WULFNET_LOG(::WulfNet::LogLevel::Error, category, __VA_ARGS__)
#define WULFNET_LOG_FATAL(category, ...) WULFNET_LOG(::WulfNet::LogLevel::Fatal, category, __VA_ARGS__)

// Convenience macros without category
#define LOG_TRACE(...) WULFNET_LOG_TRACE("", __VA_ARGS__)
#define LOG_DEBUG(...) WULFNET_LOG_DEBUG("", __VA_ARGS__)
#define LOG_INFO(...)  WULFNET_LOG_INFO("", __VA_ARGS__)
#define LOG_WARN(...)  WULFNET_LOG_WARN("", __VA_ARGS__)
#define LOG_ERROR(...) WULFNET_LOG_ERROR("", __VA_ARGS__)
#define LOG_FATAL(...) WULFNET_LOG_FATAL("", __VA_ARGS__)
