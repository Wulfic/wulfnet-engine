// =============================================================================
// WulfNet Engine - Logger.h
// =============================================================================
// Comprehensive logging infrastructure with multiple log levels and outputs.
// Integrates with Jolt's trace system when appropriate.
// =============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <memory>
#include <vector>
#include <mutex>
#include <fstream>
#include <chrono>
#include <functional>

namespace WulfNet {

// =============================================================================
// Log Levels
// =============================================================================

enum class LogLevel : uint8_t {
    Trace = 0,    // Detailed tracing for debugging
    Debug = 1,    // Debug information
    Info = 2,     // General information
    Warning = 3,  // Warnings (non-critical issues)
    Error = 4,    // Errors (recoverable)
    Fatal = 5,    // Fatal errors (unrecoverable)
    Off = 6       // Disable all logging
};

// Convert log level to string
constexpr const char* LogLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::Trace:   return "TRACE";
        case LogLevel::Debug:   return "DEBUG";
        case LogLevel::Info:    return "INFO";
        case LogLevel::Warning: return "WARN";
        case LogLevel::Error:   return "ERROR";
        case LogLevel::Fatal:   return "FATAL";
        case LogLevel::Off:     return "OFF";
        default:                return "UNKNOWN";
    }
}

// =============================================================================
// Log Entry
// =============================================================================

struct LogEntry {
    LogLevel level;
    std::string message;
    std::string category;
    std::chrono::system_clock::time_point timestamp;
    const char* file;
    int line;
    const char* function;
};

// =============================================================================
// Log Sink Interface
// =============================================================================

class ILogSink {
public:
    virtual ~ILogSink() = default;
    virtual void Write(const LogEntry& entry) = 0;
    virtual void Flush() = 0;
};

// =============================================================================
// Console Log Sink
// =============================================================================

class ConsoleLogSink : public ILogSink {
public:
    ConsoleLogSink(bool useColors = true);
    void Write(const LogEntry& entry) override;
    void Flush() override;

private:
    bool m_useColors;
};

// =============================================================================
// File Log Sink
// =============================================================================

class FileLogSink : public ILogSink {
public:
    explicit FileLogSink(const std::string& filepath);
    ~FileLogSink();

    void Write(const LogEntry& entry) override;
    void Flush() override;

    bool IsOpen() const { return m_file.is_open(); }

private:
    std::ofstream m_file;
    std::string m_filepath;
};

// =============================================================================
// Callback Log Sink
// =============================================================================

using LogCallback = std::function<void(const LogEntry&)>;

class CallbackLogSink : public ILogSink {
public:
    explicit CallbackLogSink(LogCallback callback);
    void Write(const LogEntry& entry) override;
    void Flush() override {}

private:
    LogCallback m_callback;
};

// =============================================================================
// Logger Class
// =============================================================================

class Logger {
public:
    // Singleton access
    static Logger& Get();

    // Configuration
    void SetMinLevel(LogLevel level) { m_minLevel = level; }
    LogLevel GetMinLevel() const { return m_minLevel; }

    // Sink management
    void AddSink(std::shared_ptr<ILogSink> sink);
    void RemoveSink(std::shared_ptr<ILogSink> sink);
    void ClearSinks();

    // Logging methods
    void Log(LogLevel level,
             std::string_view category,
             std::string_view message,
             const char* file = __builtin_FILE(),
             int line = __builtin_LINE(),
             const char* function = __builtin_FUNCTION());

    void Flush();

    // Convenience methods (use macros below instead for source location)
    void Trace(std::string_view category, std::string_view message);
    void Debug(std::string_view category, std::string_view message);
    void Info(std::string_view category, std::string_view message);
    void Warning(std::string_view category, std::string_view message);
    void Error(std::string_view category, std::string_view message);
    void Fatal(std::string_view category, std::string_view message);

    // Statistics
    size_t GetLogCount() const { return m_logCount; }
    size_t GetErrorCount() const { return m_errorCount; }
    size_t GetWarningCount() const { return m_warningCount; }
    void ResetStatistics();

private:
    Logger();
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    LogLevel m_minLevel = LogLevel::Info;
    std::vector<std::shared_ptr<ILogSink>> m_sinks;
    std::mutex m_mutex;

    // Statistics
    size_t m_logCount = 0;
    size_t m_errorCount = 0;
    size_t m_warningCount = 0;
};

// =============================================================================
// Logging Macros
// =============================================================================

#define WULFNET_LOG(level, category, message) \
    ::WulfNet::Logger::Get().Log(level, category, message, __FILE__, __LINE__, __func__)

#define WULFNET_TRACE(category, message) \
    WULFNET_LOG(::WulfNet::LogLevel::Trace, category, message)

#define WULFNET_DEBUG(category, message) \
    WULFNET_LOG(::WulfNet::LogLevel::Debug, category, message)

#define WULFNET_INFO(category, message) \
    WULFNET_LOG(::WulfNet::LogLevel::Info, category, message)

#define WULFNET_WARNING(category, message) \
    WULFNET_LOG(::WulfNet::LogLevel::Warning, category, message)

#define WULFNET_ERROR(category, message) \
    WULFNET_LOG(::WulfNet::LogLevel::Error, category, message)

#define WULFNET_FATAL(category, message) \
    WULFNET_LOG(::WulfNet::LogLevel::Fatal, category, message)

// =============================================================================
// Conditional Logging Macros (compile out in release builds)
// =============================================================================

#ifdef NDEBUG
    #define WULFNET_TRACE_DEBUG(category, message) ((void)0)
    #define WULFNET_DEBUG_DEBUG(category, message) ((void)0)
#else
    #define WULFNET_TRACE_DEBUG(category, message) WULFNET_TRACE(category, message)
    #define WULFNET_DEBUG_DEBUG(category, message) WULFNET_DEBUG(category, message)
#endif

} // namespace WulfNet
