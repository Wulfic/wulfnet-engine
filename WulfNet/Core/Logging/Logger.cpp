// =============================================================================
// WulfNet Engine - Logger.cpp
// =============================================================================

#include "Logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace WulfNet {

// =============================================================================
// Console Log Sink Implementation
// =============================================================================

ConsoleLogSink::ConsoleLogSink(bool useColors)
    : m_useColors(useColors)
{
#ifdef _WIN32
    // Enable ANSI colors on Windows
    if (m_useColors) {
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (GetConsoleMode(hOut, &mode)) {
            SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
#endif
}

void ConsoleLogSink::Write(const LogEntry& entry) {
    // Format timestamp
    auto time = std::chrono::system_clock::to_time_t(entry.timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        entry.timestamp.time_since_epoch()) % 1000;

    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%H:%M:%S") << "."
        << std::setfill('0') << std::setw(3) << ms.count();

    // Color codes
    const char* colorStart = "";
    const char* colorEnd = "";

    if (m_useColors) {
        colorEnd = "\033[0m";
        switch (entry.level) {
            case LogLevel::Trace:   colorStart = "\033[90m"; break;  // Gray
            case LogLevel::Debug:   colorStart = "\033[36m"; break;  // Cyan
            case LogLevel::Info:    colorStart = "\033[32m"; break;  // Green
            case LogLevel::Warning: colorStart = "\033[33m"; break;  // Yellow
            case LogLevel::Error:   colorStart = "\033[31m"; break;  // Red
            case LogLevel::Fatal:   colorStart = "\033[35;1m"; break; // Magenta Bold
            default: colorEnd = ""; break;
        }
    }

    // Output format: [TIME] [LEVEL] [CATEGORY] Message
    std::ostream& out = (entry.level >= LogLevel::Error) ? std::cerr : std::cout;

    out << colorStart
        << "[" << oss.str() << "] "
        << "[" << std::setw(5) << LogLevelToString(entry.level) << "] "
        << "[" << entry.category << "] "
        << entry.message
        << colorEnd;

    // Add source location for errors and above
    if (entry.level >= LogLevel::Error && entry.file != nullptr) {
        out << " (" << entry.file << ":" << entry.line << ")";
    }

    out << "\n";
}

void ConsoleLogSink::Flush() {
    std::cout.flush();
    std::cerr.flush();
}

// =============================================================================
// File Log Sink Implementation
// =============================================================================

FileLogSink::FileLogSink(const std::string& filepath)
    : m_filepath(filepath)
{
    m_file.open(filepath, std::ios::out | std::ios::app);
    if (!m_file.is_open()) {
        std::cerr << "[WulfNet Logger] Failed to open log file: " << filepath << "\n";
    }
}

FileLogSink::~FileLogSink() {
    if (m_file.is_open()) {
        m_file.flush();
        m_file.close();
    }
}

void FileLogSink::Write(const LogEntry& entry) {
    if (!m_file.is_open()) return;

    // Format timestamp
    auto time = std::chrono::system_clock::to_time_t(entry.timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        entry.timestamp.time_since_epoch()) % 1000;

    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif

    m_file << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << "."
           << std::setfill('0') << std::setw(3) << ms.count() << " "
           << "[" << std::setw(5) << LogLevelToString(entry.level) << "] "
           << "[" << entry.category << "] "
           << entry.message;

    if (entry.file != nullptr) {
        m_file << " (" << entry.file << ":" << entry.line
               << " in " << entry.function << ")";
    }

    m_file << "\n";
}

void FileLogSink::Flush() {
    if (m_file.is_open()) {
        m_file.flush();
    }
}

// =============================================================================
// Callback Log Sink Implementation
// =============================================================================

CallbackLogSink::CallbackLogSink(LogCallback callback)
    : m_callback(std::move(callback))
{
}

void CallbackLogSink::Write(const LogEntry& entry) {
    if (m_callback) {
        m_callback(entry);
    }
}

// =============================================================================
// Logger Implementation
// =============================================================================

Logger& Logger::Get() {
    static Logger instance;
    return instance;
}

Logger::Logger() {
    // Add default console sink
    AddSink(std::make_shared<ConsoleLogSink>(true));
}

void Logger::AddSink(std::shared_ptr<ILogSink> sink) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sinks.push_back(std::move(sink));
}

void Logger::RemoveSink(std::shared_ptr<ILogSink> sink) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sinks.erase(
        std::remove(m_sinks.begin(), m_sinks.end(), sink),
        m_sinks.end()
    );
}

void Logger::ClearSinks() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sinks.clear();
}

void Logger::Log(LogLevel level,
                 std::string_view category,
                 std::string_view message,
                 const char* file,
                 int line,
                 const char* function) {
    if (level < m_minLevel || level == LogLevel::Off) {
        return;
    }

    LogEntry entry;
    entry.level = level;
    entry.category = std::string(category);
    entry.message = std::string(message);
    entry.timestamp = std::chrono::system_clock::now();
    entry.file = file;
    entry.line = line;
    entry.function = function;

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_logCount++;
        if (level >= LogLevel::Error) m_errorCount++;
        if (level == LogLevel::Warning) m_warningCount++;

        for (auto& sink : m_sinks) {
            sink->Write(entry);
        }
    }
}

void Logger::Flush() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& sink : m_sinks) {
        sink->Flush();
    }
}

void Logger::Trace(std::string_view category, std::string_view message) {
    Log(LogLevel::Trace, category, message);
}

void Logger::Debug(std::string_view category, std::string_view message) {
    Log(LogLevel::Debug, category, message);
}

void Logger::Info(std::string_view category, std::string_view message) {
    Log(LogLevel::Info, category, message);
}

void Logger::Warning(std::string_view category, std::string_view message) {
    Log(LogLevel::Warning, category, message);
}

void Logger::Error(std::string_view category, std::string_view message) {
    Log(LogLevel::Error, category, message);
}

void Logger::Fatal(std::string_view category, std::string_view message) {
    Log(LogLevel::Fatal, category, message);
}

void Logger::ResetStatistics() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_logCount = 0;
    m_errorCount = 0;
    m_warningCount = 0;
}

} // namespace WulfNet
