// =============================================================================
// WulfNet Engine - Core Tests (Logger, Profiler)
// =============================================================================
// Tests for Logger singleton, log levels, statistics, and ManualTimer.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>

using namespace WulfNet;

// =============================================================================
// Logger Tests
// =============================================================================

void test_Logger_Singleton() {
    Logger& logger1 = Logger::Get();
    Logger& logger2 = Logger::Get();
    EXPECT_EQ(&logger1, &logger2);
}

void test_Logger_SetMinLevel() {
    Logger& logger = Logger::Get();
    logger.SetMinLevel(LogLevel::Warning);
    EXPECT_EQ(logger.GetMinLevel(), LogLevel::Warning);
    logger.SetMinLevel(LogLevel::Info); // Reset
}

void test_Logger_Statistics() {
    Logger& logger = Logger::Get();
    logger.ResetStatistics();

    size_t initialCount = logger.GetLogCount();
    logger.SetMinLevel(LogLevel::Debug);
    WULFNET_INFO("Test", "Test message");
    EXPECT_TRUE(logger.GetLogCount() > initialCount);
    logger.SetMinLevel(LogLevel::Error); // Reset
}

void test_Logger_ErrorCount() {
    Logger& logger = Logger::Get();
    logger.ResetStatistics();

    WULFNET_ERROR("Test", "Test error");
    EXPECT_EQ(logger.GetErrorCount(), static_cast<size_t>(1));
}

void test_Logger_WarningCount() {
    Logger& logger = Logger::Get();
    logger.ResetStatistics();
    logger.SetMinLevel(LogLevel::Warning);

    WULFNET_WARNING("Test", "Test warning");
    EXPECT_EQ(logger.GetWarningCount(), static_cast<size_t>(1));
    logger.SetMinLevel(LogLevel::Error); // Reset
}

void test_Logger_CallbackSink() {
    bool callbackCalled = false;
    LogLevel capturedLevel = LogLevel::Off;

    auto callback = [&](const LogEntry& entry) {
        callbackCalled = true;
        capturedLevel = entry.level;
    };

    auto sink = std::make_shared<CallbackLogSink>(callback);
    Logger::Get().AddSink(sink);
    Logger::Get().SetMinLevel(LogLevel::Debug);

    WULFNET_INFO("Test", "Callback test");

    EXPECT_TRUE(callbackCalled);
    EXPECT_EQ(capturedLevel, LogLevel::Info);

    Logger::Get().RemoveSink(sink);
    Logger::Get().SetMinLevel(LogLevel::Error);
}

// =============================================================================
// Profiler Tests
// =============================================================================

void test_ManualTimer_ElapsedTime() {
    ManualTimer timer;
    timer.Start();

    // Do some work
    volatile int sum = 0;
    for (int i = 0; i < 100000; i++) {
        sum += i;
    }
    (void)sum;

    double elapsed = timer.ElapsedMicroseconds();
    EXPECT_TRUE(elapsed > 0.0);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterCoreTests() {
    RUN_TEST("Logger_Singleton", test_Logger_Singleton);
    RUN_TEST("Logger_SetMinLevel", test_Logger_SetMinLevel);
    RUN_TEST("Logger_Statistics", test_Logger_Statistics);
    RUN_TEST("Logger_ErrorCount", test_Logger_ErrorCount);
    RUN_TEST("Logger_WarningCount", test_Logger_WarningCount);
    RUN_TEST("Logger_CallbackSink", test_Logger_CallbackSink);

    RUN_TEST("ManualTimer_ElapsedTime", test_ManualTimer_ElapsedTime);
}
