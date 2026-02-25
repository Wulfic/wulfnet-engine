// =============================================================================
// WulfNet Engine - System Monitor Unit Tests
// =============================================================================
// Validates CPU, RAM, GPU, and VRAM monitoring, singleton access, and utilities.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Core/System/SystemMonitor.h>

using namespace WulfNet;

// =============================================================================
// Initialization Tests
// =============================================================================

void test_SystemMonitor_Singleton() {
    SystemMonitor& mon1 = SystemMonitor::Get();
    SystemMonitor& mon2 = SystemMonitor::Get();
    EXPECT_EQ(&mon1, &mon2);
}

void test_SystemMonitor_Initialize() {
    SystemMonitor& mon = SystemMonitor::Get();
    bool result = mon.Initialize();
    EXPECT_TRUE(result);
    mon.Shutdown();
}

void test_SystemMonitor_DoubleInitialize() {
    SystemMonitor& mon = SystemMonitor::Get();
    mon.Initialize();
    // Should handle gracefully (not crash)
    mon.Initialize();
    mon.Shutdown();
    EXPECT_TRUE(true); // Reached here without crash
}

void test_SystemMonitor_Shutdown_Idempotent() {
    SystemMonitor& mon = SystemMonitor::Get();
    mon.Initialize();
    mon.Shutdown();
    mon.Shutdown(); // Double shutdown should not crash
    EXPECT_TRUE(true);
}

// =============================================================================
// Stats Tests
// =============================================================================

void test_SystemMonitor_InitialStats() {
    SystemMonitor& mon = SystemMonitor::Get();
    mon.Initialize();

    const SystemStats& stats = mon.GetStats();
    // Before first update, CPU usage may be 0
    EXPECT_TRUE(stats.ramTotalBytes > 0); // System should report RAM
    EXPECT_TRUE(stats.ramUsagePercent >= 0.0f);

    mon.Shutdown();
}

void test_SystemMonitor_UpdateCPU() {
    SystemMonitor& mon = SystemMonitor::Get();
    mon.Initialize();

    // Update twice to get a delta for CPU measurement
    mon.Update();
    // Do some work to create CPU load
    volatile int sum = 0;
    for (int i = 0; i < 1000000; i++) sum += i;
    (void)sum;
    mon.Update();

    const SystemStats& stats = mon.GetStats();
    // CPU usage should be in valid range [0, 100]
    EXPECT_TRUE(stats.cpuUsagePercent >= 0.0f);
    EXPECT_TRUE(stats.cpuUsagePercent <= 100.0f);

    mon.Shutdown();
}

void test_SystemMonitor_RAM() {
    SystemMonitor& mon = SystemMonitor::Get();
    mon.Initialize();
    mon.Update();

    const SystemStats& stats = mon.GetStats();
    // System must have some RAM
    EXPECT_TRUE(stats.ramTotalBytes > 0);
    EXPECT_TRUE(stats.ramUsedBytes > 0);
    EXPECT_TRUE(stats.ramUsedBytes <= stats.ramTotalBytes);
    EXPECT_TRUE(stats.ramUsagePercent > 0.0f);
    EXPECT_TRUE(stats.ramUsagePercent <= 100.0f);

    mon.Shutdown();
}

void test_SystemMonitor_ProcessMemory() {
    SystemMonitor& mon = SystemMonitor::Get();
    mon.Initialize();
    mon.Update();

    const SystemStats& stats = mon.GetStats();
    // This process should be using some memory
    EXPECT_TRUE(stats.processMemoryBytes > 0);

    mon.Shutdown();
}

void test_SystemMonitor_GPUAvailability() {
    SystemMonitor& mon = SystemMonitor::Get();
    mon.Initialize();
    mon.Update();

    // GPU monitoring may or may not be available depending on hardware
    bool gpuAvailable = mon.IsGPUMonitoringAvailable();
    const SystemStats& stats = mon.GetStats();

    if (gpuAvailable) {
        // If available, usage should be in valid range
        EXPECT_TRUE(stats.gpuUsagePercent >= 0.0f);
        EXPECT_TRUE(stats.gpuUsagePercent <= 100.0f);
        EXPECT_TRUE(!stats.gpuName.empty());
    }
    // Test passes regardless of GPU availability

    mon.Shutdown();
}

void test_SystemMonitor_VRAM() {
    SystemMonitor& mon = SystemMonitor::Get();
    mon.Initialize();
    mon.Update();

    const SystemStats& stats = mon.GetStats();

    if (stats.vramUsageAvailable) {
        EXPECT_TRUE(stats.vramTotalBytes > 0);
        EXPECT_TRUE(stats.vramUsagePercent >= 0.0f);
        EXPECT_TRUE(stats.vramUsagePercent <= 100.0f);
    }
    // Test passes regardless

    mon.Shutdown();
}

// =============================================================================
// Utility Function Tests
// =============================================================================

void test_FormatBytes_Zero() {
    std::string result = FormatBytes(0);
    EXPECT_TRUE(result.find("0") != std::string::npos);
}

void test_FormatBytes_Bytes() {
    std::string result = FormatBytes(512);
    EXPECT_TRUE(result.find("512") != std::string::npos);
    EXPECT_TRUE(result.find("B") != std::string::npos);
}

void test_FormatBytes_Kilobytes() {
    std::string result = FormatBytes(1024);
    EXPECT_TRUE(result.find("KB") != std::string::npos || result.find("1") != std::string::npos);
}

void test_FormatBytes_Megabytes() {
    std::string result = FormatBytes(1024 * 1024);
    EXPECT_TRUE(result.find("MB") != std::string::npos || result.find("1") != std::string::npos);
}

void test_FormatBytes_Gigabytes() {
    std::string result = FormatBytes(static_cast<uint64_t>(1024) * 1024 * 1024);
    EXPECT_TRUE(result.find("GB") != std::string::npos || result.find("1") != std::string::npos);
}

void test_FormatPercent_Values() {
    std::string result50 = FormatPercent(50.0f);
    EXPECT_TRUE(result50.find("50") != std::string::npos);

    std::string result0 = FormatPercent(0.0f);
    EXPECT_TRUE(result0.find("0") != std::string::npos);

    std::string result100 = FormatPercent(100.0f);
    EXPECT_TRUE(result100.find("100") != std::string::npos);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterSystemMonitorTests() {
    RUN_TEST("SystemMonitor_Singleton", test_SystemMonitor_Singleton);
    RUN_TEST("SystemMonitor_Initialize", test_SystemMonitor_Initialize);
    RUN_TEST("SystemMonitor_DoubleInitialize", test_SystemMonitor_DoubleInitialize);
    RUN_TEST("SystemMonitor_Shutdown_Idempotent", test_SystemMonitor_Shutdown_Idempotent);

    RUN_TEST("SystemMonitor_InitialStats", test_SystemMonitor_InitialStats);
    RUN_TEST("SystemMonitor_UpdateCPU", test_SystemMonitor_UpdateCPU);
    RUN_TEST("SystemMonitor_RAM", test_SystemMonitor_RAM);
    RUN_TEST("SystemMonitor_ProcessMemory", test_SystemMonitor_ProcessMemory);
    RUN_TEST("SystemMonitor_GPUAvailability", test_SystemMonitor_GPUAvailability);
    RUN_TEST("SystemMonitor_VRAM", test_SystemMonitor_VRAM);

    RUN_TEST("FormatBytes_Zero", test_FormatBytes_Zero);
    RUN_TEST("FormatBytes_Bytes", test_FormatBytes_Bytes);
    RUN_TEST("FormatBytes_Kilobytes", test_FormatBytes_Kilobytes);
    RUN_TEST("FormatBytes_Megabytes", test_FormatBytes_Megabytes);
    RUN_TEST("FormatBytes_Gigabytes", test_FormatBytes_Gigabytes);
    RUN_TEST("FormatPercent_Values", test_FormatPercent_Values);
}
