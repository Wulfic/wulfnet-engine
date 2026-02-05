// =============================================================================
// WulfNet Engine - System Resource Monitor
// =============================================================================
// Provides real-time CPU, RAM, GPU, and VRAM usage statistics.
// =============================================================================

#pragma once

#include <cstdint>
#include <string>

namespace WulfNet {

// =============================================================================
// System Resource Statistics
// =============================================================================

struct SystemStats {
    // CPU Usage (0.0 - 100.0%)
    float cpuUsagePercent = 0.0f;

    // RAM Usage
    uint64_t ramUsedBytes = 0;
    uint64_t ramTotalBytes = 0;
    float ramUsagePercent = 0.0f;

    // GPU Usage (0.0 - 100.0%) - NVIDIA only via NVML
    float gpuUsagePercent = 0.0f;
    bool gpuUsageAvailable = false;

    // GPU Memory (VRAM)
    uint64_t vramUsedBytes = 0;
    uint64_t vramTotalBytes = 0;
    float vramUsagePercent = 0.0f;
    bool vramUsageAvailable = false;

    // GPU Name
    std::string gpuName;

    // Process-specific memory
    uint64_t processMemoryBytes = 0;
    uint64_t processVirtualMemoryBytes = 0;
};

// =============================================================================
// System Monitor Class
// =============================================================================

class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    // Non-copyable
    SystemMonitor(const SystemMonitor&) = delete;
    SystemMonitor& operator=(const SystemMonitor&) = delete;

    /// Initialize the monitor (call once at startup)
    bool Initialize();

    /// Shutdown and release resources
    void Shutdown();

    /// Update statistics (call periodically, e.g., once per second)
    void Update();

    /// Get current statistics
    const SystemStats& GetStats() const { return m_stats; }

    /// Check if GPU monitoring is available
    bool IsGPUMonitoringAvailable() const { return m_gpuMonitoringAvailable; }

    // Singleton access
    static SystemMonitor& Get();

private:
    void UpdateCPUUsage();
    void UpdateRAMUsage();
    void UpdateGPUUsage();
    void UpdateProcessMemory();

    SystemStats m_stats;
    bool m_initialized = false;
    bool m_gpuMonitoringAvailable = false;

    // Platform-specific data
#ifdef _WIN32
    uint64_t m_lastCPUIdleTime = 0;
    uint64_t m_lastCPUKernelTime = 0;
    uint64_t m_lastCPUUserTime = 0;
#endif

    // NVML handles (loaded dynamically)
    void* m_nvmlLib = nullptr;
    void* m_nvmlDevice = nullptr;
};

// =============================================================================
// Utility Functions
// =============================================================================

/// Format bytes to human-readable string (KB, MB, GB)
std::string FormatBytes(uint64_t bytes);

/// Format percentage with fixed precision
std::string FormatPercent(float percent, int decimals = 1);

} // namespace WulfNet
