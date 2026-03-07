// =============================================================================
// WulfNet Engine - Performance Logger
// =============================================================================
// Logs frame timing, FPS, CPU/GPU/Memory statistics to a CSV file and provides
// real-time overlay text for HUD display. Thread-safe for the stats snapshot.
//
// Usage:
//   PerfLogger::Get().Initialize("perf_log.csv");
//   // Each frame:
//   PerfLogger::Get().BeginFrame();
//   // ... do work ...
//   PerfLogger::Get().EndFrame();
//   // On shutdown:
//   PerfLogger::Get().Shutdown();
// =============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <chrono>

namespace WulfNet {

// =============================================================================
// Per-Frame Performance Snapshot
// =============================================================================

struct PerfSnapshot {
    uint64_t frameNumber = 0;

    // Timing
    float frameTimeMs   = 0.0f;   // Wall-clock frame time in milliseconds
    float fps           = 0.0f;   // Instantaneous FPS (1000/frameTimeMs)
    float avgFps        = 0.0f;   // Smoothed FPS (exponential moving average)

    // CPU
    float cpuPercent    = 0.0f;   // System-wide CPU %
    float processCpu    = 0.0f;   // This process CPU % (can exceed 100% on multi-core)

    // RAM
    uint64_t ramUsedMB  = 0;
    uint64_t ramTotalMB = 0;
    float ramPercent    = 0.0f;

    // Process memory
    uint64_t processMemMB = 0;

    // GPU
    float gpuPercent    = 0.0f;
    bool  gpuAvailable  = false;

    // VRAM
    uint64_t vramUsedMB  = 0;
    uint64_t vramTotalMB = 0;
    float vramPercent    = 0.0f;
    bool  vramAvailable  = false;

    // GPU name
    std::string gpuName;
};

// =============================================================================
// Performance Logger
// =============================================================================

class PerfLogger {
public:
    /// Singleton access
    static PerfLogger& Get();

    /// Initialize logging. Pass empty path to disable CSV output.
    /// @param csvPath  Output CSV file path (empty = no file logging)
    /// @param updateIntervalMs  How often to poll SystemMonitor (default 500ms)
    void Initialize(const std::string& csvPath = "",
                    float updateIntervalMs = 500.0f);

    /// Shutdown — flush and close the CSV file
    void Shutdown();

    /// Call at the very start of each frame
    void BeginFrame();

    /// Call at the very end of each frame (after Present)
    void EndFrame();

    /// Record a frame using an externally-measured wall-clock delta (in milliseconds).
    /// Use this instead of BeginFrame/EndFrame when you have the true full-frame time
    /// (e.g. from Application::mClockDeltaTime) that includes rendering and present.
    void RecordFrame(float wallClockDeltaMs);

    /// Get the latest performance snapshot (read-only)
    const PerfSnapshot& GetSnapshot() const { return m_snapshot; }

    /// Format a multi-line overlay string for HUD display
    /// Lines are separated by '\n'. Example:
    ///   FPS: 142.3  (7.0ms)
    ///   CPU: 34.2%  RAM: 6.1/15.9 GB (38%)
    ///   GPU: 61%  VRAM: 2.1/8.0 GB (26%)
    ///   Process: 412 MB
    std::string FormatOverlayText() const;

    /// Check if logging is active
    bool IsActive() const { return m_active; }

private:
    PerfLogger() = default;
    ~PerfLogger();
    PerfLogger(const PerfLogger&) = delete;
    PerfLogger& operator=(const PerfLogger&) = delete;

    void PollSystemStats();
    void WriteCSVHeader();
    void WriteCSVRow();

    // State
    bool  m_active = false;
    bool  m_fileLogging = false;
    void* m_file = nullptr;   // FILE* cast to void* to avoid <cstdio> in header

    // Timing
    std::chrono::high_resolution_clock::time_point m_frameStart;
    float m_smoothFps = 0.0f;
    float m_updateIntervalMs = 500.0f;
    float m_timeSinceLastPoll = 0.0f;
    int   m_framesSincePoll = 0;

    // Snapshot
    PerfSnapshot m_snapshot;
};

} // namespace WulfNet
