// =============================================================================
// WulfNet Engine - Performance Logger Implementation
// =============================================================================

#include "PerfLogger.h"
#include "SystemMonitor.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

namespace WulfNet {

// =============================================================================
// Singleton
// =============================================================================

PerfLogger& PerfLogger::Get() {
    static PerfLogger instance;
    return instance;
}

PerfLogger::~PerfLogger() {
    Shutdown();
}

// =============================================================================
// Lifecycle
// =============================================================================

void PerfLogger::Initialize(const std::string& csvPath, float updateIntervalMs) {
    if (m_active) Shutdown();

    m_updateIntervalMs = updateIntervalMs;
    m_timeSinceLastPoll = updateIntervalMs; // Force immediate first poll
    m_smoothFps = 0.0f;
    m_snapshot = PerfSnapshot{};
    m_active = true;

    // Initialize SystemMonitor if not already done
    SystemMonitor::Get().Initialize();

    // Open CSV file if requested
    if (!csvPath.empty()) {
        FILE* f = nullptr;
#ifdef _WIN32
        fopen_s(&f, csvPath.c_str(), "w");
#else
        f = fopen(csvPath.c_str(), "w");
#endif
        if (f) {
            m_file = static_cast<void*>(f);
            m_fileLogging = true;
            WriteCSVHeader();
        }
    }
}

void PerfLogger::Shutdown() {
    if (!m_active) return;

    if (m_fileLogging && m_file) {
        FILE* f = static_cast<FILE*>(m_file);
        fflush(f);
        fclose(f);
        m_file = nullptr;
        m_fileLogging = false;
    }

    m_active = false;
}

// =============================================================================
// Frame Timing
// =============================================================================

void PerfLogger::BeginFrame() {
    if (!m_active) return;
    m_frameStart = std::chrono::high_resolution_clock::now();
}

void PerfLogger::EndFrame() {
    if (!m_active) return;

    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - m_frameStart);

    float frameMs = elapsed.count() * 0.001f;
    RecordFrame(frameMs);
}

void PerfLogger::RecordFrame(float wallClockDeltaMs) {
    if (!m_active) return;

    float frameMs = wallClockDeltaMs;
    if (frameMs < 0.001f) frameMs = 0.001f; // Avoid division by zero

    m_snapshot.frameNumber++;
    m_framesSincePoll++;
    m_timeSinceLastPoll += frameMs;

    // Update displayed FPS/frameTime only at the poll interval (every ~500ms).
    // This computes a true average over the window, not a noisy per-frame value.
    if (m_timeSinceLastPoll >= m_updateIntervalMs) {
        m_snapshot.fps = (m_timeSinceLastPoll > 0.0f)
            ? (m_framesSincePoll * 1000.0f / m_timeSinceLastPoll)
            : 0.0f;
        m_snapshot.frameTimeMs = m_timeSinceLastPoll / m_framesSincePoll;
        m_snapshot.avgFps = m_snapshot.fps;
        m_framesSincePoll = 0;
        m_timeSinceLastPoll = 0.0f;
        PollSystemStats();
    }

    // Log to CSV
    if (m_fileLogging) {
        WriteCSVRow();
    }
}

// =============================================================================
// System Stats Polling
// =============================================================================

void PerfLogger::PollSystemStats() {
    auto& mon = SystemMonitor::Get();
    mon.Update();
    const auto& stats = mon.GetStats();

    m_snapshot.cpuPercent    = stats.cpuUsagePercent;
    m_snapshot.processCpu    = stats.processCpuPercent;
    m_snapshot.ramUsedMB     = stats.ramUsedBytes / (1024 * 1024);
    m_snapshot.ramTotalMB    = stats.ramTotalBytes / (1024 * 1024);
    m_snapshot.ramPercent    = stats.ramUsagePercent;
    m_snapshot.processMemMB  = stats.processMemoryBytes / (1024 * 1024);

    m_snapshot.gpuPercent    = stats.gpuUsagePercent;
    m_snapshot.gpuAvailable  = stats.gpuUsageAvailable;
    m_snapshot.vramUsedMB    = stats.vramUsedBytes / (1024 * 1024);
    m_snapshot.vramTotalMB   = stats.vramTotalBytes / (1024 * 1024);
    m_snapshot.vramPercent   = stats.vramUsagePercent;
    m_snapshot.vramAvailable = stats.vramUsageAvailable;
    m_snapshot.gpuName       = stats.gpuName;
}

// =============================================================================
// Overlay Formatting
// =============================================================================

std::string PerfLogger::FormatOverlayText() const {
    char buf[512];

    // Line 1: FPS and frame time
    int pos = snprintf(buf, sizeof(buf),
        "FPS: %.1f  (%.1fms)  avg: %.1f\n",
        m_snapshot.fps, m_snapshot.frameTimeMs, m_snapshot.avgFps);

    // Line 2: CPU and RAM
    pos += snprintf(buf + pos, sizeof(buf) - pos,
        "CPU: %.0f%% (sys: %.0f%%)  RAM: %llu/%llu MB (%.0f%%)\n",
        m_snapshot.processCpu,
        m_snapshot.cpuPercent,
        (unsigned long long)m_snapshot.ramUsedMB,
        (unsigned long long)m_snapshot.ramTotalMB,
        m_snapshot.ramPercent);

    // Line 3: GPU and VRAM (if available)
    if (m_snapshot.gpuAvailable) {
        pos += snprintf(buf + pos, sizeof(buf) - pos,
            "GPU: %.1f%%  VRAM: %llu/%llu MB (%.0f%%)\n",
            m_snapshot.gpuPercent,
            (unsigned long long)m_snapshot.vramUsedMB,
            (unsigned long long)m_snapshot.vramTotalMB,
            m_snapshot.vramPercent);
    } else {
        pos += snprintf(buf + pos, sizeof(buf) - pos,
            "GPU: N/A  VRAM: N/A\n");
    }

    // Line 4: Process memory
    snprintf(buf + pos, sizeof(buf) - pos,
        "Process: %llu MB",
        (unsigned long long)m_snapshot.processMemMB);

    return std::string(buf);
}

// =============================================================================
// CSV Output
// =============================================================================

void PerfLogger::WriteCSVHeader() {
    if (!m_file) return;
    FILE* f = static_cast<FILE*>(m_file);
    fprintf(f, "frame,frame_time_ms,fps,avg_fps,cpu_pct,ram_used_mb,ram_total_mb,"
               "ram_pct,process_mem_mb,gpu_pct,vram_used_mb,vram_total_mb,vram_pct\n");
    fflush(f);
}

void PerfLogger::WriteCSVRow() {
    if (!m_file) return;
    FILE* f = static_cast<FILE*>(m_file);
    fprintf(f, "%llu,%.3f,%.1f,%.1f,%.1f,%llu,%llu,%.1f,%llu,%.1f,%llu,%llu,%.1f\n",
        (unsigned long long)m_snapshot.frameNumber,
        m_snapshot.frameTimeMs,
        m_snapshot.fps,
        m_snapshot.avgFps,
        m_snapshot.cpuPercent,
        (unsigned long long)m_snapshot.ramUsedMB,
        (unsigned long long)m_snapshot.ramTotalMB,
        m_snapshot.ramPercent,
        (unsigned long long)m_snapshot.processMemMB,
        m_snapshot.gpuPercent,
        (unsigned long long)m_snapshot.vramUsedMB,
        (unsigned long long)m_snapshot.vramTotalMB,
        m_snapshot.vramPercent);

    // Flush every 60 frames to balance performance and data safety
    if (m_snapshot.frameNumber % 60 == 0) {
        fflush(f);
    }
}

} // namespace WulfNet
