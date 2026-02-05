// =============================================================================
// WulfNet Engine - System Resource Monitor Implementation
// =============================================================================

#include "SystemMonitor.h"
#include <sstream>
#include <iomanip>
#include <cstring>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <psapi.h>
#endif

namespace WulfNet {

// =============================================================================
// NVML Function Types (for dynamic loading)
// =============================================================================

#ifdef _WIN32
typedef int (*nvmlInit_t)();
typedef int (*nvmlShutdown_t)();
typedef int (*nvmlDeviceGetHandleByIndex_t)(unsigned int, void**);
typedef int (*nvmlDeviceGetUtilizationRates_t)(void*, void*);
typedef int (*nvmlDeviceGetMemoryInfo_t)(void*, void*);
typedef int (*nvmlDeviceGetName_t)(void*, char*, unsigned int);

struct NvmlUtilization {
    unsigned int gpu;
    unsigned int memory;
};

struct NvmlMemory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

static nvmlInit_t s_nvmlInit = nullptr;
static nvmlShutdown_t s_nvmlShutdown = nullptr;
static nvmlDeviceGetHandleByIndex_t s_nvmlDeviceGetHandleByIndex = nullptr;
static nvmlDeviceGetUtilizationRates_t s_nvmlDeviceGetUtilizationRates = nullptr;
static nvmlDeviceGetMemoryInfo_t s_nvmlDeviceGetMemoryInfo = nullptr;
static nvmlDeviceGetName_t s_nvmlDeviceGetName = nullptr;
#endif

// =============================================================================
// Singleton
// =============================================================================

static SystemMonitor* s_instance = nullptr;

SystemMonitor& SystemMonitor::Get() {
    if (!s_instance) {
        s_instance = new SystemMonitor();
    }
    return *s_instance;
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

SystemMonitor::SystemMonitor() = default;

SystemMonitor::~SystemMonitor() {
    Shutdown();
    if (s_instance == this) {
        s_instance = nullptr;
    }
}

// =============================================================================
// Initialization
// =============================================================================

bool SystemMonitor::Initialize() {
    if (m_initialized) return true;

#ifdef _WIN32
    // Initialize CPU time tracking
    FILETIME idleTime, kernelTime, userTime;
    if (GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
        m_lastCPUIdleTime = (static_cast<uint64_t>(idleTime.dwHighDateTime) << 32) | idleTime.dwLowDateTime;
        m_lastCPUKernelTime = (static_cast<uint64_t>(kernelTime.dwHighDateTime) << 32) | kernelTime.dwLowDateTime;
        m_lastCPUUserTime = (static_cast<uint64_t>(userTime.dwHighDateTime) << 32) | userTime.dwLowDateTime;
    }

    // Try to load NVML for GPU monitoring
    m_nvmlLib = LoadLibraryA("nvml.dll");
    if (m_nvmlLib) {
        s_nvmlInit = (nvmlInit_t)GetProcAddress((HMODULE)m_nvmlLib, "nvmlInit_v2");
        s_nvmlShutdown = (nvmlShutdown_t)GetProcAddress((HMODULE)m_nvmlLib, "nvmlShutdown");
        s_nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_t)GetProcAddress((HMODULE)m_nvmlLib, "nvmlDeviceGetHandleByIndex_v2");
        s_nvmlDeviceGetUtilizationRates = (nvmlDeviceGetUtilizationRates_t)GetProcAddress((HMODULE)m_nvmlLib, "nvmlDeviceGetUtilizationRates");
        s_nvmlDeviceGetMemoryInfo = (nvmlDeviceGetMemoryInfo_t)GetProcAddress((HMODULE)m_nvmlLib, "nvmlDeviceGetMemoryInfo");
        s_nvmlDeviceGetName = (nvmlDeviceGetName_t)GetProcAddress((HMODULE)m_nvmlLib, "nvmlDeviceGetName");

        if (s_nvmlInit && s_nvmlShutdown && s_nvmlDeviceGetHandleByIndex &&
            s_nvmlDeviceGetUtilizationRates && s_nvmlDeviceGetMemoryInfo) {

            if (s_nvmlInit() == 0) {  // NVML_SUCCESS
                if (s_nvmlDeviceGetHandleByIndex(0, &m_nvmlDevice) == 0) {
                    m_gpuMonitoringAvailable = true;

                    // Get GPU name
                    if (s_nvmlDeviceGetName) {
                        char name[256] = {0};
                        if (s_nvmlDeviceGetName(m_nvmlDevice, name, sizeof(name)) == 0) {
                            m_stats.gpuName = name;
                        }
                    }
                }
            }
        }
    }
#endif

    m_initialized = true;

    // Do initial update
    Update();

    return true;
}

void SystemMonitor::Shutdown() {
    if (!m_initialized) return;

#ifdef _WIN32
    if (m_gpuMonitoringAvailable && s_nvmlShutdown) {
        s_nvmlShutdown();
    }

    if (m_nvmlLib) {
        FreeLibrary((HMODULE)m_nvmlLib);
        m_nvmlLib = nullptr;
    }
#endif

    m_gpuMonitoringAvailable = false;
    m_initialized = false;
}

// =============================================================================
// Update
// =============================================================================

void SystemMonitor::Update() {
    if (!m_initialized) return;

    UpdateCPUUsage();
    UpdateRAMUsage();
    UpdateGPUUsage();
    UpdateProcessMemory();
}

void SystemMonitor::UpdateCPUUsage() {
#ifdef _WIN32
    FILETIME idleTime, kernelTime, userTime;
    if (GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
        uint64_t idle = (static_cast<uint64_t>(idleTime.dwHighDateTime) << 32) | idleTime.dwLowDateTime;
        uint64_t kernel = (static_cast<uint64_t>(kernelTime.dwHighDateTime) << 32) | kernelTime.dwLowDateTime;
        uint64_t user = (static_cast<uint64_t>(userTime.dwHighDateTime) << 32) | userTime.dwLowDateTime;

        uint64_t idleDiff = idle - m_lastCPUIdleTime;
        uint64_t kernelDiff = kernel - m_lastCPUKernelTime;
        uint64_t userDiff = user - m_lastCPUUserTime;

        uint64_t total = kernelDiff + userDiff;
        if (total > 0) {
            // CPU usage = (total - idle) / total * 100
            m_stats.cpuUsagePercent = static_cast<float>(total - idleDiff) / static_cast<float>(total) * 100.0f;
        }

        m_lastCPUIdleTime = idle;
        m_lastCPUKernelTime = kernel;
        m_lastCPUUserTime = user;
    }
#endif
}

void SystemMonitor::UpdateRAMUsage() {
#ifdef _WIN32
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus)) {
        m_stats.ramTotalBytes = memStatus.ullTotalPhys;
        m_stats.ramUsedBytes = memStatus.ullTotalPhys - memStatus.ullAvailPhys;
        m_stats.ramUsagePercent = static_cast<float>(m_stats.ramUsedBytes) /
                                   static_cast<float>(m_stats.ramTotalBytes) * 100.0f;
    }
#endif
}

void SystemMonitor::UpdateGPUUsage() {
#ifdef _WIN32
    if (!m_gpuMonitoringAvailable || !m_nvmlDevice) {
        m_stats.gpuUsageAvailable = false;
        m_stats.vramUsageAvailable = false;
        return;
    }

    // GPU utilization
    if (s_nvmlDeviceGetUtilizationRates) {
        NvmlUtilization util = {0, 0};
        if (s_nvmlDeviceGetUtilizationRates(m_nvmlDevice, &util) == 0) {
            m_stats.gpuUsagePercent = static_cast<float>(util.gpu);
            m_stats.gpuUsageAvailable = true;
        }
    }

    // VRAM usage
    if (s_nvmlDeviceGetMemoryInfo) {
        NvmlMemory mem = {0, 0, 0};
        if (s_nvmlDeviceGetMemoryInfo(m_nvmlDevice, &mem) == 0) {
            m_stats.vramTotalBytes = mem.total;
            m_stats.vramUsedBytes = mem.used;
            m_stats.vramUsagePercent = static_cast<float>(mem.used) /
                                        static_cast<float>(mem.total) * 100.0f;
            m_stats.vramUsageAvailable = true;
        }
    }
#endif
}

void SystemMonitor::UpdateProcessMemory() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        m_stats.processMemoryBytes = pmc.WorkingSetSize;
        m_stats.processVirtualMemoryBytes = pmc.PrivateUsage;
    }
#endif
}

// =============================================================================
// Utility Functions
// =============================================================================

std::string FormatBytes(uint64_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);

    if (bytes >= 1024ULL * 1024 * 1024 * 1024) {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0 * 1024.0)) << " TB";
    } else if (bytes >= 1024ULL * 1024 * 1024) {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GB";
    } else if (bytes >= 1024ULL * 1024) {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
    } else if (bytes >= 1024ULL) {
        oss << (static_cast<double>(bytes) / 1024.0) << " KB";
    } else {
        oss << bytes << " B";
    }

    return oss.str();
}

std::string FormatPercent(float percent, int decimals) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals) << percent << "%";
    return oss.str();
}

} // namespace WulfNet
