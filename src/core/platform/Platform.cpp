// =============================================================================
// WulfNet Engine - Platform Implementation (Shared)
// =============================================================================

#include "Platform.h"

#if WULFNET_PLATFORM_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
    #include <intrin.h>
#elif WULFNET_PLATFORM_LINUX
    #include <unistd.h>
    #include <sys/sysinfo.h>
    #include <sys/mman.h>
    #include <dlfcn.h>
    #include <pthread.h>
    #include <time.h>
    #include <cpuid.h>
#endif

#include <cstring>
#include <thread>

namespace WulfNet::Platform {

// =============================================================================
// System Information
// =============================================================================

SystemInfo getSystemInfo() {
    SystemInfo info;
    
    info.numLogicalCores = std::thread::hardware_concurrency();
    info.cacheLineSize = WULFNET_CACHE_LINE;
    
    #if WULFNET_PLATFORM_WINDOWS
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        info.pageSize = sysInfo.dwPageSize;
        info.numPhysicalCores = info.numLogicalCores / 2;  // Simplified
        
        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);
        GlobalMemoryStatusEx(&memStatus);
        info.totalSystemMemory = memStatus.ullTotalPhys;
        info.availableSystemMemory = memStatus.ullAvailPhys;
        
        info.osName = "Windows";
        
        // CPU feature detection
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        info.hasSSE42 = (cpuInfo[2] & (1 << 20)) != 0;
        info.hasFMA = (cpuInfo[2] & (1 << 12)) != 0;
        
        __cpuidex(cpuInfo, 7, 0);
        info.hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;
        info.hasAVX512 = (cpuInfo[1] & (1 << 16)) != 0;
        
    #elif WULFNET_PLATFORM_LINUX
        info.pageSize = sysconf(_SC_PAGESIZE);
        info.numPhysicalCores = sysconf(_SC_NPROCESSORS_ONLN);
        
        struct sysinfo si;
        sysinfo(&si);
        info.totalSystemMemory = si.totalram * si.mem_unit;
        info.availableSystemMemory = si.freeram * si.mem_unit;
        
        info.osName = "Linux";
        
        // CPU feature detection
        unsigned int eax, ebx, ecx, edx;
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        info.hasSSE42 = (ecx & (1 << 20)) != 0;
        info.hasFMA = (ecx & (1 << 12)) != 0;
        
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            info.hasAVX2 = (ebx & (1 << 5)) != 0;
            info.hasAVX512 = (ebx & (1 << 16)) != 0;
        }
    #endif
    
    return info;
}

// =============================================================================
// Memory
// =============================================================================

void* alignedAlloc(usize size, usize alignment) {
    #if WULFNET_PLATFORM_WINDOWS
        return _aligned_malloc(size, alignment);
    #else
        void* ptr = nullptr;
        posix_memalign(&ptr, alignment, size);
        return ptr;
    #endif
}

void alignedFree(void* ptr) {
    #if WULFNET_PLATFORM_WINDOWS
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

void* virtualReserve(usize size) {
    #if WULFNET_PLATFORM_WINDOWS
        return VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
    #else
        return mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    #endif
}

bool virtualCommit(void* ptr, usize size) {
    #if WULFNET_PLATFORM_WINDOWS
        return VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) != nullptr;
    #else
        return mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0;
    #endif
}

void virtualDecommit(void* ptr, usize size) {
    #if WULFNET_PLATFORM_WINDOWS
        VirtualFree(ptr, size, MEM_DECOMMIT);
    #else
        madvise(ptr, size, MADV_DONTNEED);
    #endif
}

void virtualRelease(void* ptr, usize size) {
    #if WULFNET_PLATFORM_WINDOWS
        (void)size;
        VirtualFree(ptr, 0, MEM_RELEASE);
    #else
        munmap(ptr, size);
    #endif
}

// =============================================================================
// Threading
// =============================================================================

u64 getCurrentThreadId() {
    #if WULFNET_PLATFORM_WINDOWS
        return static_cast<u64>(GetCurrentThreadId());
    #else
        return static_cast<u64>(pthread_self());
    #endif
}

void setThreadName([[maybe_unused]] const char* name) {
    #if WULFNET_PLATFORM_WINDOWS
        // Windows 10+ SetThreadDescription
        wchar_t wname[256];
        MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, 256);
        SetThreadDescription(GetCurrentThread(), wname);
    #elif WULFNET_PLATFORM_LINUX
        pthread_setname_np(pthread_self(), name);
    #endif
}

void setThreadPriority(ThreadPriority priority) {
    #if WULFNET_PLATFORM_WINDOWS
        int winPriority;
        switch (priority) {
            case ThreadPriority::Lowest:      winPriority = THREAD_PRIORITY_LOWEST; break;
            case ThreadPriority::Low:         winPriority = THREAD_PRIORITY_BELOW_NORMAL; break;
            case ThreadPriority::Normal:      winPriority = THREAD_PRIORITY_NORMAL; break;
            case ThreadPriority::High:        winPriority = THREAD_PRIORITY_ABOVE_NORMAL; break;
            case ThreadPriority::Highest:     winPriority = THREAD_PRIORITY_HIGHEST; break;
            case ThreadPriority::TimeCritical: winPriority = THREAD_PRIORITY_TIME_CRITICAL; break;
            default: winPriority = THREAD_PRIORITY_NORMAL;
        }
        SetThreadPriority(GetCurrentThread(), winPriority);
    #elif WULFNET_PLATFORM_LINUX
        int policy;
        struct sched_param param;
        pthread_getschedparam(pthread_self(), &policy, &param);
        param.sched_priority = static_cast<int>(priority) * 10 + 50;
        pthread_setschedparam(pthread_self(), policy, &param);
    #endif
}

void setThreadAffinity(u32 coreIndex) {
    setThreadAffinityMask(1ULL << coreIndex);
}

void setThreadAffinityMask(u64 mask) {
    #if WULFNET_PLATFORM_WINDOWS
        SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(mask));
    #elif WULFNET_PLATFORM_LINUX
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int i = 0; i < 64; i++) {
            if (mask & (1ULL << i)) {
                CPU_SET(i, &cpuset);
            }
        }
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    #endif
}

// =============================================================================
// Timing
// =============================================================================

u64 getPerformanceCounter() {
    #if WULFNET_PLATFORM_WINDOWS
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        return static_cast<u64>(counter.QuadPart);
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<u64>(ts.tv_sec) * 1000000000ULL + static_cast<u64>(ts.tv_nsec);
    #endif
}

u64 getPerformanceFrequency() {
    #if WULFNET_PLATFORM_WINDOWS
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        return static_cast<u64>(freq.QuadPart);
    #else
        return 1000000000ULL;  // Nanoseconds
    #endif
}

f64 getTimeSeconds() {
    return static_cast<f64>(getPerformanceCounter()) / static_cast<f64>(getPerformanceFrequency());
}

f64 getTimeMilliseconds() {
    return getTimeSeconds() * 1000.0;
}

void sleep(u32 milliseconds) {
    #if WULFNET_PLATFORM_WINDOWS
        Sleep(milliseconds);
    #else
        usleep(milliseconds * 1000);
    #endif
}

void sleepMicroseconds(u32 microseconds) {
    #if WULFNET_PLATFORM_WINDOWS
        // Windows doesn't have microsecond sleep natively
        // Use busy-wait for short sleeps
        if (microseconds < 1000) {
            LARGE_INTEGER start, now, freq;
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            
            LONGLONG target = start.QuadPart + (freq.QuadPart * microseconds / 1000000);
            do {
                QueryPerformanceCounter(&now);
            } while (now.QuadPart < target);
        } else {
            Sleep(microseconds / 1000);
        }
    #else
        usleep(microseconds);
    #endif
}

void yieldThread() {
    std::this_thread::yield();
}

// =============================================================================
// Debug
// =============================================================================

bool isDebuggerAttached() {
    #if WULFNET_PLATFORM_WINDOWS
        return IsDebuggerPresent() != 0;
    #else
        // Linux: check /proc/self/status for TracerPid
        return false;  // Simplified
    #endif
}

void debugBreak() {
    WULFNET_DEBUGBREAK();
}

void outputDebugString([[maybe_unused]] const char* message) {
    #if WULFNET_PLATFORM_WINDOWS
        OutputDebugStringA(message);
    #endif
}

} // namespace WulfNet::Platform
