// =============================================================================
// WulfNet Engine - Platform Abstraction
// =============================================================================
// Cross-platform utilities for OS interaction
// =============================================================================

#pragma once

#include "../Types.h"
#include <string>

namespace WulfNet {
namespace Platform {

// =============================================================================
// System Information
// =============================================================================

struct SystemInfo {
    u32 numPhysicalCores = 0;
    u32 numLogicalCores = 0;
    u64 totalSystemMemory = 0;      // Bytes
    u64 availableSystemMemory = 0;  // Bytes
    u32 pageSize = 0;
    u32 cacheLineSize = 64;
    
    // NUMA information
    u32 numaNodeCount = 1;
    
    // CPU features
    bool hasSSE42 = false;
    bool hasAVX = false;
    bool hasAVX2 = false;
    bool hasAVX512 = false;
    bool hasFMA = false;
    
    // Platform name
    const char* osName = nullptr;
    const char* cpuName = nullptr;
};

SystemInfo getSystemInfo();

// =============================================================================
// Memory
// =============================================================================

// Allocate memory with specific alignment
void* alignedAlloc(usize size, usize alignment);
void alignedFree(void* ptr);

// Large page allocation (for physics data pools)
void* allocateLargePages(usize size);
void freeLargePages(void* ptr, usize size);

// Virtual memory
void* virtualReserve(usize size);
bool virtualCommit(void* ptr, usize size);
void virtualDecommit(void* ptr, usize size);
void virtualRelease(void* ptr, usize size);

// =============================================================================
// Threading
// =============================================================================

// Get current thread ID
u64 getCurrentThreadId();

// Set thread name (for debuggers)
void setThreadName(const char* name);

// Set thread priority
enum class ThreadPriority : i32 {
    Lowest = -2,
    Low = -1,
    Normal = 0,
    High = 1,
    Highest = 2,
    TimeCritical = 3
};

void setThreadPriority(ThreadPriority priority);

// Set thread affinity (pin to core)
void setThreadAffinity(u32 coreIndex);
void setThreadAffinityMask(u64 mask);

// =============================================================================
// Timing
// =============================================================================

// High-resolution timer
u64 getPerformanceCounter();
u64 getPerformanceFrequency();
f64 getTimeSeconds();
f64 getTimeMilliseconds();

// Sleep
void sleep(u32 milliseconds);
void sleepMicroseconds(u32 microseconds);

// Yield current thread
void yieldThread();

// =============================================================================
// File System
// =============================================================================

bool fileExists(const char* path);
bool directoryExists(const char* path);
bool createDirectory(const char* path);
bool createDirectories(const char* path);

u64 getFileSize(const char* path);
u64 getFileModificationTime(const char* path);

std::string getExecutablePath();
std::string getExecutableDirectory();
std::string getCurrentWorkingDirectory();
bool setCurrentWorkingDirectory(const char* path);

// =============================================================================
// Dynamic Libraries
// =============================================================================

using LibraryHandle = void*;

LibraryHandle loadLibrary(const char* path);
void unloadLibrary(LibraryHandle handle);
void* getLibrarySymbol(LibraryHandle handle, const char* symbolName);

// =============================================================================
// Debug
// =============================================================================

bool isDebuggerAttached();
void debugBreak();
void outputDebugString(const char* message);

// Stack trace
struct StackFrame {
    void* address;
    char moduleName[256];
    char functionName[256];
    char fileName[256];
    u32 lineNumber;
};

u32 captureStackTrace(StackFrame* frames, u32 maxFrames, u32 skipFrames = 0);
void printStackTrace();

} // namespace Platform
} // namespace WulfNet
