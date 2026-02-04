// =============================================================================
// WulfNet Engine - Job System
// =============================================================================
// Massively parallel job scheduler with work stealing and fiber support
// Scales to 128+ threads with minimal contention
// =============================================================================

#pragma once

#include "../Types.h"
#include "../Assert.h"
#include "../Log.h"
#include "../memory/Memory.h"
#include "../math/MathUtils.h"
#include "Job.h"
#include "WorkStealingQueue.h"
#include "Fiber.h"
#include <atomic>
#include <thread>
#include <vector>
#include <random>

namespace WulfNet {

// =============================================================================
// Job System Configuration
// =============================================================================

struct JobSystemConfig {
    u32 numWorkerThreads = 0;          // 0 = auto-detect (hardware threads - 1)
    u32 maxJobs = 4096;                 // Maximum pending jobs
    u32 fiberPoolSize = 256;            // Number of fibers for suspension
    usize fiberStackSize = 64 * 1024;   // 64KB per fiber
    bool enableFibers = true;           // Enable fiber-based waiting
    bool setThreadAffinity = true;      // Pin threads to cores
};

// =============================================================================
// Worker Thread State
// =============================================================================

struct WULFNET_CACHE_ALIGNED WorkerThread {
    std::thread thread;
    WorkStealingQueue queue;
    std::atomic<bool> isRunning{false};
    u32 threadIndex = 0;
    
    // Random number generator for work stealing target selection
    std::minstd_rand rng;
    
    // Statistics
    std::atomic<u64> jobsExecuted{0};
    std::atomic<u64> jobsStolen{0};
    std::atomic<u64> stealAttempts{0};
    
    // Default constructor
    WorkerThread() = default;
    
    // Non-copyable (std::thread is non-copyable)
    WorkerThread(const WorkerThread&) = delete;
    WorkerThread& operator=(const WorkerThread&) = delete;
    
    // Move constructor
    WorkerThread(WorkerThread&& other) noexcept
        : thread(std::move(other.thread))
        , queue(std::move(other.queue))
        , isRunning(other.isRunning.load(std::memory_order_relaxed))
        , threadIndex(other.threadIndex)
        , rng(std::move(other.rng))
        , jobsExecuted(other.jobsExecuted.load(std::memory_order_relaxed))
        , jobsStolen(other.jobsStolen.load(std::memory_order_relaxed))
        , stealAttempts(other.stealAttempts.load(std::memory_order_relaxed))
    {}
    
    // Move assignment
    WorkerThread& operator=(WorkerThread&& other) noexcept {
        if (this != &other) {
            thread = std::move(other.thread);
            queue = std::move(other.queue);
            isRunning.store(other.isRunning.load(std::memory_order_relaxed), std::memory_order_relaxed);
            threadIndex = other.threadIndex;
            rng = std::move(other.rng);
            jobsExecuted.store(other.jobsExecuted.load(std::memory_order_relaxed), std::memory_order_relaxed);
            jobsStolen.store(other.jobsStolen.load(std::memory_order_relaxed), std::memory_order_relaxed);
            stealAttempts.store(other.stealAttempts.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
};

// =============================================================================
// Job System
// =============================================================================

class JobSystem : public NonMoveable {
public:
    // =========================================================================
    // Singleton Access
    // =========================================================================
    
    static JobSystem& instance() {
        static JobSystem s_instance;
        return s_instance;
    }
    
    // =========================================================================
    // Initialization / Shutdown
    // =========================================================================
    
    bool initialize(const JobSystemConfig& config = JobSystemConfig{}) {
        if (m_initialized) {
            WULFNET_LOG_WARN("JobSystem", "Already initialized");
            return true;
        }
        
        m_config = config;
        
        // Determine worker count
        if (m_config.numWorkerThreads == 0) {
            m_config.numWorkerThreads = std::thread::hardware_concurrency();
            if (m_config.numWorkerThreads > 1) {
                m_config.numWorkerThreads--;  // Leave one for main thread
            }
        }
        
        WULFNET_LOG_INFO("JobSystem", "Initializing with %u worker threads", 
            m_config.numWorkerThreads);
        
        // Allocate job pool
        m_jobPool = static_cast<Job*>(
            Memory::systemAlloc(sizeof(Job) * m_config.maxJobs, alignof(Job)));
        if (!m_jobPool) {
            WULFNET_LOG_ERROR("JobSystem", "Failed to allocate job pool");
            return false;
        }
        
        // Initialize job free list
        for (u32 i = 0; i < m_config.maxJobs; i++) {
            new (&m_jobPool[i]) Job();
        }
        m_nextJobIndex.store(0, std::memory_order_relaxed);
        
        // Create worker threads
        m_workers.resize(m_config.numWorkerThreads);
        m_isRunning.store(true, std::memory_order_release);
        
        for (u32 i = 0; i < m_config.numWorkerThreads; i++) {
            m_workers[i].threadIndex = i;
            m_workers[i].rng.seed(i + 1);
            m_workers[i].isRunning.store(true, std::memory_order_relaxed);
            m_workers[i].thread = std::thread(&JobSystem::workerThreadMain, this, i);
            
            // Set thread affinity if requested
            if (m_config.setThreadAffinity) {
                setThreadAffinity(m_workers[i].thread, i);
            }
        }
        
        // Convert main thread to fiber if enabled
        if (m_config.enableFibers) {
            m_mainThreadFiber = Fiber::convertThreadToFiber(nullptr);
        }
        
        m_initialized = true;
        WULFNET_LOG_INFO("JobSystem", "Initialized successfully");
        return true;
    }
    
    void shutdown() {
        if (!m_initialized) return;
        
        WULFNET_LOG_INFO("JobSystem", "Shutting down...");
        
        // Signal workers to stop
        m_isRunning.store(false, std::memory_order_release);
        
        // Wake up sleeping workers
        for (auto& worker : m_workers) {
            worker.isRunning.store(false, std::memory_order_release);
        }
        
        // Join all worker threads
        for (auto& worker : m_workers) {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        }
        
        // Log statistics
        u64 totalExecuted = 0;
        u64 totalStolen = 0;
        for (const auto& worker : m_workers) {
            totalExecuted += worker.jobsExecuted.load();
            totalStolen += worker.jobsStolen.load();
        }
        WULFNET_LOG_INFO("JobSystem", "Total jobs executed: %llu, stolen: %llu",
            totalExecuted, totalStolen);
        
        // Clean up
        m_workers.clear();
        
        if (m_config.enableFibers && m_mainThreadFiber.isValid()) {
            Fiber::convertFiberToThread();
        }
        
        Memory::systemFree(m_jobPool);
        m_jobPool = nullptr;
        
        m_initialized = false;
    }
    
    // =========================================================================
    // Job Creation and Submission
    // =========================================================================
    
    Job* allocateJob() {
        u32 index = m_nextJobIndex.fetch_add(1, std::memory_order_relaxed) % m_config.maxJobs;
        return &m_jobPool[index];
    }
    
    JobHandle submit(JobFunction function, void* userData = nullptr, 
                     JobPriority priority = JobPriority::Normal,
                     JobHandle dependency = JobHandle::invalid()) {
        Job* job = allocateJob();
        job->function = function;
        job->priority = priority;
        job->parent = dependency;
        job->unfinishedJobs.store(1, std::memory_order_relaxed);
        
        if (userData) {
            job->setPayload(userData, sizeof(void*));
        }
        
        // Push to current thread's queue or main queue
        pushJob(job);
        
        return makeHandle(job);
    }
    
    template<typename T>
    JobHandle submit(JobFunction function, const T& data,
                     JobPriority priority = JobPriority::Normal) {
        static_assert(sizeof(T) <= Job::PAYLOAD_SIZE, "Payload too large");
        
        Job* job = allocateJob();
        job->function = function;
        job->priority = priority;
        job->unfinishedJobs.store(1, std::memory_order_relaxed);
        job->setPayload(data);
        
        pushJob(job);
        
        return makeHandle(job);
    }
    
    // =========================================================================
    // Parallel For
    // =========================================================================
    
    void parallelFor(u32 itemCount, ParallelForFunction function, void* userData = nullptr,
                     u32 minBatchSize = 1, JobPriority priority = JobPriority::Normal) {
        if (itemCount == 0) return;
        
        // Calculate batch size
        u32 workerCount = m_config.numWorkerThreads + 1;  // +1 for main thread
        u32 batchSize = Math::max(minBatchSize, (itemCount + workerCount * 4 - 1) / (workerCount * 4));
        u32 numBatches = (itemCount + batchSize - 1) / batchSize;
        
        // Counter to track completion
        std::atomic<u32> batchesRemaining{numBatches};
        
        struct BatchData {
            ParallelForFunction function;
            void* userData;
            u32 startIndex;
            u32 endIndex;
            std::atomic<u32>* counter;
        };
        
        // Submit batch jobs
        for (u32 batch = 0; batch < numBatches; batch++) {
            u32 start = batch * batchSize;
            u32 end = Math::min(start + batchSize, itemCount);
            
            BatchData data{function, userData, start, end, &batchesRemaining};
            
            submit([](void* payload) {
                BatchData* d = static_cast<BatchData*>(payload);
                for (u32 i = d->startIndex; i < d->endIndex; i++) {
                    d->function(i, d->userData);
                }
                d->counter->fetch_sub(1, std::memory_order_release);
            }, data, priority);
        }
        
        // Help execute until all batches complete
        while (batchesRemaining.load(std::memory_order_acquire) > 0) {
            executeOneJob();
        }
    }
    
    // =========================================================================
    // Waiting
    // =========================================================================
    
    void wait(JobHandle handle) {
        if (!handle.isValid()) return;
        
        Job* job = getJob(handle);
        if (!job) return;
        
        // Spin-wait while helping with other jobs
        while (job->unfinishedJobs.load(std::memory_order_acquire) > 0) {
            if (!executeOneJob()) {
                // No jobs available, yield
                spinWait();
            }
        }
    }
    
    void waitAll() {
        // Execute jobs until all queues are empty
        while (hasWork()) {
            if (!executeOneJob()) {
                spinWait();
            }
        }
    }
    
    // =========================================================================
    // Query
    // =========================================================================
    
    u32 getWorkerCount() const { return m_config.numWorkerThreads; }
    u32 getTotalThreadCount() const { return m_config.numWorkerThreads + 1; }
    bool isInitialized() const { return m_initialized; }
    
    u32 getCurrentThreadIndex() const {
        // Main thread is always index 0 in our scheme
        // Workers are 1..N
        for (u32 i = 0; i < m_config.numWorkerThreads; i++) {
            if (std::this_thread::get_id() == m_workers[i].thread.get_id()) {
                return i + 1;
            }
        }
        return 0;  // Main thread
    }
    
private:
    JobSystem() = default;
    ~JobSystem() { shutdown(); }
    
    // =========================================================================
    // Worker Thread
    // =========================================================================
    
    void workerThreadMain(u32 workerIndex) {
        WorkerThread& worker = m_workers[workerIndex];
        
        WULFNET_LOG_DEBUG("JobSystem", "Worker %u started", workerIndex);
        
        if (m_config.enableFibers) {
            Fiber::convertThreadToFiber(&worker);
        }
        
        while (worker.isRunning.load(std::memory_order_acquire)) {
            Job* job = getJob(workerIndex);
            
            if (job) {
                executeJob(job);
                worker.jobsExecuted.fetch_add(1, std::memory_order_relaxed);
            } else {
                // No work available - spin, then yield, then sleep
                spinWait();
            }
        }
        
        if (m_config.enableFibers) {
            Fiber::convertFiberToThread();
        }
        
        WULFNET_LOG_DEBUG("JobSystem", "Worker %u stopped", workerIndex);
    }
    
    // =========================================================================
    // Job Execution
    // =========================================================================
    
    void executeJob(Job* job) {
        if (job->function) {
            job->function(job->payload);
        }
        
        // Mark job as complete
        finishJob(job);
    }
    
    void finishJob(Job* job) {
        i32 remaining = job->unfinishedJobs.fetch_sub(1, std::memory_order_release) - 1;
        
        if (remaining == 0 && job->parent.isValid()) {
            // Decrement parent's counter
            Job* parent = getJob(job->parent);
            if (parent) {
                finishJob(parent);
            }
        }
    }
    
    bool executeOneJob() {
        u32 threadIndex = getCurrentThreadIndex();
        Job* job = getJob(threadIndex);
        
        if (job) {
            executeJob(job);
            return true;
        }
        
        return false;
    }
    
    // =========================================================================
    // Job Retrieval (Work Stealing)
    // =========================================================================
    
    Job* getJob(u32 threadIndex) {
        WorkerThread* worker = (threadIndex > 0) ? &m_workers[threadIndex - 1] : nullptr;
        
        // Try local queue first
        if (worker) {
            Job* job = worker->queue.pop();
            if (job) return job;
        }
        
        // Try stealing from random victims
        return stealJob(threadIndex);
    }
    
    Job* stealJob(u32 threadIndex) {
        if (m_workers.empty()) return nullptr;
        
        WorkerThread* currentWorker = (threadIndex > 0) ? &m_workers[threadIndex - 1] : nullptr;
        
        // Random victim selection for better load balancing
        u32 numVictims = static_cast<u32>(m_workers.size());
        u32 startVictim = currentWorker ? 
            (currentWorker->rng() % numVictims) : 0;
        
        for (u32 i = 0; i < numVictims; i++) {
            u32 victimIndex = (startVictim + i) % numVictims;
            
            if (currentWorker && victimIndex == threadIndex - 1) {
                continue;  // Don't steal from self
            }
            
            Job* job = m_workers[victimIndex].queue.steal();
            if (job) {
                if (currentWorker) {
                    currentWorker->jobsStolen.fetch_add(1, std::memory_order_relaxed);
                }
                return job;
            }
        }
        
        return nullptr;
    }
    
    // =========================================================================
    // Job Queue Management
    // =========================================================================
    
    void pushJob(Job* job) {
        u32 threadIndex = getCurrentThreadIndex();
        
        if (threadIndex > 0 && threadIndex <= m_workers.size()) {
            m_workers[threadIndex - 1].queue.push(job);
        } else if (!m_workers.empty()) {
            // Main thread - push to worker 0's queue
            m_workers[0].queue.push(job);
        }
    }
    
    bool hasWork() const {
        for (const auto& worker : m_workers) {
            if (!worker.queue.empty()) return true;
        }
        return false;
    }
    
    // =========================================================================
    // Utility
    // =========================================================================
    
    JobHandle makeHandle(Job* job) {
        JobHandle handle;
        handle.index = static_cast<u32>(job - m_jobPool);
        handle.generation = 1;  // Simplified - should track generations properly
        return handle;
    }
    
    Job* getJob(JobHandle handle) {
        if (!handle.isValid()) return nullptr;
        if (handle.index >= m_config.maxJobs) return nullptr;
        return &m_jobPool[handle.index];
    }
    
    void spinWait() {
        // Adaptive spinning with pause instruction
        for (int i = 0; i < 32; i++) {
            #if WULFNET_COMPILER_MSVC
                _mm_pause();
            #else
                __builtin_ia32_pause();
            #endif
        }
        
        // After spinning, yield to OS
        std::this_thread::yield();
    }
    
    void setThreadAffinity([[maybe_unused]] std::thread& thread, 
                           [[maybe_unused]] u32 coreIndex) {
        #if WULFNET_PLATFORM_WINDOWS
            DWORD_PTR mask = 1ULL << coreIndex;
            SetThreadAffinityMask(thread.native_handle(), mask);
        #elif WULFNET_PLATFORM_LINUX
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(coreIndex, &cpuset);
            pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        #endif
    }
    
    // =========================================================================
    // Member Variables
    // =========================================================================
    
    JobSystemConfig m_config;
    std::vector<WorkerThread> m_workers;
    Job* m_jobPool = nullptr;
    std::atomic<u32> m_nextJobIndex{0};
    std::atomic<bool> m_isRunning{false};
    bool m_initialized = false;
    
    Fiber m_mainThreadFiber;
};

// =============================================================================
// Convenience Functions
// =============================================================================

inline JobSystem& Jobs() {
    return JobSystem::instance();
}

} // namespace WulfNet
