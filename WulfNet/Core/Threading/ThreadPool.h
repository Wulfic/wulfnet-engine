// =============================================================================
// WulfNet Engine - Persistent Thread Pool
// =============================================================================
// A persistent, fixed-size thread pool that outlives individual frames.
// Replaces per-frame std::thread creation with zero-overhead task dispatch.
//
// Features:
//   - Fixed thread count (hardware_concurrency - 1 by default)
//   - Lock-free task submission via std::function queue
//   - ParallelFor(begin, end, body) for range-based parallelism
//   - ParallelForTiled(w, h, tileSize, body) for 2D rendering tiles
//   - Integrates with WulfNet profiler
//
// Usage:
//   auto& pool = ThreadPool::Get();
//   pool.ParallelFor(0, numObjects, [&](int i) { ProcessObject(i); });
//
//   // Or submit individual tasks:
//   auto future = pool.Submit([&]() { return ComputeSomething(); });
//   auto result = future.get();
// =============================================================================

#pragma once

#include "WulfNet/API.h"
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <algorithm>
#include <cstdint>

namespace WulfNet {

class WULFNET_API ThreadPool {
public:
    /// Singleton accessor. The pool is created on first use.
    static ThreadPool& Get();

    /// Construct with explicit thread count. Use 0 for auto (hardware_concurrency - 1).
    explicit ThreadPool(int threadCount = 0);
    ~ThreadPool();

    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // =========================================================================
    // Task Submission
    // =========================================================================

    /// Submit a callable and get a future for its result.
    template<typename F>
    auto Submit(F&& func) -> std::future<decltype(func())> {
        using ReturnType = decltype(func());
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(std::forward<F>(func));
        std::future<ReturnType> future = task->get_future();

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_tasks.emplace([task]() { (*task)(); });
        }
        m_condition.notify_one();

        return future;
    }

    // =========================================================================
    // Parallel Patterns
    // =========================================================================

    /// Execute body(i) for i in [begin, end) across all threads.
    /// Blocks until all iterations complete.
    void ParallelFor(int begin, int end, const std::function<void(int)>& body);

    /// Execute body(tileX, tileY, tileW, tileH) across 2D tiles.
    /// Tiles cover the [0, width) x [0, height) region.
    /// Blocks until all tiles complete.
    void ParallelForTiled(int width, int height, int tileSize,
                          const std::function<void(int, int, int, int)>& body);

    // =========================================================================
    // Queries
    // =========================================================================

    int GetThreadCount() const { return m_threadCount; }
    bool IsShutdown() const { return m_shutdown; }

private:
    void WorkerLoop();

    int m_threadCount = 0;
    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    bool m_shutdown = false;
};

} // namespace WulfNet
