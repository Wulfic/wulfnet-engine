// =============================================================================
// WulfNet Engine - Thread Pool Implementation
// =============================================================================

#include "ThreadPool.h"
#include "WulfNet/Core/Logging/Logger.h"

namespace WulfNet {

static constexpr const char* LOG_CAT = "ThreadPool";

// =============================================================================
// Singleton
// =============================================================================

ThreadPool& ThreadPool::Get() {
    static ThreadPool instance(0); // auto-detect
    return instance;
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

ThreadPool::ThreadPool(int threadCount) {
    if (threadCount <= 0) {
        int hwThreads = static_cast<int>(std::thread::hardware_concurrency());
        // Leave 1 thread for the main thread, minimum 1 worker
        m_threadCount = std::max(1, hwThreads - 1);
    } else {
        m_threadCount = threadCount;
    }

    WULFNET_INFO(LOG_CAT, "Creating thread pool with " +
                 std::to_string(m_threadCount) + " workers");

    m_workers.reserve(m_threadCount);
    for (int i = 0; i < m_threadCount; ++i) {
        m_workers.emplace_back(&ThreadPool::WorkerLoop, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_shutdown = true;
    }
    m_condition.notify_all();

    for (auto& worker : m_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

// =============================================================================
// Worker Thread Loop
// =============================================================================

void ThreadPool::WorkerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_condition.wait(lock, [this] {
                return m_shutdown || !m_tasks.empty();
            });

            if (m_shutdown && m_tasks.empty()) {
                return;
            }

            task = std::move(m_tasks.front());
            m_tasks.pop();
        }
        task();
    }
}

// =============================================================================
// ParallelFor
// =============================================================================

void ThreadPool::ParallelFor(int begin, int end, const std::function<void(int)>& body) {
    int count = end - begin;
    if (count <= 0) return;

    // For very small ranges, just run inline
    if (count <= m_threadCount || m_threadCount <= 1) {
        for (int i = begin; i < end; ++i) {
            body(i);
        }
        return;
    }

    // Work-stealing approach: atomic counter
    std::atomic<int> workIdx(begin);
    std::atomic<int> completed(0);
    int numWorkers = std::min(m_threadCount, count);

    std::mutex doneMutex;
    std::condition_variable doneCv;

    for (int t = 0; t < numWorkers; ++t) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_tasks.emplace([&workIdx, end, &body, &completed, numWorkers, &doneMutex, &doneCv]() {
                while (true) {
                    int i = workIdx.fetch_add(1);
                    if (i >= end) break;
                    body(i);
                }
                if (completed.fetch_add(1) + 1 == numWorkers) {
                    std::lock_guard<std::mutex> lk(doneMutex);
                    doneCv.notify_one();
                }
            });
        }
        m_condition.notify_one();
    }

    // Wait for all workers to finish
    std::unique_lock<std::mutex> lock(doneMutex);
    doneCv.wait(lock, [&completed, numWorkers] {
        return completed.load() >= numWorkers;
    });
}

// =============================================================================
// ParallelForTiled
// =============================================================================

void ThreadPool::ParallelForTiled(int width, int height, int tileSize,
                                   const std::function<void(int, int, int, int)>& body) {
    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;
    int totalTiles = tilesX * tilesY;

    if (totalTiles <= 0) return;

    ParallelFor(0, totalTiles, [&](int tileIdx) {
        int tx = tileIdx % tilesX;
        int ty = tileIdx / tilesX;
        int x0 = tx * tileSize;
        int y0 = ty * tileSize;
        int tw = std::min(tileSize, width - x0);
        int th = std::min(tileSize, height - y0);
        body(x0, y0, tw, th);
    });
}

} // namespace WulfNet
