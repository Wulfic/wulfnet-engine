// =============================================================================
// WulfNet Engine - Threading & Parallelism Tests
// =============================================================================
// Tests for ThreadPool, ParallelFor, ParallelForTiled, and verifies that
// OpenMP-annotated systems produce correct results.
// =============================================================================

#include "TestHarness.h"
#include "WulfNet/Core/Threading/ThreadPool.h"
#include <vector>
#include <atomic>
#include <numeric>
#include <cmath>

// =============================================================================
// ThreadPool Basic Tests
// =============================================================================

static void Test_ThreadPool_Singleton() {
    auto& pool1 = WulfNet::ThreadPool::Get();
    auto& pool2 = WulfNet::ThreadPool::Get();
    EXPECT_TRUE(&pool1 == &pool2);
}

static void Test_ThreadPool_ThreadCount() {
    auto& pool = WulfNet::ThreadPool::Get();
    EXPECT_GE(pool.GetThreadCount(), 1);
}

static void Test_ThreadPool_NotShutdown() {
    auto& pool = WulfNet::ThreadPool::Get();
    EXPECT_FALSE(pool.IsShutdown());
}

static void Test_ThreadPool_SubmitSingle() {
    auto& pool = WulfNet::ThreadPool::Get();
    auto future = pool.Submit([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

static void Test_ThreadPool_SubmitMultiple() {
    auto& pool = WulfNet::ThreadPool::Get();
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 20; ++i) {
        futures.push_back(pool.Submit([i]() { return i * i; }));
    }
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

static void Test_ThreadPool_SubmitVoid() {
    auto& pool = WulfNet::ThreadPool::Get();
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.Submit([&counter]() { counter.fetch_add(1); }));
    }
    for (auto& f : futures) f.get();
    EXPECT_EQ(counter.load(), 10);
}

// =============================================================================
// ParallelFor Tests
// =============================================================================

static void Test_ParallelFor_EmptyRange() {
    auto& pool = WulfNet::ThreadPool::Get();
    int callCount = 0;
    pool.ParallelFor(0, 0, [&](int) { callCount++; });
    EXPECT_EQ(callCount, 0);
}

static void Test_ParallelFor_SingleElement() {
    auto& pool = WulfNet::ThreadPool::Get();
    std::atomic<int> value{0};
    pool.ParallelFor(0, 1, [&](int i) { value.store(i + 1); });
    EXPECT_EQ(value.load(), 1);
}

static void Test_ParallelFor_Sum() {
    // Sum 0..999 in parallel, verify against known answer
    auto& pool = WulfNet::ThreadPool::Get();
    const int N = 1000;
    std::atomic<int> sum{0};
    pool.ParallelFor(0, N, [&](int i) { sum.fetch_add(i); });
    int expected = N * (N - 1) / 2;
    EXPECT_EQ(sum.load(), expected);
}

static void Test_ParallelFor_LargeRange() {
    // Verify every index in [0, 10000) is visited exactly once
    auto& pool = WulfNet::ThreadPool::Get();
    const int N = 10000;
    std::vector<std::atomic<int>> visited(N);
    for (int i = 0; i < N; ++i) visited[i].store(0);

    pool.ParallelFor(0, N, [&](int i) { visited[i].fetch_add(1); });

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(visited[i].load(), 1);
    }
}

static void Test_ParallelFor_OffsetRange() {
    auto& pool = WulfNet::ThreadPool::Get();
    std::atomic<int> sum{0};
    pool.ParallelFor(50, 100, [&](int i) { sum.fetch_add(i); });
    // Sum from 50..99 = (99*100/2) - (49*50/2) = 4950 - 1225 = 3725
    EXPECT_EQ(sum.load(), 3725);
}

// =============================================================================
// ParallelForTiled Tests
// =============================================================================

static void Test_ParallelForTiled_Basic() {
    auto& pool = WulfNet::ThreadPool::Get();
    const int W = 64, H = 48, TILE = 16;
    std::vector<std::atomic<int>> visited(W * H);
    for (int i = 0; i < W * H; ++i) visited[i].store(0);

    pool.ParallelForTiled(W, H, TILE, [&](int x0, int y0, int tw, int th) {
        for (int y = y0; y < y0 + th; ++y) {
            for (int x = x0; x < x0 + tw; ++x) {
                visited[y * W + x].fetch_add(1);
            }
        }
    });

    // Every pixel should be visited exactly once
    for (int i = 0; i < W * H; ++i) {
        EXPECT_EQ(visited[i].load(), 1);
    }
}

static void Test_ParallelForTiled_NonAligned() {
    // Test with width/height not evenly divisible by tile size
    auto& pool = WulfNet::ThreadPool::Get();
    const int W = 100, H = 75, TILE = 32;
    std::atomic<int> pixelCount{0};

    pool.ParallelForTiled(W, H, TILE, [&](int x0, int y0, int tw, int th) {
        // Verify tile stays in bounds
        EXPECT_GE(x0, 0);
        EXPECT_GE(y0, 0);
        EXPECT_LE(x0 + tw, W);
        EXPECT_LE(y0 + th, H);
        pixelCount.fetch_add(tw * th);
    });

    EXPECT_EQ(pixelCount.load(), W * H);
}

// =============================================================================
// Correctness Under Parallelism Tests
// =============================================================================

static void Test_ParallelFor_NoDataRace() {
    // Write to separate array indices from parallel threads
    auto& pool = WulfNet::ThreadPool::Get();
    const int N = 5000;
    std::vector<float> output(N, 0.0f);

    pool.ParallelFor(0, N, [&](int i) {
        output[i] = static_cast<float>(i) * 2.5f;
    });

    for (int i = 0; i < N; ++i) {
        float expected = static_cast<float>(i) * 2.5f;
        EXPECT_NEAR(output[i], expected, 0.001f);
    }
}

static void Test_ParallelFor_StressTest() {
    // Run many small parallel-fors back to back (tests pool reuse)
    auto& pool = WulfNet::ThreadPool::Get();
    for (int round = 0; round < 50; ++round) {
        std::atomic<int> sum{0};
        pool.ParallelFor(0, 100, [&](int i) { sum.fetch_add(1); });
        EXPECT_EQ(sum.load(), 100);
    }
}

static void Test_CustomThreadPool_Lifecycle() {
    // Create a separate pool (not the singleton), use it, destroy it
    WulfNet::ThreadPool pool(2);
    EXPECT_EQ(pool.GetThreadCount(), 2);
    EXPECT_FALSE(pool.IsShutdown());

    auto future = pool.Submit([]() { return 99; });
    EXPECT_EQ(future.get(), 99);

    std::atomic<int> sum{0};
    pool.ParallelFor(0, 50, [&](int) { sum.fetch_add(1); });
    EXPECT_EQ(sum.load(), 50);
    // Destructor cleans up threads
}

// =============================================================================
// Registration
// =============================================================================

void RegisterThreadingTests() {
    std::cout << "\n=== Threading & Parallelism Tests ===" << std::endl;

    // ThreadPool basics
    RUN_TEST("ThreadPool_Singleton", Test_ThreadPool_Singleton);
    RUN_TEST("ThreadPool_ThreadCount", Test_ThreadPool_ThreadCount);
    RUN_TEST("ThreadPool_NotShutdown", Test_ThreadPool_NotShutdown);
    RUN_TEST("ThreadPool_SubmitSingle", Test_ThreadPool_SubmitSingle);
    RUN_TEST("ThreadPool_SubmitMultiple", Test_ThreadPool_SubmitMultiple);
    RUN_TEST("ThreadPool_SubmitVoid", Test_ThreadPool_SubmitVoid);

    // ParallelFor
    RUN_TEST("ParallelFor_EmptyRange", Test_ParallelFor_EmptyRange);
    RUN_TEST("ParallelFor_SingleElement", Test_ParallelFor_SingleElement);
    RUN_TEST("ParallelFor_Sum", Test_ParallelFor_Sum);
    RUN_TEST("ParallelFor_LargeRange", Test_ParallelFor_LargeRange);
    RUN_TEST("ParallelFor_OffsetRange", Test_ParallelFor_OffsetRange);

    // ParallelForTiled
    RUN_TEST("ParallelForTiled_Basic", Test_ParallelForTiled_Basic);
    RUN_TEST("ParallelForTiled_NonAligned", Test_ParallelForTiled_NonAligned);

    // Correctness
    RUN_TEST("ParallelFor_NoDataRace", Test_ParallelFor_NoDataRace);
    RUN_TEST("ParallelFor_StressTest", Test_ParallelFor_StressTest);
    RUN_TEST("CustomThreadPool_Lifecycle", Test_CustomThreadPool_Lifecycle);
}
