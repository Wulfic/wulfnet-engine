// =============================================================================
// WulfNet Engine - Job System Tests
// =============================================================================

#include <cassert>
#include <catch2/catch_all.hpp>
#include "core/jobs/JobSystem.h"
#include "core/Types.h"
#include <atomic>
#include <vector>
#include <thread>
#include <chrono>

using namespace WulfNet;

// Helper to run job system tests
class JobSystemFixture {
public:
    JobSystemFixture() {
        JobSystemConfig config;
        config.numWorkerThreads = 4;
        JobSystem::instance().initialize(config);
    }
    
    ~JobSystemFixture() {
        JobSystem::instance().shutdown();
    }
};

TEST_CASE_METHOD(JobSystemFixture, "JobSystem basic job execution", "[jobs]") {
    std::atomic<bool> executed{false};
    
    // Submit a lambda directly
    JobHandle handle = JobSystem::instance().submit([](void* data) {
        auto** flagPtr = static_cast<std::atomic<bool>**>(data);
        (*flagPtr)->store(true);
    }, &executed);
    
    JobSystem::instance().wait(handle);
    
    REQUIRE(executed.load());
}

TEST_CASE_METHOD(JobSystemFixture, "JobSystem multiple independent jobs", "[jobs]") {
    constexpr int NUM_JOBS = 100;
    std::atomic<int> counter{0};
    
    std::vector<JobHandle> handles;
    handles.reserve(NUM_JOBS);
    
    for (int i = 0; i < NUM_JOBS; i++) {
        JobHandle h = JobSystem::instance().submit([](void* data) {
            auto** cPtr = static_cast<std::atomic<int>**>(data);
            (*cPtr)->fetch_add(1, std::memory_order_relaxed);
        }, &counter);
        handles.push_back(h);
    }
    
    // Wait for all jobs
    for (auto& h : handles) {
        JobSystem::instance().wait(h);
    }
    
    REQUIRE(counter.load() == NUM_JOBS);
}

TEST_CASE_METHOD(JobSystemFixture, "JobSystem parallel for", "[jobs]") {
    constexpr int SIZE = 10000;
    std::vector<int> data(SIZE, 0);
    
    // parallelFor signature: (itemCount, function(index, userdata), userData, minBatchSize, priority)
    // Note: implementation requires function to take (u32, void*), see JobSystem.h logic
    JobSystem::instance().parallelFor(SIZE, [](u32 i, void* userData) {
        auto* vec = static_cast<std::vector<int>*>(userData);
        (*vec)[i] = static_cast<int>(i);
    }, &data, 256);
    
    // Verify all elements were set correctly
    bool allCorrect = true;
    for (int i = 0; i < SIZE; i++) {
        if (data[i] != i) {
            allCorrect = false;
            break;
        }
    }
    
    REQUIRE(allCorrect);
}

TEST_CASE_METHOD(JobSystemFixture, "JobSystem parallel sum", "[jobs]") {
    constexpr int SIZE = 10000;
    std::vector<i64> data(SIZE);
    
    // Initialize data
    for (int i = 0; i < SIZE; i++) {
        data[i] = i + 1;  // 1 to SIZE
    }
    
    std::atomic<i64> sum{0};
    
    // Using simple parallelFor, not reduction
    // We add atomically
    JobSystem::instance().parallelFor(SIZE, [](u32 i, void* userData) {
        auto* params = static_cast<std::pair<std::vector<i64>*, std::atomic<i64>*>*>(userData);
        i64 val = (*params->first)[i];
        params->second->fetch_add(val, std::memory_order_relaxed);
    }, new std::pair<std::vector<i64>*, std::atomic<i64>*>(&data, &sum), 100);
    
    // Note: Memory leak of pair above, simplified for test. In real code use struct or persistent data.
    
    // Expected sum: n*(n+1)/2 where n = SIZE
    i64 expected = static_cast<i64>(SIZE) * (SIZE + 1) / 2;
    
    REQUIRE(sum.load() == expected);
}
