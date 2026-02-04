// =============================================================================
// WulfNet Engine - Job System Benchmark
// =============================================================================
// Benchmarks the job system with various workloads to measure scalability.
// =============================================================================

#include "core/Types.h"
#include "core/Log.h"
#include "core/jobs/JobSystem.h"
#include "core/platform/Platform.h"
#include "core/math/Math.h"

#include <vector>
#include <cmath>
#include <numeric>

using namespace WulfNet;

// Simulated physics update (heavy computation)
void simulatePhysicsChunk(Vec3* positions, Vec3* velocities, 
                          u32 start, u32 end, f32 dt) {
    const Vec3 gravity(0.0f, -9.81f, 0.0f);
    const f32 damping = 0.99f;
    
    for (u32 i = start; i < end; i++) {
        // Apply gravity
        velocities[i] = velocities[i] + gravity * dt;
        
        // Apply damping
        velocities[i] = velocities[i] * damping;
        
        // Update position
        positions[i] = positions[i] + velocities[i] * dt;
        
        // Ground collision (simple bounce)
        if (positions[i].y() < 0.0f) {
            positions[i] = Vec3(positions[i].x(), 0.0f, positions[i].z());
            velocities[i] = Vec3(velocities[i].x(), 
                                 -velocities[i].y() * 0.8f,  // Bounce
                                 velocities[i].z());
        }
    }
}

void runBenchmark(const char* name, u32 numObjects, u32 numIterations, 
                  u32 batchSize, u32 numThreads) {
    WULFNET_LOG_INFO("\n--- {} ---", name);
    WULFNET_LOG_INFO("Objects: {}, Iterations: {}, Batch: {}, Threads: {}",
                     numObjects, numIterations, batchSize, numThreads);
    
    // Initialize data
    std::vector<Vec3> positions(numObjects);
    std::vector<Vec3> velocities(numObjects);
    
    for (u32 i = 0; i < numObjects; i++) {
        f32 angle = static_cast<f32>(i) / numObjects * 2.0f * PI;
        positions[i] = Vec3(std::cos(angle) * 10.0f, 
                           static_cast<f32>(i % 100) * 0.1f,
                           std::sin(angle) * 10.0f);
        velocities[i] = Vec3(0.0f, 5.0f, 0.0f);  // Initial upward velocity
    }
    
    const f32 dt = 1.0f / 60.0f;  // 60 FPS
    
    // Configure job system
    JobSystemConfig config;
    config.numWorkerThreads = numThreads;
    JobSystem::get().initialize(config);
    
    // Warmup
    for (u32 iter = 0; iter < 10; iter++) {
        JobSystem::get().parallelFor(0, numObjects, [&](u32 start, u32 end) {
            simulatePhysicsChunk(positions.data(), velocities.data(), 
                                start, end, dt);
        }, batchSize);
    }
    
    // Benchmark
    f64 totalTime = 0.0;
    f64 minTime = 999999.0;
    f64 maxTime = 0.0;
    
    for (u32 iter = 0; iter < numIterations; iter++) {
        f64 startTime = Platform::getTimeMilliseconds();
        
        JobSystem::get().parallelFor(0, numObjects, [&](u32 start, u32 end) {
            simulatePhysicsChunk(positions.data(), velocities.data(), 
                                start, end, dt);
        }, batchSize);
        
        f64 elapsed = Platform::getTimeMilliseconds() - startTime;
        totalTime += elapsed;
        minTime = std::min(minTime, elapsed);
        maxTime = std::max(maxTime, elapsed);
    }
    
    f64 avgTime = totalTime / numIterations;
    f64 objectsPerSecond = (numObjects * 1000.0) / avgTime;
    
    WULFNET_LOG_INFO("Results:");
    WULFNET_LOG_INFO("  Average: {:.3f} ms", avgTime);
    WULFNET_LOG_INFO("  Min:     {:.3f} ms", minTime);
    WULFNET_LOG_INFO("  Max:     {:.3f} ms", maxTime);
    WULFNET_LOG_INFO("  Throughput: {:.0f} objects/sec", objectsPerSecond);
    
    JobSystem::get().shutdown();
}

void runScalabilityTest(u32 numObjects) {
    WULFNET_LOG_INFO("\n=== Scalability Test ({} objects) ===", numObjects);
    
    auto sysInfo = Platform::getSystemInfo();
    std::vector<f64> times;
    std::vector<u32> threadCounts = {1, 2, 4};
    
    // Add more thread counts up to available cores
    for (u32 t = 8; t <= sysInfo.numLogicalCores; t *= 2) {
        threadCounts.push_back(t);
    }
    if (threadCounts.back() != sysInfo.numLogicalCores) {
        threadCounts.push_back(sysInfo.numLogicalCores);
    }
    
    for (u32 threads : threadCounts) {
        if (threads > sysInfo.numLogicalCores) break;
        
        // Initialize data
        std::vector<Vec3> positions(numObjects);
        std::vector<Vec3> velocities(numObjects);
        for (u32 i = 0; i < numObjects; i++) {
            positions[i] = Vec3::zero();
            velocities[i] = Vec3(0, 5, 0);
        }
        
        const f32 dt = 1.0f / 60.0f;
        const u32 batchSize = std::max(1u, numObjects / (threads * 4));
        
        JobSystemConfig config;
        config.numWorkerThreads = threads;
        JobSystem::get().initialize(config);
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            JobSystem::get().parallelFor(0, numObjects, [&](u32 start, u32 end) {
                simulatePhysicsChunk(positions.data(), velocities.data(), 
                                    start, end, dt);
            }, batchSize);
        }
        
        // Measure
        constexpr u32 iterations = 100;
        f64 startTime = Platform::getTimeMilliseconds();
        
        for (u32 iter = 0; iter < iterations; iter++) {
            JobSystem::get().parallelFor(0, numObjects, [&](u32 start, u32 end) {
                simulatePhysicsChunk(positions.data(), velocities.data(), 
                                    start, end, dt);
            }, batchSize);
        }
        
        f64 elapsed = (Platform::getTimeMilliseconds() - startTime) / iterations;
        times.push_back(elapsed);
        
        JobSystem::get().shutdown();
    }
    
    // Print results
    WULFNET_LOG_INFO("Thread scaling results:");
    WULFNET_LOG_INFO("{:>8} {:>12} {:>12}", "Threads", "Time (ms)", "Speedup");
    
    for (size_t i = 0; i < threadCounts.size() && i < times.size(); i++) {
        f64 speedup = times[0] / times[i];
        WULFNET_LOG_INFO("{:>8} {:>12.3f} {:>12.2f}x", 
                        threadCounts[i], times[i], speedup);
    }
}

int main() {
    WULFNET_LOG_INFO("=== WulfNet Engine - Job System Benchmark ===");
    
    auto sysInfo = Platform::getSystemInfo();
    WULFNET_LOG_INFO("CPU: {} cores, {} threads",
                     sysInfo.numPhysicalCores, sysInfo.numLogicalCores);
    
    // Run various benchmarks
    runBenchmark("Small workload", 1000, 1000, 100, sysInfo.numLogicalCores);
    runBenchmark("Medium workload", 10000, 500, 256, sysInfo.numLogicalCores);
    runBenchmark("Large workload", 100000, 100, 1000, sysInfo.numLogicalCores);
    runBenchmark("Massive workload", 1000000, 20, 5000, sysInfo.numLogicalCores);
    
    // Scalability test
    runScalabilityTest(100000);
    
    WULFNET_LOG_INFO("\n=== Benchmark Complete ===");
    
    return 0;
}
