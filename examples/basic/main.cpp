// =============================================================================
// WulfNet Engine - Basic Example
// =============================================================================
// Demonstrates core engine features: logging, memory, math, and job system.
// =============================================================================

#include "core/Types.h"
#include "core/Log.h"
#include "core/memory/Memory.h"
#include "core/math/Math.h"
#include "core/jobs/JobSystem.h"
#include "core/platform/Platform.h"

#include <iostream>
#include <vector>
#include <chrono>

using namespace WulfNet;

int main() {
    // -------------------------------------------------------------------------
    // System Information
    // -------------------------------------------------------------------------
    WULFNET_LOG_INFO("=== WulfNet Engine - Basic Example ===");
    
    auto sysInfo = Platform::getSystemInfo();
    WULFNET_LOG_INFO("System: {}", sysInfo.osName);
    WULFNET_LOG_INFO("CPU Cores: {} physical, {} logical", 
                     sysInfo.numPhysicalCores, sysInfo.numLogicalCores);
    WULFNET_LOG_INFO("RAM: {} GB total", 
                     sysInfo.totalSystemMemory / (1024 * 1024 * 1024));
    WULFNET_LOG_INFO("SIMD: SSE4.2={}, AVX2={}, AVX512={}, FMA={}",
                     sysInfo.hasSSE42, sysInfo.hasAVX2, sysInfo.hasAVX512, sysInfo.hasFMA);

    // -------------------------------------------------------------------------
    // Memory Allocators Demo
    // -------------------------------------------------------------------------
    WULFNET_LOG_INFO("\n--- Memory Allocators ---");
    
    // Linear allocator for temporary frame data
    LinearAllocator frameAlloc(megabytes(1));
    
    struct Particle {
        Vec3 position;
        Vec3 velocity;
        f32 lifetime;
    };
    
    auto* particles = frameAlloc.allocArray<Particle>(1000);
    WULFNET_LOG_INFO("Allocated {} particles ({} bytes)", 
                     1000, sizeof(Particle) * 1000);
    
    // Initialize particles
    for (int i = 0; i < 1000; i++) {
        particles[i].position = Vec3(static_cast<f32>(i), 0.0f, 0.0f);
        particles[i].velocity = Vec3(0.0f, 1.0f, 0.0f);
        particles[i].lifetime = static_cast<f32>(i) * 0.01f;
    }
    
    // Reset allocator (all allocations freed)
    frameAlloc.reset();
    WULFNET_LOG_INFO("Frame allocator reset");

    // -------------------------------------------------------------------------
    // SIMD Math Demo
    // -------------------------------------------------------------------------
    WULFNET_LOG_INFO("\n--- SIMD Math Library ---");
    
    Vec3 position(10.0f, 20.0f, 30.0f);
    Vec3 velocity(1.0f, 0.0f, -1.0f);
    f32 deltaTime = 0.016f;  // ~60 FPS
    
    position = position + velocity * deltaTime;
    WULFNET_LOG_INFO("Position after update: ({}, {}, {})", 
                     position.x(), position.y(), position.z());
    
    // Quaternion rotation
    Quat rotation = Quat::fromAxisAngle(Vec3::up(), radians(45.0f));
    Vec3 forward = rotation.rotate(Vec3::forward());
    WULFNET_LOG_INFO("Forward after 45° Y rotation: ({:.3f}, {:.3f}, {:.3f})",
                     forward.x(), forward.y(), forward.z());
    
    // Matrix transforms
    Mat4 model = Mat4::translation(position) * Mat4::fromQuat(rotation);
    Mat4 view = Mat4::lookAt(Vec3(0, 5, 10), Vec3::zero(), Vec3::up());
    Mat4 proj = Mat4::perspective(radians(60.0f), 16.0f/9.0f, 0.1f, 1000.0f);
    Mat4 mvp = proj * view * model;
    WULFNET_LOG_INFO("MVP matrix created successfully");

    // -------------------------------------------------------------------------
    // Job System Demo
    // -------------------------------------------------------------------------
    WULFNET_LOG_INFO("\n--- Job System ---");
    
    JobSystemConfig jobConfig;
    jobConfig.numWorkerThreads = sysInfo.numLogicalCores - 1;  // Leave one for main
    JobSystem::get().initialize(jobConfig);
    WULFNET_LOG_INFO("Job system initialized with {} worker threads", 
                     jobConfig.numWorkerThreads);
    
    // Parallel sum benchmark
    constexpr u32 NUM_ELEMENTS = 10000000;  // 10 million
    std::vector<f32> data(NUM_ELEMENTS);
    
    // Initialize data
    for (u32 i = 0; i < NUM_ELEMENTS; i++) {
        data[i] = static_cast<f32>(i % 100) * 0.01f;
    }
    
    // Sequential sum
    auto startSeq = Platform::getTimeMilliseconds();
    f64 sumSeq = 0.0;
    for (u32 i = 0; i < NUM_ELEMENTS; i++) {
        sumSeq += data[i];
    }
    auto endSeq = Platform::getTimeMilliseconds();
    
    WULFNET_LOG_INFO("Sequential sum: {} (took {:.2f} ms)", 
                     sumSeq, endSeq - startSeq);
    
    // Parallel sum
    std::atomic<f64> sumPar{0.0};
    
    auto startPar = Platform::getTimeMilliseconds();
    JobSystem::get().parallelFor(0, NUM_ELEMENTS, [&data, &sumPar](u32 start, u32 end) {
        f64 localSum = 0.0;
        for (u32 i = start; i < end; i++) {
            localSum += data[i];
        }
        // Atomic add for f64
        f64 expected = sumPar.load(std::memory_order_relaxed);
        while (!sumPar.compare_exchange_weak(expected, expected + localSum,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
        }
    }, 10000);  // Batch size
    auto endPar = Platform::getTimeMilliseconds();
    
    WULFNET_LOG_INFO("Parallel sum: {} (took {:.2f} ms)", 
                     sumPar.load(), endPar - startPar);
    
    f64 speedup = (endSeq - startSeq) / (endPar - startPar);
    WULFNET_LOG_INFO("Speedup: {:.2f}x", speedup);
    
    JobSystem::get().shutdown();
    
    // -------------------------------------------------------------------------
    // Collision Detection Demo (AABB)
    // -------------------------------------------------------------------------
    WULFNET_LOG_INFO("\n--- Collision Detection ---");
    
    AABB boxA(Vec3(0, 0, 0), Vec3(2, 2, 2));
    AABB boxB(Vec3(1, 1, 1), Vec3(3, 3, 3));
    AABB boxC(Vec3(10, 10, 10), Vec3(12, 12, 12));
    
    WULFNET_LOG_INFO("Box A intersects Box B: {}", boxA.intersects(boxB));
    WULFNET_LOG_INFO("Box A intersects Box C: {}", boxA.intersects(boxC));
    
    Sphere sphereA(Vec3(0, 0, 0), 2.0f);
    Sphere sphereB(Vec3(3, 0, 0), 1.5f);
    
    WULFNET_LOG_INFO("Sphere A intersects Sphere B: {}", sphereA.intersects(sphereB));
    
    // -------------------------------------------------------------------------
    // Done
    // -------------------------------------------------------------------------
    WULFNET_LOG_INFO("\n=== Example Complete ===");
    
    return 0;
}
