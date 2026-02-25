// =============================================================================
// WulfNet Engine - Benchmark Harness
// =============================================================================
// Lightweight benchmarking framework for measuring system performance.
// Provides timing, iteration control, throughput calculation, and reporting.
//
// Usage:
//   BENCHMARK("AudioMixer_16Sources", {
//       mixer.MixFrame(output, 1024);
//   });
//
//   BENCHMARK_N("FluidGrid_Interpolate", 100000, {
//       grid.Interpolate(x, y, z);
//   });
// =============================================================================

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>

// =============================================================================
// Benchmark Result
// =============================================================================

struct BenchmarkResult {
    std::string name;
    int         iterations      = 0;
    double      totalTimeMs     = 0.0;
    double      meanTimeUs      = 0.0;    ///< Mean time per iteration (microseconds)
    double      medianTimeUs    = 0.0;    ///< Median time per iteration
    double      minTimeUs       = 0.0;    ///< Minimum time per iteration
    double      maxTimeUs       = 0.0;    ///< Maximum time per iteration
    double      stdDevUs        = 0.0;    ///< Standard deviation
    double      throughput      = 0.0;    ///< Operations per second
    bool        passed          = true;   ///< Whether benchmark completed without error
    std::string failureReason;
};

// =============================================================================
// Benchmark State (singleton)
// =============================================================================

struct BenchmarkState {
    std::vector<BenchmarkResult> results;
    int defaultIterations = 1000;     ///< Default iteration count
    int warmupIterations  = 100;      ///< Warmup iterations (not timed)

    static BenchmarkState& Get() {
        static BenchmarkState instance;
        return instance;
    }
};

// =============================================================================
// Core Benchmark Runner
// =============================================================================

/// Run a benchmark with a specific number of iterations.
/// The function is called `iterations` times, and per-iteration times are collected.
inline BenchmarkResult RunBenchmark(const char* name, int iterations,
                                     std::function<void()> func) {
    BenchmarkResult result;
    result.name = name;
    result.iterations = iterations;

    auto& state = BenchmarkState::Get();

    // Warmup phase (not timed)
    for (int i = 0; i < state.warmupIterations; ++i) {
        func();
    }

    // Collect per-iteration timings
    std::vector<double> timings;
    timings.reserve(iterations);

    auto totalStart = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(end - start).count();
        timings.push_back(us);
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    // Compute statistics
    std::sort(timings.begin(), timings.end());

    result.minTimeUs = timings.front();
    result.maxTimeUs = timings.back();

    // Median
    if (iterations % 2 == 0) {
        result.medianTimeUs = (timings[iterations / 2 - 1] + timings[iterations / 2]) / 2.0;
    } else {
        result.medianTimeUs = timings[iterations / 2];
    }

    // Mean
    double sum = 0.0;
    for (double t : timings) sum += t;
    result.meanTimeUs = sum / iterations;

    // Standard deviation
    double variance = 0.0;
    for (double t : timings) {
        double diff = t - result.meanTimeUs;
        variance += diff * diff;
    }
    result.stdDevUs = std::sqrt(variance / iterations);

    // Throughput (operations/second)
    if (result.meanTimeUs > 0.0) {
        result.throughput = 1000000.0 / result.meanTimeUs;
    }

    // Print inline result
    std::cout << "  BENCH [" << name << "] "
              << std::fixed << std::setprecision(2)
              << result.medianTimeUs << " us/op (median), "
              << result.meanTimeUs << " us/op (mean), "
              << std::setprecision(0)
              << result.throughput << " ops/sec"
              << std::endl;

    state.results.push_back(result);
    return result;
}

/// Run a benchmark that processes N items per iteration (for throughput reporting).
/// `itemsPerIteration` scales the throughput to "items/sec" rather than "ops/sec".
inline BenchmarkResult RunBenchmarkThroughput(const char* name, int iterations,
                                               int itemsPerIteration,
                                               std::function<void()> func) {
    BenchmarkResult result = RunBenchmark(name, iterations, func);

    // Recompute throughput as items/sec
    if (result.meanTimeUs > 0.0) {
        result.throughput = (1000000.0 / result.meanTimeUs) * itemsPerIteration;
    }

    // Overwrite the last printed line with corrected throughput
    std::cout << "         -> " << std::fixed << std::setprecision(0)
              << result.throughput << " items/sec"
              << " (" << itemsPerIteration << " items/op)"
              << std::endl;

    // Update the stored result
    if (!BenchmarkState::Get().results.empty()) {
        BenchmarkState::Get().results.back() = result;
    }

    return result;
}

// =============================================================================
// Convenience Macros
// =============================================================================

/// Run a benchmark with default iteration count
#define BENCHMARK(name, code) \
    RunBenchmark(name, BenchmarkState::Get().defaultIterations, [&]() { code; })

/// Run a benchmark with explicit iteration count
#define BENCHMARK_N(name, iters, code) \
    RunBenchmark(name, iters, [&]() { code; })

/// Run a throughput benchmark (items/sec reporting)
#define BENCHMARK_THROUGHPUT(name, iters, itemsPerIter, code) \
    RunBenchmarkThroughput(name, iters, itemsPerIter, [&]() { code; })

// =============================================================================
// Assertion Macros for Benchmarks
// =============================================================================

/// Verify a benchmark meets a minimum throughput (ops/sec)
#define BENCH_EXPECT_THROUGHPUT_GT(result, minOpsPerSec) \
    do { \
        if ((result).throughput <= (minOpsPerSec)) { \
            TestState::Get().currentTestPassed = false; \
            TestState::Get().failureReason = std::string("Throughput ") + \
                std::to_string((result).throughput) + " ops/sec below minimum " + \
                std::to_string(minOpsPerSec); \
            return; \
        } \
    } while(0)

/// Verify a benchmark stays below a maximum latency (microseconds)
#define BENCH_EXPECT_LATENCY_LT(result, maxUs) \
    do { \
        if ((result).medianTimeUs >= (maxUs)) { \
            TestState::Get().currentTestPassed = false; \
            TestState::Get().failureReason = std::string("Latency ") + \
                std::to_string((result).medianTimeUs) + " us above maximum " + \
                std::to_string(maxUs); \
            return; \
        } \
    } while(0)

// =============================================================================
// Report Printer
// =============================================================================

inline int PrintBenchmarkReport() {
    const auto& results = BenchmarkState::Get().results;

    std::cout << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "  Benchmark Results (" << results.size() << " benchmarks)" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << std::endl;

    // Header
    std::cout << std::left << std::setw(45) << "Benchmark"
              << std::right << std::setw(12) << "Median(us)"
              << std::setw(12) << "Mean(us)"
              << std::setw(12) << "Min(us)"
              << std::setw(12) << "StdDev(us)"
              << std::setw(15) << "Throughput"
              << std::endl;
    std::cout << std::string(108, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::left << std::setw(45) << r.name
                  << std::right << std::fixed
                  << std::setw(12) << std::setprecision(2) << r.medianTimeUs
                  << std::setw(12) << std::setprecision(2) << r.meanTimeUs
                  << std::setw(12) << std::setprecision(2) << r.minTimeUs
                  << std::setw(12) << std::setprecision(2) << r.stdDevUs
                  << std::setw(15) << std::setprecision(0) << r.throughput
                  << std::endl;
    }

    std::cout << std::string(108, '-') << std::endl;
    std::cout << std::endl;

    // Summary
    double totalTime = 0.0;
    for (const auto& r : results) totalTime += r.totalTimeMs;
    std::cout << "  Total benchmark time: " << std::fixed << std::setprecision(1)
              << totalTime << " ms" << std::endl;
    std::cout << "=================================================" << std::endl;

    return 0;
}

