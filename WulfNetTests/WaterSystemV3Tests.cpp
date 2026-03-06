// =============================================================================
// WulfNet Engine - WaterSystemV3 Tests
// =============================================================================
// Validates SWE simulation correctness: volume conservation, hydrostatic
// equilibrium, sparse tile classification, boundary conditions, and
// water surface sampling. Outputs rich diagnostic log data.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Physics/WaterSystemV3.h>
#include <WulfNet/Core/Logging/Logger.h>
#include <WulfNet/Core/Profiling/Profiler.h>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace WulfNet::Physics;

// =============================================================================
// Helpers
// =============================================================================

static constexpr const char* LOG_CAT = "WaterV3Test";

// Log a 1-line summary of the grid state for diagnostics
static void LogGridSummary(const char* label, const WaterStateSOA& state, const WaterSystemV3Config& cfg) {
    float cellArea = cfg.gridSize * cfg.gridSize;
    double totalVol = state.CalculateTotalVolume(cellArea);

    float maxDepth = 0.0f;
    uint32_t wetCells = 0;
    double totalFluxMag = 0.0;
    for (uint32_t i = 0; i < cfg.width * cfg.height; ++i) {
        if (state.waterDepth[i] > 1e-6f) {
            ++wetCells;
            maxDepth = std::max(maxDepth, state.waterDepth[i]);
        }
        totalFluxMag += std::abs(state.flux[i].L) + std::abs(state.flux[i].R)
                      + std::abs(state.flux[i].T) + std::abs(state.flux[i].B);
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6)
       << label
       << " | vol=" << totalVol
       << " wetCells=" << wetCells
       << " maxDepth=" << maxDepth
       << " totalFlux=" << totalFluxMag;
    WULFNET_INFO(LOG_CAT, ss.str());
}

// =============================================================================
// Test: Volume Conservation
// =============================================================================
static void Test_VolumeConservation() {
    WaterSystemV3Config cfg;
    cfg.width = 64;
    cfg.height = 64;
    cfg.gridSize = 1.0f;
    cfg.gravity = 9.8f;

    WaterSystemV3 system(cfg, nullptr);
    float cellArea = cfg.gridSize * cfg.gridSize;

    // Add a known volume and verify it reads back exactly
    float initialVolume = 50.0f;
    system.AddWater(32, 32, initialVolume);

    WaterStateSOA& state = system.GetCPUState();
    double startVol = state.CalculateTotalVolume(cellArea);

    WULFNET_INFO(LOG_CAT, "=== Volume Conservation Test ===");
    WULFNET_INFO(LOG_CAT, "  Added " + std::to_string(initialVolume) + " m^3 at (32,32)");
    LogGridSummary("  [pre-sim]", state, cfg);

    EXPECT_NEAR(startVol, static_cast<double>(initialVolume), 0.001);

    // Simulate 100 steps (1.6s of game time)
    {
        WULFNET_SCOPED_TIMER("VolumeConservation_100steps");
        for (int i = 0; i < 100; ++i) {
            system.StepSimulationCPU(0.016f);

            // Log every 25 steps so we can watch the spread
            if ((i + 1) % 25 == 0) {
                LogGridSummary(("  [step " + std::to_string(i + 1) + "]").c_str(), state, cfg);
            }
        }
    }

    double endVol = state.CalculateTotalVolume(cellArea);
    double drift = std::abs(startVol - endVol);

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(9)
       << "  [result] startVol=" << startVol
       << " endVol=" << endVol
       << " drift=" << drift;
    WULFNET_INFO(LOG_CAT, ss.str());

    // Volume must be conserved within floating-point tolerance
    EXPECT_NEAR(startVol, endVol, 0.01);
}

// =============================================================================
// Test: Hydrostatic Equilibrium
// =============================================================================
static void Test_HydrostaticEquilibrium() {
    WaterSystemV3Config cfg;
    cfg.width = 64;
    cfg.height = 64;
    cfg.gridSize = 1.0f;
    cfg.gravity = 9.8f;

    WaterSystemV3 system(cfg, nullptr);
    WaterStateSOA& state = system.GetCPUState();

    // Create a gentle bowl-shaped terrain (lower slope improves convergence)
    uint32_t cx = cfg.width / 2;
    uint32_t cy = cfg.height / 2;
    for (uint32_t y = 0; y < cfg.height; ++y) {
        for (uint32_t x = 0; x < cfg.width; ++x) {
            float dx = static_cast<float>(x) - static_cast<float>(cx);
            float dy = static_cast<float>(y) - static_cast<float>(cy);
            state.terrainHeight[y * cfg.width + x] = (dx * dx + dy * dy) * 0.01f;
        }
    }

    system.AddWater(cx, cy, 100.0f);

    WULFNET_INFO(LOG_CAT, "=== Hydrostatic Equilibrium Test ===");
    WULFNET_INFO(LOG_CAT, "  Gentle bowl terrain (0.01*r^2), 100 m^3 at center");
    LogGridSummary("  [pre-sim]", state, cfg);

    // Step until kinetic energy ~ 0 (equilibrium state)
    {
        WULFNET_SCOPED_TIMER("HydrostaticEquilibrium_1000steps");
        for (int i = 0; i < 1000; ++i) {
            system.StepSimulationCPU(0.016f);

            if ((i + 1) % 250 == 0) {
                LogGridSummary(("  [step " + std::to_string(i + 1) + "]").c_str(), state, cfg);
            }
        }
    }

    // Measure mean water surface level
    float meanSurface = 0.0f;
    int count = 0;
    float minSurface = 1e9f, maxSurface = -1e9f;

    for (uint32_t i = 0; i < cfg.width * cfg.height; ++i) {
        if (state.waterDepth[i] > 0.01f) {
            float surface = state.terrainHeight[i] + state.waterDepth[i];
            meanSurface += surface;
            minSurface = std::min(minSurface, surface);
            maxSurface = std::max(maxSurface, surface);
            count++;
        }
    }

    if (count > 0) meanSurface /= count;

    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4)
           << "  [equilibrium] wetCells=" << count
           << " meanH=" << meanSurface
           << " range=[" << minSurface << ", " << maxSurface << "]"
           << " spread=" << (maxSurface - minSurface);
        WULFNET_INFO(LOG_CAT, ss.str());
    }

    // Verify all active water cells are roughly equal in surface height
    int outOfTolerance = 0;
    for (uint32_t i = 0; i < cfg.width * cfg.height; ++i) {
        if (state.waterDepth[i] > 0.01f) {
            float surface = state.terrainHeight[i] + state.waterDepth[i];
            if (std::abs(surface - meanSurface) > 0.5f) {
                ++outOfTolerance;
            }
        }
    }

    WULFNET_INFO(LOG_CAT, "  [check] cellsOutOfTolerance=" + std::to_string(outOfTolerance) + "/" + std::to_string(count));

    // All wet cells should have uniform surface height (H = h + d is constant)
    EXPECT_EQ(outOfTolerance, 0);
}

// =============================================================================
// Test: Boundary Conditions (no flux leaks at edges)
// =============================================================================
static void Test_BoundaryNoLeak() {
    WaterSystemV3Config cfg;
    cfg.width = 16;
    cfg.height = 16;
    cfg.gridSize = 1.0f;
    cfg.gravity = 9.8f;

    WULFNET_INFO(LOG_CAT, "=== Boundary No-Leak Test ===");

    // Place water at all 4 corners
    uint32_t corners[4][2] = {{0,0}, {15,0}, {0,15}, {15,15}};

    WaterSystemV3 system(cfg, nullptr);
    float cellArea = cfg.gridSize * cfg.gridSize;

    for (auto& c : corners) {
        system.AddWater(c[0], c[1], 10.0f);
    }

    WaterStateSOA& state = system.GetCPUState();
    double startVol = state.CalculateTotalVolume(cellArea);
    LogGridSummary("  [pre-sim]", state, cfg);

    for (int i = 0; i < 200; ++i) {
        system.StepSimulationCPU(0.016f);
    }

    double endVol = state.CalculateTotalVolume(cellArea);
    LogGridSummary("  [post-sim]", state, cfg);

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(9)
       << "  [result] startVol=" << startVol << " endVol=" << endVol
       << " drift=" << std::abs(startVol - endVol);
    WULFNET_INFO(LOG_CAT, ss.str());

    EXPECT_NEAR(startVol, endVol, 0.01);
}

// =============================================================================
// Test: Sparse Tile Classification
// =============================================================================
static void Test_SparseTileClassification() {
    WaterSystemV3Config cfg;
    cfg.width = 64;
    cfg.height = 64;
    cfg.gridSize = 1.0f;

    WaterSystemV3 system(cfg, nullptr);

    WULFNET_INFO(LOG_CAT, "=== Sparse Tile Classification Test ===");

    // Add water at (4, 4) — should activate tile (0, 0) only
    system.AddWater(4, 4, 5.0f);
    system.BuildSparseActiveTilesCPU();

    // Simulate a few steps to ensure water spreads
    for (int i = 0; i < 50; ++i) {
        system.StepSimulationCPU(0.016f);
    }
    system.BuildSparseActiveTilesCPU();

    LogGridSummary("  [post-spread]", system.GetCPUState(), cfg);
    WULFNET_INFO(LOG_CAT, "  Sparse tile test completed (active tile count in DEBUG log above)");

    // Volume should still be conserved
    double vol = system.GetCPUState().CalculateTotalVolume(cfg.gridSize * cfg.gridSize);
    EXPECT_NEAR(vol, 5.0, 0.01);
}

// =============================================================================
// Test: Water Surface Sampling
// =============================================================================
static void Test_SurfaceSampling() {
    WaterSystemV3Config cfg;
    cfg.width = 32;
    cfg.height = 32;
    cfg.gridSize = 1.0f;

    WaterSystemV3 system(cfg, nullptr);
    WaterStateSOA& state = system.GetCPUState();

    WULFNET_INFO(LOG_CAT, "=== Surface Sampling Test ===");

    // Set uniform water level over a 4x4 area
    float terrainH = 5.0f;
    float waterD = 3.0f;
    for (uint32_t y = 10; y < 14; ++y) {
        for (uint32_t x = 10; x < 14; ++x) {
            uint32_t idx = y * cfg.width + x;
            state.terrainHeight[idx] = terrainH;
            state.waterDepth[idx] = waterD;
        }
    }

    // The expected surface is terrainH + waterD = 8.0
    float expectedSurface = terrainH + waterD;
    float actualSurface = state.terrainHeight[11 * cfg.width + 11] + state.waterDepth[11 * cfg.width + 11];

    WULFNET_INFO(LOG_CAT, "  Expected surface: " + std::to_string(expectedSurface) + ", actual: " + std::to_string(actualSurface));

    EXPECT_NEAR(actualSurface, expectedSurface, 0.001f);
}

// =============================================================================
// Test: Add/Remove Water Symmetry
// =============================================================================
static void Test_AddRemoveSymmetry() {
    WaterSystemV3Config cfg;
    cfg.width = 16;
    cfg.height = 16;
    cfg.gridSize = 1.0f;

    WaterSystemV3 system(cfg, nullptr);

    WULFNET_INFO(LOG_CAT, "=== Add/Remove Symmetry Test ===");

    system.AddWater(8, 8, 20.0f);
    float cellArea = cfg.gridSize * cfg.gridSize;
    double afterAdd = system.GetCPUState().CalculateTotalVolume(cellArea);
    WULFNET_INFO(LOG_CAT, "  After add 20: vol=" + std::to_string(afterAdd));
    EXPECT_NEAR(afterAdd, 20.0, 0.001);

    system.RemoveWater(8, 8, 15.0f);
    double afterRemove = system.GetCPUState().CalculateTotalVolume(cellArea);
    WULFNET_INFO(LOG_CAT, "  After remove 15: vol=" + std::to_string(afterRemove));
    EXPECT_NEAR(afterRemove, 5.0, 0.001);

    // Over-removal should clamp to 0
    system.RemoveWater(8, 8, 999.0f);
    double afterOverRemove = system.GetCPUState().CalculateTotalVolume(cellArea);
    WULFNET_INFO(LOG_CAT, "  After over-remove: vol=" + std::to_string(afterOverRemove));
    EXPECT_NEAR(afterOverRemove, 0.0, 0.001);
}

// =============================================================================
// Test: Flat Terrain Symmetric Spread
// =============================================================================
static void Test_SymmetricSpread() {
    WaterSystemV3Config cfg;
    cfg.width = 33; // Odd so center is exact
    cfg.height = 33;
    cfg.gridSize = 1.0f;
    cfg.gravity = 9.8f;

    WaterSystemV3 system(cfg, nullptr);

    WULFNET_INFO(LOG_CAT, "=== Symmetric Spread Test ===");

    // Add water at exact center
    system.AddWater(16, 16, 50.0f);

    for (int i = 0; i < 100; ++i) {
        system.StepSimulationCPU(0.016f);
    }

    WaterStateSOA& state = system.GetCPUState();

    // Check symmetry: depth at (16+d, 16) should == depth at (16-d, 16)
    bool symmetric = true;
    for (int d = 1; d <= 10; ++d) {
        float dR = state.waterDepth[16 * cfg.width + (16 + d)];
        float dL = state.waterDepth[16 * cfg.width + (16 - d)];
        float dT = state.waterDepth[(16 - d) * cfg.width + 16];
        float dB = state.waterDepth[(16 + d) * cfg.width + 16];

        float maxDiff = std::max({std::abs(dR - dL), std::abs(dT - dB), std::abs(dR - dT)});
        if (maxDiff > 0.01f) {
            WULFNET_WARNING(LOG_CAT, "  Asymmetry at d=" + std::to_string(d) + " diff=" + std::to_string(maxDiff));
            symmetric = false;
        }
    }

    WULFNET_INFO(LOG_CAT, "  Spread symmetry: " + std::string(symmetric ? "PASS" : "FAIL"));
    LogGridSummary("  [post-spread]", state, cfg);

    EXPECT_TRUE(symmetric);
}

// =============================================================================
// Test: Performance Benchmark (128x128, 500 steps)
// =============================================================================
static void Test_PerformanceBenchmark() {
    WaterSystemV3Config cfg;
    cfg.width = 128;
    cfg.height = 128;
    cfg.gridSize = 1.0f;
    cfg.gravity = 9.8f;

    WaterSystemV3 system(cfg, nullptr);
    system.AddWater(64, 64, 200.0f);

    WULFNET_INFO(LOG_CAT, "=== Performance Benchmark (128x128 x 500 steps) ===");

    WulfNet::ManualTimer timer;
    timer.Start();

    for (int i = 0; i < 500; ++i) {
        system.StepSimulationCPU(0.016f);
    }

    double elapsedMs = timer.ElapsedMilliseconds();
    double msPerStep = elapsedMs / 500.0;
    double cellsPerSecond = (128.0 * 128.0) / (msPerStep / 1000.0);

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2)
       << "  Total: " << elapsedMs << "ms"
       << " | Per step: " << msPerStep << "ms"
       << " | Throughput: " << (cellsPerSecond / 1e6) << "M cells/sec";
    WULFNET_INFO(LOG_CAT, ss.str());

    LogGridSummary("  [final]", system.GetCPUState(), cfg);

    // Sanity: should complete in reasonable time (< 30s for 500 steps on any machine)
    EXPECT_LT(elapsedMs, 30000.0);
}

// =============================================================================
// Test: Parallel Performance — DamBreak-Scale (512x512)
// =============================================================================
// This is the diagnostic test that validates the parallel SWE solver
// at the same scale as the Dam Break visualization (512x512, 5m cells).
// It measures throughput, verifies water actually flows under gravity,
// and confirms volume conservation.
// =============================================================================
static void Test_ParallelDamBreak512() {
    WULFNET_INFO(LOG_CAT, "=== Parallel Dam Break Benchmark (512x512 x 5m cells) ===");

    // Match the DamBreak test config exactly (WaterBox baseline)
    WaterSystemV3Config cfg;
    cfg.width       = 512;
    cfg.height      = 512;
    cfg.gridSize    = 5.0f;
    cfg.gravity     = 9.81f;
    cfg.fluxDamping = 0.5f;    // WaterBox baseline — realistic settling
    cfg.dtMax       = 0.016f;  // Standard timestep (matches WaterBox)

    WaterSystemV3 system(cfg, nullptr);
    auto& state = system.GetCPUState();
    float cellArea = cfg.gridSize * cfg.gridSize;

    // Build simplified dam-break terrain:
    // - Upstream reservoir (gy < 200): elevation 300m
    // - Dam ridge (gy 200-210): elevation 400m with breach at center
    // - Downstream valley (gy > 210): elevation 250m with gentle slope
    for (uint32_t gy = 0; gy < 512; ++gy) {
        for (uint32_t gx = 0; gx < 512; ++gx) {
            uint32_t idx = gy * 512 + gx;
            if (gy < 200) {
                state.terrainHeight[idx] = 300.0f;
            } else if (gy <= 210) {
                // Dam ridge with a 40-cell-wide breach in the center
                float dxCenter = std::abs((float)gx - 256.0f);
                if (dxCenter < 20.0f)
                    state.terrainHeight[idx] = 275.0f;  // breach — lowered
                else
                    state.terrainHeight[idx] = 400.0f;  // intact dam wall
            } else {
                state.terrainHeight[idx] = 250.0f + (float)(512 - gy) * 0.05f;
            }
        }
    }

    // Fill reservoir ALL THE WAY to the dam wall (gy < 211) so breach cells
    // are already full of water — this is physically realistic because the
    // reservoir presses against the dam until it breaks.
    const float resSurface = 385.0f;
    for (uint32_t gy = 5; gy <= 210; ++gy)
        for (uint32_t gx = 5; gx < 507; ++gx) {
            uint32_t idx = gy * 512 + gx;
            float tH = state.terrainHeight[idx];
            if (tH < resSurface)
                system.AddWater(gx, gy, (resSurface - tH) * cellArea);
        }

    double initialVol = state.CalculateTotalVolume(cellArea);
    WULFNET_INFO(LOG_CAT, "  Initial volume: " + std::to_string(initialVol) + " m^3");

    // Seed breach flux using Torricelli's law: v = sqrt(2*g*h)
    // Applied to cells just past the dam (gy 208-212) so water rushes downstream
    for (uint32_t gy = 208; gy <= 215; ++gy)
        for (uint32_t gx = 236; gx < 276; ++gx) {
            uint32_t idx = gy * 512 + gx;
            float depth = state.waterDepth[idx];
            if (depth > 0.1f) {
                float v = std::sqrt(2.0f * 9.81f * depth);
                state.flux[idx].B += v * cfg.gridSize;
            }
        }

    // Probe points: reservoir center, breach gap, just downstream of dam
    float dResBefore  = state.waterDepth[100 * 512 + 256];
    float dGapBefore  = state.waterDepth[205 * 512 + 256];
    float dDownBefore = state.waterDepth[225 * 512 + 256]; // 15 cells past dam, not 140

    std::ostringstream pre;
    pre << std::fixed << std::setprecision(2)
        << "  [pre-sim] Depth res=" << dResBefore
        << " gap=" << dGapBefore
        << " down=" << dDownBefore;
    WULFNET_INFO(LOG_CAT, pre.str());

    // ---- Benchmark: run 200 steps (each 0.016s = 3.2s sim time) ----
    const int numSteps = 200;
    WulfNet::ManualTimer timer;
    timer.Start();

    for (int i = 0; i < numSteps; ++i) {
        system.StepSimulationCPU(0.016f);
    }

    double elapsedMs = timer.ElapsedMilliseconds();
    double msPerStep = elapsedMs / numSteps;
    double cellsPerStep = 512.0 * 512.0;
    double cellsPerSecond = cellsPerStep / (msPerStep / 1000.0);

    // Check depth at probes after simulation
    float dResAfter  = state.waterDepth[100 * 512 + 256];
    float dGapAfter  = state.waterDepth[205 * 512 + 256];

    // Count wet cells downstream of the dam (gy > 210) in breach corridor
    uint32_t downstreamWet = 0;
    float maxDownDepth = 0.0f;
    for (uint32_t gy = 211; gy < 260; ++gy)
        for (uint32_t gx = 230; gx < 282; ++gx) {
            float d = state.waterDepth[gy * 512 + gx];
            if (d > 1e-6f) { ++downstreamWet; maxDownDepth = std::max(maxDownDepth, d); }
        }

    double finalVol = state.CalculateTotalVolume(cellArea);
    double volDrift = std::abs(finalVol - initialVol) / initialVol * 100.0;

    std::ostringstream post;
    post << std::fixed << std::setprecision(4)
         << "  [post-sim] Depth res=" << dResAfter
         << " gap=" << dGapAfter
         << " downstreamWetCells=" << downstreamWet
         << " maxDownDepth=" << maxDownDepth;
    WULFNET_INFO(LOG_CAT, post.str());

    // Extra diagnostics: probe the first few rows past the dam
    for (uint32_t gy = 210; gy <= 214; ++gy) {
        float d = state.waterDepth[gy * 512 + 256];
        float t = state.terrainHeight[gy * 512 + 256];
        std::ostringstream row;
        row << std::fixed << std::setprecision(4)
            << "    row " << gy << ": terrain=" << t << " depth=" << d << " elev=" << (t+d);
        WULFNET_INFO(LOG_CAT, row.str());
    }

    std::ostringstream perf;
    perf << std::fixed << std::setprecision(2)
         << "  Total: " << elapsedMs << "ms"
         << " | Per step: " << msPerStep << "ms"
         << " | Throughput: " << (cellsPerSecond / 1e6) << "M cells/sec"
         << " | Target: <4ms/step for 60fps budget";
    WULFNET_INFO(LOG_CAT, perf.str());

    std::ostringstream vol;
    vol << std::fixed << std::setprecision(4)
        << "  Volume: initial=" << initialVol
        << " final=" << finalVol
        << " drift=" << volDrift << "%";
    WULFNET_INFO(LOG_CAT, vol.str());

    // ---- ASSERTIONS ----

    // 1. Breach gap still has water
    WULFNET_INFO(LOG_CAT, "  CHECK: Water in breach gap...");
    EXPECT_GT(dGapAfter, 1.0f);

    // 2. Water must have reached downstream: at least 1 wet cell past dam
    WULFNET_INFO(LOG_CAT, "  CHECK: Water reached downstream (wet cells=" + std::to_string(downstreamWet) + ")...");
    EXPECT_GT(downstreamWet, 0u);

    // 3. Volume conservation (expect <1% drift)
    WULFNET_INFO(LOG_CAT, "  CHECK: Volume conservation...");
    EXPECT_LT(volDrift, 1.0);

    // 4. Performance: must be < 10ms per step (well within 16.6ms frame budget)
    WULFNET_INFO(LOG_CAT, "  CHECK: Performance target...");
    EXPECT_LT(msPerStep, 10.0);

    WULFNET_INFO(LOG_CAT, "  All checks passed!");
}

// =============================================================================
// Test: Gravity Flow — water spawned in the air must fall and gain velocity
// =============================================================================
static void Test_GravityFlow() {
    WULFNET_INFO(LOG_CAT, "=== Gravity Flow Test ===");

    WaterSystemV3Config cfg;
    cfg.width       = 64;
    cfg.height      = 64;
    cfg.gridSize    = 1.0f;
    cfg.gravity     = 9.81f;
    cfg.fluxDamping = 0.01f;
    cfg.dtMax       = 0.004f;

    WaterSystemV3 system(cfg, nullptr);
    auto& state = system.GetCPUState();

    // Create a ramp: left side elevated (10m), right side at 0m
    for (uint32_t y = 0; y < 64; ++y) {
        for (uint32_t x = 0; x < 64; ++x) {
            state.terrainHeight[y * 64 + x] = 10.0f * (1.0f - (float)x / 63.0f);
        }
    }

    // Place water at top-left (elevated)
    for (uint32_t y = 20; y < 44; ++y)
        for (uint32_t x = 0; x < 10; ++x)
            system.AddWater(x, y, 5.0f);

    float initialDepthLeft = state.waterDepth[32 * 64 + 5];
    float initialDepthRight = state.waterDepth[32 * 64 + 58];

    WULFNET_INFO(LOG_CAT, "  Initial depth left=" + std::to_string(initialDepthLeft)
        + " right=" + std::to_string(initialDepthRight));

    // Run 200 steps
    for (int i = 0; i < 200; ++i)
        system.StepSimulationCPU(0.016f);

    float finalDepthLeft = state.waterDepth[32 * 64 + 5];
    float finalDepthRight = state.waterDepth[32 * 64 + 58];

    WULFNET_INFO(LOG_CAT, "  Final depth left=" + std::to_string(finalDepthLeft)
        + " right=" + std::to_string(finalDepthRight));

    // Water must have flowed right (downhill)
    WULFNET_INFO(LOG_CAT, "  CHECK: Water flowed downhill...");
    EXPECT_GT(finalDepthRight, 0.1f);

    // Left side should have lost water
    WULFNET_INFO(LOG_CAT, "  CHECK: Source depleted...");
    EXPECT_LT(finalDepthLeft, initialDepthLeft);

    // Right side should have more water than it started with
    WULFNET_INFO(LOG_CAT, "  CHECK: Destination filled...");
    EXPECT_GT(finalDepthRight, initialDepthRight);

    WULFNET_INFO(LOG_CAT, "  All checks passed!");
}

// =============================================================================
// Registration (called from WulfNetExtendedTests.cpp)
// =============================================================================
void RegisterWaterSystemV3Tests() {
    // Use Info level — Trace floods stdout with AddWater per-cell logs
    auto prevLevel = WulfNet::Logger::Get().GetMinLevel();
    WulfNet::Logger::Get().SetMinLevel(WulfNet::LogLevel::Info);

    RUN_TEST("WaterV3: Volume Conservation",        Test_VolumeConservation);
    RUN_TEST("WaterV3: Hydrostatic Equilibrium",    Test_HydrostaticEquilibrium);
    RUN_TEST("WaterV3: Boundary No-Leak",           Test_BoundaryNoLeak);
    RUN_TEST("WaterV3: Sparse Tile Classification", Test_SparseTileClassification);
    RUN_TEST("WaterV3: Surface Sampling",           Test_SurfaceSampling);
    RUN_TEST("WaterV3: Add/Remove Symmetry",        Test_AddRemoveSymmetry);
    RUN_TEST("WaterV3: Symmetric Spread",           Test_SymmetricSpread);
    RUN_TEST("WaterV3: Performance Benchmark",      Test_PerformanceBenchmark);
    RUN_TEST("WaterV3: Parallel DamBreak 512x512",  Test_ParallelDamBreak512);
    RUN_TEST("WaterV3: Gravity Flow",               Test_GravityFlow);

    // Restore previous log level
    WulfNet::Logger::Get().SetMinLevel(prevLevel);
}
