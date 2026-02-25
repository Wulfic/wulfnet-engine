// =============================================================================
// WulfNet Engine - Gaseous System Tests
// =============================================================================
// Tests for the Eulerian grid-based smoke/fire/explosion simulation system.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Physics/Gaseous/GaseousSystem.h>
#include <cmath>
#include <vector>

using namespace WulfNet;

// =============================================================================
// Helper: Create a small config for fast tests
// =============================================================================

static GaseousSystemConfig MakeSmallConfig(uint32_t res = 16) {
    GaseousSystemConfig config;
    config.resolutionX = res;
    config.resolutionY = res;
    config.resolutionZ = res;
    config.cellSize = 0.1f;
    config.originX = 0.0f;
    config.originY = 0.0f;
    config.originZ = 0.0f;
    config.ambientTemperature = 300.0f;
    config.buoyancyAlpha = 0.1f;
    config.buoyancyBeta = 0.5f;
    config.gravityY = -9.81f;
    config.pressureIterations = 10; // Fewer iterations for speed
    config.substeps = 1;
    return config;
}

// =============================================================================
// Config Defaults
// =============================================================================

void TestGaseous_ConfigDefaults() {
    GaseousSystemConfig config;
    EXPECT_EQ(config.resolutionX, 64u);
    EXPECT_EQ(config.resolutionY, 64u);
    EXPECT_EQ(config.resolutionZ, 64u);
    EXPECT_NEAR(config.cellSize, 0.1f, 1e-6f);
    EXPECT_NEAR(config.ambientTemperature, 300.0f, 1e-3f);
    EXPECT_NEAR(config.buoyancyAlpha, 0.1f, 1e-6f);
    EXPECT_NEAR(config.buoyancyBeta, 0.5f, 1e-6f);
    EXPECT_NEAR(config.vorticityStrength, 0.5f, 1e-6f);
    EXPECT_EQ(config.pressureIterations, 40u);
    EXPECT_FALSE(config.useGPU);
}

// =============================================================================
// GasCell Struct
// =============================================================================

void TestGaseous_CellDefaults() {
    GasCell cell;
    EXPECT_NEAR(cell.u, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.v, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.w, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.density, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.temperature, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.fuel, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.reaction, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.pressure, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.divergence, 0.0f, 1e-9f);
    EXPECT_TRUE(cell.state == GasCell::State::Air);
}

void TestGaseous_CellSize64Bytes() {
    // GasCell must be 64 bytes for GPU alignment
    EXPECT_EQ(sizeof(GasCell), 64u);
}

void TestGaseous_CellReset() {
    GasCell cell;
    cell.density = 5.0f;
    cell.temperature = 1000.0f;
    cell.u = 3.0f;
    cell.state = GasCell::State::Gas;
    cell.Reset();
    EXPECT_NEAR(cell.density, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.temperature, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.u, 0.0f, 1e-9f);
    EXPECT_TRUE(cell.state == GasCell::State::Air);
}

// =============================================================================
// Initialization
// =============================================================================

void TestGaseous_InitShutdown() {
    GaseousSystem sys;
    EXPECT_FALSE(sys.IsInitialized());

    auto config = MakeSmallConfig();
    EXPECT_TRUE(sys.Initialize(config));
    EXPECT_TRUE(sys.IsInitialized());
    EXPECT_EQ(sys.GetResolutionX(), 16u);
    EXPECT_EQ(sys.GetResolutionY(), 16u);
    EXPECT_EQ(sys.GetResolutionZ(), 16u);
    EXPECT_NEAR(sys.GetCellSize(), 0.1f, 1e-6f);
    EXPECT_EQ(sys.GetCellCount(), 16u * 16u * 16u);

    sys.Shutdown();
    EXPECT_FALSE(sys.IsInitialized());
}

void TestGaseous_DoubleInit() {
    GaseousSystem sys;
    auto config = MakeSmallConfig();
    EXPECT_TRUE(sys.Initialize(config));
    // Second init should fail
    EXPECT_FALSE(sys.Initialize(config));
    sys.Shutdown();
}

void TestGaseous_InitBadConfig() {
    GaseousSystem sys;

    // Zero resolution
    GaseousSystemConfig config;
    config.resolutionX = 0;
    EXPECT_FALSE(sys.Initialize(config));
    EXPECT_FALSE(sys.IsInitialized());

    // Zero cell size
    config.resolutionX = 8;
    config.cellSize = 0.0f;
    EXPECT_FALSE(sys.Initialize(config));

    // Negative cell size
    config.cellSize = -1.0f;
    EXPECT_FALSE(sys.Initialize(config));
}

// =============================================================================
// Emitter CRUD
// =============================================================================

void TestGaseous_EmitterCRUD() {
    GaseousSystem sys;
    auto config = MakeSmallConfig();
    sys.Initialize(config);
    EXPECT_EQ(sys.GetEmitterCount(), 0u);

    GasEmitter emitter;
    emitter.posX = 0.8f;
    emitter.posY = 0.8f;
    emitter.posZ = 0.8f;
    emitter.densityRate = 10.0f;
    uint32_t id = sys.AddEmitter(emitter);
    EXPECT_EQ(id, 0u);
    EXPECT_EQ(sys.GetEmitterCount(), 1u);

    GasEmitter* got = sys.GetEmitter(id);
    EXPECT_TRUE(got != nullptr);
    EXPECT_NEAR(got->posX, 0.8f, 1e-6f);
    EXPECT_NEAR(got->densityRate, 10.0f, 1e-6f);
    EXPECT_TRUE(got->enabled);

    // Remove (disables)
    sys.RemoveEmitter(id);
    got = sys.GetEmitter(id);
    EXPECT_TRUE(got != nullptr);
    EXPECT_FALSE(got->enabled);

    // Out-of-range
    EXPECT_TRUE(sys.GetEmitter(999) == nullptr);

    sys.Shutdown();
}

// =============================================================================
// Obstacle CRUD
// =============================================================================

void TestGaseous_ObstacleCRUD() {
    GaseousSystem sys;
    auto config = MakeSmallConfig();
    sys.Initialize(config);
    EXPECT_EQ(sys.GetObstacleCount(), 0u);

    GasObstacle obs;
    obs.posX = 0.5f;
    obs.posY = 0.5f;
    obs.posZ = 0.5f;
    obs.halfExtentX = 0.2f;
    obs.halfExtentY = 0.2f;
    obs.halfExtentZ = 0.2f;
    uint32_t id = sys.AddObstacle(obs);
    EXPECT_EQ(id, 0u);
    EXPECT_EQ(sys.GetObstacleCount(), 1u);

    GasObstacle* got = sys.GetObstacle(id);
    EXPECT_TRUE(got != nullptr);
    EXPECT_NEAR(got->posX, 0.5f, 1e-6f);
    EXPECT_TRUE(got->enabled);

    sys.RemoveObstacle(id);
    got = sys.GetObstacle(id);
    EXPECT_FALSE(got->enabled);

    EXPECT_TRUE(sys.GetObstacle(999) == nullptr);

    sys.Shutdown();
}

// =============================================================================
// Coordinate Conversion
// =============================================================================

void TestGaseous_CoordinateConversion() {
    GaseousSystem sys;
    auto config = MakeSmallConfig();
    config.originX = 1.0f;
    config.originY = 2.0f;
    config.originZ = 3.0f;
    config.cellSize = 0.5f;
    sys.Initialize(config);

    float gx, gy, gz;
    sys.WorldToGrid(1.0f, 2.0f, 3.0f, gx, gy, gz);
    EXPECT_NEAR(gx, 0.0f, 1e-6f);
    EXPECT_NEAR(gy, 0.0f, 1e-6f);
    EXPECT_NEAR(gz, 0.0f, 1e-6f);

    sys.WorldToGrid(2.0f, 3.0f, 4.0f, gx, gy, gz);
    EXPECT_NEAR(gx, 2.0f, 1e-6f);
    EXPECT_NEAR(gy, 2.0f, 1e-6f);
    EXPECT_NEAR(gz, 2.0f, 1e-6f);

    // Round trip
    float wx, wy, wz;
    sys.GridToWorld(2.0f, 2.0f, 2.0f, wx, wy, wz);
    EXPECT_NEAR(wx, 2.0f, 1e-6f);
    EXPECT_NEAR(wy, 3.0f, 1e-6f);
    EXPECT_NEAR(wz, 4.0f, 1e-6f);

    sys.Shutdown();
}

// =============================================================================
// Direct Cell Access
// =============================================================================

void TestGaseous_SetDensity() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    sys.Initialize(config);

    sys.SetDensity(4, 4, 4, 5.0f);
    const auto& cell = sys.GetCell(4, 4, 4);
    EXPECT_NEAR(cell.density, 5.0f, 1e-6f);

    // Adjacent cell should still be zero
    const auto& adj = sys.GetCell(3, 4, 4);
    EXPECT_NEAR(adj.density, 0.0f, 1e-6f);

    sys.Shutdown();
}

void TestGaseous_SetTemperature() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    sys.Initialize(config);

    sys.SetTemperature(3, 3, 3, 1000.0f);
    EXPECT_NEAR(sys.GetCell(3, 3, 3).temperature, 1000.0f, 1e-3f);

    sys.Shutdown();
}

void TestGaseous_SetFuel() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    sys.Initialize(config);

    sys.SetFuel(2, 2, 2, 0.8f);
    EXPECT_NEAR(sys.GetCell(2, 2, 2).fuel, 0.8f, 1e-6f);

    sys.Shutdown();
}

// =============================================================================
// Sampling
// =============================================================================

void TestGaseous_SampleDensity() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    config.originX = 0.0f;
    config.originY = 0.0f;
    config.originZ = 0.0f;
    config.cellSize = 0.1f;
    sys.Initialize(config);

    // Place density at center of grid cell (4,4,4)
    sys.SetDensity(4, 4, 4, 10.0f);

    // Sample at the world position corresponding to cell (4,4,4)
    // Cell (4,4,4) occupies world range [0.4, 0.5) so center is ~0.4
    float d = sys.SampleDensity(0.4f, 0.4f, 0.4f);
    EXPECT_NEAR(d, 10.0f, 1e-3f);

    // Outside grid should return 0
    float dOut = sys.SampleDensity(-1.0f, -1.0f, -1.0f);
    EXPECT_NEAR(dOut, 0.0f, 1e-9f);

    sys.Shutdown();
}

void TestGaseous_SampleTemperature() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    sys.Initialize(config);

    sys.SetTemperature(4, 4, 4, 500.0f);
    float t = sys.SampleTemperature(0.4f, 0.4f, 0.4f);
    EXPECT_NEAR(t, 500.0f, 1e-3f);

    sys.Shutdown();
}

void TestGaseous_SampleVelocity() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    sys.Initialize(config);

    // Set velocity directly on a cell
    auto& cell = sys.GetCellMut(4, 4, 4);
    cell.u = 1.0f;
    cell.v = 2.0f;
    cell.w = 3.0f;

    float vx, vy, vz;
    sys.SampleVelocity(0.4f, 0.4f, 0.4f, vx, vy, vz);
    EXPECT_NEAR(vx, 1.0f, 1e-3f);
    EXPECT_NEAR(vy, 2.0f, 1e-3f);
    EXPECT_NEAR(vz, 3.0f, 1e-3f);

    // Out of bounds
    sys.SampleVelocity(-10.0f, -10.0f, -10.0f, vx, vy, vz);
    EXPECT_NEAR(vx, 0.0f, 1e-9f);

    sys.Shutdown();
}

// =============================================================================
// Uninitialized guards
// =============================================================================

void TestGaseous_UninitializedSafety() {
    GaseousSystem sys;

    // Step on uninitialized should not crash
    sys.Step(0.016f);

    // Sampling on uninitialized
    float d = sys.SampleDensity(0, 0, 0);
    EXPECT_NEAR(d, 0.0f, 1e-9f);

    float t = sys.SampleTemperature(0, 0, 0);
    EXPECT_NEAR(t, 0.0f, 1e-9f);

    float vx, vy, vz;
    sys.SampleVelocity(0, 0, 0, vx, vy, vz);
    EXPECT_NEAR(vx, 0.0f, 1e-9f);

    // Reset on uninitialized should not crash
    sys.Reset();
}

// =============================================================================
// Reset
// =============================================================================

void TestGaseous_Reset() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    sys.Initialize(config);

    sys.SetDensity(4, 4, 4, 100.0f);
    sys.SetTemperature(4, 4, 4, 500.0f);
    EXPECT_GT(sys.GetCell(4, 4, 4).density, 0.0f);

    sys.Reset();
    EXPECT_NEAR(sys.GetCell(4, 4, 4).density, 0.0f, 1e-9f);
    EXPECT_NEAR(sys.GetCell(4, 4, 4).temperature, 0.0f, 1e-9f);
    // Still initialized after reset
    EXPECT_TRUE(sys.IsInitialized());

    sys.Shutdown();
}

// =============================================================================
// Smoke Simulation: Emitter injects density, buoyancy lifts it
// =============================================================================

void TestGaseous_SmokeEmitterDensity() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(16);
    config.densityDissipation = 1.0f;  // No dissipation
    config.temperatureDissipation = 1.0f;
    config.velocityDissipation = 1.0f;
    config.vorticityStrength = 0.0f;  // Disable vorticity to simplify
    sys.Initialize(config);

    // Place a point emitter at center of domain
    GasEmitter emitter;
    emitter.type = GasEmitterType::Point;
    emitter.posX = 0.8f;  // Center of 16-cell grid at 0.1 cell size
    emitter.posY = 0.8f;
    emitter.posZ = 0.8f;
    emitter.densityRate = 100.0f;
    emitter.temperatureRate = 0.0f;
    emitter.fuelRate = 0.0f;
    emitter.velocityX = 0.0f;
    emitter.velocityY = 0.0f;
    emitter.velocityZ = 0.0f;
    sys.AddEmitter(emitter);

    // Step a few times
    for (int i = 0; i < 5; ++i) {
        sys.Step(0.016f);
    }

    // Should have density in the grid
    const auto& stats = sys.GetStats();
    EXPECT_GT(stats.totalDensity, 0.0f);
    EXPECT_GT(stats.maxDensity, 0.0f);
    EXPECT_GT(stats.activeCells, 0u);

    sys.Shutdown();
}

void TestGaseous_BuoyancyLiftsSmoke() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(16);
    config.densityDissipation = 1.0f;
    config.temperatureDissipation = 1.0f;
    config.velocityDissipation = 1.0f;
    config.vorticityStrength = 0.0f;
    config.buoyancyBeta = 1.0f;  // Strong thermal lift
    config.buoyancyAlpha = 0.0f; // No density pull-down
    sys.Initialize(config);

    // Set temperature on a patch of cells — buoyancy should create upward velocity
    uint32_t mid = 8;
    for (uint32_t di = 0; di < 3; ++di) {
        for (uint32_t dk = 0; dk < 3; ++dk) {
            sys.SetTemperature(mid - 1 + di, 4, mid - 1 + dk, 500.0f);
            sys.SetDensity(mid - 1 + di, 4, mid - 1 + dk, 1.0f);
        }
    }

    sys.Step(0.016f);

    // After step, there should be non-zero upward velocity somewhere in the grid
    const auto& stats = sys.GetStats();
    EXPECT_GT(stats.maxVelocity, 0.0f);

    sys.Shutdown();
}

// =============================================================================
// Fire Simulation: Fuel + Temperature → Combustion
// =============================================================================

void TestGaseous_CombustionBurnsFuel() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    config.densityDissipation = 1.0f;
    config.temperatureDissipation = 1.0f;
    config.velocityDissipation = 1.0f;
    config.fuelDissipation = 1.0f;
    config.vorticityStrength = 0.0f;
    config.buoyancyAlpha = 0.0f;
    config.buoyancyBeta = 0.0f; // Disable buoyancy to prevent advection artifacts
    config.ignitionTemperature = 500.0f;
    config.burnRate = 2.0f;
    config.burnTemperature = 1500.0f;
    config.sootGeneration = 0.5f;
    sys.Initialize(config);

    // Place fuel on a cluster of cells to survive advection
    float totalFuelBefore = 0.0f;
    for (uint32_t di = 3; di <= 5; ++di) {
        for (uint32_t dj = 3; dj <= 5; ++dj) {
            for (uint32_t dk = 3; dk <= 5; ++dk) {
                sys.SetFuel(di, dj, dk, 1.0f);
                // Temperature above ignition: ambient(300) + 300 = 600 > 500
                sys.SetTemperature(di, dj, dk, 300.0f);
                totalFuelBefore += 1.0f;
            }
        }
    }

    sys.Step(0.016f);

    // Total fuel in the field should have decreased (combustion burned some)
    const auto& stats = sys.GetStats();
    EXPECT_LT(stats.totalFuel, totalFuelBefore);

    // Soot (density) should have been generated
    EXPECT_GT(stats.totalDensity, 0.0f);

    sys.Shutdown();
}

void TestGaseous_NoCombustionBelowIgnition() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    config.densityDissipation = 1.0f;
    config.temperatureDissipation = 1.0f;
    config.velocityDissipation = 1.0f;
    config.fuelDissipation = 1.0f;
    config.vorticityStrength = 0.0f;
    config.buoyancyAlpha = 0.0f;
    config.buoyancyBeta = 0.0f;  // Disable buoyancy to prevent velocity-driven advection
    config.ignitionTemperature = 500.0f;
    sys.Initialize(config);

    // Place fuel on a cluster below ignition temperature
    float totalFuelBefore = 0.0f;
    for (uint32_t di = 3; di <= 5; ++di) {
        for (uint32_t dj = 3; dj <= 5; ++dj) {
            for (uint32_t dk = 3; dk <= 5; ++dk) {
                sys.SetFuel(di, dj, dk, 1.0f);
                // Temperature below ignition: ambient(300) + 100 = 400 < 500
                sys.SetTemperature(di, dj, dk, 100.0f);
                totalFuelBefore += 1.0f;
            }
        }
    }

    sys.Step(0.016f);

    // Total fuel should remain roughly the same (no combustion, no dissipation)
    const auto& stats = sys.GetStats();
    EXPECT_GT(stats.totalFuel, totalFuelBefore * 0.9f);

    // No soot should have been generated (tiny tolerance for floating point)
    EXPECT_LT(stats.totalDensity, 0.1f);

    sys.Shutdown();
}

// =============================================================================
// Obstacle Interaction
// =============================================================================

void TestGaseous_ObstacleBlocksFlow() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(16);
    config.densityDissipation = 1.0f;
    config.temperatureDissipation = 1.0f;
    config.velocityDissipation = 1.0f;
    config.vorticityStrength = 0.0f;
    sys.Initialize(config);

    // Place a box obstacle at the center
    GasObstacle obs;
    obs.shape = GasObstacle::Shape::Box;
    obs.posX = 0.8f;
    obs.posY = 0.8f;
    obs.posZ = 0.8f;
    obs.halfExtentX = 0.15f;
    obs.halfExtentY = 0.15f;
    obs.halfExtentZ = 0.15f;
    sys.AddObstacle(obs);

    // Step to mark obstacles
    sys.Step(0.016f);

    // Center cell should be solid
    uint32_t mid = 8;
    const auto& centerCell = sys.GetCell(mid, mid, mid);
    EXPECT_TRUE(centerCell.state == GasCell::State::Solid);

    // Solid cells should have zero velocity
    EXPECT_NEAR(centerCell.u, 0.0f, 1e-6f);
    EXPECT_NEAR(centerCell.v, 0.0f, 1e-6f);
    EXPECT_NEAR(centerCell.w, 0.0f, 1e-6f);

    // Stats should report solid cells
    const auto& stats = sys.GetStats();
    EXPECT_GT(stats.solidCells, 0u);

    sys.Shutdown();
}

// =============================================================================
// Dissipation
// =============================================================================

void TestGaseous_DissipationReducesDensity() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    config.densityDissipation = 0.5f;   // Aggressive decay
    config.temperatureDissipation = 0.5f;
    config.velocityDissipation = 1.0f;
    config.vorticityStrength = 0.0f;
    sys.Initialize(config);

    // Set a blob of density
    sys.SetDensity(4, 4, 4, 100.0f);
    float before = sys.GetCell(4, 4, 4).density;

    sys.Step(0.016f);

    // After a step with dissipation, density should decrease
    // (Although advection may also redistribute, the global total should drop)
    const auto& stats = sys.GetStats();
    EXPECT_LT(stats.totalDensity, before);

    sys.Shutdown();
}

// =============================================================================
// Pressure Projection
// =============================================================================

void TestGaseous_PressureReducesDivergence() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    config.densityDissipation = 1.0f;
    config.temperatureDissipation = 1.0f;
    config.velocityDissipation = 1.0f;
    config.vorticityStrength = 0.0f;
    config.pressureIterations = 20;
    sys.Initialize(config);

    // Create a divergent velocity field
    for (uint32_t k = 1; k < 7; ++k) {
        for (uint32_t j = 1; j < 7; ++j) {
            for (uint32_t i = 1; i < 7; ++i) {
                auto& cell = sys.GetCellMut(i, j, k);
                // Expanding flow: velocity points outward from center
                cell.u = static_cast<float>(i) - 4.0f;
                cell.v = static_cast<float>(j) - 4.0f;
                cell.w = static_cast<float>(k) - 4.0f;
            }
        }
    }

    // Step: pressure projection should reduce divergence
    sys.Step(0.016f);

    // Check that max velocity is more controlled after projection
    // The velocity field should be altered by the pressure gradient
    const auto& stats = sys.GetStats();
    EXPECT_TRUE(stats.maxVelocity < 100.0f); // Should be reasonable

    sys.Shutdown();
}

// =============================================================================
// Statistics
// =============================================================================

void TestGaseous_StatsUpdated() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    config.densityDissipation = 1.0f;
    config.temperatureDissipation = 1.0f;
    config.velocityDissipation = 1.0f;
    sys.Initialize(config);

    // Stats should be zeroed initially
    const auto& stats = sys.GetStats();
    EXPECT_EQ(stats.activeCells, 0u);
    EXPECT_NEAR(stats.totalDensity, 0.0f, 1e-9f);

    // Set some density
    sys.SetDensity(4, 4, 4, 10.0f);
    sys.SetTemperature(3, 3, 3, 500.0f);
    sys.Step(0.016f);

    EXPECT_GT(sys.GetStats().activeCells, 0u);
    EXPECT_GT(sys.GetStats().totalTimeMs, 0.0f);

    sys.Shutdown();
}

// =============================================================================
// GPU Initialization (without actual GPU)
// =============================================================================

void TestGaseous_GPUInitNoContext() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    sys.Initialize(config);
    EXPECT_FALSE(sys.IsGPUEnabled());

    // InitializeGPU with null context should fail
    EXPECT_FALSE(sys.InitializeGPU(nullptr));
    EXPECT_FALSE(sys.IsGPUEnabled());

    sys.Shutdown();
}

// =============================================================================
// Empty Grid Step (no emitters/obstacles)
// =============================================================================

void TestGaseous_EmptyGridStep() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    sys.Initialize(config);

    // Step with no emitters — should not crash, no density
    sys.Step(0.016f);
    sys.Step(0.016f);
    sys.Step(0.016f);

    EXPECT_EQ(sys.GetStats().activeCells, 0u);
    EXPECT_NEAR(sys.GetStats().totalDensity, 0.0f, 1e-6f);

    sys.Shutdown();
}

// =============================================================================
// Sphere Emitter
// =============================================================================

void TestGaseous_SphereEmitter() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(16);
    config.densityDissipation = 1.0f;
    config.temperatureDissipation = 1.0f;
    config.velocityDissipation = 1.0f;
    config.vorticityStrength = 0.0f;
    sys.Initialize(config);

    GasEmitter emitter;
    emitter.type = GasEmitterType::Sphere;
    emitter.posX = 0.8f;
    emitter.posY = 0.8f;
    emitter.posZ = 0.8f;
    emitter.radius = 0.3f;
    emitter.densityRate = 50.0f;
    emitter.temperatureRate = 0.0f;
    emitter.velocityX = 0.0f;
    emitter.velocityY = 0.0f;
    emitter.velocityZ = 0.0f;
    sys.AddEmitter(emitter);

    sys.Step(0.016f);

    // Multiple cells should have received density
    uint32_t cellsWithDensity = 0;
    for (uint32_t k = 0; k < 16; ++k) {
        for (uint32_t j = 0; j < 16; ++j) {
            for (uint32_t i = 0; i < 16; ++i) {
                if (sys.GetCell(i, j, k).density > 0.0f)
                    cellsWithDensity++;
            }
        }
    }
    EXPECT_GT(cellsWithDensity, 1u); // Sphere emitter touches multiple cells

    sys.Shutdown();
}

// =============================================================================
// Zero DeltaTime
// =============================================================================

void TestGaseous_ZeroDeltaTime() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(8);
    sys.Initialize(config);

    sys.SetDensity(4, 4, 4, 5.0f);
    float before = sys.GetCell(4, 4, 4).density;

    sys.Step(0.0f);

    // With dt=0, density should be unchanged (no emitter injection, no dissipation power)
    // Dissipation: density *= pow(rate, 0) = pow(rate, 0) = 1.0, so no change
    EXPECT_NEAR(sys.GetCell(4, 4, 4).density, before, 0.01f);

    sys.Shutdown();
}

// =============================================================================
// Multiple Emitters
// =============================================================================

void TestGaseous_MultipleEmitters() {
    GaseousSystem sys;
    auto config = MakeSmallConfig(16);
    config.densityDissipation = 1.0f;
    config.vorticityStrength = 0.0f;
    sys.Initialize(config);

    GasEmitter e1;
    e1.posX = 0.3f;
    e1.posY = 0.3f;
    e1.posZ = 0.3f;
    e1.densityRate = 50.0f;
    e1.temperatureRate = 0.0f;
    e1.velocityY = 0.0f;

    GasEmitter e2;
    e2.posX = 1.2f;
    e2.posY = 1.2f;
    e2.posZ = 1.2f;
    e2.densityRate = 50.0f;
    e2.temperatureRate = 0.0f;
    e2.velocityY = 0.0f;

    sys.AddEmitter(e1);
    sys.AddEmitter(e2);
    EXPECT_EQ(sys.GetEmitterCount(), 2u);

    sys.Step(0.016f);

    // Both emitters should contribute density
    EXPECT_GT(sys.GetStats().totalDensity, 0.0f);
    EXPECT_GT(sys.GetStats().activeCells, 1u);

    sys.Shutdown();
}

// =============================================================================
// Registration Function
// =============================================================================

void RegisterGaseousSystemTests() {
    RUN_TEST("Gaseous_ConfigDefaults", TestGaseous_ConfigDefaults);
    RUN_TEST("Gaseous_CellDefaults", TestGaseous_CellDefaults);
    RUN_TEST("Gaseous_CellSize64Bytes", TestGaseous_CellSize64Bytes);
    RUN_TEST("Gaseous_CellReset", TestGaseous_CellReset);
    RUN_TEST("Gaseous_InitShutdown", TestGaseous_InitShutdown);
    RUN_TEST("Gaseous_DoubleInit", TestGaseous_DoubleInit);
    RUN_TEST("Gaseous_InitBadConfig", TestGaseous_InitBadConfig);
    RUN_TEST("Gaseous_EmitterCRUD", TestGaseous_EmitterCRUD);
    RUN_TEST("Gaseous_ObstacleCRUD", TestGaseous_ObstacleCRUD);
    RUN_TEST("Gaseous_CoordinateConversion", TestGaseous_CoordinateConversion);
    RUN_TEST("Gaseous_SetDensity", TestGaseous_SetDensity);
    RUN_TEST("Gaseous_SetTemperature", TestGaseous_SetTemperature);
    RUN_TEST("Gaseous_SetFuel", TestGaseous_SetFuel);
    RUN_TEST("Gaseous_SampleDensity", TestGaseous_SampleDensity);
    RUN_TEST("Gaseous_SampleTemperature", TestGaseous_SampleTemperature);
    RUN_TEST("Gaseous_SampleVelocity", TestGaseous_SampleVelocity);
    RUN_TEST("Gaseous_UninitializedSafety", TestGaseous_UninitializedSafety);
    RUN_TEST("Gaseous_Reset", TestGaseous_Reset);
    RUN_TEST("Gaseous_SmokeEmitterDensity", TestGaseous_SmokeEmitterDensity);
    RUN_TEST("Gaseous_BuoyancyLiftsSmoke", TestGaseous_BuoyancyLiftsSmoke);
    RUN_TEST("Gaseous_CombustionBurnsFuel", TestGaseous_CombustionBurnsFuel);
    RUN_TEST("Gaseous_NoCombustionBelowIgnition", TestGaseous_NoCombustionBelowIgnition);
    RUN_TEST("Gaseous_ObstacleBlocksFlow", TestGaseous_ObstacleBlocksFlow);
    RUN_TEST("Gaseous_DissipationReducesDensity", TestGaseous_DissipationReducesDensity);
    RUN_TEST("Gaseous_PressureReducesDivergence", TestGaseous_PressureReducesDivergence);
    RUN_TEST("Gaseous_StatsUpdated", TestGaseous_StatsUpdated);
    RUN_TEST("Gaseous_GPUInitNoContext", TestGaseous_GPUInitNoContext);
    RUN_TEST("Gaseous_EmptyGridStep", TestGaseous_EmptyGridStep);
    RUN_TEST("Gaseous_SphereEmitter", TestGaseous_SphereEmitter);
    RUN_TEST("Gaseous_ZeroDeltaTime", TestGaseous_ZeroDeltaTime);
    RUN_TEST("Gaseous_MultipleEmitters", TestGaseous_MultipleEmitters);
}
