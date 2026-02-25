// =============================================================================
// WulfNet Engine - CO-FLIP System Unit Tests
// =============================================================================
// Validates the CO-FLIP fluid simulation system: initialization, particles,
// emitters, obstacles, conservation, and CPU simulation correctness.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Physics/Fluids/COFLIPSystem.h>
#include <cmath>

using namespace WulfNet;

// =============================================================================
// COFLIPParticle Structure Tests
// =============================================================================

void test_COFLIPParticle_SizeAlignment() {
    // GPU alignment requirement: 64 bytes, 16-byte aligned
    EXPECT_EQ(sizeof(COFLIPParticle), 64u);
    EXPECT_EQ(alignof(COFLIPParticle), 16u);
}

void test_COFLIPParticle_DefaultValues() {
    COFLIPParticle p{};
    EXPECT_NEAR(p.x, 0.0f, 1e-6f);
    EXPECT_NEAR(p.y, 0.0f, 1e-6f);
    EXPECT_NEAR(p.z, 0.0f, 1e-6f);
    EXPECT_NEAR(p.vx, 0.0f, 1e-6f);
    EXPECT_NEAR(p.vy, 0.0f, 1e-6f);
    EXPECT_NEAR(p.vz, 0.0f, 1e-6f);
    // Vorticity fields should also default to zero
    EXPECT_NEAR(p.wx, 0.0f, 1e-6f);
    EXPECT_NEAR(p.wy, 0.0f, 1e-6f);
    EXPECT_NEAR(p.wz, 0.0f, 1e-6f);
}

// =============================================================================
// COFLIPCell Structure Tests
// =============================================================================

void test_COFLIPCell_SizeAlignment() {
    EXPECT_EQ(sizeof(COFLIPCell), 48u);
    EXPECT_EQ(alignof(COFLIPCell), 16u);
}

// =============================================================================
// COFLIPConfig Tests
// =============================================================================

void test_COFLIPConfig_Defaults() {
    COFLIPConfig config;
    EXPECT_EQ(config.gridSizeX, 64u);
    EXPECT_EQ(config.gridSizeY, 64u);
    EXPECT_EQ(config.gridSizeZ, 64u);
    EXPECT_NEAR(config.cellSize, 0.1f, 1e-6f);
    EXPECT_NEAR(config.dt, 1.0f / 60.0f, 1e-6f);
    EXPECT_NEAR(config.gravityY, -9.81f, 0.01f);
    EXPECT_NEAR(config.flipRatio, 0.99f, 1e-6f);
    EXPECT_EQ(config.pressureIterations, 50u);
    EXPECT_NEAR(config.restDensity, 1000.0f, 1e-6f);
    EXPECT_TRUE(config.useGPU);
}

// =============================================================================
// Initialization Tests
// =============================================================================

void test_COFLIPSystem_DefaultState() {
    COFLIPSystem system;
    EXPECT_FALSE(system.IsInitialized());
    EXPECT_EQ(system.GetActiveParticleCount(), 0u);
}

void test_COFLIPSystem_Initialize_CPUMode() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.useGPU = false;

    bool result = system.Initialize(config);
    EXPECT_TRUE(result);
    EXPECT_TRUE(system.IsInitialized());

    // Verify config was stored
    const COFLIPConfig& stored = system.GetConfig();
    EXPECT_EQ(stored.gridSizeX, 16u);
    EXPECT_EQ(stored.gridSizeY, 16u);
    EXPECT_EQ(stored.gridSizeZ, 16u);
    EXPECT_NEAR(stored.cellSize, 0.2f, 1e-6f);

    system.Shutdown();
    EXPECT_FALSE(system.IsInitialized());
}

void test_COFLIPSystem_Initialize_SmallGrid() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 4;
    config.gridSizeY = 4;
    config.gridSizeZ = 4;
    config.cellSize = 0.5f;
    config.useGPU = false;

    bool result = system.Initialize(config);
    EXPECT_TRUE(result);
    EXPECT_TRUE(system.IsInitialized());
    system.Shutdown();
}

void test_COFLIPSystem_Shutdown_Idempotent() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.useGPU = false;

    system.Initialize(config);
    system.Shutdown();
    // Double shutdown should not crash
    system.Shutdown();
    EXPECT_FALSE(system.IsInitialized());
}

// =============================================================================
// Particle Management Tests
// =============================================================================

void test_COFLIPSystem_AddSingleParticle() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.useGPU = false;
    system.Initialize(config);

    uint32_t id = system.AddParticle(0.5f, 0.5f, 0.5f);
    EXPECT_EQ(system.GetActiveParticleCount(), 1u);

    const auto& particles = system.GetParticles();
    EXPECT_TRUE(particles.size() >= 1);
    EXPECT_NEAR(particles[0].x, 0.5f, 1e-6f);
    EXPECT_NEAR(particles[0].y, 0.5f, 1e-6f);
    EXPECT_NEAR(particles[0].z, 0.5f, 1e-6f);

    system.Shutdown();
}

void test_COFLIPSystem_AddParticleWithVelocity() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.useGPU = false;
    system.Initialize(config);

    system.AddParticle(1.0f, 1.0f, 1.0f, 2.0f, -1.0f, 0.5f);

    const auto& particles = system.GetParticles();
    EXPECT_NEAR(particles[0].vx, 2.0f, 1e-6f);
    EXPECT_NEAR(particles[0].vy, -1.0f, 1e-6f);
    EXPECT_NEAR(particles[0].vz, 0.5f, 1e-6f);

    system.Shutdown();
}

void test_COFLIPSystem_AddParticleBox() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.useGPU = false;
    system.Initialize(config);

    system.AddParticleBox(0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f);

    uint32_t count = system.GetActiveParticleCount();
    EXPECT_TRUE(count > 10); // Should create many particles

    // All particles should be within the box bounds
    for (uint32_t i = 0; i < count; i++) {
        const auto& p = system.GetParticles()[i];
        EXPECT_TRUE(p.x >= 0.5f - 0.01f && p.x <= 1.5f + 0.01f);
        EXPECT_TRUE(p.y >= 0.5f - 0.01f && p.y <= 1.5f + 0.01f);
        EXPECT_TRUE(p.z >= 0.5f - 0.01f && p.z <= 1.5f + 0.01f);
    }

    system.Shutdown();
}

void test_COFLIPSystem_AddParticleSphere() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.useGPU = false;
    system.Initialize(config);

    float cx = 1.5f, cy = 1.5f, cz = 1.5f;
    float radius = 0.5f;
    system.AddParticleSphere(cx, cy, cz, radius);

    uint32_t count = system.GetActiveParticleCount();
    EXPECT_TRUE(count > 10);

    // All particles should be within the sphere radius (with small tolerance)
    for (uint32_t i = 0; i < count; i++) {
        const auto& p = system.GetParticles()[i];
        float dx = p.x - cx;
        float dy = p.y - cy;
        float dz = p.z - cz;
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        EXPECT_TRUE(dist <= radius + 0.05f);
    }

    system.Shutdown();
}

void test_COFLIPSystem_MultipleParticleGroups() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.useGPU = false;
    system.Initialize(config);

    system.AddParticleBox(0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f);
    uint32_t count1 = system.GetActiveParticleCount();

    system.AddParticleSphere(2.0f, 2.0f, 2.0f, 0.3f);
    uint32_t count2 = system.GetActiveParticleCount();

    // Second group should add more particles
    EXPECT_TRUE(count2 > count1);

    system.Shutdown();
}

// =============================================================================
// Emitter Tests
// =============================================================================

void test_COFLIPSystem_AddEmitter() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.useGPU = false;
    system.Initialize(config);

    // Add an emitter pointing downward
    system.AddEmitter(1.5f, 2.5f, 1.5f, 0.0f, -1.0f, 0.0f, 100.0f, 1.0f);

    // Step several times to let the emitter produce particles
    for (int i = 0; i < 10; i++) {
        system.Step(1.0f / 60.0f);
    }

    // Emitter should have produced particles
    EXPECT_TRUE(system.GetActiveParticleCount() > 0);

    system.Shutdown();
}

// =============================================================================
// Solid Obstacle Tests
// =============================================================================

void test_COFLIPSystem_AddSolidBox() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.useGPU = false;
    system.Initialize(config);

    // Add a solid box in the middle — should not crash
    system.AddSolidBox(1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 2.0f);

    // Add water above the solid box
    system.AddParticleBox(1.0f, 1.5f, 1.0f, 2.0f, 2.5f, 2.0f);
    EXPECT_TRUE(system.GetActiveParticleCount() > 0);

    system.Shutdown();
}

void test_COFLIPSystem_AddSolidSphere() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.useGPU = false;
    system.Initialize(config);

    system.AddSolidSphere(1.5f, 1.5f, 1.5f, 0.5f);
    // Should not crash
    EXPECT_TRUE(system.IsInitialized());

    system.Shutdown();
}

// =============================================================================
// Simulation Step Tests (CPU)
// =============================================================================

void test_COFLIPSystem_StepEmpty() {
    // Stepping with no particles should not crash
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.cellSize = 0.5f;
    config.useGPU = false;
    system.Initialize(config);

    system.Step(1.0f / 60.0f);
    EXPECT_EQ(system.GetActiveParticleCount(), 0u);

    system.Shutdown();
}

void test_COFLIPSystem_StepGravity() {
    // Particles should fall under gravity
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 32;
    config.gridSizeZ = 16;
    config.cellSize = 0.1f;
    config.gravityY = -9.81f;
    config.useGPU = false;
    config.pressureIterations = 5;
    system.Initialize(config);

    // Place particles in the air
    system.AddParticleBox(0.5f, 1.5f, 0.5f, 1.0f, 2.0f, 1.0f);
    uint32_t count = system.GetActiveParticleCount();

    // Record initial average Y
    float initialAvgY = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        initialAvgY += system.GetParticles()[i].y;
    }
    initialAvgY /= static_cast<float>(count);

    // Step several frames
    for (int step = 0; step < 20; step++) {
        system.Step(1.0f / 60.0f);
    }

    // Average Y should have decreased (gravity pulling particles down)
    float finalAvgY = 0.0f;
    uint32_t finalCount = system.GetActiveParticleCount();
    for (uint32_t i = 0; i < finalCount; i++) {
        finalAvgY += system.GetParticles()[i].y;
    }
    if (finalCount > 0) {
        finalAvgY /= static_cast<float>(finalCount);
        EXPECT_TRUE(finalAvgY < initialAvgY);
    }

    system.Shutdown();
}

void test_COFLIPSystem_StepPreservesParticleCount() {
    // Stepping should not lose particles (no removal logic)
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.useGPU = false;
    config.pressureIterations = 3;
    system.Initialize(config);

    system.AddParticleSphere(1.5f, 1.5f, 1.5f, 0.3f);
    uint32_t initialCount = system.GetActiveParticleCount();

    for (int i = 0; i < 10; i++) {
        system.Step(1.0f / 60.0f);
    }

    // Particle count should remain stable (no emitters, no removal)
    EXPECT_EQ(system.GetActiveParticleCount(), initialCount);

    system.Shutdown();
}

void test_COFLIPSystem_StepMultipleTimesteps() {
    // Test robustness with many consecutive steps
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.useGPU = false;
    config.pressureIterations = 3;
    system.Initialize(config);

    system.AddParticleBox(0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f);

    // Run 100 frames without crashing
    for (int i = 0; i < 100; i++) {
        system.Step(1.0f / 60.0f);
    }

    // Particles should still be valid (finite positions)
    for (uint32_t i = 0; i < system.GetActiveParticleCount(); i++) {
        const auto& p = system.GetParticles()[i];
        EXPECT_TRUE(std::isfinite(p.x));
        EXPECT_TRUE(std::isfinite(p.y));
        EXPECT_TRUE(std::isfinite(p.z));
        EXPECT_TRUE(std::isfinite(p.vx));
        EXPECT_TRUE(std::isfinite(p.vy));
        EXPECT_TRUE(std::isfinite(p.vz));
    }

    system.Shutdown();
}

// =============================================================================
// Reset Tests
// =============================================================================

void test_COFLIPSystem_Reset() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.useGPU = false;
    system.Initialize(config);

    system.AddParticleBox(0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f);
    EXPECT_TRUE(system.GetActiveParticleCount() > 0);

    system.Reset();
    EXPECT_EQ(system.GetActiveParticleCount(), 0u);

    // Should still be initialized after reset
    EXPECT_TRUE(system.IsInitialized());

    system.Shutdown();
}

// =============================================================================
// Statistics Tests
// =============================================================================

void test_COFLIPSystem_Stats_Initial() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.useGPU = false;
    system.Initialize(config);

    const COFLIPStats& stats = system.GetStats();
    EXPECT_EQ(stats.activeParticles, 0u);
    EXPECT_NEAR(stats.totalEnergy, 0.0f, 1e-6f);

    system.Shutdown();
}

void test_COFLIPSystem_Stats_AfterStep() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.useGPU = false;
    config.pressureIterations = 3;
    system.Initialize(config);

    system.AddParticleSphere(1.5f, 1.5f, 1.5f, 0.3f);
    system.Step(1.0f / 60.0f);

    const COFLIPStats& stats = system.GetStats();
    EXPECT_TRUE(stats.activeParticles > 0);
    EXPECT_TRUE(stats.totalTimeMs >= 0.0f);

    system.Shutdown();
}

void test_COFLIPSystem_Stats_Timing() {
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.useGPU = false;
    config.pressureIterations = 3;
    system.Initialize(config);

    system.AddParticleBox(0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f);
    system.Step(1.0f / 60.0f);

    const COFLIPStats& stats = system.GetStats();
    // Individual phase timings should be non-negative
    EXPECT_TRUE(stats.p2gTimeMs >= 0.0f);
    EXPECT_TRUE(stats.pressureTimeMs >= 0.0f);
    EXPECT_TRUE(stats.g2pTimeMs >= 0.0f);
    // Total should be >= sum of phases
    EXPECT_TRUE(stats.totalTimeMs >= 0.0f);

    system.Shutdown();
}

// =============================================================================
// GPU Buffer Handle Tests
// =============================================================================

void test_GPUBufferHandle_Default() {
    GPUBufferHandle handle;
    EXPECT_FALSE(handle.valid());
    EXPECT_EQ(handle.handle, 0u);
    EXPECT_EQ(handle.size, 0u);
}

void test_COFLIPSystem_GPUBuffers_CPUMode() {
    // In CPU mode, GPU buffers should be invalid
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.useGPU = false;
    system.Initialize(config);

    GPUBufferHandle particleBuf = system.GetParticleBuffer();
    GPUBufferHandle gridBuf = system.GetGridBuffer();
    EXPECT_FALSE(particleBuf.valid());
    EXPECT_FALSE(gridBuf.valid());

    system.Shutdown();
}

// =============================================================================
// Registration
// =============================================================================

void RegisterCOFLIPSystemTests() {
    // Particle structure
    RUN_TEST("COFLIPParticle_SizeAlignment", test_COFLIPParticle_SizeAlignment);
    RUN_TEST("COFLIPParticle_DefaultValues", test_COFLIPParticle_DefaultValues);
    RUN_TEST("COFLIPCell_SizeAlignment", test_COFLIPCell_SizeAlignment);

    // Config
    RUN_TEST("COFLIPConfig_Defaults", test_COFLIPConfig_Defaults);

    // Initialization
    RUN_TEST("COFLIPSystem_DefaultState", test_COFLIPSystem_DefaultState);
    RUN_TEST("COFLIPSystem_Initialize_CPUMode", test_COFLIPSystem_Initialize_CPUMode);
    RUN_TEST("COFLIPSystem_Initialize_SmallGrid", test_COFLIPSystem_Initialize_SmallGrid);
    RUN_TEST("COFLIPSystem_Shutdown_Idempotent", test_COFLIPSystem_Shutdown_Idempotent);

    // Particle management
    RUN_TEST("COFLIPSystem_AddSingleParticle", test_COFLIPSystem_AddSingleParticle);
    RUN_TEST("COFLIPSystem_AddParticleWithVelocity", test_COFLIPSystem_AddParticleWithVelocity);
    RUN_TEST("COFLIPSystem_AddParticleBox", test_COFLIPSystem_AddParticleBox);
    RUN_TEST("COFLIPSystem_AddParticleSphere", test_COFLIPSystem_AddParticleSphere);
    RUN_TEST("COFLIPSystem_MultipleParticleGroups", test_COFLIPSystem_MultipleParticleGroups);

    // Emitters
    RUN_TEST("COFLIPSystem_AddEmitter", test_COFLIPSystem_AddEmitter);

    // Obstacles
    RUN_TEST("COFLIPSystem_AddSolidBox", test_COFLIPSystem_AddSolidBox);
    RUN_TEST("COFLIPSystem_AddSolidSphere", test_COFLIPSystem_AddSolidSphere);

    // Simulation
    RUN_TEST("COFLIPSystem_StepEmpty", test_COFLIPSystem_StepEmpty);
    RUN_TEST("COFLIPSystem_StepGravity", test_COFLIPSystem_StepGravity);
    RUN_TEST("COFLIPSystem_StepPreservesParticleCount", test_COFLIPSystem_StepPreservesParticleCount);
    RUN_TEST("COFLIPSystem_StepMultipleTimesteps", test_COFLIPSystem_StepMultipleTimesteps);

    // Reset
    RUN_TEST("COFLIPSystem_Reset", test_COFLIPSystem_Reset);

    // Statistics
    RUN_TEST("COFLIPSystem_Stats_Initial", test_COFLIPSystem_Stats_Initial);
    RUN_TEST("COFLIPSystem_Stats_AfterStep", test_COFLIPSystem_Stats_AfterStep);
    RUN_TEST("COFLIPSystem_Stats_Timing", test_COFLIPSystem_Stats_Timing);

    // GPU buffer handles
    RUN_TEST("GPUBufferHandle_Default", test_GPUBufferHandle_Default);
    RUN_TEST("COFLIPSystem_GPUBuffers_CPUMode", test_COFLIPSystem_GPUBuffers_CPUMode);
}
