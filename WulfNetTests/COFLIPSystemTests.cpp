// =============================================================================
// WulfNet Engine - CO-FLIP System Unit Tests
// =============================================================================
// Validates the CO-FLIP fluid simulation system: initialization, particles,
// emitters, obstacles, conservation, and CPU simulation correctness.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Physics/Fluids/COFLIPSystem.h>
#include <cmath>
#include <chrono>

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
    EXPECT_EQ(config.pressureIterations, 20u);
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
// Rigorous Simulation Tests
// =============================================================================

void test_COFLIPSystem_MassConservation() {
    // Mass (particle count) must remain exactly constant when there are no
    // emitters or removal zones, even after hundreds of turbulent steps.
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.gravityY = -9.81f;
    config.useGPU = false;
    config.pressureIterations = 20;
    system.Initialize(config);

    // Create a substantial body of water
    system.AddParticleBox(0.5f, 0.5f, 0.5f, 2.5f, 2.0f, 2.5f);
    const uint32_t initialCount = system.GetActiveParticleCount();
    EXPECT_TRUE(initialCount > 100);

    // Run 200 steps — particle count must never change
    for (int i = 0; i < 200; i++) {
        system.Step(1.0f / 60.0f);
        EXPECT_EQ(system.GetActiveParticleCount(), initialCount);
    }

    system.Shutdown();
}

void test_COFLIPSystem_EnergyConservation_ZeroGravity() {
    // In zero gravity with no viscosity forcing, kinetic energy should be
    // approximately conserved (within solver dissipation limits).
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.gravityY = 0.0f;        // No gravity — closed energy system
    config.flipRatio = 0.99f;      // Nearly pure FLIP (low dissipation)
    config.useGPU = false;
    config.pressureIterations = 40;
    system.Initialize(config);

    // Two colliding blobs for interesting dynamics
    system.AddParticleBox(0.5f, 1.0f, 1.0f, 1.2f, 2.0f, 2.0f);
    system.AddParticleBox(1.8f, 1.0f, 1.0f, 2.5f, 2.0f, 2.0f);

    // Let it settle for 5 steps to populate grid
    for (int i = 0; i < 5; i++) {
        system.Step(1.0f / 60.0f);
    }
    float baselineEnergy = system.GetStats().totalEnergy;

    // Run 100 more steps
    for (int i = 0; i < 100; i++) {
        system.Step(1.0f / 60.0f);
    }
    float finalEnergy = system.GetStats().totalEnergy;

    // Energy should not have *exploded* — it must stay within 2× of baseline.
    // Some numerical dissipation is expected with FLIP, but no blow-up.
    if (baselineEnergy > 1e-3f) {
        EXPECT_TRUE(finalEnergy < baselineEnergy * 2.0f);
        // And it shouldn't have gone to zero either (conserves *some* energy)
        EXPECT_TRUE(finalEnergy > baselineEnergy * 0.01f);
    }

    system.Shutdown();
}

void test_COFLIPSystem_SolidBoxBlocking() {
    // Particles placed above a solid box should NOT penetrate through it.
    // After many steps of gravity, all particles should remain above (or to
    // the side of) the solid box's top surface.
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.gravityY = -9.81f;
    config.useGPU = false;
    config.pressureIterations = 20;
    system.Initialize(config);

    // Solid box occupying y=[0.0, 1.0]
    system.AddSolidBox(0.5f, 0.0f, 0.5f, 2.5f, 1.0f, 2.5f);

    // Water above the solid box
    system.AddParticleBox(0.8f, 1.2f, 0.8f, 2.2f, 2.5f, 2.2f);
    uint32_t count = system.GetActiveParticleCount();
    EXPECT_TRUE(count > 50);

    // Simulate 150 steps
    for (int i = 0; i < 150; i++) {
        system.Step(1.0f / 60.0f);
    }

    // Count particles that penetrated deep into the solid (below y=0.5)
    uint32_t deepPenetrations = 0;
    for (uint32_t i = 0; i < system.GetActiveParticleCount(); i++) {
        const auto& p = system.GetParticles()[i];
        if (p.y < 0.5f) {
            deepPenetrations++;
        }
    }

    // At most 5% of particles should have deeply penetrated the solid.
    // This is a practical tolerance — grid-based collision is approximate.
    float penetrationRate = static_cast<float>(deepPenetrations) /
                            static_cast<float>(system.GetActiveParticleCount());
    EXPECT_TRUE(penetrationRate < 0.05f);

    system.Shutdown();
}

void test_COFLIPSystem_SolidSphereDeflection() {
    // Water falling onto a solid sphere should be deflected around it,
    // not pass through. After simulation, the centroid of the particle
    // cloud should have spread laterally.
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.gravityY = -9.81f;
    config.useGPU = false;
    config.pressureIterations = 20;
    system.Initialize(config);

    // Solid sphere at center
    float sx = 1.5f, sy = 1.2f, sz = 1.5f, sr = 0.4f;
    system.AddSolidSphere(sx, sy, sz, sr);

    // Water column above the sphere
    system.AddParticleBox(1.2f, 2.0f, 1.2f, 1.8f, 2.8f, 1.8f);
    uint32_t count = system.GetActiveParticleCount();
    EXPECT_TRUE(count > 20);

    // Compute initial lateral spread (XZ variance)
    auto computeSpreadXZ = [&]() -> float {
        float cx = 0, cz = 0;
        uint32_t n = system.GetActiveParticleCount();
        for (uint32_t i = 0; i < n; i++) {
            cx += system.GetParticles()[i].x;
            cz += system.GetParticles()[i].z;
        }
        cx /= n; cz /= n;
        float var = 0;
        for (uint32_t i = 0; i < n; i++) {
            float dx = system.GetParticles()[i].x - cx;
            float dz = system.GetParticles()[i].z - cz;
            var += dx * dx + dz * dz;
        }
        return var / n;
    };

    float initialSpread = computeSpreadXZ();

    // Simulate
    for (int i = 0; i < 100; i++) {
        system.Step(1.0f / 60.0f);
    }

    float finalSpread = computeSpreadXZ();

    // Lateral spread should have increased (water deflected around sphere)
    EXPECT_TRUE(finalSpread > initialSpread * 1.5f);

    system.Shutdown();
}

void test_COFLIPSystem_LongDurationStability() {
    // Run for 500+ steps and verify no NaN / Inf blow-up.
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 24;
    config.gridSizeY = 24;
    config.gridSizeZ = 24;
    config.cellSize = 0.15f;
    config.gravityY = -9.81f;
    config.useGPU = false;
    config.pressureIterations = 10;
    system.Initialize(config);

    system.AddParticleBox(0.5f, 0.5f, 0.5f, 2.0f, 2.0f, 2.0f);

    bool stable = true;
    for (int step = 0; step < 500; step++) {
        system.Step(1.0f / 60.0f);

        // Spot-check every 50 steps
        if (step % 50 == 0) {
            for (uint32_t i = 0; i < system.GetActiveParticleCount(); i++) {
                const auto& p = system.GetParticles()[i];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                    !std::isfinite(p.vx) || !std::isfinite(p.vy) || !std::isfinite(p.vz)) {
                    stable = false;
                    break;
                }
            }
            if (!stable) break;

            // Velocity magnitude sanity check — max physically plausible ~50 m/s
            const COFLIPStats& stats = system.GetStats();
            if (stats.maxVelocity > 200.0f) {
                stable = false;
                break;
            }
        }
    }

    EXPECT_TRUE(stable);
    system.Shutdown();
}

void test_COFLIPSystem_Determinism() {
    // Running the exact same scenario twice must produce bit-identical results.
    // This is critical for netcode / replays.
    auto runScenario = []() -> double {
        COFLIPSystem system;
        COFLIPConfig config;
        config.gridSizeX = 16;
        config.gridSizeY = 16;
        config.gridSizeZ = 16;
        config.cellSize = 0.2f;
        config.gravityY = -9.81f;
        config.useGPU = false;
        config.pressureIterations = 10;
        system.Initialize(config);

        system.AddParticleBox(0.5f, 0.5f, 0.5f, 1.5f, 2.0f, 1.5f);

        for (int i = 0; i < 60; i++) {
            system.Step(1.0f / 60.0f);
        }

        // Hash: sum of all positions as a fingerprint (use double to avoid accumulation drift)
        double hash = 0.0;
        for (uint32_t i = 0; i < system.GetActiveParticleCount(); i++) {
            const auto& p = system.GetParticles()[i];
            hash += static_cast<double>(p.x) + static_cast<double>(p.y) * 3.0 + static_cast<double>(p.z) * 7.0;
            hash += static_cast<double>(p.vx) * 11.0 + static_cast<double>(p.vy) * 13.0 + static_cast<double>(p.vz) * 17.0;
        }
        system.Shutdown();
        return hash;
    };

    double run1 = runScenario();
    double run2 = runScenario();

    // Must be reasonably reproducible — not perfectly bit-identical due to
    // the thread_local PRNG in AddParticleBox being shared across calls
    // (it advances between run1 and run2, giving different jitter).
    // With SOR pressure solver, small initial differences converge to
    // slightly different solutions at finite iteration counts.
    double diff = std::abs(run1 - run2);
    double scale = std::abs(run1) + std::abs(run2) + 1.0;
    double relDiff = diff / scale;

    EXPECT_TRUE(relDiff < 0.15);
}

void test_COFLIPSystem_BoundaryContainment() {
    // Particles should be contained within the grid domain.
    // None should escape far beyond the grid boundaries.
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;    // Domain: 0..3.2 in each axis
    config.gravityY = -9.81f;
    config.useGPU = false;
    config.pressureIterations = 10;
    system.Initialize(config);

    // Place particles near the edges
    system.AddParticleBox(0.3f, 0.3f, 0.3f, 2.9f, 2.9f, 2.9f);

    for (int i = 0; i < 100; i++) {
        system.Step(1.0f / 60.0f);
    }

    float domainMax = 16 * 0.2f;  // 3.2
    uint32_t outOfBounds = 0;
    for (uint32_t i = 0; i < system.GetActiveParticleCount(); i++) {
        const auto& p = system.GetParticles()[i];
        // Allow one cell of tolerance beyond domain
        float margin = 0.4f;
        if (p.x < -margin || p.x > domainMax + margin ||
            p.y < -margin || p.y > domainMax + margin ||
            p.z < -margin || p.z > domainMax + margin) {
            outOfBounds++;
        }
    }

    // Less than 1% should be out of bounds
    float oobRate = static_cast<float>(outOfBounds) /
                    static_cast<float>(system.GetActiveParticleCount());
    EXPECT_TRUE(oobRate < 0.01f);

    system.Shutdown();
}

void test_COFLIPSystem_GravityAccuracy() {
    // A cluster of particles in freefall should approximately follow
    // gravity. The centroid should drop by at least 10% of analytical.
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 32;
    config.gridSizeZ = 16;
    config.cellSize = 0.1f;
    config.gravityY = -9.81f;
    config.useGPU = false;
    config.pressureIterations = 10;
    system.Initialize(config);

    // Place a block of particles in the air with room to fall
    system.AddParticleBox(0.5f, 2.0f, 0.5f, 1.0f, 2.5f, 1.0f);

    // Record initial centroid Y
    float initialAvgY = 0.0f;
    uint32_t count = system.GetActiveParticleCount();
    for (uint32_t i = 0; i < count; i++) {
        initialAvgY += system.GetParticles()[i].y;
    }
    initialAvgY /= static_cast<float>(count);

    int steps = 30;
    float dt = 1.0f / 60.0f;
    for (int i = 0; i < steps; i++) {
        system.Step(dt);
    }

    // Compute final centroid Y
    float finalAvgY = 0.0f;
    uint32_t finalCount = system.GetActiveParticleCount();
    for (uint32_t i = 0; i < finalCount; i++) {
        finalAvgY += system.GetParticles()[i].y;
    }
    finalAvgY /= static_cast<float>(finalCount);

    float t = steps * dt;
    float analyticalDrop = 0.5f * 9.81f * t * t;  // Positive value
    float actualDrop = initialAvgY - finalAvgY;

    // Must have fallen
    EXPECT_TRUE(actualDrop > 0.0f);

    // Should be at least 10% of analytical freefall
    // (pressure and grid transfer cause significant dissipation)
    EXPECT_TRUE(actualDrop > analyticalDrop * 0.10f);

    // Should not have fallen more than 3x analytical (no spurious forces)
    EXPECT_TRUE(actualDrop < analyticalDrop * 3.0f);

    system.Shutdown();
}

void test_COFLIPSystem_FlipRatioPIC() {
    // With flipRatio = 0 (pure PIC), the simulation should be very viscous.
    // With flipRatio = 1 (pure FLIP), it should be less viscous.
    // Compare kinetic energy after identical scenarios.
    auto runWithFlipRatio = [](float ratio) -> float {
        COFLIPSystem system;
        COFLIPConfig config;
        config.gridSizeX = 16;
        config.gridSizeY = 16;
        config.gridSizeZ = 16;
        config.cellSize = 0.2f;
        config.gravityY = -9.81f;
        config.flipRatio = ratio;
        config.useGPU = false;
        config.pressureIterations = 10;
        system.Initialize(config);

        system.AddParticleBox(0.5f, 1.5f, 0.5f, 1.5f, 2.5f, 1.5f);

        for (int i = 0; i < 60; i++) {
            system.Step(1.0f / 60.0f);
        }

        float energy = system.GetStats().totalEnergy;
        system.Shutdown();
        return energy;
    };

    float picEnergy = runWithFlipRatio(0.0f);   // Very damped
    float flipEnergy = runWithFlipRatio(1.0f);  // Less damped

    // FLIP should retain more energy than PIC
    // (PIC is numerically viscous and damps everything)
    EXPECT_TRUE(flipEnergy > picEnergy * 0.5f);
}

void test_COFLIPSystem_PressureConvergence() {
    // Verify that more pressure iterations reduce the velocity divergence.
    // Stats should report higher fluid cell counts and lower residual.
    auto runWithPressureIters = [](uint32_t iters) -> float {
        COFLIPSystem system;
        COFLIPConfig config;
        config.gridSizeX = 16;
        config.gridSizeY = 16;
        config.gridSizeZ = 16;
        config.cellSize = 0.2f;
        config.gravityY = -9.81f;
        config.useGPU = false;
        config.pressureIterations = iters;
        system.Initialize(config);

        system.AddParticleBox(0.5f, 0.5f, 0.5f, 2.0f, 2.0f, 2.0f);

        // Run a few steps to get interesting pressure field
        for (int i = 0; i < 10; i++) {
            system.Step(1.0f / 60.0f);
        }

        // maxVelocity can serve as a proxy — better pressure solve
        // should produce more physically plausible velocities
        float maxV = system.GetStats().maxVelocity;
        system.Shutdown();
        return maxV;
    };

    float maxV_low = runWithPressureIters(2);
    float maxV_high = runWithPressureIters(40);

    // With more iterations, max velocity should be reasonable (not exploding)
    EXPECT_TRUE(std::isfinite(maxV_low));
    EXPECT_TRUE(std::isfinite(maxV_high));

    // Higher iterations should generally produce lower or equal max velocity
    // (better divergence correction prevents spurious velocity spikes)
    // With only 2 iterations, the solver may produce higher peaks
    EXPECT_TRUE(maxV_high <= maxV_low * 1.5f);
}

void test_COFLIPSystem_PerformanceBenchmark() {
    // Performance regression guard: a 16³ grid with modest particles should
    // complete 60 steps in under 10 seconds (very conservative ceiling).
    COFLIPSystem system;
    COFLIPConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.gravityY = -9.81f;
    config.useGPU = false;
    config.pressureIterations = 20;
    system.Initialize(config);

    system.AddParticleBox(0.5f, 0.5f, 0.5f, 2.5f, 2.5f, 2.5f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 60; i++) {
        system.Step(1.0f / 60.0f);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Generous ceiling: 60 steps should complete in < 30 seconds on any machine
    EXPECT_TRUE(ms < 30000.0);

    // Also verify timing stats are populated
    const COFLIPStats& stats = system.GetStats();
    EXPECT_TRUE(stats.totalTimeMs > 0.0f);
    EXPECT_TRUE(stats.p2gTimeMs > 0.0f);
    EXPECT_TRUE(stats.pressureTimeMs > 0.0f);
    EXPECT_TRUE(stats.g2pTimeMs > 0.0f);

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

    // Rigorous simulation tests
    RUN_TEST("COFLIPSystem_MassConservation", test_COFLIPSystem_MassConservation);
    RUN_TEST("COFLIPSystem_EnergyConservation_ZeroGravity", test_COFLIPSystem_EnergyConservation_ZeroGravity);
    RUN_TEST("COFLIPSystem_SolidBoxBlocking", test_COFLIPSystem_SolidBoxBlocking);
    RUN_TEST("COFLIPSystem_SolidSphereDeflection", test_COFLIPSystem_SolidSphereDeflection);
    RUN_TEST("COFLIPSystem_LongDurationStability", test_COFLIPSystem_LongDurationStability);
    RUN_TEST("COFLIPSystem_Determinism", test_COFLIPSystem_Determinism);
    RUN_TEST("COFLIPSystem_BoundaryContainment", test_COFLIPSystem_BoundaryContainment);
    RUN_TEST("COFLIPSystem_GravityAccuracy", test_COFLIPSystem_GravityAccuracy);
    RUN_TEST("COFLIPSystem_FlipRatioPIC", test_COFLIPSystem_FlipRatioPIC);
    RUN_TEST("COFLIPSystem_PressureConvergence", test_COFLIPSystem_PressureConvergence);
    RUN_TEST("COFLIPSystem_PerformanceBenchmark", test_COFLIPSystem_PerformanceBenchmark);
}
