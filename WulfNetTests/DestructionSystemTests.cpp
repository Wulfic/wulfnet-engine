// =============================================================================
// WulfNet Engine - Destruction System Tests
// =============================================================================
// Tests for the Voronoi pre-fracture destruction physics system.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Physics/Destruction/DestructionSystem.h>
#include <cmath>
#include <vector>

using namespace WulfNet;

// =============================================================================
// Config Defaults
// =============================================================================

void TestDestruction_ConfigDefaults() {
    DestructionConfig config;
    EXPECT_EQ(config.defaultCellCount, 8u);
    EXPECT_EQ(config.maxCellCount, 64u);
    EXPECT_NEAR(config.minFragmentMass, 0.1f, 1e-6f);
    EXPECT_NEAR(config.fragmentEjectionSpeed, 2.0f, 1e-6f);
    EXPECT_NEAR(config.fragmentAngularSpeed, 5.0f, 1e-6f);
    EXPECT_EQ(config.maxFragmentsPerFrame, 100u);
    EXPECT_EQ(config.maxTotalFragments, 5000u);
    EXPECT_NEAR(config.fragmentLifetime, 30.0f, 1e-3f);
    EXPECT_FALSE(config.enableSecondaryFracture);
    EXPECT_NEAR(config.globalImpulseScale, 1.0f, 1e-6f);
}

// =============================================================================
// VoronoiCell Defaults
// =============================================================================

void TestDestruction_VoronoiCellDefaults() {
    VoronoiCell cell;
    EXPECT_NEAR(cell.centerX, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.centerY, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.centerZ, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.volume, 0.0f, 1e-9f);
    EXPECT_NEAR(cell.mass, 0.0f, 1e-9f);
    EXPECT_FALSE(cell.detached);
}

// =============================================================================
// Fracture Pattern Defaults
// =============================================================================

void TestDestruction_FracturePatternDefaults() {
    FracturePattern pattern;
    EXPECT_EQ(pattern.GetCellCount(), 0u);
    EXPECT_NEAR(pattern.totalVolume, 0.0f, 1e-9f);
    EXPECT_NEAR(pattern.density, 1000.0f, 1e-3f);
}

// =============================================================================
// DestructibleBody Defaults
// =============================================================================

void TestDestruction_BodyDefaults() {
    DestructibleBody body;
    EXPECT_NEAR(body.fractureThreshold, 1000.0f, 1e-3f);
    EXPECT_NEAR(body.stressThreshold, 1.0e6f, 1.0f);
    EXPECT_FALSE(body.fractured);
    EXPECT_TRUE(body.enabled);
    EXPECT_EQ(body.fractureLevel, 0u);
    EXPECT_EQ(body.maxFractureLevel, 2u);
    EXPECT_EQ(body.fragmentBodyIds.size(), 0u);
}

// =============================================================================
// Initialization
// =============================================================================

void TestDestruction_InitShutdown() {
    DestructionSystem sys;
    EXPECT_FALSE(sys.IsInitialized());

    DestructionConfig config;
    EXPECT_TRUE(sys.Initialize(config));
    EXPECT_TRUE(sys.IsInitialized());

    sys.Shutdown();
    EXPECT_FALSE(sys.IsInitialized());
}

void TestDestruction_DoubleInit() {
    DestructionSystem sys;
    DestructionConfig config;
    EXPECT_TRUE(sys.Initialize(config));
    EXPECT_FALSE(sys.Initialize(config)); // Should fail
    sys.Shutdown();
}

// =============================================================================
// Destructible CRUD
// =============================================================================

void TestDestruction_AddDestructible() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    JPH::BodyID bodyId(42);
    uint32_t handle = sys.AddDestructible(bodyId, 500.0f, 8);
    EXPECT_EQ(handle, 0u);
    EXPECT_EQ(sys.GetDestructibleCount(), 1u);

    const DestructibleBody* body = sys.GetDestructible(handle);
    EXPECT_TRUE(body != nullptr);
    EXPECT_NEAR(body->fractureThreshold, 500.0f, 1e-3f);
    EXPECT_FALSE(body->fractured);
    EXPECT_TRUE(body->enabled);

    // Pattern should have cells
    EXPECT_EQ(body->pattern.GetCellCount(), 8u);

    sys.Shutdown();
}

void TestDestruction_RemoveDestructible() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    JPH::BodyID bodyId(1);
    uint32_t handle = sys.AddDestructible(bodyId);
    sys.RemoveDestructible(handle);

    const DestructibleBody* body = sys.GetDestructible(handle);
    EXPECT_TRUE(body != nullptr);
    EXPECT_FALSE(body->enabled);

    sys.Shutdown();
}

void TestDestruction_GetOutOfRange() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    EXPECT_TRUE(sys.GetDestructible(999) == nullptr);

    sys.Shutdown();
}

void TestDestruction_MultipleDestructibles() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t h1 = sys.AddDestructible(JPH::BodyID(1), 100.0f);
    uint32_t h2 = sys.AddDestructible(JPH::BodyID(2), 200.0f);
    uint32_t h3 = sys.AddDestructible(JPH::BodyID(3), 300.0f);

    EXPECT_EQ(sys.GetDestructibleCount(), 3u);
    EXPECT_NE(h1, h2);
    EXPECT_NE(h2, h3);

    EXPECT_NEAR(sys.GetDestructible(h1)->fractureThreshold, 100.0f, 1e-3f);
    EXPECT_NEAR(sys.GetDestructible(h2)->fractureThreshold, 200.0f, 1e-3f);
    EXPECT_NEAR(sys.GetDestructible(h3)->fractureThreshold, 300.0f, 1e-3f);

    sys.Shutdown();
}

// =============================================================================
// Box Pattern Generation
// =============================================================================

void TestDestruction_GenerateBoxPattern() {
    FracturePattern pattern = DestructionSystem::GenerateBoxPattern(
        0.5f, 0.5f, 0.5f, 8, 1000.0f);

    EXPECT_EQ(pattern.GetCellCount(), 8u);
    EXPECT_NEAR(pattern.density, 1000.0f, 1e-3f);
    EXPECT_GT(pattern.totalVolume, 0.0f);

    // Each cell should have positive volume
    for (uint32_t i = 0; i < pattern.GetCellCount(); ++i) {
        EXPECT_GT(pattern.cells[i].volume, 0.0f);
        EXPECT_GT(pattern.cells[i].mass, 0.0f);
    }

    // Total volume should be approximately the box volume (1x1x1 = 1.0)
    EXPECT_NEAR(pattern.totalVolume, 1.0f, 0.2f); // Allow ~20% tolerance for discretization

    // Centers should be within bounds
    for (uint32_t i = 0; i < pattern.GetCellCount(); ++i) {
        EXPECT_GE(pattern.cells[i].centerX, -0.5f);
        EXPECT_LE(pattern.cells[i].centerX, 0.5f);
        EXPECT_GE(pattern.cells[i].centerY, -0.5f);
        EXPECT_LE(pattern.cells[i].centerY, 0.5f);
        EXPECT_GE(pattern.cells[i].centerZ, -0.5f);
        EXPECT_LE(pattern.cells[i].centerZ, 0.5f);
    }
}

void TestDestruction_GenerateBoxPatternDifferentSizes() {
    FracturePattern p4 = DestructionSystem::GenerateBoxPattern(0.5f, 0.5f, 0.5f, 4);
    FracturePattern p16 = DestructionSystem::GenerateBoxPattern(0.5f, 0.5f, 0.5f, 16);

    EXPECT_EQ(p4.GetCellCount(), 4u);
    EXPECT_EQ(p16.GetCellCount(), 16u);

    // Both should cover the same total volume
    EXPECT_NEAR(p4.totalVolume, p16.totalVolume, 0.2f);
}

// =============================================================================
// Sphere Pattern Generation
// =============================================================================

void TestDestruction_GenerateSpherePattern() {
    FracturePattern pattern = DestructionSystem::GenerateSpherePattern(
        1.0f, 8, 2000.0f);

    EXPECT_EQ(pattern.GetCellCount(), 8u);
    EXPECT_NEAR(pattern.density, 2000.0f, 1e-3f);
    EXPECT_GT(pattern.totalVolume, 0.0f);

    // Cell centers should be within the sphere (or close to it)
    for (uint32_t i = 0; i < pattern.GetCellCount(); ++i) {
        float dist = std::sqrt(
            pattern.cells[i].centerX * pattern.cells[i].centerX +
            pattern.cells[i].centerY * pattern.cells[i].centerY +
            pattern.cells[i].centerZ * pattern.cells[i].centerZ);
        EXPECT_LE(dist, 1.0f); // Should be inside the sphere
    }
}

// =============================================================================
// Impact Evaluation
// =============================================================================

void TestDestruction_EvaluateImpactBelowThreshold() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 1000.0f);

    // Impact below threshold should not fracture
    bool fractured = sys.EvaluateImpact(h, 0.0f, 0.0f, 0.0f, 500.0f);
    EXPECT_FALSE(fractured);
    EXPECT_FALSE(sys.GetDestructible(h)->fractured);

    sys.Shutdown();
}

void TestDestruction_EvaluateImpactAboveThreshold() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 1000.0f, 4);

    // Impact above threshold should fracture
    bool fractured = sys.EvaluateImpact(h, 1.0f, 2.0f, 3.0f, 1500.0f);
    EXPECT_TRUE(fractured);
    EXPECT_TRUE(sys.GetDestructible(h)->fractured);

    // Fragments should have been generated
    EXPECT_GT(sys.GetDestructible(h)->fragmentBodyIds.size(), 0u);

    sys.Shutdown();
}

void TestDestruction_EvaluateImpactExactThreshold() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 1000.0f, 4);

    // Impact exactly at threshold should fracture
    bool fractured = sys.EvaluateImpact(h, 0.0f, 0.0f, 0.0f, 1000.0f);
    EXPECT_TRUE(fractured);

    sys.Shutdown();
}

void TestDestruction_EvaluateImpactAlreadyFractured() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 500.0f, 4);

    // First impact fractures
    sys.EvaluateImpact(h, 0, 0, 0, 600.0f);
    EXPECT_TRUE(sys.GetDestructible(h)->fractured);

    // Second impact should not re-fracture
    bool result = sys.EvaluateImpact(h, 0, 0, 0, 600.0f);
    EXPECT_FALSE(result);

    sys.Shutdown();
}

void TestDestruction_EvaluateImpactDisabledBody() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 500.0f, 4);
    sys.RemoveDestructible(h); // Disables

    bool result = sys.EvaluateImpact(h, 0, 0, 0, 1000.0f);
    EXPECT_FALSE(result);

    sys.Shutdown();
}

// =============================================================================
// Global Impulse Scale
// =============================================================================

void TestDestruction_GlobalImpulseScale() {
    DestructionSystem sys;
    DestructionConfig config;
    config.globalImpulseScale = 2.0f;
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 1000.0f, 4);

    // 600 * 2.0 = 1200 > 1000 threshold
    bool fractured = sys.EvaluateImpact(h, 0, 0, 0, 600.0f);
    EXPECT_TRUE(fractured);

    sys.Shutdown();
}

// =============================================================================
// Fracture Execution
// =============================================================================

void TestDestruction_ManualFracture() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 1000.0f, 8);

    uint32_t fragCount = sys.Fracture(h, 0.0f, 0.0f, 0.0f);
    EXPECT_GT(fragCount, 0u);
    EXPECT_TRUE(sys.GetDestructible(h)->fractured);
    EXPECT_EQ(sys.GetDestructible(h)->fragmentBodyIds.size(),
              static_cast<size_t>(fragCount));

    sys.Shutdown();
}

void TestDestruction_FractureInvalidHandle() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t fragCount = sys.Fracture(999, 0.0f, 0.0f, 0.0f);
    EXPECT_EQ(fragCount, 0u);

    sys.Shutdown();
}

void TestDestruction_FractureAlreadyFractured() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 1000.0f, 4);
    sys.Fracture(h, 0, 0, 0);
    EXPECT_TRUE(sys.GetDestructible(h)->fractured);

    // Second fracture should return 0
    uint32_t result = sys.Fracture(h, 0, 0, 0);
    EXPECT_EQ(result, 0u);

    sys.Shutdown();
}

// =============================================================================
// Fragment Tracking
// =============================================================================

void TestDestruction_FragmentTracking() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 1000.0f, 6);
    EXPECT_EQ(sys.GetActiveFragments().size(), 0u);

    sys.Fracture(h, 0, 0, 0);

    // Active fragments should be tracked globally
    EXPECT_GT(sys.GetActiveFragments().size(), 0u);

    // Fragment count should match body's fragment list
    EXPECT_EQ(sys.GetActiveFragments().size(),
              sys.GetDestructible(h)->fragmentBodyIds.size());

    sys.Shutdown();
}

// =============================================================================
// Fracture Callback
// =============================================================================

void TestDestruction_FractureCallback() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    bool callbackFired = false;
    uint32_t cbFragCount = 0;
    uint32_t cbIndex = UINT32_MAX;

    sys.SetFractureCallback([&](const FractureEvent& evt) {
        callbackFired = true;
        cbFragCount = evt.fragmentCount;
        cbIndex = evt.destructibleIndex;
    });

    uint32_t h = sys.AddDestructible(JPH::BodyID(7), 100.0f, 4);
    sys.Fracture(h, 1.0f, 2.0f, 3.0f);

    EXPECT_TRUE(callbackFired);
    EXPECT_GT(cbFragCount, 0u);
    EXPECT_EQ(cbIndex, h);

    sys.Shutdown();
}

// =============================================================================
// Statistics
// =============================================================================

void TestDestruction_StatsUpdated() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    sys.AddDestructible(JPH::BodyID(1), 1000.0f, 4);
    sys.AddDestructible(JPH::BodyID(2), 1000.0f, 4);

    sys.Step(0.016f);

    const auto& stats = sys.GetStats();
    EXPECT_EQ(stats.totalDestructibles, 2u);
    EXPECT_EQ(stats.fracturedBodies, 0u);

    // Fracture one
    sys.Fracture(0, 0, 0, 0);
    sys.Step(0.016f);

    const auto& stats2 = sys.GetStats();
    EXPECT_EQ(stats2.fracturedBodies, 1u);
    EXPECT_GT(stats2.activeFragments, 0u);

    sys.Shutdown();
}

// =============================================================================
// MinFragmentMass Filter
// =============================================================================

void TestDestruction_MinFragmentMassFilter() {
    DestructionSystem sys;
    DestructionConfig config;
    config.minFragmentMass = 1e30f; // Impossibly high — no fragments should pass
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 100.0f, 4);
    uint32_t fragCount = sys.Fracture(h, 0, 0, 0);

    // No fragments should be generated because all are below min mass
    EXPECT_EQ(fragCount, 0u);
    // But the body is still marked fractured
    EXPECT_TRUE(sys.GetDestructible(h)->fractured);

    sys.Shutdown();
}

// =============================================================================
// Config Get/Set
// =============================================================================

void TestDestruction_ConfigGetSet() {
    DestructionSystem sys;
    DestructionConfig config;
    config.defaultCellCount = 12;
    sys.Initialize(config);

    EXPECT_EQ(sys.GetConfig().defaultCellCount, 12u);

    DestructionConfig newConfig = sys.GetConfig();
    newConfig.defaultCellCount = 20;
    sys.SetConfig(newConfig);
    EXPECT_EQ(sys.GetConfig().defaultCellCount, 20u);

    sys.Shutdown();
}

// =============================================================================
// Uninitialized Safety
// =============================================================================

void TestDestruction_UninitializedSafety() {
    DestructionSystem sys;

    // AddDestructible on uninitialized should return UINT32_MAX
    uint32_t h = sys.AddDestructible(JPH::BodyID(1));
    EXPECT_EQ(h, UINT32_MAX);

    // EvaluateImpact on uninitialized
    bool result = sys.EvaluateImpact(0, 0, 0, 0, 1000.0f);
    EXPECT_FALSE(result);

    // Fracture on uninitialized
    uint32_t fragCount = sys.Fracture(0, 0, 0, 0);
    EXPECT_EQ(fragCount, 0u);

    // Step on uninitialized should not crash
    sys.Step(0.016f);
}

// =============================================================================
// Max Fracture Level
// =============================================================================

void TestDestruction_MaxFractureLevel() {
    DestructionSystem sys;
    DestructionConfig config;
    sys.Initialize(config);

    uint32_t h = sys.AddDestructible(JPH::BodyID(1), 100.0f, 4);

    // Set fracture level to max
    DestructibleBody* body = sys.GetDestructible(h);
    EXPECT_TRUE(body != nullptr);
    body->fractureLevel = body->maxFractureLevel;

    // Should not fracture because max level reached
    uint32_t fragCount = sys.Fracture(h, 0, 0, 0);
    EXPECT_EQ(fragCount, 0u);
    EXPECT_FALSE(body->fractured);

    sys.Shutdown();
}

// =============================================================================
// Registration Function
// =============================================================================

void RegisterDestructionSystemTests() {
    RUN_TEST("Destruction_ConfigDefaults", TestDestruction_ConfigDefaults);
    RUN_TEST("Destruction_VoronoiCellDefaults", TestDestruction_VoronoiCellDefaults);
    RUN_TEST("Destruction_FracturePatternDefaults", TestDestruction_FracturePatternDefaults);
    RUN_TEST("Destruction_BodyDefaults", TestDestruction_BodyDefaults);
    RUN_TEST("Destruction_InitShutdown", TestDestruction_InitShutdown);
    RUN_TEST("Destruction_DoubleInit", TestDestruction_DoubleInit);
    RUN_TEST("Destruction_AddDestructible", TestDestruction_AddDestructible);
    RUN_TEST("Destruction_RemoveDestructible", TestDestruction_RemoveDestructible);
    RUN_TEST("Destruction_GetOutOfRange", TestDestruction_GetOutOfRange);
    RUN_TEST("Destruction_MultipleDestructibles", TestDestruction_MultipleDestructibles);
    RUN_TEST("Destruction_GenerateBoxPattern", TestDestruction_GenerateBoxPattern);
    RUN_TEST("Destruction_GenerateBoxPatternSizes", TestDestruction_GenerateBoxPatternDifferentSizes);
    RUN_TEST("Destruction_GenerateSpherePattern", TestDestruction_GenerateSpherePattern);
    RUN_TEST("Destruction_ImpactBelowThreshold", TestDestruction_EvaluateImpactBelowThreshold);
    RUN_TEST("Destruction_ImpactAboveThreshold", TestDestruction_EvaluateImpactAboveThreshold);
    RUN_TEST("Destruction_ImpactExactThreshold", TestDestruction_EvaluateImpactExactThreshold);
    RUN_TEST("Destruction_ImpactAlreadyFractured", TestDestruction_EvaluateImpactAlreadyFractured);
    RUN_TEST("Destruction_ImpactDisabledBody", TestDestruction_EvaluateImpactDisabledBody);
    RUN_TEST("Destruction_GlobalImpulseScale", TestDestruction_GlobalImpulseScale);
    RUN_TEST("Destruction_ManualFracture", TestDestruction_ManualFracture);
    RUN_TEST("Destruction_FractureInvalidHandle", TestDestruction_FractureInvalidHandle);
    RUN_TEST("Destruction_FractureAlreadyFractured", TestDestruction_FractureAlreadyFractured);
    RUN_TEST("Destruction_FragmentTracking", TestDestruction_FragmentTracking);
    RUN_TEST("Destruction_FractureCallback", TestDestruction_FractureCallback);
    RUN_TEST("Destruction_StatsUpdated", TestDestruction_StatsUpdated);
    RUN_TEST("Destruction_MinFragmentMassFilter", TestDestruction_MinFragmentMassFilter);
    RUN_TEST("Destruction_ConfigGetSet", TestDestruction_ConfigGetSet);
    RUN_TEST("Destruction_UninitializedSafety", TestDestruction_UninitializedSafety);
    RUN_TEST("Destruction_MaxFractureLevel", TestDestruction_MaxFractureLevel);
}
