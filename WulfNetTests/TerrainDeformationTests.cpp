// =============================================================================
// WulfNet Engine - Terrain Deformation Tests
// =============================================================================

#include "TestHarness.h"
#include "WulfNet/Physics/Terrain/TerrainDeformation.h"
#include <cmath>
#include <cstring>

using namespace WulfNet;

// Helper: create a config with gridSize cells and cellSize so that
// worldExtent = gridSize * cellSize
static TerrainDeformConfig MakeConfig(uint32_t gridSize, float cellSize) {
    TerrainDeformConfig cfg;
    cfg.gridSizeX = gridSize;
    cfg.gridSizeZ = gridSize;
    cfg.cellSize = cellSize;
    cfg.originX = 0.0f;
    cfg.originZ = 0.0f;
    return cfg;
}

// =============================================================================
// Configuration and Initialization Tests
// =============================================================================

void test_TerrainDeform_DefaultConfig() {
    TerrainDeformConfig config;
    EXPECT_TRUE(config.gridSizeX > 0);
    EXPECT_TRUE(config.gridSizeZ > 0);
    EXPECT_TRUE(config.cellSize > 0.0f);
    EXPECT_TRUE(config.maxDeformDepth > 0.0f);
    EXPECT_TRUE(config.maxHistorySize > 0);
}

void test_TerrainDeform_InitShutdown() {
    TerrainDeformation terrain;
    auto config = MakeConfig(32, 2.0f);

    EXPECT_TRUE(terrain.Initialize(config));

    auto stats = terrain.GetStats();
    EXPECT_EQ(stats.cellsModified, 0u);
    EXPECT_EQ(stats.totalDeformations, 0u);

    terrain.Shutdown();
}

void test_TerrainDeform_DoubleInit() {
    TerrainDeformation terrain;
    auto config = MakeConfig(16, 1.0f);

    EXPECT_TRUE(terrain.Initialize(config));
    // Second init should also succeed (reinitialize)
    EXPECT_TRUE(terrain.Initialize(config));

    terrain.Shutdown();
}

// =============================================================================
// Height Field Tests
// =============================================================================

void test_TerrainDeform_FlatTerrain() {
    TerrainDeformation terrain;
    auto config = MakeConfig(16, 2.0f);   // 32 world units
    terrain.Initialize(config);

    // Default height should be 0
    float h = terrain.SampleHeight(16.0f, 16.0f);
    EXPECT_NEAR(h, 0.0f, 1e-5f);

    terrain.Shutdown();
}

void test_TerrainDeform_SetGetHeight() {
    TerrainDeformation terrain;
    auto config = MakeConfig(16, 2.0f);
    terrain.Initialize(config);

    // Set height at a cell and verify
    terrain.SetHeightAt(8, 8, 5.0f);
    float h = terrain.GetHeightAt(8, 8);
    EXPECT_NEAR(h, 5.0f, 1e-5f);

    terrain.Shutdown();
}

void test_TerrainDeform_BilinearInterpolation() {
    TerrainDeformation terrain;
    auto config = MakeConfig(4, 1.0f);    // 4 world units
    terrain.Initialize(config);

    // Set corner heights
    terrain.SetHeightAt(0, 0, 0.0f);
    terrain.SetHeightAt(1, 0, 4.0f);
    terrain.SetHeightAt(0, 1, 0.0f);
    terrain.SetHeightAt(1, 1, 4.0f);

    // Sample at midpoint between cell 0 and cell 1 in X
    float midX = 0.5f;   // Half a cell
    float h = terrain.SampleHeight(midX, 0.0f);
    // Should be approximately 2.0 (halfway between 0 and 4)
    EXPECT_NEAR(h, 2.0f, 0.5f);

    terrain.Shutdown();
}

void test_TerrainDeform_OutOfBoundsSample() {
    TerrainDeformation terrain;
    auto config = MakeConfig(8, 2.0f);
    terrain.Initialize(config);

    // Out of bounds should clamp and return something finite
    float h = terrain.SampleHeight(-100.0f, -100.0f);
    EXPECT_TRUE(std::isfinite(h));

    h = terrain.SampleHeight(1000.0f, 1000.0f);
    EXPECT_TRUE(std::isfinite(h));

    terrain.Shutdown();
}

// =============================================================================
// Material Tests
// =============================================================================

void test_TerrainDeform_DefaultMaterial() {
    TerrainDeformation terrain;
    auto config = MakeConfig(8, 1.0f);
    terrain.Initialize(config);

    const TerrainMaterial& mat = terrain.GetMaterial(4, 4);
    EXPECT_TRUE(mat.hardness > 0.0f);
    EXPECT_TRUE(mat.displacementScale > 0.0f);

    terrain.Shutdown();
}

void test_TerrainDeform_SetMaterial() {
    TerrainDeformation terrain;
    auto config = MakeConfig(8, 1.0f);
    terrain.Initialize(config);

    terrain.SetMaterial(4, 4, TerrainMaterial::Sand());
    const TerrainMaterial& mat = terrain.GetMaterial(4, 4);
    EXPECT_TRUE(mat.type == TerrainMaterialType::Sand);

    terrain.Shutdown();
}

void test_TerrainDeform_SetMaterialRegion() {
    TerrainDeformation terrain;
    auto config = MakeConfig(16, 1.0f);
    terrain.Initialize(config);

    terrain.SetMaterialRegion(2, 2, 6, 6, TerrainMaterial::Mud());

    // Check center of region
    const TerrainMaterial& mat = terrain.GetMaterial(4, 4);
    EXPECT_TRUE(mat.type == TerrainMaterialType::Mud);

    // Check outside region
    const TerrainMaterial& matOut = terrain.GetMaterial(0, 0);
    EXPECT_TRUE(matOut.type != TerrainMaterialType::Mud);

    terrain.Shutdown();
}

// =============================================================================
// Stamp Application Tests
// =============================================================================

void test_TerrainDeform_CircleStamp() {
    TerrainDeformation terrain;
    auto config = MakeConfig(32, 2.0f);   // 64 world units
    terrain.Initialize(config);

    DeformationStamp stamp;
    stamp.shape = StampShape::Circle;
    stamp.worldX = 32.0f;
    stamp.worldZ = 32.0f;
    stamp.radius = 8.0f;
    stamp.depth = 2.0f;
    stamp.falloffExponent = 1.0f;

    terrain.ApplyStamp(stamp);

    // Center should be deformed downward
    float hCenter = terrain.SampleHeight(32.0f, 32.0f);
    EXPECT_TRUE(hCenter < 0.0f);

    // Far away should be unaffected
    float hFar = terrain.SampleHeight(0.0f, 0.0f);
    EXPECT_NEAR(hFar, 0.0f, 0.5f);

    auto stats = terrain.GetStats();
    EXPECT_TRUE(stats.totalDeformations > 0);

    terrain.Shutdown();
}

void test_TerrainDeform_RectangleStamp() {
    TerrainDeformation terrain;
    auto config = MakeConfig(32, 2.0f);
    terrain.Initialize(config);

    DeformationStamp stamp;
    stamp.shape = StampShape::Rectangle;
    stamp.worldX = 32.0f;
    stamp.worldZ = 32.0f;
    stamp.width = 8.0f;       // full width
    stamp.length = 8.0f;      // full length
    stamp.depth = 1.5f;
    stamp.falloffExponent = 0.5f;

    terrain.ApplyStamp(stamp);

    float hCenter = terrain.SampleHeight(32.0f, 32.0f);
    EXPECT_TRUE(hCenter < 0.0f);

    terrain.Shutdown();
}

void test_TerrainDeform_StampDepthLimit() {
    TerrainDeformation terrain;
    TerrainDeformConfig config = MakeConfig(16, 2.0f);
    config.maxDeformDepth = 5.0f;
    terrain.Initialize(config);

    // Apply a very deep stamp
    DeformationStamp stamp;
    stamp.shape = StampShape::Circle;
    stamp.worldX = 16.0f;
    stamp.worldZ = 16.0f;
    stamp.radius = 4.0f;
    stamp.depth = 100.0f;  // Way beyond limit
    stamp.falloffExponent = 0.0f;

    terrain.ApplyStamp(stamp);

    float h = terrain.SampleHeight(16.0f, 16.0f);
    // Should be clamped to maxDeformDepth
    EXPECT_TRUE(h >= -config.maxDeformDepth - 1.0f);

    terrain.Shutdown();
}

// =============================================================================
// Explosion Tests
// =============================================================================

void test_TerrainDeform_Explosion() {
    TerrainDeformation terrain;
    auto config = MakeConfig(32, 2.0f);
    terrain.Initialize(config);

    terrain.ApplyExplosion(32.0f, 32.0f, 10.0f, 3.0f);

    // Center should be cratered (lowered)
    float hCenter = terrain.SampleHeight(32.0f, 32.0f);
    EXPECT_TRUE(hCenter < 0.0f);

    // Rim should be raised (volume conservation)
    bool foundRaisedRim = false;
    for (float r = 10.0f; r < 15.0f; r += 0.5f) {
        float h = terrain.SampleHeight(32.0f + r, 32.0f);
        if (h > 0.0f) {
            foundRaisedRim = true;
            break;
        }
    }
    EXPECT_TRUE(foundRaisedRim);

    terrain.Shutdown();
}

void test_TerrainDeform_ExplosionOnRock() {
    TerrainDeformation terrain;
    auto config = MakeConfig(32, 2.0f);
    terrain.Initialize(config);

    // Set center to rock (hard material)
    terrain.SetMaterialRegion(12, 12, 20, 20, TerrainMaterial::Rock());

    terrain.ApplyExplosion(32.0f, 32.0f, 8.0f, 3.0f);
    float hRock = terrain.SampleHeight(32.0f, 32.0f);

    // Reset and try on sand
    terrain.Reset();
    terrain.SetMaterialRegion(12, 12, 20, 20, TerrainMaterial::Sand());
    terrain.ApplyExplosion(32.0f, 32.0f, 8.0f, 3.0f);
    float hSand = terrain.SampleHeight(32.0f, 32.0f);

    // Sand should deform more than rock (lower height = deeper)
    EXPECT_TRUE(hSand < hRock);

    terrain.Shutdown();
}

// =============================================================================
// Tire Track & Footprint Tests
// =============================================================================

void test_TerrainDeform_TireTrack() {
    TerrainDeformation terrain;
    auto config = MakeConfig(64, 2.0f);   // 128 world units
    terrain.Initialize(config);

    terrain.ApplyTireTrack(
        32.0f, 64.0f,   // start
        96.0f, 64.0f,    // end
        2.0f,            // width
        0.3f,            // depth
        0.5f             // tread spacing
    );

    // Sample along the track
    float hMid = terrain.SampleHeight(64.0f, 64.0f);
    EXPECT_TRUE(hMid < 0.0f);  // Track should be depressed

    auto stats = terrain.GetStats();
    EXPECT_TRUE(stats.totalDeformations > 0);

    terrain.Shutdown();
}

void test_TerrainDeform_Footprint() {
    TerrainDeformation terrain;
    auto config = MakeConfig(32, 2.0f);
    terrain.Initialize(config);

    // ApplyFootprint(worldX, worldZ, rotation, footLength, footWidth, depth)
    terrain.ApplyFootprint(32.0f, 32.0f, 0.0f, 1.0f, 0.5f, 0.3f);

    float h = terrain.SampleHeight(32.0f, 32.0f);
    EXPECT_TRUE(h < 0.0f);

    terrain.Shutdown();
}

// =============================================================================
// Undo & Reset Tests
// =============================================================================

void test_TerrainDeform_Undo() {
    TerrainDeformation terrain;
    auto config = MakeConfig(16, 2.0f);
    terrain.Initialize(config);

    float hBefore = terrain.SampleHeight(16.0f, 16.0f);

    DeformationStamp stamp;
    stamp.shape = StampShape::Circle;
    stamp.worldX = 16.0f;
    stamp.worldZ = 16.0f;
    stamp.radius = 4.0f;
    stamp.depth = 2.0f;
    stamp.falloffExponent = 0.5f;

    terrain.ApplyStamp(stamp);

    float hAfter = terrain.SampleHeight(16.0f, 16.0f);
    EXPECT_TRUE(hAfter != hBefore);

    terrain.Undo();

    float hUndo = terrain.SampleHeight(16.0f, 16.0f);
    EXPECT_NEAR(hUndo, hBefore, 0.1f);

    terrain.Shutdown();
}

void test_TerrainDeform_Reset() {
    TerrainDeformation terrain;
    auto config = MakeConfig(16, 2.0f);
    terrain.Initialize(config);

    // Make several deformations
    for (int i = 0; i < 5; ++i) {
        DeformationStamp stamp;
        stamp.shape = StampShape::Circle;
        stamp.worldX = 16.0f;
        stamp.worldZ = 16.0f;
        stamp.radius = 3.0f;
        stamp.depth = 1.0f;
        stamp.falloffExponent = 0.5f;
        terrain.ApplyStamp(stamp);
    }

    float hDeformed = terrain.SampleHeight(16.0f, 16.0f);
    EXPECT_TRUE(hDeformed != 0.0f);

    terrain.Reset();

    float hReset = terrain.SampleHeight(16.0f, 16.0f);
    EXPECT_NEAR(hReset, 0.0f, 1e-5f);

    auto stats = terrain.GetStats();
    EXPECT_EQ(stats.totalDeformations, 0u);

    terrain.Shutdown();
}

// =============================================================================
// Dirty Region Tests
// =============================================================================

void test_TerrainDeform_DirtyRegions() {
    TerrainDeformation terrain;
    auto config = MakeConfig(32, 2.0f);
    terrain.Initialize(config);

    // Initially no dirty region
    EXPECT_TRUE(!terrain.HasDirtyRegion());

    DeformationStamp stamp;
    stamp.shape = StampShape::Circle;
    stamp.worldX = 32.0f;
    stamp.worldZ = 32.0f;
    stamp.radius = 4.0f;
    stamp.depth = 1.0f;
    stamp.falloffExponent = 0.5f;
    terrain.ApplyStamp(stamp);

    EXPECT_TRUE(terrain.HasDirtyRegion());

    terrain.ClearDirty();

    EXPECT_TRUE(!terrain.HasDirtyRegion());

    terrain.Shutdown();
}

// =============================================================================
// Normal Computation Tests
// =============================================================================

void test_TerrainDeform_FlatNormal() {
    TerrainDeformation terrain;
    auto config = MakeConfig(16, 2.0f);
    terrain.Initialize(config);

    float nx, ny, nz;
    terrain.ComputeNormal(8, 8, nx, ny, nz);

    // Flat terrain normal should be (0, 1, 0)
    EXPECT_NEAR(nx, 0.0f, 0.01f);
    EXPECT_NEAR(ny, 1.0f, 0.01f);
    EXPECT_NEAR(nz, 0.0f, 0.01f);

    terrain.Shutdown();
}

void test_TerrainDeform_SlopedNormal() {
    TerrainDeformation terrain;
    auto config = MakeConfig(16, 2.0f);
    terrain.Initialize(config);

    // Create a slope in X
    for (uint32_t x = 0; x < 16; ++x) {
        for (uint32_t z = 0; z < 16; ++z) {
            terrain.SetHeightAt(x, z, static_cast<float>(x) * 2.0f);
        }
    }

    float nx, ny, nz;
    terrain.ComputeNormal(8, 8, nx, ny, nz);

    // Normal should tilt in -X direction
    EXPECT_TRUE(nx < 0.0f);
    EXPECT_TRUE(ny > 0.0f);

    // Should be normalized
    float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    EXPECT_NEAR(len, 1.0f, 0.05f);

    terrain.Shutdown();
}

// =============================================================================
// MPM Integration Test
// =============================================================================

void test_TerrainDeform_MPMDeformation() {
    TerrainDeformation terrain;
    auto config = MakeConfig(32, 2.0f);
    terrain.Initialize(config);

    // Set center to soft material
    terrain.SetMaterialRegion(12, 12, 20, 20, TerrainMaterial::SoftSoil());

    float hBefore = terrain.SampleHeight(32.0f, 32.0f);

    // ApplyMPMDeformation(positions, forces, count, dt)
    // Single particle at center pushing down
    float positions[3] = { 32.0f, 0.0f, 32.0f };
    float forces[3] = { 0.0f, -500.0f, 0.0f };
    terrain.ApplyMPMDeformation(positions, forces, 1, 1.0f / 60.0f);

    float hAfter = terrain.SampleHeight(32.0f, 32.0f);

    // Should have deformed downward
    EXPECT_TRUE(hAfter < hBefore);

    terrain.Shutdown();
}

void test_TerrainDeform_MPMOnHardSurface() {
    TerrainDeformation terrain;
    auto config = MakeConfig(32, 2.0f);
    terrain.Initialize(config);

    terrain.SetMaterialRegion(0, 0, 31, 31, TerrainMaterial::Rock());

    float hBeforeRock = terrain.SampleHeight(32.0f, 32.0f);
    float pos[3] = { 32.0f, 0.0f, 32.0f };
    float forces[3] = { 0.0f, -500.0f, 0.0f };
    terrain.ApplyMPMDeformation(pos, forces, 1, 1.0f / 60.0f);
    float hAfterRock = terrain.SampleHeight(32.0f, 32.0f);
    float diffRock = std::abs(hAfterRock - hBeforeRock);

    // Now test soft soil
    terrain.Reset();
    terrain.SetMaterialRegion(0, 0, 31, 31, TerrainMaterial::SoftSoil());
    float hBeforeSoft = terrain.SampleHeight(32.0f, 32.0f);
    terrain.ApplyMPMDeformation(pos, forces, 1, 1.0f / 60.0f);
    float hAfterSoft = terrain.SampleHeight(32.0f, 32.0f);
    float diffSoft = std::abs(hAfterSoft - hBeforeSoft);

    // Soft soil should deform more than rock
    EXPECT_TRUE(diffSoft > diffRock);

    terrain.Shutdown();
}

// =============================================================================
// Edge Cases
// =============================================================================

void test_TerrainDeform_ZeroRadiusStamp() {
    TerrainDeformation terrain;
    auto config = MakeConfig(16, 2.0f);
    terrain.Initialize(config);

    DeformationStamp stamp;
    stamp.shape = StampShape::Circle;
    stamp.worldX = 16.0f;
    stamp.worldZ = 16.0f;
    stamp.radius = 0.0f;
    stamp.depth = 1.0f;
    stamp.falloffExponent = 0.0f;

    // Should not crash
    terrain.ApplyStamp(stamp);

    EXPECT_TRUE(true);

    terrain.Shutdown();
}

void test_TerrainDeform_LargeResolution() {
    TerrainDeformation terrain;
    auto config = MakeConfig(128, 2.0f);   // 256 world units
    terrain.Initialize(config);

    DeformationStamp stamp;
    stamp.shape = StampShape::Circle;
    stamp.worldX = 128.0f;
    stamp.worldZ = 128.0f;
    stamp.radius = 20.0f;
    stamp.depth = 3.0f;
    stamp.falloffExponent = 1.0f;

    terrain.ApplyStamp(stamp);

    float h = terrain.SampleHeight(128.0f, 128.0f);
    EXPECT_TRUE(h < 0.0f);
    EXPECT_TRUE(std::isfinite(h));

    terrain.Shutdown();
}

// =============================================================================
// Registration
// =============================================================================

void RegisterTerrainDeformationTests() {
    // Config & Init
    RUN_TEST("TerrainDeform_DefaultConfig", test_TerrainDeform_DefaultConfig);
    RUN_TEST("TerrainDeform_InitShutdown", test_TerrainDeform_InitShutdown);
    RUN_TEST("TerrainDeform_DoubleInit", test_TerrainDeform_DoubleInit);

    // Height Field
    RUN_TEST("TerrainDeform_FlatTerrain", test_TerrainDeform_FlatTerrain);
    RUN_TEST("TerrainDeform_SetGetHeight", test_TerrainDeform_SetGetHeight);
    RUN_TEST("TerrainDeform_BilinearInterp", test_TerrainDeform_BilinearInterpolation);
    RUN_TEST("TerrainDeform_OutOfBounds", test_TerrainDeform_OutOfBoundsSample);

    // Materials
    RUN_TEST("TerrainDeform_DefaultMaterial", test_TerrainDeform_DefaultMaterial);
    RUN_TEST("TerrainDeform_SetMaterial", test_TerrainDeform_SetMaterial);
    RUN_TEST("TerrainDeform_MaterialRegion", test_TerrainDeform_SetMaterialRegion);

    // Stamps
    RUN_TEST("TerrainDeform_CircleStamp", test_TerrainDeform_CircleStamp);
    RUN_TEST("TerrainDeform_RectStamp", test_TerrainDeform_RectangleStamp);
    RUN_TEST("TerrainDeform_StampDepthLimit", test_TerrainDeform_StampDepthLimit);

    // Explosions
    RUN_TEST("TerrainDeform_Explosion", test_TerrainDeform_Explosion);
    RUN_TEST("TerrainDeform_ExplosionOnRock", test_TerrainDeform_ExplosionOnRock);

    // Tire Tracks & Footprints
    RUN_TEST("TerrainDeform_TireTrack", test_TerrainDeform_TireTrack);
    RUN_TEST("TerrainDeform_Footprint", test_TerrainDeform_Footprint);

    // Undo & Reset
    RUN_TEST("TerrainDeform_Undo", test_TerrainDeform_Undo);
    RUN_TEST("TerrainDeform_Reset", test_TerrainDeform_Reset);

    // Dirty Regions
    RUN_TEST("TerrainDeform_DirtyRegions", test_TerrainDeform_DirtyRegions);

    // Normals
    RUN_TEST("TerrainDeform_FlatNormal", test_TerrainDeform_FlatNormal);
    RUN_TEST("TerrainDeform_SlopedNormal", test_TerrainDeform_SlopedNormal);

    // MPM Integration
    RUN_TEST("TerrainDeform_MPMDeformation", test_TerrainDeform_MPMDeformation);
    RUN_TEST("TerrainDeform_MPMOnHardSurface", test_TerrainDeform_MPMOnHardSurface);

    // Edge Cases
    RUN_TEST("TerrainDeform_ZeroRadius", test_TerrainDeform_ZeroRadiusStamp);
    RUN_TEST("TerrainDeform_LargeResolution", test_TerrainDeform_LargeResolution);
}
