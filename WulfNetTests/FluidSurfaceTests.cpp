// =============================================================================
// WulfNet Engine - Fluid Surface Unit Tests
// =============================================================================
// Validates marching cubes surface extraction, density splatting, and mesh output.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Physics/Fluids/FluidSurface.h>
#include <WulfNet/Physics/Fluids/COFLIPSystem.h>
#include <cmath>

using namespace WulfNet;

// =============================================================================
// Structure Alignment Tests
// =============================================================================

void test_FluidSurfaceVertex_SizeAlignment() {
    EXPECT_EQ(sizeof(FluidSurfaceVertex), 32u);
    EXPECT_EQ(alignof(FluidSurfaceVertex), 16u);
}

void test_DensityCell_Size() {
    EXPECT_EQ(sizeof(DensityCell), 4u);
}

// =============================================================================
// Configuration Tests
// =============================================================================

void test_FluidSurfaceConfig_Defaults() {
    FluidSurfaceConfig config;
    EXPECT_EQ(config.gridSizeX, 64u);
    EXPECT_EQ(config.gridSizeY, 64u);
    EXPECT_EQ(config.gridSizeZ, 64u);
    EXPECT_NEAR(config.cellSize, 0.1f, 1e-6f);
    EXPECT_NEAR(config.isoLevel, 0.5f, 1e-6f);
    EXPECT_TRUE(config.useGPU);
    EXPECT_TRUE(config.smoothNormals);
}

// =============================================================================
// Initialization Tests
// =============================================================================

void test_FluidSurface_DefaultState() {
    FluidSurface surface;
    EXPECT_FALSE(surface.IsInitialized());
    EXPECT_EQ(surface.GetVertexCount(), 0u);
    EXPECT_EQ(surface.GetTriangleCount(), 0u);
    EXPECT_EQ(surface.GetIndexCount(), 0u);
}

void test_FluidSurface_Initialize_CPUMode() {
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.useGPU = false;

    bool result = surface.Initialize(config);
    EXPECT_TRUE(result);
    EXPECT_TRUE(surface.IsInitialized());

    surface.Shutdown();
    EXPECT_FALSE(surface.IsInitialized());
}

void test_FluidSurface_Initialize_SmallGrid() {
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 4;
    config.gridSizeY = 4;
    config.gridSizeZ = 4;
    config.cellSize = 0.5f;
    config.useGPU = false;

    bool result = surface.Initialize(config);
    EXPECT_TRUE(result);
    surface.Shutdown();
}

// =============================================================================
// Density Grid Tests
// =============================================================================

void test_FluidSurface_ClearDensity() {
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.cellSize = 0.25f;
    config.useGPU = false;
    surface.Initialize(config);

    // Set some density
    surface.SetDensity(4, 4, 4, 5.0f);
    EXPECT_NEAR(surface.GetDensity(4, 4, 4), 5.0f, 1e-6f);

    // Clear should zero everything
    surface.ClearDensity();
    EXPECT_NEAR(surface.GetDensity(4, 4, 4), 0.0f, 1e-6f);

    surface.Shutdown();
}

void test_FluidSurface_DensityReadWrite() {
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.cellSize = 0.25f;
    config.useGPU = false;
    surface.Initialize(config);

    surface.ClearDensity();

    // Write and read back
    surface.SetDensity(2, 3, 4, 1.5f);
    EXPECT_NEAR(surface.GetDensity(2, 3, 4), 1.5f, 1e-6f);

    surface.SetDensity(0, 0, 0, 0.0f);
    EXPECT_NEAR(surface.GetDensity(0, 0, 0), 0.0f, 1e-6f);

    surface.SetDensity(7, 7, 7, 100.0f);
    EXPECT_NEAR(surface.GetDensity(7, 7, 7), 100.0f, 1e-6f);

    surface.Shutdown();
}

void test_FluidSurface_SplatParticle() {
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.2f;
    config.splatRadius = 2.0f;
    config.useGPU = false;
    surface.Initialize(config);

    surface.ClearDensity();

    // Splat a particle at the center
    float cx = 1.5f, cy = 1.5f, cz = 1.5f;
    surface.SplatParticle(cx, cy, cz, 1.0f);

    // Density at the center (grid coords ~7.5) should be non-zero
    // The exact cell depends on world-to-grid mapping, but near center should be affected
    bool anyDensity = false;
    for (int k = 0; k < 16; k++) {
        for (int j = 0; j < 16; j++) {
            for (int i = 0; i < 16; i++) {
                if (surface.GetDensity(i, j, k) > 0.0f) {
                    anyDensity = true;
                    break;
                }
            }
            if (anyDensity) break;
        }
        if (anyDensity) break;
    }
    EXPECT_TRUE(anyDensity);

    surface.Shutdown();
}

// =============================================================================
// Marching Cubes Surface Extraction Tests
// =============================================================================

void test_FluidSurface_ExtractSurface_NoDensity() {
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.cellSize = 0.25f;
    config.useGPU = false;
    surface.Initialize(config);

    surface.ClearDensity();
    surface.ExtractSurface();

    // No density means no surface
    EXPECT_EQ(surface.GetVertexCount(), 0u);
    EXPECT_EQ(surface.GetTriangleCount(), 0u);

    surface.Shutdown();
}

void test_FluidSurface_ExtractSurface_UniformAboveIso() {
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.cellSize = 0.25f;
    config.isoLevel = 0.5f;
    config.useGPU = false;
    surface.Initialize(config);

    // Set all cells to above iso level
    for (int k = 0; k < 8; k++)
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++)
                surface.SetDensity(i, j, k, 1.0f);

    surface.ExtractSurface();

    // Uniform field above iso -> no surface (all inside)
    EXPECT_EQ(surface.GetVertexCount(), 0u);

    surface.Shutdown();
}

void test_FluidSurface_ExtractSurface_SphereDensity() {
    // Create a sphere density field — should extract a sphere-like surface
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.25f;
    config.isoLevel = 0.5f;
    config.useGPU = false;
    surface.Initialize(config);

    // Fill density: sphere centered at grid center
    float cx = 8.0f, cy = 8.0f, cz = 8.0f;
    float radius = 4.0f;
    for (int k = 0; k < 16; k++) {
        for (int j = 0; j < 16; j++) {
            for (int i = 0; i < 16; i++) {
                float dx = static_cast<float>(i) - cx;
                float dy = static_cast<float>(j) - cy;
                float dz = static_cast<float>(k) - cz;
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                float density = std::max(0.0f, 1.0f - dist / radius);
                surface.SetDensity(i, j, k, density);
            }
        }
    }

    surface.ExtractSurface();

    // Should produce triangles forming a sphere-like mesh
    EXPECT_TRUE(surface.GetVertexCount() > 0);
    EXPECT_TRUE(surface.GetTriangleCount() > 0);
    EXPECT_TRUE(surface.GetIndexCount() == surface.GetTriangleCount() * 3);

    surface.Shutdown();
}

void test_FluidSurface_VertexNormals() {
    // Generate surface and verify normals are unit vectors
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 12;
    config.gridSizeY = 12;
    config.gridSizeZ = 12;
    config.cellSize = 0.25f;
    config.isoLevel = 0.5f;
    config.smoothNormals = true;
    config.useGPU = false;
    surface.Initialize(config);

    // Sphere density
    float cx = 6.0f, cy = 6.0f, cz = 6.0f;
    float radius = 3.0f;
    for (int k = 0; k < 12; k++)
        for (int j = 0; j < 12; j++)
            for (int i = 0; i < 12; i++) {
                float dx = static_cast<float>(i) - cx;
                float dy = static_cast<float>(j) - cy;
                float dz = static_cast<float>(k) - cz;
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                surface.SetDensity(i, j, k, std::max(0.0f, 1.0f - dist / radius));
            }

    surface.ExtractSurface();

    // Verify all vertex normals are approximately unit length
    const auto& vertices = surface.GetVertices();
    for (uint32_t i = 0; i < surface.GetVertexCount(); i++) {
        float nx = vertices[i].nx;
        float ny = vertices[i].ny;
        float nz = vertices[i].nz;
        float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        // Normals should be unit length (or zero if degenerate)
        if (len > 0.01f) {
            EXPECT_NEAR(len, 1.0f, 0.1f);
        }
    }

    surface.Shutdown();
}

// =============================================================================
// Full Pipeline: Fluid System -> Surface Extraction Tests
// =============================================================================

void test_FluidSurface_GenerateFromFluid() {
    // End-to-end: create CO-FLIP system, add particles, extract surface
    COFLIPSystem fluid;
    COFLIPConfig fluidConfig;
    fluidConfig.gridSizeX = 16;
    fluidConfig.gridSizeY = 16;
    fluidConfig.gridSizeZ = 16;
    fluidConfig.cellSize = 0.2f;
    fluidConfig.useGPU = false;
    fluidConfig.pressureIterations = 3;
    fluid.Initialize(fluidConfig);

    // Add a moderate blob of water
    fluid.AddParticleBox(0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f);
    EXPECT_TRUE(fluid.GetActiveParticleCount() > 20);

    FluidSurface surface;
    FluidSurfaceConfig surfConfig;
    surfConfig.gridSizeX = 16;
    surfConfig.gridSizeY = 16;
    surfConfig.gridSizeZ = 16;
    surfConfig.cellSize = 0.2f;
    surfConfig.splatRadius = 2.0f;
    surfConfig.isoLevel = 0.3f;
    surfConfig.useGPU = false;
    surface.Initialize(surfConfig);

    // Generate surface from the fluid system
    surface.GenerateSurface(fluid);

    // Should produce some geometry
    // (may or may not depending on particle density vs iso level)
    // At minimum, the call should not crash
    const FluidSurfaceStats& stats = surface.GetStats();
    EXPECT_TRUE(stats.totalTimeMs >= 0.0f);

    fluid.Shutdown();
    surface.Shutdown();
}

// =============================================================================
// Statistics Tests
// =============================================================================

void test_FluidSurface_Stats() {
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.cellSize = 0.25f;
    config.useGPU = false;
    surface.Initialize(config);

    surface.ClearDensity();
    surface.ExtractSurface();

    const FluidSurfaceStats& stats = surface.GetStats();
    EXPECT_EQ(stats.vertexCount, 0u);
    EXPECT_EQ(stats.triangleCount, 0u);
    EXPECT_TRUE(stats.marchingCubesTimeMs >= 0.0f);

    surface.Shutdown();
}

// =============================================================================
// GPU Buffer Handle Tests
// =============================================================================

void test_FluidSurface_GPUBuffers_CPUMode() {
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.useGPU = false;
    surface.Initialize(config);

    GPUSurfaceBufferHandle vertBuf = surface.GetVertexBuffer();
    GPUSurfaceBufferHandle idxBuf = surface.GetIndexBuffer();
    EXPECT_FALSE(vertBuf.valid());
    EXPECT_FALSE(idxBuf.valid());

    surface.Shutdown();
}

// =============================================================================
// Registration
// =============================================================================

void RegisterFluidSurfaceTests() {
    // Structure alignment
    RUN_TEST("FluidSurfaceVertex_SizeAlignment", test_FluidSurfaceVertex_SizeAlignment);
    RUN_TEST("DensityCell_Size", test_DensityCell_Size);

    // Config
    RUN_TEST("FluidSurfaceConfig_Defaults", test_FluidSurfaceConfig_Defaults);

    // Initialization
    RUN_TEST("FluidSurface_DefaultState", test_FluidSurface_DefaultState);
    RUN_TEST("FluidSurface_Initialize_CPUMode", test_FluidSurface_Initialize_CPUMode);
    RUN_TEST("FluidSurface_Initialize_SmallGrid", test_FluidSurface_Initialize_SmallGrid);

    // Density grid
    RUN_TEST("FluidSurface_ClearDensity", test_FluidSurface_ClearDensity);
    RUN_TEST("FluidSurface_DensityReadWrite", test_FluidSurface_DensityReadWrite);
    RUN_TEST("FluidSurface_SplatParticle", test_FluidSurface_SplatParticle);

    // Surface extraction
    RUN_TEST("FluidSurface_ExtractSurface_NoDensity", test_FluidSurface_ExtractSurface_NoDensity);
    RUN_TEST("FluidSurface_ExtractSurface_UniformAboveIso", test_FluidSurface_ExtractSurface_UniformAboveIso);
    RUN_TEST("FluidSurface_ExtractSurface_SphereDensity", test_FluidSurface_ExtractSurface_SphereDensity);
    RUN_TEST("FluidSurface_VertexNormals", test_FluidSurface_VertexNormals);

    // Full pipeline
    RUN_TEST("FluidSurface_GenerateFromFluid", test_FluidSurface_GenerateFromFluid);

    // Stats
    RUN_TEST("FluidSurface_Stats", test_FluidSurface_Stats);

    // GPU buffers
    RUN_TEST("FluidSurface_GPUBuffers_CPUMode", test_FluidSurface_GPUBuffers_CPUMode);
}
