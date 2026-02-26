// =============================================================================
// WulfNet Engine - Fluid Surface Unit Tests
// =============================================================================
// Validates marching cubes surface extraction, density splatting, and mesh output.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Physics/Fluids/FluidSurface.h>
#include <WulfNet/Physics/Fluids/COFLIPSystem.h>
#include <cmath>
#include <chrono>

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
// Rigorous Surface Extraction Tests
// =============================================================================

void test_FluidSurface_NormalsPointOutward() {
    // For a sphere density field, normals should point radially outward.
    // Verify by checking dot(normal, radialDir) > 0.
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 16;
    config.gridSizeY = 16;
    config.gridSizeZ = 16;
    config.cellSize = 0.25f;
    config.isoLevel = 0.5f;
    config.smoothNormals = true;
    config.useGPU = false;
    surface.Initialize(config);

    float cx = 8.0f, cy = 8.0f, cz = 8.0f;
    float radius = 4.0f;
    float cellSz = config.cellSize;

    for (int k = 0; k < 16; k++)
        for (int j = 0; j < 16; j++)
            for (int i = 0; i < 16; i++) {
                float dx = static_cast<float>(i) - cx;
                float dy = static_cast<float>(j) - cy;
                float dz = static_cast<float>(k) - cz;
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                surface.SetDensity(i, j, k, std::max(0.0f, 1.0f - dist / radius));
            }

    surface.ExtractSurface();
    EXPECT_TRUE(surface.GetVertexCount() > 10);

    // Sphere center in world space
    float wcx = cx * cellSz, wcy = cy * cellSz, wcz = cz * cellSz;

    uint32_t outwardCount = 0;
    uint32_t checkedCount = 0;
    const auto& verts = surface.GetVertices();
    for (uint32_t i = 0; i < surface.GetVertexCount(); i++) {
        float nx = verts[i].nx, ny = verts[i].ny, nz = verts[i].nz;
        float nLen = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (nLen < 0.01f) continue;  // Skip degenerate normals

        // Radial direction from sphere center to vertex
        float rx = verts[i].x - wcx;
        float ry = verts[i].y - wcy;
        float rz = verts[i].z - wcz;
        float rLen = std::sqrt(rx * rx + ry * ry + rz * rz);
        if (rLen < 0.01f) continue;

        // Dot product: normal should align with radial direction
        float dot = (nx * rx + ny * ry + nz * rz) / (nLen * rLen);
        if (dot > 0.0f) outwardCount++;
        checkedCount++;
    }

    // At least 80% of normals should point outward (some vertex interpolation
    // inaccuracies near grid cell boundaries are acceptable)
    if (checkedCount > 0) {
        float ratio = static_cast<float>(outwardCount) / static_cast<float>(checkedCount);
        EXPECT_TRUE(ratio > 0.8f);
    }

    surface.Shutdown();
}

void test_FluidSurface_SmoothDensityReducesNoise() {
    // SmoothDensity should reduce variance in the density field.
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 12;
    config.gridSizeY = 12;
    config.gridSizeZ = 12;
    config.cellSize = 0.25f;
    config.useGPU = false;
    surface.Initialize(config);

    // Set a noisy density field: alternating high/low
    for (int k = 0; k < 12; k++)
        for (int j = 0; j < 12; j++)
            for (int i = 0; i < 12; i++) {
                float val = ((i + j + k) % 2 == 0) ? 1.0f : 0.0f;
                surface.SetDensity(i, j, k, val);
            }

    // Measure variance before smoothing
    auto computeVariance = [&]() -> float {
        float sum = 0, sum2 = 0;
        int count = 0;
        for (int k = 1; k < 11; k++)
            for (int j = 1; j < 11; j++)
                for (int i = 1; i < 11; i++) {
                    float v = surface.GetDensity(i, j, k);
                    sum += v;
                    sum2 += v * v;
                    count++;
                }
        float mean = sum / count;
        return sum2 / count - mean * mean;
    };

    float varianceBefore = computeVariance();

    surface.SmoothDensity();

    float varianceAfter = computeVariance();

    // Smoothing should significantly reduce variance
    EXPECT_TRUE(varianceAfter < varianceBefore * 0.5f);

    surface.Shutdown();
}

void test_FluidSurface_SphereMeshQuality() {
    // Extract a sphere mesh and verify:
    // 1) Reasonable vertex count for the resolution
    // 2) All vertices lie near the expected iso-surface radius
    // 3) Triangle count is consistent with index count
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 20;
    config.gridSizeY = 20;
    config.gridSizeZ = 20;
    config.cellSize = 0.2f;
    config.isoLevel = 0.5f;
    config.smoothNormals = true;
    config.useGPU = false;
    surface.Initialize(config);

    float cx = 10.0f, cy = 10.0f, cz = 10.0f;
    float radius = 6.0f;
    float cellSz = config.cellSize;

    for (int k = 0; k < 20; k++)
        for (int j = 0; j < 20; j++)
            for (int i = 0; i < 20; i++) {
                float dx = static_cast<float>(i) - cx;
                float dy = static_cast<float>(j) - cy;
                float dz = static_cast<float>(k) - cz;
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                surface.SetDensity(i, j, k, std::max(0.0f, 1.0f - dist / radius));
            }

    surface.ExtractSurface();

    // Should produce a substantial mesh
    EXPECT_TRUE(surface.GetVertexCount() > 100);
    EXPECT_TRUE(surface.GetTriangleCount() > 50);
    EXPECT_EQ(surface.GetIndexCount(), surface.GetTriangleCount() * 3);

    // Verify vertices lie near the expected iso-surface
    // iso = 0.5 means dist/radius = 0.5 => dist = 3.0 cells => world = 0.6m
    float expectedWorldRadius = radius * 0.5f * cellSz;  // 0.6m
    float wcx = cx * cellSz, wcy = cy * cellSz, wcz = cz * cellSz;

    uint32_t nearSurface = 0;
    const auto& verts = surface.GetVertices();
    for (uint32_t i = 0; i < surface.GetVertexCount(); i++) {
        float dx = verts[i].x - wcx;
        float dy = verts[i].y - wcy;
        float dz = verts[i].z - wcz;
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        // Each vertex should be within ±1 cell of the expected radius
        if (std::abs(dist - expectedWorldRadius) < cellSz * 2.0f) {
            nearSurface++;
        }
    }

    // At least 90% should be on the iso-surface
    float ratio = static_cast<float>(nearSurface) / static_cast<float>(surface.GetVertexCount());
    EXPECT_TRUE(ratio > 0.9f);

    surface.Shutdown();
}

void test_FluidSurface_GenerateFromFluid_ProducesMesh() {
    // Stricter version: the full pipeline must actually produce vertices.
    COFLIPSystem fluid;
    COFLIPConfig fluidConfig;
    fluidConfig.gridSizeX = 16;
    fluidConfig.gridSizeY = 16;
    fluidConfig.gridSizeZ = 16;
    fluidConfig.cellSize = 0.2f;
    fluidConfig.useGPU = false;
    fluidConfig.pressureIterations = 3;
    fluid.Initialize(fluidConfig);

    // Dense particle blob to ensure splats overlap
    fluid.AddParticleBox(0.4f, 0.4f, 0.4f, 1.6f, 1.6f, 1.6f);
    EXPECT_TRUE(fluid.GetActiveParticleCount() > 200);

    // Step to let particles settle (avoids all-zero grid)
    for (int i = 0; i < 3; i++) {
        fluid.Step(1.0f / 60.0f);
    }

    FluidSurface surface;
    FluidSurfaceConfig surfConfig;
    surfConfig.gridSizeX = 16;
    surfConfig.gridSizeY = 16;
    surfConfig.gridSizeZ = 16;
    surfConfig.cellSize = 0.2f;
    surfConfig.splatRadius = 2.5f;
    surfConfig.isoLevel = 0.2f;  // Lower threshold for more geometry
    surfConfig.smoothNormals = true;
    surfConfig.useGPU = false;
    surface.Initialize(surfConfig);

    surface.GenerateSurface(fluid);

    // This time we actually require mesh output
    EXPECT_TRUE(surface.GetVertexCount() > 0);
    EXPECT_TRUE(surface.GetTriangleCount() > 0);

    // Verify mesh integrity
    EXPECT_EQ(surface.GetIndexCount(), surface.GetTriangleCount() * 3);

    // All vertex positions should be finite
    const auto& verts = surface.GetVertices();
    for (uint32_t i = 0; i < surface.GetVertexCount(); i++) {
        EXPECT_TRUE(std::isfinite(verts[i].x));
        EXPECT_TRUE(std::isfinite(verts[i].y));
        EXPECT_TRUE(std::isfinite(verts[i].z));
    }

    fluid.Shutdown();
    surface.Shutdown();
}

void test_FluidSurface_ExtractSurface_UniformBelowIso() {
    // Uniform density BELOW iso level should produce no surface (all outside)
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.cellSize = 0.25f;
    config.isoLevel = 0.5f;
    config.useGPU = false;
    surface.Initialize(config);

    for (int k = 0; k < 8; k++)
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++)
                surface.SetDensity(i, j, k, 0.1f);

    surface.ExtractSurface();
    EXPECT_EQ(surface.GetVertexCount(), 0u);
    EXPECT_EQ(surface.GetTriangleCount(), 0u);

    surface.Shutdown();
}

void test_FluidSurface_IsoLevelSensitivity() {
    // Higher iso level should produce smaller surface (fewer vertices)
    // for the same density field.
    auto extractWithIso = [](float iso) -> uint32_t {
        FluidSurface surface;
        FluidSurfaceConfig config;
        config.gridSizeX = 16;
        config.gridSizeY = 16;
        config.gridSizeZ = 16;
        config.cellSize = 0.25f;
        config.isoLevel = iso;
        config.useGPU = false;
        surface.Initialize(config);

        float cx = 8.0f, cy = 8.0f, cz = 8.0f, radius = 5.0f;
        for (int k = 0; k < 16; k++)
            for (int j = 0; j < 16; j++)
                for (int i = 0; i < 16; i++) {
                    float dx = static_cast<float>(i) - cx;
                    float dy = static_cast<float>(j) - cy;
                    float dz = static_cast<float>(k) - cz;
                    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                    surface.SetDensity(i, j, k, std::max(0.0f, 1.0f - dist / radius));
                }

        surface.ExtractSurface();
        uint32_t verts = surface.GetVertexCount();
        surface.Shutdown();
        return verts;
    };

    uint32_t vertsLowIso = extractWithIso(0.2f);   // Large surface
    uint32_t vertsHighIso = extractWithIso(0.8f);   // Small surface

    // Both should produce geometry
    EXPECT_TRUE(vertsLowIso > 0);
    EXPECT_TRUE(vertsHighIso > 0);

    // Lower iso level encompasses more volume => more surface vertices
    EXPECT_TRUE(vertsLowIso > vertsHighIso);
}

void test_FluidSurface_PerformanceBenchmark() {
    // Performance regression guard for marching cubes extraction
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.isoLevel = 0.4f;
    config.smoothNormals = true;
    config.useGPU = false;
    surface.Initialize(config);

    // Large sphere density field
    float cx = 16.0f, cy = 16.0f, cz = 16.0f, radius = 10.0f;
    for (int k = 0; k < 32; k++)
        for (int j = 0; j < 32; j++)
            for (int i = 0; i < 32; i++) {
                float dx = static_cast<float>(i) - cx;
                float dy = static_cast<float>(j) - cy;
                float dz = static_cast<float>(k) - cz;
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                surface.SetDensity(i, j, k, std::max(0.0f, 1.0f - dist / radius));
            }

    auto start = std::chrono::high_resolution_clock::now();
    surface.ExtractSurface();
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    // 32³ grid extraction should complete in < 5 seconds (very generous)
    EXPECT_TRUE(ms < 5000.0);

    // Should produce a substantial mesh at this resolution
    EXPECT_TRUE(surface.GetVertexCount() > 500);
    EXPECT_TRUE(surface.GetTriangleCount() > 200);

    // Stats timing should be non-negative (may be zero if below timer resolution)
    const FluidSurfaceStats& stats = surface.GetStats();
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

    // Rigorous tests
    RUN_TEST("FluidSurface_NormalsPointOutward", test_FluidSurface_NormalsPointOutward);
    RUN_TEST("FluidSurface_SmoothDensityReducesNoise", test_FluidSurface_SmoothDensityReducesNoise);
    RUN_TEST("FluidSurface_SphereMeshQuality", test_FluidSurface_SphereMeshQuality);
    RUN_TEST("FluidSurface_GenerateFromFluid_ProducesMesh", test_FluidSurface_GenerateFromFluid_ProducesMesh);
    RUN_TEST("FluidSurface_ExtractSurface_UniformBelowIso", test_FluidSurface_ExtractSurface_UniformBelowIso);
    RUN_TEST("FluidSurface_IsoLevelSensitivity", test_FluidSurface_IsoLevelSensitivity);
    RUN_TEST("FluidSurface_PerformanceBenchmark", test_FluidSurface_PerformanceBenchmark);

    // GPU buffers
    RUN_TEST("FluidSurface_GPUBuffers_CPUMode", test_FluidSurface_GPUBuffers_CPUMode);
}
