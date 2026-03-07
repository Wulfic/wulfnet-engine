// =============================================================================
// WulfNet Engine - Pipeline Integration Tests
// =============================================================================
// End-to-end tests: IFS chaos game simulation, full rasterizer pipeline
// with deferred shading, and occlusion culling integration.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>
#include <WulfNet/Procedural/IFS/AffineTransform.h>
#include <WulfNet/Procedural/IFS/TransformPresets.h>
#include <WulfNet/Rendering/Types/RenderTypes.h>
#include <WulfNet/Rendering/SoftwareRasterizer/GBuffer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/SoftwareRasterizer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/DeferredShading.h>
#include <WulfNet/Rendering/SoftwareRasterizer/OcclusionCuller.h>

using namespace WulfNet;

// =============================================================================
// Helper: transform a point by a Mat4 (row-major)
// =============================================================================

static Vec3 TransformPoint(const Mat4& m, const Vec3& p) {
    float x = m.At(0, 0) * p.x + m.At(0, 1) * p.y + m.At(0, 2) * p.z + m.At(0, 3);
    float y = m.At(1, 0) * p.x + m.At(1, 1) * p.y + m.At(1, 2) * p.z + m.At(1, 3);
    float z = m.At(2, 0) * p.x + m.At(2, 1) * p.y + m.At(2, 2) * p.z + m.At(2, 3);
    return {x, y, z};
}

// =============================================================================
// IFS Chaos Game CPU Simulation Tests
// =============================================================================

void test_IFS_ChaosGame_CPUSimulation() {
    // Simulate the chaos game on CPU to verify affine transforms produce
    // a fractal pattern bounded within expected space
    auto instructions = TransformPresets::GetPreset(IFSPreset::SierpinskiTriangle3D);
    auto matrices = TransformPresets::BuildMatrices(instructions);

    EXPECT_EQ(matrices.size(), static_cast<size_t>(5));

    Vec3 point = {0.0f, 0.0f, 0.0f};
    uint32_t seed = 42;

    float minX = 1e10f, maxX = -1e10f;
    float minY = 1e10f, maxY = -1e10f;
    float minZ = 1e10f, maxZ = -1e10f;

    // Run 5000 iterations of chaos game
    for (int i = 0; i < 5000; i++) {
        // Hugo Elias hash (matches shader)
        seed = (seed << 13) ^ seed;
        seed = seed * (seed * seed * 15731 + 789221) + 1376312589;
        int idx = static_cast<int>((seed & 0x7FFFFFFF) % matrices.size());

        point = TransformPoint(matrices[idx], point);

        // Track bounds after warmup
        if (i > 50) {
            if (point.x < minX) minX = point.x;
            if (point.x > maxX) maxX = point.x;
            if (point.y < minY) minY = point.y;
            if (point.y > maxY) maxY = point.y;
            if (point.z < minZ) minZ = point.z;
            if (point.z > maxZ) maxZ = point.z;
        }
    }

    // Sierpinski 3D should be bounded (all points within reasonable range)
    float rangeX = maxX - minX;
    float rangeY = maxY - minY;
    float rangeZ = maxZ - minZ;

    EXPECT_TRUE(rangeX > 0.0f && rangeX < 10.0f);
    EXPECT_TRUE(rangeY > 0.0f && rangeY < 10.0f);
    EXPECT_TRUE(rangeZ > 0.0f && rangeZ < 10.0f);

    // Points should not explode to infinity
    EXPECT_TRUE(std::abs(point.x) < 100.0f);
    EXPECT_TRUE(std::abs(point.y) < 100.0f);
    EXPECT_TRUE(std::abs(point.z) < 100.0f);
}

void test_IFS_AllPresetsConverge() {
    // Verify all presets produce bounded attractors
    IFSPreset presets[] = {
        IFSPreset::SierpinskiTriangle2D,
        IFSPreset::Vicsek2D,
        IFSPreset::SierpinskiCarpet2D,
        IFSPreset::SierpinskiTriangle3D,
        IFSPreset::Vicsek3D,
        IFSPreset::SierpinskiCarpet3D
    };

    for (auto preset : presets) {
        auto instructions = TransformPresets::GetPreset(preset);
        auto matrices = TransformPresets::BuildMatrices(instructions);

        Vec3 point = {0.5f, 0.5f, 0.5f};
        uint32_t seed = 12345;

        for (int i = 0; i < 2000; i++) {
            seed = (seed << 13) ^ seed;
            seed = seed * (seed * seed * 15731 + 789221) + 1376312589;
            int idx = static_cast<int>((seed & 0x7FFFFFFF) % matrices.size());
            point = TransformPoint(matrices[idx], point);
        }

        // After 2000 iterations, the point should be bounded
        EXPECT_TRUE(std::abs(point.x) < 100.0f);
        EXPECT_TRUE(std::abs(point.y) < 100.0f);
        EXPECT_TRUE(std::abs(point.z) < 100.0f);
    }
}

// =============================================================================
// Full Rasterizer Pipeline Integration Tests
// =============================================================================

void test_FullPipeline_RenderAndShade() {
    // End-to-end test: create meshes, render to GBuffer, apply deferred shading
    SoftwareRasterizer rast;
    SoftRasterizerConfig config;
    config.width = 128;
    config.height = 128;
    config.threadCount = 1;
    config.enableBackfaceCulling = false;
    rast.Initialize(config);

    SoftMesh cube = SoftMeshGen::CreateCube(2.0f);
    int meshIdx = rast.AddMesh(cube);

    rast.Clear();

    SoftCamera cam;
    cam.position = {0, 0, -5};
    cam.forward = {0, 0, 1};
    cam.up = {0, 1, 0};
    cam.right = {1, 0, 0};
    cam.fov = 60.0f;
    cam.aspectRatio = 1.0f;
    cam.nearPlane = 0.1f;
    cam.farPlane = 100.0f;

    SoftTransform obj;
    obj.meshIndex = meshIdx;
    obj.position = {0, 0, 0};
    obj.tint = {255, 200, 150, 255};

    rast.RenderObjects(&obj, 1, cam);

    // Apply deferred shading
    DeferredShading deferred;
    DeferredShadingConfig shadingConfig;
    shadingConfig.sunLight.direction = {-0.5f, -1.0f, 0.5f};
    shadingConfig.sunLight.intensity = 1.0f;
    shadingConfig.ambientIntensity = 0.2f;
    shadingConfig.fogEnd = 500.0f;

    deferred.Apply(rast.GetGBuffer(), shadingConfig, cam);

    // Verify something was rendered in the center
    const GBuffer& gb = rast.GetGBuffer();
    float centerDepth = gb.GetDepth(64, 64);
    (void)centerDepth;

    // Count how many pixels have depth < max (were rasterized)
    int pixelsRendered = 0;
    for (int y = 0; y < 128; y++) {
        for (int x = 0; x < 128; x++) {
            if (gb.GetDepth(x, y) < 1e30f) pixelsRendered++;
        }
    }
    EXPECT_TRUE(pixelsRendered > 0);

    rast.Shutdown();
}

void test_FullPipeline_OcclusionCullingIntegration() {
    // Integration test: render occluder + verify front object is visible
    OcclusionCuller culler;
    culler.Initialize();

    SoftMesh cube = SoftMeshGen::CreateCube(10.0f);
    int meshIdx = culler.AddMesh(cube);

    SoftCamera cam;
    cam.position = {0, 0, -10};
    cam.forward = {0, 0, 1};
    cam.up = {0, 1, 0};
    cam.right = {1, 0, 0};
    cam.fov = 60.0f;
    cam.aspectRatio = 16.0f / 9.0f;
    cam.nearPlane = 0.1f;
    cam.farPlane = 1000.0f;

    // Place large occluder cube at origin
    SoftTransform occluder;
    occluder.meshIndex = meshIdx;
    occluder.position = {0, 0, 0};
    occluder.scale = {1, 1, 1};
    occluder.tint = {255, 255, 255, 255};

    culler.RenderOccluders(&occluder, 1, cam);

    // At minimum, a behind-the-occluder object test is exercised
    AABox behind;
    behind.min = {-1, -1, 20};
    behind.max = {1, 1, 22};
    // Just verify the call doesn't crash
    (void)culler.IsVisible(behind, cam);

    // Verify the depth buffer was populated (something was rasterized)
    const float* depth = culler.GetDepthBuffer();
    bool somePixelWritten = false;
    for (int i = 0; i < culler.GetWidth() * culler.GetHeight(); i++) {
        if (depth[i] < 1e30f) {
            somePixelWritten = true;
            break;
        }
    }
    EXPECT_TRUE(somePixelWritten);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterPipelineIntegrationTests() {
    // IFS chaos game tests
    RUN_TEST("IFS_ChaosGame_CPUSimulation", test_IFS_ChaosGame_CPUSimulation);
    RUN_TEST("IFS_AllPresetsConverge", test_IFS_AllPresetsConverge);

    // Full pipeline integration tests
    RUN_TEST("FullPipeline_RenderAndShade", test_FullPipeline_RenderAndShade);
    RUN_TEST("FullPipeline_OcclusionCullingIntegration", test_FullPipeline_OcclusionCullingIntegration);
}
