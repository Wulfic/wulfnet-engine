// =============================================================================
// WulfNet Engine - Integration & Stress Tests
// =============================================================================
// Cross-system integration tests, multi-system pipelines, and stress tests
// that exercise the engine holistically.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>
#include <WulfNet/Procedural/IFS/AffineTransform.h>
#include <WulfNet/Procedural/IFS/TransformPresets.h>
#include <WulfNet/Procedural/IFS/TransformBlender.h>
#include <WulfNet/Rendering/Types/RenderTypes.h>
#include <WulfNet/Rendering/SoftwareRasterizer/GBuffer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/SoftwareRasterizer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/DeferredShading.h>
#include <WulfNet/Rendering/SoftwareRasterizer/OcclusionCuller.h>
#include <WulfNet/Core/System/SystemMonitor.h>

#include <WulfNet/Jolt/Physics/Collision/Shape/BoxShape.h>
#include <WulfNet/Jolt/Physics/Collision/Shape/SphereShape.h>
#include <WulfNet/Jolt/Physics/Body/BodyCreationSettings.h>

#include <cmath>
#include <vector>
#include <random>

using namespace WulfNet;
using namespace JPH::literals;

// =============================================================================
// Multi-System Integration Tests
// =============================================================================

void test_Integration_RasterizerAndOcclusion() {
    // Render scene with rasterizer, then use occlusion culler
    SoftwareRasterizer rast;
    SoftRasterizerConfig rastConfig;
    rastConfig.width = 128;
    rastConfig.height = 128;
    rastConfig.threadCount = 1;
    rastConfig.enableBackfaceCulling = false;
    rast.Initialize(rastConfig);

    OcclusionCuller culler;
    culler.Initialize();

    // Create mesh
    SoftMesh cube = SoftMeshGen::CreateCube(2.0f);
    int rastMeshIdx = rast.AddMesh(cube);
    int cullMeshIdx = culler.AddMesh(cube);

    SoftCamera cam;
    cam.position = {0, 0, -10};
    cam.forward = {0, 0, 1};
    cam.up = {0, 1, 0};
    cam.right = {1, 0, 0};
    cam.fov = 60.0f;
    cam.aspectRatio = 1.0f;
    cam.nearPlane = 0.1f;
    cam.farPlane = 100.0f;

    // Render objects
    SoftTransform obj;
    obj.meshIndex = rastMeshIdx;
    obj.position = {0, 0, 0};
    obj.tint = {200, 150, 100, 255};

    rast.Clear();
    rast.RenderObjects(&obj, 1, cam);

    // Apply deferred shading
    DeferredShading deferred;
    DeferredShadingConfig shadingConfig;
    shadingConfig.sunLight.direction = {-0.5f, -1.0f, 0.5f};
    shadingConfig.sunLight.intensity = 1.0f;
    shadingConfig.ambientIntensity = 0.2f;
    deferred.Apply(rast.GetGBuffer(), shadingConfig, cam);

    // Run occlusion culling
    SoftTransform occluder;
    occluder.meshIndex = cullMeshIdx;
    occluder.position = {0, 0, 0};
    occluder.tint = {255, 255, 255, 255};
    culler.RenderOccluders(&occluder, 1, cam);

    // Test visibility
    AABox visible;
    visible.min = {-1, -1, -5};
    visible.max = {1, 1, -3};
    EXPECT_TRUE(culler.IsVisible(visible, cam));

    rast.Shutdown();
}

void test_Integration_FractalAndRasterizer() {
    // Generate fractal points and visualize with software rasterizer
    auto instructions = TransformPresets::GetPreset(IFSPreset::SierpinskiTriangle3D);
    auto matrices = TransformPresets::BuildMatrices(instructions);

    // Run chaos game
    Vec3 point = {0.5f, 0.5f, 0.5f};
    uint32_t seed = 42;
    std::vector<Vec3> fractalPoints;

    for (int i = 0; i < 2000; i++) {
        seed = (seed << 13) ^ seed;
        seed = seed * (seed * seed * 15731 + 789221) + 1376312589;
        int idx = static_cast<int>((seed & 0x7FFFFFFF) % matrices.size());

        float x = matrices[idx].At(0, 0) * point.x + matrices[idx].At(0, 1) * point.y + matrices[idx].At(0, 2) * point.z + matrices[idx].At(0, 3);
        float y = matrices[idx].At(1, 0) * point.x + matrices[idx].At(1, 1) * point.y + matrices[idx].At(1, 2) * point.z + matrices[idx].At(1, 3);
        float z = matrices[idx].At(2, 0) * point.x + matrices[idx].At(2, 1) * point.y + matrices[idx].At(2, 2) * point.z + matrices[idx].At(2, 3);
        point = {x, y, z};

        if (i > 50) { // Skip initial convergence
            fractalPoints.push_back(point);
        }
    }

    // Verify fractal points are bounded
    EXPECT_TRUE(fractalPoints.size() > 1000);
    for (const auto& p : fractalPoints) {
        EXPECT_TRUE(std::abs(p.x) < 50.0f);
        EXPECT_TRUE(std::abs(p.y) < 50.0f);
        EXPECT_TRUE(std::abs(p.z) < 50.0f);
    }

    // Setup rasterizer (just initialize/shutdown to ensure no conflicts)
    SoftwareRasterizer rast;
    SoftRasterizerConfig config;
    config.width = 64;
    config.height = 64;
    config.threadCount = 1;
    rast.Initialize(config);
    rast.Clear();
    rast.Shutdown();
}

void test_Integration_SystemMonitorDuringPhysics() {
    // Monitor system resources while running physics
    SystemMonitor& mon = SystemMonitor::Get();
    mon.Initialize();

    PhysicsWorld world;
    world.Initialize();

    JPH::BodyInterface& bi = world.GetBodyInterface();
    for (int i = 0; i < 50; i++) {
        JPH::BodyCreationSettings s(
            new JPH::SphereShape(0.2f),
            JPH::RVec3(static_cast<float>(i % 10) * 0.5_r, static_cast<float>(i) * 0.5_r, 0.0_r),
            JPH::Quat::sIdentity(),
            JPH::EMotionType::Dynamic,
            Layers::MOVING
        );
        bi.CreateAndAddBody(s, JPH::EActivation::Activate);
    }

    // Step while monitoring
    for (int frame = 0; frame < 10; frame++) {
        mon.Update();
        world.Step(1.0f / 60.0f);
    }

    const SystemStats& stats = mon.GetStats();
    EXPECT_TRUE(stats.ramTotalBytes > 0);
    EXPECT_TRUE(stats.processMemoryBytes > 0);

    world.Shutdown();
    mon.Shutdown();
}

// =============================================================================
// Stress Tests
// =============================================================================

void test_Stress_RapidRasterizerRender() {
    SoftwareRasterizer rast;
    SoftRasterizerConfig config;
    config.width = 128;
    config.height = 128;
    config.threadCount = 1;
    config.enableBackfaceCulling = false;
    rast.Initialize(config);

    SoftMesh cube = SoftMeshGen::CreateCube(2.0f);
    int meshIdx = rast.AddMesh(cube);

    SoftCamera cam;
    cam.position = {0, 0, -5};
    cam.forward = {0, 0, 1};
    cam.up = {0, 1, 0};
    cam.right = {1, 0, 0};
    cam.fov = 60.0f;
    cam.aspectRatio = 1.0f;
    cam.nearPlane = 0.1f;
    cam.farPlane = 100.0f;

    // Render 50 frames rapidly
    for (int frame = 0; frame < 50; frame++) {
        rast.Clear();

        SoftTransform obj;
        obj.meshIndex = meshIdx;
        // Orbit the camera position
        float angle = static_cast<float>(frame) * 0.1f;
        obj.position = {std::cos(angle) * 2.0f, 0.0f, std::sin(angle) * 2.0f};
        obj.tint = {255, 200, 100, 255};

        rast.RenderObjects(&obj, 1, cam);
    }

    // Verify last frame rendered something
    const GBuffer& gb = rast.GetGBuffer();
    bool anyRendered = false;
    for (int i = 0; i < 128 * 128; i++) {
        if (gb.GetDepthBuffer()[i] < 1e30f) {
            anyRendered = true;
            break;
        }
    }
    EXPECT_TRUE(anyRendered);

    rast.Shutdown();
}

// =============================================================================
// Blender Continuous Morphing Stress Test
// =============================================================================

void test_Stress_BlenderContinuousMorphing() {
    TransformBlender blender;
    IFSPreset presets[] = {
        IFSPreset::SierpinskiTriangle2D,
        IFSPreset::Vicsek2D,
        IFSPreset::SierpinskiCarpet2D,
        IFSPreset::SierpinskiTriangle3D,
        IFSPreset::Vicsek3D,
        IFSPreset::SierpinskiCarpet3D
    };
    int numPresets = 6;

    auto set0 = TransformPresets::GetPreset(presets[0]);
    auto set1 = TransformPresets::GetPreset(presets[1]);
    blender.SetSets(set0, set1);

    // Simulate rapid preset switching over 200 frames
    for (int frame = 0; frame < 200; frame++) {
        blender.Update(0.016f, 5.0f);

        if (frame % 30 == 0 && frame > 0) {
            int nextPreset = (frame / 30 + 1) % numPresets;
            auto nextSet = TransformPresets::GetPreset(presets[nextPreset]);
            blender.SwitchTarget(nextSet);
        }

        // Get blended matrices every frame
        auto matrices = blender.GetBlendedMatrices();
        EXPECT_TRUE(!matrices.empty());

        // Verify all matrices have finite values
        for (const auto& mat : matrices) {
            for (int i = 0; i < 16; i++) {
                EXPECT_TRUE(std::isfinite(mat.m[i]));
            }
        }
    }
}

// =============================================================================
// Registration
// =============================================================================

void RegisterIntegrationTests() {
    // Multi-system integration
    RUN_TEST("Integration_RasterizerAndOcclusion", test_Integration_RasterizerAndOcclusion);
    RUN_TEST("Integration_FractalAndRasterizer", test_Integration_FractalAndRasterizer);
    RUN_TEST("Integration_SystemMonitorDuringPhysics", test_Integration_SystemMonitorDuringPhysics);

    // Stress tests
    RUN_TEST("Stress_RapidRasterizerRender", test_Stress_RapidRasterizerRender);
    RUN_TEST("Stress_BlenderContinuousMorphing", test_Stress_BlenderContinuousMorphing);
}
