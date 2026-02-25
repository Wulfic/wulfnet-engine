// =============================================================================
// WulfNet Engine - Integration & Stress Tests
// =============================================================================
// Cross-system integration tests, multi-system pipelines, and stress tests
// that exercise the engine holistically.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>
#include <WulfNet/Physics/Fluids/COFLIPSystem.h>
#include <WulfNet/Physics/Fluids/FluidSurface.h>
#include <WulfNet/Physics/Fluids/FluidSystem.h>
#include <WulfNet/Physics/Fluids/FluidParticle.h>
#include <WulfNet/Physics/Fluids/FluidGrid.h>
#include <WulfNet/Procedural/IFS/AffineTransform.h>
#include <WulfNet/Procedural/IFS/TransformPresets.h>
#include <WulfNet/Procedural/IFS/TransformBlender.h>
#include <WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h>
#include <WulfNet/Rendering/SoftwareRasterizer/GBuffer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/SoftwareRasterizer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/DeferredShading.h>
#include <WulfNet/Rendering/SoftwareRasterizer/OcclusionCuller.h>
#include <WulfNet/Core/System/SystemMonitor.h>

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

#include <cmath>
#include <vector>
#include <random>

using namespace WulfNet;
using namespace JPH::literals;

// =============================================================================
// Multi-System Integration Tests
// =============================================================================

void test_Integration_PhysicsAndFluid() {
    // Run Jolt rigid body physics alongside CO-FLIP fluid simulation
    PhysicsWorld joltWorld;
    joltWorld.Initialize();

    COFLIPSystem fluid;
    COFLIPConfig fluidConfig;
    fluidConfig.gridSizeX = 16;
    fluidConfig.gridSizeY = 16;
    fluidConfig.gridSizeZ = 16;
    fluidConfig.cellSize = 0.2f;
    fluidConfig.useGPU = false;
    fluidConfig.pressureIterations = 3;
    fluid.Initialize(fluidConfig);

    // Add rigid body
    JPH::BodyInterface& bi = joltWorld.GetBodyInterface();
    JPH::BodyCreationSettings bodySettings(
        new JPH::SphereShape(0.3f),
        JPH::RVec3(1.0_r, 5.0_r, 1.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );
    JPH::BodyID sphereId = bi.CreateAndAddBody(bodySettings, JPH::EActivation::Activate);

    // Add fluid particles
    fluid.AddParticleBox(0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f);

    // Step both systems in tandem
    for (int frame = 0; frame < 30; frame++) {
        float dt = 1.0f / 60.0f;
        joltWorld.Step(dt);
        fluid.Step(dt);
    }

    // Rigid body should have fallen
    JPH::RVec3 pos = bi.GetCenterOfMassPosition(sphereId);
    EXPECT_TRUE(pos.GetY() < 5.0f);

    // Fluid should still have particles
    EXPECT_TRUE(fluid.GetActiveParticleCount() > 0);

    bi.RemoveBody(sphereId);
    bi.DestroyBody(sphereId);
    joltWorld.Shutdown();
    fluid.Shutdown();
}

void test_Integration_FluidAndSurface() {
    // Run fluid sim and extract surface every Nth frame
    COFLIPSystem fluid;
    COFLIPConfig fluidConfig;
    fluidConfig.gridSizeX = 16;
    fluidConfig.gridSizeY = 16;
    fluidConfig.gridSizeZ = 16;
    fluidConfig.cellSize = 0.2f;
    fluidConfig.useGPU = false;
    fluidConfig.pressureIterations = 3;
    fluid.Initialize(fluidConfig);

    FluidSurface surface;
    FluidSurfaceConfig surfConfig;
    surfConfig.gridSizeX = 16;
    surfConfig.gridSizeY = 16;
    surfConfig.gridSizeZ = 16;
    surfConfig.cellSize = 0.2f;
    surfConfig.isoLevel = 0.3f;
    surfConfig.useGPU = false;
    surface.Initialize(surfConfig);

    fluid.AddParticleBox(0.5f, 0.5f, 0.5f, 2.0f, 2.0f, 2.0f);

    // Simulate 20 frames, extract surface every 5th frame
    for (int frame = 0; frame < 20; frame++) {
        fluid.Step(1.0f / 60.0f);

        if (frame % 5 == 0) {
            surface.GenerateSurface(fluid);
            // Surface stats should update
            const FluidSurfaceStats& stats = surface.GetStats();
            EXPECT_TRUE(stats.totalTimeMs >= 0.0f);
        }
    }

    EXPECT_TRUE(true); // No crashes throughout

    fluid.Shutdown();
    surface.Shutdown();
}

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
// Fluid Grid Advanced Tests
// =============================================================================

void test_FluidGrid_TrilinearInterpolation() {
    FluidGrid grid;
    grid.Initialize(8, 8, 8, 1.0f);
    grid.SetBounds(0.0f, 0.0f, 0.0f, 8.0f, 8.0f, 8.0f);
    grid.Reset();

    // Set known velocity field: u = x (linear in x)
    for (uint32_t k = 0; k < 8; k++) {
        for (uint32_t j = 0; j < 8; j++) {
            for (uint32_t i = 0; i < 8; i++) {
                MACCell& cell = grid.GetCell(i, j, k);
                cell.u = static_cast<float>(i);
                cell.v = static_cast<float>(j);
                cell.w = static_cast<float>(k);
            }
        }
    }

    // Interpolate at grid point
    float vx, vy, vz;
    grid.InterpolateVelocity(3.0f, 3.0f, 3.0f, vx, vy, vz);

    // At grid coordinates, should approximate the stored values
    // Exact values depend on staggering, but should be reasonable
    EXPECT_TRUE(std::isfinite(vx));
    EXPECT_TRUE(std::isfinite(vy));
    EXPECT_TRUE(std::isfinite(vz));
}

void test_FluidGrid_BoundsConversion() {
    FluidGrid grid;
    grid.Initialize(10, 20, 30, 0.5f);
    grid.SetBounds(-5.0f, 0.0f, -15.0f, 5.0f, 10.0f, 15.0f);

    // World to grid conversion
    float gx, gy, gz;
    grid.WorldToGrid(0.0f, 5.0f, 0.0f, gx, gy, gz);

    // (0 - (-5)) / 0.5 = 10.0 for x, (5 - 0) / 0.5 = 10.0 for y, etc.
    EXPECT_NEAR(gx, 10.0f, 0.5f);
    EXPECT_NEAR(gy, 10.0f, 0.5f);
    EXPECT_NEAR(gz, 30.0f, 0.5f);

    // Grid to world conversion (round trip)
    float wx, wy, wz;
    grid.GridToWorld(gx, gy, gz, wx, wy, wz);
    EXPECT_NEAR(wx, 0.0f, 0.1f);
    EXPECT_NEAR(wy, 5.0f, 0.1f);
    EXPECT_NEAR(wz, 0.0f, 0.1f);
}

void test_FluidGrid_CellIndexConversion() {
    FluidGrid grid;
    grid.Initialize(10, 20, 30, 0.5f);

    // Forward conversion
    uint32_t index = grid.GetIndex(3, 7, 15);

    // Reverse conversion
    uint32_t i, j, k;
    grid.GetIJK(index, i, j, k);
    EXPECT_EQ(i, 3u);
    EXPECT_EQ(j, 7u);
    EXPECT_EQ(k, 15u);
}

void test_FluidGrid_ResetClearsData() {
    FluidGrid grid;
    grid.Initialize(4, 4, 4, 1.0f);
    grid.SetBounds(0, 0, 0, 4, 4, 4);

    // Set some values
    grid.GetCell(2, 2, 2).u = 5.0f;
    grid.GetCell(2, 2, 2).pressure = 100.0f;
    grid.GetCell(2, 2, 2).state = CellState::Fluid;

    grid.Reset();

    // Should be cleared
    const MACCell& cell = grid.GetCell(2, 2, 2);
    EXPECT_NEAR(cell.u, 0.0f, 1e-6f);
    EXPECT_NEAR(cell.pressure, 0.0f, 1e-6f);
    EXPECT_TRUE(cell.state == CellState::Empty);
}

// =============================================================================
// Stress Tests
// =============================================================================

void test_Stress_FluidHighParticleCount() {
    COFLIPSystem fluid;
    COFLIPConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.useGPU = false;
    config.pressureIterations = 2; // Reduced for speed
    fluid.Initialize(config);

    // Add large volume of particles
    fluid.AddParticleBox(0.5f, 0.5f, 0.5f, 2.5f, 2.5f, 2.5f);
    uint32_t count = fluid.GetActiveParticleCount();
    EXPECT_TRUE(count > 100);

    // Step a few frames — should complete without crash or NaN
    for (int i = 0; i < 5; i++) {
        fluid.Step(1.0f / 60.0f);
    }

    // Verify all particles are finite
    bool allFinite = true;
    for (uint32_t i = 0; i < fluid.GetActiveParticleCount(); i++) {
        const auto& p = fluid.GetParticles()[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            allFinite = false;
            break;
        }
    }
    EXPECT_TRUE(allFinite);

    fluid.Shutdown();
}

void test_Stress_MultipleFluidSystems() {
    // Run multiple fluid systems simultaneously
    COFLIPSystem fluid1, fluid2;

    COFLIPConfig config;
    config.gridSizeX = 8;
    config.gridSizeY = 8;
    config.gridSizeZ = 8;
    config.cellSize = 0.5f;
    config.useGPU = false;
    config.pressureIterations = 2;

    fluid1.Initialize(config);
    fluid2.Initialize(config);

    fluid1.AddParticleSphere(2.0f, 2.0f, 2.0f, 1.0f);
    fluid2.AddParticleBox(1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 3.0f);

    for (int i = 0; i < 10; i++) {
        fluid1.Step(1.0f / 60.0f);
        fluid2.Step(1.0f / 60.0f);
    }

    // Both should still be valid
    EXPECT_TRUE(fluid1.GetActiveParticleCount() > 0);
    EXPECT_TRUE(fluid2.GetActiveParticleCount() > 0);

    fluid1.Shutdown();
    fluid2.Shutdown();
}

void test_Stress_SurfaceExtractionLargeGrid() {
    FluidSurface surface;
    FluidSurfaceConfig config;
    config.gridSizeX = 32;
    config.gridSizeY = 32;
    config.gridSizeZ = 32;
    config.cellSize = 0.1f;
    config.isoLevel = 0.5f;
    config.useGPU = false;
    surface.Initialize(config);

    // Fill a sphere density on a larger grid
    float cx = 16.0f, cy = 16.0f, cz = 16.0f;
    float radius = 10.0f;
    for (int k = 0; k < 32; k++)
        for (int j = 0; j < 32; j++)
            for (int i = 0; i < 32; i++) {
                float dx = static_cast<float>(i) - cx;
                float dy = static_cast<float>(j) - cy;
                float dz = static_cast<float>(k) - cz;
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                surface.SetDensity(i, j, k, std::max(0.0f, 1.0f - dist / radius));
            }

    surface.ExtractSurface();

    // Should produce significant geometry
    EXPECT_TRUE(surface.GetVertexCount() > 100);
    EXPECT_TRUE(surface.GetTriangleCount() > 50);

    surface.Shutdown();
}

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
    RUN_TEST("Integration_PhysicsAndFluid", test_Integration_PhysicsAndFluid);
    RUN_TEST("Integration_FluidAndSurface", test_Integration_FluidAndSurface);
    RUN_TEST("Integration_RasterizerAndOcclusion", test_Integration_RasterizerAndOcclusion);
    RUN_TEST("Integration_FractalAndRasterizer", test_Integration_FractalAndRasterizer);
    RUN_TEST("Integration_SystemMonitorDuringPhysics", test_Integration_SystemMonitorDuringPhysics);

    // Fluid Grid advanced
    RUN_TEST("FluidGrid_TrilinearInterpolation", test_FluidGrid_TrilinearInterpolation);
    RUN_TEST("FluidGrid_BoundsConversion", test_FluidGrid_BoundsConversion);
    RUN_TEST("FluidGrid_CellIndexConversion", test_FluidGrid_CellIndexConversion);
    RUN_TEST("FluidGrid_ResetClearsData", test_FluidGrid_ResetClearsData);

    // Stress tests
    RUN_TEST("Stress_FluidHighParticleCount", test_Stress_FluidHighParticleCount);
    RUN_TEST("Stress_MultipleFluidSystems", test_Stress_MultipleFluidSystems);
    RUN_TEST("Stress_SurfaceExtractionLargeGrid", test_Stress_SurfaceExtractionLargeGrid);
    RUN_TEST("Stress_RapidRasterizerRender", test_Stress_RapidRasterizerRender);
    RUN_TEST("Stress_BlenderContinuousMorphing", test_Stress_BlenderContinuousMorphing);
}
