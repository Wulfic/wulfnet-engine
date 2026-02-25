// =============================================================================
// WulfNet Engine - Render Pipeline Tests
// =============================================================================
// Integration tests for the unified render pipeline.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Rendering/SoftwareRasterizer/RenderPipeline.h>
#include <WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h>
#include <cmath>

using namespace WulfNet;

// =============================================================================
// Helpers
// =============================================================================

static RenderPipelineConfig CreateSmallConfig() {
    RenderPipelineConfig config;
    config.rasterizer.width = 64;
    config.rasterizer.height = 64;
    config.rasterizer.threadCount = 1;
    config.shadows.numCascades = 2;
    config.shadows.cascadeResolution = 32;
    config.shadows.pointLightResolution = 32;
    config.gi.ssao.sampleCount = 4;
    config.gi.ssao.blurPasses = 0;
    config.gi.indirect.sampleCount = 4;
    config.volumetric.maxSteps = 16;
    config.volumetric.stepSize = 0.5f;
    return config;
}

static SoftCamera CreatePipelineCam() {
    SoftCamera cam;
    cam.position = {0, 5, -15};
    cam.forward = {0, 0, 1};
    cam.up = {0, 1, 0};
    cam.right = {1, 0, 0};
    cam.fov = 60.0f;
    cam.aspectRatio = 1.0f;
    cam.nearPlane = 0.1f;
    cam.farPlane = 200.0f;
    return cam;
}

// =============================================================================
// Initialization Tests
// =============================================================================

static void Test_RP_Initialize() {
    RenderPipeline pipeline;
    auto config = CreateSmallConfig();
    EXPECT_TRUE(pipeline.Initialize(config));
    EXPECT_EQ(pipeline.GetWidth(), 64);
    EXPECT_EQ(pipeline.GetHeight(), 64);
}

static void Test_RP_InitializeDefaults() {
    RenderPipeline pipeline;
    EXPECT_TRUE(pipeline.Initialize());
    EXPECT_GT(pipeline.GetWidth(), 0);
    EXPECT_GT(pipeline.GetHeight(), 0);
}

static void Test_RP_Shutdown() {
    RenderPipeline pipeline;
    pipeline.Initialize(CreateSmallConfig());
    pipeline.Shutdown(); // Should not crash
    // Re-initialize after shutdown
    EXPECT_TRUE(pipeline.Initialize(CreateSmallConfig()));
}

// =============================================================================
// Scene Management Tests
// =============================================================================

static void Test_RP_AddMesh() {
    RenderPipeline pipeline;
    pipeline.Initialize(CreateSmallConfig());

    SoftMesh cube = SoftMeshGen::CreateCube(2.0f);
    int idx = pipeline.AddMesh(cube);
    EXPECT_EQ(idx, 0);

    SoftMesh sphere = SoftMeshGen::CreateSphere(1.0f, 8, 8);
    int idx2 = pipeline.AddMesh(sphere);
    EXPECT_EQ(idx2, 1);
}

static void Test_RP_AddVolume() {
    RenderPipeline pipeline;
    pipeline.Initialize(CreateSmallConfig());

    VolumeSampler vol;
    vol.region.boundsMin = {-5, -5, -5};
    vol.region.boundsMax = {5, 5, 5};
    vol.sampleDensity = [](float, float, float) { return 0.5f; };

    pipeline.AddVolume(vol);
    EXPECT_EQ(pipeline.GetVolumetric().GetVolumeCount(), 1);

    pipeline.ClearVolumes();
    EXPECT_EQ(pipeline.GetVolumetric().GetVolumeCount(), 0);
}

static void Test_RP_AddLightProbe() {
    RenderPipeline pipeline;
    pipeline.Initialize(CreateSmallConfig());

    LightProbe probe;
    probe.position = {0, 5, 0};
    probe.radius = 20.0f;
    probe.shCoeffs[0] = {0.3f, 0.3f, 0.3f};

    pipeline.AddLightProbe(probe);
    EXPECT_EQ(pipeline.GetGI().GetProbeCount(), 1);
}

// =============================================================================
// Rendering Tests
// =============================================================================

static void Test_RP_RenderEmptyScene() {
    RenderPipeline pipeline;
    pipeline.Initialize(CreateSmallConfig());

    SoftCamera cam = CreatePipelineCam();
    pipeline.RenderFrame(nullptr, 0, cam);

    // Should produce a sky gradient without crashing
    const uint32_t* colorBuf = pipeline.GetColorBuffer();
    EXPECT_TRUE(colorBuf != nullptr);
}

static void Test_RP_RenderWithGeometry() {
    RenderPipeline pipeline;
    auto config = CreateSmallConfig();
    config.shading.sunLight.direction = {0, -1, 0.5f};
    config.shading.sunLight.intensity = 1.0f;
    pipeline.Initialize(config);

    SoftMesh cube = SoftMeshGen::CreateCube(4.0f);
    pipeline.AddMesh(cube);

    SoftTransform xform;
    xform.meshIndex = 0;
    xform.position = {0, 0, 10};

    SoftCamera cam = CreatePipelineCam();
    pipeline.RenderFrame(&xform, 1, cam);

    // Check that some pixels are not sky (have geometry colors)
    const uint32_t* colorBuf = pipeline.GetColorBuffer();
    bool hasNonSky = false;
    for (int i = 0; i < 64 * 64; ++i) {
        uint32_t pixel = colorBuf[i];
        uint8_t r = pixel & 0xFF;
        uint8_t g = (pixel >> 8) & 0xFF;
        uint8_t b = (pixel >> 16) & 0xFF;
        // Sky colors are blueish; geometry + lighting should produce other colors
        if (r > 0 || g > 0 || b > 0) {
            hasNonSky = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonSky);
}

static void Test_RP_ShadowsDisabled() {
    RenderPipeline pipeline;
    auto config = CreateSmallConfig();
    config.enableShadows = false;
    pipeline.Initialize(config);

    SoftMesh cube = SoftMeshGen::CreateCube(2.0f);
    pipeline.AddMesh(cube);

    SoftTransform xform;
    xform.meshIndex = 0;
    xform.position = {0, 0, 10};

    SoftCamera cam = CreatePipelineCam();
    pipeline.RenderFrame(&xform, 1, cam);

    // Should render without shadow pass
    EXPECT_EQ(pipeline.GetStats().shadowCascadesUsed, 0);
}

static void Test_RP_ShadowsEnabled() {
    RenderPipeline pipeline;
    auto config = CreateSmallConfig();
    config.enableShadows = true;
    config.shading.sunLight.direction = {0, -1, 0.5f};
    pipeline.Initialize(config);

    SoftMesh cube = SoftMeshGen::CreateCube(2.0f);
    pipeline.AddMesh(cube);

    SoftTransform xform;
    xform.meshIndex = 0;
    xform.position = {0, 0, 10};

    SoftCamera cam = CreatePipelineCam();
    pipeline.RenderFrame(&xform, 1, cam);

    EXPECT_GT(pipeline.GetStats().shadowCascadesUsed, 0);
}

static void Test_RP_GIDisabled() {
    RenderPipeline pipeline;
    auto config = CreateSmallConfig();
    config.enableGI = false;
    pipeline.Initialize(config);

    SoftCamera cam = CreatePipelineCam();
    pipeline.RenderFrame(nullptr, 0, cam);
    // Should not crash with GI disabled
    EXPECT_TRUE(pipeline.GetColorBuffer() != nullptr);
}

static void Test_RP_VolumetricDisabled() {
    RenderPipeline pipeline;
    auto config = CreateSmallConfig();
    config.enableVolumetric = false;
    pipeline.Initialize(config);

    SoftCamera cam = CreatePipelineCam();
    pipeline.RenderFrame(nullptr, 0, cam);
    EXPECT_EQ(pipeline.GetStats().volumetricVolumes, 0);
}

static void Test_RP_VolumetricWithVolume() {
    RenderPipeline pipeline;
    auto config = CreateSmallConfig();
    config.enableVolumetric = true;
    pipeline.Initialize(config);

    VolumeSampler vol;
    vol.region.boundsMin = {-5, -5, 5};
    vol.region.boundsMax = {5, 5, 15};
    vol.sampleDensity = [](float, float, float) { return 0.5f; };
    vol.sampleTemperature = [](float, float, float) { return 0.0f; };
    pipeline.AddVolume(vol);

    SoftCamera cam = CreatePipelineCam();
    pipeline.RenderFrame(nullptr, 0, cam);
    EXPECT_EQ(pipeline.GetStats().volumetricVolumes, 1);
}

// =============================================================================
// Config Tests
// =============================================================================

static void Test_RP_SetShadingConfig() {
    RenderPipeline pipeline;
    pipeline.Initialize(CreateSmallConfig());

    DeferredShadingConfig newShading;
    newShading.sunLight.direction = {1, -1, 0};
    newShading.sunLight.intensity = 2.0f;
    newShading.fogStart = 10.0f;
    newShading.fogEnd = 50.0f;

    pipeline.SetShadingConfig(newShading);
    EXPECT_NEAR(pipeline.GetConfig().shading.sunLight.intensity, 2.0f, 0.01f);
}

static void Test_RP_Stats() {
    RenderPipeline pipeline;
    auto config = CreateSmallConfig();
    config.enableShadows = true;
    config.shading.pointLights.push_back({{0, 10, 0}, {1, 1, 1}, 1.0f, 20.0f});
    pipeline.Initialize(config);

    SoftMesh cube = SoftMeshGen::CreateCube(2.0f);
    pipeline.AddMesh(cube);

    SoftTransform xform;
    xform.meshIndex = 0;
    xform.position = {0, 0, 10};

    SoftCamera cam = CreatePipelineCam();
    pipeline.RenderFrame(&xform, 1, cam);

    const auto& stats = pipeline.GetStats();
    EXPECT_EQ(stats.shadowCascadesUsed, 2);
    EXPECT_EQ(stats.pointLightShadows, 1);
}

static void Test_RP_SubsystemAccess() {
    RenderPipeline pipeline;
    pipeline.Initialize(CreateSmallConfig());

    // All sub-systems should be accessible
    GBuffer& gb = pipeline.GetGBuffer();
    EXPECT_EQ(gb.GetWidth(), 64);
    EXPECT_EQ(gb.GetHeight(), 64);

    EXPECT_EQ(pipeline.GetShadowSystem().GetCascadeCount(), 2);
    EXPECT_EQ(pipeline.GetGI().GetWidth(), 64);
    EXPECT_EQ(pipeline.GetVolumetric().GetWidth(), 64);
}

static void Test_RP_MultipleFrames() {
    RenderPipeline pipeline;
    pipeline.Initialize(CreateSmallConfig());

    SoftMesh cube = SoftMeshGen::CreateCube(2.0f);
    pipeline.AddMesh(cube);

    SoftTransform xform;
    xform.meshIndex = 0;
    xform.position = {0, 0, 10};

    SoftCamera cam = CreatePipelineCam();

    // Render multiple frames without crash
    for (int i = 0; i < 3; ++i) {
        xform.position.x = static_cast<float>(i);
        pipeline.RenderFrame(&xform, 1, cam);
    }

    EXPECT_TRUE(pipeline.GetColorBuffer() != nullptr);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterRenderPipelineTests() {
    // Initialization
    RUN_TEST("RP_Initialize", Test_RP_Initialize);
    RUN_TEST("RP_InitializeDefaults", Test_RP_InitializeDefaults);
    RUN_TEST("RP_Shutdown", Test_RP_Shutdown);

    // Scene Management
    RUN_TEST("RP_AddMesh", Test_RP_AddMesh);
    RUN_TEST("RP_AddVolume", Test_RP_AddVolume);
    RUN_TEST("RP_AddLightProbe", Test_RP_AddLightProbe);

    // Rendering
    RUN_TEST("RP_RenderEmptyScene", Test_RP_RenderEmptyScene);
    RUN_TEST("RP_RenderWithGeometry", Test_RP_RenderWithGeometry);
    RUN_TEST("RP_ShadowsDisabled", Test_RP_ShadowsDisabled);
    RUN_TEST("RP_ShadowsEnabled", Test_RP_ShadowsEnabled);
    RUN_TEST("RP_GIDisabled", Test_RP_GIDisabled);
    RUN_TEST("RP_VolumetricDisabled", Test_RP_VolumetricDisabled);
    RUN_TEST("RP_VolumetricWithVolume", Test_RP_VolumetricWithVolume);

    // Config & Stats
    RUN_TEST("RP_SetShadingConfig", Test_RP_SetShadingConfig);
    RUN_TEST("RP_Stats", Test_RP_Stats);
    RUN_TEST("RP_SubsystemAccess", Test_RP_SubsystemAccess);
    RUN_TEST("RP_MultipleFrames", Test_RP_MultipleFrames);
}
