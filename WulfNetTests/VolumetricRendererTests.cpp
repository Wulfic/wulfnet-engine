// =============================================================================
// WulfNet Engine - Volumetric Renderer Tests
// =============================================================================
// Tests for ray-marching, AABB intersection, emission ramp, phase function,
// and volume compositing.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Rendering/SoftwareRasterizer/VolumetricRenderer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/GBuffer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h>
#include <cmath>

using namespace WulfNet;

// =============================================================================
// Helper: create a uniform density volume
// =============================================================================
static VolumeSampler CreateUniformVolume(float density, float temperature,
                                          SoftVec3 bMin, SoftVec3 bMax) {
    VolumeSampler sampler;
    sampler.region.boundsMin = bMin;
    sampler.region.boundsMax = bMax;
    sampler.sampleDensity = [density](float, float, float) { return density; };
    sampler.sampleTemperature = [temperature](float, float, float) { return temperature; };
    return sampler;
}

static SoftCamera CreateVolCamera() {
    SoftCamera cam;
    cam.position = {0, 0, -10};
    cam.forward = {0, 0, 1};
    cam.up = {0, 1, 0};
    cam.right = {1, 0, 0};
    cam.fov = 60.0f;
    cam.aspectRatio = 1.0f;
    cam.nearPlane = 0.1f;
    cam.farPlane = 100.0f;
    return cam;
}

// =============================================================================
// Initialization Tests
// =============================================================================

static void Test_Vol_Initialize() {
    VolumetricRenderer renderer;
    EXPECT_TRUE(renderer.Initialize(64, 48));
    EXPECT_EQ(renderer.GetWidth(), 64);
    EXPECT_EQ(renderer.GetHeight(), 48);
}

static void Test_Vol_InitializeInvalid() {
    VolumetricRenderer renderer;
    EXPECT_FALSE(renderer.Initialize(0, 48));
    EXPECT_FALSE(renderer.Initialize(64, -1));
}

static void Test_Vol_AddAndClearVolumes() {
    VolumetricRenderer renderer;
    renderer.Initialize(32, 32);

    VolumeSampler vol = CreateUniformVolume(0.5f, 300.0f, {-5, -5, -5}, {5, 5, 5});
    renderer.AddVolume(vol);
    EXPECT_EQ(renderer.GetVolumeCount(), 1);

    renderer.AddVolume(vol);
    EXPECT_EQ(renderer.GetVolumeCount(), 2);

    renderer.ClearVolumes();
    EXPECT_EQ(renderer.GetVolumeCount(), 0);
}

// =============================================================================
// Ray-AABB Intersection Tests
// =============================================================================

static void Test_Vol_RayAABB_Hit() {
    SoftVec3 origin = {0, 0, -10};
    SoftVec3 dir = {0, 0, 1};
    SoftVec3 invDir = {1e8f, 1e8f, 1.0f};
    SoftVec3 boxMin = {-5, -5, -5};
    SoftVec3 boxMax = {5, 5, 5};

    float tNear, tFar;
    bool hit = VolumetricRenderer::RayAABBIntersect(origin, invDir, boxMin, boxMax, tNear, tFar);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(tNear, 5.0f, 0.01f);
    EXPECT_NEAR(tFar, 15.0f, 0.01f);
}

static void Test_Vol_RayAABB_Miss() {
    SoftVec3 origin = {0, 0, -10};
    SoftVec3 dir = {1, 0, 0}; // Ray going sideways, not toward box
    SoftVec3 invDir = {1.0f, 1e8f, 1e8f};
    SoftVec3 boxMin = {-5, -5, 5};
    SoftVec3 boxMax = {5, 5, 15};

    float tNear, tFar;
    bool hit = VolumetricRenderer::RayAABBIntersect(origin, invDir, boxMin, boxMax, tNear, tFar);
    // Ray along X won't hit a box in the +Z direction
    // Actually the Z component is zero, invDir.z = 1e8, so let's test carefully
    // The box is at z=[5,15], origin z=-10, dir.z=0 → infinite inv → large t
    // This should miss because the ray never reaches z=5
    EXPECT_FALSE(hit);
}

static void Test_Vol_RayAABB_Behind() {
    SoftVec3 origin = {0, 0, 10};
    SoftVec3 dir = {0, 0, 1};
    SoftVec3 invDir = {1e8f, 1e8f, 1.0f};
    SoftVec3 boxMin = {-5, -5, -5};
    SoftVec3 boxMax = {5, 5, 5};

    float tNear, tFar;
    bool hit = VolumetricRenderer::RayAABBIntersect(origin, invDir, boxMin, boxMax, tNear, tFar);
    // Box is behind (z: -5 to 5), origin at z=10, ray going +Z → box is behind
    EXPECT_FALSE(hit);
}

static void Test_Vol_RayAABB_InsideBox() {
    SoftVec3 origin = {0, 0, 0}; // Inside the box
    SoftVec3 dir = {0, 0, 1};
    SoftVec3 invDir = {1e8f, 1e8f, 1.0f};
    SoftVec3 boxMin = {-5, -5, -5};
    SoftVec3 boxMax = {5, 5, 5};

    float tNear, tFar;
    bool hit = VolumetricRenderer::RayAABBIntersect(origin, invDir, boxMin, boxMax, tNear, tFar);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(tNear, 0.0f, 0.01f); // Clamped to 0 when inside
    EXPECT_NEAR(tFar, 5.0f, 0.01f);
}

// =============================================================================
// Phase Function Tests
// =============================================================================

static void Test_Vol_PhaseHG_Isotropic() {
    // g=0 → isotropic → constant 1/(4π)
    float expected = 1.0f / (4.0f * 3.14159265f);
    float result = VolumetricRenderer::PhaseHG(0.5f, 0.0f);
    EXPECT_NEAR(result, expected, 0.01f);
}

static void Test_Vol_PhaseHG_ForwardScatter() {
    // g=0.8, cosTheta=1 (forward scattering) → should be highest
    float forward = VolumetricRenderer::PhaseHG(1.0f, 0.8f);
    float side = VolumetricRenderer::PhaseHG(0.0f, 0.8f);
    float back = VolumetricRenderer::PhaseHG(-1.0f, 0.8f);

    EXPECT_GT(forward, side);
    EXPECT_GT(side, back);
}

static void Test_Vol_PhaseHG_Symmetry() {
    // g=0, phase should be same for any angle
    float a = VolumetricRenderer::PhaseHG(0.5f, 0.0f);
    float b = VolumetricRenderer::PhaseHG(-0.5f, 0.0f);
    EXPECT_NEAR(a, b, 0.01f);
}

// =============================================================================
// Emission Tests
// =============================================================================

static void Test_Vol_Emission_BelowThreshold() {
    VolumetricRenderer renderer;
    VolumetricConfig config;
    config.emissionRamp = {{300.0f, {1, 0, 0}, 1.0f}};
    renderer.Initialize(32, 32, config);

    SoftVec3 emission = renderer.EvaluateEmission(100.0f); // Below 300
    EXPECT_NEAR(emission.x, 0.0f, 0.01f);
    EXPECT_NEAR(emission.y, 0.0f, 0.01f);
    EXPECT_NEAR(emission.z, 0.0f, 0.01f);
}

static void Test_Vol_Emission_AboveMax() {
    VolumetricRenderer renderer;
    VolumetricConfig config;
    config.emissionRamp = {
        {300.0f, {1, 0, 0}, 1.0f},
        {600.0f, {1, 1, 0}, 2.0f}
    };
    config.emissionIntensity = 1.0f;
    renderer.Initialize(32, 32, config);

    SoftVec3 emission = renderer.EvaluateEmission(1000.0f); // Above 600
    // Should use last keyframe
    EXPECT_NEAR(emission.x, 1.0f * 2.0f, 0.01f); // color * intensity * emissionIntensity
    EXPECT_NEAR(emission.y, 1.0f * 2.0f, 0.01f);
}

static void Test_Vol_Emission_Interpolation() {
    VolumetricRenderer renderer;
    VolumetricConfig config;
    config.emissionRamp = {
        {300.0f, {1, 0, 0}, 1.0f},
        {600.0f, {0, 1, 0}, 1.0f}
    };
    config.emissionIntensity = 1.0f;
    renderer.Initialize(32, 32, config);

    // Midpoint: 450 = 50% between 300 and 600
    SoftVec3 emission = renderer.EvaluateEmission(450.0f);
    EXPECT_NEAR(emission.x, 0.5f, 0.1f); // Lerp between red and green
    EXPECT_NEAR(emission.y, 0.5f, 0.1f);
}

static void Test_Vol_Emission_EmptyRamp() {
    VolumetricRenderer renderer;
    VolumetricConfig config;
    config.emissionRamp.clear();
    renderer.Initialize(32, 32, config);

    SoftVec3 emission = renderer.EvaluateEmission(500.0f);
    EXPECT_NEAR(emission.x, 0.0f, 0.01f);
}

// =============================================================================
// Ray Marching Tests
// =============================================================================

static void Test_Vol_MarchRay_EmptyVolume() {
    VolumetricRenderer renderer;
    renderer.Initialize(32, 32);

    VolumeSampler vol = CreateUniformVolume(0.0f, 0.0f, {-5, -5, -5}, {5, 5, 5});

    VolumetricSample result = renderer.MarchRay({0, 0, -10}, {0, 0, 1}, 50.0f, vol);
    EXPECT_NEAR(result.transmittance, 1.0f, 0.01f);
    EXPECT_NEAR(result.color.x, 0.0f, 0.01f);
}

static void Test_Vol_MarchRay_DenseVolume() {
    VolumetricRenderer renderer;
    VolumetricConfig config;
    config.absorptionCoeff = 2.0f;
    config.scatteringCoeff = 0.0f;
    config.densityMultiplier = 1.0f;
    config.stepSize = 0.5f;
    config.maxSteps = 100;
    renderer.Initialize(32, 32, config);

    VolumeSampler vol = CreateUniformVolume(1.0f, 0.0f, {-5, -5, -5}, {5, 5, 5});

    VolumetricSample result = renderer.MarchRay({0, 0, -10}, {0, 0, 1}, 50.0f, vol);
    // After marching through 10 units of density=1.0, absorption=2.0:
    // transmittance ≈ exp(-2.0 * 1.0 * 10) ≈ very small
    EXPECT_LT(result.transmittance, 0.1f);
}

static void Test_Vol_MarchRay_NoSampler() {
    VolumetricRenderer renderer;
    renderer.Initialize(32, 32);

    VolumeSampler vol;
    vol.region.boundsMin = {-5, -5, -5};
    vol.region.boundsMax = {5, 5, 5};
    // sampleDensity is null

    VolumetricSample result = renderer.MarchRay({0, 0, -10}, {0, 0, 1}, 50.0f, vol);
    EXPECT_NEAR(result.transmittance, 1.0f, 0.01f);
}

static void Test_Vol_MarchRay_MissVolume() {
    VolumetricRenderer renderer;
    renderer.Initialize(32, 32);

    VolumeSampler vol = CreateUniformVolume(1.0f, 0.0f, {100, 100, 100}, {110, 110, 110});

    // Ray going toward Z, volume is at (100,100,100)
    VolumetricSample result = renderer.MarchRay({0, 0, -10}, {0, 0, 1}, 50.0f, vol);
    EXPECT_NEAR(result.transmittance, 1.0f, 0.01f);
}

static void Test_Vol_MarchRay_WithEmission() {
    VolumetricRenderer renderer;
    VolumetricConfig config;
    config.absorptionCoeff = 0.1f;
    config.scatteringCoeff = 0.0f;
    config.densityMultiplier = 1.0f;
    config.emissionIntensity = 2.0f;
    config.emissionRamp = {
        {200.0f, {1, 0.5f, 0}, 1.0f},
        {800.0f, {1, 1, 0.5f}, 2.0f}
    };
    config.stepSize = 0.5f;
    config.maxSteps = 100;
    renderer.Initialize(32, 32, config);

    // Volume with both density and high temperature
    VolumeSampler vol = CreateUniformVolume(0.5f, 500.0f, {-5, -5, -5}, {5, 5, 5});

    VolumetricSample result = renderer.MarchRay({0, 0, -10}, {0, 0, 1}, 50.0f, vol);
    // Should have accumulated emission color
    EXPECT_GT(result.color.x, 0.0f);
}

// =============================================================================
// Render / Compositing Tests
// =============================================================================

static void Test_Vol_Render_NoVolumes() {
    VolumetricRenderer renderer;
    renderer.Initialize(32, 32);

    GBuffer gb;
    gb.Initialize(32, 32);
    gb.Clear();

    SoftCamera cam = CreateVolCamera();
    renderer.Render(gb, cam);

    // No volumes → scene should be unchanged (sky gradient)
    // Check that it didn't crash and colors are valid
    SoftColorRGBA8 pixel = gb.GetColor(16, 16);
    EXPECT_TRUE(pixel.r > 0 || pixel.g > 0 || pixel.b > 0); // Sky isn't black
}

static void Test_Vol_Render_WithVolume() {
    VolumetricRenderer renderer;
    VolumetricConfig config;
    config.absorptionCoeff = 0.5f;
    config.scatteringCoeff = 0.3f;
    config.densityMultiplier = 1.0f;
    config.stepSize = 0.5f;
    config.maxSteps = 50;
    renderer.Initialize(32, 32, config);

    GBuffer gb;
    gb.Initialize(32, 32);
    gb.Clear();

    // Fill entire GBuffer with geometry
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x)
            gb.SetDepth(x, y, 20.0f);

    VolumeSampler vol = CreateUniformVolume(0.3f, 0.0f, {-20, -20, -5}, {20, 20, 5});
    renderer.AddVolume(vol);

    SoftCamera cam = CreateVolCamera();
    renderer.Render(gb, cam);

    // Check that the volume buffer has reduced transmittance
    const VolumetricSample* buf = renderer.GetVolumetricBuffer();
    bool hasVolume = false;
    for (int i = 0; i < 32 * 32; ++i) {
        if (buf[i].transmittance < 0.99f) {
            hasVolume = true;
            break;
        }
    }
    EXPECT_TRUE(hasVolume);
}

static void Test_Vol_ConfigAccessors() {
    VolumetricRenderer renderer;
    VolumetricConfig config;
    config.maxSteps = 128;
    config.stepSize = 0.1f;
    config.absorptionCoeff = 3.0f;
    config.phaseG = 0.5f;
    renderer.Initialize(64, 64, config);

    const auto& cfg = renderer.GetConfig();
    EXPECT_EQ(cfg.maxSteps, 128);
    EXPECT_NEAR(cfg.stepSize, 0.1f, 0.001f);
    EXPECT_NEAR(cfg.absorptionCoeff, 3.0f, 0.01f);
    EXPECT_NEAR(cfg.phaseG, 0.5f, 0.01f);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterVolumetricRendererTests() {
    // Initialization
    RUN_TEST("Vol_Initialize", Test_Vol_Initialize);
    RUN_TEST("Vol_InitializeInvalid", Test_Vol_InitializeInvalid);
    RUN_TEST("Vol_AddAndClearVolumes", Test_Vol_AddAndClearVolumes);

    // Ray-AABB
    RUN_TEST("Vol_RayAABB_Hit", Test_Vol_RayAABB_Hit);
    RUN_TEST("Vol_RayAABB_Miss", Test_Vol_RayAABB_Miss);
    RUN_TEST("Vol_RayAABB_Behind", Test_Vol_RayAABB_Behind);
    RUN_TEST("Vol_RayAABB_InsideBox", Test_Vol_RayAABB_InsideBox);

    // Phase Function
    RUN_TEST("Vol_PhaseHG_Isotropic", Test_Vol_PhaseHG_Isotropic);
    RUN_TEST("Vol_PhaseHG_ForwardScatter", Test_Vol_PhaseHG_ForwardScatter);
    RUN_TEST("Vol_PhaseHG_Symmetry", Test_Vol_PhaseHG_Symmetry);

    // Emission
    RUN_TEST("Vol_Emission_BelowThreshold", Test_Vol_Emission_BelowThreshold);
    RUN_TEST("Vol_Emission_AboveMax", Test_Vol_Emission_AboveMax);
    RUN_TEST("Vol_Emission_Interpolation", Test_Vol_Emission_Interpolation);
    RUN_TEST("Vol_Emission_EmptyRamp", Test_Vol_Emission_EmptyRamp);

    // Ray Marching
    RUN_TEST("Vol_MarchRay_EmptyVolume", Test_Vol_MarchRay_EmptyVolume);
    RUN_TEST("Vol_MarchRay_DenseVolume", Test_Vol_MarchRay_DenseVolume);
    RUN_TEST("Vol_MarchRay_NoSampler", Test_Vol_MarchRay_NoSampler);
    RUN_TEST("Vol_MarchRay_MissVolume", Test_Vol_MarchRay_MissVolume);
    RUN_TEST("Vol_MarchRay_WithEmission", Test_Vol_MarchRay_WithEmission);

    // Render/Compositing
    RUN_TEST("Vol_Render_NoVolumes", Test_Vol_Render_NoVolumes);
    RUN_TEST("Vol_Render_WithVolume", Test_Vol_Render_WithVolume);
    RUN_TEST("Vol_ConfigAccessors", Test_Vol_ConfigAccessors);
}
