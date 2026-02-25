// =============================================================================
// WulfNet Engine - Global Illumination Tests
// =============================================================================
// Tests for SSAO, indirect lighting, light probes, and the GI system.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Rendering/SoftwareRasterizer/GlobalIllumination.h>
#include <WulfNet/Rendering/SoftwareRasterizer/GBuffer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h>
#include <cmath>

using namespace WulfNet;

// =============================================================================
// Helper: create a minimal GBuffer with some geometry written
// =============================================================================
static GBuffer CreateTestGBuffer(int w, int h) {
    GBuffer gb;
    gb.Initialize(w, h);
    gb.Clear();
    return gb;
}

static void FillGBufferRegion(GBuffer& gb, int x0, int y0, int x1, int y1,
                               SoftColorRGBA8 color, SoftVec3 normal, float depth) {
    // Pack normal
    SoftColorRGBA8 packedN;
    packedN.r = static_cast<uint8_t>((normal.x * 0.5f + 0.5f) * 255.0f);
    packedN.g = static_cast<uint8_t>((normal.y * 0.5f + 0.5f) * 255.0f);
    packedN.b = static_cast<uint8_t>((normal.z * 0.5f + 0.5f) * 255.0f);
    packedN.a = 255;

    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            gb.SetColor(x, y, color);
            gb.SetNormal(x, y, packedN);
            gb.SetDepth(x, y, depth);
        }
    }
}

static SoftCamera CreateTestCamera() {
    SoftCamera cam;
    cam.position = {0, 5, -10};
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

static void Test_GI_Initialize() {
    GlobalIllumination gi;
    EXPECT_TRUE(gi.Initialize(64, 48));
    EXPECT_EQ(gi.GetWidth(), 64);
    EXPECT_EQ(gi.GetHeight(), 48);
}

static void Test_GI_InitializeInvalid() {
    GlobalIllumination gi;
    EXPECT_FALSE(gi.Initialize(0, 48));
    EXPECT_FALSE(gi.Initialize(64, -1));
    EXPECT_FALSE(gi.Initialize(-1, -1));
}

static void Test_GI_DefaultAOIsOne() {
    GlobalIllumination gi;
    gi.Initialize(32, 32);

    // Before Compute(), AO should be 1.0 (no occlusion)
    const float* ao = gi.GetAOBuffer();
    for (int i = 0; i < 32 * 32; ++i) {
        EXPECT_NEAR(ao[i], 1.0f, 0.01f);
    }
}

// =============================================================================
// SSAO Tests
// =============================================================================

static void Test_GI_SSAO_SkyPixelsAreOne() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    config.ssaoEnabled = true;
    config.ssao.sampleCount = 4;
    gi.Initialize(32, 32, config);

    GBuffer gb = CreateTestGBuffer(32, 32);
    // Default GBuffer has max depth (sky) — no clearing needed
    SoftCamera cam = CreateTestCamera();

    gi.Compute(gb, cam);

    // All sky pixels should have AO = 1.0
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            EXPECT_NEAR(gi.SampleAO(x, y), 1.0f, 0.01f);
        }
    }
}

static void Test_GI_SSAO_GeometryGetsOcclusion() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    config.ssaoEnabled = true;
    config.ssao.sampleCount = 16;
    config.ssao.radius = 2.0f;
    config.ssao.blurPasses = 0; // No blur for deterministic testing
    config.indirect.enabled = false;
    gi.Initialize(64, 64, config);

    GBuffer gb = CreateTestGBuffer(64, 64);
    SoftCamera cam = CreateTestCamera();

    // Fill entire GBuffer with geometry at varying depths
    // Center region is close, surroundings are far → creates occlusion at edges
    FillGBufferRegion(gb, 0, 0, 64, 64, {128, 128, 128, 255}, {0, 1, 0}, 10.0f);
    // Create a corner pocket (closer depth) that would occlude neighbors
    FillGBufferRegion(gb, 0, 0, 16, 16, {128, 128, 128, 255}, {0, 1, 0}, 5.0f);

    gi.Compute(gb, cam);

    // AO values should exist in [0,1] range for geometry pixels
    const float* ao = gi.GetAOBuffer();
    bool hasNonOne = false;
    for (int i = 0; i < 64 * 64; ++i) {
        EXPECT_TRUE(ao[i] >= 0.0f && ao[i] <= 1.0f);
        if (ao[i] < 0.99f) hasNonOne = true;
    }
    // With depth discontinuity, SOME pixels should have AO < 1.0
    EXPECT_TRUE(hasNonOne);
}

static void Test_GI_SSAO_Disabled() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    config.ssaoEnabled = false;
    gi.Initialize(32, 32, config);

    GBuffer gb = CreateTestGBuffer(32, 32);
    FillGBufferRegion(gb, 0, 0, 32, 32, {200, 100, 50, 255}, {0, 1, 0}, 10.0f);
    SoftCamera cam = CreateTestCamera();

    gi.Compute(gb, cam);

    // With SSAO disabled, all should be 1.0
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            EXPECT_NEAR(gi.SampleAO(x, y), 1.0f, 0.01f);
        }
    }
}

static void Test_GI_SSAO_SampleAOBounds() {
    GlobalIllumination gi;
    gi.Initialize(16, 16);

    // Out-of-bounds should return 1.0
    EXPECT_NEAR(gi.SampleAO(-1, 0), 1.0f, 0.01f);
    EXPECT_NEAR(gi.SampleAO(0, -1), 1.0f, 0.01f);
    EXPECT_NEAR(gi.SampleAO(16, 0), 1.0f, 0.01f);
    EXPECT_NEAR(gi.SampleAO(0, 16), 1.0f, 0.01f);
}

// =============================================================================
// Blur Tests
// =============================================================================

static void Test_GI_BlurAOBuffer() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    config.ssao.blurKernelSize = 3;
    config.ssao.blurPasses = 0; // We'll call manually
    gi.Initialize(16, 16, config);

    // Set a spike in the AO buffer
    float* ao = gi.GetAOBuffer();
    for (int i = 0; i < 16 * 16; ++i) ao[i] = 1.0f;
    ao[8 * 16 + 8] = 0.0f; // center dark

    gi.BlurAOBuffer();

    // After blur, the center should be partially smoothed
    float blurred = ao[8 * 16 + 8];
    EXPECT_GT(blurred, 0.0f);  // Was 0, now partially raised
    EXPECT_LT(blurred, 1.0f);  // But not fully 1.0
}

static void Test_GI_BlurMultiplePasses() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    config.ssao.blurKernelSize = 3;
    config.ssao.blurPasses = 0;
    gi.Initialize(16, 16, config);

    float* ao = gi.GetAOBuffer();
    for (int i = 0; i < 16 * 16; ++i) ao[i] = 1.0f;
    ao[8 * 16 + 8] = 0.0f;

    gi.BlurAOBuffer();
    float afterOne = ao[8 * 16 + 8];

    // After one blur, the center dark pixel should be partially averaged
    EXPECT_GT(afterOne, 0.0f);

    // Check that a pixel 2 away from center was NOT affected by the first blur
    // (kernel size 3 only reaches 1 pixel away)
    float edge2AfterOne = ao[6 * 16 + 8]; // 2 pixels above center

    gi.BlurAOBuffer();
    float edge2AfterTwo = ao[6 * 16 + 8]; // 2 pixels above center after 2nd blur

    // Second blur pass should spread the effect to pixels that weren't touched by the first
    EXPECT_LT(edge2AfterTwo, edge2AfterOne);
}

// =============================================================================
// Indirect Lighting Tests
// =============================================================================

static void Test_GI_IndirectLighting_SkyReturnZero() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    config.ssaoEnabled = false;
    config.indirect.enabled = true;
    config.indirect.sampleCount = 4;
    gi.Initialize(32, 32, config);

    GBuffer gb = CreateTestGBuffer(32, 32);
    // All sky (default depth > 9999)
    SoftCamera cam = CreateTestCamera();

    gi.Compute(gb, cam);

    SoftVec3 ind = gi.SampleIndirect(16, 16);
    EXPECT_NEAR(ind.x, 0.0f, 0.01f);
    EXPECT_NEAR(ind.y, 0.0f, 0.01f);
    EXPECT_NEAR(ind.z, 0.0f, 0.01f);
}

static void Test_GI_IndirectLighting_Disabled() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    config.indirect.enabled = false;
    gi.Initialize(32, 32, config);

    GBuffer gb = CreateTestGBuffer(32, 32);
    FillGBufferRegion(gb, 0, 0, 32, 32, {255, 0, 0, 255}, {0, 1, 0}, 10.0f);
    SoftCamera cam = CreateTestCamera();

    gi.Compute(gb, cam);

    SoftVec3 ind = gi.SampleIndirect(16, 16);
    EXPECT_NEAR(ind.x, 0.0f, 0.01f);
    EXPECT_NEAR(ind.y, 0.0f, 0.01f);
    EXPECT_NEAR(ind.z, 0.0f, 0.01f);
}

static void Test_GI_IndirectLighting_BoundsCheck() {
    GlobalIllumination gi;
    gi.Initialize(16, 16);

    SoftVec3 oob = gi.SampleIndirect(-1, -1);
    EXPECT_NEAR(oob.x, 0.0f, 0.01f);
    EXPECT_NEAR(oob.y, 0.0f, 0.01f);
    EXPECT_NEAR(oob.z, 0.0f, 0.01f);
}

static void Test_GI_IndirectLighting_WithGeometry() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    config.ssaoEnabled = false;
    config.indirect.enabled = true;
    config.indirect.sampleCount = 8;
    config.indirect.bounceRadius = 2.0f;
    config.indirect.bounceIntensity = 1.0f;
    gi.Initialize(64, 64, config);

    GBuffer gb = CreateTestGBuffer(64, 64);
    // Fill with bright red surface — normal facing UP so hemisphere bounce produces valid weights
    FillGBufferRegion(gb, 0, 0, 64, 64, {255, 0, 0, 255}, {0, 1, 0}, 10.0f);
    SoftCamera cam = CreateTestCamera();

    gi.Compute(gb, cam);

    // The indirect buffer should have non-zero values
    const SoftVec3* indirect = gi.GetIndirectBuffer();
    bool hasIndirect = false;
    for (int i = 0; i < 64 * 64; ++i) {
        if (indirect[i].x > 0.001f || indirect[i].y > 0.001f || indirect[i].z > 0.001f) {
            hasIndirect = true;
            break;
        }
    }
    EXPECT_TRUE(hasIndirect);
}

// =============================================================================
// Light Probe Tests
// =============================================================================

static void Test_GI_LightProbe_Evaluate() {
    LightProbe probe;
    probe.position = {0, 0, 0};
    probe.radius = 10.0f;

    // Set L0 (constant ambient) to white
    probe.shCoeffs[0] = {1.0f, 1.0f, 1.0f};
    // No L1 terms
    probe.shCoeffs[1] = {};
    probe.shCoeffs[2] = {};
    probe.shCoeffs[3] = {};

    SoftVec3 irr = probe.Evaluate({0, 1, 0});
    // L0 scaled by 0.282095
    EXPECT_NEAR(irr.x, 0.282095f, 0.01f);
    EXPECT_NEAR(irr.y, 0.282095f, 0.01f);
    EXPECT_NEAR(irr.z, 0.282095f, 0.01f);
}

static void Test_GI_LightProbe_DirectionalSH() {
    LightProbe probe;
    probe.position = {0, 0, 0};
    probe.radius = 10.0f;

    // L0 = 0, L1_y = bright (light from above)
    probe.shCoeffs[0] = {0.5f, 0.5f, 0.5f}; // ambient
    probe.shCoeffs[1] = {1.0f, 1.0f, 1.0f}; // Y direction

    // Normal pointing up should get more light
    SoftVec3 upIrr = probe.Evaluate({0, 1, 0});
    // Normal pointing down should get less
    SoftVec3 downIrr = probe.Evaluate({0, -1, 0});

    EXPECT_GT(upIrr.x, downIrr.x);
}

static void Test_GI_AddProbe() {
    GlobalIllumination gi;
    gi.Initialize(32, 32);

    LightProbe probe;
    probe.position = {0, 5, 0};
    probe.radius = 20.0f;
    probe.shCoeffs[0] = {0.3f, 0.3f, 0.3f};

    gi.AddProbe(probe);
    EXPECT_EQ(gi.GetProbeCount(), 1);

    gi.AddProbe(probe);
    EXPECT_EQ(gi.GetProbeCount(), 2);
}

static void Test_GI_EvaluateProbes_InRange() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    LightProbe probe;
    probe.position = {0, 0, 0};
    probe.radius = 10.0f;
    probe.shCoeffs[0] = {1.0f, 0.5f, 0.25f};
    config.probes.push_back(probe);
    config.probesEnabled = true;
    gi.Initialize(32, 32, config);

    // Point inside the probe radius
    SoftVec3 result = gi.EvaluateProbes({2, 0, 0}, {0, 1, 0});
    EXPECT_GT(result.x, 0.0f);
    EXPECT_GT(result.y, 0.0f);
    EXPECT_GT(result.z, 0.0f);
}

static void Test_GI_EvaluateProbes_OutOfRange() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    LightProbe probe;
    probe.position = {0, 0, 0};
    probe.radius = 5.0f;
    probe.shCoeffs[0] = {1.0f, 1.0f, 1.0f};
    config.probes.push_back(probe);
    gi.Initialize(32, 32, config);

    // Point outside the probe radius
    SoftVec3 result = gi.EvaluateProbes({100, 0, 0}, {0, 1, 0});
    EXPECT_NEAR(result.x, 0.0f, 0.01f);
    EXPECT_NEAR(result.y, 0.0f, 0.01f);
    EXPECT_NEAR(result.z, 0.0f, 0.01f);
}

static void Test_GI_EvaluateProbes_Falloff() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    LightProbe probe;
    probe.position = {0, 0, 0};
    probe.radius = 10.0f;
    probe.shCoeffs[0] = {1.0f, 1.0f, 1.0f};
    config.probes.push_back(probe);
    gi.Initialize(32, 32, config);

    // Close to probe should have more contribution
    SoftVec3 close = gi.EvaluateProbes({1, 0, 0}, {0, 1, 0});
    SoftVec3 far   = gi.EvaluateProbes({8, 0, 0}, {0, 1, 0});

    EXPECT_GT(close.x, far.x); // Closer = more light
}

// =============================================================================
// Config Tests
// =============================================================================

static void Test_GI_ConfigAccessors() {
    GlobalIllumination gi;
    GlobalIlluminationConfig config;
    config.ssaoEnabled = false;
    config.ssao.sampleCount = 32;
    config.ssao.radius = 3.0f;
    config.indirect.bounceIntensity = 0.8f;
    gi.Initialize(64, 64, config);

    const auto& cfg = gi.GetConfig();
    EXPECT_FALSE(cfg.ssaoEnabled);
    EXPECT_EQ(cfg.ssao.sampleCount, 32);
    EXPECT_NEAR(cfg.ssao.radius, 3.0f, 0.01f);
    EXPECT_NEAR(cfg.indirect.bounceIntensity, 0.8f, 0.01f);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterGlobalIlluminationTests() {
    // Initialization
    RUN_TEST("GI_Initialize", Test_GI_Initialize);
    RUN_TEST("GI_InitializeInvalid", Test_GI_InitializeInvalid);
    RUN_TEST("GI_DefaultAOIsOne", Test_GI_DefaultAOIsOne);

    // SSAO
    RUN_TEST("GI_SSAO_SkyPixelsAreOne", Test_GI_SSAO_SkyPixelsAreOne);
    RUN_TEST("GI_SSAO_GeometryGetsOcclusion", Test_GI_SSAO_GeometryGetsOcclusion);
    RUN_TEST("GI_SSAO_Disabled", Test_GI_SSAO_Disabled);
    RUN_TEST("GI_SSAO_SampleAOBounds", Test_GI_SSAO_SampleAOBounds);

    // Blur
    RUN_TEST("GI_BlurAOBuffer", Test_GI_BlurAOBuffer);
    RUN_TEST("GI_BlurMultiplePasses", Test_GI_BlurMultiplePasses);

    // Indirect
    RUN_TEST("GI_IndirectLighting_SkyReturnZero", Test_GI_IndirectLighting_SkyReturnZero);
    RUN_TEST("GI_IndirectLighting_Disabled", Test_GI_IndirectLighting_Disabled);
    RUN_TEST("GI_IndirectLighting_BoundsCheck", Test_GI_IndirectLighting_BoundsCheck);
    RUN_TEST("GI_IndirectLighting_WithGeometry", Test_GI_IndirectLighting_WithGeometry);

    // Light Probes
    RUN_TEST("GI_LightProbe_Evaluate", Test_GI_LightProbe_Evaluate);
    RUN_TEST("GI_LightProbe_DirectionalSH", Test_GI_LightProbe_DirectionalSH);
    RUN_TEST("GI_AddProbe", Test_GI_AddProbe);
    RUN_TEST("GI_EvaluateProbes_InRange", Test_GI_EvaluateProbes_InRange);
    RUN_TEST("GI_EvaluateProbes_OutOfRange", Test_GI_EvaluateProbes_OutOfRange);
    RUN_TEST("GI_EvaluateProbes_Falloff", Test_GI_EvaluateProbes_Falloff);

    // Config
    RUN_TEST("GI_ConfigAccessors", Test_GI_ConfigAccessors);
}
