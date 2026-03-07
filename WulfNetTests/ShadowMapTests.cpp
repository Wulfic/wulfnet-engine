// =============================================================================
// WulfNet Engine - Shadow Mapping Tests
// =============================================================================
// Tests for ShadowCascade, PointLightShadow, and ShadowSystem.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Rendering/Lighting/ShadowMap.h>
#include <WulfNet/Rendering/Types/RenderTypes.h>
#include <cmath>
#include <limits>

using namespace WulfNet;

// =============================================================================
// ShadowCascade Tests
// =============================================================================

static void Test_ShadowCascade_Initialize() {
    ShadowCascade cascade;
    EXPECT_TRUE(cascade.Initialize(256));
    EXPECT_EQ(cascade.GetResolution(), 256);
    EXPECT_TRUE(cascade.GetDepthBuffer() != nullptr);
}

static void Test_ShadowCascade_InitializeInvalid() {
    ShadowCascade cascade;
    EXPECT_FALSE(cascade.Initialize(0));
    EXPECT_FALSE(cascade.Initialize(-1));
}

static void Test_ShadowCascade_Clear() {
    ShadowCascade cascade;
    cascade.Initialize(64);

    // Write some depth
    cascade.ComputeLightMatrix({0, -1, 0}, {0, 0, 0}, 10.0f, 0.1f, 100.0f);
    cascade.WriteDepth(0.0f, 0.0f, 0.5f);

    // Clear should reset all depths to max
    cascade.Clear();
    const float* buf = cascade.GetDepthBuffer();
    bool allMax = true;
    for (int i = 0; i < 64 * 64; ++i) {
        if (buf[i] < 9999.0f) { allMax = false; break; }
    }
    EXPECT_TRUE(allMax);
}

static void Test_ShadowCascade_ComputeLightMatrix() {
    ShadowCascade cascade;
    cascade.Initialize(128);

    // Light pointing down (-Y)
    cascade.ComputeLightMatrix({0, -1, 0}, {0, 0, 0}, 20.0f, 0.1f, 50.0f);

    EXPECT_NEAR(cascade.GetOrthoSize(), 20.0f, 0.001f);

    // Light forward should be normalized (0,-1,0)
    Vec3 fwd = cascade.GetLightForward();
    EXPECT_NEAR(fwd.Length(), 1.0f, 0.01f);
    EXPECT_NEAR(fwd.y, -1.0f, 0.01f);
}

static void Test_ShadowCascade_WriteAndReadDepth() {
    ShadowCascade cascade;
    cascade.Initialize(128);
    cascade.ComputeLightMatrix({0, -1, 0.5f}, {0, 0, 0}, 20.0f, 0.1f, 100.0f);

    // Write at center: NDC (0,0)
    EXPECT_TRUE(cascade.WriteDepth(0.0f, 0.0f, 0.3f));

    // Writing a farther value at same pixel should fail
    EXPECT_FALSE(cascade.WriteDepth(0.0f, 0.0f, 0.5f));

    // Writing a closer value should succeed
    EXPECT_TRUE(cascade.WriteDepth(0.0f, 0.0f, 0.1f));
}

static void Test_ShadowCascade_WriteOutOfBounds() {
    ShadowCascade cascade;
    cascade.Initialize(64);
    cascade.ComputeLightMatrix({0, -1, 0}, {0, 0, 0}, 10.0f, 0.1f, 50.0f);

    // NDC values outside [-1,1] should fail
    EXPECT_FALSE(cascade.WriteDepth(2.0f, 0.0f, 0.5f));
    EXPECT_FALSE(cascade.WriteDepth(0.0f, -1.5f, 0.5f));

    // Invalid depth
    EXPECT_FALSE(cascade.WriteDepth(0.0f, 0.0f, -0.1f));
    EXPECT_FALSE(cascade.WriteDepth(0.0f, 0.0f, 1.1f));
}

static void Test_ShadowCascade_SampleShadow_Lit() {
    ShadowCascade cascade;
    cascade.Initialize(128);
    cascade.ComputeLightMatrix({0, -1, 0}, {0, 10, 0}, 30.0f, 0.1f, 100.0f);

    // No occluders written — everything should be lit
    Vec3 testPoint = {0.0f, 5.0f, 0.0f};
    float shadow = cascade.SampleShadow(testPoint, 0.005f);
    EXPECT_NEAR(shadow, 1.0f, 0.01f);
}

static void Test_ShadowCascade_SampleShadow_OutsideFrustum() {
    ShadowCascade cascade;
    cascade.Initialize(64);
    cascade.ComputeLightMatrix({0, -1, 0}, {0, 0, 0}, 10.0f, 0.1f, 50.0f);

    // Far outside the cascade's ortho box should return lit
    Vec3 farAway = {500.0f, 0.0f, 500.0f};
    EXPECT_NEAR(cascade.SampleShadow(farAway, 0.005f), 1.0f, 0.01f);
}

static void Test_ShadowCascade_WorldToLightNDC() {
    ShadowCascade cascade;
    cascade.Initialize(64);

    // Light pointing straight down
    cascade.ComputeLightMatrix({0, -1, 0}, {0, 0, 0}, 10.0f, 0.1f, 50.0f);

    // The focus center (0,0,0) projected into light NDC should be near center
    Vec3 ndc = cascade.WorldToLightNDC({0, 0, 0});
    EXPECT_NEAR(ndc.x, 0.0f, 0.5f);
    EXPECT_NEAR(ndc.y, 0.0f, 0.5f);
    // Depth should be between 0 and 1
    EXPECT_TRUE(ndc.z >= 0.0f && ndc.z <= 1.0f);
}

// =============================================================================
// PointLightShadow Tests
// =============================================================================

static void Test_PointLightShadow_Initialize() {
    PointLightShadow shadow;
    EXPECT_TRUE(shadow.Initialize(128));
    EXPECT_EQ(shadow.GetResolution(), 128);

    // All 6 faces should have buffers
    for (int f = 0; f < 6; ++f) {
        EXPECT_TRUE(shadow.GetFaceDepthBuffer(f) != nullptr);
    }
}

static void Test_PointLightShadow_InitializeInvalid() {
    PointLightShadow shadow;
    EXPECT_FALSE(shadow.Initialize(0));
    EXPECT_FALSE(shadow.Initialize(-5));
}

static void Test_PointLightShadow_DirectionToFace() {
    // +X, -X, +Y, -Y, +Z, -Z
    EXPECT_EQ(PointLightShadow::DirectionToFace({1, 0, 0}), 0);
    EXPECT_EQ(PointLightShadow::DirectionToFace({-1, 0, 0}), 1);
    EXPECT_EQ(PointLightShadow::DirectionToFace({0, 1, 0}), 2);
    EXPECT_EQ(PointLightShadow::DirectionToFace({0, -1, 0}), 3);
    EXPECT_EQ(PointLightShadow::DirectionToFace({0, 0, 1}), 4);
    EXPECT_EQ(PointLightShadow::DirectionToFace({0, 0, -1}), 5);
}

static void Test_PointLightShadow_DirectionDiagonal() {
    // Diagonal: dominant axis determines face
    int face = PointLightShadow::DirectionToFace({0.9f, 0.1f, 0.1f});
    EXPECT_EQ(face, 0); // +X dominant

    face = PointLightShadow::DirectionToFace({0.1f, -0.9f, 0.1f});
    EXPECT_EQ(face, 3); // -Y dominant
}

static void Test_PointLightShadow_SetLightPosition() {
    PointLightShadow shadow;
    shadow.Initialize(64);
    shadow.SetLightPosition({5, 10, 3}, 25.0f);
    EXPECT_NEAR(shadow.GetPosition().x, 5.0f, 0.01f);
    EXPECT_NEAR(shadow.GetPosition().y, 10.0f, 0.01f);
    EXPECT_NEAR(shadow.GetRange(), 25.0f, 0.01f);
}

static void Test_PointLightShadow_Clear() {
    PointLightShadow shadow;
    shadow.Initialize(32);
    shadow.SetLightPosition({0, 0, 0}, 10.0f);

    // Write depth on face 0
    shadow.WriteDepth(0, 0.0f, 0.0f, 0.5f);

    // Clear
    shadow.Clear();

    // All should be max again
    const float* buf = shadow.GetFaceDepthBuffer(0);
    EXPECT_TRUE(buf[16 * 32 + 16] > 9999.0f); // center pixel
}

static void Test_PointLightShadow_WriteDepth() {
    PointLightShadow shadow;
    shadow.Initialize(64);
    shadow.SetLightPosition({0, 0, 0}, 10.0f);

    // Write at center of face 0
    EXPECT_TRUE(shadow.WriteDepth(0, 0.0f, 0.0f, 0.5f));
    // Farther value should not overwrite
    EXPECT_FALSE(shadow.WriteDepth(0, 0.0f, 0.0f, 0.7f));
    // Closer value should overwrite
    EXPECT_TRUE(shadow.WriteDepth(0, 0.0f, 0.0f, 0.2f));
}

static void Test_PointLightShadow_SampleLit() {
    PointLightShadow shadow;
    shadow.Initialize(64);
    shadow.SetLightPosition({0, 0, 0}, 20.0f);

    // No occluders — everything in range should be lit
    Vec3 testPoint = {5, 0, 0};
    float result = shadow.SampleShadow(testPoint, 0.005f);
    EXPECT_NEAR(result, 1.0f, 0.01f);
}

static void Test_PointLightShadow_SampleOutOfRange() {
    PointLightShadow shadow;
    shadow.Initialize(64);
    shadow.SetLightPosition({0, 0, 0}, 10.0f);

    // Point beyond range
    Vec3 farPoint = {100, 0, 0};
    float result = shadow.SampleShadow(farPoint, 0.005f);
    EXPECT_NEAR(result, 1.0f, 0.01f); // Out of range = lit
}

static void Test_PointLightShadow_InvalidFace() {
    PointLightShadow shadow;
    shadow.Initialize(64);
    EXPECT_TRUE(shadow.GetFaceDepthBuffer(-1) == nullptr);
    EXPECT_TRUE(shadow.GetFaceDepthBuffer(6) == nullptr);
    EXPECT_FALSE(shadow.WriteDepth(-1, 0, 0, 0.5f));
    EXPECT_FALSE(shadow.WriteDepth(6, 0, 0, 0.5f));
}

// =============================================================================
// ShadowSystem Tests
// =============================================================================

static void Test_ShadowSystem_Initialize() {
    ShadowSystem system;
    ShadowSystemConfig config;
    config.numCascades = 3;
    config.cascadeResolution = 128;

    EXPECT_TRUE(system.Initialize(config));
    EXPECT_EQ(system.GetCascadeCount(), 3);
}

static void Test_ShadowSystem_InitializeDefaults() {
    ShadowSystem system;
    EXPECT_TRUE(system.Initialize());
    EXPECT_EQ(system.GetCascadeCount(), 3); // default
}

static void Test_ShadowSystem_CascadeSplits() {
    ShadowSystem system;
    ShadowSystemConfig config;
    config.numCascades = 3;
    config.maxShadowDistance = 100.0f;
    config.cascadeSplitLambda = 0.5f;
    system.Initialize(config);

    SoftCamera camera;
    camera.position = {0, 0, 0};
    camera.forward = {0, 0, 1};
    camera.nearPlane = 0.1f;
    camera.farPlane = 1000.0f;

    system.ComputeCascadeSplits(camera);

    const auto& splits = system.GetCascadeSplits();
    EXPECT_EQ(static_cast<int>(splits.size()), 4); // numCascades + 1

    // Splits should be monotonically increasing
    for (int i = 1; i < static_cast<int>(splits.size()); ++i) {
        EXPECT_GT(splits[i], splits[i - 1]);
    }

    // First split is near plane, last is max shadow distance
    EXPECT_NEAR(splits[0], 0.1f, 0.01f);
    EXPECT_NEAR(splits[3], 100.0f, 0.01f);
}

static void Test_ShadowSystem_SelectCascade() {
    ShadowSystem system;
    ShadowSystemConfig config;
    config.numCascades = 3;
    config.maxShadowDistance = 90.0f;
    config.cascadeSplitLambda = 0.0f; // Pure linear for predictability
    system.Initialize(config);

    SoftCamera camera;
    camera.position = {0, 0, 0};
    camera.forward = {0, 0, 1};
    camera.nearPlane = 0.1f;
    camera.farPlane = 1000.0f;

    system.ComputeCascadeSplits(camera);

    // Linear splits: ~0.1, ~30.07, ~60.03, ~90.0
    // A point at z=10 should be cascade 0
    int c0 = system.SelectCascade({0, 0, 10}, camera);
    EXPECT_EQ(c0, 0);

    // A point at z=50 should be cascade 1
    int c1 = system.SelectCascade({0, 0, 50}, camera);
    EXPECT_EQ(c1, 1);

    // A point at z=80 should be cascade 2
    int c2 = system.SelectCascade({0, 0, 80}, camera);
    EXPECT_EQ(c2, 2);
}

static void Test_ShadowSystem_RenderDirectionalShadows() {
    ShadowSystem system;
    ShadowSystemConfig config;
    config.numCascades = 2;
    config.cascadeResolution = 64;
    config.maxShadowDistance = 100.0f;
    system.Initialize(config);

    // Create a simple ground plane mesh - large and centered near the camera
    SoftMesh plane;
    plane.name = "Ground";
    plane.vertices = {
        {{-5, 0, -5}, {0, 1, 0}, {0, 0}},
        {{ 5, 0, -5}, {0, 1, 0}, {1, 0}},
        {{ 5, 0,  5}, {0, 1, 0}, {1, 1}},
        {{-5, 0,  5}, {0, 1, 0}, {0, 1}}
    };
    plane.indices = {0, 1, 2, 0, 2, 3};
    plane.ComputeFaceNormals();

    std::vector<SoftMesh> meshes = {plane};
    SoftTransform xform;
    xform.meshIndex = 0;
    xform.position = {0, 0, 15}; // Centered in frustum at Z=10..20

    SoftDirectionalLight sun;
    sun.direction = {0.0f, -1.0f, 0.0f}; // Straight down

    SoftCamera camera;
    camera.position = {0, 0, 0};
    camera.forward = {0, 0, 1};
    camera.nearPlane = 0.1f;
    camera.farPlane = 200.0f;

    system.RenderDirectionalShadows(sun, camera, &xform, 1, meshes);

    // After rendering, some depth values should have been written in at least one cascade
    bool hasWrittenPixels = false;
    for (int c = 0; c < config.numCascades && !hasWrittenPixels; ++c) {
        const ShadowCascade& cascade = system.GetCascade(c);
        const float* buf = cascade.GetDepthBuffer();
        int res = cascade.GetResolution();
        for (int i = 0; i < res * res; ++i) {
            if (buf[i] < 9999.0f) { hasWrittenPixels = true; break; }
        }
    }
    EXPECT_TRUE(hasWrittenPixels);
}

static void Test_ShadowSystem_SampleDirectionalShadow_NoOccluders() {
    ShadowSystem system;
    ShadowSystemConfig config;
    config.numCascades = 2;
    config.cascadeResolution = 64;
    config.maxShadowDistance = 100.0f;
    system.Initialize(config);

    SoftCamera camera;
    camera.position = {0, 0, 0};
    camera.forward = {0, 0, 1};
    camera.nearPlane = 0.1f;
    camera.farPlane = 200.0f;

    // Render with no objects
    SoftDirectionalLight sun;
    sun.direction = {0, -1, 0};
    std::vector<SoftMesh> meshes;
    system.RenderDirectionalShadows(sun, camera, nullptr, 0, meshes);

    // Should be fully lit
    float shadow = system.SampleDirectionalShadow({0, 0, 10});
    EXPECT_NEAR(shadow, 1.0f, 0.01f);
}

static void Test_ShadowSystem_PointLightShadow() {
    ShadowSystem system;
    ShadowSystemConfig config;
    config.pointLightResolution = 32;
    system.Initialize(config);

    SoftPointLight light;
    light.position = {0, 5, 0};
    light.range = 20.0f;

    // Create a small cube occluder
    SoftMesh cube = SoftMeshGen::CreateCube(2.0f);
    std::vector<SoftMesh> meshes = {cube};
    SoftTransform xform;
    xform.meshIndex = 0;
    xform.position = {0, 2, 0}; // Below the light

    system.RenderPointLightShadow(0, light, &xform, 1, meshes);

    EXPECT_EQ(system.GetPointLightShadowCount(), 1);

    // Sample shadow at a point that's NOT occluded (off to the side)
    float litSample = system.SamplePointLightShadow(0, {10, 5, 0});
    EXPECT_NEAR(litSample, 1.0f, 0.01f);
}

static void Test_ShadowSystem_ClearAll() {
    ShadowSystem system;
    ShadowSystemConfig config;
    config.numCascades = 2;
    config.cascadeResolution = 32;
    system.Initialize(config);

    // Write something
    SoftMesh cube = SoftMeshGen::CreateCube(1.0f);
    std::vector<SoftMesh> meshes = {cube};
    SoftTransform xform;
    xform.meshIndex = 0;
    xform.position = {0, 0, 10};

    SoftDirectionalLight sun;
    sun.direction = {0, -1, 0};

    SoftCamera camera;
    camera.position = {0, 0, 0};
    camera.forward = {0, 0, 1};

    system.RenderDirectionalShadows(sun, camera, &xform, 1, meshes);
    system.ClearAll();

    // All cascade depths should be max after clear
    const float* buf = system.GetCascade(0).GetDepthBuffer();
    int res = system.GetCascade(0).GetResolution();
    bool allMax = true;
    for (int i = 0; i < res * res; ++i) {
        if (buf[i] < 9999.0f) { allMax = false; break; }
    }
    EXPECT_TRUE(allMax);
}

static void Test_ShadowSystem_MultiplePointLights() {
    ShadowSystem system;
    ShadowSystemConfig config;
    config.pointLightResolution = 32;
    system.Initialize(config);

    SoftPointLight light0, light1;
    light0.position = {0, 5, 0};   light0.range = 15.0f;
    light1.position = {10, 5, 10}; light1.range = 15.0f;

    std::vector<SoftMesh> meshes;
    // Render shadows for both lights (no occluders)
    system.RenderPointLightShadow(0, light0, nullptr, 0, meshes);
    system.RenderPointLightShadow(1, light1, nullptr, 0, meshes);

    EXPECT_EQ(system.GetPointLightShadowCount(), 2);

    // Both should be lit (no occluders)
    EXPECT_NEAR(system.SamplePointLightShadow(0, {0, 2, 0}), 1.0f, 0.01f);
    EXPECT_NEAR(system.SamplePointLightShadow(1, {10, 2, 10}), 1.0f, 0.01f);
}

static void Test_ShadowSystem_InvalidPointLightIndex() {
    ShadowSystem system;
    system.Initialize();

    // Non-existent point light should return lit
    float result = system.SamplePointLightShadow(99, {0, 0, 0});
    EXPECT_NEAR(result, 1.0f, 0.01f);
}

static void Test_ShadowSystem_ConfigAccessors() {
    ShadowSystem system;
    ShadowSystemConfig config;
    config.numCascades = 4;
    config.cascadeResolution = 256;
    config.maxShadowDistance = 200.0f;
    config.shadowBias = 0.01f;
    config.pcfSamples = 3;
    system.Initialize(config);

    const auto& cfg = system.GetConfig();
    EXPECT_EQ(cfg.numCascades, 4);
    EXPECT_EQ(cfg.cascadeResolution, 256);
    EXPECT_NEAR(cfg.maxShadowDistance, 200.0f, 0.01f);
    EXPECT_NEAR(cfg.shadowBias, 0.01f, 0.001f);
    EXPECT_EQ(cfg.pcfSamples, 3);
}

static void Test_ShadowCascade_DepthComparison() {
    // Verify that an occluder closer to the light creates shadow
    ShadowCascade cascade;
    cascade.Initialize(128);
    cascade.ComputeLightMatrix({0, -1, 0}, {0, 0, 0}, 20.0f, 0.1f, 100.0f);

    // Write a shallow depth (close to light) at center
    cascade.WriteDepth(0.0f, 0.0f, 0.2f);

    // A point behind the occluder (deeper) should be in shadow
    // Manually check: the stored depth is 0.2, a fragment at depth 0.5 should fail
    // We need to construct a world pos that maps to NDC center with depth 0.5
    Vec3 lightPos = cascade.GetLightPosition();
    Vec3 lightFwd = cascade.GetLightForward();
    float nearC = cascade.GetNearClip();
    float farC = cascade.GetFarClip();

    // A point along light forward at normalized depth 0.5
    float worldDepth = nearC + 0.5f * (farC - nearC);
    Vec3 shadowedPoint = lightPos + lightFwd * worldDepth;

    float shadow = cascade.SampleShadow(shadowedPoint, 0.005f);
    EXPECT_NEAR(shadow, 0.0f, 0.01f); // Should be shadowed
}

// =============================================================================
// Registration
// =============================================================================

void RegisterShadowMapTests() {
    // ShadowCascade
    RUN_TEST("ShadowCascade_Initialize", Test_ShadowCascade_Initialize);
    RUN_TEST("ShadowCascade_InitializeInvalid", Test_ShadowCascade_InitializeInvalid);
    RUN_TEST("ShadowCascade_Clear", Test_ShadowCascade_Clear);
    RUN_TEST("ShadowCascade_ComputeLightMatrix", Test_ShadowCascade_ComputeLightMatrix);
    RUN_TEST("ShadowCascade_WriteAndReadDepth", Test_ShadowCascade_WriteAndReadDepth);
    RUN_TEST("ShadowCascade_WriteOutOfBounds", Test_ShadowCascade_WriteOutOfBounds);
    RUN_TEST("ShadowCascade_SampleShadow_Lit", Test_ShadowCascade_SampleShadow_Lit);
    RUN_TEST("ShadowCascade_SampleShadow_OutsideFrustum", Test_ShadowCascade_SampleShadow_OutsideFrustum);
    RUN_TEST("ShadowCascade_WorldToLightNDC", Test_ShadowCascade_WorldToLightNDC);
    RUN_TEST("ShadowCascade_DepthComparison", Test_ShadowCascade_DepthComparison);

    // PointLightShadow
    RUN_TEST("PointLightShadow_Initialize", Test_PointLightShadow_Initialize);
    RUN_TEST("PointLightShadow_InitializeInvalid", Test_PointLightShadow_InitializeInvalid);
    RUN_TEST("PointLightShadow_DirectionToFace", Test_PointLightShadow_DirectionToFace);
    RUN_TEST("PointLightShadow_DirectionDiagonal", Test_PointLightShadow_DirectionDiagonal);
    RUN_TEST("PointLightShadow_SetLightPosition", Test_PointLightShadow_SetLightPosition);
    RUN_TEST("PointLightShadow_Clear", Test_PointLightShadow_Clear);
    RUN_TEST("PointLightShadow_WriteDepth", Test_PointLightShadow_WriteDepth);
    RUN_TEST("PointLightShadow_SampleLit", Test_PointLightShadow_SampleLit);
    RUN_TEST("PointLightShadow_SampleOutOfRange", Test_PointLightShadow_SampleOutOfRange);
    RUN_TEST("PointLightShadow_InvalidFace", Test_PointLightShadow_InvalidFace);

    // ShadowSystem
    RUN_TEST("ShadowSystem_Initialize", Test_ShadowSystem_Initialize);
    RUN_TEST("ShadowSystem_InitializeDefaults", Test_ShadowSystem_InitializeDefaults);
    RUN_TEST("ShadowSystem_CascadeSplits", Test_ShadowSystem_CascadeSplits);
    RUN_TEST("ShadowSystem_SelectCascade", Test_ShadowSystem_SelectCascade);
    RUN_TEST("ShadowSystem_RenderDirectionalShadows", Test_ShadowSystem_RenderDirectionalShadows);
    RUN_TEST("ShadowSystem_SampleDirectionalShadow_NoOccluders", Test_ShadowSystem_SampleDirectionalShadow_NoOccluders);
    RUN_TEST("ShadowSystem_PointLightShadow", Test_ShadowSystem_PointLightShadow);
    RUN_TEST("ShadowSystem_ClearAll", Test_ShadowSystem_ClearAll);
    RUN_TEST("ShadowSystem_MultiplePointLights", Test_ShadowSystem_MultiplePointLights);
    RUN_TEST("ShadowSystem_InvalidPointLightIndex", Test_ShadowSystem_InvalidPointLightIndex);
    RUN_TEST("ShadowSystem_ConfigAccessors", Test_ShadowSystem_ConfigAccessors);
}
