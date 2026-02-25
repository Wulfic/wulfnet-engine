// =============================================================================
// WulfNet Engine - Acoustic System Tests
// =============================================================================
// Tests for AcousticMaterial, AcousticSystem (occlusion, obstruction,
// impulse response, room estimation), RT60 calculations, and distance
// attenuation.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Audio/Acoustics/AcousticSystem.h>
#include <cmath>
#include <vector>

using namespace WulfNet;

static constexpr float kTestEpsilon = 0.001f;

// =============================================================================
// Helper: Simple box room ray caster
// =============================================================================
// Simulates a box room centered at origin with given half-extents.
// All walls have the same material ID. Returns the closest wall hit.
static AcousticRayCastFn MakeBoxRoomRayCast(float halfX, float halfY, float halfZ, int materialId = 0) {
    return [=](float ox, float oy, float oz,
               float dx, float dy, float dz, float maxDist) -> AcousticRayHit {
        AcousticRayHit closest;
        closest.hit = false;
        closest.distance = maxDist + 1.0f;

        // Test all 6 axis-aligned planes
        struct Plane { float normal[3]; float d; };
        Plane planes[6] = {
            {{ 1, 0, 0}, halfX}, {{-1, 0, 0}, halfX},
            {{ 0, 1, 0}, halfY}, {{ 0,-1, 0}, halfY},
            {{ 0, 0, 1}, halfZ}, {{ 0, 0,-1}, halfZ}
        };

        for (auto& p : planes) {
            float denom = dx * p.normal[0] + dy * p.normal[1] + dz * p.normal[2];
            if (std::abs(denom) < 1e-8f) continue;

            float planeD = p.d * p.normal[0] + p.d * p.normal[1] + p.d * p.normal[2];
            // Point on plane: p.normal[0]*x + p.normal[1]*y + p.normal[2]*z = p.d
            // For axis-aligned: simply the coordinate of the plane along the normal
            float originDot = ox * p.normal[0] + oy * p.normal[1] + oz * p.normal[2];
            float t = (p.d - originDot) / denom;

            if (t <= 0.0f || t > maxDist) continue;

            // Check hit point is within the box face
            float hx = ox + dx * t;
            float hy = oy + dy * t;
            float hz = oz + dz * t;

            bool inBounds = (hx >= -halfX - 0.01f && hx <= halfX + 0.01f &&
                             hy >= -halfY - 0.01f && hy <= halfY + 0.01f &&
                             hz >= -halfZ - 0.01f && hz <= halfZ + 0.01f);

            if (inBounds && t < closest.distance) {
                closest.hit = true;
                closest.distance = t;
                closest.normalX = p.normal[0];
                closest.normalY = p.normal[1];
                closest.normalZ = p.normal[2];
                closest.materialId = materialId;
            }
        }

        return closest;
    };
}

// A blocker plane perpendicular to the X axis at a given x position
static AcousticRayCastFn MakeWallBlocker(float wallX, int materialId = 0) {
    return [=](float ox, float oy, float oz,
               float dx, float dy, float dz, float maxDist) -> AcousticRayHit {
        AcousticRayHit hit;
        if (std::abs(dx) < 1e-8f) return hit;

        float t = (wallX - ox) / dx;
        if (t <= 0.0f || t > maxDist) return hit;

        hit.hit = true;
        hit.distance = t;
        hit.normalX = (dx > 0) ? -1.0f : 1.0f;
        hit.normalY = 0.0f;
        hit.normalZ = 0.0f;
        hit.materialId = materialId;
        return hit;
    };
}

// A ray caster that never hits anything (open space)
static AcousticRayCastFn MakeOpenSpaceRayCast() {
    return [](float, float, float, float, float, float, float) -> AcousticRayHit {
        return AcousticRayHit{};
    };
}

// =============================================================================
// AcousticMaterial Tests
// =============================================================================

static void Test_AcousticMaterial_DefaultValues() {
    AcousticMaterial m;
    for (int i = 0; i < AcousticMaterial::kNumBands; ++i) {
        EXPECT_NEAR(m.absorption[i], 0.1f, kTestEpsilon);
    }
    EXPECT_NEAR(m.scattering, 0.5f, kTestEpsilon);
    EXPECT_NEAR(m.transmission, 0.0f, kTestEpsilon);
}

static void Test_AcousticMaterial_ConcretePreset() {
    AcousticMaterial m = AcousticMaterial::Concrete();
    // Concrete: very low absorption across all bands
    for (int i = 0; i < AcousticMaterial::kNumBands; ++i) {
        EXPECT_TRUE(m.absorption[i] < 0.05f);
    }
    EXPECT_TRUE(m.scattering < 0.5f);
    EXPECT_NEAR(m.transmission, 0.0f, kTestEpsilon);
}

static void Test_AcousticMaterial_CarpetPreset() {
    AcousticMaterial m = AcousticMaterial::Carpet();
    // Carpet: high absorption at high frequencies
    EXPECT_TRUE(m.absorption[4] > 0.3f); // 2kHz
    EXPECT_TRUE(m.absorption[5] > 0.5f); // 4kHz
    // Low absorption at low frequencies
    EXPECT_TRUE(m.absorption[0] < 0.1f); // 125Hz
}

static void Test_AcousticMaterial_AllPresets() {
    // All presets should produce valid coefficients in [0,1]
    AcousticMaterial presets[] = {
        AcousticMaterial::Concrete(),
        AcousticMaterial::Wood(),
        AcousticMaterial::Glass(),
        AcousticMaterial::Carpet(),
        AcousticMaterial::Curtain(),
        AcousticMaterial::Metal()
    };

    for (const auto& mat : presets) {
        for (int b = 0; b < AcousticMaterial::kNumBands; ++b) {
            EXPECT_GE(mat.absorption[b], 0.0f);
            EXPECT_LE(mat.absorption[b], 1.0f);
        }
        EXPECT_GE(mat.scattering, 0.0f);
        EXPECT_LE(mat.scattering, 1.0f);
        EXPECT_GE(mat.transmission, 0.0f);
        EXPECT_LE(mat.transmission, 1.0f);
    }
}

// =============================================================================
// AcousticSystem Init Tests
// =============================================================================

static void Test_AcousticSystem_Initialize() {
    AcousticSystem sys;
    EXPECT_TRUE(!sys.IsInitialized());

    bool ok = sys.Initialize();
    EXPECT_TRUE(ok);
    EXPECT_TRUE(sys.IsInitialized());
    EXPECT_EQ(sys.GetConfig().maxBounces, 6);
    EXPECT_EQ(sys.GetConfig().numRays, 128);
}

static void Test_AcousticSystem_InitializeInvalid() {
    AcousticSystem sys;

    // Invalid: 0 rays
    AcousticConfig cfg;
    cfg.numRays = 0;
    EXPECT_TRUE(!sys.Initialize(cfg));
    EXPECT_TRUE(!sys.IsInitialized());

    // Invalid: negative max distance
    AcousticConfig cfg2;
    cfg2.maxDistance = -1.0f;
    EXPECT_TRUE(!sys.Initialize(cfg2));
}

static void Test_AcousticSystem_Shutdown() {
    AcousticSystem sys;
    sys.Initialize();
    sys.AddMaterial(AcousticMaterial::Concrete());
    EXPECT_EQ(sys.GetMaterialCount(), 1);

    sys.Shutdown();
    EXPECT_TRUE(!sys.IsInitialized());
    EXPECT_EQ(sys.GetMaterialCount(), 0);
}

// =============================================================================
// Material Management
// =============================================================================

static void Test_AcousticSystem_AddMaterial() {
    AcousticSystem sys;
    sys.Initialize();

    int id0 = sys.AddMaterial(AcousticMaterial::Concrete());
    int id1 = sys.AddMaterial(AcousticMaterial::Wood());
    int id2 = sys.AddMaterial(AcousticMaterial::Carpet());

    EXPECT_EQ(id0, 0);
    EXPECT_EQ(id1, 1);
    EXPECT_EQ(id2, 2);
    EXPECT_EQ(sys.GetMaterialCount(), 3);
}

static void Test_AcousticSystem_GetMaterial() {
    AcousticSystem sys;
    sys.Initialize();

    int id = sys.AddMaterial(AcousticMaterial::Metal());
    const AcousticMaterial& m = sys.GetMaterial(id);
    EXPECT_NEAR(m.absorption[0], 0.04f, kTestEpsilon);

    // Invalid ID should return default material (Concrete)
    const AcousticMaterial& def = sys.GetMaterial(-1);
    EXPECT_NEAR(def.absorption[0], AcousticMaterial::Concrete().absorption[0], kTestEpsilon);
}

// =============================================================================
// Occlusion Tests
// =============================================================================

static void Test_AcousticSystem_OcclusionClear() {
    AcousticSystem sys;
    sys.Initialize();
    sys.SetRayCastFunction(MakeOpenSpaceRayCast());

    // No geometry → full clear
    float occ = sys.ComputeOcclusion(0, 0, 0, 10, 0, 0);
    EXPECT_NEAR(occ, 1.0f, kTestEpsilon);
}

static void Test_AcousticSystem_OcclusionBlocked() {
    AcousticSystem sys;
    sys.Initialize();

    int matId = sys.AddMaterial(AcousticMaterial::Concrete());
    sys.SetRayCastFunction(MakeWallBlocker(5.0f, matId));

    // Wall at x=5 between source (0,0,0) and listener (10,0,0)
    float occ = sys.ComputeOcclusion(0, 0, 0, 10, 0, 0);
    EXPECT_NEAR(occ, 0.0f, kTestEpsilon); // Concrete has 0 transmission
}

static void Test_AcousticSystem_OcclusionPartial() {
    AcousticSystem sys;
    sys.Initialize();

    // Glass has some transmission
    int matId = sys.AddMaterial(AcousticMaterial::Glass());
    sys.SetRayCastFunction(MakeWallBlocker(5.0f, matId));

    float occ = sys.ComputeOcclusion(0, 0, 0, 10, 0, 0);
    EXPECT_TRUE(occ > 0.0f);   // Some sound gets through glass
    EXPECT_TRUE(occ < 1.0f);   // But not full
    EXPECT_NEAR(occ, 0.1f, 0.05f); // Glass transmission ~= 0.1
}

static void Test_AcousticSystem_OcclusionNoRayCast() {
    AcousticSystem sys;
    sys.Initialize();
    // No ray cast function set → should return 1.0 (clear)
    float occ = sys.ComputeOcclusion(0, 0, 0, 10, 0, 0);
    EXPECT_NEAR(occ, 1.0f, kTestEpsilon);
}

static void Test_AcousticSystem_OcclusionSamePosition() {
    AcousticSystem sys;
    sys.Initialize();
    sys.SetRayCastFunction(MakeOpenSpaceRayCast());

    // Source and listener at same position
    float occ = sys.ComputeOcclusion(5, 5, 5, 5, 5, 5);
    EXPECT_NEAR(occ, 1.0f, kTestEpsilon);
}

// =============================================================================
// Obstruction Tests
// =============================================================================

static void Test_AcousticSystem_ObstructionClear() {
    AcousticSystem sys;
    sys.Initialize();
    sys.SetRayCastFunction(MakeOpenSpaceRayCast());

    float obs = sys.ComputeObstruction(0, 0, 0, 10, 0, 0, 16);
    EXPECT_NEAR(obs, 1.0f, kTestEpsilon); // Fully clear
}

static void Test_AcousticSystem_ObstructionBlocked() {
    AcousticSystem sys;
    sys.Initialize();

    int matId = sys.AddMaterial(AcousticMaterial::Concrete());
    sys.SetRayCastFunction(MakeWallBlocker(5.0f, matId));

    // Large wall blocks most sample rays
    float obs = sys.ComputeObstruction(0, 0, 0, 10, 0, 0, 32);
    EXPECT_TRUE(obs < 0.5f); // Mostly blocked
}

static void Test_AcousticSystem_ObstructionNoRayCast() {
    AcousticSystem sys;
    sys.Initialize();
    float obs = sys.ComputeObstruction(0, 0, 0, 10, 0, 0, 8);
    EXPECT_NEAR(obs, 1.0f, kTestEpsilon); // No geometry = clear
}

// =============================================================================
// Impulse Response Tests
// =============================================================================

static void Test_AcousticSystem_IRDirectPath() {
    AcousticSystem sys;
    AcousticConfig cfg;
    cfg.numRays = 32;
    cfg.maxBounces = 4;
    sys.Initialize(cfg);
    sys.SetRayCastFunction(MakeOpenSpaceRayCast());

    // Open space: only direct path, no reflections
    ImpulseResponse ir = sys.TraceImpulseResponse(0, 0, 0, 10, 0, 0);

    float expectedTime = 10.0f / 343.0f;
    EXPECT_NEAR(ir.directTime, expectedTime, 0.001f);
    EXPECT_TRUE(ir.directEnergy > 0.0f);
    EXPECT_TRUE(!ir.directOccluded);
    EXPECT_EQ(static_cast<int>(ir.taps.size()), 0); // No reflections in open space
}

static void Test_AcousticSystem_IRBoxRoom() {
    AcousticSystem sys;
    AcousticConfig cfg;
    cfg.numRays = 64;
    cfg.maxBounces = 4;
    cfg.maxDistance = 50.0f;
    sys.Initialize(cfg);

    int matId = sys.AddMaterial(AcousticMaterial::Concrete());
    sys.SetRayCastFunction(MakeBoxRoomRayCast(5.0f, 3.0f, 5.0f, matId));

    ImpulseResponse ir = sys.TraceImpulseResponse(-2, 0, 0, 2, 0, 0);

    // Should have some reflections from the box walls
    EXPECT_TRUE(ir.taps.size() > 0);
    EXPECT_TRUE(ir.directEnergy > 0.0f);
    EXPECT_TRUE(!ir.directOccluded);
}

static void Test_AcousticSystem_IROccludedDirect() {
    AcousticSystem sys;
    AcousticConfig cfg;
    cfg.numRays = 16;
    cfg.maxBounces = 2;
    sys.Initialize(cfg);

    int matId = sys.AddMaterial(AcousticMaterial::Concrete());
    sys.SetRayCastFunction(MakeWallBlocker(5.0f, matId));

    // Wall blocks line of sight
    ImpulseResponse ir = sys.TraceImpulseResponse(0, 0, 0, 10, 0, 0);
    EXPECT_TRUE(ir.directOccluded);
    EXPECT_NEAR(ir.directEnergy, 0.0f, 0.01f); // Concrete blocks all
}

static void Test_AcousticSystem_IRNotInitialized() {
    AcousticSystem sys;
    // Not initialized → empty impulse response
    ImpulseResponse ir = sys.TraceImpulseResponse(0, 0, 0, 10, 0, 0);
    EXPECT_NEAR(ir.directEnergy, 0.0f, kTestEpsilon);
    EXPECT_EQ(static_cast<int>(ir.taps.size()), 0);
}

static void Test_AcousticSystem_IRReflectionTiming() {
    AcousticSystem sys;
    AcousticConfig cfg;
    cfg.numRays = 64;
    cfg.maxBounces = 4;
    cfg.maxDistance = 50.0f;
    sys.Initialize(cfg);

    int matId = sys.AddMaterial(AcousticMaterial::Wood());
    sys.SetRayCastFunction(MakeBoxRoomRayCast(5.0f, 3.0f, 5.0f, matId));

    ImpulseResponse ir = sys.TraceImpulseResponse(0, 0, 0, 2, 0, 0);

    // All reflection taps should arrive after the direct path
    for (const auto& tap : ir.taps) {
        EXPECT_TRUE(tap.time >= ir.directTime);
        EXPECT_TRUE(tap.energy >= 0.0f);
        EXPECT_TRUE(tap.bounces > 0);
    }
}

// =============================================================================
// ImpulseResponse ToAudioBuffer
// =============================================================================

static void Test_AcousticSystem_IRToAudioBuffer() {
    ImpulseResponse ir;
    ir.directTime = 0.01f;
    ir.directEnergy = 0.8f;

    ReflectionTap tap1;
    tap1.time = 0.05f;
    tap1.energy = 0.3f;
    tap1.bounces = 1;
    ir.taps.push_back(tap1);

    ReflectionTap tap2;
    tap2.time = 0.1f;
    tap2.energy = 0.15f;
    tap2.bounces = 2;
    ir.taps.push_back(tap2);

    AudioBuffer buf = ir.ToAudioBuffer(44100, 0.5f);
    EXPECT_TRUE(buf.GetFrameCount() > 0);
    EXPECT_EQ(buf.GetFormat().channels, 1);

    // Verify the buffer contains the direct energy somewhere near the expected sample
    const float* data = buf.GetData();
    int numFrames = buf.GetFrameCount();

    // Find the max value in the buffer (should be the direct energy)
    float maxVal = 0.0f;
    for (int i = 0; i < numFrames; ++i) {
        if (data[i] > maxVal) maxVal = data[i];
    }
    EXPECT_NEAR(maxVal, 0.8f, 0.01f);

    // Verify the tap energy is also present
    float secondMax = 0.0f;
    for (int i = 0; i < numFrames; ++i) {
        if (data[i] > secondMax && data[i] < maxVal - 0.01f) {
            secondMax = data[i];
        }
    }
    EXPECT_NEAR(secondMax, 0.3f, 0.01f);
}

static void Test_AcousticSystem_IRToAudioBufferEmpty() {
    ImpulseResponse ir;
    AudioBuffer buf = ir.ToAudioBuffer(44100, 0.0f);
    EXPECT_EQ(buf.GetFrameCount(), 0);
}

// =============================================================================
// Room Estimation Tests
// =============================================================================

static void Test_AcousticSystem_EstimateRoom() {
    AcousticSystem sys;
    AcousticConfig cfg;
    cfg.roomProbeRays = 64;
    sys.Initialize(cfg);

    int matId = sys.AddMaterial(AcousticMaterial::Concrete());
    sys.SetRayCastFunction(MakeBoxRoomRayCast(5.0f, 3.0f, 5.0f, matId));

    RoomEstimate room = sys.EstimateRoom(0, 0, 0);

    // Mean free path should be reasonable for a 10×6×10 room
    EXPECT_TRUE(room.meanFreePath > 0.0f);
    EXPECT_TRUE(room.volume > 0.0f);
    EXPECT_TRUE(room.surfaceArea > 0.0f);
    EXPECT_TRUE(room.rt60 > 0.0f);
    EXPECT_TRUE(room.avgAbsorption > 0.0f);
    EXPECT_TRUE(room.avgAbsorption < 1.0f);
}

static void Test_AcousticSystem_EstimateRoomNoGeometry() {
    AcousticSystem sys;
    sys.Initialize();
    sys.SetRayCastFunction(MakeOpenSpaceRayCast());

    RoomEstimate room = sys.EstimateRoom(0, 0, 0);
    // No hits → no room estimate
    EXPECT_NEAR(room.volume, 0.0f, kTestEpsilon);
}

static void Test_AcousticSystem_EstimateRoomNoRayCast() {
    AcousticSystem sys;
    sys.Initialize();

    RoomEstimate room = sys.EstimateRoom(0, 0, 0);
    EXPECT_NEAR(room.volume, 0.0f, kTestEpsilon);
}

// =============================================================================
// RT60 Calculation Tests
// =============================================================================

static void Test_AcousticSystem_RT60Sabine() {
    // Sabine: RT60 = 0.161 * V / (S * α)
    float rt60 = AcousticSystem::ComputeRT60_Sabine(100.0f, 120.0f, 0.2f);
    float expected = 0.161f * 100.0f / (120.0f * 0.2f);
    EXPECT_NEAR(rt60, expected, 0.001f);
}

static void Test_AcousticSystem_RT60SabineEdgeCases() {
    EXPECT_NEAR(AcousticSystem::ComputeRT60_Sabine(100.0f, 0.0f, 0.2f), 0.0f, kTestEpsilon);
    EXPECT_NEAR(AcousticSystem::ComputeRT60_Sabine(100.0f, 120.0f, 0.0f), 0.0f, kTestEpsilon);
}

static void Test_AcousticSystem_RT60Eyring() {
    // Eyring: RT60 = 0.161 * V / (-S * ln(1-α))
    float rt60 = AcousticSystem::ComputeRT60_Eyring(100.0f, 120.0f, 0.2f);
    float expected = 0.161f * 100.0f / (-120.0f * std::log(1.0f - 0.2f));
    EXPECT_NEAR(rt60, expected, 0.001f);
}

static void Test_AcousticSystem_RT60EyringEdgeCases() {
    EXPECT_NEAR(AcousticSystem::ComputeRT60_Eyring(100.0f, 0.0f, 0.2f), 0.0f, kTestEpsilon);
    EXPECT_NEAR(AcousticSystem::ComputeRT60_Eyring(100.0f, 120.0f, 0.0f), 0.0f, kTestEpsilon);
    EXPECT_NEAR(AcousticSystem::ComputeRT60_Eyring(100.0f, 120.0f, 1.0f), 0.0f, kTestEpsilon); // α=1 invalid for Eyring
}

static void Test_AcousticSystem_RT60Comparison() {
    // For the same room, Eyring should give a shorter (or equal) RT60 than Sabine
    float sabine = AcousticSystem::ComputeRT60_Sabine(200.0f, 240.0f, 0.15f);
    float eyring = AcousticSystem::ComputeRT60_Eyring(200.0f, 240.0f, 0.15f);
    EXPECT_TRUE(eyring <= sabine);
    EXPECT_TRUE(eyring > 0.0f);
}

// =============================================================================
// Distance Attenuation Tests
// =============================================================================

static void Test_AcousticSystem_DistanceAttenuation() {
    // At reference distance → 1.0
    float a = AcousticSystem::DistanceAttenuation(1.0f);
    EXPECT_NEAR(a, 1.0f, kTestEpsilon);

    // At twice reference distance → 0.5
    float b = AcousticSystem::DistanceAttenuation(2.0f);
    EXPECT_NEAR(b, 0.5f, kTestEpsilon);

    // At 10x reference distance → 0.1
    float c = AcousticSystem::DistanceAttenuation(10.0f);
    EXPECT_NEAR(c, 0.1f, kTestEpsilon);
}

static void Test_AcousticSystem_DistanceAttenuationClamping() {
    // Below reference distance → 1.0
    float a = AcousticSystem::DistanceAttenuation(0.5f);
    EXPECT_NEAR(a, 1.0f, kTestEpsilon);

    // Beyond max distance → 0.0
    float b = AcousticSystem::DistanceAttenuation(200.0f, 1.0f, 100.0f);
    EXPECT_NEAR(b, 0.0f, kTestEpsilon);
}

static void Test_AcousticSystem_DistanceAttenuationRolloff() {
    // Higher rolloff = faster falloff
    float normal = AcousticSystem::DistanceAttenuation(4.0f, 1.0f, 100.0f, 1.0f);
    float steep  = AcousticSystem::DistanceAttenuation(4.0f, 1.0f, 100.0f, 2.0f);
    EXPECT_TRUE(steep < normal);
}

// =============================================================================
// Air Absorption Tests
// =============================================================================

static void Test_AcousticSystem_AirAbsorption() {
    AcousticSystem sys;
    sys.Initialize();

    // At distance 0, absorption is 1.0 (no loss)
    float a = sys.AirAbsorption(0.0f);
    EXPECT_NEAR(a, 1.0f, kTestEpsilon);

    // Energy decreases with distance
    float b = sys.AirAbsorption(10.0f);
    float c = sys.AirAbsorption(50.0f);
    EXPECT_TRUE(b < 1.0f);
    EXPECT_TRUE(c < b);
    EXPECT_TRUE(c > 0.0f);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterAcousticSystemTests() {
    // Material tests (4)
    RUN_TEST("AC_Material_DefaultValues", Test_AcousticMaterial_DefaultValues);
    RUN_TEST("AC_Material_ConcretePreset", Test_AcousticMaterial_ConcretePreset);
    RUN_TEST("AC_Material_CarpetPreset", Test_AcousticMaterial_CarpetPreset);
    RUN_TEST("AC_Material_AllPresets", Test_AcousticMaterial_AllPresets);

    // Init tests (3)
    RUN_TEST("AC_Initialize", Test_AcousticSystem_Initialize);
    RUN_TEST("AC_InitializeInvalid", Test_AcousticSystem_InitializeInvalid);
    RUN_TEST("AC_Shutdown", Test_AcousticSystem_Shutdown);

    // Material management (2)
    RUN_TEST("AC_AddMaterial", Test_AcousticSystem_AddMaterial);
    RUN_TEST("AC_GetMaterial", Test_AcousticSystem_GetMaterial);

    // Occlusion tests (5)
    RUN_TEST("AC_OcclusionClear", Test_AcousticSystem_OcclusionClear);
    RUN_TEST("AC_OcclusionBlocked", Test_AcousticSystem_OcclusionBlocked);
    RUN_TEST("AC_OcclusionPartial", Test_AcousticSystem_OcclusionPartial);
    RUN_TEST("AC_OcclusionNoRayCast", Test_AcousticSystem_OcclusionNoRayCast);
    RUN_TEST("AC_OcclusionSamePosition", Test_AcousticSystem_OcclusionSamePosition);

    // Obstruction tests (3)
    RUN_TEST("AC_ObstructionClear", Test_AcousticSystem_ObstructionClear);
    RUN_TEST("AC_ObstructionBlocked", Test_AcousticSystem_ObstructionBlocked);
    RUN_TEST("AC_ObstructionNoRayCast", Test_AcousticSystem_ObstructionNoRayCast);

    // Impulse response tests (5)
    RUN_TEST("AC_IRDirectPath", Test_AcousticSystem_IRDirectPath);
    RUN_TEST("AC_IRBoxRoom", Test_AcousticSystem_IRBoxRoom);
    RUN_TEST("AC_IROccludedDirect", Test_AcousticSystem_IROccludedDirect);
    RUN_TEST("AC_IRNotInitialized", Test_AcousticSystem_IRNotInitialized);
    RUN_TEST("AC_IRReflectionTiming", Test_AcousticSystem_IRReflectionTiming);

    // IR to AudioBuffer (2)
    RUN_TEST("AC_IRToAudioBuffer", Test_AcousticSystem_IRToAudioBuffer);
    RUN_TEST("AC_IRToAudioBufferEmpty", Test_AcousticSystem_IRToAudioBufferEmpty);

    // Room estimation (3)
    RUN_TEST("AC_EstimateRoom", Test_AcousticSystem_EstimateRoom);
    RUN_TEST("AC_EstimateRoomNoGeometry", Test_AcousticSystem_EstimateRoomNoGeometry);
    RUN_TEST("AC_EstimateRoomNoRayCast", Test_AcousticSystem_EstimateRoomNoRayCast);

    // RT60 tests (5)
    RUN_TEST("AC_RT60Sabine", Test_AcousticSystem_RT60Sabine);
    RUN_TEST("AC_RT60SabineEdgeCases", Test_AcousticSystem_RT60SabineEdgeCases);
    RUN_TEST("AC_RT60Eyring", Test_AcousticSystem_RT60Eyring);
    RUN_TEST("AC_RT60EyringEdgeCases", Test_AcousticSystem_RT60EyringEdgeCases);
    RUN_TEST("AC_RT60Comparison", Test_AcousticSystem_RT60Comparison);

    // Distance attenuation (3)
    RUN_TEST("AC_DistanceAttenuation", Test_AcousticSystem_DistanceAttenuation);
    RUN_TEST("AC_DistanceAttenuationClamping", Test_AcousticSystem_DistanceAttenuationClamping);
    RUN_TEST("AC_DistanceAttenuationRolloff", Test_AcousticSystem_DistanceAttenuationRolloff);

    // Air absorption (1)
    RUN_TEST("AC_AirAbsorption", Test_AcousticSystem_AirAbsorption);
}
