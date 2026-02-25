// =============================================================================
// WulfNet Engine - Spatial Audio Tests
// =============================================================================
// Tests for HRTF, Ambisonics, Doppler effect, distance attenuation,
// and coordinate transform utilities.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Audio/Spatial/SpatialAudio.h>
#include <cmath>
#include <vector>

using namespace WulfNet;

static constexpr float kSpatialEps = 0.01f;
static constexpr float kSpPi = 3.14159265358979323846f;

// =============================================================================
// AttenuationCurve Tests
// =============================================================================

static void Test_AttenuationCurve_Inverse() {
    AttenuationCurve curve;
    curve.model = AttenuationModel::Inverse;
    curve.refDistance = 1.0f;
    curve.maxDistance = 100.0f;
    curve.rolloff = 1.0f;

    EXPECT_NEAR(curve.Evaluate(1.0f), 1.0f, kSpatialEps);
    EXPECT_NEAR(curve.Evaluate(2.0f), 0.5f, kSpatialEps);
    EXPECT_NEAR(curve.Evaluate(10.0f), 0.1f, kSpatialEps);
}

static void Test_AttenuationCurve_Linear() {
    AttenuationCurve curve;
    curve.model = AttenuationModel::Linear;
    curve.refDistance = 1.0f;
    curve.maxDistance = 11.0f;

    EXPECT_NEAR(curve.Evaluate(1.0f), 1.0f, kSpatialEps);
    EXPECT_NEAR(curve.Evaluate(6.0f), 0.5f, kSpatialEps);
    EXPECT_NEAR(curve.Evaluate(11.0f), 0.0f, kSpatialEps);
}

static void Test_AttenuationCurve_Exponential() {
    AttenuationCurve curve;
    curve.model = AttenuationModel::Exponential;
    curve.refDistance = 1.0f;
    curve.maxDistance = 100.0f;
    curve.rolloff = 2.0f;

    EXPECT_NEAR(curve.Evaluate(1.0f), 1.0f, kSpatialEps);
    float mid = curve.Evaluate(50.5f);
    EXPECT_TRUE(mid > 0.0f);
    EXPECT_TRUE(mid < 1.0f);
    EXPECT_NEAR(curve.Evaluate(100.0f), 0.0f, kSpatialEps);
}

static void Test_AttenuationCurve_None() {
    AttenuationCurve curve;
    curve.model = AttenuationModel::None;

    EXPECT_NEAR(curve.Evaluate(50.0f), 1.0f, kSpatialEps);
    EXPECT_NEAR(curve.Evaluate(0.5f), 1.0f, kSpatialEps);
}

static void Test_AttenuationCurve_Clamping() {
    AttenuationCurve curve;
    curve.model = AttenuationModel::Inverse;
    curve.refDistance = 1.0f;
    curve.maxDistance = 100.0f;

    // Below ref → 1.0
    EXPECT_NEAR(curve.Evaluate(0.5f), 1.0f, kSpatialEps);
    // Beyond max → 0.0
    EXPECT_NEAR(curve.Evaluate(200.0f), 0.0f, kSpatialEps);
}

// =============================================================================
// SpatialAudio Init Tests
// =============================================================================

static void Test_SpatialAudio_Initialize() {
    SpatialAudio sa;
    EXPECT_TRUE(!sa.IsInitialized());

    bool ok = sa.Initialize(44100);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(sa.IsInitialized());
    EXPECT_EQ(sa.GetSampleRate(), 44100);
}

static void Test_SpatialAudio_InitializeInvalid() {
    SpatialAudio sa;
    EXPECT_TRUE(!sa.Initialize(0));
    EXPECT_TRUE(!sa.Initialize(-1));
    EXPECT_TRUE(!sa.IsInitialized());
}

static void Test_SpatialAudio_Shutdown() {
    SpatialAudio sa;
    sa.Initialize();
    EXPECT_TRUE(sa.IsInitialized());
    sa.Shutdown();
    EXPECT_TRUE(!sa.IsInitialized());
}

// =============================================================================
// HRTF Tests
// =============================================================================

static void Test_HRTF_Centered() {
    SpatialAudio sa;
    sa.Initialize(44100);

    // Source directly in front (along -Z in local space)
    HRTFResult r = sa.ComputeHRTF(0.0f, 0.0f, -10.0f);

    // Centered: equal gains, zero delays
    EXPECT_NEAR(r.leftGain, 1.0f, kSpatialEps);
    EXPECT_NEAR(r.rightGain, 1.0f, kSpatialEps);
    EXPECT_EQ(r.leftDelay, 0);
    EXPECT_EQ(r.rightDelay, 0);
}

static void Test_HRTF_Right() {
    SpatialAudio sa;
    sa.Initialize(44100);

    // Source to the right (+X)
    HRTFResult r = sa.ComputeHRTF(10.0f, 0.0f, 0.0f);

    // Right ear should be louder, left ear should have delay
    EXPECT_TRUE(r.rightGain > r.leftGain);
    EXPECT_TRUE(r.leftDelay > 0);
    EXPECT_EQ(r.rightDelay, 0);
}

static void Test_HRTF_Left() {
    SpatialAudio sa;
    sa.Initialize(44100);

    // Source to the left (-X)
    HRTFResult r = sa.ComputeHRTF(-10.0f, 0.0f, 0.0f);

    // Left ear should be louder, right ear should have delay
    EXPECT_TRUE(r.leftGain > r.rightGain);
    EXPECT_EQ(r.leftDelay, 0);
    EXPECT_TRUE(r.rightDelay > 0);
}

static void Test_HRTF_Symmetry() {
    SpatialAudio sa;
    sa.Initialize(44100);

    HRTFResult right = sa.ComputeHRTF(5.0f, 0.0f, -5.0f);
    HRTFResult left  = sa.ComputeHRTF(-5.0f, 0.0f, -5.0f);

    // Gains should be mirrored
    EXPECT_NEAR(right.rightGain, left.leftGain, kSpatialEps);
    EXPECT_NEAR(right.leftGain, left.rightGain, kSpatialEps);
    EXPECT_EQ(right.rightDelay, left.leftDelay);
    EXPECT_EQ(right.leftDelay, left.rightDelay);
}

static void Test_HRTF_AtOrigin() {
    SpatialAudio sa;
    sa.Initialize(44100);

    // Source at listener position
    HRTFResult r = sa.ComputeHRTF(0.0f, 0.0f, 0.0f);
    EXPECT_NEAR(r.leftGain, 1.0f, kSpatialEps);
    EXPECT_NEAR(r.rightGain, 1.0f, kSpatialEps);
    EXPECT_EQ(r.leftDelay, 0);
    EXPECT_EQ(r.rightDelay, 0);
}

// =============================================================================
// HRTF Apply Tests
// =============================================================================

static void Test_HRTF_ApplyBinaural() {
    SpatialAudio sa;
    sa.Initialize(44100);

    // Create a short mono buffer
    AudioBuffer mono = AudioBuffer::GenerateSine(440.0f, 0.01f, 44100);

    // Source to the right
    HRTFResult hrtf = sa.ComputeHRTF(10.0f, 0.0f, 0.0f);
    AudioBuffer binaural = sa.ApplyHRTF(mono, hrtf);

    // Output should be stereo
    EXPECT_EQ(binaural.GetFormat().channels, 2);
    EXPECT_EQ(binaural.GetFrameCount(), mono.GetFrameCount());

    // Right channel should be louder than left (source is to the right)
    float leftRms = 0.0f, rightRms = 0.0f;
    const float* data = binaural.GetData();
    int frames = binaural.GetFrameCount();
    for (int i = 0; i < frames; ++i) {
        leftRms += data[i * 2] * data[i * 2];
        rightRms += data[i * 2 + 1] * data[i * 2 + 1];
    }
    leftRms = std::sqrt(leftRms / frames);
    rightRms = std::sqrt(rightRms / frames);

    EXPECT_TRUE(rightRms > leftRms);
}

static void Test_HRTF_ApplyInvalidInput() {
    SpatialAudio sa;
    sa.Initialize(44100);

    // Stereo input should be rejected (HRTF expects mono)
    AudioFormat stereoFmt;
    stereoFmt.channels = 2;
    AudioBuffer stereo;
    stereo.Initialize(stereoFmt, 100);

    HRTFResult hrtf;
    AudioBuffer result = sa.ApplyHRTF(stereo, hrtf);
    EXPECT_EQ(result.GetFrameCount(), 0);
}

// =============================================================================
// Ambisonics Tests
// =============================================================================

static void Test_Ambisonics_EncodeFront() {
    // Source directly in front (azimuth=0, elevation=0)
    AmbisonicsBFormat bf = SpatialAudio::EncodeAmbisonics(0.0f, 0.0f, 1.0f);

    EXPECT_NEAR(bf.W, 0.707107f, 0.001f);  // 1/sqrt(2)
    EXPECT_NEAR(bf.X, 1.0f, kSpatialEps);  // cos(0)*cos(0) = 1
    EXPECT_NEAR(bf.Y, 0.0f, kSpatialEps);  // sin(0)*cos(0) = 0
    EXPECT_NEAR(bf.Z, 0.0f, kSpatialEps);  // sin(0) = 0
}

static void Test_Ambisonics_EncodeRight() {
    // Source to the right (azimuth = -π/2)
    AmbisonicsBFormat bf = SpatialAudio::EncodeAmbisonics(-kSpPi / 2.0f, 0.0f, 1.0f);

    EXPECT_NEAR(bf.W, 0.707107f, 0.001f);
    EXPECT_NEAR(bf.X, 0.0f, kSpatialEps);
    EXPECT_NEAR(bf.Y, -1.0f, kSpatialEps); // sin(-π/2) = -1
    EXPECT_NEAR(bf.Z, 0.0f, kSpatialEps);
}

static void Test_Ambisonics_EncodeAbove() {
    // Source directly above (elevation = π/2)
    AmbisonicsBFormat bf = SpatialAudio::EncodeAmbisonics(0.0f, kSpPi / 2.0f, 1.0f);

    EXPECT_NEAR(bf.W, 0.707107f, 0.001f);
    EXPECT_NEAR(bf.Z, 1.0f, kSpatialEps);  // sin(π/2) = 1
    // X and Y should be near zero (cos(π/2) ≈ 0)
    EXPECT_TRUE(std::abs(bf.X) < 0.01f);
    EXPECT_TRUE(std::abs(bf.Y) < 0.01f);
}

static void Test_Ambisonics_DecodeToStereo() {
    // Source in front
    AmbisonicsBFormat bf = SpatialAudio::EncodeAmbisonics(0.0f, 0.0f, 1.0f);
    auto speakers = SpatialAudio::CreateStereoLayout();
    auto feeds = SpatialAudio::DecodeAmbisonics(bf, speakers);

    EXPECT_EQ(static_cast<int>(feeds.size()), 2);
    // Front source → both speakers should have roughly equal signal
    EXPECT_TRUE(std::abs(feeds[0] - feeds[1]) < 0.3f);
    EXPECT_TRUE(feeds[0] > 0.0f);
    EXPECT_TRUE(feeds[1] > 0.0f);
}

static void Test_Ambisonics_DecodeToQuad() {
    // Source in front
    AmbisonicsBFormat bf = SpatialAudio::EncodeAmbisonics(0.0f, 0.0f, 1.0f);
    auto speakers = SpatialAudio::CreateQuadLayout();
    auto feeds = SpatialAudio::DecodeAmbisonics(bf, speakers);

    EXPECT_EQ(static_cast<int>(feeds.size()), 4);
    // Front speakers should be louder than rear speakers for front source
    float frontSum = feeds[0] + feeds[1];
    float rearSum  = feeds[2] + feeds[3];
    EXPECT_TRUE(frontSum > rearSum);
}

static void Test_Ambisonics_StereoLayout() {
    auto layout = SpatialAudio::CreateStereoLayout();
    EXPECT_EQ(static_cast<int>(layout.size()), 2);
    EXPECT_TRUE(layout[0].azimuth > 0.0f);  // Left at positive azimuth
    EXPECT_TRUE(layout[1].azimuth < 0.0f);  // Right at negative azimuth
}

static void Test_Ambisonics_QuadLayout() {
    auto layout = SpatialAudio::CreateQuadLayout();
    EXPECT_EQ(static_cast<int>(layout.size()), 4);
}

static void Test_Ambisonics_DecodeEmpty() {
    AmbisonicsBFormat bf;
    std::vector<AmbisonicsSpeaker> empty;
    auto feeds = SpatialAudio::DecodeAmbisonics(bf, empty);
    EXPECT_EQ(static_cast<int>(feeds.size()), 0);
}

// =============================================================================
// Doppler Effect Tests
// =============================================================================

static void Test_Doppler_Stationary() {
    SpatialAudio sa;
    sa.Initialize();

    // Both source and listener stationary
    float shift = sa.ComputeDopplerShift(
        0, 0, 0,    // source pos
        0, 0, 0,    // source vel
        10, 0, 0,   // listener pos
        0, 0, 0     // listener vel
    );

    EXPECT_NEAR(shift, 1.0f, kSpatialEps);
}

static void Test_Doppler_ApproachingSource() {
    SpatialAudio sa;
    sa.Initialize();

    // Source moving toward listener
    float shift = sa.ComputeDopplerShift(
        0, 0, 0,     // source pos
        50, 0, 0,    // source vel (toward listener)
        100, 0, 0,   // listener pos
        0, 0, 0      // listener vel
    );

    EXPECT_TRUE(shift > 1.0f); // Higher pitch
}

static void Test_Doppler_RecedingSource() {
    SpatialAudio sa;
    sa.Initialize();

    // Source moving away from listener
    float shift = sa.ComputeDopplerShift(
        0, 0, 0,      // source pos
        -50, 0, 0,    // source vel (away from listener)
        100, 0, 0,    // listener pos
        0, 0, 0       // listener vel
    );

    EXPECT_TRUE(shift < 1.0f); // Lower pitch
}

static void Test_Doppler_ApproachingListener() {
    SpatialAudio sa;
    sa.Initialize();

    // Listener moving toward source
    float shift = sa.ComputeDopplerShift(
        0, 0, 0,      // source pos
        0, 0, 0,       // source vel
        100, 0, 0,     // listener pos
        -50, 0, 0      // listener vel (toward source)
    );

    EXPECT_TRUE(shift > 1.0f); // Higher pitch
}

static void Test_Doppler_SamePosition() {
    SpatialAudio sa;
    sa.Initialize();

    float shift = sa.ComputeDopplerShift(
        5, 5, 5,   0, 0, 0,
        5, 5, 5,   0, 0, 0
    );
    EXPECT_NEAR(shift, 1.0f, kSpatialEps);
}

static void Test_Doppler_MaxShift() {
    SpatialAudio sa;
    sa.Initialize();

    DopplerConfig cfg;
    cfg.maxShift = 2.0f;
    sa.SetDopplerConfig(cfg);

    // Source moving very fast toward listener
    float shift = sa.ComputeDopplerShift(
        0, 0, 0,      // source pos
        300, 0, 0,    // source vel (near speed of sound)
        100, 0, 0,    // listener pos
        0, 0, 0       // listener vel
    );

    // Should be clamped to maxShift
    EXPECT_TRUE(shift <= 2.0f);
}

// =============================================================================
// Distance Gain Tests
// =============================================================================

static void Test_SpatialAudio_DistanceGain() {
    SpatialAudio sa;
    sa.Initialize();

    float gain1 = sa.ComputeDistanceGain(1.0f);
    float gain10 = sa.ComputeDistanceGain(10.0f);
    float gain50 = sa.ComputeDistanceGain(50.0f);

    EXPECT_NEAR(gain1, 1.0f, kSpatialEps);
    EXPECT_TRUE(gain10 < gain1);
    EXPECT_TRUE(gain50 < gain10);
}

// =============================================================================
// Coordinate Transform Tests
// =============================================================================

static void Test_WorldToListenerLocal_Front() {
    float lx, ly, lz;
    // Listener at origin, facing -Z, up +Y
    SpatialAudio::WorldToListenerLocal(
        0, 0, -10,     // source position
        0, 0, 0,       // listener position
        0, 0, -1,      // listener forward
        0, 1, 0,       // listener up
        lx, ly, lz
    );

    // Source is directly in front → localZ should be negative (forward)
    EXPECT_NEAR(lx, 0.0f, kSpatialEps);
    EXPECT_NEAR(ly, 0.0f, kSpatialEps);
    EXPECT_TRUE(lz < 0.0f);
}

static void Test_WorldToListenerLocal_Right() {
    float lx, ly, lz;
    // Source to the right of listener
    SpatialAudio::WorldToListenerLocal(
        10, 0, 0,      // source position
        0, 0, 0,       // listener position
        0, 0, -1,      // listener forward
        0, 1, 0,       // listener up
        lx, ly, lz
    );

    // Source should be to the right (+X in local space)
    EXPECT_TRUE(lx > 0.0f);
    EXPECT_NEAR(ly, 0.0f, kSpatialEps);
}

static void Test_CartesianToSpherical() {
    float az, el, dist;

    // Front (-Z axis)
    SpatialAudio::CartesianToSpherical(0, 0, -10, az, el, dist);
    EXPECT_NEAR(az, 0.0f, kSpatialEps);
    EXPECT_NEAR(el, 0.0f, kSpatialEps);
    EXPECT_NEAR(dist, 10.0f, kSpatialEps);
}

static void Test_CartesianToSpherical_Right() {
    float az, el, dist;

    // Right (+X axis)
    SpatialAudio::CartesianToSpherical(10, 0, 0, az, el, dist);
    EXPECT_NEAR(az, kSpPi / 2.0f, 0.05f); // ~90 degrees
    EXPECT_NEAR(el, 0.0f, kSpatialEps);
    EXPECT_NEAR(dist, 10.0f, kSpatialEps);
}

static void Test_CartesianToSpherical_Above() {
    float az, el, dist;

    // Above (+Y axis)
    SpatialAudio::CartesianToSpherical(0, 10, 0, az, el, dist);
    EXPECT_NEAR(el, kSpPi / 2.0f, 0.05f); // ~90 degrees up
    EXPECT_NEAR(dist, 10.0f, kSpatialEps);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterSpatialAudioTests() {
    // Attenuation curve (5)
    RUN_TEST("SA_AttenuationCurve_Inverse", Test_AttenuationCurve_Inverse);
    RUN_TEST("SA_AttenuationCurve_Linear", Test_AttenuationCurve_Linear);
    RUN_TEST("SA_AttenuationCurve_Exponential", Test_AttenuationCurve_Exponential);
    RUN_TEST("SA_AttenuationCurve_None", Test_AttenuationCurve_None);
    RUN_TEST("SA_AttenuationCurve_Clamping", Test_AttenuationCurve_Clamping);

    // Init (3)
    RUN_TEST("SA_Initialize", Test_SpatialAudio_Initialize);
    RUN_TEST("SA_InitializeInvalid", Test_SpatialAudio_InitializeInvalid);
    RUN_TEST("SA_Shutdown", Test_SpatialAudio_Shutdown);

    // HRTF (7)
    RUN_TEST("SA_HRTF_Centered", Test_HRTF_Centered);
    RUN_TEST("SA_HRTF_Right", Test_HRTF_Right);
    RUN_TEST("SA_HRTF_Left", Test_HRTF_Left);
    RUN_TEST("SA_HRTF_Symmetry", Test_HRTF_Symmetry);
    RUN_TEST("SA_HRTF_AtOrigin", Test_HRTF_AtOrigin);
    RUN_TEST("SA_HRTF_ApplyBinaural", Test_HRTF_ApplyBinaural);
    RUN_TEST("SA_HRTF_ApplyInvalidInput", Test_HRTF_ApplyInvalidInput);

    // Ambisonics (8)
    RUN_TEST("SA_Ambisonics_EncodeFront", Test_Ambisonics_EncodeFront);
    RUN_TEST("SA_Ambisonics_EncodeRight", Test_Ambisonics_EncodeRight);
    RUN_TEST("SA_Ambisonics_EncodeAbove", Test_Ambisonics_EncodeAbove);
    RUN_TEST("SA_Ambisonics_DecodeToStereo", Test_Ambisonics_DecodeToStereo);
    RUN_TEST("SA_Ambisonics_DecodeToQuad", Test_Ambisonics_DecodeToQuad);
    RUN_TEST("SA_Ambisonics_StereoLayout", Test_Ambisonics_StereoLayout);
    RUN_TEST("SA_Ambisonics_QuadLayout", Test_Ambisonics_QuadLayout);
    RUN_TEST("SA_Ambisonics_DecodeEmpty", Test_Ambisonics_DecodeEmpty);

    // Doppler (6)
    RUN_TEST("SA_Doppler_Stationary", Test_Doppler_Stationary);
    RUN_TEST("SA_Doppler_ApproachingSource", Test_Doppler_ApproachingSource);
    RUN_TEST("SA_Doppler_RecedingSource", Test_Doppler_RecedingSource);
    RUN_TEST("SA_Doppler_ApproachingListener", Test_Doppler_ApproachingListener);
    RUN_TEST("SA_Doppler_SamePosition", Test_Doppler_SamePosition);
    RUN_TEST("SA_Doppler_MaxShift", Test_Doppler_MaxShift);

    // Distance gain (1)
    RUN_TEST("SA_DistanceGain", Test_SpatialAudio_DistanceGain);

    // Coordinate transforms (5)
    RUN_TEST("SA_WorldToLocal_Front", Test_WorldToListenerLocal_Front);
    RUN_TEST("SA_WorldToLocal_Right", Test_WorldToListenerLocal_Right);
    RUN_TEST("SA_CartesianToSpherical", Test_CartesianToSpherical);
    RUN_TEST("SA_CartesianToSpherical_Right", Test_CartesianToSpherical_Right);
    RUN_TEST("SA_CartesianToSpherical_Above", Test_CartesianToSpherical_Above);
}
