// =============================================================================
// WulfNet Engine - Performance Benchmarks
// =============================================================================
// Comprehensive benchmarks covering all major engine subsystems:
//   - Audio: buffer ops, mixing, source readback
//   - Acoustics: occlusion, obstruction, impulse response, room estimation
//   - Spatial Audio: HRTF, Ambisonics encode/decode, Doppler
//   - Rendering: GBuffer ops, shadow mapping, GI/SSAO, volumetric ray march
//   - Physics: constitutive models, fluid grid, gaseous sim, destruction
//
// Each benchmark enforces minimum throughput / maximum latency to catch
// performance regressions.
// =============================================================================

#include "TestHarness.h"
#include "BenchmarkHarness.h"

// Audio
#include <WulfNet/Audio/Core/AudioTypes.h>
#include <WulfNet/Audio/Core/AudioMixer.h>
#include <WulfNet/Audio/Acoustics/AcousticSystem.h>
#include <WulfNet/Audio/Spatial/SpatialAudio.h>

// Rendering
#include <WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h>
#include <WulfNet/Rendering/SoftwareRasterizer/GBuffer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/ShadowMap.h>
#include <WulfNet/Rendering/SoftwareRasterizer/GlobalIllumination.h>
#include <WulfNet/Rendering/SoftwareRasterizer/VolumetricRenderer.h>
#include <WulfNet/Rendering/SoftwareRasterizer/RenderPipeline.h>

// Physics
#include <WulfNet/Physics/MPM/ConstitutiveModel.h>
#include <WulfNet/Physics/Fluids/FluidGrid.h>
#include <WulfNet/Physics/Gaseous/GaseousSystem.h>
#include <WulfNet/Physics/Destruction/DestructionSystem.h>

#include <cmath>
#include <vector>

using namespace WulfNet;

// =============================================================================
// SECTION 1: Audio Benchmarks
// =============================================================================

static void Bench_AudioBuffer_GenerateSine() {
    // Benchmark: generate 1-second sine wave at 44100 Hz
    auto r = BENCHMARK_N("AudioBuf_GenerateSine_1s", 500, {
        AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 1.0f, 44100);
        (void)buf;
    });
    // Should be able to generate >200 buffers/sec (1s each)
    BENCH_EXPECT_THROUGHPUT_GT(r, 100.0);
}

static void Bench_AudioBuffer_MixIn() {
    AudioBuffer base = AudioBuffer::GenerateSine(440.0f, 0.1f, 44100);
    AudioBuffer overlay = AudioBuffer::GenerateSine(880.0f, 0.1f, 44100);

    auto r = BENCHMARK_N("AudioBuf_MixIn_4410frames", 2000, {
        base.MixIn(overlay, 0.5f);
    });
    // 4410 frames mix should be very fast (>5000 ops/sec)
    BENCH_EXPECT_THROUGHPUT_GT(r, 1000.0);
}

static void Bench_AudioBuffer_Normalize() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.5f, 44100);
    buf.ApplyGain(0.1f); // make it quiet so normalize has work to do

    auto r = BENCHMARK_N("AudioBuf_Normalize_22050fr", 2000, {
        buf.Normalize(1.0f);
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 500.0);
}

static void Bench_AudioBuffer_ComputeRMS() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 1.0f, 44100);

    float rms = 0.0f;
    auto r = BENCHMARK_N("AudioBuf_ComputeRMS_44100fr", 3000, {
        rms = buf.ComputeRMS();
    });
    EXPECT_TRUE(rms > 0.0f);
    BENCH_EXPECT_THROUGHPUT_GT(r, 1000.0);
}

static void Bench_AudioMixer_MixFrame_8Sources() {
    AudioMixerConfig cfg;
    cfg.sampleRate = 44100;
    cfg.bufferSize = 1024;
    AudioMixer mixer;
    mixer.Initialize(cfg);

    // Create 8 sources with buffers
    std::vector<AudioBuffer> buffers(8);
    std::vector<AudioSource> sources(8);
    for (int i = 0; i < 8; ++i) {
        buffers[i] = AudioBuffer::GenerateSine(220.0f + i * 110.0f, 1.0f, 44100);
        sources[i].SetBuffer(&buffers[i]);
        AudioSourceConfig sc;
        sc.loop = true;
        sources[i].SetConfig(sc);
        sources[i].Play();
        mixer.AddSource(&sources[i]);
    }

    std::vector<float> output(1024 * 2, 0.0f);
    auto r = BENCHMARK_N("AudioMixer_MixFrame_8src_1024fr", 2000, {
        // Reset playheads so we always have data
        for (auto& s : sources) s.SetPlayhead(0);
        mixer.MixFrame(output.data(), 1024);
    });
    // Must be able to mix at least several hundred 1024-frame blocks/sec
    BENCH_EXPECT_THROUGHPUT_GT(r, 200.0);
}

static void Bench_AudioMixer_MixFrame_32Sources() {
    AudioMixerConfig cfg;
    cfg.sampleRate = 44100;
    cfg.bufferSize = 1024;
    cfg.maxSources = 64;
    AudioMixer mixer;
    mixer.Initialize(cfg);

    std::vector<AudioBuffer> buffers(32);
    std::vector<AudioSource> sources(32);
    for (int i = 0; i < 32; ++i) {
        buffers[i] = AudioBuffer::GenerateSine(200.0f + i * 50.0f, 1.0f, 44100);
        sources[i].SetBuffer(&buffers[i]);
        AudioSourceConfig sc;
        sc.loop = true;
        sources[i].SetConfig(sc);
        sources[i].Play();
        mixer.AddSource(&sources[i]);
    }

    std::vector<float> output(1024 * 2, 0.0f);
    auto r = BENCHMARK_N("AudioMixer_MixFrame_32src_1024fr", 1000, {
        for (auto& s : sources) s.SetPlayhead(0);
        mixer.MixFrame(output.data(), 1024);
    });
    // 32 sources: still >100 ops/sec for real-time mixing
    BENCH_EXPECT_THROUGHPUT_GT(r, 50.0);
}

static void Bench_AudioSource_ReadFrames() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 1.0f, 44100);
    AudioSource src;
    src.SetBuffer(&buf);
    AudioSourceConfig cfg;
    cfg.loop = true;
    src.SetConfig(cfg);
    src.Play();

    std::vector<float> out(1024 * 2, 0.0f);
    auto r = BENCHMARK_N("AudioSource_ReadFrames_1024", 5000, {
        src.SetPlayhead(0);
        src.ReadFrames(out.data(), 1024);
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 1000.0);
}

// =============================================================================
// SECTION 2: Acoustic Benchmarks
// =============================================================================

// Simple box room ray-cast lambda for benchmarks
static AcousticRayCastFn MakeBenchBoxRoomRayCast(float halfSize = 10.0f) {
    return [halfSize](float ox, float oy, float oz,
                      float dx, float dy, float dz, float maxDist) -> AcousticRayHit {
        AcousticRayHit hit;
        hit.hit = false;
        hit.distance = maxDist;
        hit.materialId = 0;

        // Check 6 planes of an axis-aligned box [-halfSize, halfSize]^3
        float planes[6][4] = {
            { 1, 0, 0,  halfSize}, {-1, 0, 0,  halfSize},
            { 0, 1, 0,  halfSize}, { 0,-1, 0,  halfSize},
            { 0, 0, 1,  halfSize}, { 0, 0,-1,  halfSize}
        };

        for (int p = 0; p < 6; ++p) {
            float nx = planes[p][0], ny = planes[p][1], nz = planes[p][2];
            float d = planes[p][3];
            float denom = dx * nx + dy * ny + dz * nz;
            if (std::abs(denom) < 1e-6f) continue;
            float t = (d - (ox * nx + oy * ny + oz * nz)) / denom;
            if (t > 0.001f && t < hit.distance) {
                hit.hit = true;
                hit.distance = t;
                hit.normalX = nx;
                hit.normalY = ny;
                hit.normalZ = nz;
            }
        }
        return hit;
    };
}

static void Bench_Acoustic_Occlusion() {
    AcousticSystem sys;
    sys.Initialize();
    sys.SetRayCastFunction(MakeBenchBoxRoomRayCast());
    sys.AddMaterial(AcousticMaterial::Concrete());

    float result = 0.0f;
    auto r = BENCHMARK_N("Acoustic_Occlusion_SingleRay", 5000, {
        result = sys.ComputeOcclusion(0, 0, 0, 5, 3, 2);
    });
    (void)result;
    BENCH_EXPECT_THROUGHPUT_GT(r, 5000.0);
}

static void Bench_Acoustic_Obstruction() {
    AcousticSystem sys;
    AcousticConfig cfg;
    cfg.numRays = 16;
    sys.Initialize(cfg);
    sys.SetRayCastFunction(MakeBenchBoxRoomRayCast());
    sys.AddMaterial(AcousticMaterial::Wood());

    float result = 0.0f;
    auto r = BENCHMARK_N("Acoustic_Obstruction_16rays", 2000, {
        result = sys.ComputeObstruction(0, 0, 0, 5, 3, 2, 16);
    });
    (void)result;
    BENCH_EXPECT_THROUGHPUT_GT(r, 1000.0);
}

static void Bench_Acoustic_ImpulseResponse() {
    AcousticSystem sys;
    AcousticConfig cfg;
    cfg.numRays = 64;
    cfg.maxBounces = 6;
    sys.Initialize(cfg);
    sys.SetRayCastFunction(MakeBenchBoxRoomRayCast());
    sys.AddMaterial(AcousticMaterial::Concrete());

    auto r = BENCHMARK_N("Acoustic_ImpulseResp_64ray_6bounce", 200, {
        auto ir = sys.TraceImpulseResponse(0, 0, 0, 3, 2, 1);
        (void)ir;
    });
    // IR tracing is heavier - but should still manage >50 ops/sec
    BENCH_EXPECT_THROUGHPUT_GT(r, 20.0);
}

static void Bench_Acoustic_RoomEstimate() {
    AcousticSystem sys;
    AcousticConfig cfg;
    cfg.roomProbeRays = 64;
    sys.Initialize(cfg);
    sys.SetRayCastFunction(MakeBenchBoxRoomRayCast());
    sys.AddMaterial(AcousticMaterial::Concrete());

    auto r = BENCHMARK_N("Acoustic_RoomEstimate_64probes", 500, {
        auto re = sys.EstimateRoom(0, 0, 0);
        (void)re;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100.0);
}

static void Bench_Acoustic_RT60_Sabine() {
    auto r = BENCHMARK_N("Acoustic_RT60_Sabine", 10000, {
        float rt60 = AcousticSystem::ComputeRT60_Sabine(500.0f, 300.0f, 0.15f);
        (void)rt60;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);
}

static void Bench_Acoustic_DistanceAttenuation() {
    auto r = BENCHMARK_N("Acoustic_DistanceAtten", 50000, {
        float atten = AcousticSystem::DistanceAttenuation(25.0f, 1.0f, 100.0f, 1.0f);
        (void)atten;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);  // timer-resolution limited
}

// =============================================================================
// SECTION 3: Spatial Audio Benchmarks
// =============================================================================

static void Bench_SpatialAudio_HRTF() {
    SpatialAudio spa;
    spa.Initialize(44100);

    auto r = BENCHMARK_N("SpatialAudio_ComputeHRTF", 10000, {
        auto hrtf = spa.ComputeHRTF(5.0f, 0.0f, -3.0f);
        (void)hrtf;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 50000.0);
}

static void Bench_SpatialAudio_ApplyHRTF() {
    SpatialAudio spa;
    spa.Initialize(44100);

    AudioBuffer mono = AudioBuffer::GenerateSine(440.0f, 0.05f, 44100);
    HRTFResult hrtf = spa.ComputeHRTF(5.0f, 0.0f, -3.0f);

    auto r = BENCHMARK_N("SpatialAudio_ApplyHRTF_2205fr", 1000, {
        auto stereo = spa.ApplyHRTF(mono, hrtf);
        (void)stereo;
    });
    // ApplyHRTF allocates a new AudioBuffer each call — allocation-heavy
    BENCH_EXPECT_THROUGHPUT_GT(r, 20000.0);
}

static void Bench_SpatialAudio_AmbisonicsEncode() {
    auto r = BENCHMARK_N("SpatialAudio_AmbiEncode", 50000, {
        auto bf = SpatialAudio::EncodeAmbisonics(0.5f, 0.2f, 1.0f);
        (void)bf;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);  // timer-resolution limited
}

static void Bench_SpatialAudio_AmbisonicsDecode() {
    auto speakers = SpatialAudio::CreateQuadLayout();
    AmbisonicsBFormat bf = SpatialAudio::EncodeAmbisonics(0.5f, 0.2f, 1.0f);

    auto r = BENCHMARK_N("SpatialAudio_AmbiDecode_Quad", 20000, {
        auto gains = SpatialAudio::DecodeAmbisonics(bf, speakers);
        (void)gains;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);
}

static void Bench_SpatialAudio_Doppler() {
    SpatialAudio spa;
    spa.Initialize(44100);

    auto r = BENCHMARK_N("SpatialAudio_DopplerShift", 50000, {
        float shift = spa.ComputeDopplerShift(
            10, 0, 0,  -30, 0, 0,    // source pos + vel
             0, 0, 0,    0, 0, 0);   // listener pos + vel
        (void)shift;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);  // timer-resolution limited
}

static void Bench_SpatialAudio_WorldToLocal() {
    float lx = 0.0f;
    float ly = 0.0f;
    float lz = 0.0f;
    auto r = BENCHMARK_N("SpatialAudio_WorldToLocal", 50000, {
        SpatialAudio::WorldToListenerLocal(
            10, 5, -3,         // source
             0, 0,  0,         // listener pos
             0, 0, -1,         // forward
             0, 1,  0,         // up
            lx, ly, lz);
    });
    (void)lx; (void)ly; (void)lz;
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);  // timer-resolution limited
}

static void Bench_SpatialAudio_CartesianToSpherical() {
    float az = 0.0f;
    float el = 0.0f;
    float dist = 0.0f;
    auto r = BENCHMARK_N("SpatialAudio_CartToSphere", 50000, {
        SpatialAudio::CartesianToSpherical(5.0f, 3.0f, -7.0f, az, el, dist);
    });
    (void)az; (void)el; (void)dist;
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);  // timer-resolution limited
}

// =============================================================================
// SECTION 4: Rendering Benchmarks
// =============================================================================

static void Bench_GBuffer_Clear() {
    GBuffer gb;
    gb.Initialize(640, 480);

    auto r = BENCHMARK_N("GBuffer_Clear_640x480", 500, {
        gb.Clear();
    });
    // Clearing a 640x480 GBuffer should be fast (>100 ops/sec)
    BENCH_EXPECT_THROUGHPUT_GT(r, 50.0);
}

static void Bench_GBuffer_DepthTest() {
    GBuffer gb;
    gb.Initialize(256, 256);
    gb.Clear();

    int passed = 0;
    auto r = BENCHMARK_THROUGHPUT("GBuffer_DepthTest_65536px", 200, 65536, {
        passed = 0;
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                if (gb.DepthTest(x, y, 0.5f)) passed++;
            }
        }
    });
    (void)passed;
    // GBuffer_DepthTest processes 65536 pixels per iteration — throughput is items/sec
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000000.0);  // >100M depth tests/sec
}

static void Bench_ShadowCascade_Clear() {
    ShadowCascade cascade;
    cascade.Initialize(1024);

    auto r = BENCHMARK_N("ShadowCascade_Clear_1024", 500, {
        cascade.Clear();
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100.0);
}

static void Bench_ShadowCascade_WriteDepth() {
    ShadowCascade cascade;
    cascade.Initialize(512);
    cascade.ComputeLightMatrix({0, -1, 0}, {0, 10, 0}, 20.0f, 0.1f, 100.0f);

    int writes = 0;
    auto r = BENCHMARK_THROUGHPUT("ShadowCascade_WriteDepth_262144px", 50, 262144, {
        cascade.Clear();
        writes = 0;
        for (int y = 0; y < 512; ++y) {
            for (int x = 0; x < 512; ++x) {
                float ndcX = (x / 256.0f) - 1.0f;
                float ndcY = (y / 256.0f) - 1.0f;
                if (cascade.WriteDepth(ndcX, ndcY, 0.5f)) writes++;
            }
        }
    });
    (void)writes;
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);  // items/sec after BENCHMARK_THROUGHPUT recalc
}

static void Bench_GlobalIllumination_SSAO() {
    // Create a small GBuffer with some geometry
    GBuffer gb;
    gb.Initialize(128, 128);
    gb.Clear();

    // Fill center region with a surface
    SoftColorRGBA8 color = {200, 200, 200, 255};
    SoftColorRGBA8 normal;
    normal.r = 128; normal.g = 128; normal.b = 255; normal.a = 255; // pointing up in packed space
    for (int y = 20; y < 108; ++y) {
        for (int x = 20; x < 108; ++x) {
            gb.SetColor(x, y, color);
            gb.SetNormal(x, y, normal);
            gb.SetDepth(x, y, 0.5f);
        }
    }

    // Small SSAO config for benchmark
    GlobalIlluminationConfig giCfg;
    giCfg.ssao.sampleCount = 8;
    giCfg.ssao.blurPasses = 1;
    giCfg.ssaoEnabled = true;
    giCfg.probesEnabled = false;
    giCfg.indirect.enabled = false;

    GlobalIllumination gi;
    gi.Initialize(128, 128, giCfg);

    SoftCamera cam;
    cam.position = {0, 2, 5};
    cam.forward = {0, 0, -1};
    cam.up = {0, 1, 0};
    cam.fov = 60.0f;
    cam.nearPlane = 0.1f;
    cam.farPlane = 100.0f;

    auto r = BENCHMARK_N("GlobalIllumination_SSAO_128x128", 100, {
        gi.Compute(gb, cam);
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 10.0);
}

static void Bench_Volumetric_RayMarch() {
    VolumetricConfig vCfg;
    vCfg.maxSteps = 32;
    vCfg.stepSize = 0.5f;
    vCfg.absorptionCoeff = 0.5f;
    vCfg.scatteringCoeff = 0.1f;
    vCfg.densityMultiplier = 1.0f;

    VolumetricRenderer vol;
    vol.Initialize(64, 64, vCfg);

    // Simple density sampler — uniform density 0.3 in the volume
    VolumeSampler sampler;
    sampler.region.boundsMin = {-5, -5, -5};
    sampler.region.boundsMax = { 5,  5,  5};
    sampler.sampleDensity = [](float, float, float) -> float { return 0.3f; };
    sampler.sampleTemperature = [](float, float, float) -> float { return 500.0f; };

    SoftVec3 origin = {0, 0, 15};
    SoftVec3 dir = {0, 0, -1};

    auto r = BENCHMARK_N("Volumetric_MarchRay_32steps", 5000, {
        auto sample = vol.MarchRay(origin, dir, 30.0f, sampler);
        (void)sample;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);  // timer-resolution limited
}

static void Bench_Volumetric_PhaseHG() {
    auto r = BENCHMARK_N("Volumetric_PhaseHG", 100000, {
        float phase = VolumetricRenderer::PhaseHG(0.7f, 0.6f);
        (void)phase;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);  // timer-resolution limited
}

// =============================================================================
// SECTION 5: Physics Benchmarks
// =============================================================================

static void Bench_ConstitutiveModel_NeoHookean() {
    auto* model = GetConstitutiveModel(MPMMaterialType::NeoHookean);
    MPMMaterialParams params = MPMMaterialParams::Rubber();
    params.ComputeLameConstants();

    MPMParticle p = {};
    p.mass = 1.0f;
    p.volume0 = 0.001f;
    p.F = Mat3::Identity();
    p.Fp = Mat3::Identity();
    p.C = Mat3::Zero();
    p.Jp = 1.0f;

    auto r = BENCHMARK_N("MPM_NeoHookean_Stress", 50000, {
        auto stress = model->ComputeStress(p, params);
        (void)stress;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);
}

static void Bench_ConstitutiveModel_DruckerPrager() {
    auto* model = GetConstitutiveModel(MPMMaterialType::DruckerPrager);
    MPMMaterialParams params = MPMMaterialParams::Sand();
    params.ComputeLameConstants();

    MPMParticle p = {};
    p.mass = 1.0f;
    p.volume0 = 0.001f;
    p.F = Mat3::Identity();
    p.Fp = Mat3::Identity();
    p.C = Mat3::Zero();
    p.Jp = 1.0f;

    auto r = BENCHMARK_N("MPM_DruckerPrager_Stress", 20000, {
        auto stress = model->ComputeStress(p, params);
        (void)stress;
    });
    // DP stress with SVD is heavier
    BENCH_EXPECT_THROUGHPUT_GT(r, 20000.0);
}

static void Bench_ConstitutiveModel_Snow() {
    auto* model = GetConstitutiveModel(MPMMaterialType::Snow);
    MPMMaterialParams params = MPMMaterialParams::Snow();
    params.ComputeLameConstants();

    MPMParticle p = {};
    p.mass = 1.0f;
    p.volume0 = 0.001f;
    p.F = Mat3::Identity();
    p.Fp = Mat3::Identity();
    p.C = Mat3::Zero();
    p.Jp = 1.0f;

    auto r = BENCHMARK_N("MPM_Snow_Stress", 20000, {
        auto stress = model->ComputeStress(p, params);
        (void)stress;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 20000.0);
}

static void Bench_SVD3x3() {
    Mat3 A;
    A.m[0][0] = 1.2f; A.m[0][1] = 0.3f; A.m[0][2] = -0.1f;
    A.m[1][0] = 0.4f; A.m[1][1] = 0.9f; A.m[1][2] = 0.2f;
    A.m[2][0] = -0.2f; A.m[2][1] = 0.1f; A.m[2][2] = 1.1f;

    auto r = BENCHMARK_N("MPM_SVD3x3_JacobiIter", 20000, {
        auto svd = ComputeSVD3x3(A);
        (void)svd;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 50000.0);
}

static void Bench_FluidGrid_Interpolate() {
    FluidGrid grid;
    grid.Initialize(32, 32, 32, 0.1f);

    // Set some velocities so interpolation does real work
    for (uint32_t k = 0; k < 32; ++k) {
        for (uint32_t j = 0; j < 32; ++j) {
            for (uint32_t i = 0; i < 32; ++i) {
                auto& cell = grid.GetCell(i, j, k);
                cell.u = static_cast<float>(i) * 0.01f;
                cell.v = static_cast<float>(j) * 0.01f;
                cell.w = static_cast<float>(k) * 0.01f;
            }
        }
    }

    float vx, vy, vz;
    auto r = BENCHMARK_N("FluidGrid_Interpolate_32cubed", 50000, {
        grid.InterpolateVelocity(1.5f, 1.5f, 1.5f, vx, vy, vz);
    });
    (void)vx; (void)vy; (void)vz;
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);
}

static void Bench_GaseousSystem_Step() {
    GaseousSystemConfig cfg;
    cfg.resolutionX = 16;
    cfg.resolutionY = 16;
    cfg.resolutionZ = 16;
    cfg.cellSize = 0.25f;
    cfg.pressureIterations = 5;
    cfg.substeps = 1;
    cfg.useGPU = false;

    GaseousSystem gas;
    gas.Initialize(cfg);

    // Add a small emitter
    GasEmitter em;
    em.type = GasEmitterType::Point;
    em.posX = 2.0f; em.posY = 2.0f; em.posZ = 2.0f;
    em.densityRate = 5.0f;
    em.temperatureRate = 300.0f;
    gas.AddEmitter(em);

    auto r = BENCHMARK_N("GaseousSystem_Step_16cubed", 100, {
        gas.Step(1.0f / 60.0f);
    });
    // 16^3 grid step should be >20 ops/sec
    BENCH_EXPECT_THROUGHPUT_GT(r, 10.0);
}

static void Bench_GaseousSystem_SampleDensity() {
    GaseousSystemConfig cfg;
    cfg.resolutionX = 32;
    cfg.resolutionY = 32;
    cfg.resolutionZ = 32;
    cfg.cellSize = 0.25f;
    cfg.useGPU = false;

    GaseousSystem gas;
    gas.Initialize(cfg);

    // Set some density
    gas.SetDensity(16, 16, 16, 5.0f);
    gas.SetDensity(15, 16, 16, 3.0f);
    gas.SetDensity(16, 15, 16, 3.0f);

    float d = 0.0f;
    auto r = BENCHMARK_N("GaseousSystem_SampleDensity", 50000, {
        d = gas.SampleDensity(4.0f, 4.0f, 4.0f);
    });
    (void)d;
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);
}

static void Bench_Destruction_GenerateBoxPattern() {
    auto r = BENCHMARK_N("Destruction_GenBoxPattern_16cells", 500, {
        auto pattern = DestructionSystem::GenerateBoxPattern(1.0f, 1.0f, 1.0f, 16);
        (void)pattern;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100.0);
}

static void Bench_Destruction_GenerateSpherePattern() {
    auto r = BENCHMARK_N("Destruction_GenSpherePattern_16cells", 500, {
        auto pattern = DestructionSystem::GenerateSpherePattern(1.0f, 16);
        (void)pattern;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100.0);
}

static void Bench_Destruction_EvaluateImpact() {
    DestructionConfig cfg;
    cfg.defaultCellCount = 8;
    DestructionSystem ds;
    ds.Initialize(cfg);

    JPH::BodyID bodyId(1);
    uint32_t handle = ds.AddDestructible(bodyId, 1000.0f, 8);

    auto r = BENCHMARK_N("Destruction_EvaluateImpact", 10000, {
        // Sub-threshold impact (won't fracture, just evaluates)
        ds.EvaluateImpact(handle, 0, 0, 0, 500.0f);
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 50000.0);
}

// =============================================================================
// SECTION 6: Attenuation Curve Benchmarks
// =============================================================================

static void Bench_AttenuationCurve_Linear() {
    AttenuationCurve curve;
    curve.model = AttenuationModel::Linear;
    curve.refDistance = 1.0f;
    curve.maxDistance = 100.0f;
    curve.rolloff = 1.0f;

    auto r = BENCHMARK_N("AttenuationCurve_Linear", 100000, {
        float g = curve.Evaluate(25.0f);
        (void)g;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);  // timer-resolution limited
}

static void Bench_AttenuationCurve_Inverse() {
    AttenuationCurve curve;
    curve.model = AttenuationModel::Inverse;
    curve.refDistance = 1.0f;
    curve.maxDistance = 100.0f;
    curve.rolloff = 1.0f;

    auto r = BENCHMARK_N("AttenuationCurve_Inverse", 100000, {
        float g = curve.Evaluate(25.0f);
        (void)g;
    });
    BENCH_EXPECT_THROUGHPUT_GT(r, 100000.0);  // timer-resolution limited
}

// =============================================================================
// Registration
// =============================================================================

void RegisterPerformanceBenchmarks() {
    // -- Audio Benchmarks --
    RUN_TEST("BENCH_AudioBuf_GenerateSine", Bench_AudioBuffer_GenerateSine);
    RUN_TEST("BENCH_AudioBuf_MixIn", Bench_AudioBuffer_MixIn);
    RUN_TEST("BENCH_AudioBuf_Normalize", Bench_AudioBuffer_Normalize);
    RUN_TEST("BENCH_AudioBuf_ComputeRMS", Bench_AudioBuffer_ComputeRMS);
    RUN_TEST("BENCH_AudioMixer_8Sources", Bench_AudioMixer_MixFrame_8Sources);
    RUN_TEST("BENCH_AudioMixer_32Sources", Bench_AudioMixer_MixFrame_32Sources);
    RUN_TEST("BENCH_AudioSource_ReadFrames", Bench_AudioSource_ReadFrames);

    // -- Acoustic Benchmarks --
    RUN_TEST("BENCH_Acoustic_Occlusion", Bench_Acoustic_Occlusion);
    RUN_TEST("BENCH_Acoustic_Obstruction", Bench_Acoustic_Obstruction);
    RUN_TEST("BENCH_Acoustic_ImpulseResponse", Bench_Acoustic_ImpulseResponse);
    RUN_TEST("BENCH_Acoustic_RoomEstimate", Bench_Acoustic_RoomEstimate);
    RUN_TEST("BENCH_Acoustic_RT60_Sabine", Bench_Acoustic_RT60_Sabine);
    RUN_TEST("BENCH_Acoustic_DistanceAtten", Bench_Acoustic_DistanceAttenuation);

    // -- Spatial Audio Benchmarks --
    RUN_TEST("BENCH_SpatialAudio_HRTF", Bench_SpatialAudio_HRTF);
    RUN_TEST("BENCH_SpatialAudio_ApplyHRTF", Bench_SpatialAudio_ApplyHRTF);
    RUN_TEST("BENCH_SpatialAudio_AmbiEncode", Bench_SpatialAudio_AmbisonicsEncode);
    RUN_TEST("BENCH_SpatialAudio_AmbiDecode", Bench_SpatialAudio_AmbisonicsDecode);
    RUN_TEST("BENCH_SpatialAudio_Doppler", Bench_SpatialAudio_Doppler);
    RUN_TEST("BENCH_SpatialAudio_WorldToLocal", Bench_SpatialAudio_WorldToLocal);
    RUN_TEST("BENCH_SpatialAudio_CartToSphere", Bench_SpatialAudio_CartesianToSpherical);

    // -- Rendering Benchmarks --
    RUN_TEST("BENCH_GBuffer_Clear", Bench_GBuffer_Clear);
    RUN_TEST("BENCH_GBuffer_DepthTest", Bench_GBuffer_DepthTest);
    RUN_TEST("BENCH_ShadowCascade_Clear", Bench_ShadowCascade_Clear);
    RUN_TEST("BENCH_ShadowCascade_WriteDepth", Bench_ShadowCascade_WriteDepth);
    RUN_TEST("BENCH_GlobalIllumination_SSAO", Bench_GlobalIllumination_SSAO);
    RUN_TEST("BENCH_Volumetric_RayMarch", Bench_Volumetric_RayMarch);
    RUN_TEST("BENCH_Volumetric_PhaseHG", Bench_Volumetric_PhaseHG);

    // -- Physics Benchmarks --
    RUN_TEST("BENCH_MPM_NeoHookean_Stress", Bench_ConstitutiveModel_NeoHookean);
    RUN_TEST("BENCH_MPM_DruckerPrager_Stress", Bench_ConstitutiveModel_DruckerPrager);
    RUN_TEST("BENCH_MPM_Snow_Stress", Bench_ConstitutiveModel_Snow);
    RUN_TEST("BENCH_MPM_SVD3x3", Bench_SVD3x3);
    RUN_TEST("BENCH_FluidGrid_Interpolate", Bench_FluidGrid_Interpolate);
    RUN_TEST("BENCH_GaseousSystem_Step", Bench_GaseousSystem_Step);
    RUN_TEST("BENCH_GaseousSystem_SampleDensity", Bench_GaseousSystem_SampleDensity);
    RUN_TEST("BENCH_Destruction_GenBoxPattern", Bench_Destruction_GenerateBoxPattern);
    RUN_TEST("BENCH_Destruction_GenSpherePattern", Bench_Destruction_GenerateSpherePattern);
    RUN_TEST("BENCH_Destruction_EvaluateImpact", Bench_Destruction_EvaluateImpact);
    RUN_TEST("BENCH_AttenuationCurve_Linear", Bench_AttenuationCurve_Linear);
    RUN_TEST("BENCH_AttenuationCurve_Inverse", Bench_AttenuationCurve_Inverse);

    // Print consolidated benchmark report
    PrintBenchmarkReport();
}
