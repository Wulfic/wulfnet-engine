// =============================================================================
// WulfNet Engine - Audio Engine Tests
// =============================================================================
// Tests for AudioBuffer, AudioSource, and AudioMixer.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/Audio/Core/AudioTypes.h>
#include <WulfNet/Audio/Core/AudioMixer.h>
#include <cmath>
#include <vector>

using namespace WulfNet;

static constexpr float kPi = 3.14159265358979323846f;

// =============================================================================
// AudioFormat Tests
// =============================================================================

static void Test_AudioFormat_Defaults() {
    AudioFormat fmt;
    EXPECT_EQ(fmt.sampleRate, 44100);
    EXPECT_EQ(fmt.channels, 1);
    EXPECT_EQ(fmt.BytesPerSample(), 4); // Float32
    EXPECT_EQ(fmt.BytesPerFrame(), 4);  // 1 channel * 4 bytes
}

static void Test_AudioFormat_Stereo() {
    AudioFormat fmt;
    fmt.channels = 2;
    EXPECT_EQ(fmt.BytesPerFrame(), 8); // 2 channels * 4 bytes
}

static void Test_AudioFormat_Int16() {
    AudioFormat fmt;
    fmt.format = AudioSampleFormat::Int16;
    EXPECT_EQ(fmt.BytesPerSample(), 2);
    fmt.channels = 2;
    EXPECT_EQ(fmt.BytesPerFrame(), 4);
}

static void Test_AudioFormat_Equality() {
    AudioFormat a, b;
    EXPECT_TRUE(a == b);
    b.sampleRate = 48000;
    EXPECT_TRUE(a != b);
}

// =============================================================================
// AudioBuffer Tests
// =============================================================================

static void Test_AudioBuffer_Initialize() {
    AudioBuffer buf;
    AudioFormat fmt;
    fmt.sampleRate = 44100;
    fmt.channels = 1;
    EXPECT_TRUE(buf.Initialize(fmt, 1024));
    EXPECT_EQ(buf.GetFrameCount(), 1024);
    EXPECT_EQ(buf.GetSampleCount(), 1024);
    EXPECT_EQ(buf.GetSampleRate(), 44100);
    EXPECT_EQ(buf.GetChannels(), 1);
    EXPECT_TRUE(buf.IsValid());
}

static void Test_AudioBuffer_InitializeInvalid() {
    AudioBuffer buf;
    AudioFormat fmt;
    EXPECT_FALSE(buf.Initialize(fmt, 0));
    EXPECT_FALSE(buf.Initialize(fmt, -1));
    fmt.sampleRate = 0;
    EXPECT_FALSE(buf.Initialize(fmt, 100));
}

static void Test_AudioBuffer_InitializeStereo() {
    AudioBuffer buf;
    AudioFormat fmt;
    fmt.channels = 2;
    EXPECT_TRUE(buf.Initialize(fmt, 512));
    EXPECT_EQ(buf.GetFrameCount(), 512);
    EXPECT_EQ(buf.GetSampleCount(), 1024); // 512 frames * 2 channels
}

static void Test_AudioBuffer_GenerateSine() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.1f, 44100, 1.0f);
    EXPECT_TRUE(buf.IsValid());
    EXPECT_EQ(buf.GetFrameCount(), 4410); // 0.1s * 44100
    EXPECT_EQ(buf.GetChannels(), 1);

    // Peak should be close to 1.0
    float peak = buf.ComputePeak();
    EXPECT_GT(peak, 0.99f);
    EXPECT_LE(peak, 1.01f);

    // RMS of a sine wave = amplitude / sqrt(2) ≈ 0.707
    float rms = buf.ComputeRMS();
    EXPECT_NEAR(rms, 0.7071f, 0.02f);
}

static void Test_AudioBuffer_GenerateNoise() {
    AudioBuffer buf = AudioBuffer::GenerateNoise(0.1f, 44100, 1.0f);
    EXPECT_TRUE(buf.IsValid());
    EXPECT_EQ(buf.GetFrameCount(), 4410);

    // Noise should have non-zero peak and RMS
    EXPECT_GT(buf.ComputePeak(), 0.1f);
    EXPECT_GT(buf.ComputeRMS(), 0.1f);
}

static void Test_AudioBuffer_GenerateSilence() {
    AudioBuffer buf = AudioBuffer::GenerateSilence(0.05f, 44100);
    EXPECT_TRUE(buf.IsValid());
    EXPECT_EQ(buf.GetFrameCount(), 2205);
    EXPECT_NEAR(buf.ComputePeak(), 0.0f, 1e-6f);
    EXPECT_NEAR(buf.ComputeRMS(), 0.0f, 1e-6f);
}

static void Test_AudioBuffer_Duration() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 1.0f, 44100);
    EXPECT_NEAR(buf.GetDuration(), 1.0f, 0.001f);
}

static void Test_AudioBuffer_Clear() {
    AudioBuffer buf = AudioBuffer::GenerateNoise(0.01f, 44100);
    EXPECT_GT(buf.ComputePeak(), 0.0f);
    buf.Clear();
    EXPECT_NEAR(buf.ComputePeak(), 0.0f, 1e-6f);
}

static void Test_AudioBuffer_ApplyGain() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.01f, 44100, 1.0f);
    float origPeak = buf.ComputePeak();
    buf.ApplyGain(0.5f);
    EXPECT_NEAR(buf.ComputePeak(), origPeak * 0.5f, 0.01f);
}

static void Test_AudioBuffer_Normalize() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.01f, 44100, 0.3f);
    EXPECT_NEAR(buf.ComputePeak(), 0.3f, 0.01f);
    buf.Normalize(1.0f);
    EXPECT_NEAR(buf.ComputePeak(), 1.0f, 0.01f);
}

static void Test_AudioBuffer_LoadFromFloat() {
    float samples[4] = {0.1f, 0.5f, -0.3f, 0.8f};
    AudioBuffer buf;
    AudioFormat fmt;
    fmt.channels = 1;
    EXPECT_TRUE(buf.LoadFromFloat(samples, 4, fmt));
    EXPECT_EQ(buf.GetFrameCount(), 4);
    EXPECT_NEAR(buf.GetData()[0], 0.1f, 1e-6f);
    EXPECT_NEAR(buf.GetData()[3], 0.8f, 1e-6f);
}

static void Test_AudioBuffer_Resize() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.01f, 44100);
    int origFrames = buf.GetFrameCount();
    buf.Resize(origFrames * 2);
    EXPECT_EQ(buf.GetFrameCount(), origFrames * 2);
    // New frames should be zero
    EXPECT_NEAR(buf.GetData()[origFrames], 0.0f, 1e-6f);
}

static void Test_AudioBuffer_MixIn() {
    AudioBuffer dst = AudioBuffer::GenerateSilence(0.01f, 44100);
    AudioBuffer src = AudioBuffer::GenerateSine(440.0f, 0.01f, 44100, 0.5f);

    dst.MixIn(src, 1.0f);
    EXPECT_NEAR(dst.ComputePeak(), 0.5f, 0.02f);

    // Mix again — should double
    dst.Clear();
    dst.MixIn(src, 1.0f);
    dst.MixIn(src, 1.0f);
    EXPECT_NEAR(dst.ComputePeak(), 1.0f, 0.02f);
}

// =============================================================================
// AudioSource Tests
// =============================================================================

static void Test_AudioSource_DefaultState() {
    AudioSource src;
    EXPECT_FALSE(src.IsPlaying());
    EXPECT_TRUE(src.GetBuffer() == nullptr);
    EXPECT_EQ(src.GetPlayhead(), 0);
    EXPECT_NEAR(src.GetGain(), 1.0f, 0.001f);
    EXPECT_NEAR(src.GetPan(), 0.0f, 0.001f);
}

static void Test_AudioSource_SetBuffer() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.1f);
    AudioSource src;
    src.SetBuffer(&buf);
    EXPECT_TRUE(src.GetBuffer() != nullptr);
}

static void Test_AudioSource_PlayStop() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.1f);
    AudioSource src;
    src.SetBuffer(&buf);
    src.Play();
    EXPECT_TRUE(src.IsPlaying());
    src.Stop();
    EXPECT_FALSE(src.IsPlaying());
    EXPECT_EQ(src.GetPlayhead(), 0);
}

static void Test_AudioSource_PauseResume() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.1f);
    AudioSource src;
    src.SetBuffer(&buf);
    src.Play();
    EXPECT_TRUE(src.IsPlaying());

    // Read some frames to advance playhead
    std::vector<float> out(256 * 2);
    src.ReadFrames(out.data(), 256);
    int pos = src.GetPlayhead();
    EXPECT_GT(pos, 0);

    src.Pause();
    EXPECT_FALSE(src.IsPlaying());
    EXPECT_EQ(src.GetPlayhead(), pos); // Position preserved

    src.Resume();
    EXPECT_TRUE(src.IsPlaying());
}

static void Test_AudioSource_ReadFrames() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.1f, 44100, 0.8f);
    AudioSource src;
    src.SetBuffer(&buf);
    src.Play();

    std::vector<float> out(512 * 2, 0.0f);
    int read = src.ReadFrames(out.data(), 512);
    EXPECT_EQ(read, 512);

    // Should have non-zero output (center pan = equal L/R)
    bool hasOutput = false;
    for (int i = 0; i < 512 * 2; ++i) {
        if (std::abs(out[i]) > 0.01f) { hasOutput = true; break; }
    }
    EXPECT_TRUE(hasOutput);
}

static void Test_AudioSource_ReadFramesStopped() {
    AudioSource src;
    std::vector<float> out(256 * 2, 1.0f);
    int read = src.ReadFrames(out.data(), 256);
    EXPECT_EQ(read, 0);
    // Output should be zeroed
    EXPECT_NEAR(out[0], 0.0f, 1e-6f);
}

static void Test_AudioSource_Gain() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.01f, 44100, 1.0f);
    AudioSource src;
    src.SetBuffer(&buf);

    // Full gain
    src.SetGain(1.0f);
    src.Play();
    std::vector<float> out1(441 * 2, 0.0f);
    src.ReadFrames(out1.data(), 441);

    // Half gain
    src.SetGain(0.5f);
    src.Play(); // Reset playhead
    std::vector<float> out2(441 * 2, 0.0f);
    src.ReadFrames(out2.data(), 441);

    // Find peak of each
    float peak1 = 0.0f, peak2 = 0.0f;
    for (int i = 0; i < 441 * 2; ++i) {
        peak1 = std::max(peak1, std::abs(out1[i]));
        peak2 = std::max(peak2, std::abs(out2[i]));
    }

    // Half gain should produce roughly half peak
    EXPECT_GT(peak1, peak2);
    EXPECT_NEAR(peak2 / peak1, 0.5f, 0.1f);
}

static void Test_AudioSource_Pan() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.01f, 44100, 1.0f);
    AudioSource src;
    src.SetBuffer(&buf);

    // Pan hard left
    src.SetPan(-1.0f);
    src.Play();
    std::vector<float> outL(441 * 2, 0.0f);
    src.ReadFrames(outL.data(), 441);

    float leftPeak = 0.0f, rightPeak = 0.0f;
    for (int i = 0; i < 441; ++i) {
        leftPeak = std::max(leftPeak, std::abs(outL[i * 2 + 0]));
        rightPeak = std::max(rightPeak, std::abs(outL[i * 2 + 1]));
    }
    EXPECT_GT(leftPeak, rightPeak);  // Left channel louder

    // Pan hard right
    src.SetPan(1.0f);
    src.Play();
    std::vector<float> outR(441 * 2, 0.0f);
    src.ReadFrames(outR.data(), 441);

    leftPeak = 0.0f; rightPeak = 0.0f;
    for (int i = 0; i < 441; ++i) {
        leftPeak = std::max(leftPeak, std::abs(outR[i * 2 + 0]));
        rightPeak = std::max(rightPeak, std::abs(outR[i * 2 + 1]));
    }
    EXPECT_GT(rightPeak, leftPeak);  // Right channel louder
}

static void Test_AudioSource_Loop() {
    // Create a very short buffer
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.01f, 44100, 0.5f);
    AudioSource src;
    src.SetBuffer(&buf);

    AudioSourceConfig cfg;
    cfg.loop = true;
    src.SetConfig(cfg);
    src.Play();

    // Read more frames than the buffer contains
    int bufFrames = buf.GetFrameCount();
    int readTotal = bufFrames * 3;
    std::vector<float> out(readTotal * 2, 0.0f);
    int read = src.ReadFrames(out.data(), readTotal);

    EXPECT_EQ(read, readTotal);
    EXPECT_TRUE(src.IsPlaying()); // Still playing (looped)
}

static void Test_AudioSource_EndOfBuffer() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.01f, 44100, 0.5f);
    AudioSource src;
    src.SetBuffer(&buf);
    src.Play();

    int bufFrames = buf.GetFrameCount();
    // Read entire buffer plus extra
    std::vector<float> out((bufFrames + 100) * 2, 0.0f);
    int read = src.ReadFrames(out.data(), bufFrames + 100);

    // Should have read only up to buffer end
    EXPECT_EQ(read, bufFrames);
    EXPECT_FALSE(src.IsPlaying()); // Stopped at end
}

static void Test_AudioSource_FadeIn() {
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.5f, 44100, 1.0f);
    AudioSource src;
    src.SetBuffer(&buf);

    AudioSourceConfig cfg;
    cfg.fadeInSec = 0.1f; // 4410 frames
    src.SetConfig(cfg);
    src.Play();

    // Read early frames (should be quiet)
    std::vector<float> early(100 * 2, 0.0f);
    src.ReadFrames(early.data(), 100);

    float earlyPeak = 0.0f;
    for (int i = 0; i < 100 * 2; ++i) earlyPeak = std::max(earlyPeak, std::abs(early[i]));

    // Read later frames (should be louder) — skip ahead
    src.SetPlayhead(4000);
    std::vector<float> late(100 * 2, 0.0f);
    src.ReadFrames(late.data(), 100);

    float latePeak = 0.0f;
    for (int i = 0; i < 100 * 2; ++i) latePeak = std::max(latePeak, std::abs(late[i]));

    EXPECT_GT(latePeak, earlyPeak);
}

static void Test_AudioSource_Position3D() {
    AudioSource src;
    AudioSource::Position3D pos = {1.0f, 2.0f, 3.0f};
    src.SetPosition(pos);
    EXPECT_NEAR(src.GetPosition().x, 1.0f, 0.001f);
    EXPECT_NEAR(src.GetPosition().y, 2.0f, 0.001f);
    EXPECT_NEAR(src.GetPosition().z, 3.0f, 0.001f);
}

// =============================================================================
// AudioMixer Tests
// =============================================================================

static void Test_AudioMixer_Initialize() {
    AudioMixer mixer;
    AudioMixerConfig cfg;
    cfg.sampleRate = 44100;
    cfg.bufferSize = 1024;
    EXPECT_TRUE(mixer.Initialize(cfg));
    EXPECT_EQ(mixer.GetSourceCount(), 0);
}

static void Test_AudioMixer_InitializeInvalid() {
    AudioMixer mixer;
    AudioMixerConfig cfg;
    cfg.sampleRate = 0;
    EXPECT_FALSE(mixer.Initialize(cfg));
    cfg.sampleRate = 44100;
    cfg.bufferSize = 0;
    EXPECT_FALSE(mixer.Initialize(cfg));
}

static void Test_AudioMixer_AddRemoveSource() {
    AudioMixer mixer;
    mixer.Initialize();

    AudioSource src1, src2;
    int idx = mixer.AddSource(&src1);
    EXPECT_EQ(idx, 0);
    EXPECT_EQ(mixer.GetSourceCount(), 1);

    mixer.AddSource(&src2);
    EXPECT_EQ(mixer.GetSourceCount(), 2);

    EXPECT_TRUE(mixer.RemoveSource(&src1));
    EXPECT_EQ(mixer.GetSourceCount(), 1);
}

static void Test_AudioMixer_AddNull() {
    AudioMixer mixer;
    mixer.Initialize();
    EXPECT_EQ(mixer.AddSource(nullptr), -1);
}

static void Test_AudioMixer_MaxSources() {
    AudioMixer mixer;
    AudioMixerConfig cfg;
    cfg.maxSources = 2;
    mixer.Initialize(cfg);

    AudioSource s1, s2, s3;
    EXPECT_GE(mixer.AddSource(&s1), 0);
    EXPECT_GE(mixer.AddSource(&s2), 0);
    EXPECT_EQ(mixer.AddSource(&s3), -1); // Exceeds max
}

static void Test_AudioMixer_MixSilence() {
    AudioMixer mixer;
    mixer.Initialize();

    std::vector<float> out(1024 * 2, 1.0f);
    int frames = mixer.MixFrame(out.data(), 1024);
    EXPECT_EQ(frames, 1024);

    // No sources = silence
    float peak = 0.0f;
    for (float s : out) peak = std::max(peak, std::abs(s));
    EXPECT_NEAR(peak, 0.0f, 1e-6f);
}

static void Test_AudioMixer_MixSingleSource() {
    AudioMixer mixer;
    mixer.Initialize();

    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.1f, 44100, 0.8f);
    AudioSource src;
    src.SetBuffer(&buf);
    src.Play();

    mixer.AddSource(&src);

    std::vector<float> out(512 * 2, 0.0f);
    mixer.MixFrame(out.data(), 512);

    // Should have audio output
    const auto& stats = mixer.GetStats();
    EXPECT_EQ(stats.activeSources, 1);
    EXPECT_GT(stats.peakLevel, 0.1f);
    EXPECT_GT(stats.rmsLevel, 0.01f);
}

static void Test_AudioMixer_MixMultipleSources() {
    AudioMixer mixer;
    mixer.Initialize();

    AudioBuffer buf1 = AudioBuffer::GenerateSine(440.0f, 0.1f, 44100, 0.3f);
    AudioBuffer buf2 = AudioBuffer::GenerateSine(880.0f, 0.1f, 44100, 0.3f);

    AudioSource src1, src2;
    src1.SetBuffer(&buf1); src1.Play();
    src2.SetBuffer(&buf2); src2.Play();

    mixer.AddSource(&src1);
    mixer.AddSource(&src2);

    std::vector<float> out(256 * 2, 0.0f);
    mixer.MixFrame(out.data(), 256);

    EXPECT_EQ(mixer.GetStats().activeSources, 2);
    EXPECT_GT(mixer.GetStats().peakLevel, 0.1f);
}

static void Test_AudioMixer_MasterGain() {
    AudioMixer mixer;
    mixer.Initialize();

    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.1f, 44100, 0.5f);
    AudioSource src;
    src.SetBuffer(&buf);
    src.Play();
    mixer.AddSource(&src);

    // Mix at master gain 1.0
    std::vector<float> out1(256 * 2, 0.0f);
    mixer.MixFrame(out1.data(), 256);
    float peak1 = mixer.GetStats().peakLevel;

    // Reset and mix at master gain 0.5
    src.Play();
    mixer.SetMasterGain(0.5f);
    std::vector<float> out2(256 * 2, 0.0f);
    mixer.MixFrame(out2.data(), 256);
    float peak2 = mixer.GetStats().peakLevel;

    EXPECT_GT(peak1, peak2);
}

static void Test_AudioMixer_Limiter() {
    AudioMixer mixer;
    AudioMixerConfig cfg;
    cfg.limiterEnabled = true;
    cfg.limiterThreshold = 0.5f;
    mixer.Initialize(cfg);

    // Create a loud source
    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.1f, 44100, 2.0f);
    AudioSource src;
    src.SetBuffer(&buf);
    src.Play();
    mixer.AddSource(&src);

    std::vector<float> out(256 * 2, 0.0f);
    mixer.MixFrame(out.data(), 256);

    // Limiter should keep peak below ~1.0 (tanh clips smoothly)
    EXPECT_LE(mixer.GetStats().peakLevel, 1.1f);
    EXPECT_GT(mixer.GetStats().clipCount, 0);
}

static void Test_AudioMixer_MixIntoBuffer() {
    AudioMixer mixer;
    mixer.Initialize();

    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.1f, 44100, 0.5f);
    AudioSource src;
    src.SetBuffer(&buf);
    src.Play();
    mixer.AddSource(&src);

    AudioBuffer output;
    mixer.MixFrame(output, 256);

    EXPECT_TRUE(output.IsValid());
    EXPECT_EQ(output.GetFrameCount(), 256);
    EXPECT_EQ(output.GetChannels(), 2);
}

static void Test_AudioMixer_ClearSources() {
    AudioMixer mixer;
    mixer.Initialize();

    AudioSource s1, s2;
    mixer.AddSource(&s1);
    mixer.AddSource(&s2);
    EXPECT_EQ(mixer.GetSourceCount(), 2);

    mixer.ClearSources();
    EXPECT_EQ(mixer.GetSourceCount(), 0);
}

static void Test_AudioMixer_Shutdown() {
    AudioMixer mixer;
    mixer.Initialize();

    AudioSource src;
    mixer.AddSource(&src);
    mixer.Shutdown();

    EXPECT_EQ(mixer.GetSourceCount(), 0);
    // Should not crash after shutdown
    std::vector<float> out(128 * 2);
    EXPECT_EQ(mixer.MixFrame(out.data(), 128), 0);
}

static void Test_AudioMixer_Stats() {
    AudioMixer mixer;
    mixer.Initialize();

    AudioBuffer buf = AudioBuffer::GenerateSine(440.0f, 0.1f, 44100, 0.5f);
    AudioSource src;
    src.SetBuffer(&buf);
    src.Play();
    mixer.AddSource(&src);

    std::vector<float> out(256 * 2);
    mixer.MixFrame(out.data(), 256);

    const auto& stats = mixer.GetStats();
    EXPECT_EQ(stats.framesProcessed, 256);
    EXPECT_EQ(stats.totalSources, 1);
    EXPECT_EQ(stats.activeSources, 1);

    // Mix another frame
    mixer.MixFrame(out.data(), 256);
    EXPECT_EQ(mixer.GetStats().framesProcessed, 512);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterAudioEngineTests() {
    // AudioFormat
    RUN_TEST("AudioFormat_Defaults", Test_AudioFormat_Defaults);
    RUN_TEST("AudioFormat_Stereo", Test_AudioFormat_Stereo);
    RUN_TEST("AudioFormat_Int16", Test_AudioFormat_Int16);
    RUN_TEST("AudioFormat_Equality", Test_AudioFormat_Equality);

    // AudioBuffer
    RUN_TEST("AudioBuffer_Initialize", Test_AudioBuffer_Initialize);
    RUN_TEST("AudioBuffer_InitializeInvalid", Test_AudioBuffer_InitializeInvalid);
    RUN_TEST("AudioBuffer_InitializeStereo", Test_AudioBuffer_InitializeStereo);
    RUN_TEST("AudioBuffer_GenerateSine", Test_AudioBuffer_GenerateSine);
    RUN_TEST("AudioBuffer_GenerateNoise", Test_AudioBuffer_GenerateNoise);
    RUN_TEST("AudioBuffer_GenerateSilence", Test_AudioBuffer_GenerateSilence);
    RUN_TEST("AudioBuffer_Duration", Test_AudioBuffer_Duration);
    RUN_TEST("AudioBuffer_Clear", Test_AudioBuffer_Clear);
    RUN_TEST("AudioBuffer_ApplyGain", Test_AudioBuffer_ApplyGain);
    RUN_TEST("AudioBuffer_Normalize", Test_AudioBuffer_Normalize);
    RUN_TEST("AudioBuffer_LoadFromFloat", Test_AudioBuffer_LoadFromFloat);
    RUN_TEST("AudioBuffer_Resize", Test_AudioBuffer_Resize);
    RUN_TEST("AudioBuffer_MixIn", Test_AudioBuffer_MixIn);

    // AudioSource
    RUN_TEST("AudioSource_DefaultState", Test_AudioSource_DefaultState);
    RUN_TEST("AudioSource_SetBuffer", Test_AudioSource_SetBuffer);
    RUN_TEST("AudioSource_PlayStop", Test_AudioSource_PlayStop);
    RUN_TEST("AudioSource_PauseResume", Test_AudioSource_PauseResume);
    RUN_TEST("AudioSource_ReadFrames", Test_AudioSource_ReadFrames);
    RUN_TEST("AudioSource_ReadFramesStopped", Test_AudioSource_ReadFramesStopped);
    RUN_TEST("AudioSource_Gain", Test_AudioSource_Gain);
    RUN_TEST("AudioSource_Pan", Test_AudioSource_Pan);
    RUN_TEST("AudioSource_Loop", Test_AudioSource_Loop);
    RUN_TEST("AudioSource_EndOfBuffer", Test_AudioSource_EndOfBuffer);
    RUN_TEST("AudioSource_FadeIn", Test_AudioSource_FadeIn);
    RUN_TEST("AudioSource_Position3D", Test_AudioSource_Position3D);

    // AudioMixer
    RUN_TEST("AudioMixer_Initialize", Test_AudioMixer_Initialize);
    RUN_TEST("AudioMixer_InitializeInvalid", Test_AudioMixer_InitializeInvalid);
    RUN_TEST("AudioMixer_AddRemoveSource", Test_AudioMixer_AddRemoveSource);
    RUN_TEST("AudioMixer_AddNull", Test_AudioMixer_AddNull);
    RUN_TEST("AudioMixer_MaxSources", Test_AudioMixer_MaxSources);
    RUN_TEST("AudioMixer_MixSilence", Test_AudioMixer_MixSilence);
    RUN_TEST("AudioMixer_MixSingleSource", Test_AudioMixer_MixSingleSource);
    RUN_TEST("AudioMixer_MixMultipleSources", Test_AudioMixer_MixMultipleSources);
    RUN_TEST("AudioMixer_MasterGain", Test_AudioMixer_MasterGain);
    RUN_TEST("AudioMixer_Limiter", Test_AudioMixer_Limiter);
    RUN_TEST("AudioMixer_MixIntoBuffer", Test_AudioMixer_MixIntoBuffer);
    RUN_TEST("AudioMixer_ClearSources", Test_AudioMixer_ClearSources);
    RUN_TEST("AudioMixer_Shutdown", Test_AudioMixer_Shutdown);
    RUN_TEST("AudioMixer_Stats", Test_AudioMixer_Stats);
}
