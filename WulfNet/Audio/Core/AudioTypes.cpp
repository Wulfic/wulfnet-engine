// =============================================================================
// WulfNet Engine - Audio Types Implementation
// =============================================================================

#include "WulfNet/Audio/Core/AudioTypes.h"
#include <cstring>
#include <numeric>

namespace WulfNet {

// =============================================================================
// AudioBuffer
// =============================================================================

bool AudioBuffer::Initialize(const AudioFormat& format, int frameCount) {
    if (frameCount <= 0 || format.sampleRate <= 0 || format.channels <= 0) return false;
    if (format.channels > 8) return false; // Sanity limit

    m_format = format;
    m_frameCount = frameCount;
    m_samples.assign(frameCount * format.channels, 0.0f);
    return true;
}

AudioBuffer AudioBuffer::GenerateSine(float frequencyHz, float durationSec,
                                       int sampleRate, float amplitude) {
    AudioBuffer buf;
    AudioFormat fmt;
    fmt.sampleRate = sampleRate;
    fmt.channels = 1;
    fmt.format = AudioSampleFormat::Float32;

    int frames = static_cast<int>(durationSec * sampleRate);
    if (frames <= 0) return buf;

    buf.Initialize(fmt, frames);
    float* data = buf.GetData();

    constexpr float kTwoPi = 6.28318530717958647693f;
    float phase = 0.0f;
    float phaseInc = kTwoPi * frequencyHz / static_cast<float>(sampleRate);

    for (int i = 0; i < frames; ++i) {
        data[i] = amplitude * std::sin(phase);
        phase += phaseInc;
        // Keep phase in range to avoid precision loss
        if (phase > kTwoPi) phase -= kTwoPi;
    }

    return buf;
}

AudioBuffer AudioBuffer::GenerateNoise(float durationSec, int sampleRate, float amplitude) {
    AudioBuffer buf;
    AudioFormat fmt;
    fmt.sampleRate = sampleRate;
    fmt.channels = 1;
    fmt.format = AudioSampleFormat::Float32;

    int frames = static_cast<int>(durationSec * sampleRate);
    if (frames <= 0) return buf;

    buf.Initialize(fmt, frames);
    float* data = buf.GetData();

    // Deterministic pseudo-random noise (no external RNG dependency)
    uint32_t seed = 12345;
    for (int i = 0; i < frames; ++i) {
        // xorshift32
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        float r = static_cast<float>(seed) / static_cast<float>(0xFFFFFFFF);
        data[i] = amplitude * (r * 2.0f - 1.0f);
    }

    return buf;
}

AudioBuffer AudioBuffer::GenerateSilence(float durationSec, int sampleRate) {
    AudioBuffer buf;
    AudioFormat fmt;
    fmt.sampleRate = sampleRate;
    fmt.channels = 1;
    fmt.format = AudioSampleFormat::Float32;

    int frames = static_cast<int>(durationSec * sampleRate);
    if (frames <= 0) return buf;

    buf.Initialize(fmt, frames);
    return buf;
}

bool AudioBuffer::LoadFromFloat(const float* data, int frameCount, const AudioFormat& fmt) {
    if (!data || frameCount <= 0) return false;
    if (!Initialize(fmt, frameCount)) return false;
    std::memcpy(m_samples.data(), data, frameCount * fmt.channels * sizeof(float));
    return true;
}

float AudioBuffer::ComputeRMS() const {
    if (m_samples.empty()) return 0.0f;

    double sumSq = 0.0;
    for (float s : m_samples) {
        sumSq += static_cast<double>(s) * static_cast<double>(s);
    }
    return static_cast<float>(std::sqrt(sumSq / static_cast<double>(m_samples.size())));
}

float AudioBuffer::ComputePeak() const {
    float peak = 0.0f;
    for (float s : m_samples) {
        float a = std::abs(s);
        if (a > peak) peak = a;
    }
    return peak;
}

void AudioBuffer::Clear() {
    std::fill(m_samples.begin(), m_samples.end(), 0.0f);
}

void AudioBuffer::Resize(int newFrameCount) {
    if (newFrameCount <= 0) return;
    m_frameCount = newFrameCount;
    m_samples.resize(newFrameCount * m_format.channels, 0.0f);
}

void AudioBuffer::MixIn(const AudioBuffer& other, float gain, int destOffset) {
    if (!other.IsValid() || !IsValid()) return;

    int srcFrames = other.GetFrameCount();
    int srcChannels = other.GetChannels();
    int dstChannels = m_format.channels;
    const float* srcData = other.GetData();

    for (int f = 0; f < srcFrames; ++f) {
        int dstFrame = f + destOffset;
        if (dstFrame < 0) continue;
        if (dstFrame >= m_frameCount) break;

        // Simple channel mapping: mono->stereo duplicates, stereo->mono averages
        for (int dc = 0; dc < dstChannels; ++dc) {
            int sc = (srcChannels == 1) ? 0 : std::min(dc, srcChannels - 1);
            float srcSample = srcData[f * srcChannels + sc] * gain;
            m_samples[dstFrame * dstChannels + dc] += srcSample;
        }
    }
}

void AudioBuffer::ApplyGain(float gain) {
    for (float& s : m_samples) {
        s *= gain;
    }
}

void AudioBuffer::Normalize(float targetPeak) {
    float peak = ComputePeak();
    if (peak < 1e-8f) return;
    float scale = targetPeak / peak;
    ApplyGain(scale);
}

// =============================================================================
// AudioSource
// =============================================================================

void AudioSource::SetBuffer(const AudioBuffer* buffer) {
    m_buffer = buffer;
    m_playhead = 0;
    m_playing = false;
}

int AudioSource::ReadFrames(float* outStereo, int frameCount) {
    if (!m_playing || !m_buffer || !m_buffer->IsValid() || frameCount <= 0) {
        // Fill with silence
        std::memset(outStereo, 0, frameCount * 2 * sizeof(float));
        return 0;
    }

    const float* srcData = m_buffer->GetData();
    int srcFrames = m_buffer->GetFrameCount();
    int srcChannels = m_buffer->GetChannels();

    // Compute stereo panning gains (constant-power)
    float panAngle = (m_config.pan + 1.0f) * 0.25f * 3.14159265f; // Map [-1,1] to [0, pi/2]
    float leftGain  = m_config.gain * std::cos(panAngle);
    float rightGain = m_config.gain * std::sin(panAngle);

    int framesWritten = 0;

    for (int f = 0; f < frameCount; ++f) {
        if (m_playhead >= srcFrames) {
            if (m_config.loop) {
                m_playhead = 0;
            } else {
                m_playing = false;
                // Fill remaining with silence
                for (int r = f; r < frameCount; ++r) {
                    outStereo[r * 2 + 0] = 0.0f;
                    outStereo[r * 2 + 1] = 0.0f;
                }
                break;
            }
        }

        // Read source sample(s)
        float monoSample = 0.0f;
        if (srcChannels == 1) {
            monoSample = srcData[m_playhead];
        } else {
            // Stereo source: average for mono processing, then re-pan
            float left  = srcData[m_playhead * srcChannels + 0];
            float right = srcData[m_playhead * srcChannels + 1];
            monoSample = (left + right) * 0.5f;
        }

        // Apply fade-in
        float fadeGain = 1.0f;
        if (m_config.fadeInSec > 0.0f) {
            float fadeInFrames = m_config.fadeInSec * m_buffer->GetSampleRate();
            if (m_playhead < static_cast<int>(fadeInFrames)) {
                fadeGain *= static_cast<float>(m_playhead) / fadeInFrames;
            }
        }

        // Apply fade-out
        if (m_config.fadeOutSec > 0.0f) {
            float fadeOutFrames = m_config.fadeOutSec * m_buffer->GetSampleRate();
            int fadeOutStart = srcFrames - static_cast<int>(fadeOutFrames);
            if (m_playhead >= fadeOutStart && fadeOutStart < srcFrames) {
                float remaining = static_cast<float>(srcFrames - m_playhead);
                fadeGain *= remaining / fadeOutFrames;
            }
        }

        monoSample *= fadeGain;

        outStereo[f * 2 + 0] = monoSample * leftGain;
        outStereo[f * 2 + 1] = monoSample * rightGain;

        m_playhead++;
        framesWritten++;
    }

    return framesWritten;
}

} // namespace WulfNet
