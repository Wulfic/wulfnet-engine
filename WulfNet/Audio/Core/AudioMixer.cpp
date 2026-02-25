// =============================================================================
// WulfNet Engine - Audio Mixer Implementation
// =============================================================================

#include "WulfNet/Audio/Core/AudioMixer.h"
#include <cstring>
#include <cmath>

namespace WulfNet {

bool AudioMixer::Initialize(const AudioMixerConfig& config) {
    if (config.sampleRate <= 0 || config.bufferSize <= 0) return false;
    if (config.maxSources <= 0) return false;

    m_config = config;
    m_sources.clear();
    m_sources.reserve(config.maxSources);
    m_tempBuffer.resize(config.bufferSize * 2, 0.0f); // Stereo scratch
    m_stats = {};
    m_initialized = true;
    return true;
}

void AudioMixer::Shutdown() {
    m_sources.clear();
    m_tempBuffer.clear();
    m_stats = {};
    m_initialized = false;
}

int AudioMixer::AddSource(AudioSource* source) {
    if (!m_initialized || !source) return -1;
    if (static_cast<int>(m_sources.size()) >= m_config.maxSources) return -1;

    m_sources.push_back(source);
    m_stats.totalSources = static_cast<int>(m_sources.size());
    return static_cast<int>(m_sources.size()) - 1;
}

bool AudioMixer::RemoveSource(AudioSource* source) {
    if (!source) return false;
    for (auto it = m_sources.begin(); it != m_sources.end(); ++it) {
        if (*it == source) {
            m_sources.erase(it);
            m_stats.totalSources = static_cast<int>(m_sources.size());
            return true;
        }
    }
    return false;
}

bool AudioMixer::RemoveSource(int index) {
    if (index < 0 || index >= static_cast<int>(m_sources.size())) return false;
    m_sources.erase(m_sources.begin() + index);
    m_stats.totalSources = static_cast<int>(m_sources.size());
    return true;
}

AudioSource* AudioMixer::GetSource(int index) {
    if (index < 0 || index >= static_cast<int>(m_sources.size())) return nullptr;
    return m_sources[index];
}

const AudioSource* AudioMixer::GetSource(int index) const {
    if (index < 0 || index >= static_cast<int>(m_sources.size())) return nullptr;
    return m_sources[index];
}

int AudioMixer::MixFrame(float* outputStereo, int frameCount) {
    if (!m_initialized || !outputStereo || frameCount <= 0) return 0;

    int totalSamples = frameCount * 2; // Stereo

    // Clear output
    std::memset(outputStereo, 0, totalSamples * sizeof(float));

    // Ensure temp buffer is large enough
    if (static_cast<int>(m_tempBuffer.size()) < totalSamples) {
        m_tempBuffer.resize(totalSamples, 0.0f);
    }

    int activeSources = 0;

    // Mix each source
    for (auto* source : m_sources) {
        if (!source || !source->IsPlaying()) continue;

        // Read frames from source into temp buffer
        int framesRead = source->ReadFrames(m_tempBuffer.data(), frameCount);
        if (framesRead <= 0) continue;

        activeSources++;

        // Add temp buffer into output
        for (int i = 0; i < framesRead * 2; ++i) {
            outputStereo[i] += m_tempBuffer[i];
        }
    }

    // Apply master gain and limiter
    float peak = 0.0f;
    double sumSq = 0.0;
    int clipCount = 0;

    for (int i = 0; i < totalSamples; ++i) {
        outputStereo[i] *= m_config.masterGain;

        // Soft-clip limiter
        if (m_config.limiterEnabled) {
            float absVal = std::abs(outputStereo[i]);
            if (absVal > m_config.limiterThreshold) {
                outputStereo[i] = SoftClip(outputStereo[i]);
                clipCount++;
            }
        }

        float absVal = std::abs(outputStereo[i]);
        if (absVal > peak) peak = absVal;
        sumSq += static_cast<double>(outputStereo[i]) * static_cast<double>(outputStereo[i]);
    }

    // Update stats
    m_stats.activeSources = activeSources;
    m_stats.peakLevel = peak;
    m_stats.rmsLevel = (totalSamples > 0)
        ? static_cast<float>(std::sqrt(sumSq / totalSamples))
        : 0.0f;
    m_stats.framesProcessed += frameCount;
    m_stats.clipCount += clipCount;

    return frameCount;
}

int AudioMixer::MixFrame(AudioBuffer& output, int frameCount) {
    if (frameCount <= 0) return 0;

    AudioFormat stereoFmt;
    stereoFmt.sampleRate = m_config.sampleRate;
    stereoFmt.channels = 2;
    stereoFmt.format = AudioSampleFormat::Float32;

    if (!output.IsValid() || output.GetFrameCount() < frameCount ||
        output.GetChannels() != 2) {
        output.Initialize(stereoFmt, frameCount);
    }

    return MixFrame(output.GetData(), frameCount);
}

void AudioMixer::ClearSources() {
    m_sources.clear();
    m_stats.totalSources = 0;
}

float AudioMixer::SoftClip(float sample) const {
    // tanh-based soft clipping preserves dynamics while preventing harsh distortion
    return std::tanh(sample);
}

} // namespace WulfNet
