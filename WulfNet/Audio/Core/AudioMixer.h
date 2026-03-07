// =============================================================================
// WulfNet Engine - Audio Mixer
// =============================================================================
// Multi-source audio mixer that combines AudioSource outputs into a single
// stereo output buffer. Supports master gain, per-source gain/pan, and a
// simple limiter to prevent clipping.
//
// Usage:
//   AudioMixer mixer;
//   mixer.Initialize(config);
//   mixer.AddSource(&source1);
//   mixer.AddSource(&source2);
//   mixer.MixFrame(outputBuffer, 1024);
// =============================================================================

#pragma once

#include "WulfNet/Audio/Core/AudioTypes.h"
#include "WulfNet/API.h"
#include <vector>
#include <cstdint>
#include <algorithm>

namespace WulfNet {

// =============================================================================
// Configuration
// =============================================================================

struct AudioMixerConfig {
    int   sampleRate    = 44100;    ///< Output sample rate
    int   bufferSize    = 1024;     ///< Mix buffer size in frames
    float masterGain    = 1.0f;     ///< Master volume
    bool  limiterEnabled = true;    ///< Soft-clip limiter
    float limiterThreshold = 0.95f; ///< Threshold for limiter activation
    int   maxSources    = 64;       ///< Maximum simultaneous sources

    /// Validate configuration and return true if valid.
    bool Validate() const {
        if (sampleRate <= 0 || sampleRate > 192000) return false;
        if (bufferSize <= 0 || bufferSize > 65536) return false;
        if (masterGain < 0.0f) return false;
        if (limiterThreshold <= 0.0f || limiterThreshold > 1.0f) return false;
        if (maxSources <= 0) return false;
        return true;
    }
};

struct AudioMixerStats {
    int   activeSources   = 0;     ///< Number of currently playing sources
    int   totalSources    = 0;     ///< Number of registered sources
    float peakLevel       = 0.0f;  ///< Peak amplitude in last mix frame
    float rmsLevel        = 0.0f;  ///< RMS level in last mix frame
    int   framesProcessed = 0;     ///< Total frames mixed since init
    int   clipCount       = 0;     ///< Number of samples that hit the limiter
};

// =============================================================================
// Audio Mixer
// =============================================================================

class WULFNET_API AudioMixer {
public:
    AudioMixer() = default;

    /// Initialize the mixer
    bool Initialize(const AudioMixerConfig& config = {});

    /// Shutdown and release resources
    void Shutdown();

    /// Add a source to the mixer. Returns index or -1 on failure.
    int AddSource(AudioSource* source);

    /// Remove a source by pointer
    bool RemoveSource(AudioSource* source);

    /// Remove a source by index
    bool RemoveSource(int index);

    /// Get a source by index
    AudioSource* GetSource(int index);
    const AudioSource* GetSource(int index) const;

    /// Mix all active sources into the output buffer (interleaved stereo float).
    /// Returns the number of frames written.
    int MixFrame(float* outputStereo, int frameCount);

    /// Mix into an AudioBuffer (creates/resizes as needed)
    int MixFrame(AudioBuffer& output, int frameCount);

    /// Accessors
    const AudioMixerConfig& GetConfig() const { return m_config; }
    const AudioMixerStats& GetStats() const { return m_stats; }

    void SetMasterGain(float gain) { m_config.masterGain = gain; }
    float GetMasterGain() const { return m_config.masterGain; }

    int GetSourceCount() const { return static_cast<int>(m_sources.size()); }

    /// Remove all sources
    void ClearSources();

private:
    /// Soft-clip limiter (tanh-based)
    float SoftClip(float sample) const;

    AudioMixerConfig         m_config;
    AudioMixerStats          m_stats;
    std::vector<AudioSource*> m_sources;
    std::vector<float>       m_tempBuffer; ///< Scratch buffer for per-source output
    bool                     m_initialized = false;
};

} // namespace WulfNet
