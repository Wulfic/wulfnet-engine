// =============================================================================
// WulfNet Engine - Audio Types
// =============================================================================
// Core audio data types: AudioBuffer (PCM container), AudioSource (playback
// instance), and AudioFormat (sample rate/channels/bit depth).
//
// All processing is performed on CPU with float32 samples internally.
// External audio hardware integration is out of scope; these types enable
// offline processing, mixing, and spatial audio computation in tests and
// pipeline code.
// =============================================================================

#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <cmath>
#include <algorithm>

namespace WulfNet {

// =============================================================================
// Audio Format
// =============================================================================

enum class AudioSampleFormat {
    Int16,      ///< 16-bit signed integer PCM
    Float32     ///< 32-bit IEEE float
};

struct AudioFormat {
    int   sampleRate  = 44100;  ///< Samples per second (Hz)
    int   channels    = 1;      ///< Number of channels (1=mono, 2=stereo)
    AudioSampleFormat format = AudioSampleFormat::Float32;

    /// Bytes per single sample (one channel)
    int BytesPerSample() const {
        return (format == AudioSampleFormat::Float32) ? 4 : 2;
    }

    /// Bytes per frame (all channels for one sample)
    int BytesPerFrame() const { return BytesPerSample() * channels; }

    bool operator==(const AudioFormat& o) const {
        return sampleRate == o.sampleRate &&
               channels   == o.channels   &&
               format     == o.format;
    }
    bool operator!=(const AudioFormat& o) const { return !(*this == o); }
};

// =============================================================================
// Audio Buffer — owns PCM sample data in float32
// =============================================================================

class AudioBuffer {
public:
    AudioBuffer() = default;

    /// Allocate a buffer with the given format and frame count
    bool Initialize(const AudioFormat& format, int frameCount);

    /// Create a mono buffer at a given sample rate with sine wave data (for testing)
    static AudioBuffer GenerateSine(float frequencyHz, float durationSec,
                                     int sampleRate = 44100, float amplitude = 1.0f);

    /// Create a mono buffer filled with white noise
    static AudioBuffer GenerateNoise(float durationSec, int sampleRate = 44100,
                                      float amplitude = 1.0f);

    /// Create a mono buffer of silence
    static AudioBuffer GenerateSilence(float durationSec, int sampleRate = 44100);

    /// Create from interleaved float samples
    bool LoadFromFloat(const float* data, int frameCount, const AudioFormat& fmt);

    /// Access raw sample data (interleaved: L R L R … for stereo)
    float*       GetData()       { return m_samples.data(); }
    const float* GetData() const { return m_samples.data(); }

    /// Number of sample frames (= total samples / channels)
    int GetFrameCount() const { return m_frameCount; }

    /// Total sample count (frames × channels)
    int GetSampleCount() const { return static_cast<int>(m_samples.size()); }

    /// Duration in seconds
    float GetDuration() const {
        if (m_format.sampleRate <= 0) return 0.0f;
        return static_cast<float>(m_frameCount) / static_cast<float>(m_format.sampleRate);
    }

    /// Format accessors
    const AudioFormat& GetFormat() const { return m_format; }
    int GetSampleRate() const { return m_format.sampleRate; }
    int GetChannels() const { return m_format.channels; }

    /// Compute RMS amplitude
    float ComputeRMS() const;

    /// Compute peak amplitude
    float ComputePeak() const;

    /// Clear all samples to zero
    void Clear();

    /// Resize the buffer (preserves existing data where possible)
    void Resize(int newFrameCount);

    /// Check if buffer has any data
    bool IsValid() const { return m_frameCount > 0 && !m_samples.empty(); }

    /// Mix another buffer into this one (additive, with gain)
    void MixIn(const AudioBuffer& other, float gain = 1.0f, int destOffset = 0);

    /// Apply gain to all samples
    void ApplyGain(float gain);

    /// Normalize to a target peak amplitude
    void Normalize(float targetPeak = 1.0f);

private:
    AudioFormat        m_format;
    int                m_frameCount = 0;
    std::vector<float> m_samples;
};

// =============================================================================
// Audio Source — a playback instance referencing an AudioBuffer
// =============================================================================

struct AudioSourceConfig {
    float gain      = 1.0f;    ///< Linear gain [0..∞)
    float pan       = 0.0f;    ///< Stereo pan [-1 (left) .. +1 (right)]
    bool  loop      = false;   ///< Loop playback
    float fadeInSec  = 0.0f;   ///< Fade-in duration (seconds)
    float fadeOutSec = 0.0f;   ///< Fade-out duration (seconds)
    float pitch     = 1.0f;    ///< Playback pitch multiplier (1.0 = normal)
};

class AudioSource {
public:
    AudioSource() = default;

    /// Bind a buffer (does not take ownership)
    void SetBuffer(const AudioBuffer* buffer);
    const AudioBuffer* GetBuffer() const { return m_buffer; }

    /// Configuration
    void SetConfig(const AudioSourceConfig& cfg) { m_config = cfg; }
    const AudioSourceConfig& GetConfig() const { return m_config; }

    /// Gain/pan shortcuts
    void SetGain(float g) { m_config.gain = g; }
    float GetGain() const { return m_config.gain; }
    void SetPan(float p) { m_config.pan = std::max(-1.0f, std::min(1.0f, p)); }
    float GetPan() const { return m_config.pan; }

    /// Playback control
    void Play()  { m_playing = true; m_playhead = 0; }
    void Stop()  { m_playing = false; m_playhead = 0; }
    void Pause() { m_playing = false; }
    void Resume() { m_playing = true; }

    bool IsPlaying() const { return m_playing; }
    int  GetPlayhead() const { return m_playhead; }
    void SetPlayhead(int frame) { m_playhead = frame; }

    /// Read the next `frameCount` frames into a stereo output buffer.
    /// Returns the number of frames actually written.
    int ReadFrames(float* outStereo, int frameCount);

    /// 3D position for spatial audio
    struct Position3D {
        float x = 0.0f, y = 0.0f, z = 0.0f;
    };
    void SetPosition(const Position3D& pos) { m_position = pos; }
    const Position3D& GetPosition() const { return m_position; }

    void SetVelocity(const Position3D& vel) { m_velocity = vel; }
    const Position3D& GetVelocity() const { return m_velocity; }

private:
    const AudioBuffer*  m_buffer  = nullptr;
    AudioSourceConfig   m_config;
    bool                m_playing = false;
    int                 m_playhead = 0;    ///< Current frame position
    Position3D          m_position;
    Position3D          m_velocity;
};

// =============================================================================
// Audio Listener — represents the listener position for spatial audio
// =============================================================================

struct AudioListener {
    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
    float fwdX = 0.0f, fwdY = 0.0f, fwdZ = 1.0f;
    float upX  = 0.0f, upY  = 1.0f, upZ  = 0.0f;
    float velX = 0.0f, velY = 0.0f, velZ = 0.0f;
};

} // namespace WulfNet
