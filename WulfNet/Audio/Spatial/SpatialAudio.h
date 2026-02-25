// =============================================================================
// WulfNet Engine - Spatial Audio System
// =============================================================================
// Provides 3D audio spatialization features:
//   - Head-Related Transfer Function (HRTF) simplified model
//     using interaural time difference (ITD) and interaural level
//     difference (ILD) for binaural rendering.
//   - First-order Ambisonics (FOA) B-format encoding/decoding
//     for surround-sound and VR audio.
//   - Doppler effect simulation based on relative velocity.
//   - Distance attenuation curves (linear, inverse, exponential).
//
// All processing is purely mathematical (no audio hardware dependency).
// =============================================================================

#pragma once

#include "WulfNet/Audio/Core/AudioTypes.h"
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace WulfNet {

// =============================================================================
// Distance Attenuation Models
// =============================================================================

enum class AttenuationModel : uint8_t {
    None,
    Linear,
    Inverse,
    Exponential
};

struct AttenuationCurve {
    AttenuationModel model  = AttenuationModel::Inverse;
    float refDistance        = 1.0f;   ///< Reference distance (full volume)
    float maxDistance        = 100.0f; ///< Beyond this distance, signal is silent
    float rolloff           = 1.0f;   ///< Rolloff exponent (for Inverse / Exponential)

    /// Compute gain for a given distance
    float Evaluate(float distance) const;
};

// =============================================================================
// HRTF (Head-Related Transfer Function) - Simplified Model
// =============================================================================

/// Simplified HRTF parameters for binaural spatialization.
/// Uses interaural time difference (ITD) and interaural level difference (ILD).
struct HRTFParams {
    float headRadius      = 0.0875f;  ///< Average human head radius (~8.75 cm) in meters
    float speedOfSound    = 343.0f;   ///< Speed of sound in m/s
    float maxITD          = 0.0f;     ///< Maximum ITD (auto-computed from headRadius)
    float ildExponent     = 1.0f;     ///< ILD frequency shadowing exponent

    HRTFParams() {
        maxITD = headRadius / speedOfSound; // ~0.000255 sec = 0.255 ms
    }
};

/// Result of HRTF computation for one source
struct HRTFResult {
    float leftGain   = 1.0f;   ///< Left ear gain [0..1]
    float rightGain  = 1.0f;   ///< Right ear gain [0..1]
    int   leftDelay  = 0;      ///< Left ear delay in samples
    int   rightDelay = 0;      ///< Right ear delay in samples
    float azimuth    = 0.0f;   ///< Source azimuth relative to listener (radians)
    float elevation  = 0.0f;   ///< Source elevation relative to listener (radians)
};

// =============================================================================
// Ambisonics (First Order - B-Format)
// =============================================================================

/// First-order Ambisonics B-format channels: W (omnidirectional), X, Y, Z
struct AmbisonicsBFormat {
    float W = 0.0f;  ///< Pressure (omnidirectional)
    float X = 0.0f;  ///< Front-back (cos θ cos φ)
    float Y = 0.0f;  ///< Left-right (sin θ cos φ)
    float Z = 0.0f;  ///< Up-down (sin φ)
};

/// Speaker layout for decoding Ambisonics to speaker feeds
struct AmbisonicsSpeaker {
    float azimuth   = 0.0f;  ///< Horizontal angle (radians, 0 = front, +π/2 = left)
    float elevation = 0.0f;  ///< Vertical angle (radians, 0 = horizontal)
    float gain      = 1.0f;  ///< Per-speaker gain
};

// =============================================================================
// Doppler Effect
// =============================================================================

struct DopplerConfig {
    float speedOfSound = 343.0f;  ///< Speed of sound in m/s
    float maxShift     = 4.0f;    ///< Maximum pitch shift factor
    float smoothing    = 0.5f;    ///< Smoothing factor [0..1] to reduce artifacts
};

// =============================================================================
// SpatialAudio System
// =============================================================================

class SpatialAudio {
public:
    SpatialAudio() = default;

    /// Initialize with default settings
    bool Initialize(int sampleRate = 44100);

    /// Shutdown and release resources
    void Shutdown();

    // =========================================================================
    // HRTF
    // =========================================================================

    /// Set HRTF parameters
    void SetHRTFParams(const HRTFParams& params) { m_hrtfParams = params; }
    const HRTFParams& GetHRTFParams() const { return m_hrtfParams; }

    /// Compute HRTF gains and delays for a source position relative to listener.
    /// Positions are in listener-local space (listener at origin, facing -Z, Y up, X right).
    HRTFResult ComputeHRTF(float srcX, float srcY, float srcZ) const;

    /// Apply HRTF to a mono audio buffer, producing a stereo binaural output.
    /// The output buffer will have twice the sample count (interleaved L/R).
    AudioBuffer ApplyHRTF(const AudioBuffer& monoInput, const HRTFResult& hrtf) const;

    // =========================================================================
    // Ambisonics
    // =========================================================================

    /// Encode a mono source at the given direction into B-format
    static AmbisonicsBFormat EncodeAmbisonics(float azimuth, float elevation, float gain = 1.0f);

    /// Decode B-format to a set of virtual speaker feeds
    static std::vector<float> DecodeAmbisonics(const AmbisonicsBFormat& bformat,
                                                const std::vector<AmbisonicsSpeaker>& speakers);

    /// Convenience: create a standard stereo speaker layout (L at +30°, R at -30°)
    static std::vector<AmbisonicsSpeaker> CreateStereoLayout();

    /// Convenience: create a standard quad speaker layout
    static std::vector<AmbisonicsSpeaker> CreateQuadLayout();

    // =========================================================================
    // Doppler Effect
    // =========================================================================

    /// Set Doppler configuration
    void SetDopplerConfig(const DopplerConfig& config) { m_dopplerConfig = config; }
    const DopplerConfig& GetDopplerConfig() const { return m_dopplerConfig; }

    /// Compute Doppler pitch shift given source and listener velocities and positions.
    /// Returns pitch multiplier (>1 = higher pitch, <1 = lower pitch).
    float ComputeDopplerShift(float srcX, float srcY, float srcZ,
                               float srcVX, float srcVY, float srcVZ,
                               float lstX, float lstY, float lstZ,
                               float lstVX, float lstVY, float lstVZ) const;

    // =========================================================================
    // Distance Attenuation
    // =========================================================================

    /// Set the attenuation curve
    void SetAttenuationCurve(const AttenuationCurve& curve) { m_attenuation = curve; }
    const AttenuationCurve& GetAttenuationCurve() const { return m_attenuation; }

    /// Compute distance attenuation for a given distance
    float ComputeDistanceGain(float distance) const;

    // =========================================================================
    // Utility
    // =========================================================================

    /// Transform world-space source position into listener-local coordinates.
    /// Listener defined by position, forward (-Z in local), and up (Y in local).
    static void WorldToListenerLocal(float srcX, float srcY, float srcZ,
                                      float lstX, float lstY, float lstZ,
                                      float lstFwdX, float lstFwdY, float lstFwdZ,
                                      float lstUpX, float lstUpY, float lstUpZ,
                                      float& localX, float& localY, float& localZ);

    /// Compute azimuth and elevation from local-space position
    static void CartesianToSpherical(float x, float y, float z,
                                      float& azimuth, float& elevation, float& distance);

    // =========================================================================
    // State
    // =========================================================================

    bool IsInitialized() const { return m_initialized; }
    int GetSampleRate() const { return m_sampleRate; }

private:
    bool              m_initialized  = false;
    int               m_sampleRate   = 44100;
    HRTFParams        m_hrtfParams;
    DopplerConfig     m_dopplerConfig;
    AttenuationCurve  m_attenuation;
};

} // namespace WulfNet
