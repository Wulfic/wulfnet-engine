// =============================================================================
// WulfNet Engine - Spatial Audio Implementation
// =============================================================================

#include "WulfNet/Audio/Spatial/SpatialAudio.h"
#include <cstring>
#include <algorithm>

namespace WulfNet {

static constexpr float kSpPi = 3.14159265358979323846f;
static constexpr float kSpHalfPi = kSpPi * 0.5f;

// =============================================================================
// AttenuationCurve
// =============================================================================

float AttenuationCurve::Evaluate(float distance) const {
    if (distance <= refDistance) return 1.0f;
    if (distance >= maxDistance) return 0.0f;

    switch (model) {
        case AttenuationModel::None:
            return 1.0f;

        case AttenuationModel::Linear: {
            float t = (distance - refDistance) / (maxDistance - refDistance);
            return 1.0f - t;
        }

        case AttenuationModel::Inverse: {
            return std::pow(refDistance / distance, rolloff);
        }

        case AttenuationModel::Exponential: {
            float norm = (distance - refDistance) / (maxDistance - refDistance);
            return std::pow(1.0f - norm, rolloff);
        }
    }

    return 1.0f;
}

// =============================================================================
// SpatialAudio
// =============================================================================

bool SpatialAudio::Initialize(int sampleRate) {
    if (sampleRate <= 0) return false;

    m_sampleRate = sampleRate;
    m_hrtfParams = HRTFParams();
    m_dopplerConfig = DopplerConfig();
    m_attenuation = AttenuationCurve();
    m_initialized = true;
    return true;
}

void SpatialAudio::Shutdown() {
    m_initialized = false;
}

// =============================================================================
// HRTF
// =============================================================================

HRTFResult SpatialAudio::ComputeHRTF(float srcX, float srcY, float srcZ) const {
    HRTFResult result;

    float azimuth, elevation, distance;
    CartesianToSpherical(srcX, srcY, srcZ, azimuth, elevation, distance);

    result.azimuth = azimuth;
    result.elevation = elevation;

    if (distance < 1e-6f) {
        // Source at listener position — no spatialization
        result.leftGain = 1.0f;
        result.rightGain = 1.0f;
        result.leftDelay = 0;
        result.rightDelay = 0;
        return result;
    }

    // Interaural Time Difference (ITD)
    // Woodworth-Schlosberg model: ITD = (r/c)(sin(θ) + θ)  for |θ| < π/2
    // Simplified: ITD ≈ (r/c) * sin(azimuth)
    float sinAz = std::sin(azimuth);
    float itd = m_hrtfParams.headRadius / m_hrtfParams.speedOfSound * sinAz;

    // Convert ITD to sample delay
    int delaySamples = static_cast<int>(std::abs(itd) * m_sampleRate + 0.5f);

    if (sinAz > 0) {
        // Source is to the right → right ear is closer, left ear has delay
        result.leftDelay = delaySamples;
        result.rightDelay = 0;
    } else {
        // Source is to the left → left ear is closer, right ear has delay
        result.leftDelay = 0;
        result.rightDelay = delaySamples;
    }

    // Interaural Level Difference (ILD) — head shadow effect
    // Simple cosine-based model: ear closer to source gets more signal
    // ILD increases with frequency (we model an average)
    float ild = 0.5f * std::pow(std::abs(sinAz), m_hrtfParams.ildExponent);

    if (sinAz > 0) {
        // Source to the right
        result.rightGain = 1.0f;
        result.leftGain = 1.0f - ild;
    } else if (sinAz < 0) {
        // Source to the left
        result.leftGain = 1.0f;
        result.rightGain = 1.0f - ild;
    } else {
        // Centered
        result.leftGain = 1.0f;
        result.rightGain = 1.0f;
    }

    return result;
}

AudioBuffer SpatialAudio::ApplyHRTF(const AudioBuffer& monoInput, const HRTFResult& hrtf) const {
    AudioBuffer output;

    if (monoInput.GetFormat().channels != 1 || monoInput.GetFrameCount() == 0) {
        return output;
    }

    // Create stereo output
    AudioFormat stereoFmt;
    stereoFmt.sampleRate = monoInput.GetFormat().sampleRate;
    stereoFmt.channels = 2;
    stereoFmt.format = AudioSampleFormat::Float32;

    int frames = monoInput.GetFrameCount();
    output.Initialize(stereoFmt, frames);

    const float* src = monoInput.GetData();
    float* dst = output.GetData();

    int maxDelay = std::max(hrtf.leftDelay, hrtf.rightDelay);

    for (int i = 0; i < frames; ++i) {
        // Left channel — apply delay and gain
        int leftIdx = i - hrtf.leftDelay;
        float leftSample = (leftIdx >= 0 && leftIdx < frames) ? src[leftIdx] : 0.0f;
        dst[i * 2 + 0] = leftSample * hrtf.leftGain;

        // Right channel — apply delay and gain
        int rightIdx = i - hrtf.rightDelay;
        float rightSample = (rightIdx >= 0 && rightIdx < frames) ? src[rightIdx] : 0.0f;
        dst[i * 2 + 1] = rightSample * hrtf.rightGain;
    }

    return output;
}

// =============================================================================
// Ambisonics
// =============================================================================

AmbisonicsBFormat SpatialAudio::EncodeAmbisonics(float azimuth, float elevation, float gain) {
    AmbisonicsBFormat bf;

    float cosElev = std::cos(elevation);
    float sinElev = std::sin(elevation);
    float cosAz   = std::cos(azimuth);
    float sinAz   = std::sin(azimuth);

    // FuMa normalization (traditional first-order)
    // W channel: omnidirectional = 1/√2
    bf.W = gain * 0.707107f; // 1/sqrt(2)

    // X: front-back = cos(azimuth) * cos(elevation)
    bf.X = gain * cosAz * cosElev;

    // Y: left-right = sin(azimuth) * cos(elevation)
    bf.Y = gain * sinAz * cosElev;

    // Z: up-down = sin(elevation)
    bf.Z = gain * sinElev;

    return bf;
}

std::vector<float> SpatialAudio::DecodeAmbisonics(const AmbisonicsBFormat& bf,
                                                    const std::vector<AmbisonicsSpeaker>& speakers) {
    std::vector<float> feeds(speakers.size(), 0.0f);

    if (speakers.empty()) return feeds;

    for (size_t i = 0; i < speakers.size(); ++i) {
        const auto& spk = speakers[i];
        float cosE = std::cos(spk.elevation);
        float sinE = std::sin(spk.elevation);
        float cosA = std::cos(spk.azimuth);
        float sinA = std::sin(spk.azimuth);

        // Simple decode: project B-format onto speaker direction
        // feed = W + X*cos(az)*cos(el) + Y*sin(az)*cos(el) + Z*sin(el)
        float feed = bf.W
                   + bf.X * cosA * cosE
                   + bf.Y * sinA * cosE
                   + bf.Z * sinE;

        feeds[i] = feed * spk.gain;
    }

    return feeds;
}

std::vector<AmbisonicsSpeaker> SpatialAudio::CreateStereoLayout() {
    std::vector<AmbisonicsSpeaker> speakers(2);
    // Left speaker at +30° azimuth
    speakers[0].azimuth = kSpPi / 6.0f;  // +30° = π/6
    speakers[0].elevation = 0.0f;
    speakers[0].gain = 1.0f;
    // Right speaker at -30° azimuth
    speakers[1].azimuth = -kSpPi / 6.0f; // -30° = -π/6
    speakers[1].elevation = 0.0f;
    speakers[1].gain = 1.0f;
    return speakers;
}

std::vector<AmbisonicsSpeaker> SpatialAudio::CreateQuadLayout() {
    std::vector<AmbisonicsSpeaker> speakers(4);
    // Front Left (+45°)
    speakers[0].azimuth = kSpPi / 4.0f;
    speakers[0].elevation = 0.0f;
    speakers[0].gain = 1.0f;
    // Front Right (-45°)
    speakers[1].azimuth = -kSpPi / 4.0f;
    speakers[1].elevation = 0.0f;
    speakers[1].gain = 1.0f;
    // Rear Left (+135°)
    speakers[2].azimuth = 3.0f * kSpPi / 4.0f;
    speakers[2].elevation = 0.0f;
    speakers[2].gain = 1.0f;
    // Rear Right (-135°)
    speakers[3].azimuth = -3.0f * kSpPi / 4.0f;
    speakers[3].elevation = 0.0f;
    speakers[3].gain = 1.0f;
    return speakers;
}

// =============================================================================
// Doppler Effect
// =============================================================================

float SpatialAudio::ComputeDopplerShift(float srcX, float srcY, float srcZ,
                                          float srcVX, float srcVY, float srcVZ,
                                          float lstX, float lstY, float lstZ,
                                          float lstVX, float lstVY, float lstVZ) const {
    // Direction from source to listener
    float dx = lstX - srcX;
    float dy = lstY - srcY;
    float dz = lstZ - srcZ;
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (dist < 1e-6f) return 1.0f; // Same position — no Doppler

    // Unit direction source→listener
    float invDist = 1.0f / dist;
    float ndx = dx * invDist;
    float ndy = dy * invDist;
    float ndz = dz * invDist;

    // Project velocities onto the source→listener axis
    // vSrc positive = source moving toward listener (along source→listener direction)
    float vSrc = srcVX * ndx + srcVY * ndy + srcVZ * ndz;
    // vLst positive = listener moving toward source (OPPOSITE of source→listener direction)
    // So we negate the projection to get "toward source" convention
    float vLst = -(lstVX * ndx + lstVY * ndy + lstVZ * ndz);

    float c = m_dopplerConfig.speedOfSound;

    // Clamp velocities to prevent singularities (source can't exceed speed of sound)
    float maxV = c * 0.9f;
    vSrc = std::max(-maxV, std::min(maxV, vSrc));
    vLst = std::max(-maxV, std::min(maxV, vLst));

    // Doppler formula: f_observed/f_emitted = (c + v_listener) / (c + v_source)
    // Note: positive v means moving toward each other
    // With our convention: vSrc positive = source moving toward listener
    //                      vLst positive = listener moving toward source
    // So: f_ratio = (c + vLst) / (c - vSrc)
    // Wait, standard: f' = f * (c + vL) / (c - vS) where vL toward source, vS toward listener
    // vSrc is already projected along source→listener, so it's "toward listener" = positive toward listener
    float denom = c - vSrc;
    float numer = c + vLst;

    if (std::abs(denom) < 1e-6f) denom = 1e-6f; // Prevent division by zero

    float shift = numer / denom;

    // Clamp to prevent extreme shifts
    shift = std::max(1.0f / m_dopplerConfig.maxShift, std::min(m_dopplerConfig.maxShift, shift));

    return shift;
}

// =============================================================================
// Distance Attenuation
// =============================================================================

float SpatialAudio::ComputeDistanceGain(float distance) const {
    return m_attenuation.Evaluate(distance);
}

// =============================================================================
// Coordinate Transforms
// =============================================================================

void SpatialAudio::WorldToListenerLocal(float srcX, float srcY, float srcZ,
                                          float lstX, float lstY, float lstZ,
                                          float lstFwdX, float lstFwdY, float lstFwdZ,
                                          float lstUpX, float lstUpY, float lstUpZ,
                                          float& localX, float& localY, float& localZ) {
    // Compute right vector = forward × up
    float rightX = lstFwdY * lstUpZ - lstFwdZ * lstUpY;
    float rightY = lstFwdZ * lstUpX - lstFwdX * lstUpZ;
    float rightZ = lstFwdX * lstUpY - lstFwdY * lstUpX;

    // Normalize right vector
    float rLen = std::sqrt(rightX * rightX + rightY * rightY + rightZ * rightZ);
    if (rLen > 1e-6f) {
        rightX /= rLen; rightY /= rLen; rightZ /= rLen;
    }

    // Delta from listener to source (in world space)
    float dx = srcX - lstX;
    float dy = srcY - lstY;
    float dz = srcZ - lstZ;

    // Project onto listener's local axes
    // X = right axis (positive to listener's right)
    localX = dx * rightX + dy * rightY + dz * rightZ;
    // Y = up axis
    localY = dx * lstUpX + dy * lstUpY + dz * lstUpZ;
    // Z = negative forward axis (so -Z = forward, matching OpenGL/audio convention)
    localZ = -(dx * lstFwdX + dy * lstFwdY + dz * lstFwdZ);
}

void SpatialAudio::CartesianToSpherical(float x, float y, float z,
                                          float& azimuth, float& elevation, float& distance) {
    distance = std::sqrt(x * x + y * y + z * z);

    if (distance < 1e-6f) {
        azimuth = 0.0f;
        elevation = 0.0f;
        return;
    }

    // Azimuth: angle in the horizontal plane from forward (+Z or -Z convention)
    // We use: 0 = directly in front (along +Z), positive = right (+X)
    // atan2(x, z) gives angle from +Z axis toward +X
    azimuth = std::atan2(x, -z); // -Z is forward

    // Elevation: angle above the horizontal plane
    float horizontal = std::sqrt(x * x + z * z);
    elevation = std::atan2(y, horizontal);
}

} // namespace WulfNet
