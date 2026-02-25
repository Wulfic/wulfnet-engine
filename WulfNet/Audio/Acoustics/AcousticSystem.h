// =============================================================================
// WulfNet Engine - Acoustic Simulation System
// =============================================================================
// CPU-based acoustic ray tracer that computes:
//   - Occlusion & obstruction (line-of-sight attenuation)
//   - Impulse response generation (ray-traced reverb)
//   - Late reverb estimation (statistical Sabine/Eyring model)
//   - Material-based absorption coefficients
//
// The system uses a pluggable ray-casting interface so it can work with
// Jolt's BroadPhaseQuery or a standalone geometry representation.
//
// This is a CPU reference implementation for testing and offline processing.
// Real-time integration would use GPU-accelerated ray tracing.
// =============================================================================

#pragma once

#include "WulfNet/Audio/Core/AudioTypes.h"
#include <vector>
#include <cmath>
#include <functional>
#include <cstdint>

namespace WulfNet {

// =============================================================================
// Acoustic Material
// =============================================================================

/// Surface absorption coefficients per octave band (125 Hz - 4 kHz)
struct AcousticMaterial {
    static constexpr int kNumBands = 6; // 125, 250, 500, 1k, 2k, 4k Hz

    float absorption[kNumBands] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float scattering = 0.5f;  ///< Diffuse scattering coefficient [0..1]
    float transmission = 0.0f; ///< Sound transmission through surface [0..1]

    /// Named presets
    static AcousticMaterial Concrete();
    static AcousticMaterial Wood();
    static AcousticMaterial Glass();
    static AcousticMaterial Carpet();
    static AcousticMaterial Curtain();
    static AcousticMaterial Metal();
};

// =============================================================================
// Ray-Cast Interface
// =============================================================================

/// Result of a single ray cast
struct AcousticRayHit {
    bool  hit       = false;
    float distance  = 0.0f;
    float normalX   = 0.0f, normalY = 0.0f, normalZ = 0.0f;
    int   materialId = 0;   ///< Index into the material table
};

/// Callback for casting a ray through the scene.
/// Parameters: originXYZ, directionXYZ, maxDistance → AcousticRayHit
using AcousticRayCastFn = std::function<AcousticRayHit(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float maxDist)>;

// =============================================================================
// Impulse Response
// =============================================================================

/// Represents a discrete reflection arrival (tap in an impulse response)
struct ReflectionTap {
    float time      = 0.0f;   ///< Arrival time (seconds)
    float energy    = 0.0f;   ///< Energy at arrival [0..1]
    float direction[3] = {};  ///< Arrival direction (unit vector)
    int   bounces   = 0;      ///< Number of wall bounces
};

/// Full impulse response for a source→listener pair
struct ImpulseResponse {
    std::vector<ReflectionTap> taps;
    float directEnergy    = 0.0f;  ///< Direct path energy
    float directTime      = 0.0f;  ///< Direct path arrival time (seconds)
    float earlyEnergy     = 0.0f;  ///< Sum of early reflections (< 80ms)
    float lateEnergy      = 0.0f;  ///< Estimated late reverb energy
    float rt60            = 0.0f;  ///< Estimated RT60 (reverberation time)
    bool  directOccluded  = false; ///< Direct path blocked by geometry

    /// Convert impulse response to a mono AudioBuffer
    AudioBuffer ToAudioBuffer(int sampleRate = 44100, float durationSec = 1.0f) const;
};

// =============================================================================
// Room Estimation
// =============================================================================

struct RoomEstimate {
    float volume      = 0.0f;   ///< Estimated room volume (m³)
    float surfaceArea = 0.0f;   ///< Estimated total surface area (m²)
    float meanFreePath = 0.0f;  ///< Mean free path (m) = 4V/S
    float rt60        = 0.0f;   ///< Sabine RT60 estimate (seconds)
    float avgAbsorption = 0.0f; ///< Average absorption coefficient
};

// =============================================================================
// Configuration
// =============================================================================

struct AcousticConfig {
    int   maxBounces        = 6;       ///< Maximum ray bounces for reflections
    int   numRays           = 128;     ///< Number of rays to cast
    float maxDistance        = 100.0f;  ///< Maximum propagation distance (m)
    float speedOfSound      = 343.0f;  ///< Speed of sound (m/s)
    float airAbsorption     = 0.001f;  ///< Absorption per meter of air
    float earlyLateBoundary = 0.08f;   ///< Early / late reflection boundary (sec)
    float energyThreshold   = 0.001f;  ///< Min energy to continue tracing a ray
    int   roomProbeRays     = 64;      ///< Rays for room volume estimation
};

// =============================================================================
// Acoustic System
// =============================================================================

class AcousticSystem {
public:
    AcousticSystem() = default;

    /// Initialize with configuration and material set
    bool Initialize(const AcousticConfig& config = {});

    /// Shutdown
    void Shutdown();

    /// Set the ray-cast function (must be set before tracing)
    void SetRayCastFunction(AcousticRayCastFn fn);

    /// Add an acoustic material. Returns the material ID.
    int AddMaterial(const AcousticMaterial& material);

    /// Get material by ID
    const AcousticMaterial& GetMaterial(int id) const;
    int GetMaterialCount() const { return static_cast<int>(m_materials.size()); }

    // =========================================================================
    // Occlusion / Obstruction
    // =========================================================================

    /// Compute occlusion factor between source and listener [0=fully blocked, 1=clear].
    /// Casts a single ray from source to listener.
    float ComputeOcclusion(float srcX, float srcY, float srcZ,
                           float lstX, float lstY, float lstZ) const;

    /// Compute obstruction factor — how much geometry blocks independent
    /// of full occlusion (partial covers). Uses multiple sample rays.
    float ComputeObstruction(float srcX, float srcY, float srcZ,
                              float lstX, float lstY, float lstZ,
                              int numSampleRays = 8) const;

    // =========================================================================
    // Impulse Response
    // =========================================================================

    /// Trace acoustic rays from source position and compute impulse response
    /// at listener position.
    ImpulseResponse TraceImpulseResponse(float srcX, float srcY, float srcZ,
                                          float lstX, float lstY, float lstZ) const;

    // =========================================================================
    // Room Estimation
    // =========================================================================

    /// Estimate room volume and surface area by casting probe rays from a point.
    RoomEstimate EstimateRoom(float posX, float posY, float posZ) const;

    /// Compute Sabine RT60 from room parameters
    static float ComputeRT60_Sabine(float volume, float surfaceArea, float avgAbsorption);

    /// Compute Eyring RT60 from room parameters
    static float ComputeRT60_Eyring(float volume, float surfaceArea, float avgAbsorption);

    // =========================================================================
    // Distance Attenuation
    // =========================================================================

    /// Inverse-square-law attenuation with configurable rolloff
    static float DistanceAttenuation(float distance, float refDistance = 1.0f,
                                     float maxDistance = 100.0f, float rolloff = 1.0f);

    /// Air absorption (frequency-dependent high-frequency loss)
    float AirAbsorption(float distance) const;

    // =========================================================================
    // Accessors
    // =========================================================================

    const AcousticConfig& GetConfig() const { return m_config; }
    bool IsInitialized() const { return m_initialized; }

private:
    /// Generate a uniform direction on the unit sphere (deterministic)
    void GenerateRayDirection(int index, int total, float& dx, float& dy, float& dz) const;

    /// Reflect direction about normal
    static void Reflect(float dx, float dy, float dz,
                        float nx, float ny, float nz,
                        float& rx, float& ry, float& rz);

    AcousticConfig              m_config;
    std::vector<AcousticMaterial> m_materials;
    AcousticRayCastFn           m_rayCast;
    bool                        m_initialized = false;

    static const AcousticMaterial kDefaultMaterial;
};

} // namespace WulfNet
