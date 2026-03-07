// =============================================================================
// WulfNet Engine - Global Illumination System
// =============================================================================
// Screen-Space Ambient Occlusion (SSAO) and indirect lighting for the
// deferred rendering pipeline. CPU reference implementation.
//
// Features:
//   - SSAO: hemisphere sampling of the depth buffer to estimate local occlusion
//   - Indirect diffuse: one-bounce approximation from GBuffer color+normal
//   - Light probes: baked spherical harmonic probes for static GI
//   - Temporal accumulation for noise reduction
//
// Integration:
//   Replaces/augments the hemisphere ambient term in DeferredShading.
//   Call Compute() after the GBuffer is filled, then query GetAOBuffer()
//   or sample indirect lighting per-pixel.
// =============================================================================

#pragma once

#include "WulfNet/Rendering/Types/RenderTypes.h"
#include "WulfNet/Rendering/SoftwareRasterizer/GBuffer.h"
#include <vector>
#include <cmath>

namespace WulfNet {

// =============================================================================
// Configuration
// =============================================================================

struct SSAOConfig {
    int   sampleCount    = 16;     ///< Number of hemisphere samples per pixel
    float radius         = 1.5f;   ///< Sampling radius in world units
    float bias           = 0.025f; ///< Depth bias to prevent self-occlusion
    float intensity      = 1.5f;   ///< AO darkening intensity
    float power          = 2.0f;   ///< AO falloff power
    int   blurPasses     = 1;      ///< Number of blur passes for noise reduction
    int   blurKernelSize = 3;      ///< Blur kernel size (must be odd)
};

struct IndirectLightConfig {
    int   sampleCount     = 8;     ///< Samples for indirect bounce
    float bounceRadius    = 3.0f;  ///< How far to sample for indirect light
    float bounceIntensity = 0.4f;  ///< Strength of indirect diffuse bounce
    bool  enabled         = true;  ///< Enable/disable indirect lighting
};

/// Spherical Harmonic light probe (L1 = 4 coefficients per color channel)
struct LightProbe {
    SoftVec3 position;
    float    radius = 10.0f;       ///< Influence radius

    // L0 + L1 SH coefficients (4 per channel, 12 total)
    SoftVec3 shCoeffs[4] = {};     ///< [0]=L0, [1-3]=L1 (x,y,z)

    /// Evaluate SH irradiance for a given normal direction
    SoftVec3 Evaluate(const SoftVec3& normal) const {
        // L0: constant term
        SoftVec3 color = shCoeffs[0] * 0.282095f;
        // L1: directional terms
        color = color + shCoeffs[1] * (0.488603f * normal.y);
        color = color + shCoeffs[2] * (0.488603f * normal.z);
        color = color + shCoeffs[3] * (0.488603f * normal.x);
        return color;
    }
};

struct GlobalIlluminationConfig {
    SSAOConfig          ssao;
    IndirectLightConfig indirect;
    std::vector<LightProbe> probes;

    bool ssaoEnabled     = true;
    bool probesEnabled   = false;
};

// =============================================================================
// Global Illumination System
// =============================================================================

class GlobalIllumination {
public:
    GlobalIllumination() = default;

    /// Initialize with resolution matching the GBuffer
    bool Initialize(int width, int height, const GlobalIlluminationConfig& config = {});

    /// Compute SSAO and indirect lighting from the current GBuffer state
    void Compute(const GBuffer& gbuffer, const SoftCamera& camera);

    /// Get the AO buffer (float per pixel, 0=fully occluded, 1=no occlusion)
    const float* GetAOBuffer() const { return m_aoBuffer.data(); }
    float* GetAOBuffer() { return m_aoBuffer.data(); }

    /// Get the indirect lighting buffer (RGB per pixel)
    const SoftVec3* GetIndirectBuffer() const { return m_indirectBuffer.data(); }

    /// Sample AO at a pixel
    float SampleAO(int x, int y) const;

    /// Sample indirect lighting at a pixel
    SoftVec3 SampleIndirect(int x, int y) const;

    /// Evaluate light probe contribution at a world position with given normal
    SoftVec3 EvaluateProbes(const SoftVec3& worldPos, const SoftVec3& normal) const;

    /// Apply a box blur to the AO buffer for noise reduction
    void BlurAOBuffer();

    /// Accessors
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    const GlobalIlluminationConfig& GetConfig() const { return m_config; }

    /// Add a light probe
    void AddProbe(const LightProbe& probe);
    int GetProbeCount() const { return static_cast<int>(m_config.probes.size()); }

private:
    /// Compute SSAO for a single pixel
    float ComputeSSAOPixel(const GBuffer& gbuffer, const SoftCamera& camera,
                           int x, int y) const;

    /// Compute indirect lighting for a single pixel
    SoftVec3 ComputeIndirectPixel(const GBuffer& gbuffer, const SoftCamera& camera,
                                   int x, int y) const;

    /// Generate a pseudo-random sample direction in a hemisphere
    SoftVec3 HemisphereSample(int sampleIndex, int totalSamples,
                              const SoftVec3& normal) const;

    /// Reconstruct world position from screen coords + depth
    SoftVec3 ReconstructWorldPos(int x, int y, float depth,
                                 const SoftCamera& camera) const;

    /// Unpack normal from GBuffer
    SoftVec3 UnpackNormal(SoftColorRGBA8 packed) const;

    int m_width = 0;
    int m_height = 0;

    GlobalIlluminationConfig m_config;

    std::vector<float> m_aoBuffer;         ///< AO per pixel [0..1]
    std::vector<SoftVec3> m_indirectBuffer; ///< Indirect lighting color per pixel
    std::vector<float> m_aoTempBuffer;     ///< Temporary buffer for blur
};

} // namespace WulfNet
