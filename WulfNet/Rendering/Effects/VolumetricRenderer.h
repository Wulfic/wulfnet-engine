// =============================================================================
// WulfNet Engine - Volumetric Renderer
// =============================================================================
// CPU ray-marching for rendering gaseous/fluid volumes into the scene.
// Integrates with both the GaseousSystem (for density/temperature fields)
// and the deferred rendering GBuffer pipeline.
//
// Features:
//   - Ray-marched density integration (Beer-Lambert absorption)
//   - Emission from hot gas (fire, combustion)
//   - In-scattering from directional and point lights
//   - Compositing volumetric results over the scene color buffer
//
// Integration:
//   After DeferredShading::Apply(), call VolumetricRenderer::Render() to
//   composite volumetric volumes over the final image.
// =============================================================================

#pragma once

#include "WulfNet/Rendering/Types/RenderTypes.h"
#include "WulfNet/Rendering/SoftwareRasterizer/GBuffer.h"
#include <vector>
#include <cmath>
#include <functional>

namespace WulfNet {

// Forward declaration — only needed for the callback interface
// We don't include GaseousSystem.h to keep rendering decoupled from physics

// =============================================================================
// Configuration
// =============================================================================

struct VolumeRegion {
    SoftVec3 boundsMin = {0, 0, 0};     ///< World-space AABB minimum
    SoftVec3 boundsMax = {10, 10, 10};   ///< World-space AABB maximum
};

/// Color ramp keyframe for mapping temperature to emission color
struct EmissionKeyframe {
    float temperature;     ///< Temperature threshold
    SoftVec3 color;        ///< RGB emission color
    float intensity;       ///< Emission strength
};

struct VolumetricConfig {
    int   maxSteps         = 64;       ///< Max ray-march steps per pixel
    float stepSize         = 0.2f;     ///< Step size in world units
    float absorptionCoeff  = 1.0f;     ///< Beer-Lambert coefficient (sigma_a)
    float scatteringCoeff  = 0.5f;     ///< Scattering coefficient (sigma_s)
    float densityMultiplier = 2.0f;    ///< Scale density for visual intensity
    float emissionIntensity = 3.0f;    ///< Scale for fire/hot emission

    /// Temperature-to-color mapping for fire
    std::vector<EmissionKeyframe> emissionRamp = {
        {300.0f,  {1.0f, 0.2f, 0.0f}, 0.5f},   // Low fire: dark orange
        {600.0f,  {1.0f, 0.6f, 0.1f}, 1.0f},   // Mid fire: orange-yellow
        {1000.0f, {1.0f, 1.0f, 0.8f}, 2.0f},   // Hot fire: white-yellow
    };

    /// Light direction for in-scattering (usually sun direction)
    SoftVec3 lightDirection = {0.0f, -1.0f, 0.5f};
    SoftVec3 lightColor = {1.0f, 0.95f, 0.9f};
    float lightIntensity = 1.0f;

    /// Henyey-Greenstein phase function asymmetry (0=isotropic, >0=forward scatter)
    float phaseG = 0.3f;
};

// =============================================================================
// Density/Temperature sampling callback
// =============================================================================

/// Abstracted volume field sampler — allows rendering volumes from any source
/// without coupling to GaseousSystem
struct VolumeSampler {
    /// Sample density at world position. Returns density [0..∞)
    std::function<float(float wx, float wy, float wz)> sampleDensity;

    /// Sample temperature at world position. Returns temperature in Kelvin
    std::function<float(float wx, float wy, float wz)> sampleTemperature;

    /// The AABB of the volume
    VolumeRegion region;
};

// =============================================================================
// Per-pixel volumetric result
// =============================================================================

struct VolumetricSample {
    SoftVec3 color = {};          ///< Accumulated color (emission + in-scattering)
    float    transmittance = 1.0f; ///< Remaining transparency [0..1]
};

// =============================================================================
// Volumetric Renderer
// =============================================================================

class VolumetricRenderer {
public:
    VolumetricRenderer() = default;

    /// Initialize with framebuffer dimensions
    bool Initialize(int width, int height, const VolumetricConfig& config = {});

    /// Add a volume to render
    void AddVolume(const VolumeSampler& sampler);

    /// Clear all registered volumes
    void ClearVolumes();

    /// Render all volumes and composite onto the GBuffer color buffer
    void Render(GBuffer& gbuffer, const SoftCamera& camera);

    /// Get the volumetric accumulation buffer (for debug)
    const VolumetricSample* GetVolumetricBuffer() const { return m_volumeBuffer.data(); }

    /// Ray-march a single ray through a volume
    VolumetricSample MarchRay(const SoftVec3& rayOrigin, const SoftVec3& rayDir,
                               float maxDist, const VolumeSampler& sampler) const;

    /// Compute ray-AABB intersection (returns tNear, tFar)
    static bool RayAABBIntersect(const SoftVec3& origin, const SoftVec3& invDir,
                                  const SoftVec3& boxMin, const SoftVec3& boxMax,
                                  float& tNear, float& tFar);

    /// Evaluate emission color from temperature
    SoftVec3 EvaluateEmission(float temperature) const;

    /// Henyey-Greenstein phase function
    static float PhaseHG(float cosTheta, float g);

    /// Accessors
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    int GetVolumeCount() const { return static_cast<int>(m_volumes.size()); }
    const VolumetricConfig& GetConfig() const { return m_config; }

private:
    /// Reconstruct a ray direction from screen pixel
    SoftVec3 PixelToRayDir(int x, int y, const SoftCamera& camera) const;

    int m_width = 0;
    int m_height = 0;
    VolumetricConfig m_config;

    std::vector<VolumeSampler> m_volumes;
    std::vector<VolumetricSample> m_volumeBuffer;
};

} // namespace WulfNet
