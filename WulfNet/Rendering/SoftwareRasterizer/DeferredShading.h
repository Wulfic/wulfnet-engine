// =============================================================================
// WulfNet Engine - Deferred Shading Pass
// =============================================================================
// Screen-space deferred lighting: directional light, point lights,
// hemisphere ambient, distance fog, fresnel reflections, specular.
// Ported from Differed() in BG-C-Software-Renderer/MainEngine.cpp.
// =============================================================================

#pragma once

#include "WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h"
#include "WulfNet/Rendering/SoftwareRasterizer/GBuffer.h"
#include <vector>

namespace WulfNet {

struct DeferredShadingConfig {
    // Directional light
    SoftDirectionalLight sunLight;

    // Ambient
    SoftVec3 ambientSkyColor = {0.3f, 0.4f, 0.6f};
    SoftVec3 ambientGroundColor = {0.15f, 0.1f, 0.08f};
    float ambientIntensity = 0.3f;

    // Fog
    float fogStart = 50.0f;
    float fogEnd = 200.0f;
    SoftVec3 fogColor = {0.7f, 0.8f, 0.9f};

    // Reflections / Specular
    float fresnelPower = 3.0f;
    float specularIntensity = 0.5f;

    // Point lights
    std::vector<SoftPointLight> pointLights;
};

class DeferredShading {
public:
    DeferredShading() = default;

    /// Apply deferred shading to a GBuffer
    void Apply(GBuffer& gbuffer, const DeferredShadingConfig& config,
               const SoftCamera& camera);

private:
    SoftVec3 ComputeLighting(const SoftVec3& worldNormal, const SoftVec3& viewDir,
                              float depth, const SoftVec3& albedo,
                              const DeferredShadingConfig& config,
                              const SoftVec3& worldPos) const;
};

} // namespace WulfNet
