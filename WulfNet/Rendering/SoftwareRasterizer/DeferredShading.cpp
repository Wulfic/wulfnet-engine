// =============================================================================
// WulfNet Engine - Deferred Shading Implementation
// =============================================================================

#include "WulfNet/Rendering/SoftwareRasterizer/DeferredShading.h"
#include <cmath>
#include <algorithm>

namespace WulfNet {

static float Clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

static SoftVec3 ClampVec(const SoftVec3& v) {
    return {Clamp01(v.x), Clamp01(v.y), Clamp01(v.z)};
}

void DeferredShading::Apply(GBuffer& gbuffer, const DeferredShadingConfig& config,
                             const SoftCamera& camera) {
    int width = gbuffer.GetWidth();
    int height = gbuffer.GetHeight();
    float fovScale = 1.0f / std::tan(camera.fov * 0.5f * 3.14159265f / 180.0f);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float depth = gbuffer.GetDepth(x, y);

            // Skip sky pixels (max depth)
            if (depth > 9999.0f) continue;

            // Unpack normal from GBuffer
            SoftColorRGBA8 packedNormal = gbuffer.GetNormal(x, y);
            SoftVec3 normal = {
                (packedNormal.r / 255.0f) * 2.0f - 1.0f,
                (packedNormal.g / 255.0f) * 2.0f - 1.0f,
                (packedNormal.b / 255.0f) * 2.0f - 1.0f
            };
            normal = normal.Normalized();

            // Reconstruct view direction from screen coords
            float ndcX = (2.0f * (x + 0.5f) / float(width) - 1.0f) * camera.aspectRatio;
            float ndcY = 1.0f - 2.0f * (y + 0.5f) / float(height);
            SoftVec3 viewDir = (camera.forward * fovScale + camera.right * ndcX + camera.up * ndcY).Normalized();

            // Approximate world position
            SoftVec3 worldPos = camera.position + viewDir * depth;

            // Get albedo from color buffer
            SoftColorRGBA8 albedoColor = gbuffer.GetColor(x, y);
            SoftVec3 albedo = {albedoColor.r / 255.0f, albedoColor.g / 255.0f, albedoColor.b / 255.0f};

            // Compute lighting
            SoftVec3 lit = ComputeLighting(normal, viewDir, depth, albedo, config, worldPos);

            gbuffer.SetColor(x, y, SoftColorRGBA8::FromFloat(lit.x, lit.y, lit.z));
        }
    }
}

SoftVec3 DeferredShading::ComputeLighting(const SoftVec3& worldNormal, const SoftVec3& viewDir,
                                            float depth, const SoftVec3& albedo,
                                            const DeferredShadingConfig& config,
                                            const SoftVec3& worldPos) const {
    SoftVec3 color = {};

    // --- Hemisphere ambient ---
    float upFactor = worldNormal.y * 0.5f + 0.5f;
    SoftVec3 ambient = SoftVec3::Lerp(config.ambientGroundColor, config.ambientSkyColor, upFactor)
                       * config.ambientIntensity;
    color = color + SoftVec3{albedo.x * ambient.x, albedo.y * ambient.y, albedo.z * ambient.z};

    // --- Directional (sun) light ---
    SoftVec3 lightDir = config.sunLight.direction.Normalized() * -1.0f;
    float NdotL = std::max(0.0f, worldNormal.Dot(lightDir));
    SoftVec3 diffuse = {
        albedo.x * config.sunLight.color.x * NdotL * config.sunLight.intensity,
        albedo.y * config.sunLight.color.y * NdotL * config.sunLight.intensity,
        albedo.z * config.sunLight.color.z * NdotL * config.sunLight.intensity
    };
    color = color + diffuse;

    // Specular (Blinn-Phong)
    SoftVec3 halfVec = (lightDir + viewDir * -1.0f).Normalized();
    float NdotH = std::max(0.0f, worldNormal.Dot(halfVec));
    float specular = std::pow(NdotH, 32.0f) * config.specularIntensity;
    color = color + config.sunLight.color * specular;

    // --- Point lights ---
    for (const auto& light : config.pointLights) {
        SoftVec3 toLight = light.position - worldPos;
        float dist = toLight.Length();

        if (dist > light.range) continue;

        SoftVec3 lightDirP = toLight / dist;
        float NdotLP = std::max(0.0f, worldNormal.Dot(lightDirP));
        float attenuation = 1.0f - Clamp01(dist / light.range);
        attenuation *= attenuation;

        color = color + SoftVec3{
            albedo.x * light.color.x * NdotLP * light.intensity * attenuation,
            albedo.y * light.color.y * NdotLP * light.intensity * attenuation,
            albedo.z * light.color.z * NdotLP * light.intensity * attenuation
        };
    }

    // --- Fresnel reflections ---
    float fresnel = 1.0f - std::max(0.0f, worldNormal.Dot(viewDir * -1.0f));
    fresnel = std::pow(fresnel, config.fresnelPower);
    color = SoftVec3::Lerp(color, config.ambientSkyColor, fresnel * 0.3f);

    // --- Distance fog ---
    float fogFactor = Clamp01((depth - config.fogStart) / (config.fogEnd - config.fogStart));
    color = SoftVec3::Lerp(color, config.fogColor, fogFactor);

    return ClampVec(color);
}

} // namespace WulfNet
