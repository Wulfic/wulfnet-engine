// =============================================================================
// WulfNet Engine - Volumetric Renderer Implementation
// =============================================================================

#include "WulfNet/Rendering/Effects/VolumetricRenderer.h"
#include <algorithm>
#include <limits>

#ifdef WULFNET_HAS_OPENMP
#include <omp.h>
#endif

namespace WulfNet {

// =============================================================================
// Helpers
// =============================================================================

static float VolClamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

static constexpr float kVolPi = 3.14159265358979323846f;

// =============================================================================
// VolumetricRenderer
// =============================================================================

bool VolumetricRenderer::Initialize(int width, int height, const VolumetricConfig& config) {
    if (width <= 0 || height <= 0) return false;

    m_width = width;
    m_height = height;
    m_config = config;

    m_volumeBuffer.resize(width * height, VolumetricSample{});

    return true;
}

void VolumetricRenderer::AddVolume(const VolumeSampler& sampler) {
    m_volumes.push_back(sampler);
}

void VolumetricRenderer::ClearVolumes() {
    m_volumes.clear();
}

SoftVec3 VolumetricRenderer::PixelToRayDir(int x, int y, const SoftCamera& camera) const {
    float fovScale = 1.0f / std::tan(camera.fov * 0.5f * kVolPi / 180.0f);
    float ndcX = (2.0f * (x + 0.5f) / static_cast<float>(m_width) - 1.0f) * camera.aspectRatio;
    float ndcY = 1.0f - 2.0f * (y + 0.5f) / static_cast<float>(m_height);
    return (camera.forward * fovScale + camera.right * ndcX + camera.up * ndcY).Normalized();
}

bool VolumetricRenderer::RayAABBIntersect(const SoftVec3& origin, const SoftVec3& invDir,
                                            const SoftVec3& boxMin, const SoftVec3& boxMax,
                                            float& tNear, float& tFar) {
    float t1 = (boxMin.x - origin.x) * invDir.x;
    float t2 = (boxMax.x - origin.x) * invDir.x;
    float t3 = (boxMin.y - origin.y) * invDir.y;
    float t4 = (boxMax.y - origin.y) * invDir.y;
    float t5 = (boxMin.z - origin.z) * invDir.z;
    float t6 = (boxMax.z - origin.z) * invDir.z;

    tNear = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
    tFar  = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

    // If tFar < 0 the box is behind us
    if (tFar < 0.0f) return false;
    // If tNear > tFar, no intersection
    if (tNear > tFar) return false;

    // Clamp tNear to 0 if we're inside the box
    if (tNear < 0.0f) tNear = 0.0f;

    return true;
}

SoftVec3 VolumetricRenderer::EvaluateEmission(float temperature) const {
    const auto& ramp = m_config.emissionRamp;
    if (ramp.empty() || temperature < ramp[0].temperature) return {};

    // Find the two keyframes we're between
    for (size_t i = 0; i + 1 < ramp.size(); ++i) {
        if (temperature >= ramp[i].temperature && temperature < ramp[i + 1].temperature) {
            float t = (temperature - ramp[i].temperature) /
                      (ramp[i + 1].temperature - ramp[i].temperature);
            SoftVec3 color = SoftVec3::Lerp(ramp[i].color, ramp[i + 1].color, t);
            float intensity = ramp[i].intensity + (ramp[i + 1].intensity - ramp[i].intensity) * t;
            return color * intensity * m_config.emissionIntensity;
        }
    }

    // Above the last keyframe — use the last entry
    const auto& last = ramp.back();
    return last.color * last.intensity * m_config.emissionIntensity;
}

float VolumetricRenderer::PhaseHG(float cosTheta, float g) {
    // Henyey-Greenstein phase function
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    if (denom < 1e-6f) denom = 1e-6f;
    return (1.0f / (4.0f * kVolPi)) * (1.0f - g2) / (denom * std::sqrt(denom));
}

VolumetricSample VolumetricRenderer::MarchRay(const SoftVec3& rayOrigin, const SoftVec3& rayDir,
                                                float maxDist, const VolumeSampler& sampler) const {
    VolumetricSample result;
    result.transmittance = 1.0f;
    result.color = {};

    if (!sampler.sampleDensity) return result;

    // Intersect ray with volume AABB
    SoftVec3 invDir = {
        std::abs(rayDir.x) > 1e-8f ? 1.0f / rayDir.x : 1e8f,
        std::abs(rayDir.y) > 1e-8f ? 1.0f / rayDir.y : 1e8f,
        std::abs(rayDir.z) > 1e-8f ? 1.0f / rayDir.z : 1e8f
    };

    float tNear, tFar;
    if (!RayAABBIntersect(rayOrigin, invDir, sampler.region.boundsMin,
                           sampler.region.boundsMax, tNear, tFar)) {
        return result;
    }

    // Clamp to scene depth
    tFar = std::min(tFar, maxDist);
    if (tNear >= tFar) return result;

    // Light direction for in-scattering
    SoftVec3 lightDir = m_config.lightDirection.Normalized() * -1.0f;
    float cosTheta = rayDir.Dot(lightDir);
    float phase = PhaseHG(cosTheta, m_config.phaseG);

    float stepSize = m_config.stepSize;
    float t = tNear;
    int steps = 0;

    while (t < tFar && steps < m_config.maxSteps && result.transmittance > 0.001f) {
        SoftVec3 pos = rayOrigin + rayDir * t;

        // Sample density
        float density = sampler.sampleDensity(pos.x, pos.y, pos.z) * m_config.densityMultiplier;

        if (density > 0.001f) {
            // Beer-Lambert absorption
            float extinction = (m_config.absorptionCoeff + m_config.scatteringCoeff) * density * stepSize;
            float sampleTransmittance = std::exp(-extinction);

            // In-scattering contribution
            SoftVec3 scatterColor = m_config.lightColor * m_config.lightIntensity
                                    * m_config.scatteringCoeff * density * phase;

            // Emission from temperature
            if (sampler.sampleTemperature) {
                float temp = sampler.sampleTemperature(pos.x, pos.y, pos.z);
                SoftVec3 emission = EvaluateEmission(temp);
                scatterColor = scatterColor + emission * density;
            }

            // Integrate: accumulate color weighted by transmittance and step
            SoftVec3 colorContrib = scatterColor * ((1.0f - sampleTransmittance) / std::max(extinction, 0.001f));
            result.color = result.color + colorContrib * result.transmittance;
            result.transmittance *= sampleTransmittance;
        }

        t += stepSize;
        steps++;
    }

    return result;
}

void VolumetricRenderer::Render(GBuffer& gbuffer, const SoftCamera& camera) {
    int w = gbuffer.GetWidth();
    int h = gbuffer.GetHeight();

    // Ensure buffer is sized correctly
    if (m_width != w || m_height != h) {
        m_width = w;
        m_height = h;
        m_volumeBuffer.resize(w * h);
    }

    // Clear volume buffer
    for (auto& s : m_volumeBuffer) {
        s.color = {};
        s.transmittance = 1.0f;
    }

    if (m_volumes.empty()) return;

    // For each pixel, march through all volumes
#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sceneDepth = gbuffer.GetDepth(x, y);
            float maxDist = (sceneDepth > 9999.0f) ? 1000.0f : sceneDepth;

            SoftVec3 rayDir = PixelToRayDir(x, y, camera);

            VolumetricSample accumulated;
            accumulated.transmittance = 1.0f;
            accumulated.color = {};

            for (const auto& vol : m_volumes) {
                VolumetricSample sample = MarchRay(camera.position, rayDir, maxDist, vol);

                // Combine: the second volume sees through the first
                accumulated.color = accumulated.color + sample.color * accumulated.transmittance;
                accumulated.transmittance *= sample.transmittance;
            }

            m_volumeBuffer[y * w + x] = accumulated;

            // Composite over scene color
            SoftColorRGBA8 sceneColor = gbuffer.GetColor(x, y);
            SoftVec3 scene = {sceneColor.r / 255.0f, sceneColor.g / 255.0f, sceneColor.b / 255.0f};

            // Front-to-back blending
            SoftVec3 final_color = accumulated.color + scene * accumulated.transmittance;

            // Clamp
            final_color.x = VolClamp01(final_color.x);
            final_color.y = VolClamp01(final_color.y);
            final_color.z = VolClamp01(final_color.z);

            gbuffer.SetColor(x, y, SoftColorRGBA8::FromFloat(final_color.x, final_color.y, final_color.z));
        }
    }
}

} // namespace WulfNet
