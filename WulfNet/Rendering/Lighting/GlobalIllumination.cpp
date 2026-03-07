// =============================================================================
// WulfNet Engine - Global Illumination Implementation
// =============================================================================

#include "WulfNet/Rendering/Lighting/GlobalIllumination.h"
#include <cstring>
#include <algorithm>

#ifdef WULFNET_HAS_OPENMP
#include <omp.h>
#endif

namespace WulfNet {

// =============================================================================
// Helpers
// =============================================================================

static float GIClamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

static constexpr float kGIPi = 3.14159265358979323846f;

// Simple hash for deterministic pseudo-random sampling (no external RNG needed)
static float HashFloat(int x, int y, int s) {
    uint32_t h = static_cast<uint32_t>(x * 374761393 + y * 668265263 + s * 1274126177);
    h = (h ^ (h >> 13)) * 1274126177;
    h = h ^ (h >> 16);
    return static_cast<float>(h & 0xFFFF) / 65535.0f;
}

// =============================================================================
// GlobalIllumination
// =============================================================================

bool GlobalIllumination::Initialize(int width, int height, const GlobalIlluminationConfig& config) {
    if (width <= 0 || height <= 0) return false;

    m_width = width;
    m_height = height;
    m_config = config;

    int pixelCount = width * height;
    m_aoBuffer.resize(pixelCount, 1.0f);
    m_aoTempBuffer.resize(pixelCount, 1.0f);
    m_indirectBuffer.resize(pixelCount, SoftVec3{0, 0, 0});

    return true;
}

void GlobalIllumination::Compute(const GBuffer& gbuffer, const SoftCamera& camera) {
#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            float depth = gbuffer.GetDepth(x, y);
            int idx = y * m_width + x;

            // Skip sky pixels
            if (depth > 9999.0f) {
                m_aoBuffer[idx] = 1.0f;
                m_indirectBuffer[idx] = {};
                continue;
            }

            // SSAO
            if (m_config.ssaoEnabled) {
                m_aoBuffer[idx] = ComputeSSAOPixel(gbuffer, camera, x, y);
            } else {
                m_aoBuffer[idx] = 1.0f;
            }

            // Indirect lighting
            if (m_config.indirect.enabled) {
                m_indirectBuffer[idx] = ComputeIndirectPixel(gbuffer, camera, x, y);
            } else {
                m_indirectBuffer[idx] = {};
            }
        }
    }

    // Blur AO to reduce noise
    if (m_config.ssaoEnabled) {
        for (int pass = 0; pass < m_config.ssao.blurPasses; ++pass) {
            BlurAOBuffer();
        }
    }
}

float GlobalIllumination::SampleAO(int x, int y) const {
    if (x < 0 || x >= m_width || y < 0 || y >= m_height) return 1.0f;
    return m_aoBuffer[y * m_width + x];
}

SoftVec3 GlobalIllumination::SampleIndirect(int x, int y) const {
    if (x < 0 || x >= m_width || y < 0 || y >= m_height) return {};
    return m_indirectBuffer[y * m_width + x];
}

SoftVec3 GlobalIllumination::EvaluateProbes(const SoftVec3& worldPos, const SoftVec3& normal) const {
    SoftVec3 result = {};

    for (const auto& probe : m_config.probes) {
        SoftVec3 toProbe = probe.position - worldPos;
        float dist = toProbe.Length();

        if (dist > probe.radius) continue;

        // Smooth falloff — quadratic attenuation with distance
        float t = dist / probe.radius;
        float weight = (1.0f - t) * (1.0f - t);

        SoftVec3 irradiance = probe.Evaluate(normal);
        result = result + irradiance * weight;
    }

    return result;
}

void GlobalIllumination::BlurAOBuffer() {
    int halfK = m_config.ssao.blurKernelSize / 2;

    // Copy to temp
    std::copy(m_aoBuffer.begin(), m_aoBuffer.end(), m_aoTempBuffer.begin());

#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            float sum = 0.0f;
            int count = 0;

            for (int ky = -halfK; ky <= halfK; ++ky) {
                for (int kx = -halfK; kx <= halfK; ++kx) {
                    int sx = x + kx;
                    int sy = y + ky;
                    if (sx >= 0 && sx < m_width && sy >= 0 && sy < m_height) {
                        sum += m_aoTempBuffer[sy * m_width + sx];
                        count++;
                    }
                }
            }

            m_aoBuffer[y * m_width + x] = (count > 0) ? (sum / count) : 1.0f;
        }
    }
}

void GlobalIllumination::AddProbe(const LightProbe& probe) {
    m_config.probes.push_back(probe);
}

// =============================================================================
// Private methods
// =============================================================================

SoftVec3 GlobalIllumination::UnpackNormal(SoftColorRGBA8 packed) const {
    SoftVec3 n = {
        (packed.r / 255.0f) * 2.0f - 1.0f,
        (packed.g / 255.0f) * 2.0f - 1.0f,
        (packed.b / 255.0f) * 2.0f - 1.0f
    };
    float len = n.Length();
    if (len > 0.001f) n = n * (1.0f / len);
    return n;
}

SoftVec3 GlobalIllumination::ReconstructWorldPos(int x, int y, float depth,
                                                  const SoftCamera& camera) const {
    float fovScale = 1.0f / std::tan(camera.fov * 0.5f * kGIPi / 180.0f);
    float ndcX = (2.0f * (x + 0.5f) / static_cast<float>(m_width) - 1.0f) * camera.aspectRatio;
    float ndcY = 1.0f - 2.0f * (y + 0.5f) / static_cast<float>(m_height);
    SoftVec3 viewDir = (camera.forward * fovScale + camera.right * ndcX + camera.up * ndcY).Normalized();
    return camera.position + viewDir * depth;
}

SoftVec3 GlobalIllumination::HemisphereSample(int sampleIndex, int totalSamples,
                                                const SoftVec3& normal) const {
    // Fibonacci hemisphere sampling (deterministic, uniform distribution)
    float goldenRatio = (1.0f + std::sqrt(5.0f)) * 0.5f;
    float theta = 2.0f * kGIPi * sampleIndex / goldenRatio;
    float cosTheta = 1.0f - (static_cast<float>(sampleIndex) + 0.5f) / static_cast<float>(totalSamples);
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));

    SoftVec3 sample = {
        sinTheta * std::cos(theta),
        sinTheta * std::sin(theta),
        cosTheta
    };

    // Orient to hemisphere around normal using Gram-Schmidt
    SoftVec3 tangent;
    if (std::abs(normal.x) < 0.9f)
        tangent = SoftVec3{1, 0, 0}.Cross(normal).Normalized();
    else
        tangent = SoftVec3{0, 1, 0}.Cross(normal).Normalized();

    SoftVec3 bitangent = normal.Cross(tangent);

    return (tangent * sample.x + bitangent * sample.y + normal * sample.z).Normalized();
}

float GlobalIllumination::ComputeSSAOPixel(const GBuffer& gbuffer, const SoftCamera& camera,
                                            int x, int y) const {
    float depth = gbuffer.GetDepth(x, y);
    SoftColorRGBA8 packedN = gbuffer.GetNormal(x, y);
    SoftVec3 normal = UnpackNormal(packedN);
    SoftVec3 worldPos = ReconstructWorldPos(x, y, depth, camera);

    float fovScale = 1.0f / std::tan(camera.fov * 0.5f * kGIPi / 180.0f);

    float occlusion = 0.0f;
    int sampleCount = m_config.ssao.sampleCount;

    for (int s = 0; s < sampleCount; ++s) {
        // Get sample direction in hemisphere around normal
        SoftVec3 sampleDir = HemisphereSample(s, sampleCount, normal);

        // Add per-pixel randomization using hash
        float randRotation = HashFloat(x, y, s) * 2.0f * kGIPi;
        float c = std::cos(randRotation);
        float sn = std::sin(randRotation);
        SoftVec3 rotated = {
            sampleDir.x * c - sampleDir.y * sn,
            sampleDir.x * sn + sampleDir.y * c,
            sampleDir.z
        };
        // Re-orient to hemisphere (make sure it faces the normal side)
        if (rotated.Dot(normal) < 0.0f) rotated = rotated * -1.0f;

        // Scale by radius with distance-based distribution
        float scale = (static_cast<float>(s) + 1.0f) / static_cast<float>(sampleCount);
        scale = 0.1f + scale * scale * 0.9f; // Accelerating distribution

        SoftVec3 samplePos = worldPos + rotated * (m_config.ssao.radius * scale);

        // Project sample to screen
        SoftVec3 toSample = samplePos - camera.position;
        float sampleDepth = toSample.Dot(camera.forward);
        if (sampleDepth <= 0.0f) continue;

        float projX = toSample.Dot(camera.right) / sampleDepth * fovScale;
        float projY = toSample.Dot(camera.up) / sampleDepth * fovScale;

        int screenX = static_cast<int>((projX + camera.aspectRatio) / (2.0f * camera.aspectRatio) * m_width);
        int screenY = static_cast<int>((1.0f - projY) * 0.5f * m_height);

        if (screenX < 0 || screenX >= m_width || screenY < 0 || screenY >= m_height) continue;

        float sceneDepth = gbuffer.GetDepth(screenX, screenY);

        // Range check: ignore samples too far away
        float rangeCheck = GIClamp01(m_config.ssao.radius / std::abs(depth - sceneDepth + 0.001f));

        if (sceneDepth < sampleDepth - m_config.ssao.bias) {
            occlusion += rangeCheck;
        }
    }

    occlusion = occlusion / static_cast<float>(sampleCount);
    float ao = 1.0f - (occlusion * m_config.ssao.intensity);
    ao = std::pow(GIClamp01(ao), m_config.ssao.power);

    return ao;
}

SoftVec3 GlobalIllumination::ComputeIndirectPixel(const GBuffer& gbuffer, const SoftCamera& camera,
                                                    int x, int y) const {
    float depth = gbuffer.GetDepth(x, y);
    SoftColorRGBA8 packedN = gbuffer.GetNormal(x, y);
    SoftVec3 normal = UnpackNormal(packedN);
    SoftVec3 worldPos = ReconstructWorldPos(x, y, depth, camera);

    float fovScale = 1.0f / std::tan(camera.fov * 0.5f * kGIPi / 180.0f);

    SoftVec3 indirectColor = {};
    int sampleCount = m_config.indirect.sampleCount;
    int validSamples = 0;

    for (int s = 0; s < sampleCount; ++s) {
        SoftVec3 sampleDir = HemisphereSample(s, sampleCount, normal);

        // Randomize per-pixel
        float angle = HashFloat(x, y, s + 1000) * 2.0f * kGIPi;
        float c = std::cos(angle);
        float sn = std::sin(angle);
        SoftVec3 rotated = {
            sampleDir.x * c - sampleDir.y * sn,
            sampleDir.x * sn + sampleDir.y * c,
            sampleDir.z
        };
        if (rotated.Dot(normal) < 0.0f) rotated = rotated * -1.0f;

        SoftVec3 samplePos = worldPos + rotated * m_config.indirect.bounceRadius;

        // Project to screen
        SoftVec3 toSample = samplePos - camera.position;
        float sampleDepth = toSample.Dot(camera.forward);
        if (sampleDepth <= 0.0f) continue;

        float projX = toSample.Dot(camera.right) / sampleDepth * fovScale;
        float projY = toSample.Dot(camera.up) / sampleDepth * fovScale;

        int screenX = static_cast<int>((projX + camera.aspectRatio) / (2.0f * camera.aspectRatio) * m_width);
        int screenY = static_cast<int>((1.0f - projY) * 0.5f * m_height);

        if (screenX < 0 || screenX >= m_width || screenY < 0 || screenY >= m_height) continue;

        // Read color from GBuffer at the neighbor pixel (one-bounce: light that hit there bounces here)
        SoftColorRGBA8 neighborColor = gbuffer.GetColor(screenX, screenY);

        // Weight by cosine at receiver — standard SSGI hemisphere integration
        // (rotated is already in our normal's hemisphere, so dot >= 0)
        float weight = std::max(0.0f, normal.Dot(rotated));

        indirectColor = indirectColor + SoftVec3{
            neighborColor.r / 255.0f * weight,
            neighborColor.g / 255.0f * weight,
            neighborColor.b / 255.0f * weight
        };
        validSamples++;
    }

    if (validSamples > 0) {
        indirectColor = indirectColor * (m_config.indirect.bounceIntensity / static_cast<float>(validSamples));
    }

    // Add light probe contribution
    if (m_config.probesEnabled && !m_config.probes.empty()) {
        SoftVec3 probeContrib = EvaluateProbes(worldPos, normal);
        indirectColor = indirectColor + probeContrib;
    }

    return indirectColor;
}

} // namespace WulfNet
