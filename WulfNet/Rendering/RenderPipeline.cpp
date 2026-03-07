// =============================================================================
// WulfNet Engine - Render Pipeline Implementation
// =============================================================================

#include "WulfNet/Rendering/RenderPipeline.h"
#include <cmath>
#include <algorithm>

#ifdef WULFNET_HAS_OPENMP
#include <omp.h>
#endif

namespace WulfNet {

static float RPClamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

bool RenderPipeline::Initialize(const RenderPipelineConfig& config) {
    m_config = config;

    // Initialize rasterizer
    if (!m_rasterizer.Initialize(config.rasterizer)) return false;

    // Initialize shadow system
    if (config.enableShadows) {
        if (!m_shadows.Initialize(config.shadows)) return false;
    }

    // Initialize GI
    if (config.enableGI) {
        if (!m_gi.Initialize(config.rasterizer.width, config.rasterizer.height, config.gi))
            return false;
    }

    // Initialize volumetric renderer
    if (config.enableVolumetric) {
        if (!m_volumetric.Initialize(config.rasterizer.width, config.rasterizer.height, config.volumetric))
            return false;
    }

    m_initialized = true;
    return true;
}

void RenderPipeline::Shutdown() {
    m_rasterizer.Shutdown();
    m_meshCache.clear();
    m_initialized = false;
}

// =============================================================================
// IRenderer interface — frame-oriented API
// =============================================================================

void RenderPipeline::BeginFrame() {
    // Software rasterizer resets state in RenderFrame, nothing to do here.
}

void RenderPipeline::Submit(const RenderableList& renderables, const SoftCamera& camera) {
    // Convert RenderCommands into the legacy per-frame rendering path.
    // Collect mesh transforms and volumes, then call RenderFrame.

    // First, handle volumes from render commands
    ClearVolumes();

    // Build a transform array from mesh instance commands
    std::vector<SoftTransform> transforms;
    transforms.reserve(renderables.GetCount());

    for (const auto& cmd : renderables.GetCommands()) {
        switch (cmd.type) {
            case RenderCommandType::MeshInstance:
                transforms.push_back(cmd.transform);
                break;
            case RenderCommandType::Volume:
                if (cmd.volumeSampler) {
                    AddVolume(*cmd.volumeSampler);
                }
                break;
            default:
                // Debug primitives — not yet handled by software rasterizer
                break;
        }
    }

    // Render the frame with collected transforms
    if (!transforms.empty()) {
        RenderFrame(transforms.data(), static_cast<int>(transforms.size()), camera);
    } else {
        // Still need to render (for volumetrics, lighting on empty scene, etc.)
        RenderFrame(nullptr, 0, camera);
    }
}

void RenderPipeline::EndFrame() {
    // Software rasterizer produces the final buffer in RenderFrame/GetColorBuffer.
    // Nothing to finalize here.
}

int RenderPipeline::AddMesh(const SoftMesh& mesh) {
    m_meshCache.push_back(mesh);
    return m_rasterizer.AddMesh(mesh);
}

int RenderPipeline::AddTexture(const SoftTexture& texture) {
    return m_rasterizer.AddTexture(texture);
}

void RenderPipeline::AddVolume(const VolumeSampler& sampler) {
    m_volumetric.AddVolume(sampler);
}

void RenderPipeline::ClearVolumes() {
    m_volumetric.ClearVolumes();
}

void RenderPipeline::AddLightProbe(const LightProbe& probe) {
    m_gi.AddProbe(probe);
}

void RenderPipeline::RenderFrame(const SoftTransform* transforms, int transformCount,
                                   const SoftCamera& camera) {
    m_stats = {}; // Reset stats

    // Pass 1: Shadow maps
    if (m_config.enableShadows) {
        PassShadow(transforms, transformCount, camera);
    }

    // Pass 2: GBuffer (geometry rasterization)
    PassGBuffer(transforms, transformCount, camera);

    // Pass 3: Global Illumination (SSAO + indirect)
    if (m_config.enableGI) {
        PassGI(camera);
    }

    // Pass 4: Deferred lighting (with shadow integration)
    PassLighting(camera);

    // Pass 5: Volumetric rendering
    if (m_config.enableVolumetric && m_volumetric.GetVolumeCount() > 0) {
        PassVolumetric(camera);
    }
}

const uint32_t* RenderPipeline::GetColorBuffer() const {
    return m_rasterizer.GetColorBuffer();
}

// =============================================================================
// Pass Implementations
// =============================================================================

void RenderPipeline::PassShadow(const SoftTransform* transforms, int transformCount,
                                  const SoftCamera& camera) {
    m_shadows.ClearAll();

    // Render directional light shadows
    m_shadows.RenderDirectionalShadows(m_config.shading.sunLight, camera,
                                        transforms, transformCount, m_meshCache);

    m_stats.shadowCascadesUsed = m_shadows.GetCascadeCount();

    // Render point light shadows
    for (int i = 0; i < static_cast<int>(m_config.shading.pointLights.size()); ++i) {
        m_shadows.RenderPointLightShadow(i, m_config.shading.pointLights[i],
                                           transforms, transformCount, m_meshCache);
        m_stats.pointLightShadows++;
    }
}

void RenderPipeline::PassGBuffer(const SoftTransform* transforms, int transformCount,
                                   const SoftCamera& camera) {
    m_rasterizer.Clear();
    m_rasterizer.RenderObjects(transforms, transformCount, camera);
}

void RenderPipeline::PassGI(const SoftCamera& camera) {
    m_gi.Compute(m_rasterizer.GetGBuffer(), camera);
}

void RenderPipeline::PassLighting(const SoftCamera& camera) {
    GBuffer& gbuffer = m_rasterizer.GetGBuffer();
    int width = gbuffer.GetWidth();
    int height = gbuffer.GetHeight();

    float fovScale = 1.0f / std::tan(camera.fov * 0.5f * 3.14159265f / 180.0f);

#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float depth = gbuffer.GetDepth(x, y);
            if (depth > 9999.0f) continue;

            // Unpack normal
            SoftColorRGBA8 packedNormal = gbuffer.GetNormal(x, y);
            Vec3 normal = {
                (packedNormal.r / 255.0f) * 2.0f - 1.0f,
                (packedNormal.g / 255.0f) * 2.0f - 1.0f,
                (packedNormal.b / 255.0f) * 2.0f - 1.0f
            };
            float nlen = normal.Length();
            if (nlen > 0.001f) normal = normal * (1.0f / nlen);

            // Reconstruct view direction
            float ndcX = (2.0f * (x + 0.5f) / float(width) - 1.0f) * camera.aspectRatio;
            float ndcY = 1.0f - 2.0f * (y + 0.5f) / float(height);
            Vec3 viewDir = (camera.forward * fovScale + camera.right * ndcX + camera.up * ndcY).Normalized();
            Vec3 worldPos = camera.position + viewDir * depth;

            // Get albedo
            SoftColorRGBA8 albedoColor = gbuffer.GetColor(x, y);
            Vec3 albedo = {albedoColor.r / 255.0f, albedoColor.g / 255.0f, albedoColor.b / 255.0f};

            Vec3 color = {};

            // --- Ambient + GI ---
            float upFactor = normal.y * 0.5f + 0.5f;
            Vec3 ambient = Vec3::Lerp(m_config.shading.ambientGroundColor,
                                                m_config.shading.ambientSkyColor, upFactor)
                               * m_config.shading.ambientIntensity;

            // Apply SSAO
            float ao = 1.0f;
            if (m_config.enableGI) {
                ao = m_gi.SampleAO(x, y);
            }

            color = color + Vec3{albedo.x * ambient.x, albedo.y * ambient.y, albedo.z * ambient.z} * ao;

            // Add indirect lighting
            if (m_config.enableGI) {
                Vec3 indirect = m_gi.SampleIndirect(x, y);
                color = color + Vec3{
                    albedo.x * indirect.x, albedo.y * indirect.y, albedo.z * indirect.z
                };
            }

            // --- Directional (sun) light with shadow ---
            Vec3 lightDir = m_config.shading.sunLight.direction.Normalized() * -1.0f;
            float NdotL = std::max(0.0f, normal.Dot(lightDir));

            float shadowFactor = 1.0f;
            if (m_config.enableShadows) {
                shadowFactor = m_shadows.SampleDirectionalShadow(worldPos);
            }

            Vec3 diffuse = {
                albedo.x * m_config.shading.sunLight.color.x * NdotL * m_config.shading.sunLight.intensity * shadowFactor,
                albedo.y * m_config.shading.sunLight.color.y * NdotL * m_config.shading.sunLight.intensity * shadowFactor,
                albedo.z * m_config.shading.sunLight.color.z * NdotL * m_config.shading.sunLight.intensity * shadowFactor
            };
            color = color + diffuse;

            // Specular (Blinn-Phong) with shadow
            Vec3 halfVec = (lightDir + viewDir * -1.0f).Normalized();
            float NdotH = std::max(0.0f, normal.Dot(halfVec));
            float specular = std::pow(NdotH, 32.0f) * m_config.shading.specularIntensity * shadowFactor;
            color = color + m_config.shading.sunLight.color * specular;

            // --- Point lights with shadow ---
            for (int i = 0; i < static_cast<int>(m_config.shading.pointLights.size()); ++i) {
                const auto& light = m_config.shading.pointLights[i];
                Vec3 toLight = light.position - worldPos;
                float dist = toLight.Length();
                if (dist > light.range) continue;

                Vec3 lightDirP = toLight / dist;
                float NdotLP = std::max(0.0f, normal.Dot(lightDirP));
                float attenuation = 1.0f - RPClamp01(dist / light.range);
                attenuation *= attenuation;

                float pointShadow = 1.0f;
                if (m_config.enableShadows) {
                    pointShadow = m_shadows.SamplePointLightShadow(i, worldPos);
                }

                color = color + Vec3{
                    albedo.x * light.color.x * NdotLP * light.intensity * attenuation * pointShadow,
                    albedo.y * light.color.y * NdotLP * light.intensity * attenuation * pointShadow,
                    albedo.z * light.color.z * NdotLP * light.intensity * attenuation * pointShadow
                };
            }

            // --- Fresnel ---
            float fresnel = 1.0f - std::max(0.0f, normal.Dot(viewDir * -1.0f));
            fresnel = std::pow(fresnel, m_config.shading.fresnelPower);
            color = Vec3::Lerp(color, m_config.shading.ambientSkyColor, fresnel * 0.3f);

            // --- Distance fog ---
            float fogFactor = RPClamp01((depth - m_config.shading.fogStart) /
                                         (m_config.shading.fogEnd - m_config.shading.fogStart));
            color = Vec3::Lerp(color, m_config.shading.fogColor, fogFactor);

            // Clamp and write
            color.x = RPClamp01(color.x);
            color.y = RPClamp01(color.y);
            color.z = RPClamp01(color.z);
            gbuffer.SetColor(x, y, SoftColorRGBA8::FromFloat(color.x, color.y, color.z));
        }
    }
}

void RenderPipeline::PassVolumetric(const SoftCamera& camera) {
    m_stats.volumetricVolumes = m_volumetric.GetVolumeCount();
    m_volumetric.Render(m_rasterizer.GetGBuffer(), camera);
}

} // namespace WulfNet
