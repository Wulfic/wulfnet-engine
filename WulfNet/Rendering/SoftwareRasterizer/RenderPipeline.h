// =============================================================================
// WulfNet Engine - Render Pipeline
// =============================================================================
// Unified render pipeline orchestrator that manages the full rendering flow:
//
//   1. Shadow Pass  — Render shadow maps from light POVs
//   2. GBuffer Pass — Rasterize scene geometry into the GBuffer
//   3. SSAO/GI Pass — Compute screen-space ambient occlusion + indirect
//   4. Lighting Pass — Deferred shading with shadow factor + GI integration
//   5. Volumetric Pass — Ray-march through gas/fluid volumes
//   6. Post-process  — (future: tone mapping, bloom, etc.)
//
// This class owns and coordinates all sub-systems and provides a single
// RenderFrame() entry point.
// =============================================================================

#pragma once

#include "WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h"
#include "WulfNet/Rendering/SoftwareRasterizer/SoftwareRasterizer.h"
#include "WulfNet/Rendering/SoftwareRasterizer/DeferredShading.h"
#include "WulfNet/Rendering/SoftwareRasterizer/ShadowMap.h"
#include "WulfNet/Rendering/SoftwareRasterizer/GlobalIllumination.h"
#include "WulfNet/Rendering/SoftwareRasterizer/VolumetricRenderer.h"
#include <vector>

namespace WulfNet {

// =============================================================================
// Render Pipeline Configuration
// =============================================================================

struct RenderPipelineConfig {
    SoftRasterizerConfig     rasterizer;
    DeferredShadingConfig    shading;
    ShadowSystemConfig       shadows;
    GlobalIlluminationConfig gi;
    VolumetricConfig         volumetric;

    bool enableShadows       = true;
    bool enableGI            = true;
    bool enableVolumetric    = true;
};

// =============================================================================
// Frame Statistics
// =============================================================================

struct RenderStats {
    int trianglesRendered    = 0;
    int shadowCascadesUsed   = 0;
    int pointLightShadows    = 0;
    int volumetricVolumes    = 0;
    float shadowPassMs       = 0.0f;
    float gbufferPassMs      = 0.0f;
    float giPassMs           = 0.0f;
    float lightingPassMs     = 0.0f;
    float volumetricPassMs   = 0.0f;
    float totalFrameMs       = 0.0f;
};

// =============================================================================
// Render Pipeline
// =============================================================================

class RenderPipeline {
public:
    RenderPipeline() = default;

    /// Initialize all sub-systems
    bool Initialize(const RenderPipelineConfig& config = {});

    /// Shut down and release resources
    void Shutdown();

    // =========================================================================
    // Scene Management (delegates to SoftwareRasterizer)
    // =========================================================================

    int AddMesh(const SoftMesh& mesh);
    int AddTexture(const SoftTexture& texture);
    void AddVolume(const VolumeSampler& sampler);
    void ClearVolumes();
    void AddLightProbe(const LightProbe& probe);

    // =========================================================================
    // Rendering
    // =========================================================================

    /// Full render frame: shadow → GBuffer → GI → lighting → volumetric
    void RenderFrame(const SoftTransform* transforms, int transformCount,
                     const SoftCamera& camera);

    /// Get the final color buffer for display
    const uint32_t* GetColorBuffer() const;

    /// Get individual sub-system references (for advanced customization)
    SoftwareRasterizer& GetRasterizer() { return m_rasterizer; }
    DeferredShading& GetDeferredShading() { return m_deferred; }
    ShadowSystem& GetShadowSystem() { return m_shadows; }
    GlobalIllumination& GetGI() { return m_gi; }
    VolumetricRenderer& GetVolumetric() { return m_volumetric; }
    GBuffer& GetGBuffer() { return m_rasterizer.GetGBuffer(); }

    const RenderStats& GetStats() const { return m_stats; }
    const RenderPipelineConfig& GetConfig() const { return m_config; }

    int GetWidth() const { return m_config.rasterizer.width; }
    int GetHeight() const { return m_config.rasterizer.height; }

    /// Update shading config (lights, fog, etc.) between frames
    void SetShadingConfig(const DeferredShadingConfig& config) { m_config.shading = config; }

private:
    /// Individual pass methods
    void PassShadow(const SoftTransform* transforms, int transformCount,
                    const SoftCamera& camera);
    void PassGBuffer(const SoftTransform* transforms, int transformCount,
                     const SoftCamera& camera);
    void PassGI(const SoftCamera& camera);
    void PassLighting(const SoftCamera& camera);
    void PassVolumetric(const SoftCamera& camera);

    RenderPipelineConfig m_config;
    RenderStats m_stats;

    // Sub-systems
    SoftwareRasterizer m_rasterizer;
    DeferredShading    m_deferred;
    ShadowSystem       m_shadows;
    GlobalIllumination m_gi;
    VolumetricRenderer m_volumetric;

    // Cached mesh list for shadow rendering
    std::vector<SoftMesh> m_meshCache;
    bool m_initialized = false;
};

} // namespace WulfNet
