// =============================================================================
// WulfNet Engine - Software Rasterizer Core
// =============================================================================
// CPU scanline rasterizer with backface culling, perspective-correct
// interpolation, depth testing, and multi-threaded object-level parallelism.
// Ported from RenderObjectsPooled() in BG-C-Software-Renderer/MainEngine.cpp.
// =============================================================================

#pragma once

#include "WulfNet/Rendering/Types/RenderTypes.h"
#include "WulfNet/Rendering/SoftwareRasterizer/GBuffer.h"
#include <vector>
#include <functional>

namespace WulfNet {

struct SoftRasterizerConfig {
    int width = 1280;
    int height = 720;
    int threadCount = 0;  // 0 = auto-detect
    bool enableBackfaceCulling = true;
};

class SoftwareRasterizer {
public:
    SoftwareRasterizer();
    ~SoftwareRasterizer();

    bool Initialize(const SoftRasterizerConfig& config);
    void Shutdown();

    // ==========================================================================
    // Scene Management
    // ==========================================================================

    int AddMesh(const SoftMesh& mesh);
    int AddTexture(const SoftTexture& texture);

    // ==========================================================================
    // Rendering
    // ==========================================================================

    /// Clear all buffers
    void Clear(const Vec3& skyTop = {0.4f, 0.6f, 0.9f},
               const Vec3& skyBottom = {0.8f, 0.85f, 0.95f});

    /// Render objects to the GBuffer
    void RenderObjects(const SoftTransform* objects, int count, const SoftCamera& camera);

    /// Get the GBuffer for post-processing
    GBuffer& GetGBuffer() { return m_gbuffer; }
    const GBuffer& GetGBuffer() const { return m_gbuffer; }

    /// Get final color buffer pointer (for display)
    const uint32_t* GetColorBuffer() const { return m_gbuffer.GetColorBuffer(); }

    int GetWidth() const { return m_config.width; }
    int GetHeight() const { return m_config.height; }

private:
    // Per-triangle rasterization
    void RasterizeTriangle(const SoftVertex& v0, const SoftVertex& v1, const SoftVertex& v2,
                           const Vec3& faceNormal, const SoftMaterial& material,
                           const SoftCamera& camera, const SoftColorRGBA8& tint);

    // World-to-screen projection
    Vec3 ProjectToScreen(const Vec3& worldPos, const SoftCamera& camera) const;

    // Transform vertex to world space
    Vec3 TransformPoint(const Vec3& point, const SoftTransform& transform) const;
    Vec3 TransformNormal(const Vec3& normal, const SoftTransform& transform) const;

    SoftRasterizerConfig m_config;
    GBuffer m_gbuffer;

    // Scene data
    std::vector<SoftMesh> m_meshes;
    std::vector<SoftTexture> m_textures;

    // Threading (uses Core/Threading/ThreadPool)
    int m_threadCount = 1;
};

} // namespace WulfNet
