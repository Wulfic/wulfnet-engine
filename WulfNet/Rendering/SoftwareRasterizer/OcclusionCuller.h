// =============================================================================
// WulfNet Engine - Occlusion Culler
// =============================================================================
// Low-resolution CPU occlusion culling using the software rasterizer.
// Renders occluder geometry at reduced resolution and tests AABBs against
// the resulting depth buffer.
// =============================================================================

#pragma once

#include "WulfNet/Rendering/SoftwareRasterizer/SoftwareRasterizer.h"
#include <vector>

namespace WulfNet {

struct AABox {
    SoftVec3 min;
    SoftVec3 max;

    SoftVec3 Center() const { return (min + max) * 0.5f; }
    SoftVec3 Extent() const { return (max - min) * 0.5f; }
};

struct OcclusionCullerConfig {
    int width = 256;
    int height = 144;
};

class OcclusionCuller {
public:
    OcclusionCuller() = default;

    bool Initialize(const OcclusionCullerConfig& config = {});

    /// Render occluder meshes to the low-res depth buffer
    void RenderOccluders(const SoftTransform* occluders, int count, const SoftCamera& camera);

    /// Test a single AABB against the depth buffer
    bool IsVisible(const AABox& worldBounds, const SoftCamera& camera) const;

    /// Batch test multiple AABBs
    void TestVisibility(const AABox* bounds, bool* results, int count,
                        const SoftCamera& camera) const;

    /// Add a mesh to the rasterizer's mesh pool
    int AddMesh(const SoftMesh& mesh) { return m_rasterizer.AddMesh(mesh); }

    /// Get the low-res depth buffer (for debug visualization)
    const float* GetDepthBuffer() const { return m_rasterizer.GetGBuffer().GetDepthBuffer(); }
    int GetWidth() const { return m_config.width; }
    int GetHeight() const { return m_config.height; }

private:
    SoftVec3 ProjectToScreen(const SoftVec3& worldPos, const SoftCamera& camera) const;

    OcclusionCullerConfig m_config;
    SoftwareRasterizer m_rasterizer;
};

} // namespace WulfNet
