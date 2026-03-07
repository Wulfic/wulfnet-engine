// =============================================================================
// WulfNet Engine - Occlusion Culler Implementation
// =============================================================================

#include "WulfNet/Rendering/SoftwareRasterizer/OcclusionCuller.h"
#include <cmath>
#include <algorithm>

namespace WulfNet {

bool OcclusionCuller::Initialize(const OcclusionCullerConfig& config) {
    m_config = config;

    SoftRasterizerConfig rastConfig;
    rastConfig.width = config.width;
    rastConfig.height = config.height;
    rastConfig.threadCount = 1;  // Low-res, single-threaded is fine
    rastConfig.enableBackfaceCulling = true;

    return m_rasterizer.Initialize(rastConfig);
}

void OcclusionCuller::RenderOccluders(const SoftTransform* occluders, int count,
                                       const SoftCamera& camera) {
    m_rasterizer.Clear();
    m_rasterizer.RenderObjects(occluders, count, camera);
}

Vec3 OcclusionCuller::ProjectToScreen(const Vec3& worldPos, const SoftCamera& camera) const {
    Vec3 rel = worldPos - camera.position;
    float vx = rel.Dot(camera.right);
    float vy = rel.Dot(camera.up);
    float vz = rel.Dot(camera.forward);

    if (vz <= camera.nearPlane) vz = camera.nearPlane;

    float fovScale = 1.0f / std::tan(camera.fov * 0.5f * 3.14159265f / 180.0f);
    float invZ = 1.0f / vz;

    float ndcX = vx * fovScale * invZ / camera.aspectRatio;
    float ndcY = vy * fovScale * invZ;

    float screenX = (ndcX * 0.5f + 0.5f) * static_cast<float>(m_config.width);
    float screenY = (0.5f - ndcY * 0.5f) * static_cast<float>(m_config.height);

    return {screenX, screenY, vz};
}

bool OcclusionCuller::IsVisible(const AABox& worldBounds, const SoftCamera& camera) const {
    // Project all 8 corners of the AABB to screen space
    Vec3 corners[8] = {
        {worldBounds.min.x, worldBounds.min.y, worldBounds.min.z},
        {worldBounds.max.x, worldBounds.min.y, worldBounds.min.z},
        {worldBounds.min.x, worldBounds.max.y, worldBounds.min.z},
        {worldBounds.max.x, worldBounds.max.y, worldBounds.min.z},
        {worldBounds.min.x, worldBounds.min.y, worldBounds.max.z},
        {worldBounds.max.x, worldBounds.min.y, worldBounds.max.z},
        {worldBounds.min.x, worldBounds.max.y, worldBounds.max.z},
        {worldBounds.max.x, worldBounds.max.y, worldBounds.max.z},
    };

    float minScreenX = 1e10f, maxScreenX = -1e10f;
    float minScreenY = 1e10f, maxScreenY = -1e10f;
    float minDepth = 1e10f;

    bool anyInFront = false;

    for (int i = 0; i < 8; ++i) {
        Vec3 projected = ProjectToScreen(corners[i], camera);

        if (projected.z > camera.nearPlane) {
            anyInFront = true;
            minScreenX = std::min(minScreenX, projected.x);
            maxScreenX = std::max(maxScreenX, projected.x);
            minScreenY = std::min(minScreenY, projected.y);
            maxScreenY = std::max(maxScreenY, projected.y);
            minDepth = std::min(minDepth, projected.z);
        }
    }

    if (!anyInFront) return false;

    // Clip to screen
    int startX = std::max(0, static_cast<int>(std::floor(minScreenX)));
    int endX = std::min(m_config.width - 1, static_cast<int>(std::ceil(maxScreenX)));
    int startY = std::max(0, static_cast<int>(std::floor(minScreenY)));
    int endY = std::min(m_config.height - 1, static_cast<int>(std::ceil(maxScreenY)));

    if (startX > endX || startY > endY) return false;

    // Conservative test: check if ANY pixel in the AABB's screen rect
    // has a depth value >= our minimum depth (meaning we can see through)
    const float* depthBuffer = m_rasterizer.GetGBuffer().GetDepthBuffer();

    // Sample a grid of points for efficiency (not every pixel)
    int stepX = std::max(1, (endX - startX) / 4);
    int stepY = std::max(1, (endY - startY) / 4);

    for (int py = startY; py <= endY; py += stepY) {
        for (int px = startX; px <= endX; px += stepX) {
            float bufferDepth = depthBuffer[py * m_config.width + px];
            if (minDepth <= bufferDepth) {
                return true;  // At least one sample is closer than occluder
            }
        }
    }

    return false;
}

void OcclusionCuller::TestVisibility(const AABox* bounds, bool* results, int count,
                                      const SoftCamera& camera) const {
    for (int i = 0; i < count; ++i) {
        results[i] = IsVisible(bounds[i], camera);
    }
}

} // namespace WulfNet
