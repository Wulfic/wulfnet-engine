// =============================================================================
// WulfNet Engine - Software Rasterizer Core Implementation
// =============================================================================

#include "WulfNet/Rendering/SoftwareRasterizer/SoftwareRasterizer.h"
#include "WulfNet/Core/Threading/ThreadPool.h"
#include <algorithm>
#include <cmath>

namespace WulfNet {

SoftwareRasterizer::SoftwareRasterizer() = default;
SoftwareRasterizer::~SoftwareRasterizer() { Shutdown(); }

bool SoftwareRasterizer::Initialize(const SoftRasterizerConfig& config) {
    m_config = config;
    m_threadCount = config.threadCount > 0 ? config.threadCount
                    : static_cast<int>(std::thread::hardware_concurrency());
    if (m_threadCount < 1) m_threadCount = 1;

    return m_gbuffer.Initialize(config.width, config.height);
}

void SoftwareRasterizer::Shutdown() {
    m_meshes.clear();
    m_textures.clear();
}

int SoftwareRasterizer::AddMesh(const SoftMesh& mesh) {
    m_meshes.push_back(mesh);
    return static_cast<int>(m_meshes.size()) - 1;
}

int SoftwareRasterizer::AddTexture(const SoftTexture& texture) {
    m_textures.push_back(texture);
    return static_cast<int>(m_textures.size()) - 1;
}

void SoftwareRasterizer::Clear(const Vec3& skyTop, const Vec3& skyBottom) {
    m_gbuffer.Clear(skyTop, skyBottom);
}

Vec3 SoftwareRasterizer::TransformPoint(const Vec3& point, const SoftTransform& transform) const {
    // Apply scale
    Vec3 p = {point.x * transform.scale.x, point.y * transform.scale.y, point.z * transform.scale.z};

    // Apply rotation (simplified Euler YXZ)
    float yRad = transform.rotation.y * 3.14159265f / 180.0f;
    float xRad = transform.rotation.x * 3.14159265f / 180.0f;
    float zRad = transform.rotation.z * 3.14159265f / 180.0f;

    float cy = std::cos(yRad), sy = std::sin(yRad);
    float cx = std::cos(xRad), sx = std::sin(xRad);
    float cz = std::cos(zRad), sz = std::sin(zRad);

    // Rotation matrix YXZ
    Vec3 r;
    r.x = (cy*cz + sy*sx*sz) * p.x + (-cy*sz + sy*sx*cz) * p.y + sy*cx * p.z;
    r.y = cx*sz * p.x + cx*cz * p.y + (-sx) * p.z;
    r.z = (-sy*cz + cy*sx*sz) * p.x + (sy*sz + cy*sx*cz) * p.y + cy*cx * p.z;

    return r + transform.position;
}

Vec3 SoftwareRasterizer::TransformNormal(const Vec3& normal, const SoftTransform& transform) const {
    float yRad = transform.rotation.y * 3.14159265f / 180.0f;
    float xRad = transform.rotation.x * 3.14159265f / 180.0f;
    float zRad = transform.rotation.z * 3.14159265f / 180.0f;

    float cy = std::cos(yRad), sy = std::sin(yRad);
    float cx = std::cos(xRad), sx = std::sin(xRad);
    float cz = std::cos(zRad), sz = std::sin(zRad);

    Vec3 r;
    r.x = (cy*cz + sy*sx*sz) * normal.x + (-cy*sz + sy*sx*cz) * normal.y + sy*cx * normal.z;
    r.y = cx*sz * normal.x + cx*cz * normal.y + (-sx) * normal.z;
    r.z = (-sy*cz + cy*sx*sz) * normal.x + (sy*sz + cy*sx*cz) * normal.y + cy*cx * normal.z;

    return r.Normalized();
}

Vec3 SoftwareRasterizer::ProjectToScreen(const Vec3& worldPos, const SoftCamera& camera) const {
    // View space
    Vec3 rel = worldPos - camera.position;
    float vx = rel.Dot(camera.right);
    float vy = rel.Dot(camera.up);
    float vz = rel.Dot(camera.forward);

    if (vz <= camera.nearPlane) {
        vz = camera.nearPlane;  // Clamp to near plane
    }

    // Perspective projection
    float fovScale = 1.0f / std::tan(camera.fov * 0.5f * 3.14159265f / 180.0f);
    float invZ = 1.0f / vz;

    float ndcX = vx * fovScale * invZ / camera.aspectRatio;
    float ndcY = vy * fovScale * invZ;

    // NDC to screen
    float screenX = (ndcX * 0.5f + 0.5f) * static_cast<float>(m_config.width);
    float screenY = (0.5f - ndcY * 0.5f) * static_cast<float>(m_config.height);

    return {screenX, screenY, vz};
}

void SoftwareRasterizer::RenderObjects(const SoftTransform* objects, int count,
                                        const SoftCamera& camera) {
    // Object-level parallelism via persistent ThreadPool
    auto renderObject = [&](int objIdx) {
        const SoftTransform& obj = objects[objIdx];
        if (obj.meshIndex < 0 || obj.meshIndex >= static_cast<int>(m_meshes.size())) return;
        const SoftMesh& mesh = m_meshes[obj.meshIndex];

        size_t triCount = mesh.indices.size() / 3;
        for (size_t tri = 0; tri < triCount; ++tri) {
            uint32_t i0 = mesh.indices[tri * 3 + 0];
            uint32_t i1 = mesh.indices[tri * 3 + 1];
            uint32_t i2 = mesh.indices[tri * 3 + 2];

            // Transform vertices to world space
            SoftVertex wv0 = mesh.vertices[i0];
            SoftVertex wv1 = mesh.vertices[i1];
            SoftVertex wv2 = mesh.vertices[i2];

            wv0.position = TransformPoint(wv0.position, obj);
            wv1.position = TransformPoint(wv1.position, obj);
            wv2.position = TransformPoint(wv2.position, obj);
            wv0.normal = TransformNormal(wv0.normal, obj);
            wv1.normal = TransformNormal(wv1.normal, obj);
            wv2.normal = TransformNormal(wv2.normal, obj);

            Vec3 faceNormal = tri < mesh.faceNormals.size()
                ? TransformNormal(mesh.faceNormals[tri], obj)
                : (wv1.position - wv0.position).Cross(wv2.position - wv0.position).Normalized();

            // Backface culling
            if (m_config.enableBackfaceCulling) {
                Vec3 viewDir = (wv0.position - camera.position).Normalized();
                if (faceNormal.Dot(viewDir) > 0.0f) continue;
            }

            RasterizeTriangle(wv0, wv1, wv2, faceNormal, mesh.material, camera, obj.tint);
        }
    };

    if (m_threadCount <= 1 || count <= 4) {
        for (int i = 0; i < count; ++i) {
            renderObject(i);
        }
    } else {
        ThreadPool::Get().ParallelFor(0, count, renderObject);
    }
}

void SoftwareRasterizer::RasterizeTriangle(const SoftVertex& v0, const SoftVertex& v1,
                                            const SoftVertex& v2, const Vec3& faceNormal,
                                            const SoftMaterial& material, const SoftCamera& camera,
                                            const SoftColorRGBA8& tint) {
    // Project to screen space
    Vec3 s0 = ProjectToScreen(v0.position, camera);
    Vec3 s1 = ProjectToScreen(v1.position, camera);
    Vec3 s2 = ProjectToScreen(v2.position, camera);

    // Skip triangles behind near plane
    if (s0.z <= camera.nearPlane && s1.z <= camera.nearPlane && s2.z <= camera.nearPlane)
        return;

    // Screen-space bounding box
    float minX = std::min({s0.x, s1.x, s2.x});
    float maxX = std::max({s0.x, s1.x, s2.x});
    float minY = std::min({s0.y, s1.y, s2.y});
    float maxY = std::max({s0.y, s1.y, s2.y});

    // Clip to viewport
    int startX = std::max(0, static_cast<int>(std::floor(minX)));
    int endX = std::min(m_config.width - 1, static_cast<int>(std::ceil(maxX)));
    int startY = std::max(0, static_cast<int>(std::floor(minY)));
    int endY = std::min(m_config.height - 1, static_cast<int>(std::ceil(maxY)));

    if (startX > endX || startY > endY) return;

    // Precompute edge function denominators
    float area = (s1.x - s0.x) * (s2.y - s0.y) - (s2.x - s0.x) * (s1.y - s0.y);
    if (std::abs(area) < 0.001f) return;  // Degenerate triangle
    float invArea = 1.0f / area;

    // Perspective-correct interpolation: 1/z values
    float invZ0 = 1.0f / s0.z;
    float invZ1 = 1.0f / s1.z;
    float invZ2 = 1.0f / s2.z;

    // --- Incremental barycentric stepping (10.5.2) ---
    // Edge function gradients: dw/dx and dw/dy are constant across the triangle.
    // This replaces per-pixel recomputation (12 muls + 6 subs) with 3 adds per pixel.
    float dw0_dx = (s1.y - s2.y) * invArea;
    float dw1_dx = (s2.y - s0.y) * invArea;
    float dw2_dx = (s0.y - s1.y) * invArea;
    float dw0_dy = (s2.x - s1.x) * invArea;
    float dw1_dy = (s0.x - s2.x) * invArea;
    float dw2_dy = (s1.x - s0.x) * invArea;

    // Compute initial barycentric coordinates at (startX + 0.5, startY + 0.5)
    float fx0 = static_cast<float>(startX) + 0.5f;
    float fy0 = static_cast<float>(startY) + 0.5f;
    float w0_row = ((s1.x - fx0) * (s2.y - fy0) - (s2.x - fx0) * (s1.y - fy0)) * invArea;
    float w1_row = ((s2.x - fx0) * (s0.y - fy0) - (s0.x - fx0) * (s2.y - fy0)) * invArea;
    float w2_row = ((s0.x - fx0) * (s1.y - fy0) - (s1.x - fx0) * (s0.y - fy0)) * invArea;

    // Scanline rasterization with incremental barycentrics
    for (int py = startY; py <= endY; ++py) {
        float w0 = w0_row;
        float w1 = w1_row;
        float w2 = w2_row;

        for (int px = startX; px <= endX; ++px) {
            // Inside test (all barycentrics >= 0)
            if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
                w0 += dw0_dx; w1 += dw1_dx; w2 += dw2_dx;
                continue;
            }

            // Perspective-correct depth
            float invZInterp = w0 * invZ0 + w1 * invZ1 + w2 * invZ2;
            float depth = 1.0f / invZInterp;

            // Depth test
            if (!m_gbuffer.DepthTest(px, py, depth)) continue;

            // Perspective-correct UV interpolation
            float u = (w0 * v0.uv.x * invZ0 + w1 * v1.uv.x * invZ1 + w2 * v2.uv.x * invZ2) * depth;
            float v = (w0 * v0.uv.y * invZ0 + w1 * v1.uv.y * invZ1 + w2 * v2.uv.y * invZ2) * depth;

            // Perspective-correct normal interpolation
            Vec3 normal;
            normal.x = (w0 * v0.normal.x * invZ0 + w1 * v1.normal.x * invZ1 + w2 * v2.normal.x * invZ2) * depth;
            normal.y = (w0 * v0.normal.y * invZ0 + w1 * v1.normal.y * invZ1 + w2 * v2.normal.y * invZ2) * depth;
            normal.z = (w0 * v0.normal.z * invZ0 + w1 * v1.normal.z * invZ1 + w2 * v2.normal.z * invZ2) * depth;
            normal = normal.Normalized();

            // Texture sampling
            SoftColorRGBA8 texColor = material.color;
            if (material.textureIndex >= 0 && material.textureIndex < static_cast<int>(m_textures.size())) {
                texColor = m_textures[material.textureIndex].Sample(u, v);
            }

            // Apply tint
            uint8_t finalR = static_cast<uint8_t>((texColor.r * tint.r) / 255);
            uint8_t finalG = static_cast<uint8_t>((texColor.g * tint.g) / 255);
            uint8_t finalB = static_cast<uint8_t>((texColor.b * tint.b) / 255);

            m_gbuffer.SetColor(px, py, {finalR, finalG, finalB, 255});

            // Pack normal to 0-255 range: (normal * 0.5 + 0.5) * 255
            uint8_t nx = static_cast<uint8_t>((normal.x * 0.5f + 0.5f) * 255.0f);
            uint8_t ny = static_cast<uint8_t>((normal.y * 0.5f + 0.5f) * 255.0f);
            uint8_t nz = static_cast<uint8_t>((normal.z * 0.5f + 0.5f) * 255.0f);
            m_gbuffer.SetNormal(px, py, {nx, ny, nz, 255});

            // Step barycentrics for next pixel
            w0 += dw0_dx; w1 += dw1_dx; w2 += dw2_dx;
        }

        // Step barycentrics for next scanline
        w0_row += dw0_dy;
        w1_row += dw1_dy;
        w2_row += dw2_dy;
    }
}

} // namespace WulfNet
