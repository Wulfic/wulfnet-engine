// =============================================================================
// WulfNet Engine - Shadow Mapping Implementation
// =============================================================================

#include "WulfNet/Rendering/SoftwareRasterizer/ShadowMap.h"
#include <cstring>
#include <limits>

namespace WulfNet {

// =============================================================================
// Helper functions
// =============================================================================

static float ShadowClamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

static constexpr float kPi = 3.14159265358979323846f;

/// Rotate a vector by Euler angles (degrees) in ZYX order (matching SoftwareRasterizer)
static SoftVec3 RotateEuler(const SoftVec3& v, const SoftVec3& eulerDeg) {
    float rx = eulerDeg.x * kPi / 180.0f;
    float ry = eulerDeg.y * kPi / 180.0f;
    float rz = eulerDeg.z * kPi / 180.0f;

    float cx = std::cos(rx), sx = std::sin(rx);
    float cy = std::cos(ry), sy = std::sin(ry);
    float cz = std::cos(rz), sz = std::sin(rz);

    // ZYX rotation matrix
    SoftVec3 result;
    result.x = (cy * cz) * v.x + (sx * sy * cz - cx * sz) * v.y + (cx * sy * cz + sx * sz) * v.z;
    result.y = (cy * sz) * v.x + (sx * sy * sz + cx * cz) * v.y + (cx * sy * sz - sx * cz) * v.z;
    result.z = (-sy)     * v.x + (sx * cy)                * v.y + (cx * cy)                * v.z;
    return result;
}

// =============================================================================
// ShadowCascade
// =============================================================================

bool ShadowCascade::Initialize(int resolution) {
    if (resolution <= 0) return false;
    m_resolution = resolution;
    m_depthBuffer.resize(resolution * resolution, std::numeric_limits<float>::max());
    return true;
}

void ShadowCascade::ComputeLightMatrix(const SoftVec3& lightDir, const SoftVec3& focusCenter,
                                        float orthoSize, float nearClip, float farClip) {
    m_orthoSize = orthoSize;
    m_nearClip = nearClip;
    m_farClip = farClip;

    // Light forward = normalized light direction
    m_lightForward = lightDir.Normalized();

    // Choose an 'up' hint that isn't parallel to light direction
    SoftVec3 upHint = {0.0f, 1.0f, 0.0f};
    if (std::abs(m_lightForward.Dot(upHint)) > 0.99f) {
        upHint = {1.0f, 0.0f, 0.0f};
    }

    // Build orthonormal basis
    m_lightRight = m_lightForward.Cross(upHint).Normalized();
    m_lightUp = m_lightRight.Cross(m_lightForward).Normalized();

    // Position the light "behind" the focus center along the light direction
    m_lightPosition = focusCenter - m_lightForward * (farClip * 0.5f);
}

void ShadowCascade::Clear() {
    std::fill(m_depthBuffer.begin(), m_depthBuffer.end(), std::numeric_limits<float>::max());
}

SoftVec3 ShadowCascade::WorldToLightNDC(const SoftVec3& worldPos) const {
    // Transform to light space
    SoftVec3 toPoint = worldPos - m_lightPosition;
    float lightX = toPoint.Dot(m_lightRight);
    float lightY = toPoint.Dot(m_lightUp);
    float lightZ = toPoint.Dot(m_lightForward);

    // Orthographic projection to NDC [-1, 1]
    float ndcX = lightX / m_orthoSize;
    float ndcY = lightY / m_orthoSize;
    // Depth normalized to [0, 1]
    float ndcZ = (lightZ - m_nearClip) / (m_farClip - m_nearClip);

    return {ndcX, ndcY, ndcZ};
}

bool ShadowCascade::WriteDepth(float ndcX, float ndcY, float depth) {
    // Convert NDC [-1,1] to pixel coordinates
    float px = (ndcX * 0.5f + 0.5f) * m_resolution;
    float py = (ndcY * 0.5f + 0.5f) * m_resolution;

    int ix = static_cast<int>(px);
    int iy = static_cast<int>(py);

    if (ix < 0 || ix >= m_resolution || iy < 0 || iy >= m_resolution) return false;
    if (depth < 0.0f || depth > 1.0f) return false;

    int idx = iy * m_resolution + ix;
    if (depth < m_depthBuffer[idx]) {
        m_depthBuffer[idx] = depth;
        return true;
    }
    return false;
}

float ShadowCascade::SampleShadow(const SoftVec3& worldPos, float bias) const {
    SoftVec3 ndc = WorldToLightNDC(worldPos);

    // Outside cascade frustum = lit
    if (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f) return 1.0f;
    if (ndc.z < 0.0f || ndc.z > 1.0f) return 1.0f;

    // Convert to texel coords
    float px = (ndc.x * 0.5f + 0.5f) * m_resolution;
    float py = (ndc.y * 0.5f + 0.5f) * m_resolution;

    int ix = static_cast<int>(px);
    int iy = static_cast<int>(py);
    ix = std::max(0, std::min(ix, m_resolution - 1));
    iy = std::max(0, std::min(iy, m_resolution - 1));

    float storedDepth = m_depthBuffer[iy * m_resolution + ix];
    float currentDepth = ndc.z;

    // Depth comparison with bias
    return (currentDepth - bias <= storedDepth) ? 1.0f : 0.0f;
}

// =============================================================================
// PointLightShadow
// =============================================================================

bool PointLightShadow::Initialize(int resolution) {
    if (resolution <= 0) return false;
    m_resolution = resolution;

    // +X, -X, +Y, -Y, +Z, -Z
    SoftVec3 forwards[6] = {
        { 1, 0, 0}, {-1, 0, 0},
        { 0, 1, 0}, { 0,-1, 0},
        { 0, 0, 1}, { 0, 0,-1}
    };
    SoftVec3 ups[6] = {
        {0, 1, 0}, {0, 1, 0},
        {0, 0,-1}, {0, 0, 1},
        {0, 1, 0}, {0, 1, 0}
    };

    for (int i = 0; i < 6; ++i) {
        m_faces[i].forward = forwards[i];
        m_faces[i].up = ups[i];
        m_faces[i].right = forwards[i].Cross(ups[i]).Normalized();
        m_faces[i].depthBuffer.resize(resolution * resolution, std::numeric_limits<float>::max());
    }

    return true;
}

void PointLightShadow::Clear() {
    for (int i = 0; i < 6; ++i) {
        std::fill(m_faces[i].depthBuffer.begin(), m_faces[i].depthBuffer.end(),
                  std::numeric_limits<float>::max());
    }
}

void PointLightShadow::SetLightPosition(const SoftVec3& position, float range) {
    m_position = position;
    m_range = range;
}

int PointLightShadow::DirectionToFace(const SoftVec3& dir) {
    float absX = std::abs(dir.x);
    float absY = std::abs(dir.y);
    float absZ = std::abs(dir.z);

    if (absX >= absY && absX >= absZ) {
        return dir.x > 0.0f ? 0 : 1; // +X or -X
    }
    if (absY >= absX && absY >= absZ) {
        return dir.y > 0.0f ? 2 : 3; // +Y or -Y
    }
    return dir.z > 0.0f ? 4 : 5;     // +Z or -Z
}

SoftVec3 PointLightShadow::WorldToFaceNDC(const SoftVec3& worldPos, int face) const {
    if (face < 0 || face >= 6) return {0, 0, 0};

    SoftVec3 toPoint = worldPos - m_position;
    const CubeFace& f = m_faces[face];

    // Project onto face basis
    float fwdDist = toPoint.Dot(f.forward);
    if (fwdDist <= 0.0f) return {0, 0, -1}; // Behind the face

    float rightDist = toPoint.Dot(f.right);
    float upDist = toPoint.Dot(f.up);

    // Perspective projection (90-degree FOV per face)
    float ndcX = rightDist / fwdDist;
    float ndcY = upDist / fwdDist;
    float ndcZ = fwdDist / m_range; // Normalize to [0,1] by range

    return {ndcX, ndcY, ndcZ};
}

bool PointLightShadow::WriteDepth(int face, float ndcX, float ndcY, float depth) {
    if (face < 0 || face >= 6) return false;

    // Convert NDC [-1,1] to pixel
    float px = (ndcX * 0.5f + 0.5f) * m_resolution;
    float py = (ndcY * 0.5f + 0.5f) * m_resolution;

    int ix = static_cast<int>(px);
    int iy = static_cast<int>(py);

    if (ix < 0 || ix >= m_resolution || iy < 0 || iy >= m_resolution) return false;
    if (depth < 0.0f || depth > 1.0f) return false;

    int idx = iy * m_resolution + ix;
    if (depth < m_faces[face].depthBuffer[idx]) {
        m_faces[face].depthBuffer[idx] = depth;
        return true;
    }
    return false;
}

float PointLightShadow::SampleShadow(const SoftVec3& worldPos, float bias) const {
    SoftVec3 toPoint = worldPos - m_position;
    float dist = toPoint.Length();

    if (dist > m_range || dist < 0.001f) return 1.0f;

    SoftVec3 dir = toPoint / dist;
    int face = DirectionToFace(dir);

    SoftVec3 ndc = WorldToFaceNDC(worldPos, face);

    // Outside face frustum
    if (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f) return 1.0f;
    if (ndc.z < 0.0f || ndc.z > 1.0f) return 1.0f;

    // Sample depth buffer
    float px = (ndc.x * 0.5f + 0.5f) * m_resolution;
    float py = (ndc.y * 0.5f + 0.5f) * m_resolution;

    int ix = std::max(0, std::min(static_cast<int>(px), m_resolution - 1));
    int iy = std::max(0, std::min(static_cast<int>(py), m_resolution - 1));

    float storedDepth = m_faces[face].depthBuffer[iy * m_resolution + ix];
    float currentDepth = ndc.z;

    return (currentDepth - bias <= storedDepth) ? 1.0f : 0.0f;
}

const float* PointLightShadow::GetFaceDepthBuffer(int face) const {
    if (face < 0 || face >= 6) return nullptr;
    return m_faces[face].depthBuffer.data();
}

// =============================================================================
// ShadowSystem
// =============================================================================

bool ShadowSystem::Initialize(const ShadowSystemConfig& config) {
    m_config = config;

    // Initialize cascades
    m_cascades.resize(config.numCascades);
    for (int i = 0; i < config.numCascades; ++i) {
        if (!m_cascades[i].Initialize(config.cascadeResolution)) return false;
    }

    // Pre-allocate cascade splits
    m_cascadeSplits.resize(config.numCascades + 1, 0.0f);

    return true;
}

void ShadowSystem::ComputeCascadeSplits(const SoftCamera& camera) {
    m_cachedCameraPos = camera.position;
    m_cachedCameraFwd = camera.forward;

    float nearClip = camera.nearPlane;
    float farClip = std::min(camera.farPlane, m_config.maxShadowDistance);
    float lambda = m_config.cascadeSplitLambda;
    int n = m_config.numCascades;

    m_cascadeSplits[0] = nearClip;

    for (int i = 1; i <= n; ++i) {
        float p = static_cast<float>(i) / static_cast<float>(n);

        // Logarithmic split
        float logSplit = nearClip * std::pow(farClip / nearClip, p);
        // Linear split
        float linearSplit = nearClip + (farClip - nearClip) * p;
        // Blend
        float split = lambda * logSplit + (1.0f - lambda) * linearSplit;

        m_cascadeSplits[i] = split;
    }
}

int ShadowSystem::SelectCascade(const SoftVec3& worldPos, const SoftCamera& camera) const {
    // Distance from camera along the camera's forward axis
    SoftVec3 toPoint = worldPos - camera.position;
    float viewDist = toPoint.Dot(camera.forward);

    for (int i = 0; i < static_cast<int>(m_cascades.size()); ++i) {
        if (viewDist < m_cascadeSplits[i + 1]) {
            return i;
        }
    }
    return static_cast<int>(m_cascades.size()) - 1;
}

void ShadowSystem::ClearAll() {
    for (auto& cascade : m_cascades) cascade.Clear();
    for (auto& shadow : m_pointShadows)  shadow.Clear();
}

SoftVec3 ShadowSystem::TransformPoint(const SoftVec3& point, const SoftTransform& transform) const {
    // Scale
    SoftVec3 scaled = {
        point.x * transform.scale.x,
        point.y * transform.scale.y,
        point.z * transform.scale.z
    };

    // Rotate (Euler ZYX)
    SoftVec3 rotated = RotateEuler(scaled, transform.rotation);

    // Translate
    return rotated + transform.position;
}

void ShadowSystem::RasterizeShadowTriangle(ShadowCascade& cascade,
                                             const SoftVec3& v0, const SoftVec3& v1, const SoftVec3& v2) {
    // Project all 3 vertices to light NDC
    SoftVec3 ndc0 = cascade.WorldToLightNDC(v0);
    SoftVec3 ndc1 = cascade.WorldToLightNDC(v1);
    SoftVec3 ndc2 = cascade.WorldToLightNDC(v2);

    // Early reject if all vertices are outside
    if ((ndc0.z < 0 && ndc1.z < 0 && ndc2.z < 0) ||
        (ndc0.z > 1 && ndc1.z > 1 && ndc2.z > 1)) return;

    // Convert NDC to screen coords
    int res = cascade.GetResolution();
    float s0x = (ndc0.x * 0.5f + 0.5f) * res;
    float s0y = (ndc0.y * 0.5f + 0.5f) * res;
    float s1x = (ndc1.x * 0.5f + 0.5f) * res;
    float s1y = (ndc1.y * 0.5f + 0.5f) * res;
    float s2x = (ndc2.x * 0.5f + 0.5f) * res;
    float s2y = (ndc2.y * 0.5f + 0.5f) * res;

    // Bounding box
    int minX = std::max(0, static_cast<int>(std::floor(std::min({s0x, s1x, s2x}))));
    int maxX = std::min(res - 1, static_cast<int>(std::ceil(std::max({s0x, s1x, s2x}))));
    int minY = std::max(0, static_cast<int>(std::floor(std::min({s0y, s1y, s2y}))));
    int maxY = std::min(res - 1, static_cast<int>(std::ceil(std::max({s0y, s1y, s2y}))));

    // Rasterize with barycentric coordinates
    float denom = (s1y - s2y) * (s0x - s2x) + (s2x - s1x) * (s0y - s2y);
    if (std::abs(denom) < 1e-8f) return; // Degenerate triangle

    float invDenom = 1.0f / denom;

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            float px = x + 0.5f;
            float py = y + 0.5f;

            float w0 = ((s1y - s2y) * (px - s2x) + (s2x - s1x) * (py - s2y)) * invDenom;
            float w1 = ((s2y - s0y) * (px - s2x) + (s0x - s2x) * (py - s2y)) * invDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
                float depth = w0 * ndc0.z + w1 * ndc1.z + w2 * ndc2.z;
                depth = ShadowClamp01(depth);
                cascade.WriteDepth(
                    (px / res) * 2.0f - 1.0f,
                    (py / res) * 2.0f - 1.0f,
                    depth
                );
            }
        }
    }
}

void ShadowSystem::RasterizeShadowTriangleFace(PointLightShadow& shadow, int face,
                                                  const SoftVec3& v0, const SoftVec3& v1, const SoftVec3& v2) {
    SoftVec3 ndc0 = shadow.WorldToFaceNDC(v0, face);
    SoftVec3 ndc1 = shadow.WorldToFaceNDC(v1, face);
    SoftVec3 ndc2 = shadow.WorldToFaceNDC(v2, face);

    // Early reject
    if ((ndc0.z < 0 && ndc1.z < 0 && ndc2.z < 0) ||
        (ndc0.z > 1 && ndc1.z > 1 && ndc2.z > 1)) return;

    int res = shadow.GetResolution();
    float s0x = (ndc0.x * 0.5f + 0.5f) * res;
    float s0y = (ndc0.y * 0.5f + 0.5f) * res;
    float s1x = (ndc1.x * 0.5f + 0.5f) * res;
    float s1y = (ndc1.y * 0.5f + 0.5f) * res;
    float s2x = (ndc2.x * 0.5f + 0.5f) * res;
    float s2y = (ndc2.y * 0.5f + 0.5f) * res;

    int minX = std::max(0, static_cast<int>(std::floor(std::min({s0x, s1x, s2x}))));
    int maxX = std::min(res - 1, static_cast<int>(std::ceil(std::max({s0x, s1x, s2x}))));
    int minY = std::max(0, static_cast<int>(std::floor(std::min({s0y, s1y, s2y}))));
    int maxY = std::min(res - 1, static_cast<int>(std::ceil(std::max({s0y, s1y, s2y}))));

    float denom = (s1y - s2y) * (s0x - s2x) + (s2x - s1x) * (s0y - s2y);
    if (std::abs(denom) < 1e-8f) return;

    float invDenom = 1.0f / denom;

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            float px = x + 0.5f;
            float py = y + 0.5f;

            float w0 = ((s1y - s2y) * (px - s2x) + (s2x - s1x) * (py - s2y)) * invDenom;
            float w1 = ((s2y - s0y) * (px - s2x) + (s0x - s2x) * (py - s2y)) * invDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
                float depth = w0 * ndc0.z + w1 * ndc1.z + w2 * ndc2.z;
                depth = ShadowClamp01(depth);
                shadow.WriteDepth(face,
                    (px / res) * 2.0f - 1.0f,
                    (py / res) * 2.0f - 1.0f,
                    depth
                );
            }
        }
    }
}

void ShadowSystem::RenderDirectionalShadows(const SoftDirectionalLight& light,
                                              const SoftCamera& camera,
                                              const SoftTransform* transforms, int transformCount,
                                              const std::vector<SoftMesh>& meshes) {
    ComputeCascadeSplits(camera);

    SoftVec3 lightDir = light.direction.Normalized();

    // Set up each cascade's light matrix
    for (int c = 0; c < static_cast<int>(m_cascades.size()); ++c) {
        float nearDist = m_cascadeSplits[c];
        float farDist = m_cascadeSplits[c + 1];
        float midDist = (nearDist + farDist) * 0.5f;

        // Focus center = camera position + forward * midDist
        SoftVec3 focusCenter = camera.position + camera.forward * midDist;

        // Ortho size scales with cascade distance
        float orthoSize = farDist * 0.6f;

        m_cascades[c].Clear();
        m_cascades[c].ComputeLightMatrix(lightDir, focusCenter, orthoSize, 0.1f, farDist * 2.0f);
    }

    // Rasterize each triangle of each transform into the appropriate cascades
    for (int t = 0; t < transformCount; ++t) {
        const SoftTransform& xform = transforms[t];
        if (xform.meshIndex < 0 || xform.meshIndex >= static_cast<int>(meshes.size())) continue;

        const SoftMesh& mesh = meshes[xform.meshIndex];
        const auto& indices = mesh.indices;
        const auto& verts = mesh.vertices;

        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            SoftVec3 wv0 = TransformPoint(verts[indices[i + 0]].position, xform);
            SoftVec3 wv1 = TransformPoint(verts[indices[i + 1]].position, xform);
            SoftVec3 wv2 = TransformPoint(verts[indices[i + 2]].position, xform);

            // Insert into all cascades that overlap this triangle
            for (int c = 0; c < static_cast<int>(m_cascades.size()); ++c) {
                RasterizeShadowTriangle(m_cascades[c], wv0, wv1, wv2);
            }
        }
    }
}

void ShadowSystem::RenderPointLightShadow(int lightIndex,
                                            const SoftPointLight& light,
                                            const SoftTransform* transforms, int transformCount,
                                            const std::vector<SoftMesh>& meshes) {
    // Ensure we have enough point light shadows
    if (lightIndex >= static_cast<int>(m_pointShadows.size())) {
        m_pointShadows.resize(lightIndex + 1);
    }

    auto& shadow = m_pointShadows[lightIndex];
    if (shadow.GetResolution() == 0) {
        shadow.Initialize(m_config.pointLightResolution);
    }

    shadow.Clear();
    shadow.SetLightPosition(light.position, light.range);

    // Rasterize each triangle into all 6 faces
    for (int t = 0; t < transformCount; ++t) {
        const SoftTransform& xform = transforms[t];
        if (xform.meshIndex < 0 || xform.meshIndex >= static_cast<int>(meshes.size())) continue;

        const SoftMesh& mesh = meshes[xform.meshIndex];
        const auto& indices = mesh.indices;
        const auto& verts = mesh.vertices;

        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            SoftVec3 wv0 = TransformPoint(verts[indices[i + 0]].position, xform);
            SoftVec3 wv1 = TransformPoint(verts[indices[i + 1]].position, xform);
            SoftVec3 wv2 = TransformPoint(verts[indices[i + 2]].position, xform);

            // Determine which faces this triangle could affect
            SoftVec3 triCenter = (wv0 + wv1 + wv2) * (1.0f / 3.0f);
            SoftVec3 toTri = triCenter - light.position;
            float dist = toTri.Length();

            if (dist > light.range) continue;

            // Rasterize to the primary face and adjacent faces
            int primaryFace = PointLightShadow::DirectionToFace(toTri);
            RasterizeShadowTriangleFace(shadow, primaryFace, wv0, wv1, wv2);

            // Also try adjacent faces for triangles near edges
            for (int f = 0; f < 6; ++f) {
                if (f != primaryFace) {
                    RasterizeShadowTriangleFace(shadow, f, wv0, wv1, wv2);
                }
            }
        }
    }
}

float ShadowSystem::SampleDirectionalShadow(const SoftVec3& worldPos) const {
    if (m_cascades.empty()) return 1.0f;

    // Select cascade based on view distance
    SoftVec3 toPoint = worldPos - m_cachedCameraPos;
    float viewDist = toPoint.Dot(m_cachedCameraFwd);

    int cascadeIdx = -1;
    for (int i = 0; i < static_cast<int>(m_cascades.size()); ++i) {
        if (i + 1 < static_cast<int>(m_cascadeSplits.size()) && viewDist < m_cascadeSplits[i + 1]) {
            cascadeIdx = i;
            break;
        }
    }

    if (cascadeIdx < 0) cascadeIdx = static_cast<int>(m_cascades.size()) - 1;

    return m_cascades[cascadeIdx].SampleShadow(worldPos, m_config.shadowBias);
}

float ShadowSystem::SamplePointLightShadow(int lightIndex, const SoftVec3& worldPos) const {
    if (lightIndex < 0 || lightIndex >= static_cast<int>(m_pointShadows.size())) return 1.0f;
    return m_pointShadows[lightIndex].SampleShadow(worldPos, m_config.shadowBias);
}

} // namespace WulfNet
