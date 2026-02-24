// =============================================================================
// WulfNet Engine - Software Rasterizer Types
// =============================================================================
// Core data types for the CPU software rasterizer with SIMD support.
// Ported from BG-C-Software-Renderer struct definitions.
// =============================================================================

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <array>

namespace WulfNet {

// =============================================================================
// Basic Math Types (self-contained, no external deps)
// =============================================================================

struct SoftVec2 {
    float x = 0.0f, y = 0.0f;
    SoftVec2() = default;
    SoftVec2(float x, float y) : x(x), y(y) {}
};

struct SoftVec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    SoftVec3() = default;
    SoftVec3(float x, float y, float z) : x(x), y(y), z(z) {}

    SoftVec3 operator+(const SoftVec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    SoftVec3 operator-(const SoftVec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    SoftVec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    SoftVec3 operator/(float s) const { float inv = 1.0f / s; return {x * inv, y * inv, z * inv}; }

    float Dot(const SoftVec3& o) const { return x * o.x + y * o.y + z * o.z; }
    SoftVec3 Cross(const SoftVec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    float Length() const { return std::sqrt(x * x + y * y + z * z); }
    float LengthSquared() const { return x * x + y * y + z * z; }
    SoftVec3 Normalized() const { float len = Length(); return len > 0.0f ? *this / len : SoftVec3(); }

    static SoftVec3 Lerp(const SoftVec3& a, const SoftVec3& b, float t) {
        return a + (b - a) * t;
    }
};

struct SoftVec4 {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
    SoftVec4() = default;
    SoftVec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    SoftVec4(const SoftVec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}
    SoftVec3 xyz() const { return {x, y, z}; }
};

// =============================================================================
// Color Types
// =============================================================================

struct SoftColorRGBA8 {
    uint8_t r = 0, g = 0, b = 0, a = 255;

    SoftColorRGBA8() = default;
    SoftColorRGBA8(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) : r(r), g(g), b(b), a(a) {}

    static SoftColorRGBA8 FromFloat(float fr, float fg, float fb, float fa = 1.0f) {
        auto clamp = [](float v) -> uint8_t {
            return static_cast<uint8_t>(v < 0.0f ? 0.0f : (v > 1.0f ? 255.0f : v * 255.0f));
        };
        return {clamp(fr), clamp(fg), clamp(fb), clamp(fa)};
    }

    uint32_t ToUint32() const {
        return static_cast<uint32_t>(r) |
               (static_cast<uint32_t>(g) << 8) |
               (static_cast<uint32_t>(b) << 16) |
               (static_cast<uint32_t>(a) << 24);
    }
};

// =============================================================================
// Vertex / Mesh / Material Types
// =============================================================================

struct SoftVertex {
    SoftVec3 position;
    SoftVec3 normal;
    SoftVec2 uv;
};

struct SoftMaterial {
    SoftColorRGBA8 color = {255, 255, 255, 255};
    float metalness = 0.0f;
    float roughness = 0.5f;
    int textureIndex = -1;  // -1 = no texture
};

struct SoftMesh {
    std::vector<SoftVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SoftVec3> faceNormals;  // Pre-computed per-triangle normals
    SoftMaterial material;
    std::string name;

    void ComputeFaceNormals() {
        size_t triCount = indices.size() / 3;
        faceNormals.resize(triCount);
        for (size_t i = 0; i < triCount; ++i) {
            const auto& v0 = vertices[indices[i * 3 + 0]].position;
            const auto& v1 = vertices[indices[i * 3 + 1]].position;
            const auto& v2 = vertices[indices[i * 3 + 2]].position;
            faceNormals[i] = (v1 - v0).Cross(v2 - v0).Normalized();
        }
    }
};

// =============================================================================
// Transform / Camera / Lights
// =============================================================================

struct SoftTransform {
    SoftVec3 position;
    SoftVec3 rotation;      // Euler angles in degrees
    SoftVec3 scale = {1.0f, 1.0f, 1.0f};
    int meshIndex = 0;
    SoftColorRGBA8 tint = {255, 255, 255, 255};
};

struct SoftCamera {
    SoftVec3 position;
    SoftVec3 forward = {0.0f, 0.0f, 1.0f};
    SoftVec3 up = {0.0f, 1.0f, 0.0f};
    SoftVec3 right = {1.0f, 0.0f, 0.0f};
    float fov = 60.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
    float aspectRatio = 16.0f / 9.0f;
};

struct SoftPointLight {
    SoftVec3 position;
    SoftVec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float range = 10.0f;
};

struct SoftDirectionalLight {
    SoftVec3 direction = {0.0f, -1.0f, 0.5f};
    SoftVec3 color = {1.0f, 0.95f, 0.9f};
    float intensity = 1.0f;
};

// =============================================================================
// Texture
// =============================================================================

struct SoftTexture {
    std::vector<SoftColorRGBA8> pixels;
    int width = 0;
    int height = 0;
    std::string name;

    SoftColorRGBA8 Sample(float u, float v) const {
        if (pixels.empty()) return {255, 255, 255, 255};

        // Wrap UVs
        u = u - std::floor(u);
        v = v - std::floor(v);

        int px = static_cast<int>(u * (width - 1));
        int py = static_cast<int>(v * (height - 1));
        px = px < 0 ? 0 : (px >= width ? width - 1 : px);
        py = py < 0 ? 0 : (py >= height ? height - 1 : py);

        return pixels[py * width + px];
    }
};

// =============================================================================
// Simple Mesh Generators
// =============================================================================

namespace SoftMeshGen {

inline SoftMesh CreateCube(float size = 1.0f) {
    SoftMesh mesh;
    mesh.name = "Cube";
    float h = size * 0.5f;

    // 8 corner positions
    SoftVec3 corners[8] = {
        {-h, -h, -h}, { h, -h, -h}, { h,  h, -h}, {-h,  h, -h},
        {-h, -h,  h}, { h, -h,  h}, { h,  h,  h}, {-h,  h,  h}
    };

    // 6 faces, each with 4 vertices and a normal
    struct Face { int v[4]; SoftVec3 normal; };
    Face faces[6] = {
        {{0, 1, 2, 3}, { 0,  0, -1}}, // front
        {{5, 4, 7, 6}, { 0,  0,  1}}, // back
        {{4, 0, 3, 7}, {-1,  0,  0}}, // left
        {{1, 5, 6, 2}, { 1,  0,  0}}, // right
        {{3, 2, 6, 7}, { 0,  1,  0}}, // top
        {{4, 5, 1, 0}, { 0, -1,  0}}, // bottom
    };

    SoftVec2 uvs[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

    for (int f = 0; f < 6; ++f) {
        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        for (int v = 0; v < 4; ++v) {
            mesh.vertices.push_back({corners[faces[f].v[v]], faces[f].normal, uvs[v]});
        }
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 3);
    }

    mesh.ComputeFaceNormals();
    return mesh;
}

inline SoftMesh CreateSphere(float radius = 0.5f, int segments = 16, int rings = 16) {
    SoftMesh mesh;
    mesh.name = "Sphere";

    for (int r = 0; r <= rings; ++r) {
        float phi = 3.14159265f * float(r) / float(rings);
        for (int s = 0; s <= segments; ++s) {
            float theta = 2.0f * 3.14159265f * float(s) / float(segments);

            SoftVec3 pos = {
                radius * std::sin(phi) * std::cos(theta),
                radius * std::cos(phi),
                radius * std::sin(phi) * std::sin(theta)
            };
            SoftVec3 normal = pos.Normalized();
            SoftVec2 uv = {float(s) / float(segments), float(r) / float(rings)};

            mesh.vertices.push_back({pos, normal, uv});
        }
    }

    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < segments; ++s) {
            uint32_t a = r * (segments + 1) + s;
            uint32_t b = a + segments + 1;

            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(a + 1);

            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(b);
            mesh.indices.push_back(b + 1);
        }
    }

    mesh.ComputeFaceNormals();
    return mesh;
}

} // namespace SoftMeshGen

} // namespace WulfNet
