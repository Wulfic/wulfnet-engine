// =============================================================================
// WulfNet Engine - Affine Transform Builder
// =============================================================================
// CPU-side affine matrix construction for IFS fractal systems.
// Ported from AffineTransformations.cs (Unity IFS reference).
// =============================================================================

#pragma once

#include <cstdint>
#include <cmath>
#include <array>
#include <vector>

namespace WulfNet {

// =============================================================================
// GPU-compatible 4x4 matrix (row-major, flat float[16])
// =============================================================================

struct GPUMat4x4 {
    float m[16];

    static GPUMat4x4 Identity() {
        GPUMat4x4 mat = {};
        mat.m[0] = 1.0f; mat.m[5] = 1.0f; mat.m[10] = 1.0f; mat.m[15] = 1.0f;
        return mat;
    }

    // Row-major: row r, column c -> m[r*4 + c]
    float& At(int r, int c) { return m[r * 4 + c]; }
    float At(int r, int c) const { return m[r * 4 + c]; }

    void SetRow(int r, float x, float y, float z, float w) {
        m[r * 4 + 0] = x; m[r * 4 + 1] = y; m[r * 4 + 2] = z; m[r * 4 + 3] = w;
    }

    GPUMat4x4 operator*(const GPUMat4x4& other) const {
        GPUMat4x4 result = {};
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    sum += At(r, k) * other.At(k, c);
                }
                result.At(r, c) = sum;
            }
        }
        return result;
    }
};

// =============================================================================
// Transform Instructions (matches Unity TransformSet.TransformInstructions)
// =============================================================================

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3() = default;
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }

    static Vec3 Lerp(const Vec3& a, const Vec3& b, float t) {
        return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
    }

    static Vec3 Scale(const Vec3& a, const Vec3& b) {
        return {a.x * b.x, a.y * b.y, a.z * b.z};
    }
};

struct TransformInstructions {
    Vec3 scale = {1.0f, 1.0f, 1.0f};
    Vec3 shearX;
    Vec3 shearY;
    Vec3 shearZ;
    Vec3 rotate;
    Vec3 translate;

    static TransformInstructions Identity() {
        TransformInstructions t;
        t.scale = {1.0f, 1.0f, 1.0f};
        return t;
    }

    TransformInstructions operator+(const TransformInstructions& b) const {
        TransformInstructions result;
        result.scale = Vec3::Scale(scale, b.scale);
        result.shearX = shearX + b.shearX;
        result.shearY = shearY + b.shearY;
        result.shearZ = shearZ + b.shearZ;
        result.rotate = rotate + b.rotate;  // simplified, original uses quaternion multiply
        result.translate = translate + b.translate;
        return result;
    }
};

// =============================================================================
// Affine Transform Builder
// =============================================================================

namespace AffineTransform {

constexpr float DEG2RAD = 3.14159265358979f / 180.0f;

GPUMat4x4 MakeScale(const Vec3& s);
GPUMat4x4 MakeShearX(const Vec3& s);
GPUMat4x4 MakeShearY(const Vec3& s);
GPUMat4x4 MakeShearZ(const Vec3& s);
GPUMat4x4 MakeTranslate(const Vec3& t);
GPUMat4x4 MakeRotation(const Vec3& eulerDegrees);

/// Build a complete affine matrix from transform instructions
/// Order: scale * rotation * shear * translate
GPUMat4x4 FromInstructions(const TransformInstructions& inst);

/// Linear interpolation between two affine matrices (row-by-row lerp)
GPUMat4x4 Interpolate(const GPUMat4x4& m1, const GPUMat4x4& m2, float t);

} // namespace AffineTransform

} // namespace WulfNet
