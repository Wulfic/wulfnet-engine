// =============================================================================
// WulfNet Engine - Unified Math Types
// =============================================================================
// Canonical vector, matrix, and quaternion types for the entire engine.
// All modules should use these types instead of defining their own.
//
// Design:
//   - GPU-compatible layouts (alignas(16) for Vec4/Mat4 where appropriate)
//   - Row-major matrix storage (consistent with existing codebase convention)
//   - Jolt-compatible conversions via free functions (not in this header)
//   - Superset of all previously-separate math types
// =============================================================================

#pragma once

#include <cmath>
#include <cstdint>

// SIMD headers — guarded by build-system defines
#if defined(WULFNET_HAS_SSE2) || defined(WULFNET_HAS_AVX2)
    #include <immintrin.h>
    #define WULFNET_SIMD_SSE 1
#endif

namespace WulfNet {

// =============================================================================
// Vec2 — 2D vector
// =============================================================================

struct Vec2 {
    float x = 0.0f, y = 0.0f;

    Vec2() = default;
    Vec2(float x, float y) : x(x), y(y) {}

    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(float s) const { return {x * s, y * s}; }
    Vec2 operator/(float s) const { float inv = 1.0f / s; return {x * inv, y * inv}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
    Vec2& operator*=(float s) { x *= s; y *= s; return *this; }

    float Dot(const Vec2& o) const { return x * o.x + y * o.y; }
    float Length() const { return std::sqrt(x * x + y * y); }
    float LengthSquared() const { return x * x + y * y; }
    Vec2 Normalized() const {
        float len = Length();
        return len > 0.0f ? *this / len : Vec2();
    }

    static Vec2 Lerp(const Vec2& a, const Vec2& b, float t) {
        return a + (b - a) * t;
    }
};

// =============================================================================
// Vec3 — 3D vector
// =============================================================================

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3() = default;
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { float inv = 1.0f / s; return {x * inv, y * inv, z * inv}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    float Dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }

    Vec3 Cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    float Length() const { return std::sqrt(x * x + y * y + z * z); }
    float LengthSquared() const { return x * x + y * y + z * z; }

    Vec3 Normalized() const {
        float len = Length();
        return len > 0.0f ? *this / len : Vec3();
    }

    static Vec3 Lerp(const Vec3& a, const Vec3& b, float t) {
        return a + (b - a) * t;
    }

    /// Component-wise multiply (used by IFS TransformInstructions)
    static Vec3 Scale(const Vec3& a, const Vec3& b) {
        return {a.x * b.x, a.y * b.y, a.z * b.z};
    }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

// =============================================================================
// Vec4 — 4D vector (GPU-compatible alignment, optional SSE backing)
// =============================================================================

struct alignas(16) Vec4 {
#if WULFNET_SIMD_SSE
    union {
        __m128 simd;
        struct { float x, y, z, w; };
    };

    Vec4() : simd(_mm_setzero_ps()) {}
    Vec4(float x, float y, float z, float w) : simd(_mm_set_ps(w, z, y, x)) {}
    Vec4(const Vec3& v, float w) : simd(_mm_set_ps(w, v.z, v.y, v.x)) {}
    Vec4(__m128 v) : simd(v) {}
#else
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;

    Vec4() = default;
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    Vec4(const Vec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}
#endif

    Vec3 xyz() const { return {x, y, z}; }

#if WULFNET_SIMD_SSE
    Vec4 operator+(const Vec4& o) const { return Vec4(_mm_add_ps(simd, o.simd)); }
    Vec4 operator-(const Vec4& o) const { return Vec4(_mm_sub_ps(simd, o.simd)); }
    Vec4 operator*(float s) const { return Vec4(_mm_mul_ps(simd, _mm_set1_ps(s))); }

    float Dot(const Vec4& o) const {
        __m128 mul = _mm_mul_ps(simd, o.simd);
        // Horizontal add: (a+b, c+d, a+b, c+d) then (sum, sum, sum, sum)
        __m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(mul, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }
#else
    Vec4 operator+(const Vec4& o) const { return {x + o.x, y + o.y, z + o.z, w + o.w}; }
    Vec4 operator-(const Vec4& o) const { return {x - o.x, y - o.y, z - o.z, w - o.w}; }
    Vec4 operator*(float s) const { return {x * s, y * s, z * s, w * s}; }

    float Dot(const Vec4& o) const { return x * o.x + y * o.y + z * o.z + w * o.w; }
#endif
};

// =============================================================================
// Mat3 — 3×3 matrix (row-major)
// =============================================================================
// Primary use: deformation gradients, stress tensors, rotation matrices (MPM)

struct Mat3 {
    float m[3][3] = {};

    static Mat3 Identity() {
        Mat3 I;
        I.m[0][0] = I.m[1][1] = I.m[2][2] = 1.0f;
        return I;
    }

    static Mat3 Zero() { return Mat3{}; }

    Mat3 operator+(const Mat3& b) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][j] + b.m[i][j];
        return r;
    }

    Mat3 operator-(const Mat3& b) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][j] - b.m[i][j];
        return r;
    }

    Mat3 operator*(float s) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][j] * s;
        return r;
    }

    Mat3 operator*(const Mat3& b) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                r.m[i][j] = 0.0f;
                for (int k = 0; k < 3; ++k)
                    r.m[i][j] += m[i][k] * b.m[k][j];
            }
        return r;
    }

    Mat3 Transpose() const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[j][i];
        return r;
    }

    float Determinant() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    float Trace() const {
        return m[0][0] + m[1][1] + m[2][2];
    }

    float FrobeniusNorm() const {
        float sum = 0.0f;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                sum += m[i][j] * m[i][j];
        return std::sqrt(sum);
    }

    // Inverse via cofactor method (safe for singular matrices — returns identity)
    Mat3 Inverse() const {
        float det = Determinant();
        if (std::abs(det) < 1e-12f) return Identity();

        float invDet = 1.0f / det;
        Mat3 r;
        r.m[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * invDet;
        r.m[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet;
        r.m[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet;
        r.m[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * invDet;
        r.m[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet;
        r.m[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * invDet;
        r.m[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * invDet;
        r.m[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * invDet;
        r.m[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * invDet;
        return r;
    }

    // Inverse-Transpose (needed for Piola-Kirchhoff stress)
    Mat3 InverseTranspose() const {
        return Inverse().Transpose();
    }
};

// =============================================================================
// Mat4 — 4×4 matrix (row-major, GPU-compatible flat float[16])
// =============================================================================
// Primary use: affine transforms, projection, GPU uniform data

struct Mat4 {
    float m[16] = {};

    static Mat4 Identity() {
        Mat4 mat;
        mat.m[0] = 1.0f; mat.m[5] = 1.0f; mat.m[10] = 1.0f; mat.m[15] = 1.0f;
        return mat;
    }

    static Mat4 Zero() { return Mat4{}; }

    // Row-major: row r, column c -> m[r*4 + c]
    float& At(int r, int c) { return m[r * 4 + c]; }
    float At(int r, int c) const { return m[r * 4 + c]; }

    void SetRow(int r, float x, float y, float z, float w) {
        m[r * 4 + 0] = x; m[r * 4 + 1] = y; m[r * 4 + 2] = z; m[r * 4 + 3] = w;
    }

    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
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

    Mat4 operator*(float s) const {
        Mat4 r;
        for (int i = 0; i < 16; ++i) r.m[i] = m[i] * s;
        return r;
    }

    Mat4 operator+(const Mat4& b) const {
        Mat4 r;
        for (int i = 0; i < 16; ++i) r.m[i] = m[i] + b.m[i];
        return r;
    }

    /// Transform a Vec3 as a point (w=1), returning the result with perspective divide
    Vec3 TransformPoint(const Vec3& v) const {
        float rx = At(0,0)*v.x + At(0,1)*v.y + At(0,2)*v.z + At(0,3);
        float ry = At(1,0)*v.x + At(1,1)*v.y + At(1,2)*v.z + At(1,3);
        float rz = At(2,0)*v.x + At(2,1)*v.y + At(2,2)*v.z + At(2,3);
        float rw = At(3,0)*v.x + At(3,1)*v.y + At(3,2)*v.z + At(3,3);
        if (std::abs(rw) > 1e-12f) {
            float inv = 1.0f / rw;
            return {rx * inv, ry * inv, rz * inv};
        }
        return {rx, ry, rz};
    }

    /// Transform a Vec3 as a direction (w=0), no perspective divide
    Vec3 TransformDirection(const Vec3& v) const {
        return {
            At(0,0)*v.x + At(0,1)*v.y + At(0,2)*v.z,
            At(1,0)*v.x + At(1,1)*v.y + At(1,2)*v.z,
            At(2,0)*v.x + At(2,1)*v.y + At(2,2)*v.z
        };
    }

    /// Lerp all 16 elements (useful for interpolating affine transforms)
    static Mat4 Lerp(const Mat4& a, const Mat4& b, float t) {
        Mat4 r;
        for (int i = 0; i < 16; ++i) {
            r.m[i] = a.m[i] + (b.m[i] - a.m[i]) * t;
        }
        return r;
    }
};

// =============================================================================
// Quat — Quaternion (for future use, Jolt-compatible layout)
// =============================================================================

struct Quat {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f;

    Quat() = default;
    Quat(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    static Quat Identity() { return {0.0f, 0.0f, 0.0f, 1.0f}; }

    float Length() const { return std::sqrt(x*x + y*y + z*z + w*w); }

    Quat Normalized() const {
        float len = Length();
        if (len > 0.0f) {
            float inv = 1.0f / len;
            return {x * inv, y * inv, z * inv, w * inv};
        }
        return Identity();
    }

    Quat operator*(const Quat& q) const {
        return {
            w*q.x + x*q.w + y*q.z - z*q.y,
            w*q.y - x*q.z + y*q.w + z*q.x,
            w*q.z + x*q.y - y*q.x + z*q.w,
            w*q.w - x*q.x - y*q.y - z*q.z
        };
    }

    Vec3 Rotate(const Vec3& v) const {
        // q * v * q^-1 (optimized)
        Vec3 u{x, y, z};
        float s = w;
        return u * (2.0f * u.Dot(v)) + v * (s * s - u.Dot(u)) + u.Cross(v) * (2.0f * s);
    }
};

} // namespace WulfNet
