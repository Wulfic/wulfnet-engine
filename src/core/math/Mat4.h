// =============================================================================
// WulfNet Engine - Mat4 (4x4 Matrix)
// =============================================================================
// SIMD-optimized 4x4 matrix for transforms and projections
// Column-major layout for GPU compatibility
// =============================================================================

#pragma once

#include "SIMD.h"
#include "MathUtils.h"
#include "Vec3.h"
#include "Vec4.h"
#include "Quat.h"

namespace WulfNet {

// =============================================================================
// Mat4 - 4x4 Matrix (Column-Major)
// =============================================================================
// Layout: columns[0] = first column, etc.
// Memory: [m00, m10, m20, m30, m01, m11, m21, m31, ...]

struct WULFNET_ALIGNAS(64) Mat4 {
    union {
        Vec4 columns[4];
        f32 data[16];
        SIMD::Vec4f simd[4];
    };
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    Mat4() {
        columns[0] = Vec4::unitX();
        columns[1] = Vec4::unitY();
        columns[2] = Vec4::unitZ();
        columns[3] = Vec4::unitW();
    }
    
    explicit Mat4(f32 diagonal) {
        columns[0] = Vec4(diagonal, 0.0f, 0.0f, 0.0f);
        columns[1] = Vec4(0.0f, diagonal, 0.0f, 0.0f);
        columns[2] = Vec4(0.0f, 0.0f, diagonal, 0.0f);
        columns[3] = Vec4(0.0f, 0.0f, 0.0f, diagonal);
    }
    
    Mat4(const Vec4& c0, const Vec4& c1, const Vec4& c2, const Vec4& c3) {
        columns[0] = c0;
        columns[1] = c1;
        columns[2] = c2;
        columns[3] = c3;
    }
    
    // =========================================================================
    // Static Factory Methods
    // =========================================================================
    
    static Mat4 identity() { return Mat4(1.0f); }
    static Mat4 zero() { return Mat4(0.0f); }
    
    // Translation matrix
    static Mat4 translation(const Vec3& t) {
        Mat4 result;
        result.columns[3] = Vec4(t, 1.0f);
        return result;
    }
    
    static Mat4 translation(f32 x, f32 y, f32 z) {
        return translation(Vec3(x, y, z));
    }
    
    // Scale matrix
    static Mat4 scale(const Vec3& s) {
        Mat4 result;
        result.columns[0].x = s.x;
        result.columns[1].y = s.y;
        result.columns[2].z = s.z;
        return result;
    }
    
    static Mat4 scale(f32 s) {
        return scale(Vec3(s));
    }
    
    static Mat4 scale(f32 x, f32 y, f32 z) {
        return scale(Vec3(x, y, z));
    }
    
    // Rotation matrices
    static Mat4 rotationX(f32 angle) {
        f32 c = std::cos(angle);
        f32 s = std::sin(angle);
        Mat4 result;
        result.columns[1] = Vec4(0.0f, c, s, 0.0f);
        result.columns[2] = Vec4(0.0f, -s, c, 0.0f);
        return result;
    }
    
    static Mat4 rotationY(f32 angle) {
        f32 c = std::cos(angle);
        f32 s = std::sin(angle);
        Mat4 result;
        result.columns[0] = Vec4(c, 0.0f, -s, 0.0f);
        result.columns[2] = Vec4(s, 0.0f, c, 0.0f);
        return result;
    }
    
    static Mat4 rotationZ(f32 angle) {
        f32 c = std::cos(angle);
        f32 s = std::sin(angle);
        Mat4 result;
        result.columns[0] = Vec4(c, s, 0.0f, 0.0f);
        result.columns[1] = Vec4(-s, c, 0.0f, 0.0f);
        return result;
    }
    
    static Mat4 rotation(const Vec3& axis, f32 angle) {
        return rotation(Quat::fromAxisAngle(axis, angle));
    }
    
    // Rotation from quaternion
    static Mat4 rotation(const Quat& q) {
        f32 xx = q.x * q.x;
        f32 yy = q.y * q.y;
        f32 zz = q.z * q.z;
        f32 xy = q.x * q.y;
        f32 xz = q.x * q.z;
        f32 yz = q.y * q.z;
        f32 wx = q.w * q.x;
        f32 wy = q.w * q.y;
        f32 wz = q.w * q.z;
        
        Mat4 result;
        result.columns[0] = Vec4(1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy), 0.0f);
        result.columns[1] = Vec4(2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx), 0.0f);
        result.columns[2] = Vec4(2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy), 0.0f);
        result.columns[3] = Vec4(0.0f, 0.0f, 0.0f, 1.0f);
        return result;
    }
    
    // TRS (Translation, Rotation, Scale) transform
    static Mat4 TRS(const Vec3& translation, const Quat& rotation, const Vec3& scale) {
        Mat4 r = Mat4::rotation(rotation);
        
        r.columns[0] *= scale.x;
        r.columns[1] *= scale.y;
        r.columns[2] *= scale.z;
        r.columns[3] = Vec4(translation, 1.0f);
        
        return r;
    }
    
    // Look-at view matrix (right-handed)
    static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
        Vec3 f = (target - eye).normalized();  // Forward
        Vec3 r = f.cross(up).normalized();     // Right
        Vec3 u = r.cross(f);                   // Up
        
        Mat4 result;
        result.columns[0] = Vec4(r.x, u.x, -f.x, 0.0f);
        result.columns[1] = Vec4(r.y, u.y, -f.y, 0.0f);
        result.columns[2] = Vec4(r.z, u.z, -f.z, 0.0f);
        result.columns[3] = Vec4(-r.dot(eye), -u.dot(eye), f.dot(eye), 1.0f);
        return result;
    }
    
    // Perspective projection (right-handed, Vulkan/DX12 clip space [0, 1])
    static Mat4 perspective(f32 fovY, f32 aspect, f32 nearPlane, f32 farPlane) {
        f32 tanHalfFov = std::tan(fovY * 0.5f);
        
        Mat4 result = zero();
        result.columns[0].x = 1.0f / (aspect * tanHalfFov);
        result.columns[1].y = 1.0f / tanHalfFov;
        result.columns[2].z = farPlane / (nearPlane - farPlane);
        result.columns[2].w = -1.0f;
        result.columns[3].z = (nearPlane * farPlane) / (nearPlane - farPlane);
        return result;
    }
    
    // Infinite perspective (for reverse-Z rendering)
    static Mat4 perspectiveInfinite(f32 fovY, f32 aspect, f32 nearPlane) {
        f32 tanHalfFov = std::tan(fovY * 0.5f);
        
        Mat4 result = zero();
        result.columns[0].x = 1.0f / (aspect * tanHalfFov);
        result.columns[1].y = 1.0f / tanHalfFov;
        result.columns[2].z = 0.0f;
        result.columns[2].w = -1.0f;
        result.columns[3].z = nearPlane;
        return result;
    }
    
    // Orthographic projection
    static Mat4 orthographic(f32 left, f32 right, f32 bottom, f32 top, 
                             f32 nearPlane, f32 farPlane) {
        Mat4 result = zero();
        result.columns[0].x = 2.0f / (right - left);
        result.columns[1].y = 2.0f / (top - bottom);
        result.columns[2].z = 1.0f / (nearPlane - farPlane);
        result.columns[3].x = -(right + left) / (right - left);
        result.columns[3].y = -(top + bottom) / (top - bottom);
        result.columns[3].z = nearPlane / (nearPlane - farPlane);
        result.columns[3].w = 1.0f;
        return result;
    }
    
    // =========================================================================
    // Accessors
    // =========================================================================
    
    Vec4& operator[](usize i) { return columns[i]; }
    const Vec4& operator[](usize i) const { return columns[i]; }
    
    f32& at(usize row, usize col) { return columns[col][row]; }
    const f32& at(usize row, usize col) const { return columns[col][row]; }
    
    // =========================================================================
    // Matrix Operations
    // =========================================================================
    
    // Matrix-Matrix multiplication (SIMD optimized)
    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        
        for (int i = 0; i < 4; i++) {
            SIMD::Vec4f col = SIMD::mul4f(simd[0], SIMD::xxxx(other.simd[i]));
            col = SIMD::fmadd4f(simd[1], SIMD::yyyy(other.simd[i]), col);
            col = SIMD::fmadd4f(simd[2], SIMD::zzzz(other.simd[i]), col);
            col = SIMD::fmadd4f(simd[3], SIMD::wwww(other.simd[i]), col);
            result.simd[i] = col;
        }
        
        return result;
    }
    
    Mat4& operator*=(const Mat4& other) {
        *this = *this * other;
        return *this;
    }
    
    // Matrix-Vector multiplication
    Vec4 operator*(const Vec4& v) const {
        SIMD::Vec4f result = SIMD::mul4f(simd[0], SIMD::xxxx(v.simd));
        result = SIMD::fmadd4f(simd[1], SIMD::yyyy(v.simd), result);
        result = SIMD::fmadd4f(simd[2], SIMD::zzzz(v.simd), result);
        result = SIMD::fmadd4f(simd[3], SIMD::wwww(v.simd), result);
        return Vec4(result);
    }
    
    // Transform point (w = 1)
    Vec3 transformPoint(const Vec3& p) const {
        Vec4 result = *this * Vec4(p, 1.0f);
        return result.homogenize();
    }
    
    // Transform direction (w = 0, no translation)
    Vec3 transformDirection(const Vec3& d) const {
        Vec4 result = *this * Vec4(d, 0.0f);
        return result.xyz();
    }
    
    // Transform normal (use inverse transpose for correct normals)
    Vec3 transformNormal(const Vec3& n) const {
        // For uniform scale or orthonormal matrix, this works
        // For non-uniform scale, use inverse transpose
        return transformDirection(n).normalized();
    }
    
    // =========================================================================
    // Matrix Properties
    // =========================================================================
    
    Mat4 transposed() const {
        Mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.at(i, j) = at(j, i);
            }
        }
        return result;
    }
    
    // Determinant
    f32 determinant() const {
        f32 sub00 = columns[2].z * columns[3].w - columns[3].z * columns[2].w;
        f32 sub01 = columns[2].y * columns[3].w - columns[3].y * columns[2].w;
        f32 sub02 = columns[2].y * columns[3].z - columns[3].y * columns[2].z;
        f32 sub03 = columns[2].x * columns[3].w - columns[3].x * columns[2].w;
        f32 sub04 = columns[2].x * columns[3].z - columns[3].x * columns[2].z;
        f32 sub05 = columns[2].x * columns[3].y - columns[3].x * columns[2].y;
        
        Vec4 cofactors(
            +(columns[1].y * sub00 - columns[1].z * sub01 + columns[1].w * sub02),
            -(columns[1].x * sub00 - columns[1].z * sub03 + columns[1].w * sub04),
            +(columns[1].x * sub01 - columns[1].y * sub03 + columns[1].w * sub05),
            -(columns[1].x * sub02 - columns[1].y * sub04 + columns[1].z * sub05)
        );
        
        return columns[0].x * cofactors.x + columns[0].y * cofactors.y + 
               columns[0].z * cofactors.z + columns[0].w * cofactors.w;
    }
    
    // Inverse (general 4x4 matrix inverse)
    Mat4 inverse() const {
        f32 coef00 = columns[2].z * columns[3].w - columns[3].z * columns[2].w;
        f32 coef02 = columns[1].z * columns[3].w - columns[3].z * columns[1].w;
        f32 coef03 = columns[1].z * columns[2].w - columns[2].z * columns[1].w;
        f32 coef04 = columns[2].y * columns[3].w - columns[3].y * columns[2].w;
        f32 coef06 = columns[1].y * columns[3].w - columns[3].y * columns[1].w;
        f32 coef07 = columns[1].y * columns[2].w - columns[2].y * columns[1].w;
        f32 coef08 = columns[2].y * columns[3].z - columns[3].y * columns[2].z;
        f32 coef10 = columns[1].y * columns[3].z - columns[3].y * columns[1].z;
        f32 coef11 = columns[1].y * columns[2].z - columns[2].y * columns[1].z;
        f32 coef12 = columns[2].x * columns[3].w - columns[3].x * columns[2].w;
        f32 coef14 = columns[1].x * columns[3].w - columns[3].x * columns[1].w;
        f32 coef15 = columns[1].x * columns[2].w - columns[2].x * columns[1].w;
        f32 coef16 = columns[2].x * columns[3].z - columns[3].x * columns[2].z;
        f32 coef18 = columns[1].x * columns[3].z - columns[3].x * columns[1].z;
        f32 coef19 = columns[1].x * columns[2].z - columns[2].x * columns[1].z;
        f32 coef20 = columns[2].x * columns[3].y - columns[3].x * columns[2].y;
        f32 coef22 = columns[1].x * columns[3].y - columns[3].x * columns[1].y;
        f32 coef23 = columns[1].x * columns[2].y - columns[2].x * columns[1].y;
        
        Vec4 fac0(coef00, coef00, coef02, coef03);
        Vec4 fac1(coef04, coef04, coef06, coef07);
        Vec4 fac2(coef08, coef08, coef10, coef11);
        Vec4 fac3(coef12, coef12, coef14, coef15);
        Vec4 fac4(coef16, coef16, coef18, coef19);
        Vec4 fac5(coef20, coef20, coef22, coef23);
        
        Vec4 vec0(columns[1].x, columns[0].x, columns[0].x, columns[0].x);
        Vec4 vec1(columns[1].y, columns[0].y, columns[0].y, columns[0].y);
        Vec4 vec2(columns[1].z, columns[0].z, columns[0].z, columns[0].z);
        Vec4 vec3(columns[1].w, columns[0].w, columns[0].w, columns[0].w);
        
        Vec4 inv0(vec1 * fac0 - vec2 * fac1 + vec3 * fac2);
        Vec4 inv1(vec0 * fac0 - vec2 * fac3 + vec3 * fac4);
        Vec4 inv2(vec0 * fac1 - vec1 * fac3 + vec3 * fac5);
        Vec4 inv3(vec0 * fac2 - vec1 * fac4 + vec2 * fac5);
        
        Vec4 signA(+1, -1, +1, -1);
        Vec4 signB(-1, +1, -1, +1);
        
        Mat4 inv(inv0 * signA, inv1 * signB, inv2 * signA, inv3 * signB);
        
        Vec4 row0(inv.columns[0].x, inv.columns[1].x, inv.columns[2].x, inv.columns[3].x);
        f32 det = columns[0].dot(row0);
        
        return inv * (1.0f / det);
    }
    
    // Fast inverse for affine transforms (rotation + translation + uniform scale)
    Mat4 inverseAffine() const {
        Mat4 result;
        
        // Transpose 3x3 rotation part
        result.columns[0] = Vec4(columns[0].x, columns[1].x, columns[2].x, 0.0f);
        result.columns[1] = Vec4(columns[0].y, columns[1].y, columns[2].y, 0.0f);
        result.columns[2] = Vec4(columns[0].z, columns[1].z, columns[2].z, 0.0f);
        
        // Negate translation
        Vec3 t = -columns[3].xyz();
        result.columns[3] = Vec4(
            result.columns[0].xyz().dot(t),
            result.columns[1].xyz().dot(t),
            result.columns[2].xyz().dot(t),
            1.0f
        );
        
        return result;
    }
    
    // =========================================================================
    // Decomposition
    // =========================================================================
    
    Vec3 getTranslation() const {
        return columns[3].xyz();
    }
    
    Vec3 getScale() const {
        return Vec3(
            columns[0].xyz().length(),
            columns[1].xyz().length(),
            columns[2].xyz().length()
        );
    }
    
    Quat getRotation() const {
        Vec3 s = getScale();
        Mat4 rotMat;
        rotMat.columns[0] = Vec4(columns[0].xyz() / s.x, 0.0f);
        rotMat.columns[1] = Vec4(columns[1].xyz() / s.y, 0.0f);
        rotMat.columns[2] = Vec4(columns[2].xyz() / s.z, 0.0f);
        rotMat.columns[3] = Vec4::unitW();
        
        // Convert rotation matrix to quaternion
        f32 trace = rotMat[0][0] + rotMat[1][1] + rotMat[2][2];
        
        if (trace > 0) {
            f32 s = 0.5f / std::sqrt(trace + 1.0f);
            return Quat(
                (rotMat[1][2] - rotMat[2][1]) * s,
                (rotMat[2][0] - rotMat[0][2]) * s,
                (rotMat[0][1] - rotMat[1][0]) * s,
                0.25f / s
            );
        } else if (rotMat[0][0] > rotMat[1][1] && rotMat[0][0] > rotMat[2][2]) {
            f32 s = 2.0f * std::sqrt(1.0f + rotMat[0][0] - rotMat[1][1] - rotMat[2][2]);
            return Quat(
                0.25f * s,
                (rotMat[1][0] + rotMat[0][1]) / s,
                (rotMat[2][0] + rotMat[0][2]) / s,
                (rotMat[1][2] - rotMat[2][1]) / s
            );
        } else if (rotMat[1][1] > rotMat[2][2]) {
            f32 s = 2.0f * std::sqrt(1.0f + rotMat[1][1] - rotMat[0][0] - rotMat[2][2]);
            return Quat(
                (rotMat[1][0] + rotMat[0][1]) / s,
                0.25f * s,
                (rotMat[2][1] + rotMat[1][2]) / s,
                (rotMat[2][0] - rotMat[0][2]) / s
            );
        } else {
            f32 s = 2.0f * std::sqrt(1.0f + rotMat[2][2] - rotMat[0][0] - rotMat[1][1]);
            return Quat(
                (rotMat[2][0] + rotMat[0][2]) / s,
                (rotMat[2][1] + rotMat[1][2]) / s,
                0.25f * s,
                (rotMat[0][1] - rotMat[1][0]) / s
            );
        }
    }
    
    // =========================================================================
    // Scalar Operations
    // =========================================================================
    
    Mat4 operator*(f32 scalar) const {
        Mat4 result;
        SIMD::Vec4f s = SIMD::broadcast4f(scalar);
        for (int i = 0; i < 4; i++) {
            result.simd[i] = SIMD::mul4f(simd[i], s);
        }
        return result;
    }
    
    Mat4& operator*=(f32 scalar) {
        SIMD::Vec4f s = SIMD::broadcast4f(scalar);
        for (int i = 0; i < 4; i++) {
            simd[i] = SIMD::mul4f(simd[i], s);
        }
        return *this;
    }
};

// =============================================================================
// External Operators
// =============================================================================

inline Mat4 operator*(f32 scalar, const Mat4& m) {
    return m * scalar;
}

// =============================================================================
// Free Functions
// =============================================================================

inline Mat4 transpose(const Mat4& m) { return m.transposed(); }
inline Mat4 inverse(const Mat4& m) { return m.inverse(); }
inline f32 determinant(const Mat4& m) { return m.determinant(); }

// Ensure proper alignment
static_assert(sizeof(Mat4) == 64, "Mat4 must be 64 bytes");
static_assert(alignof(Mat4) == 64, "Mat4 must be 64-byte aligned");

} // namespace WulfNet
