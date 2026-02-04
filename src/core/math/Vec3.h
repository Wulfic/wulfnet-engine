// =============================================================================
// WulfNet Engine - Vec3 (3D Vector)
// =============================================================================
// SIMD-optimized 3D vector for physics calculations
// =============================================================================

#pragma once

#include "SIMD.h"
#include "MathUtils.h"

namespace WulfNet {

struct Vec4;  // Forward declaration

// =============================================================================
// Vec3 - 3D Vector (padded to 16 bytes for SIMD)
// =============================================================================

struct WULFNET_ALIGNAS(16) Vec3 {
    union {
        struct { f32 x, y, z; };
        struct { f32 r, g, b; };
        f32 data[4];  // Padded for SIMD (w is unused)
        SIMD::Vec4f simd;
    };
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    Vec3() : simd(SIMD::zero4f()) {}
    explicit Vec3(f32 scalar) : simd(SIMD::broadcast4f(scalar)) {}
    Vec3(f32 x_, f32 y_, f32 z_) : simd(SIMD::set4f(x_, y_, z_, 0.0f)) {}
    Vec3(SIMD::Vec4f v) : simd(v) {}
    
    // =========================================================================
    // Static Factory Methods
    // =========================================================================
    
    static Vec3 zero() { return Vec3(0.0f); }
    static Vec3 one() { return Vec3(1.0f); }
    static Vec3 unitX() { return Vec3(1.0f, 0.0f, 0.0f); }
    static Vec3 unitY() { return Vec3(0.0f, 1.0f, 0.0f); }
    static Vec3 unitZ() { return Vec3(0.0f, 0.0f, 1.0f); }
    static Vec3 up() { return unitY(); }
    static Vec3 down() { return Vec3(0.0f, -1.0f, 0.0f); }
    static Vec3 forward() { return Vec3(0.0f, 0.0f, -1.0f); }  // Right-handed
    static Vec3 back() { return unitZ(); }
    static Vec3 right() { return unitX(); }
    static Vec3 left() { return Vec3(-1.0f, 0.0f, 0.0f); }
    
    // =========================================================================
    // Accessors
    // =========================================================================
    
    f32& operator[](usize i) { return data[i]; }
    const f32& operator[](usize i) const { return data[i]; }
    
    // =========================================================================
    // Arithmetic Operators
    // =========================================================================
    
    Vec3 operator+(const Vec3& other) const {
        return Vec3(SIMD::add4f(simd, other.simd));
    }
    
    Vec3 operator-(const Vec3& other) const {
        return Vec3(SIMD::sub4f(simd, other.simd));
    }
    
    Vec3 operator*(const Vec3& other) const {
        return Vec3(SIMD::mul4f(simd, other.simd));
    }
    
    Vec3 operator/(const Vec3& other) const {
        return Vec3(SIMD::div4f(simd, other.simd));
    }
    
    Vec3 operator*(f32 scalar) const {
        return Vec3(SIMD::mul4f(simd, SIMD::broadcast4f(scalar)));
    }
    
    Vec3 operator/(f32 scalar) const {
        return Vec3(SIMD::div4f(simd, SIMD::broadcast4f(scalar)));
    }
    
    Vec3 operator-() const {
        return Vec3(SIMD::neg4f(simd));
    }
    
    // =========================================================================
    // Compound Assignment Operators
    // =========================================================================
    
    Vec3& operator+=(const Vec3& other) {
        simd = SIMD::add4f(simd, other.simd);
        return *this;
    }
    
    Vec3& operator-=(const Vec3& other) {
        simd = SIMD::sub4f(simd, other.simd);
        return *this;
    }
    
    Vec3& operator*=(const Vec3& other) {
        simd = SIMD::mul4f(simd, other.simd);
        return *this;
    }
    
    Vec3& operator/=(const Vec3& other) {
        simd = SIMD::div4f(simd, other.simd);
        return *this;
    }
    
    Vec3& operator*=(f32 scalar) {
        simd = SIMD::mul4f(simd, SIMD::broadcast4f(scalar));
        return *this;
    }
    
    Vec3& operator/=(f32 scalar) {
        simd = SIMD::div4f(simd, SIMD::broadcast4f(scalar));
        return *this;
    }
    
    // =========================================================================
    // Comparison Operators
    // =========================================================================
    
    bool operator==(const Vec3& other) const {
        return Math::isNearlyEqual(x, other.x) && 
               Math::isNearlyEqual(y, other.y) && 
               Math::isNearlyEqual(z, other.z);
    }
    
    bool operator!=(const Vec3& other) const {
        return !(*this == other);
    }
    
    // =========================================================================
    // Vector Operations
    // =========================================================================
    
    f32 dot(const Vec3& other) const {
        return SIMD::dot3(simd, other.simd);
    }
    
    Vec3 cross(const Vec3& other) const {
        return Vec3(SIMD::cross3(simd, other.simd));
    }
    
    f32 lengthSq() const {
        return dot(*this);
    }
    
    f32 length() const {
        return std::sqrt(lengthSq());
    }
    
    f32 lengthFast() const {
        return Math::fastSqrt(lengthSq());
    }
    
    Vec3 normalized() const {
        f32 len = length();
        if (len > Math::EPSILON) {
            return *this / len;
        }
        return Vec3::zero();
    }
    
    Vec3 normalizedFast() const {
        f32 lenSq = lengthSq();
        if (lenSq > Math::EPSILON) {
            return *this * Math::fastInvSqrt(lenSq);
        }
        return Vec3::zero();
    }
    
    void normalize() {
        *this = normalized();
    }
    
    f32 distance(const Vec3& other) const {
        return (*this - other).length();
    }
    
    f32 distanceSq(const Vec3& other) const {
        return (*this - other).lengthSq();
    }
    
    // =========================================================================
    // Component-wise Operations
    // =========================================================================
    
    Vec3 abs() const {
        return Vec3(SIMD::abs4f(simd));
    }
    
    Vec3 min(const Vec3& other) const {
        return Vec3(SIMD::min4f(simd, other.simd));
    }
    
    Vec3 max(const Vec3& other) const {
        return Vec3(SIMD::max4f(simd, other.simd));
    }
    
    Vec3 clamp(const Vec3& minVal, const Vec3& maxVal) const {
        return max(minVal).min(maxVal);
    }
    
    Vec3 lerp(const Vec3& other, f32 t) const {
        return *this + (other - *this) * t;
    }
    
    f32 minComponent() const {
        return Math::min(Math::min(x, y), z);
    }
    
    f32 maxComponent() const {
        return Math::max(Math::max(x, y), z);
    }
    
    // =========================================================================
    // Geometric Operations
    // =========================================================================
    
    Vec3 reflect(const Vec3& normal) const {
        return *this - normal * (2.0f * dot(normal));
    }
    
    Vec3 project(const Vec3& onto) const {
        f32 d = onto.dot(onto);
        if (d > Math::EPSILON) {
            return onto * (dot(onto) / d);
        }
        return Vec3::zero();
    }
    
    Vec3 reject(const Vec3& from) const {
        return *this - project(from);
    }
    
    // =========================================================================
    // Utility
    // =========================================================================
    
    bool isZero(f32 epsilon = Math::EPSILON) const {
        return lengthSq() <= epsilon * epsilon;
    }
    
    bool isNormalized(f32 epsilon = Math::EPSILON) const {
        return Math::isNearlyEqual(lengthSq(), 1.0f, epsilon);
    }
    
    bool isFinite() const {
        return Math::isFinite(x) && Math::isFinite(y) && Math::isFinite(z);
    }
};

// =============================================================================
// External Operators
// =============================================================================

inline Vec3 operator*(f32 scalar, const Vec3& v) {
    return v * scalar;
}

// =============================================================================
// Free Functions
// =============================================================================

inline f32 dot(const Vec3& a, const Vec3& b) { return a.dot(b); }
inline Vec3 cross(const Vec3& a, const Vec3& b) { return a.cross(b); }
inline Vec3 normalize(const Vec3& v) { return v.normalized(); }
inline f32 length(const Vec3& v) { return v.length(); }
inline f32 distance(const Vec3& a, const Vec3& b) { return a.distance(b); }
inline Vec3 lerp(const Vec3& a, const Vec3& b, f32 t) { return a.lerp(b, t); }
inline Vec3 reflect(const Vec3& v, const Vec3& n) { return v.reflect(n); }

// Ensure proper alignment
static_assert(sizeof(Vec3) == 16, "Vec3 must be 16 bytes for SIMD");
static_assert(alignof(Vec3) == 16, "Vec3 must be 16-byte aligned");

} // namespace WulfNet
