// =============================================================================
// WulfNet Engine - Vec4 (4D Vector)
// =============================================================================
// SIMD-optimized 4D vector for homogeneous coordinates and colors
// =============================================================================

#pragma once

#include "SIMD.h"
#include "MathUtils.h"
#include "Vec3.h"

namespace WulfNet {

// =============================================================================
// Vec4 - 4D Vector (perfectly aligned for SIMD)
// =============================================================================

struct WULFNET_ALIGNAS(16) Vec4 {
    union {
        struct { f32 x, y, z, w; };
        struct { f32 r, g, b, a; };
        f32 data[4];
        SIMD::Vec4f simd;
    };
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    Vec4() : simd(SIMD::zero4f()) {}
    explicit Vec4(f32 scalar) : simd(SIMD::broadcast4f(scalar)) {}
    Vec4(f32 x_, f32 y_, f32 z_, f32 w_) : simd(SIMD::set4f(x_, y_, z_, w_)) {}
    Vec4(const Vec3& v, f32 w_) : simd(SIMD::set4f(v.x, v.y, v.z, w_)) {}
    Vec4(SIMD::Vec4f v) : simd(v) {}
    
    // =========================================================================
    // Static Factory Methods
    // =========================================================================
    
    static Vec4 zero() { return Vec4(0.0f); }
    static Vec4 one() { return Vec4(1.0f); }
    static Vec4 unitX() { return Vec4(1.0f, 0.0f, 0.0f, 0.0f); }
    static Vec4 unitY() { return Vec4(0.0f, 1.0f, 0.0f, 0.0f); }
    static Vec4 unitZ() { return Vec4(0.0f, 0.0f, 1.0f, 0.0f); }
    static Vec4 unitW() { return Vec4(0.0f, 0.0f, 0.0f, 1.0f); }
    
    // Color presets
    static Vec4 white() { return Vec4(1.0f, 1.0f, 1.0f, 1.0f); }
    static Vec4 black() { return Vec4(0.0f, 0.0f, 0.0f, 1.0f); }
    static Vec4 red() { return Vec4(1.0f, 0.0f, 0.0f, 1.0f); }
    static Vec4 green() { return Vec4(0.0f, 1.0f, 0.0f, 1.0f); }
    static Vec4 blue() { return Vec4(0.0f, 0.0f, 1.0f, 1.0f); }
    
    // =========================================================================
    // Conversion
    // =========================================================================
    
    Vec3 xyz() const { return Vec3(x, y, z); }
    Vec3 rgb() const { return Vec3(r, g, b); }
    
    // Homogeneous coordinate operations
    Vec3 homogenize() const {
        if (Math::isNearlyZero(w)) return Vec3(x, y, z);
        f32 invW = 1.0f / w;
        return Vec3(x * invW, y * invW, z * invW);
    }
    
    // =========================================================================
    // Accessors
    // =========================================================================
    
    f32& operator[](usize i) { return data[i]; }
    const f32& operator[](usize i) const { return data[i]; }
    
    // =========================================================================
    // Arithmetic Operators
    // =========================================================================
    
    Vec4 operator+(const Vec4& other) const {
        return Vec4(SIMD::add4f(simd, other.simd));
    }
    
    Vec4 operator-(const Vec4& other) const {
        return Vec4(SIMD::sub4f(simd, other.simd));
    }
    
    Vec4 operator*(const Vec4& other) const {
        return Vec4(SIMD::mul4f(simd, other.simd));
    }
    
    Vec4 operator/(const Vec4& other) const {
        return Vec4(SIMD::div4f(simd, other.simd));
    }
    
    Vec4 operator*(f32 scalar) const {
        return Vec4(SIMD::mul4f(simd, SIMD::broadcast4f(scalar)));
    }
    
    Vec4 operator/(f32 scalar) const {
        return Vec4(SIMD::div4f(simd, SIMD::broadcast4f(scalar)));
    }
    
    Vec4 operator-() const {
        return Vec4(SIMD::neg4f(simd));
    }
    
    // =========================================================================
    // Compound Assignment Operators
    // =========================================================================
    
    Vec4& operator+=(const Vec4& other) {
        simd = SIMD::add4f(simd, other.simd);
        return *this;
    }
    
    Vec4& operator-=(const Vec4& other) {
        simd = SIMD::sub4f(simd, other.simd);
        return *this;
    }
    
    Vec4& operator*=(const Vec4& other) {
        simd = SIMD::mul4f(simd, other.simd);
        return *this;
    }
    
    Vec4& operator/=(const Vec4& other) {
        simd = SIMD::div4f(simd, other.simd);
        return *this;
    }
    
    Vec4& operator*=(f32 scalar) {
        simd = SIMD::mul4f(simd, SIMD::broadcast4f(scalar));
        return *this;
    }
    
    Vec4& operator/=(f32 scalar) {
        simd = SIMD::div4f(simd, SIMD::broadcast4f(scalar));
        return *this;
    }
    
    // =========================================================================
    // Comparison Operators
    // =========================================================================
    
    bool operator==(const Vec4& other) const {
        return Math::isNearlyEqual(x, other.x) && 
               Math::isNearlyEqual(y, other.y) && 
               Math::isNearlyEqual(z, other.z) &&
               Math::isNearlyEqual(w, other.w);
    }
    
    bool operator!=(const Vec4& other) const {
        return !(*this == other);
    }
    
    // =========================================================================
    // Vector Operations
    // =========================================================================
    
    f32 dot(const Vec4& other) const {
        return SIMD::dot4(simd, other.simd);
    }
    
    f32 lengthSq() const {
        return dot(*this);
    }
    
    f32 length() const {
        return std::sqrt(lengthSq());
    }
    
    Vec4 normalized() const {
        f32 len = length();
        if (len > Math::EPSILON) {
            return *this / len;
        }
        return Vec4::zero();
    }
    
    void normalize() {
        *this = normalized();
    }
    
    // =========================================================================
    // Component-wise Operations
    // =========================================================================
    
    Vec4 abs() const {
        return Vec4(SIMD::abs4f(simd));
    }
    
    Vec4 min(const Vec4& other) const {
        return Vec4(SIMD::min4f(simd, other.simd));
    }
    
    Vec4 max(const Vec4& other) const {
        return Vec4(SIMD::max4f(simd, other.simd));
    }
    
    Vec4 clamp(const Vec4& minVal, const Vec4& maxVal) const {
        return max(minVal).min(maxVal);
    }
    
    Vec4 saturate() const {
        return clamp(Vec4::zero(), Vec4::one());
    }
    
    Vec4 lerp(const Vec4& other, f32 t) const {
        return *this + (other - *this) * t;
    }
    
    // =========================================================================
    // Utility
    // =========================================================================
    
    bool isZero(f32 epsilon = Math::EPSILON) const {
        return lengthSq() <= epsilon * epsilon;
    }
    
    bool isFinite() const {
        return Math::isFinite(x) && Math::isFinite(y) && 
               Math::isFinite(z) && Math::isFinite(w);
    }
};

// =============================================================================
// External Operators
// =============================================================================

inline Vec4 operator*(f32 scalar, const Vec4& v) {
    return v * scalar;
}

// =============================================================================
// Free Functions
// =============================================================================

inline f32 dot(const Vec4& a, const Vec4& b) { return a.dot(b); }
inline Vec4 normalize(const Vec4& v) { return v.normalized(); }
inline f32 length(const Vec4& v) { return v.length(); }
inline Vec4 lerp(const Vec4& a, const Vec4& b, f32 t) { return a.lerp(b, t); }

// Ensure proper alignment
static_assert(sizeof(Vec4) == 16, "Vec4 must be 16 bytes");
static_assert(alignof(Vec4) == 16, "Vec4 must be 16-byte aligned");

} // namespace WulfNet
