// =============================================================================
// WulfNet Engine - Math Utilities
// =============================================================================
// Common math functions and constants
// =============================================================================

#pragma once

#include "../Types.h"
#include <cmath>

namespace WulfNet {
namespace Math {

// =============================================================================
// Constants
// =============================================================================

constexpr f32 PI        = 3.14159265358979323846f;
constexpr f32 TWO_PI    = 6.28318530717958647692f;
constexpr f32 HALF_PI   = 1.57079632679489661923f;
constexpr f32 INV_PI    = 0.31830988618379067154f;
constexpr f32 INV_TWO_PI = 0.15915494309189533577f;

constexpr f32 E         = 2.71828182845904523536f;
constexpr f32 SQRT2     = 1.41421356237309504880f;
constexpr f32 INV_SQRT2 = 0.70710678118654752440f;
constexpr f32 SQRT3     = 1.73205080756887729352f;

constexpr f32 EPSILON   = 1e-6f;
constexpr f32 SMALL_NUM = 1e-10f;
constexpr f32 LARGE_NUM = 1e10f;

constexpr f32 DEG_TO_RAD = PI / 180.0f;
constexpr f32 RAD_TO_DEG = 180.0f / PI;

// =============================================================================
// Basic Math Functions
// =============================================================================

template<typename T>
constexpr T min(T a, T b) {
    return (a < b) ? a : b;
}

template<typename T>
constexpr T max(T a, T b) {
    return (a > b) ? a : b;
}

template<typename T>
constexpr T clamp(T value, T minVal, T maxVal) {
    return min(max(value, minVal), maxVal);
}

template<typename T>
constexpr T saturate(T value) {
    return clamp(value, T(0), T(1));
}

template<typename T>
constexpr T abs(T value) {
    return (value < T(0)) ? -value : value;
}

template<typename T>
constexpr T sign(T value) {
    return (value > T(0)) ? T(1) : ((value < T(0)) ? T(-1) : T(0));
}

template<typename T>
constexpr T lerp(T a, T b, T t) {
    return a + t * (b - a);
}

template<typename T>
constexpr T smoothstep(T edge0, T edge1, T x) {
    T t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (T(3) - T(2) * t);
}

template<typename T>
constexpr T smootherstep(T edge0, T edge1, T x) {
    T t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * t * (t * (t * T(6) - T(15)) + T(10));
}

template<typename T>
constexpr T remap(T value, T inMin, T inMax, T outMin, T outMax) {
    return outMin + (value - inMin) * (outMax - outMin) / (inMax - inMin);
}

// =============================================================================
// Angle Functions
// =============================================================================

WULFNET_FORCEINLINE f32 radians(f32 degrees) {
    return degrees * DEG_TO_RAD;
}

WULFNET_FORCEINLINE f32 degrees(f32 radians) {
    return radians * RAD_TO_DEG;
}

WULFNET_FORCEINLINE f32 wrapAngle(f32 angle) {
    // Wrap angle to [-PI, PI]
    angle = std::fmod(angle + PI, TWO_PI);
    if (angle < 0) angle += TWO_PI;
    return angle - PI;
}

// =============================================================================
// Floating Point Utilities
// =============================================================================

WULFNET_FORCEINLINE bool isNearlyEqual(f32 a, f32 b, f32 epsilon = EPSILON) {
    return abs(a - b) <= epsilon;
}

WULFNET_FORCEINLINE bool isNearlyZero(f32 a, f32 epsilon = EPSILON) {
    return abs(a) <= epsilon;
}

WULFNET_FORCEINLINE bool isFinite(f32 value) {
    return std::isfinite(value);
}

WULFNET_FORCEINLINE bool isNaN(f32 value) {
    return std::isnan(value);
}

WULFNET_FORCEINLINE bool isInf(f32 value) {
    return std::isinf(value);
}

// =============================================================================
// Fast Math Approximations
// =============================================================================

// Fast inverse square root (Quake III algorithm, modernized)
WULFNET_FORCEINLINE f32 fastInvSqrt(f32 x) {
    f32 xhalf = 0.5f * x;
    i32 i = *reinterpret_cast<i32*>(&x);
    i = 0x5f375a86 - (i >> 1);
    x = *reinterpret_cast<f32*>(&i);
    x = x * (1.5f - xhalf * x * x);  // Newton-Raphson iteration
    return x;
}

// Fast square root using SSE
WULFNET_FORCEINLINE f32 fastSqrt(f32 x) {
    return x * fastInvSqrt(x);
}

// Fast approximation of sin (Bhaskara approximation)
WULFNET_FORCEINLINE f32 fastSin(f32 x) {
    // Reduce to [-PI, PI]
    x = wrapAngle(x);
    
    // Bhaskara I's sine approximation
    f32 y = (16.0f * x * (PI - abs(x))) / 
            (5.0f * PI * PI - 4.0f * abs(x) * (PI - abs(x)));
    return y;
}

// Fast approximation of cos
WULFNET_FORCEINLINE f32 fastCos(f32 x) {
    return fastSin(x + HALF_PI);
}

// Fast atan2 approximation
WULFNET_FORCEINLINE f32 fastAtan2(f32 y, f32 x) {
    f32 absY = abs(y) + SMALL_NUM;  // Prevent 0/0
    f32 angle;
    
    if (x >= 0) {
        f32 r = (x - absY) / (x + absY);
        angle = 0.1963f * r * r * r - 0.9817f * r + HALF_PI * 0.5f;
    } else {
        f32 r = (x + absY) / (absY - x);
        angle = 0.1963f * r * r * r - 0.9817f * r + HALF_PI * 1.5f;
    }
    
    return (y < 0) ? -angle : angle;
}

// =============================================================================
// Bit Manipulation
// =============================================================================

WULFNET_FORCEINLINE u32 countLeadingZeros(u32 value) {
    if (value == 0) return 32;
    #if WULFNET_COMPILER_MSVC
        unsigned long index;
        _BitScanReverse(&index, value);
        return 31 - index;
    #else
        return __builtin_clz(value);
    #endif
}

WULFNET_FORCEINLINE u32 countTrailingZeros(u32 value) {
    if (value == 0) return 32;
    #if WULFNET_COMPILER_MSVC
        unsigned long index;
        _BitScanForward(&index, value);
        return index;
    #else
        return __builtin_ctz(value);
    #endif
}

WULFNET_FORCEINLINE u32 popCount(u32 value) {
    #if WULFNET_COMPILER_MSVC
        return __popcnt(value);
    #else
        return __builtin_popcount(value);
    #endif
}

} // namespace Math
} // namespace WulfNet
