// =============================================================================
// WulfNet Engine - Math Utilities
// =============================================================================
// Common math functions shared across all engine modules.
// =============================================================================

#pragma once

#include <cmath>
#include <algorithm>

namespace WulfNet {

// =============================================================================
// Constants
// =============================================================================

constexpr float Pi       = 3.14159265358979323846f;
constexpr float TwoPi    = 6.28318530717958647692f;
constexpr float HalfPi   = 1.57079632679489661923f;
constexpr float Epsilon  = 1e-6f;
constexpr float Deg2Rad  = Pi / 180.0f;
constexpr float Rad2Deg  = 180.0f / Pi;

// =============================================================================
// Scalar Utilities
// =============================================================================

/// Clamp a value to [lo, hi]
inline float Clamp(float x, float lo, float hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

/// Linear interpolation between a and b
inline float Lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

/// Smooth Hermite interpolation (0 at edge0, 1 at edge1)
inline float Smoothstep(float edge0, float edge1, float x) {
    float t = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/// Remap x from [inMin, inMax] to [outMin, outMax]
inline float Remap(float x, float inMin, float inMax, float outMin, float outMax) {
    float t = (x - inMin) / (inMax - inMin);
    return outMin + t * (outMax - outMin);
}

/// Convert degrees to radians
inline float DegreesToRadians(float degrees) {
    return degrees * Deg2Rad;
}

/// Convert radians to degrees
inline float RadiansToDegrees(float radians) {
    return radians * Rad2Deg;
}

/// Approximately equal within epsilon
inline bool ApproxEqual(float a, float b, float eps = Epsilon) {
    return std::abs(a - b) < eps;
}

/// Sign function: returns -1, 0, or 1
inline float Sign(float x) {
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

/// Square of a value
inline float Square(float x) { return x * x; }

} // namespace WulfNet
