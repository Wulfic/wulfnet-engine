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

// Use unified math types from Core
#include "WulfNet/Core/Math/MathTypes.h"

namespace WulfNet {

// =============================================================================
// Backward-compatible alias: GPUMat4x4 → Mat4
// =============================================================================
// [[deprecated("Use WulfNet::Mat4 instead")]]
using GPUMat4x4 = Mat4;

// Vec3 is now defined in Core/Math/MathTypes.h — no alias needed since the
// name is the same and it's in the same WulfNet namespace.

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

Mat4 MakeScale(const Vec3& s);
Mat4 MakeShearX(const Vec3& s);
Mat4 MakeShearY(const Vec3& s);
Mat4 MakeShearZ(const Vec3& s);
Mat4 MakeTranslate(const Vec3& t);
Mat4 MakeRotation(const Vec3& eulerDegrees);

/// Build a complete affine matrix from transform instructions
/// Order: scale * rotation * shear * translate
Mat4 FromInstructions(const TransformInstructions& inst);

/// Linear interpolation between two affine matrices (row-by-row lerp)
Mat4 Interpolate(const Mat4& m1, const Mat4& m2, float t);

} // namespace AffineTransform

} // namespace WulfNet
