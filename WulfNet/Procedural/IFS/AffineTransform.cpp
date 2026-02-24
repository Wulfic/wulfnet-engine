// =============================================================================
// WulfNet Engine - Affine Transform Builder Implementation
// =============================================================================

#include "WulfNet/Procedural/IFS/AffineTransform.h"
#include <cmath>

namespace WulfNet {
namespace AffineTransform {

GPUMat4x4 MakeScale(const Vec3& s) {
    GPUMat4x4 mat = GPUMat4x4::Identity();
    mat.SetRow(0, s.x, 0.0f, 0.0f, 0.0f);
    mat.SetRow(1, 0.0f, s.y, 0.0f, 0.0f);
    mat.SetRow(2, 0.0f, 0.0f, s.z, 0.0f);
    return mat;
}

GPUMat4x4 MakeShearX(const Vec3& s) {
    GPUMat4x4 mat = GPUMat4x4::Identity();
    mat.SetRow(0, 1.0f, s.y, s.z, 0.0f);
    return mat;
}

GPUMat4x4 MakeShearY(const Vec3& s) {
    GPUMat4x4 mat = GPUMat4x4::Identity();
    mat.SetRow(1, s.x, 1.0f, s.z, 0.0f);
    return mat;
}

GPUMat4x4 MakeShearZ(const Vec3& s) {
    GPUMat4x4 mat = GPUMat4x4::Identity();
    mat.SetRow(2, s.x, s.y, 1.0f, 0.0f);
    return mat;
}

GPUMat4x4 MakeTranslate(const Vec3& t) {
    GPUMat4x4 mat = GPUMat4x4::Identity();
    mat.SetRow(0, 1.0f, 0.0f, 0.0f, t.x);
    mat.SetRow(1, 0.0f, 1.0f, 0.0f, t.y);
    mat.SetRow(2, 0.0f, 0.0f, 1.0f, t.z);
    return mat;
}

GPUMat4x4 MakeRotation(const Vec3& eulerDegrees) {
    float xRad = eulerDegrees.x * DEG2RAD;
    float yRad = eulerDegrees.y * DEG2RAD;
    float zRad = eulerDegrees.z * DEG2RAD;

    float cx = std::cos(xRad), sx = std::sin(xRad);
    float cy = std::cos(yRad), sy = std::sin(yRad);
    float cz = std::cos(zRad), sz = std::sin(zRad);

    GPUMat4x4 rotX = GPUMat4x4::Identity();
    rotX.SetRow(1, 0.0f, cx, -sx, 0.0f);
    rotX.SetRow(2, 0.0f, sx, cx, 0.0f);

    GPUMat4x4 rotY = GPUMat4x4::Identity();
    rotY.SetRow(0, cy, 0.0f, sy, 0.0f);
    rotY.SetRow(2, -sy, 0.0f, cy, 0.0f);

    GPUMat4x4 rotZ = GPUMat4x4::Identity();
    rotZ.SetRow(0, cz, -sz, 0.0f, 0.0f);
    rotZ.SetRow(1, sz, cz, 0.0f, 0.0f);

    // Matches Unity convention: Y * X * Z
    return rotY * rotX * rotZ;
}

GPUMat4x4 FromInstructions(const TransformInstructions& inst) {
    GPUMat4x4 scale = MakeScale(inst.scale);
    GPUMat4x4 shearX = MakeShearX(inst.shearX);
    GPUMat4x4 shearY = MakeShearY(inst.shearY);
    GPUMat4x4 shearZ = MakeShearZ(inst.shearZ);
    GPUMat4x4 shear = shearZ * shearY * shearX;
    GPUMat4x4 translate = MakeTranslate(inst.translate);
    GPUMat4x4 rotation = MakeRotation(inst.rotate);

    // Order: scale * rotation * shear * translate (matches Unity reference)
    return scale * rotation * shear * translate;
}

GPUMat4x4 Interpolate(const GPUMat4x4& m1, const GPUMat4x4& m2, float t) {
    GPUMat4x4 result = {};
    for (int i = 0; i < 16; i++) {
        result.m[i] = m1.m[i] + (m2.m[i] - m1.m[i]) * t;
    }
    return result;
}

} // namespace AffineTransform
} // namespace WulfNet
