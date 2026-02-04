// =============================================================================
// WulfNet Engine - Quat (Quaternion)
// =============================================================================
// SIMD-optimized quaternion for 3D rotations
// =============================================================================

#pragma once

#include "SIMD.h"
#include "MathUtils.h"
#include "Vec3.h"
#include "Vec4.h"

namespace WulfNet {

struct Mat4;  // Forward declaration

// =============================================================================
// Quat - Unit Quaternion for Rotations
// =============================================================================
// Layout: (x, y, z, w) where w is the scalar part
// Represents rotation: q = cos(θ/2) + sin(θ/2) * (axis)

struct WULFNET_ALIGNAS(16) Quat {
    union {
        struct { f32 x, y, z, w; };
        f32 data[4];
        SIMD::Vec4f simd;
    };
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    Quat() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}  // Identity
    Quat(f32 x_, f32 y_, f32 z_, f32 w_) : simd(SIMD::set4f(x_, y_, z_, w_)) {}
    Quat(SIMD::Vec4f v) : simd(v) {}
    
    // =========================================================================
    // Static Factory Methods
    // =========================================================================
    
    static Quat identity() { 
        return Quat(0.0f, 0.0f, 0.0f, 1.0f); 
    }
    
    // Create from axis and angle (radians)
    static Quat fromAxisAngle(const Vec3& axis, f32 angle) {
        f32 halfAngle = angle * 0.5f;
        f32 s = std::sin(halfAngle);
        f32 c = std::cos(halfAngle);
        Vec3 normalizedAxis = axis.normalized();
        return Quat(
            normalizedAxis.x * s,
            normalizedAxis.y * s,
            normalizedAxis.z * s,
            c
        );
    }
    
    // Create from Euler angles (in radians) - applies in order: yaw (Y), pitch (X), roll (Z)
    static Quat fromEuler(f32 pitch, f32 yaw, f32 roll) {
        f32 cy = std::cos(yaw * 0.5f);
        f32 sy = std::sin(yaw * 0.5f);
        f32 cp = std::cos(pitch * 0.5f);
        f32 sp = std::sin(pitch * 0.5f);
        f32 cr = std::cos(roll * 0.5f);
        f32 sr = std::sin(roll * 0.5f);
        
        return Quat(
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * cp * cr + sy * sp * sr
        );
    }
    
    static Quat fromEulerDegrees(f32 pitch, f32 yaw, f32 roll) {
        return fromEuler(
            Math::radians(pitch), 
            Math::radians(yaw), 
            Math::radians(roll)
        );
    }
    
    // Create rotation from one vector to another
    static Quat fromTo(const Vec3& from, const Vec3& to) {
        Vec3 f = from.normalized();
        Vec3 t = to.normalized();
        
        f32 d = f.dot(t);
        
        if (d >= 1.0f - Math::EPSILON) {
            // Vectors are parallel (same direction)
            return identity();
        }
        
        if (d <= -1.0f + Math::EPSILON) {
            // Vectors are anti-parallel (opposite directions)
            // Find orthogonal axis
            Vec3 axis = Vec3::unitX().cross(f);
            if (axis.lengthSq() < Math::EPSILON) {
                axis = Vec3::unitY().cross(f);
            }
            return fromAxisAngle(axis.normalized(), Math::PI);
        }
        
        Vec3 c = f.cross(t);
        f32 s = std::sqrt((1.0f + d) * 2.0f);
        f32 invS = 1.0f / s;
        
        return Quat(
            c.x * invS,
            c.y * invS,
            c.z * invS,
            s * 0.5f
        );
    }
    
    // Create quaternion that looks in a direction
    static Quat lookRotation(const Vec3& forward, const Vec3& up = Vec3::up()) {
        Vec3 f = forward.normalized();
        Vec3 r = up.cross(f).normalized();
        Vec3 u = f.cross(r);
        
        // Convert rotation matrix to quaternion
        f32 trace = r.x + u.y + f.z;
        
        if (trace > 0.0f) {
            f32 s = 0.5f / std::sqrt(trace + 1.0f);
            return Quat(
                (u.z - f.y) * s,
                (f.x - r.z) * s,
                (r.y - u.x) * s,
                0.25f / s
            );
        } else if (r.x > u.y && r.x > f.z) {
            f32 s = 2.0f * std::sqrt(1.0f + r.x - u.y - f.z);
            return Quat(
                0.25f * s,
                (u.x + r.y) / s,
                (f.x + r.z) / s,
                (u.z - f.y) / s
            );
        } else if (u.y > f.z) {
            f32 s = 2.0f * std::sqrt(1.0f + u.y - r.x - f.z);
            return Quat(
                (u.x + r.y) / s,
                0.25f * s,
                (f.y + u.z) / s,
                (f.x - r.z) / s
            );
        } else {
            f32 s = 2.0f * std::sqrt(1.0f + f.z - r.x - u.y);
            return Quat(
                (f.x + r.z) / s,
                (f.y + u.z) / s,
                0.25f * s,
                (r.y - u.x) / s
            );
        }
    }
    
    // =========================================================================
    // Accessors
    // =========================================================================
    
    f32& operator[](usize i) { return data[i]; }
    const f32& operator[](usize i) const { return data[i]; }
    
    // =========================================================================
    // Quaternion Operations
    // =========================================================================
    
    // Quaternion multiplication (combines rotations)
    Quat operator*(const Quat& other) const {
        return Quat(
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y + y * other.w + z * other.x - x * other.z,
            w * other.z + z * other.w + x * other.y - y * other.x,
            w * other.w - x * other.x - y * other.y - z * other.z
        );
    }
    
    Quat& operator*=(const Quat& other) {
        *this = *this * other;
        return *this;
    }
    
    // Rotate a vector by this quaternion
    Vec3 operator*(const Vec3& v) const {
        // Optimized quaternion-vector rotation
        // q * v * q^-1 = v + 2w(u × v) + 2(u × (u × v))
        Vec3 u(x, y, z);
        Vec3 uv = u.cross(v);
        Vec3 uuv = u.cross(uv);
        return v + ((uv * w) + uuv) * 2.0f;
    }
    
    // Conjugate (inverse for unit quaternion)
    Quat conjugate() const {
        return Quat(-x, -y, -z, w);
    }
    
    // Inverse (same as conjugate for unit quaternion)
    Quat inverse() const {
        return conjugate();
    }
    
    // =========================================================================
    // Normalization
    // =========================================================================
    
    f32 lengthSq() const {
        return x * x + y * y + z * z + w * w;
    }
    
    f32 length() const {
        return std::sqrt(lengthSq());
    }
    
    Quat normalized() const {
        f32 len = length();
        if (len > Math::EPSILON) {
            f32 invLen = 1.0f / len;
            return Quat(x * invLen, y * invLen, z * invLen, w * invLen);
        }
        return identity();
    }
    
    void normalize() {
        *this = normalized();
    }
    
    bool isNormalized(f32 epsilon = Math::EPSILON) const {
        return Math::isNearlyEqual(lengthSq(), 1.0f, epsilon);
    }
    
    // =========================================================================
    // Interpolation
    // =========================================================================
    
    // Spherical linear interpolation
    static Quat slerp(const Quat& a, const Quat& b, f32 t) {
        f32 d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        
        Quat target = b;
        
        // If dot product is negative, negate one quaternion to take shortest path
        if (d < 0.0f) {
            d = -d;
            target = Quat(-b.x, -b.y, -b.z, -b.w);
        }
        
        // If quaternions are very close, use linear interpolation
        if (d > 0.9995f) {
            Quat result(
                a.x + t * (target.x - a.x),
                a.y + t * (target.y - a.y),
                a.z + t * (target.z - a.z),
                a.w + t * (target.w - a.w)
            );
            return result.normalized();
        }
        
        f32 theta0 = std::acos(d);
        f32 theta = theta0 * t;
        f32 sinTheta = std::sin(theta);
        f32 sinTheta0 = std::sin(theta0);
        
        f32 s0 = std::cos(theta) - d * sinTheta / sinTheta0;
        f32 s1 = sinTheta / sinTheta0;
        
        return Quat(
            a.x * s0 + target.x * s1,
            a.y * s0 + target.y * s1,
            a.z * s0 + target.z * s1,
            a.w * s0 + target.w * s1
        );
    }
    
    // Normalized linear interpolation (faster than slerp, good for small angles)
    static Quat nlerp(const Quat& a, const Quat& b, f32 t) {
        f32 d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        f32 sign = (d < 0.0f) ? -1.0f : 1.0f;
        
        Quat result(
            a.x + t * (b.x * sign - a.x),
            a.y + t * (b.y * sign - a.y),
            a.z + t * (b.z * sign - a.z),
            a.w + t * (b.w * sign - a.w)
        );
        return result.normalized();
    }
    
    // =========================================================================
    // Conversion
    // =========================================================================
    
    // Get rotation axis and angle
    void toAxisAngle(Vec3& axis, f32& angle) const {
        angle = 2.0f * std::acos(Math::clamp(w, -1.0f, 1.0f));
        f32 s = std::sqrt(1.0f - w * w);
        
        if (s < Math::EPSILON) {
            axis = Vec3::unitX();
        } else {
            f32 invS = 1.0f / s;
            axis = Vec3(x * invS, y * invS, z * invS);
        }
    }
    
    // Get Euler angles (pitch, yaw, roll) in radians
    Vec3 toEuler() const {
        Vec3 euler;
        
        // Roll (x-axis rotation)
        f32 sinr_cosp = 2.0f * (w * x + y * z);
        f32 cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
        euler.z = std::atan2(sinr_cosp, cosr_cosp);
        
        // Pitch (y-axis rotation)
        f32 sinp = 2.0f * (w * y - z * x);
        if (Math::abs(sinp) >= 1.0f) {
            euler.x = std::copysign(Math::HALF_PI, sinp);  // Gimbal lock
        } else {
            euler.x = std::asin(sinp);
        }
        
        // Yaw (z-axis rotation)
        f32 siny_cosp = 2.0f * (w * z + x * y);
        f32 cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
        euler.y = std::atan2(siny_cosp, cosy_cosp);
        
        return euler;
    }
    
    Vec3 toEulerDegrees() const {
        Vec3 euler = toEuler();
        return Vec3(
            Math::degrees(euler.x),
            Math::degrees(euler.y),
            Math::degrees(euler.z)
        );
    }
    
    // Get direction vectors
    Vec3 forward() const { return *this * Vec3::forward(); }
    Vec3 right() const { return *this * Vec3::right(); }
    Vec3 up() const { return *this * Vec3::up(); }
    
    // =========================================================================
    // Comparison
    // =========================================================================
    
    bool operator==(const Quat& other) const {
        return Math::isNearlyEqual(x, other.x) && 
               Math::isNearlyEqual(y, other.y) && 
               Math::isNearlyEqual(z, other.z) &&
               Math::isNearlyEqual(w, other.w);
    }
    
    bool operator!=(const Quat& other) const {
        return !(*this == other);
    }
    
    // Check if represents same rotation (q and -q represent same rotation)
    bool isEquivalent(const Quat& other, f32 epsilon = Math::EPSILON) const {
        f32 dot = x * other.x + y * other.y + z * other.z + w * other.w;
        return Math::abs(Math::abs(dot) - 1.0f) < epsilon;
    }
    
    // =========================================================================
    // Angular Operations
    // =========================================================================
    
    // Get angle between this quaternion and another (in radians)
    f32 angleTo(const Quat& other) const {
        f32 d = Math::abs(x * other.x + y * other.y + z * other.z + w * other.w);
        d = Math::clamp(d, 0.0f, 1.0f);
        return 2.0f * std::acos(d);
    }
    
    // Get the angular velocity to rotate from this to target in one second
    Vec3 angularVelocityTo(const Quat& target) const {
        Quat delta = target * conjugate();
        Vec3 axis;
        f32 angle;
        delta.toAxisAngle(axis, angle);
        return axis * angle;
    }
};

// =============================================================================
// Free Functions
// =============================================================================

inline Quat slerp(const Quat& a, const Quat& b, f32 t) { return Quat::slerp(a, b, t); }
inline Quat nlerp(const Quat& a, const Quat& b, f32 t) { return Quat::nlerp(a, b, t); }
inline Quat normalize(const Quat& q) { return q.normalized(); }
inline Quat inverse(const Quat& q) { return q.inverse(); }
inline Quat conjugate(const Quat& q) { return q.conjugate(); }

// Ensure proper alignment
static_assert(sizeof(Quat) == 16, "Quat must be 16 bytes");
static_assert(alignof(Quat) == 16, "Quat must be 16-byte aligned");

} // namespace WulfNet
