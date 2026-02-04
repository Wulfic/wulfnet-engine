// =============================================================================
// WulfNet Engine - Math Library
// =============================================================================
// Unified header for all math types
// =============================================================================

#pragma once

#include "SIMD.h"
#include "MathUtils.h"
#include "Vec3.h"
#include "Vec4.h"
#include "Quat.h"
#include "Mat4.h"

namespace WulfNet {

// =============================================================================
// Additional Math Types (Compact versions for storage)
// =============================================================================

// Compact Vec3 for storage (no padding)
struct Vec3Packed {
    f32 x, y, z;
    
    Vec3Packed() : x(0), y(0), z(0) {}
    Vec3Packed(f32 x_, f32 y_, f32 z_) : x(x_), y(y_), z(z_) {}
    Vec3Packed(const Vec3& v) : x(v.x), y(v.y), z(v.z) {}
    
    operator Vec3() const { return Vec3(x, y, z); }
};

// Compact Vec4 for storage (no alignment requirement)
struct Vec4Packed {
    f32 x, y, z, w;
    
    Vec4Packed() : x(0), y(0), z(0), w(0) {}
    Vec4Packed(f32 x_, f32 y_, f32 z_, f32 w_) : x(x_), y(y_), z(z_), w(w_) {}
    Vec4Packed(const Vec4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
    
    operator Vec4() const { return Vec4(x, y, z, w); }
};

// =============================================================================
// Transform Structure (Position, Rotation, Scale)
// =============================================================================

struct Transform {
    Vec3 position;
    Quat rotation;
    Vec3 scale;
    
    Transform() 
        : position(Vec3::zero())
        , rotation(Quat::identity())
        , scale(Vec3::one()) 
    {}
    
    Transform(const Vec3& pos) 
        : position(pos)
        , rotation(Quat::identity())
        , scale(Vec3::one()) 
    {}
    
    Transform(const Vec3& pos, const Quat& rot) 
        : position(pos)
        , rotation(rot)
        , scale(Vec3::one()) 
    {}
    
    Transform(const Vec3& pos, const Quat& rot, const Vec3& scl) 
        : position(pos)
        , rotation(rot)
        , scale(scl) 
    {}
    
    Mat4 toMatrix() const {
        return Mat4::TRS(position, rotation, scale);
    }
    
    static Transform fromMatrix(const Mat4& matrix) {
        Transform t;
        t.position = matrix.getTranslation();
        t.scale = matrix.getScale();
        t.rotation = matrix.getRotation();
        return t;
    }
    
    Transform inverse() const {
        Quat invRot = rotation.inverse();
        Vec3 invScale = Vec3(1.0f / scale.x, 1.0f / scale.y, 1.0f / scale.z);
        Vec3 invPos = invRot * (position * -invScale);
        return Transform(invPos, invRot, invScale);
    }
    
    Vec3 transformPoint(const Vec3& point) const {
        return rotation * (point * scale) + position;
    }
    
    Vec3 transformDirection(const Vec3& dir) const {
        return rotation * dir;
    }
    
    Vec3 inverseTransformPoint(const Vec3& point) const {
        return rotation.inverse() * (point - position) / scale;
    }
    
    Vec3 inverseTransformDirection(const Vec3& dir) const {
        return rotation.inverse() * dir;
    }
    
    Transform operator*(const Transform& child) const {
        Transform result;
        result.scale = scale * child.scale;
        result.rotation = rotation * child.rotation;
        result.position = transformPoint(child.position);
        return result;
    }
    
    static Transform lerp(const Transform& a, const Transform& b, f32 t) {
        return Transform(
            WulfNet::lerp(a.position, b.position, t),
            Quat::slerp(a.rotation, b.rotation, t),
            WulfNet::lerp(a.scale, b.scale, t)
        );
    }
};

// =============================================================================
// Axis-Aligned Bounding Box (Math-only)
// =============================================================================

struct MathAABB {
    Vec3 min;
    Vec3 max;
    
    MathAABB() : min(Vec3(Math::LARGE_NUM)), max(Vec3(-Math::LARGE_NUM)) {}
    MathAABB(const Vec3& min_, const Vec3& max_) : min(min_), max(max_) {}
    
    static MathAABB fromCenterExtents(const Vec3& center, const Vec3& extents) {
        return MathAABB(center - extents, center + extents);
    }
    
    Vec3 center() const { return (min + max) * 0.5f; }
    Vec3 extents() const { return (max - min) * 0.5f; }
    Vec3 size() const { return max - min; }
    f32 volume() const { Vec3 s = size(); return s.x * s.y * s.z; }
    f32 surfaceArea() const { 
        Vec3 s = size(); 
        return 2.0f * (s.x * s.y + s.y * s.z + s.z * s.x); 
    }
    
    bool isValid() const { 
        return min.x <= max.x && min.y <= max.y && min.z <= max.z; 
    }
    
    bool contains(const Vec3& point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z;
    }
    
    bool contains(const MathAABB& other) const {
        return min.x <= other.min.x && max.x >= other.max.x &&
               min.y <= other.min.y && max.y >= other.max.y &&
               min.z <= other.min.z && max.z >= other.max.z;
    }
    
    bool intersects(const MathAABB& other) const {
        return min.x <= other.max.x && max.x >= other.min.x &&
               min.y <= other.max.y && max.y >= other.min.y &&
               min.z <= other.max.z && max.z >= other.min.z;
    }
    
    MathAABB merged(const MathAABB& other) const {
        return MathAABB(min.min(other.min), max.max(other.max));
    }
    
    MathAABB expanded(f32 amount) const {
        Vec3 e(amount);
        return MathAABB(min - e, max + e);
    }
    
    void expand(const Vec3& point) {
        min = min.min(point);
        max = max.max(point);
    }
    
    void expand(const MathAABB& other) {
        min = min.min(other.min);
        max = max.max(other.max);
    }
    
    // Transform AABB (results in axis-aligned bounding box of transformed box)
    MathAABB transformed(const Mat4& matrix) const {
        Vec3 corners[8] = {
            Vec3(min.x, min.y, min.z),
            Vec3(max.x, min.y, min.z),
            Vec3(min.x, max.y, min.z),
            Vec3(max.x, max.y, min.z),
            Vec3(min.x, min.y, max.z),
            Vec3(max.x, min.y, max.z),
            Vec3(min.x, max.y, max.z),
            Vec3(max.x, max.y, max.z)
        };
        
        MathAABB result;
        for (int i = 0; i < 8; i++) {
            result.expand(matrix.transformPoint(corners[i]));
        }
        return result;
    }
};

// =============================================================================
// Sphere
// =============================================================================

struct Sphere {
    Vec3 center;
    f32 radius;
    
    Sphere() : center(Vec3::zero()), radius(0.0f) {}
    Sphere(const Vec3& c, f32 r) : center(c), radius(r) {}
    
    bool contains(const Vec3& point) const {
        return center.distanceSq(point) <= radius * radius;
    }
    
    bool intersects(const Sphere& other) const {
        f32 sumRadius = radius + other.radius;
        return center.distanceSq(other.center) <= sumRadius * sumRadius;
    }
    
    bool intersects(const MathAABB& aabb) const {
        // Find closest point on AABB to sphere center
        Vec3 closest = center.clamp(aabb.min, aabb.max);
        return center.distanceSq(closest) <= radius * radius;
    }
};

// =============================================================================
// Ray
// =============================================================================

struct Ray {
    Vec3 origin;
    Vec3 direction;  // Should be normalized
    
    Ray() : origin(Vec3::zero()), direction(Vec3::forward()) {}
    Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d.normalized()) {}
    
    Vec3 at(f32 t) const { return origin + direction * t; }
    
    // Ray-AABB intersection (returns t value, negative if no hit)
    f32 intersect(const MathAABB& aabb) const {
        Vec3 invDir(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
        
        Vec3 t1 = (aabb.min - origin) * invDir;
        Vec3 t2 = (aabb.max - origin) * invDir;
        
        Vec3 tMin = t1.min(t2);
        Vec3 tMax = t1.max(t2);
        
        f32 tNear = tMin.maxComponent();
        f32 tFar = tMax.minComponent();
        
        if (tNear > tFar || tFar < 0.0f) {
            return -1.0f;
        }
        
        return tNear >= 0.0f ? tNear : tFar;
    }
    
    // Ray-Sphere intersection
    f32 intersect(const Sphere& sphere) const {
        Vec3 oc = origin - sphere.center;
        f32 b = oc.dot(direction);
        f32 c = oc.dot(oc) - sphere.radius * sphere.radius;
        f32 discriminant = b * b - c;
        
        if (discriminant < 0.0f) {
            return -1.0f;
        }
        
        f32 sqrtD = std::sqrt(discriminant);
        f32 t1 = -b - sqrtD;
        f32 t2 = -b + sqrtD;
        
        if (t1 >= 0.0f) return t1;
        if (t2 >= 0.0f) return t2;
        return -1.0f;
    }
    
    // Ray-Plane intersection
    f32 intersect(const Vec3& planeNormal, f32 planeDistance) const {
        f32 denom = direction.dot(planeNormal);
        if (Math::abs(denom) < Math::EPSILON) {
            return -1.0f;  // Ray parallel to plane
        }
        
        f32 t = -(origin.dot(planeNormal) + planeDistance) / denom;
        return t >= 0.0f ? t : -1.0f;
    }
};

// =============================================================================
// Plane
// =============================================================================

struct Plane {
    Vec3 normal;
    f32 distance;  // Distance from origin (d in ax + by + cz + d = 0)
    
    Plane() : normal(Vec3::up()), distance(0.0f) {}
    Plane(const Vec3& n, f32 d) : normal(n.normalized()), distance(d) {}
    Plane(const Vec3& n, const Vec3& point) 
        : normal(n.normalized()), distance(-n.normalized().dot(point)) {}
    
    // From three points (counter-clockwise winding)
    static Plane fromPoints(const Vec3& a, const Vec3& b, const Vec3& c) {
        Vec3 normal = (b - a).cross(c - a).normalized();
        return Plane(normal, a);
    }
    
    f32 signedDistance(const Vec3& point) const {
        return normal.dot(point) + distance;
    }
    
    Vec3 closestPoint(const Vec3& point) const {
        return point - normal * signedDistance(point);
    }
    
    bool isOnPositiveSide(const Vec3& point) const {
        return signedDistance(point) > 0.0f;
    }
};

} // namespace WulfNet
