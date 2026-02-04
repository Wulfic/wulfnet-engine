// =============================================================================
// WulfNet Engine - Bounding Sphere
// =============================================================================
// Simple sphere bounds for collision detection.
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/Log.h"
#include "core/math/Vec3.h"
#include "core/math/Mat4.h"
#include "AABB.h"

namespace WulfNet {

// =============================================================================
// Bounding Sphere Structure
// =============================================================================

struct BoundingSphere {
    Vec3 center;
    f32 radius;
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    BoundingSphere()
        : center(Vec3::zero())
        , radius(0.0f)
    {}
    
    BoundingSphere(const Vec3& c, f32 r)
        : center(c)
        , radius(r)
    {}
    
    // Create from AABB (enclosing sphere)
    static BoundingSphere fromAABB(const AABB& aabb) {
        Vec3 c = aabb.center();
        f32 r = aabb.halfExtents().length();
        return BoundingSphere(c, r);
    }
    
    // =========================================================================
    // Properties
    // =========================================================================
    
    f32 volume() const {
        return (4.0f / 3.0f) * Math::PI * radius * radius * radius;
    }
    
    f32 surfaceArea() const {
        return 4.0f * Math::PI * radius * radius;
    }
    
    AABB toAABB() const {
        return AABB::fromSphere(center, radius);
    }
    
    // =========================================================================
    // Intersection Tests
    // =========================================================================
    
    bool intersects(const BoundingSphere& other) const {
        f32 distSq = (center - other.center).lengthSq();
        f32 radiusSum = radius + other.radius;
        return distSq <= radiusSum * radiusSum;
    }
    
    bool intersects(const AABB& aabb) const {
        // Find closest point on AABB to sphere center
        Vec3 closest = center.max(aabb.min).min(aabb.max);
        f32 distSq = (center - closest).lengthSq();
        return distSq <= radius * radius;
    }
    
    bool contains(const Vec3& point) const {
        return (point - center).lengthSq() <= radius * radius;
    }
    
    bool contains(const BoundingSphere& other) const {
        f32 dist = (other.center - center).length();
        return dist + other.radius <= radius;
    }
    
    // Ray intersection
    bool rayIntersects(const Vec3& origin, const Vec3& direction, f32& t) const {
        Vec3 oc = origin - center;
        f32 a = direction.dot(direction);
        f32 b = 2.0f * oc.dot(direction);
        f32 c = oc.dot(oc) - radius * radius;
        
        f32 discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0) {
            return false;
        }
        
        t = (-b - std::sqrt(discriminant)) / (2.0f * a);
        if (t < 0) {
            t = (-b + std::sqrt(discriminant)) / (2.0f * a);
        }
        
        return t >= 0;
    }
    
    // =========================================================================
    // Operations
    // =========================================================================
    
    void expand(const Vec3& point) {
        f32 dist = (point - center).length();
        if (dist > radius) {
            f32 newRadius = (radius + dist) * 0.5f;
            f32 k = (newRadius - radius) / dist;
            radius = newRadius;
            center = center + (point - center) * k;
        }
    }
    
    void expand(const BoundingSphere& other) {
        Vec3 diff = other.center - center;
        f32 dist = diff.length();
        
        if (dist + other.radius <= radius) {
            // Other is inside this
            return;
        }
        
        if (dist + radius <= other.radius) {
            // This is inside other
            center = other.center;
            radius = other.radius;
            return;
        }
        
        // Merge spheres
        f32 newRadius = (dist + radius + other.radius) * 0.5f;
        if (dist > 0) {
            center = center + diff * ((newRadius - radius) / dist);
        }
        radius = newRadius;
    }
    
    BoundingSphere transformed(const Mat4& transform) const {
        // Transform center
        Vec4 c4(center.x, center.y, center.z, 1.0f);
        Vec4 newCenter4 = transform * c4;
        Vec3 newCenter(newCenter4.x, newCenter4.y, newCenter4.z);
        
        // Scale radius by maximum scale factor
        // Extract scale from transform by transforming unit vectors
        Vec4 sx = transform * Vec4(1, 0, 0, 0);
        Vec4 sy = transform * Vec4(0, 1, 0, 0);
        Vec4 sz = transform * Vec4(0, 0, 1, 0);
        
        f32 scaleX = Vec3(sx.x, sx.y, sx.z).length();
        f32 scaleY = Vec3(sy.x, sy.y, sy.z).length();
        f32 scaleZ = Vec3(sz.x, sz.y, sz.z).length();
        f32 maxScale = Math::max(Math::max(scaleX, scaleY), scaleZ);
        
        return BoundingSphere(newCenter, radius * maxScale);
    }
    
    // Merge two spheres
    static BoundingSphere merge(const BoundingSphere& a, const BoundingSphere& b) {
        BoundingSphere result = a;
        result.expand(b);
        return result;
    }
};

} // namespace WulfNet
