// =============================================================================
// WulfNet Engine - Axis-Aligned Bounding Box (AABB)
// =============================================================================
// SIMD-optimized AABB for broadphase collision detection.
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/Log.h"
#include "core/math/Vec3.h"
#include "core/math/Mat4.h"
#include <algorithm>
#include <cfloat>

namespace WulfNet {

// =============================================================================
// AABB Structure
// =============================================================================

struct alignas(16) AABB {
    Vec3 min;
    Vec3 max;
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    AABB() 
        : min(Vec3(FLT_MAX, FLT_MAX, FLT_MAX))
        , max(Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX)) 
    {
        // Invalid AABB by default (empty)
    }
    
    AABB(const Vec3& minPt, const Vec3& maxPt)
        : min(minPt)
        , max(maxPt)
    {}
    
    // Create AABB from center and half-extents
    static AABB fromCenterExtents(const Vec3& center, const Vec3& halfExtents) {
        return AABB(center - halfExtents, center + halfExtents);
    }
    
    // Create AABB from center and radius (sphere bound)
    static AABB fromSphere(const Vec3& center, f32 radius) {
        Vec3 r(radius, radius, radius);
        return AABB(center - r, center + r);
    }
    
    // Create AABB from two points
    static AABB fromPoints(const Vec3& a, const Vec3& b) {
        return AABB(a.min(b), a.max(b));
    }

    Vec3 getSize() const { return max - min; }
    Vec3 getCenter() const { return (min + max) * 0.5f; }
    
    // =========================================================================
    // Properties
    // =========================================================================
    
    Vec3 center() const {
        return (min + max) * 0.5f;
    }
    
    Vec3 extents() const {
        return max - min;
    }
    
    Vec3 halfExtents() const {
        return (max - min) * 0.5f;
    }
    
    f32 volume() const {
        Vec3 e = extents();
        return e.x * e.y * e.z;
    }
    
    f32 surfaceArea() const {
        Vec3 e = extents();
        return 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
    }
    
    bool isValid() const {
        return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }
    
    // =========================================================================
    // Intersection Tests
    // =========================================================================
    
    bool intersects(const AABB& other) const {
        WULFNET_LOG_TRACE("AABB", "Testing intersection: [({:.2f},{:.2f},{:.2f})-({:.2f},{:.2f},{:.2f})] vs [({:.2f},{:.2f},{:.2f})-({:.2f},{:.2f},{:.2f})]",
            min.x, min.y, min.z, max.x, max.y, max.z,
            other.min.x, other.min.y, other.min.z, other.max.x, other.max.y, other.max.z);
        
        if (max.x < other.min.x || min.x > other.max.x) return false;
        if (max.y < other.min.y || min.y > other.max.y) return false;
        if (max.z < other.min.z || min.z > other.max.z) return false;
        return true;
    }
    
    bool contains(const Vec3& point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z;
    }
    
    bool contains(const AABB& other) const {
        return other.min.x >= min.x && other.max.x <= max.x &&
               other.min.y >= min.y && other.max.y <= max.y &&
               other.min.z >= min.z && other.max.z <= max.z;
    }
    
    // Ray intersection using slab method
    bool rayIntersects(const Vec3& origin, const Vec3& dirInv, f32& tMin, f32& tMax) const {
        WULFNET_LOG_TRACE("AABB", "Ray intersection test from ({:.2f},{:.2f},{:.2f})",
            origin.x, origin.y, origin.z);
        
        Vec3 t1 = (min - origin) * dirInv;
        Vec3 t2 = (max - origin) * dirInv;
        
        Vec3 tNear = t1.min(t2);
        Vec3 tFar = t1.max(t2);
        
        tMin = std::max({tNear.x, tNear.y, tNear.z, 0.0f});
        tMax = std::min({tFar.x, tFar.y, tFar.z});
        
        return tMin <= tMax;
    }
    
    // =========================================================================
    // Operations
    // =========================================================================
    
    void expand(const Vec3& point) {
        min = min.min(point);
        max = max.max(point);
    }
    
    void expand(const AABB& other) {
        min = min.min(other.min);
        max = max.max(other.max);
    }
    
    void expand(f32 amount) {
        Vec3 a(amount, amount, amount);
        min = min - a;
        max = max + a;
    }
    
    AABB expanded(f32 amount) const {
        AABB result = *this;
        result.expand(amount);
        return result;
    }
    
    // Expand AABB in direction of movement (for swept tests)
    AABB swept(const Vec3& velocity) const {
        AABB result = *this;
        if (velocity.x > 0) result.max = Vec3(result.max.x + velocity.x, result.max.y, result.max.z);
        else result.min = Vec3(result.min.x + velocity.x, result.min.y, result.min.z);
        
        if (velocity.y > 0) result.max = Vec3(result.max.x, result.max.y + velocity.y, result.max.z);
        else result.min = Vec3(result.min.x, result.min.y + velocity.y, result.min.z);
        
        if (velocity.z > 0) result.max = Vec3(result.max.x, result.max.y, result.max.z + velocity.z);
        else result.min = Vec3(result.min.x, result.min.y, result.min.z + velocity.z);
        
        return result;
    }
    
    // Transform AABB by matrix (produces new AABB that encloses transformed corners)
    AABB transformed(const Mat4& transform) const {
        // Get all 8 corners
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
        
        AABB result;
        for (int i = 0; i < 8; ++i) {
            Vec4 pt4(corners[i].x, corners[i].y, corners[i].z, 1.0f);
            Vec4 transformed = transform * pt4;
            result.expand(Vec3(transformed.x, transformed.y, transformed.z));
        }
        
        return result;
    }
    
    // Merge two AABBs
    static AABB merge(const AABB& a, const AABB& b) {
        return AABB(a.min.min(b.min), a.max.max(b.max));
    }
    
    // Intersection of two AABBs
    static AABB intersection(const AABB& a, const AABB& b) {
        return AABB(a.min.max(b.min), a.max.min(b.max));
    }
};

} // namespace WulfNet
