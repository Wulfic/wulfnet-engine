// =============================================================================
// WulfNet Engine - Collision Shape Base Class
// =============================================================================
// Abstract base class for all collision shapes.
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/Log.h"
#include "core/Assert.h"
#include "core/math/Vec3.h"
#include "core/math/Mat4.h"
#include "CollisionTypes.h"
#include "AABB.h"
#include "BoundingSphere.h"
#include <cmath>
#include <vector>

namespace WulfNet {

// =============================================================================
// Collision Shape Interface
// =============================================================================

class CollisionShape : public NonCopyable {
public:
    virtual ~CollisionShape() = default;
    
    // Type Information
    virtual ShapeType getType() const = 0;
    virtual const char* getTypeName() const { return shapeTypeToString(getType()); }
    
    // Bounds
    virtual AABB getLocalAABB() const = 0;
    virtual BoundingSphere getLocalBoundingSphere() const = 0;
    
    AABB getWorldAABB(const Mat4& transform) const {
        return getLocalAABB().transformed(transform);
    }
    
    BoundingSphere getWorldBoundingSphere(const Mat4& transform) const {
        return getLocalBoundingSphere().transformed(transform);
    }
    
    // Properties
    virtual f32 getVolume() const = 0;
    virtual Vec3 getCenterOfMass() const { return Vec3::zero(); }
    virtual Mat4 getLocalInertiaTensor() const = 0;
    
    // Raycast (in local space)
    virtual bool raycast(const Vec3& origin, const Vec3& direction, 
                        f32& hitDistance, Vec3& hitNormal) const = 0;
    
    // Support function for GJK (in local space)
    virtual Vec3 support(const Vec3& direction) const = 0;
};

// =============================================================================
// Sphere Shape
// =============================================================================

class SphereShape : public CollisionShape {
public:
    explicit SphereShape(f32 radius) 
        : m_radius(radius)
    {
        WULFNET_LOG_DEBUG("SphereShape", "Created with radius={:.3f}", radius);
    }
    
    ShapeType getType() const override { return ShapeType::Sphere; }
    
    f32 getRadius() const { return m_radius; }
    void setRadius(f32 r) { m_radius = r; }
    
    AABB getLocalAABB() const override {
        return AABB::fromSphere(Vec3::zero(), m_radius);
    }
    
    BoundingSphere getLocalBoundingSphere() const override {
        return BoundingSphere(Vec3::zero(), m_radius);
    }
    
    f32 getVolume() const override {
        return (4.0f / 3.0f) * Math::PI * m_radius * m_radius * m_radius;
    }
    
    Mat4 getLocalInertiaTensor() const override {
        f32 i = (2.0f / 5.0f) * m_radius * m_radius;
        return Mat4::scale(Vec3(i, i, i));
    }
    
    bool raycast(const Vec3& origin, const Vec3& direction,
                f32& hitDistance, Vec3& hitNormal) const override {
        f32 a = direction.dot(direction);
        f32 b = 2.0f * origin.dot(direction);
        f32 c = origin.dot(origin) - m_radius * m_radius;
        
        f32 discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0) {
            return false;
        }
        
        hitDistance = (-b - std::sqrt(discriminant)) / (2.0f * a);
        if (hitDistance < 0) {
            hitDistance = (-b + std::sqrt(discriminant)) / (2.0f * a);
            if (hitDistance < 0) {
                return false;
            }
        }
        
        Vec3 hitPoint = origin + direction * hitDistance;
        hitNormal = hitPoint.normalized();
        
        WULFNET_LOG_TRACE("SphereShape", "Ray hit at distance={:.3f}", hitDistance);
        return true;
    }
    
    Vec3 support(const Vec3& direction) const override {
        return direction.normalized() * m_radius;
    }
    
private:
    f32 m_radius;
};

// =============================================================================
// Box Shape
// =============================================================================

class BoxShape : public CollisionShape {
public:
    explicit BoxShape(const Vec3& halfExtents)
        : m_halfExtents(halfExtents)
    {
        WULFNET_LOG_DEBUG("BoxShape", "Created with halfExtents=({:.3f},{:.3f},{:.3f})",
            halfExtents.x, halfExtents.y, halfExtents.z);
    }
    
    BoxShape(f32 halfX, f32 halfY, f32 halfZ)
        : m_halfExtents(halfX, halfY, halfZ)
    {
        WULFNET_LOG_DEBUG("BoxShape", "Created with halfExtents=({:.3f},{:.3f},{:.3f})",
            halfX, halfY, halfZ);
    }
    
    ShapeType getType() const override { return ShapeType::Box; }
    
    const Vec3& getHalfExtents() const { return m_halfExtents; }
    void setHalfExtents(const Vec3& he) { m_halfExtents = he; }
    
    AABB getLocalAABB() const override {
        return AABB(-m_halfExtents, m_halfExtents);
    }
    
    BoundingSphere getLocalBoundingSphere() const override {
        return BoundingSphere(Vec3::zero(), m_halfExtents.length());
    }
    
    f32 getVolume() const override {
        return 8.0f * m_halfExtents.x * m_halfExtents.y * m_halfExtents.z;
    }
    
    Mat4 getLocalInertiaTensor() const override {
        Vec3 size = m_halfExtents * 2.0f;
        f32 ix = (1.0f / 12.0f) * (size.y * size.y + size.z * size.z);
        f32 iy = (1.0f / 12.0f) * (size.x * size.x + size.z * size.z);
        f32 iz = (1.0f / 12.0f) * (size.x * size.x + size.y * size.y);
        return Mat4::scale(Vec3(ix, iy, iz));
    }
    
    bool raycast(const Vec3& origin, const Vec3& direction,
                f32& hitDistance, Vec3& hitNormal) const override {
        AABB aabb = getLocalAABB();
        
        Vec3 dirInv(
            std::abs(direction.x) > Math::EPSILON ? 1.0f / direction.x : (direction.x >= 0 ? Math::LARGE_NUM : -Math::LARGE_NUM),
            std::abs(direction.y) > Math::EPSILON ? 1.0f / direction.y : (direction.y >= 0 ? Math::LARGE_NUM : -Math::LARGE_NUM),
            std::abs(direction.z) > Math::EPSILON ? 1.0f / direction.z : (direction.z >= 0 ? Math::LARGE_NUM : -Math::LARGE_NUM)
        );
        
        Vec3 t1 = (aabb.min - origin) * dirInv;
        Vec3 t2 = (aabb.max - origin) * dirInv;
        
        Vec3 tNear = t1.min(t2);
        Vec3 tFar = t1.max(t2);
        
        hitDistance = std::max({tNear.x, tNear.y, tNear.z});
        f32 tMax = std::min({tFar.x, tFar.y, tFar.z});
        
        if (hitDistance > tMax || hitDistance < 0) {
            return false;
        }
        
        // Determine which face was hit
        if (tNear.x >= tNear.y && tNear.x >= tNear.z) {
            hitNormal = Vec3(direction.x < 0 ? 1.0f : -1.0f, 0, 0);
        } else if (tNear.y >= tNear.z) {
            hitNormal = Vec3(0, direction.y < 0 ? 1.0f : -1.0f, 0);
        } else {
            hitNormal = Vec3(0, 0, direction.z < 0 ? 1.0f : -1.0f);
        }
        
        WULFNET_LOG_TRACE("BoxShape", "Ray hit at distance={:.3f}", hitDistance);
        return true;
    }
    
    Vec3 support(const Vec3& direction) const override {
        return Vec3(
            direction.x >= 0 ? m_halfExtents.x : -m_halfExtents.x,
            direction.y >= 0 ? m_halfExtents.y : -m_halfExtents.y,
            direction.z >= 0 ? m_halfExtents.z : -m_halfExtents.z
        );
    }
    
private:
    Vec3 m_halfExtents;
};

// =============================================================================
// Capsule Shape
// =============================================================================

class CapsuleShape : public CollisionShape {
public:
    CapsuleShape(f32 radius, f32 halfHeight) 
        : m_radius(radius)
        , m_halfHeight(halfHeight)
    {
        WULFNET_LOG_DEBUG("CapsuleShape", "Created with radius={:.3f} halfHeight={:.3f}",
            radius, halfHeight);
    }
    
    ShapeType getType() const override { return ShapeType::Capsule; }
    
    f32 getRadius() const { return m_radius; }
    f32 getHalfHeight() const { return m_halfHeight; }
    f32 getTotalHeight() const { return m_halfHeight * 2.0f + m_radius * 2.0f; }
    
    void setRadius(f32 r) { m_radius = r; }
    void setHalfHeight(f32 h) { m_halfHeight = h; }
    
    AABB getLocalAABB() const override {
        f32 totalHalfHeight = m_halfHeight + m_radius;
        return AABB(
            Vec3(-m_radius, -totalHalfHeight, -m_radius),
            Vec3(m_radius, totalHalfHeight, m_radius)
        );
    }
    
    BoundingSphere getLocalBoundingSphere() const override {
        return BoundingSphere(Vec3::zero(), m_halfHeight + m_radius);
    }
    
    f32 getVolume() const override {
        f32 sphereVol = (4.0f / 3.0f) * Math::PI * m_radius * m_radius * m_radius;
        f32 cylinderVol = Math::PI * m_radius * m_radius * m_halfHeight * 2.0f;
        return sphereVol + cylinderVol;
    }
    
    Mat4 getLocalInertiaTensor() const override {
        // Approximate as cylinder + two hemispheres
        f32 r2 = m_radius * m_radius;
        f32 h = m_halfHeight * 2.0f;
        
        f32 cylinderIy = (1.0f / 12.0f) * (3.0f * r2 + h * h);
        f32 cylinderIxz = 0.5f * r2;
        
        // Approximate for now
        return Mat4::scale(Vec3(cylinderIy, cylinderIxz, cylinderIy));
    }
    
    bool raycast(const Vec3& origin, const Vec3& direction,
                f32& hitDistance, Vec3& hitNormal) const override {
        f32 bestT = Math::LARGE_NUM;
        Vec3 bestNormal;
        
        // Test top hemisphere
        Vec3 topCenter(0, m_halfHeight, 0);
        BoundingSphere topSphere(topCenter, m_radius);
        f32 t;
        if (topSphere.rayIntersects(origin, direction, t) && t < bestT && t >= 0) {
            bestT = t;
            bestNormal = (origin + direction * t - topCenter).normalized();
        }
        
        // Test bottom hemisphere
        Vec3 bottomCenter(0, -m_halfHeight, 0);
        BoundingSphere bottomSphere(bottomCenter, m_radius);
        if (bottomSphere.rayIntersects(origin, direction, t) && t < bestT && t >= 0) {
            bestT = t;
            bestNormal = (origin + direction * t - bottomCenter).normalized();
        }
        
        // Test cylinder
        f32 a = direction.x * direction.x + direction.z * direction.z;
        f32 b = 2.0f * (origin.x * direction.x + origin.z * direction.z);
        f32 c = origin.x * origin.x + origin.z * origin.z - m_radius * m_radius;
        
        f32 discriminant = b * b - 4.0f * a * c;
        if (discriminant >= 0 && a > Math::EPSILON) {
            f32 sqrtD = std::sqrt(discriminant);
            f32 t1 = (-b - sqrtD) / (2.0f * a);
            f32 t2 = (-b + sqrtD) / (2.0f * a);
            
            for (f32 tc : {t1, t2}) {
                if (tc >= 0 && tc < bestT) {
                    f32 y = origin.y + direction.y * tc;
                    if (y >= -m_halfHeight && y <= m_halfHeight) {
                        bestT = tc;
                        Vec3 hitPoint = origin + direction * tc;
                        bestNormal = Vec3(hitPoint.x, 0, hitPoint.z).normalized();
                    }
                }
            }
        }
        
        if (bestT < Math::LARGE_NUM) {
            hitDistance = bestT;
            hitNormal = bestNormal;
            return true;
        }
        
        return false;
    }
    
    Vec3 support(const Vec3& direction) const override {
        Vec3 capsuleDir = Vec3::unitY();
        Vec3 sphereCenter = capsuleDir * (direction.y >= 0 ? m_halfHeight : -m_halfHeight);
        return sphereCenter + direction.normalized() * m_radius;
    }
    
private:
    f32 m_radius;
    f32 m_halfHeight;
};

// =============================================================================
// Plane Shape (Infinite plane for static geometry)
// =============================================================================

class PlaneShape : public CollisionShape {
public:
    PlaneShape(const Vec3& normal, f32 distance)
        : m_normal(normal.normalized())
        , m_distance(distance)
    {
        WULFNET_LOG_DEBUG("PlaneShape", "Created with normal=({:.3f},{:.3f},{:.3f}) distance={:.3f}",
            m_normal.x, m_normal.y, m_normal.z, distance);
    }
    
    ShapeType getType() const override { return ShapeType::Plane; }
    
    const Vec3& getNormal() const { return m_normal; }
    f32 getDistance() const { return m_distance; }
    
    void setNormal(const Vec3& n) { m_normal = n.normalized(); }
    void setDistance(f32 d) { m_distance = d; }
    
    AABB getLocalAABB() const override {
        // Plane is infinite - return large AABB
        constexpr f32 size = 1e6f;
        return AABB(Vec3(-size), Vec3(size));
    }
    
    BoundingSphere getLocalBoundingSphere() const override {
        return BoundingSphere(Vec3::zero(), 1e6f);
    }
    
    f32 getVolume() const override {
        return 0.0f;  // Planes have no volume
    }
    
    Mat4 getLocalInertiaTensor() const override {
        return Mat4::identity();  // Infinite inertia effectively
    }
    
    bool raycast(const Vec3& origin, const Vec3& direction,
                f32& hitDistance, Vec3& hitNormal) const override {
        f32 denom = m_normal.dot(direction);
        if (std::abs(denom) < Math::EPSILON) {
            return false;  // Ray parallel to plane
        }
        
        Vec3 planePoint = m_normal * m_distance;
        hitDistance = (planePoint - origin).dot(m_normal) / denom;
        
        if (hitDistance >= 0) {
            hitNormal = denom < 0 ? m_normal : -m_normal;
            return true;
        }
        
        return false;
    }
    
    Vec3 support(const Vec3& direction) const override {
        // For infinite plane, project direction onto plane
        constexpr f32 size = 1e6f;
        if (direction.dot(m_normal) > 0) {
            return m_normal * m_distance + direction * size;
        } else {
            return m_normal * m_distance - direction * size;
        }
    }
    
    // Utility: distance from point to plane
    f32 distanceToPoint(const Vec3& point) const {
        return m_normal.dot(point) - m_distance;
    }
    
    Vec3 closestPointOnPlane(const Vec3& point) const {
        return point - m_normal * distanceToPoint(point);
    }
    
private:
    Vec3 m_normal;
    f32 m_distance;
};

// =============================================================================
// Convex Hull Shape
// =============================================================================

class ConvexHullShape : public CollisionShape {
public:
    ConvexHullShape(const std::vector<Vec3>& vertices)
        : m_vertices(vertices)
    {
        WULFNET_ASSERT_MSG(!vertices.empty(), "ConvexHullShape must have at least one vertex");
        recalcBounds();
        WULFNET_LOG_DEBUG("ConvexHullShape", "Created with {} vertices", vertices.size());
    }
    
    // Move constructor
    ConvexHullShape(std::vector<Vec3>&& vertices)
        : m_vertices(std::move(vertices))
    {
        WULFNET_ASSERT_MSG(!m_vertices.empty(), "ConvexHullShape must have at least one vertex");
        recalcBounds();
        WULFNET_LOG_DEBUG("ConvexHullShape", "Created with {} vertices (move)", m_vertices.size());
    }

    ShapeType getType() const override { return ShapeType::ConvexHull; }
    
    const std::vector<Vec3>& getVertices() const { return m_vertices; }
    
    AABB getLocalAABB() const override { return m_aabb; }
    
    BoundingSphere getLocalBoundingSphere() const override { return m_boundingSphere; }
    
    f32 getVolume() const override {
        // Approximate volume using AABB (50% of AABB volume is a rough heuristic)
        Vec3 size = m_aabb.getSize();
        return size.x * size.y * size.z * 0.5f;
    }
    
    Mat4 getLocalInertiaTensor() const override {
        // Approximate as a box
        Vec3 size = m_aabb.getSize();
        f32 x2 = size.x * size.x;
        f32 y2 = size.y * size.y;
        f32 z2 = size.z * size.z;
        
        f32 ix = (1.0f / 12.0f) * (y2 + z2);
        f32 iy = (1.0f / 12.0f) * (x2 + z2);
        f32 iz = (1.0f / 12.0f) * (x2 + y2);
        return Mat4::scale(Vec3(ix, iy, iz));
    }
    
    bool raycast(const Vec3& origin, const Vec3& direction,
                f32& hitDistance, Vec3& hitNormal) const override {
        // Fallback: AABB raycast
        // Note: This is an approximation. True convex hull raycast requires face normals.
        Vec3 dirInv(
            std::abs(direction.x) > Math::EPSILON ? 1.0f / direction.x : (direction.x >= 0 ? Math::LARGE_NUM : -Math::LARGE_NUM),
            std::abs(direction.y) > Math::EPSILON ? 1.0f / direction.y : (direction.y >= 0 ? Math::LARGE_NUM : -Math::LARGE_NUM),
            std::abs(direction.z) > Math::EPSILON ? 1.0f / direction.z : (direction.z >= 0 ? Math::LARGE_NUM : -Math::LARGE_NUM)
        );

        f32 tMin, tMax;
        if (m_aabb.rayIntersects(origin, dirInv, tMin, tMax)) {
            if (tMin >= 0) {
                hitDistance = tMin;
                // Approximate normal
                hitNormal = -direction.normalized(); 
                return true;
            }
        }
        return false;
    }
    
    Vec3 support(const Vec3& direction) const override {
        // Optimized support function
        // Start with the first vertex
        f32 maxDot = m_vertices[0].dot(direction);
        Vec3 bestPoint = m_vertices[0];
        
        // Iterate through all vertices
        for (size_t i = 1; i < m_vertices.size(); ++i) {
            f32 dot = m_vertices[i].dot(direction);
            if (dot > maxDot) {
                maxDot = dot;
                bestPoint = m_vertices[i];
            }
        }
        return bestPoint;
    }

private:
    void recalcBounds() {
        Vec3 min(Math::LARGE_NUM);
        Vec3 max(-Math::LARGE_NUM);
        
        for (const auto& v : m_vertices) {
            min = min.min(v);
            max = max.max(v);
        }
        m_aabb = AABB(min, max);
        
        Vec3 center = m_aabb.getCenter();
        f32 maxDistSq = 0.0f;
        for (const auto& v : m_vertices) {
            f32 d = (v - center).lengthSq();
            if (d > maxDistSq) {
                maxDistSq = d;
            }
        }
        m_boundingSphere = BoundingSphere(center, std::sqrt(maxDistSq));
    }

    std::vector<Vec3> m_vertices;
    AABB m_aabb;
    BoundingSphere m_boundingSphere;
};

} // namespace WulfNet
