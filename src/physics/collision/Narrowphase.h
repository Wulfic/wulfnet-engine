// =============================================================================
// WulfNet Engine - Narrowphase Collision Detection
// =============================================================================
// Contact generation for primitive shapes.
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/Log.h"
#include "core/math/Vec3.h"
#include "core/math/Math.h"
#include "physics/dynamics/RigidBody.h"
#include "CollisionTypes.h"

namespace WulfNet {

class Narrowphase {
public:
    bool generateContacts(const RigidBody& bodyA, const RigidBody& bodyB, ContactManifold& outManifold) const;

private:
    bool sphereSphere(const RigidBody& a, const SphereShape& shapeA,
                      const RigidBody& b, const SphereShape& shapeB,
                      ContactManifold& outManifold) const;

    bool sphereBox(const RigidBody& sphereBody, const SphereShape& sphere,
                   const RigidBody& boxBody, const BoxShape& box,
                   ContactManifold& outManifold, bool sphereIsA) const;

    bool boxBoxApprox(const RigidBody& a, const BoxShape& shapeA,
                      const RigidBody& b, const BoxShape& shapeB,
                      ContactManifold& outManifold) const;

    bool sphereCapsule(const RigidBody& sphereBody, const SphereShape& sphere,
                       const RigidBody& capsuleBody, const CapsuleShape& capsule,
                       ContactManifold& outManifold, bool sphereIsA) const;

    bool capsuleCapsule(const RigidBody& a, const CapsuleShape& shapeA,
                        const RigidBody& b, const CapsuleShape& shapeB,
                        ContactManifold& outManifold) const;

    bool capsuleBoxApprox(const RigidBody& capsuleBody, const CapsuleShape& capsule,
                          const RigidBody& boxBody, const BoxShape& box,
                          ContactManifold& outManifold, bool capsuleIsA) const;

    bool convexConvex(const RigidBody& a, const CollisionShape& shapeA, 
                      const RigidBody& b, const CollisionShape& shapeB,
                      ContactManifold& outManifold) const;

    f32 computeWorldRadius(const SphereShape& sphere, const Transform& transform) const;

    void getCapsuleSegmentWorld(const RigidBody& body, const CapsuleShape& capsule,
                                Vec3& outP0, Vec3& outP1, f32& outRadius) const;

    static f32 distanceSqPointAABB(const Vec3& point, const AABB& aabb, Vec3& outClosest);
    static f32 distanceSqSegmentAABB(const Vec3& p0, const Vec3& p1, const AABB& aabb,
                                     Vec3& outSegPoint, Vec3& outAabbPoint);
    static void closestPointsSegmentSegment(const Vec3& p1, const Vec3& q1,
                                            const Vec3& p2, const Vec3& q2,
                                            Vec3& c1, Vec3& c2);
};

} // namespace WulfNet
