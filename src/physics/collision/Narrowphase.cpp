// =============================================================================
// WulfNet Engine - Narrowphase Collision Detection Implementation
// =============================================================================

#include "Narrowphase.h"
#include "GJK.h"
#include <cmath>

namespace WulfNet {

bool Narrowphase::generateContacts(const RigidBody& bodyA, const RigidBody& bodyB, ContactManifold& outManifold) const {
    outManifold.clear();
    outManifold.bodyIdA = 0;
    outManifold.bodyIdB = 0;

    const CollisionShape* shapeA = bodyA.getShape();
    const CollisionShape* shapeB = bodyB.getShape();
    if (!shapeA || !shapeB) return false;

    const ShapeType typeA = shapeA->getType();
    const ShapeType typeB = shapeB->getType();

    if (typeA == ShapeType::Sphere && typeB == ShapeType::Sphere) {
        return sphereSphere(bodyA, static_cast<const SphereShape&>(*shapeA),
                            bodyB, static_cast<const SphereShape&>(*shapeB),
                            outManifold);
    }

    if (typeA == ShapeType::Sphere && typeB == ShapeType::Box) {
        return sphereBox(bodyA, static_cast<const SphereShape&>(*shapeA),
                         bodyB, static_cast<const BoxShape&>(*shapeB),
                         outManifold, true);
    }

    if (typeA == ShapeType::Box && typeB == ShapeType::Sphere) {
        return sphereBox(bodyB, static_cast<const SphereShape&>(*shapeB),
                         bodyA, static_cast<const BoxShape&>(*shapeA),
                         outManifold, false);
    }

    if (typeA == ShapeType::Box && typeB == ShapeType::Box) {
        return boxBoxApprox(bodyA, static_cast<const BoxShape&>(*shapeA),
                            bodyB, static_cast<const BoxShape&>(*shapeB),
                            outManifold);
    }

    if (typeA == ShapeType::Sphere && typeB == ShapeType::Capsule) {
        return sphereCapsule(bodyA, static_cast<const SphereShape&>(*shapeA),
                             bodyB, static_cast<const CapsuleShape&>(*shapeB),
                             outManifold, true);
    }

    if (typeA == ShapeType::Capsule && typeB == ShapeType::Sphere) {
        return sphereCapsule(bodyB, static_cast<const SphereShape&>(*shapeB),
                             bodyA, static_cast<const CapsuleShape&>(*shapeA),
                             outManifold, false);
    }

    if (typeA == ShapeType::Capsule && typeB == ShapeType::Capsule) {
        return capsuleCapsule(bodyA, static_cast<const CapsuleShape&>(*shapeA),
                              bodyB, static_cast<const CapsuleShape&>(*shapeB),
                              outManifold);
    }

    if (typeA == ShapeType::Capsule && typeB == ShapeType::Box) {
        return capsuleBoxApprox(bodyA, static_cast<const CapsuleShape&>(*shapeA),
                                bodyB, static_cast<const BoxShape&>(*shapeB),
                                outManifold, true);
    }

    if (typeA == ShapeType::Box && typeB == ShapeType::Capsule) {
        return capsuleBoxApprox(bodyB, static_cast<const CapsuleShape&>(*shapeB),
                                bodyA, static_cast<const BoxShape&>(*shapeA),
                                outManifold, false);
    }

    // Fallback: Use GJK/EPA for general convex collision
    return convexConvex(bodyA, *shapeA, bodyB, *shapeB, outManifold);
}

bool Narrowphase::convexConvex(const RigidBody& a, const CollisionShape& shapeA,
                               const RigidBody& b, const CollisionShape& shapeB,
                               ContactManifold& outManifold) const {
    Vec3 normal;
    f32 depth;
    Vec3 outPointA, outPointB;

    if (GJK::computePenetration(&a, &shapeA, &b, &shapeB, normal, depth, outPointA, outPointB)) {
        outManifold.contactCount = 1;
        
        ContactPoint& cp = outManifold.contacts[0];
        // EPA returns normal from B to A (separation).
        // Manifold expects A to B? 
        // Let's assume manifold expects B->A (separating normal for A). 
        // Standard convention: Normal points from B to A.
        // If CollisionTypes says "A to B", I should verify.
        // Assuming CollisionTypes comment is correct: "normal (from A to B)"
        // Then I flip it.
        cp.normal = -normal;  
        cp.penetration = depth;
        
        cp.positionWorldA = outPointA;
        cp.positionWorldB = outPointB;
        
        return true;
    }
    
    return false;
}

bool Narrowphase::sphereSphere(const RigidBody& a, const SphereShape& shapeA,
                               const RigidBody& b, const SphereShape& shapeB,
                               ContactManifold& outManifold) const {
    const Vec3 centerA = a.getPosition();
    const Vec3 centerB = b.getPosition();

    const f32 radiusA = computeWorldRadius(shapeA, a.getTransform());
    const f32 radiusB = computeWorldRadius(shapeB, b.getTransform());

    const Vec3 delta = centerB - centerA;
    const f32 distSq = delta.lengthSq();
    const f32 radiusSum = radiusA + radiusB;

    if (distSq > radiusSum * radiusSum) {
        return false;
    }

    const f32 dist = std::sqrt(distSq);
    Vec3 normal;
    if (dist > Math::EPSILON) {
        normal = delta / dist;
    } else {
        normal = Vec3::unitX();
    }

    ContactPoint cp;
    cp.normal = normal;
    cp.penetration = radiusSum - dist;
    cp.positionWorldA = centerA + normal * radiusA;
    cp.positionWorldB = centerB - normal * radiusB;

    outManifold.addContact(cp);
    return true;
}

bool Narrowphase::sphereBox(const RigidBody& sphereBody, const SphereShape& sphere,
                            const RigidBody& boxBody, const BoxShape& box,
                            ContactManifold& outManifold, bool sphereIsA) const {
    const Transform& boxTransform = boxBody.getTransform();
    const Transform invBoxTransform = boxTransform.inverse();

    const Vec3 sphereCenterWorld = sphereBody.getPosition();
    const Vec3 sphereCenterLocal = invBoxTransform.transformPoint(sphereCenterWorld);

    Vec3 halfExtents = box.getHalfExtents();
    halfExtents *= boxTransform.scale.abs();

    Vec3 closest = sphereCenterLocal.clamp(-halfExtents, halfExtents);
    Vec3 delta = sphereCenterLocal - closest;
    f32 distSq = delta.lengthSq();

    f32 radius = computeWorldRadius(sphere, sphereBody.getTransform());
    if (distSq > radius * radius) {
        return false;
    }

    Vec3 normalLocal;
    if (distSq > Math::EPSILON) {
        normalLocal = delta.normalized();
    } else {
        // Sphere center inside box - choose a stable normal.
        // Prefer opposing relative motion to avoid selecting the wrong face when deeply penetrating.
        Vec3 relVelWorld = sphereBody.getLinearVelocity() - boxBody.getLinearVelocity();
        if (relVelWorld.lengthSq() > Math::EPSILON) {
            Vec3 relVelLocal = invBoxTransform.transformDirection(relVelWorld);
            Vec3 absVel = relVelLocal.abs();

            if (absVel.x >= absVel.y && absVel.x >= absVel.z) {
                normalLocal = Vec3(relVelLocal.x >= 0.0f ? -1.0f : 1.0f, 0.0f, 0.0f);
                closest.x = normalLocal.x > 0.0f ? halfExtents.x : -halfExtents.x;
            } else if (absVel.y >= absVel.z) {
                normalLocal = Vec3(0.0f, relVelLocal.y >= 0.0f ? -1.0f : 1.0f, 0.0f);
                closest.y = normalLocal.y > 0.0f ? halfExtents.y : -halfExtents.y;
            } else {
                normalLocal = Vec3(0.0f, 0.0f, relVelLocal.z >= 0.0f ? -1.0f : 1.0f);
                closest.z = normalLocal.z > 0.0f ? halfExtents.z : -halfExtents.z;
            }
        } else {
            // Fallback: pick closest face based on penetration depth.
            Vec3 distances = halfExtents - sphereCenterLocal.abs();
            if (distances.x <= distances.y && distances.x <= distances.z) {
                normalLocal = Vec3(sphereCenterLocal.x >= 0 ? 1.0f : -1.0f, 0.0f, 0.0f);
                closest.x = sphereCenterLocal.x >= 0 ? halfExtents.x : -halfExtents.x;
            } else if (distances.y <= distances.z) {
                normalLocal = Vec3(0.0f, sphereCenterLocal.y >= 0 ? 1.0f : -1.0f, 0.0f);
                closest.y = sphereCenterLocal.y >= 0 ? halfExtents.y : -halfExtents.y;
            } else {
                normalLocal = Vec3(0.0f, 0.0f, sphereCenterLocal.z >= 0 ? 1.0f : -1.0f);
                closest.z = sphereCenterLocal.z >= 0 ? halfExtents.z : -halfExtents.z;
            }
        }
        distSq = 0.0f;
    }

    Vec3 normalWorld = boxTransform.transformDirection(normalLocal).normalized();
    Vec3 pointOnBoxWorld = boxTransform.transformPoint(closest);
    Vec3 pointOnSphereWorld = sphereCenterWorld - normalWorld * radius;

    ContactPoint cp;
    cp.normal = sphereIsA ? -normalWorld : normalWorld;
    cp.penetration = radius - std::sqrt(distSq);
    if (sphereIsA) {
        cp.positionWorldA = pointOnSphereWorld;
        cp.positionWorldB = pointOnBoxWorld;
    } else {
        cp.positionWorldA = pointOnBoxWorld;
        cp.positionWorldB = pointOnSphereWorld;
    }

    outManifold.addContact(cp);
    return true;
}

bool Narrowphase::boxBoxApprox(const RigidBody& a, const BoxShape& shapeA,
                               const RigidBody& b, const BoxShape& shapeB,
                               ContactManifold& outManifold) const {
    (void)shapeA;
    (void)shapeB;

    const AABB aabbA = a.getWorldAABB();
    const AABB aabbB = b.getWorldAABB();

    if (!aabbA.intersects(aabbB)) {
        return false;
    }

    Vec3 overlapMin = aabbA.min.max(aabbB.min);
    Vec3 overlapMax = aabbA.max.min(aabbB.max);
    Vec3 overlap = overlapMax - overlapMin;

    Vec3 normal = Vec3::unitX();
    f32 penetration = overlap.x;

    if (overlap.y < penetration) {
        penetration = overlap.y;
        normal = Vec3::unitY();
    }
    if (overlap.z < penetration) {
        penetration = overlap.z;
        normal = Vec3::unitZ();
    }

    Vec3 centerA = aabbA.center();
    Vec3 centerB = aabbB.center();
    if ((centerB - centerA).dot(normal) < 0.0f) {
        normal = -normal;
    }

    Vec3 contactCenter = (overlapMin + overlapMax) * 0.5f;

    Vec3 u, v;
    if (Math::abs(normal.x) > 0.5f) {
        u = Vec3::unitY();
        v = Vec3::unitZ();
    } else if (Math::abs(normal.y) > 0.5f) {
        u = Vec3::unitX();
        v = Vec3::unitZ();
    } else {
        u = Vec3::unitX();
        v = Vec3::unitY();
    }

    Vec3 halfSize = (overlapMax - overlapMin) * 0.5f;
    f32 uExtent = Math::abs(u.x * halfSize.x + u.y * halfSize.y + u.z * halfSize.z);
    f32 vExtent = Math::abs(v.x * halfSize.x + v.y * halfSize.y + v.z * halfSize.z);

    Vec3 corners[4] = {
        contactCenter + u * uExtent + v * vExtent,
        contactCenter + u * uExtent - v * vExtent,
        contactCenter - u * uExtent + v * vExtent,
        contactCenter - u * uExtent - v * vExtent
    };

    for (int i = 0; i < 4; ++i) {
        ContactPoint cp;
        cp.normal = normal;
        cp.penetration = penetration;
        cp.positionWorldA = corners[i] - normal * (penetration * 0.5f);
        cp.positionWorldB = corners[i] + normal * (penetration * 0.5f);
        outManifold.addContact(cp);
    }
    return true;
}

f32 Narrowphase::computeWorldRadius(const SphereShape& sphere, const Transform& transform) const {
    Vec3 scale = transform.scale.abs();
    f32 maxScale = Math::max(Math::max(scale.x, scale.y), scale.z);
    return sphere.getRadius() * maxScale;
}

void Narrowphase::getCapsuleSegmentWorld(const RigidBody& body, const CapsuleShape& capsule,
                                         Vec3& outP0, Vec3& outP1, f32& outRadius) const {
    const Transform& transform = body.getTransform();
    Vec3 scale = transform.scale.abs();
    f32 maxScale = Math::max(Math::max(scale.x, scale.y), scale.z);
    outRadius = capsule.getRadius() * maxScale;

    Vec3 localA(0.0f, -capsule.getHalfHeight(), 0.0f);
    Vec3 localB(0.0f, capsule.getHalfHeight(), 0.0f);
    localA *= transform.scale;
    localB *= transform.scale;

    outP0 = transform.transformPoint(localA);
    outP1 = transform.transformPoint(localB);
}

bool Narrowphase::sphereCapsule(const RigidBody& sphereBody, const SphereShape& sphere,
                                const RigidBody& capsuleBody, const CapsuleShape& capsule,
                                ContactManifold& outManifold, bool sphereIsA) const {
    Vec3 p0, p1;
    f32 capsuleRadius;
    getCapsuleSegmentWorld(capsuleBody, capsule, p0, p1, capsuleRadius);

    Vec3 sphereCenter = sphereBody.getPosition();
    f32 sphereRadius = computeWorldRadius(sphere, sphereBody.getTransform());

    Vec3 segDir = p1 - p0;
    f32 segLenSq = segDir.lengthSq();
    f32 t = 0.0f;
    if (segLenSq > Math::EPSILON) {
        t = (sphereCenter - p0).dot(segDir) / segLenSq;
        t = Math::clamp(t, 0.0f, 1.0f);
    }
    Vec3 closest = p0 + segDir * t;

    Vec3 delta = sphereCenter - closest;
    f32 distSq = delta.lengthSq();
    f32 radiusSum = sphereRadius + capsuleRadius;

    if (distSq > radiusSum * radiusSum) {
        return false;
    }

    f32 dist = std::sqrt(distSq);
    Vec3 normal = dist > Math::EPSILON ? delta / dist : Vec3::unitX();

    ContactPoint cp;
    cp.penetration = radiusSum - dist;
    if (sphereIsA) {
        cp.normal = normal;
        cp.positionWorldA = sphereCenter - normal * sphereRadius;
        cp.positionWorldB = closest + normal * capsuleRadius;
    } else {
        cp.normal = -normal;
        cp.positionWorldA = closest + normal * capsuleRadius;
        cp.positionWorldB = sphereCenter - normal * sphereRadius;
    }

    outManifold.addContact(cp);
    return true;
}

bool Narrowphase::capsuleCapsule(const RigidBody& a, const CapsuleShape& shapeA,
                                 const RigidBody& b, const CapsuleShape& shapeB,
                                 ContactManifold& outManifold) const {
    Vec3 a0, a1, b0, b1;
    f32 radiusA, radiusB;
    getCapsuleSegmentWorld(a, shapeA, a0, a1, radiusA);
    getCapsuleSegmentWorld(b, shapeB, b0, b1, radiusB);

    Vec3 c1, c2;
    closestPointsSegmentSegment(a0, a1, b0, b1, c1, c2);

    Vec3 delta = c2 - c1;
    f32 distSq = delta.lengthSq();
    f32 radiusSum = radiusA + radiusB;

    if (distSq > radiusSum * radiusSum) {
        return false;
    }

    f32 dist = std::sqrt(distSq);
    Vec3 normal = dist > Math::EPSILON ? delta / dist : Vec3::unitX();

    ContactPoint cp;
    cp.normal = normal;
    cp.penetration = radiusSum - dist;
    cp.positionWorldA = c1 + normal * radiusA;
    cp.positionWorldB = c2 - normal * radiusB;

    outManifold.addContact(cp);
    return true;
}

bool Narrowphase::capsuleBoxApprox(const RigidBody& capsuleBody, const CapsuleShape& capsule,
                                   const RigidBody& boxBody, const BoxShape& box,
                                   ContactManifold& outManifold, bool capsuleIsA) const {
    (void)box;
    Vec3 p0, p1;
    f32 capsuleRadius;
    getCapsuleSegmentWorld(capsuleBody, capsule, p0, p1, capsuleRadius);

    AABB boxAabb = boxBody.getWorldAABB();

    bool hit = false;
    Vec3 endpoints[2] = { p0, p1 };
    for (int i = 0; i < 2; ++i) {
        Vec3 closest;
        f32 distSq = distanceSqPointAABB(endpoints[i], boxAabb, closest);
        if (distSq <= capsuleRadius * capsuleRadius) {
            f32 dist = std::sqrt(distSq);
            Vec3 normal = dist > Math::EPSILON ? (endpoints[i] - closest) / dist : Vec3::unitX();

            ContactPoint cp;
            cp.penetration = capsuleRadius - dist;
            if (capsuleIsA) {
                cp.normal = normal;
                cp.positionWorldA = endpoints[i] - normal * capsuleRadius;
                cp.positionWorldB = closest;
            } else {
                cp.normal = -normal;
                cp.positionWorldA = closest;
                cp.positionWorldB = endpoints[i] - normal * capsuleRadius;
            }

            outManifold.addContact(cp);
            hit = true;
        }
    }

    if (hit) {
        return true;
    }

    Vec3 segPoint, boxPoint;
    f32 distSq = distanceSqSegmentAABB(p0, p1, boxAabb, segPoint, boxPoint);
    if (distSq > capsuleRadius * capsuleRadius) {
        return false;
    }

    f32 dist = std::sqrt(distSq);
    Vec3 normal = dist > Math::EPSILON ? (segPoint - boxPoint) / dist : Vec3::unitX();

    ContactPoint cp;
    cp.penetration = capsuleRadius - dist;
    if (capsuleIsA) {
        cp.normal = normal;
        cp.positionWorldA = segPoint - normal * capsuleRadius;
        cp.positionWorldB = boxPoint;
    } else {
        cp.normal = -normal;
        cp.positionWorldA = boxPoint;
        cp.positionWorldB = segPoint - normal * capsuleRadius;
    }

    outManifold.addContact(cp);
    return true;
}

f32 Narrowphase::distanceSqPointAABB(const Vec3& point, const AABB& aabb, Vec3& outClosest) {
    outClosest = point.clamp(aabb.min, aabb.max);
    return (point - outClosest).lengthSq();
}

f32 Narrowphase::distanceSqSegmentAABB(const Vec3& p0, const Vec3& p1, const AABB& aabb,
                                       Vec3& outSegPoint, Vec3& outAabbPoint) {
    // Ternary search along segment (convex distance function)
    f32 t0 = 0.0f;
    f32 t1 = 1.0f;
    Vec3 d = p1 - p0;

    for (int i = 0; i < 12; ++i) {
        f32 tA = (2.0f * t0 + t1) / 3.0f;
        f32 tB = (t0 + 2.0f * t1) / 3.0f;

        Vec3 pA = p0 + d * tA;
        Vec3 pB = p0 + d * tB;

        Vec3 cA, cB;
        f32 distA = distanceSqPointAABB(pA, aabb, cA);
        f32 distB = distanceSqPointAABB(pB, aabb, cB);

        if (distA < distB) {
            t1 = tB;
        } else {
            t0 = tA;
        }
    }

    f32 t = (t0 + t1) * 0.5f;
    outSegPoint = p0 + d * t;
    return distanceSqPointAABB(outSegPoint, aabb, outAabbPoint);
}

void Narrowphase::closestPointsSegmentSegment(const Vec3& p1, const Vec3& q1,
                                              const Vec3& p2, const Vec3& q2,
                                              Vec3& c1, Vec3& c2) {
    Vec3 d1 = q1 - p1;
    Vec3 d2 = q2 - p2;
    Vec3 r = p1 - p2;
    f32 a = d1.dot(d1);
    f32 e = d2.dot(d2);
    f32 f = d2.dot(r);

    f32 s = 0.0f;
    f32 t = 0.0f;

    if (a <= Math::EPSILON && e <= Math::EPSILON) {
        c1 = p1;
        c2 = p2;
        return;
    }

    if (a <= Math::EPSILON) {
        s = 0.0f;
        t = Math::clamp(f / e, 0.0f, 1.0f);
    } else {
        f32 c = d1.dot(r);
        if (e <= Math::EPSILON) {
            t = 0.0f;
            s = Math::clamp(-c / a, 0.0f, 1.0f);
        } else {
            f32 b = d1.dot(d2);
            f32 denom = a * e - b * b;
            if (denom != 0.0f) {
                s = Math::clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            } else {
                s = 0.0f;
            }
            t = (b * s + f) / e;

            if (t < 0.0f) {
                t = 0.0f;
                s = Math::clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = Math::clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }

    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
}

} // namespace WulfNet
