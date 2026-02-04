// =============================================================================
// WulfNet Engine - GJK/EPA Collision Detection
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/math/Vec3.h"
#include "CollisionShape.h"
#include "physics/dynamics/RigidBody.h"
#include <vector>

namespace WulfNet {

struct SupportPoint {
    Vec3 p; // Minkowski difference (a - b)
    Vec3 a; // Point on A
    Vec3 b; // Point on B
    
    SupportPoint() = default;
    SupportPoint(const Vec3& diff, const Vec3& pA, const Vec3& pB) 
        : p(diff), a(pA), b(pB) {}
};

struct Simplex {
    SupportPoint points[4];
    int size = 0;

    void push_front(SupportPoint point) {
        for(int i = 2; i >= 0; i--) {
            points[i+1] = points[i];
        }
        points[0] = point;
        size = std::min(size + 1, 4);
    }
    
    SupportPoint& operator[](int i) { return points[i]; }
    const SupportPoint& operator[](int i) const { return points[i]; }
};

class GJK {
public:
    // Returns true if shapes overlap
    static bool intersect(const RigidBody* bodyA, const CollisionShape* shapeA,
                          const RigidBody* bodyB, const CollisionShape* shapeB,
                          Simplex* outSimplex = nullptr);

    // Returns true if overlap, fills contact info (EPA)
    static bool computePenetration(const RigidBody* bodyA, const CollisionShape* shapeA,
                                   const RigidBody* bodyB, const CollisionShape* shapeB,
                                   Vec3& outNormal, f32& outDepth, Vec3& outContactPointA, Vec3& outContactPointB);

private:
    static SupportPoint support(const RigidBody* bodyA, const CollisionShape* shapeA,
                        const RigidBody* bodyB, const CollisionShape* shapeB,
                        Vec3 dir);
                        
    static bool handleSimplex(Simplex& simplex, Vec3& direction);
    static bool lineCase(Simplex& simplex, Vec3& direction);
    static bool triangleCase(Simplex& simplex, Vec3& direction);
    static bool tetrahedronCase(Simplex& simplex, Vec3& direction);
};

}
