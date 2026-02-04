// =============================================================================
// WulfNet Engine - Physics Collision Tests
// =============================================================================

#include <cassert>
#include <cmath>
#include <catch2/catch_all.hpp>
#include "physics/collision/AABB.h"
#include "physics/collision/BoundingSphere.h"
#include "physics/collision/CollisionShape.h"
#include "physics/collision/Broadphase.h"
#include "physics/collision/Narrowphase.h"
#include "physics/dynamics/RigidBody.h"
#include "physics/dynamics/RigidBodyWorld.h"
#include "physics/softbody/Cloth.h"
#include "core/Types.h"
#include "core/math/Vec3.h"

using namespace WulfNet;

// =============================================================================
// AABB Tests
// =============================================================================

TEST_CASE("AABB construction and properties", "[physics][aabb]") {
    SECTION("Default constructor creates invalid AABB") {
        AABB aabb;
        REQUIRE_FALSE(aabb.isValid());
    }
    
    SECTION("Min/max constructor") {
        AABB aabb(Vec3(-1, -2, -3), Vec3(1, 2, 3));
        REQUIRE(aabb.isValid());
        REQUIRE(aabb.min.x == Catch::Approx(-1.0f));
        REQUIRE(aabb.max.z == Catch::Approx(3.0f));
    }
    
    SECTION("From center and extents") {
        AABB aabb = AABB::fromCenterExtents(Vec3(0, 0, 0), Vec3(1, 1, 1));
        REQUIRE(aabb.min.x == Catch::Approx(-1.0f));
        REQUIRE(aabb.max.x == Catch::Approx(1.0f));
        REQUIRE(aabb.center().x == Catch::Approx(0.0f));
    }
    
    SECTION("From sphere") {
        AABB aabb = AABB::fromSphere(Vec3(0, 0, 0), 2.0f);
        REQUIRE(aabb.min.x == Catch::Approx(-2.0f));
        REQUIRE(aabb.max.y == Catch::Approx(2.0f));
    }
    
    SECTION("Volume calculation") {
        AABB aabb(Vec3(0, 0, 0), Vec3(2, 3, 4));
        REQUIRE(aabb.volume() == Catch::Approx(24.0f));
    }
}

TEST_CASE("AABB intersection tests", "[physics][aabb]") {
    SECTION("Overlapping AABBs") {
        AABB a(Vec3(0, 0, 0), Vec3(2, 2, 2));
        AABB b(Vec3(1, 1, 1), Vec3(3, 3, 3));
        REQUIRE(a.intersects(b));
        // AABB::intersects is symmetric
        AABB b2(Vec3(1, 1, 1), Vec3(3, 3, 3));
        REQUIRE(b2.intersects(a));
    }
    
    SECTION("Non-overlapping AABBs") {
        AABB a(Vec3(0, 0, 0), Vec3(1, 1, 1));
        AABB b(Vec3(2, 2, 2), Vec3(3, 3, 3));
        REQUIRE_FALSE(a.intersects(b));
    }
    
    SECTION("Touching AABBs (edge contact)") {
        AABB a(Vec3(0, 0, 0), Vec3(1, 1, 1));
        AABB b(Vec3(1, 0, 0), Vec3(2, 1, 1));
        REQUIRE(a.intersects(b));  // Touching counts as intersecting
    }
    
    SECTION("Point containment") {
        AABB aabb(Vec3(-1, -1, -1), Vec3(1, 1, 1));
        REQUIRE(aabb.contains(Vec3(0, 0, 0)));
        REQUIRE(aabb.contains(Vec3(1, 1, 1)));  // On boundary
        REQUIRE_FALSE(aabb.contains(Vec3(2, 0, 0)));
    }
}

TEST_CASE("AABB ray intersection", "[physics][aabb]") {
    AABB aabb(Vec3(-1, -1, -1), Vec3(1, 1, 1));
    
    SECTION("Ray hitting AABB") {
        Vec3 origin(5, 0, 0);
        Vec3 direction(-1, 0, 0);
        Vec3 dirInv(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z); // .z because div by zero handle not here
        // Safe dirInv manually for test
        if(direction.x == 0) dirInv.x = 1e30f; else dirInv.x = 1.0f/direction.x;
        if(direction.y == 0) dirInv.y = 1e30f; else dirInv.y = 1.0f/direction.y;
        if(direction.z == 0) dirInv.z = 1e30f; else dirInv.z = 1.0f/direction.z;

        f32 tMin, tMax;
        REQUIRE(aabb.rayIntersects(origin, dirInv, tMin, tMax));
        REQUIRE(tMin == Catch::Approx(4.0f));  // Hits at x=1
        REQUIRE(tMax == Catch::Approx(6.0f));  // Exits at x=-1
    }
    
    SECTION("Ray missing AABB") {
        Vec3 origin(5, 5, 0);
        Vec3 direction(-1, 0, 0);
        Vec3 dirInv;
        if(direction.x == 0) dirInv.x = 1e30f; else dirInv.x = 1.0f/direction.x;
        if(direction.y == 0) dirInv.y = 1e30f; else dirInv.y = 1.0f/direction.y;
        if(direction.z == 0) dirInv.z = 1e30f; else dirInv.z = 1.0f/direction.z;
        
        f32 tMin, tMax;
        REQUIRE_FALSE(aabb.rayIntersects(origin, dirInv, tMin, tMax));
    }
}

// =============================================================================
// BoundingSphere Tests
// =============================================================================

TEST_CASE("BoundingSphere construction", "[physics][sphere]") {
    SECTION("Default constructor") {
        BoundingSphere sphere;
        REQUIRE(sphere.center.x == Catch::Approx(0.0f));
        REQUIRE(sphere.radius == Catch::Approx(0.0f));
    }
    
    SECTION("Parameter constructor") {
        BoundingSphere sphere(Vec3(1, 2, 3), 5.0f);
        REQUIRE(sphere.center.y == Catch::Approx(2.0f));
        REQUIRE(sphere.radius == Catch::Approx(5.0f));
    }
    
    SECTION("From AABB") {
        AABB aabb(Vec3(-1, -1, -1), Vec3(1, 1, 1));
        BoundingSphere sphere = BoundingSphere::fromAABB(aabb);
        REQUIRE(sphere.center.x == Catch::Approx(0.0f));
        // Radius should be sqrt(3) for unit cube
        REQUIRE(sphere.radius == Catch::Approx(std::sqrt(3.0f)));
    }
}

TEST_CASE("BoundingSphere intersection tests", "[physics][sphere]") {
    SECTION("Overlapping spheres") {
        BoundingSphere a(Vec3(0, 0, 0), 2.0f);
        BoundingSphere b(Vec3(3, 0, 0), 2.0f);
        REQUIRE(a.intersects(b));
    }
    
    SECTION("Non-overlapping spheres") {
        BoundingSphere a(Vec3(0, 0, 0), 1.0f);
        BoundingSphere b(Vec3(5, 0, 0), 1.0f);
        REQUIRE_FALSE(a.intersects(b));
    }
    
    SECTION("Touching spheres") {
        BoundingSphere a(Vec3(0, 0, 0), 1.0f);
        BoundingSphere b(Vec3(2, 0, 0), 1.0f);
        REQUIRE(a.intersects(b));
    }
    
    SECTION("Sphere-AABB intersection") {
        BoundingSphere sphere(Vec3(0, 0, 0), 2.0f);
        AABB aabb(Vec3(1, 0, 0), Vec3(3, 1, 1));
        REQUIRE(sphere.intersects(aabb));
    }
}

// =============================================================================
// CollisionShape Tests
// =============================================================================

TEST_CASE("SphereShape", "[physics][shape]") {
    SphereShape sphere(2.0f);
    
    SECTION("Properties") {
        REQUIRE(sphere.getType() == ShapeType::Sphere);
        REQUIRE(sphere.getRadius() == Catch::Approx(2.0f));
        // Volume = 4/3 * pi * r^3
        REQUIRE(sphere.getVolume() == Catch::Approx(33.51032f).margin(0.01f));
    }
    
    SECTION("Local AABB") {
        AABB aabb = sphere.getLocalAABB();
        REQUIRE(aabb.min.x == Catch::Approx(-2.0f));
        REQUIRE(aabb.max.x == Catch::Approx(2.0f));
    }
    
    SECTION("Support function") {
        Vec3 support = sphere.support(Vec3(1, 0, 0));
        REQUIRE(support.x == Catch::Approx(2.0f));
        REQUIRE(support.y == Catch::Approx(0.0f));
    }
    
    SECTION("Raycast hit") {
        f32 hitDist;
        Vec3 hitNormal;
        // Raycast(origin, dir, hitDist, hitNormal)
        REQUIRE(sphere.raycast(Vec3(5, 0, 0), Vec3(-1, 0, 0), hitDist, hitNormal));
        REQUIRE(hitDist == Catch::Approx(3.0f));  // 5 - 2 (radius)
        REQUIRE(hitNormal.x == Catch::Approx(1.0f));
    }
    
    SECTION("Raycast miss") {
        f32 hitDist;
        Vec3 hitNormal;
        REQUIRE_FALSE(sphere.raycast(Vec3(5, 5, 0), Vec3(-1, 0, 0), hitDist, hitNormal));
    }
}

TEST_CASE("BoxShape", "[physics][shape]") {
    BoxShape box(Vec3(1, 2, 3));
    
    SECTION("Properties") {
        REQUIRE(box.getType() == ShapeType::Box);
        REQUIRE(box.getHalfExtents().x == Catch::Approx(1.0f));
        // Volume = 2 * 4 * 6 = 48
        REQUIRE(box.getVolume() == Catch::Approx(48.0f));
    }
    
    SECTION("Support function") {
        Vec3 support = box.support(Vec3(1, 1, 1));
        REQUIRE(support.x == Catch::Approx(1.0f));
        REQUIRE(support.y == Catch::Approx(2.0f));
        REQUIRE(support.z == Catch::Approx(3.0f));
        
        Vec3 supportNeg = box.support(Vec3(-1, -1, -1));
        REQUIRE(supportNeg.x == Catch::Approx(-1.0f));
    }
    
    SECTION("Raycast") {
        f32 hitDist;
        Vec3 hitNormal;
        REQUIRE(box.raycast(Vec3(5, 0, 0), Vec3(-1, 0, 0), hitDist, hitNormal));
        REQUIRE(hitDist == Catch::Approx(4.0f));  // 5 - 1 (half extent)
    }
}


// =============================================================================
// Broadphase Tests
// =============================================================================

TEST_CASE("SpatialHashBroadphase basic operations", "[physics][broadphase]") {
    SpatialHashBroadphase broadphase(4.0f);
    
    SECTION("Add and remove entries") {
        u32 id1 = broadphase.createProxy(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)));
        u32 id2 = broadphase.createProxy(AABB(Vec3(2, 0, 0), Vec3(3, 1, 1)));
        
        REQUIRE(broadphase.getProxyCount() == 2);
        
        broadphase.destroyProxy(id1);
        REQUIRE(broadphase.getProxyCount() == 1);
        
        broadphase.clear();
        REQUIRE(broadphase.getProxyCount() == 0);
    }
}

TEST_CASE("SpatialHashBroadphase pair finding", "[physics][broadphase]") {
    SpatialHashBroadphase broadphase(4.0f);
    
    SECTION("Overlapping objects generate pairs") {
        u32 id1 = broadphase.createProxy(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)));
        u32 id2 = broadphase.createProxy(AABB(Vec3(1, 1, 1), Vec3(3, 3, 3)));
        
        std::vector<CollisionPair> pairs;
        broadphase.findOverlappingPairs(pairs);
        
        REQUIRE(pairs.size() == 1);
        REQUIRE(((pairs[0].bodyIdA == id1 && pairs[0].bodyIdB == id2) ||
                 (pairs[0].bodyIdA == id2 && pairs[0].bodyIdB == id1)));
    }
    
    SECTION("Non-overlapping objects don't generate pairs") {
        broadphase.createProxy(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)));
        broadphase.createProxy(AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)));
        
        std::vector<CollisionPair> pairs;
        broadphase.findOverlappingPairs(pairs);
        
        REQUIRE(pairs.empty());
    }
}

TEST_CASE("GpuSpatialHashBroadphase pair finding", "[physics][broadphase]") {
    GpuSpatialHashBroadphase broadphase(4.0f);

    SECTION("Overlapping objects generate pairs") {
        u32 id1 = broadphase.createProxy(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)));
        u32 id2 = broadphase.createProxy(AABB(Vec3(1, 1, 1), Vec3(3, 3, 3)));

        std::vector<CollisionPair> pairs;
        broadphase.findOverlappingPairs(pairs);

        REQUIRE(pairs.size() == 1);
        REQUIRE(((pairs[0].bodyIdA == id1 && pairs[0].bodyIdB == id2) ||
                 (pairs[0].bodyIdA == id2 && pairs[0].bodyIdB == id1)));
    }

    SECTION("Non-overlapping objects don't generate pairs") {
        broadphase.createProxy(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)));
        broadphase.createProxy(AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)));

        std::vector<CollisionPair> pairs;
        broadphase.findOverlappingPairs(pairs);

        REQUIRE(pairs.empty());
    }
}

// =============================================================================
// Narrowphase Tests
// =============================================================================

TEST_CASE("Narrowphase sphere-sphere contact", "[physics][narrowphase]") {
    SphereShape sphereA(1.0f);
    SphereShape sphereB(1.0f);

    RigidBodyConfig configA;
    configA.shape = &sphereA;
    configA.transform.position = Vec3(0, 0, 0);

    RigidBodyConfig configB;
    configB.shape = &sphereB;
    configB.transform.position = Vec3(1.5f, 0, 0);

    RigidBody bodyA(configA);
    RigidBody bodyB(configB);

    Narrowphase narrowphase;
    ContactManifold manifold;

    REQUIRE(narrowphase.generateContacts(bodyA, bodyB, manifold));
    REQUIRE(manifold.contactCount == 1);
    REQUIRE(manifold.contacts[0].penetration > 0.0f);
    REQUIRE(manifold.contacts[0].normal.x == Catch::Approx(1.0f));
}

TEST_CASE("Narrowphase sphere-box contact", "[physics][narrowphase]") {
    SphereShape sphere(1.0f);
    BoxShape box(Vec3(1.0f, 1.0f, 1.0f));

    RigidBodyConfig configSphere;
    configSphere.shape = &sphere;
    configSphere.transform.position = Vec3(1.5f, 0, 0);

    RigidBodyConfig configBox;
    configBox.shape = &box;
    configBox.transform.position = Vec3(0, 0, 0);

    RigidBody sphereBody(configSphere);
    RigidBody boxBody(configBox);

    Narrowphase narrowphase;
    ContactManifold manifold;

    REQUIRE(narrowphase.generateContacts(sphereBody, boxBody, manifold));
    REQUIRE(manifold.contactCount == 1);
    REQUIRE(manifold.contacts[0].penetration > 0.0f);
}

TEST_CASE("Narrowphase box-box contact (AABB approximation)", "[physics][narrowphase]") {
    BoxShape boxA(Vec3(1.0f, 1.0f, 1.0f));
    BoxShape boxB(Vec3(1.0f, 1.0f, 1.0f));

    RigidBodyConfig configA;
    configA.shape = &boxA;
    configA.transform.position = Vec3(0, 0, 0);

    RigidBodyConfig configB;
    configB.shape = &boxB;
    configB.transform.position = Vec3(1.5f, 0, 0);

    RigidBody bodyA(configA);
    RigidBody bodyB(configB);

    Narrowphase narrowphase;
    ContactManifold manifold;

    REQUIRE(narrowphase.generateContacts(bodyA, bodyB, manifold));
    REQUIRE(manifold.contactCount == 4);
    REQUIRE(manifold.contacts[0].penetration > 0.0f);
}

TEST_CASE("Narrowphase sphere-capsule contact", "[physics][narrowphase]") {
    SphereShape sphere(1.0f);
    CapsuleShape capsule(0.5f, 1.0f);

    RigidBodyConfig configSphere;
    configSphere.shape = &sphere;
    configSphere.transform.position = Vec3(1.4f, 0, 0);

    RigidBodyConfig configCapsule;
    configCapsule.shape = &capsule;
    configCapsule.transform.position = Vec3(0, 0, 0);

    RigidBody sphereBody(configSphere);
    RigidBody capsuleBody(configCapsule);

    Narrowphase narrowphase;
    ContactManifold manifold;

    REQUIRE(narrowphase.generateContacts(sphereBody, capsuleBody, manifold));
    REQUIRE(manifold.contactCount == 1);
    REQUIRE(manifold.contacts[0].penetration > 0.0f);
}

TEST_CASE("Narrowphase capsule-capsule contact", "[physics][narrowphase]") {
    CapsuleShape capsuleA(0.5f, 1.0f);
    CapsuleShape capsuleB(0.5f, 1.0f);

    RigidBodyConfig configA;
    configA.shape = &capsuleA;
    configA.transform.position = Vec3(0, 0, 0);

    RigidBodyConfig configB;
    configB.shape = &capsuleB;
    configB.transform.position = Vec3(0.8f, 0, 0);

    RigidBody bodyA(configA);
    RigidBody bodyB(configB);

    Narrowphase narrowphase;
    ContactManifold manifold;

    REQUIRE(narrowphase.generateContacts(bodyA, bodyB, manifold));
    REQUIRE(manifold.contactCount == 1);
    REQUIRE(manifold.contacts[0].penetration > 0.0f);
}

TEST_CASE("RigidBodyWorld broadphase pair output", "[physics][world]") {
    RigidBodyWorldConfig worldConfig;
    worldConfig.gravity = Vec3(0, 0, 0);
    RigidBodyWorld world(worldConfig);

    BoxShape boxA(Vec3(1.0f, 1.0f, 1.0f));
    BoxShape boxB(Vec3(1.0f, 1.0f, 1.0f));

    RigidBodyConfig configA;
    configA.shape = &boxA;
    configA.transform.position = Vec3(0, 0, 0);

    RigidBodyConfig configB;
    configB.shape = &boxB;
    configB.transform.position = Vec3(1.5f, 0, 0); // overlap

    world.createBody(configA);
    world.createBody(configB);

    world.step(1.0f / 60.0f);

    const auto& pairs = world.getBroadphasePairs();
    REQUIRE(pairs.size() == 1);
    REQUIRE(pairs[0].bodyIdA != 0);
    REQUIRE(pairs[0].bodyIdB != 0);
}

TEST_CASE("RigidBodyWorld gravity integration", "[physics][world]") {
    RigidBodyWorldConfig worldConfig;
    worldConfig.gravity = Vec3(0, -9.81f, 0);
    RigidBodyWorld world(worldConfig);

    SphereShape sphere(0.5f);
    RigidBodyConfig config;
    config.shape = &sphere;
    config.transform.position = Vec3(0, 10.0f, 0);
    config.linearDamping = 0.0f;

    auto handle = world.createBody(config);

    for (int i = 0; i < 60; ++i) {
        world.step(1.0f / 60.0f);
    }

    const RigidBody* body = world.getBody(handle);
    REQUIRE(body != nullptr);
    REQUIRE(body->getPosition().y < 10.0f);
    REQUIRE(body->getPosition().y > 4.0f);
    REQUIRE(body->getLinearVelocity().y == Catch::Approx(-9.81f).margin(0.15f));
}

TEST_CASE("RigidBodyWorld contact manifold impulses", "[physics][solver]") {
    RigidBodyWorldConfig worldConfig;
    worldConfig.gravity = Vec3(0, -9.81f, 0);
    RigidBodyWorld world(worldConfig);

    BoxShape groundShape(Vec3(10.0f, 0.5f, 10.0f));
    SphereShape sphereShape(0.5f);

    RigidBodyConfig groundConfig;
    groundConfig.shape = &groundShape;
    groundConfig.transform.position = Vec3(0, -0.5f, 0);
    groundConfig.type = MotionType::Static;

    RigidBodyConfig sphereConfig;
    sphereConfig.shape = &sphereShape;
    sphereConfig.transform.position = Vec3(0, 1.0f, 0);
    sphereConfig.restitution = 0.0f;

    world.createBody(groundConfig);
    auto sphereHandle = world.createBody(sphereConfig);

    for (int i = 0; i < 120; ++i) {
        world.step(1.0f / 60.0f);
    }

    const auto& contacts = world.getContactManifolds();
    bool foundImpulse = false;
    for (const auto& manifold : contacts) {
        if (manifold.contactCount == 0) continue;
        if (manifold.bodyIdA == sphereHandle.value || manifold.bodyIdB == sphereHandle.value) {
            if (manifold.contacts[0].normalImpulse > 0.0f) {
                foundImpulse = true;
                break;
            }
        }
    }

    REQUIRE(foundImpulse);
}

TEST_CASE("RigidBodyWorld reduces overlap", "[physics][solver]") {
    RigidBodyWorldConfig worldConfig;
    worldConfig.gravity = Vec3(0, 0, 0);
    RigidBodyWorld world(worldConfig);

    BoxShape boxA(Vec3(1.0f, 1.0f, 1.0f));
    BoxShape boxB(Vec3(1.0f, 1.0f, 1.0f));

    RigidBodyConfig configA;
    configA.shape = &boxA;
    configA.transform.position = Vec3(0, 0, 0);

    RigidBodyConfig configB;
    configB.shape = &boxB;
    configB.transform.position = Vec3(1.5f, 0, 0); // 0.5 overlap

    auto handleA = world.createBody(configA);
    auto handleB = world.createBody(configB);

    const AABB aabbBeforeA = world.getBody(handleA)->getWorldAABB();
    const AABB aabbBeforeB = world.getBody(handleB)->getWorldAABB();
    f32 overlapBefore = aabbBeforeA.max.x - aabbBeforeB.min.x;

    for (int i = 0; i < 10; ++i) {
        world.step(1.0f / 60.0f);
    }

    const AABB aabbAfterA = world.getBody(handleA)->getWorldAABB();
    const AABB aabbAfterB = world.getBody(handleB)->getWorldAABB();
    f32 overlapAfter = aabbAfterA.max.x - aabbAfterB.min.x;

    REQUIRE(overlapAfter < overlapBefore);
}

TEST_CASE("RigidBodyWorld stacking stability", "[physics][solver]") {
    RigidBodyWorldConfig worldConfig;
    worldConfig.gravity = Vec3(0, -9.81f, 0);
    RigidBodyWorld world(worldConfig);

    BoxShape groundShape(Vec3(10.0f, 1.0f, 10.0f));
    BoxShape boxShape(Vec3(1.0f, 1.0f, 1.0f));

    RigidBodyConfig groundConfig;
    groundConfig.shape = &groundShape;
    groundConfig.transform.position = Vec3(0, -1.0f, 0);
    groundConfig.type = MotionType::Static;

    RigidBodyConfig boxAConfig;
    boxAConfig.shape = &boxShape;
    boxAConfig.transform.position = Vec3(0, 1.2f, 0);

    RigidBodyConfig boxBConfig;
    boxBConfig.shape = &boxShape;
    boxBConfig.transform.position = Vec3(0, 3.3f, 0);

    world.createBody(groundConfig);
    auto boxA = world.createBody(boxAConfig);
    auto boxB = world.createBody(boxBConfig);

    for (int i = 0; i < 240; ++i) {
        world.step(1.0f / 60.0f);
    }

    const AABB aabbA = world.getBody(boxA)->getWorldAABB();
    const AABB aabbB = world.getBody(boxB)->getWorldAABB();

    REQUIRE(aabbA.min.y >= -0.05f);
    REQUIRE(aabbB.min.y >= aabbA.max.y - 0.1f);
}

TEST_CASE("RigidBodyWorld restitution bounce", "[physics][solver]") {
    RigidBodyWorldConfig worldConfig;
    worldConfig.gravity = Vec3(0, -9.81f, 0);
    RigidBodyWorld world(worldConfig);

    BoxShape groundShape(Vec3(10.0f, 0.5f, 10.0f));
    SphereShape sphereShape(0.5f);

    RigidBodyConfig groundConfig;
    groundConfig.shape = &groundShape;
    groundConfig.transform.position = Vec3(0, -0.5f, 0);
    groundConfig.type = MotionType::Static;

    RigidBodyConfig sphereConfig;
    sphereConfig.shape = &sphereShape;
    sphereConfig.transform.position = Vec3(0, 5.0f, 0);
    sphereConfig.restitution = 0.8f;

    world.createBody(groundConfig);
    auto sphereHandle = world.createBody(sphereConfig);

    bool bounced = false;
    f32 maxAfterBounce = -1.0f;
    f32 previousVy = 0.0f;

    for (int i = 0; i < 480; ++i) {
        world.step(1.0f / 60.0f);
        RigidBody* sphere = world.getBody(sphereHandle);
        f32 vy = sphere->getLinearVelocity().y;

        if (!bounced && previousVy < 0.0f && vy > 0.0f) {
            bounced = true;
            maxAfterBounce = sphere->getPosition().y;
        }

        if (bounced) {
            maxAfterBounce = Math::max(maxAfterBounce, sphere->getPosition().y);
            if (previousVy > 0.0f && vy <= 0.0f) {
                break; // reached peak
            }
        }

        previousVy = vy;
    }

    REQUIRE(bounced);
    REQUIRE(maxAfterBounce > 0.45f);
    REQUIRE(maxAfterBounce < 5.0f);
}

TEST_CASE("RigidBodyWorld distance joint", "[physics][solver]") {
    RigidBodyWorldConfig worldConfig;
    worldConfig.gravity = Vec3(0, 0, 0);
    RigidBodyWorld world(worldConfig);

    SphereShape sphereShape(0.5f);

    RigidBodyConfig configA;
    configA.shape = &sphereShape;
    configA.transform.position = Vec3(0, 0, 0);

    RigidBodyConfig configB;
    configB.shape = &sphereShape;
    configB.transform.position = Vec3(4.0f, 0, 0);

    auto handleA = world.createBody(configA);
    auto handleB = world.createBody(configB);

    DistanceJointConfig jointConfig;
    jointConfig.bodyIdA = handleA.value;
    jointConfig.bodyIdB = handleB.value;
    jointConfig.restLength = 2.0f;
    jointConfig.stiffness = 0.8f;
    jointConfig.damping = 0.2f;

    world.createDistanceJoint(jointConfig);

    f32 distBefore = (world.getBody(handleB)->getPosition() - world.getBody(handleA)->getPosition()).length();

    for (int i = 0; i < 120; ++i) {
        world.step(1.0f / 60.0f);
    }

    f32 distAfter = (world.getBody(handleB)->getPosition() - world.getBody(handleA)->getPosition()).length();

    REQUIRE(distAfter < distBefore);
    REQUIRE(distAfter < 3.0f);
}

TEST_CASE("RigidBodyWorld ball joint", "[physics][solver]") {
    RigidBodyWorldConfig worldConfig;
    worldConfig.gravity = Vec3(0, 0, 0);
    RigidBodyWorld world(worldConfig);

    SphereShape sphereShape(0.5f);

    RigidBodyConfig configA;
    configA.shape = &sphereShape;
    configA.transform.position = Vec3(0, 0, 0);
    configA.type = MotionType::Static;

    RigidBodyConfig configB;
    configB.shape = &sphereShape;
    configB.transform.position = Vec3(3.0f, 0, 0);

    auto handleA = world.createBody(configA);
    auto handleB = world.createBody(configB);

    BallJointConfig jointConfig;
    jointConfig.bodyIdA = handleA.value;
    jointConfig.bodyIdB = handleB.value;
    jointConfig.stiffness = 0.9f;
    jointConfig.damping = 0.2f;

    world.createBallJoint(jointConfig);

    f32 distBefore = (world.getBody(handleB)->getPosition() - world.getBody(handleA)->getPosition()).length();

    for (int i = 0; i < 120; ++i) {
        world.step(1.0f / 60.0f);
    }

    f32 distAfter = (world.getBody(handleB)->getPosition() - world.getBody(handleA)->getPosition()).length();

    REQUIRE(distAfter < distBefore);
    REQUIRE(distAfter < 2.0f);
}

TEST_CASE("RigidBodyWorld fixed joint", "[physics][solver]") {
    RigidBodyWorldConfig worldConfig;
    worldConfig.gravity = Vec3(0, 0, 0);
    RigidBodyWorld world(worldConfig);

    BoxShape boxShape(Vec3(0.5f, 0.5f, 0.5f));

    RigidBodyConfig configA;
    configA.shape = &boxShape;
    configA.transform.position = Vec3(0, 0, 0);
    configA.type = MotionType::Static;

    RigidBodyConfig configB;
    configB.shape = &boxShape;
    configB.transform.position = Vec3(2.0f, 0, 0);

    auto handleA = world.createBody(configA);
    auto handleB = world.createBody(configB);

    FixedJointConfig jointConfig;
    jointConfig.bodyIdA = handleA.value;
    jointConfig.bodyIdB = handleB.value;
    jointConfig.stiffness = 0.9f;
    jointConfig.damping = 0.2f;

    world.createFixedJoint(jointConfig);

    RigidBody* bodyB = world.getBody(handleB);
    REQUIRE(bodyB != nullptr);
    bodyB->setPosition(Vec3(4.0f, 0, 0));
    bodyB->setOrientation(Quat::fromAxisAngle(Vec3::unitY(), Math::radians(45.0f)));

    f32 distBefore = (bodyB->getPosition() - world.getBody(handleA)->getPosition()).length();
    f32 angleBefore = world.getBody(handleA)->getOrientation().angleTo(bodyB->getOrientation());

    for (int i = 0; i < 120; ++i) {
        world.step(1.0f / 60.0f);
    }

    f32 distAfter = (bodyB->getPosition() - world.getBody(handleA)->getPosition()).length();
    f32 angleAfter = world.getBody(handleA)->getOrientation().angleTo(bodyB->getOrientation());

    REQUIRE(distAfter < distBefore);
    REQUIRE(distAfter < 3.0f);
    REQUIRE(angleAfter < angleBefore);
}

TEST_CASE("RigidBodyWorld hinge joint", "[physics][solver]") {
    RigidBodyWorldConfig worldConfig;
    worldConfig.gravity = Vec3(0, 0, 0);
    RigidBodyWorld world(worldConfig);

    BoxShape boxShape(Vec3(0.5f, 0.5f, 0.5f));

    RigidBodyConfig configA;
    configA.shape = &boxShape;
    configA.transform.position = Vec3(0, 0, 0);
    configA.type = MotionType::Static;

    RigidBodyConfig configB;
    configB.shape = &boxShape;
    configB.transform.position = Vec3(2.0f, 0, 0);
    configB.transform.rotation = Quat::fromAxisAngle(Vec3::unitX(), Math::radians(60.0f));

    auto handleA = world.createBody(configA);
    auto handleB = world.createBody(configB);

    HingeJointConfig jointConfig;
    jointConfig.bodyIdA = handleA.value;
    jointConfig.bodyIdB = handleB.value;
    jointConfig.localAxisA = Vec3::unitY();
    jointConfig.localAxisB = Vec3::unitY();
    jointConfig.stiffness = 0.9f;
    jointConfig.damping = 0.2f;

    world.createHingeJoint(jointConfig);

    RigidBody* bodyA = world.getBody(handleA);
    RigidBody* bodyB = world.getBody(handleB);
    REQUIRE(bodyA != nullptr);
    REQUIRE(bodyB != nullptr);

    Vec3 axisAStart = (bodyA->getOrientation() * jointConfig.localAxisA).normalized();
    Vec3 axisBStart = (bodyB->getOrientation() * jointConfig.localAxisB).normalized();
    f32 alignmentBefore = Math::abs(axisAStart.dot(axisBStart));

    for (int i = 0; i < 120; ++i) {
        world.step(1.0f / 60.0f);
    }

    Vec3 axisAEnd = (bodyA->getOrientation() * jointConfig.localAxisA).normalized();
    Vec3 axisBEnd = (bodyB->getOrientation() * jointConfig.localAxisB).normalized();
    f32 alignmentAfter = Math::abs(axisAEnd.dot(axisBEnd));

    f32 distAfter = (bodyB->getPosition() - bodyA->getPosition()).length();

    REQUIRE(alignmentAfter > alignmentBefore);
    REQUIRE(distAfter < 3.0f);
}

// =============================================================================
// Soft Body (Cloth) Tests
// =============================================================================

TEST_CASE("ClothSimulation maintains rest length", "[physics][softbody]") {
    ClothSimulation cloth;
    cloth.initializeGrid(2, 2, 1.0f, Vec3(0.0f, 0.0f, 0.0f), 1.0f, 0.0f);

    // Pin top row
    cloth.setParticleInvMass(0, 0.0f);
    cloth.setParticleInvMass(1, 0.0f);

    cloth.setGravity(Vec3(0.0f, -9.81f, 0.0f));

    for (int i = 0; i < 60; ++i) {
        cloth.step(1.0f / 60.0f, 8);
    }

    const auto& particles = cloth.getParticles();
    REQUIRE(particles.size() == 4);

    f32 distLeft = (particles[0].position - particles[2].position).length();
    f32 distRight = (particles[1].position - particles[3].position).length();

    REQUIRE(distLeft == Catch::Approx(1.0f).margin(0.05f));
    REQUIRE(distRight == Catch::Approx(1.0f).margin(0.05f));
}

TEST_CASE("ClothSimulation shear constraints count", "[physics][softbody]") {
    ClothSimulation cloth;
    cloth.initializeGrid(2, 2, 1.0f, Vec3(0.0f, 0.0f, 0.0f), 1.0f, 0.0f,
        true, false, 0.0f, 0.0f);

    const auto& constraints = cloth.getConstraints();
    REQUIRE(constraints.size() == 6);
}
