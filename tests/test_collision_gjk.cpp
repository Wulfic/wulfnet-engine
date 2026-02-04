#include <catch2/catch_test_macros.hpp>
#include "physics/collision/GJK.h"
#include "physics/collision/CollisionShape.h"
#include "physics/dynamics/RigidBody.h"
#include "core/math/Vec3.h"

using namespace WulfNet;

TEST_CASE("GJK Intersection Test", "[physics][collision][gjk]") {
    // Create shapes
    std::vector<Vec3> boxVerts = {
        Vec3(-1,-1,-1), Vec3(1,-1,-1), Vec3(1,1,-1), Vec3(-1,1,-1),
        Vec3(-1,-1,1), Vec3(1,-1,1), Vec3(1,1,1), Vec3(-1,1,1)
    };
    ConvexHullShape boxShape(boxVerts);
    
    // Create bodies
    RigidBodyConfig config;
    config.mass = 1.0f;
    config.shape = &boxShape;
    config.transform = Transform(Vec3::zero());
    
    RigidBody bodyA(config); // At 0,0,0
    
    RigidBody bodyB(config); // At 0,0,0 initially
    
    SECTION("Identical bodies overlap") {
        REQUIRE(GJK::intersect(&bodyA, &boxShape, &bodyB, &boxShape));
    }
    
    SECTION("Separated bodies do not overlap") {
        // Box is size 2x2x2. Extents are +/- 1.
        
        // Move bodyB to (3,0,0). Min X is 2. Max X of bodyA is 1. Gap is 1.
        bodyB.setPosition(Vec3(3.0f, 0.0f, 0.0f)); 
        REQUIRE_FALSE(GJK::intersect(&bodyA, &boxShape, &bodyB, &boxShape));
        
        // Move bodyB to (1.5,0,0). Min X is 0.5. Max X of A is 1. Overlap 0.5.
        bodyB.setPosition(Vec3(1.5f, 0.0f, 0.0f)); 
        REQUIRE(GJK::intersect(&bodyA, &boxShape, &bodyB, &boxShape));
    }
    
    SECTION("Sphere vs Box") {
        SphereShape sphere(1.0f);
        RigidBodyConfig sConf = config;
        sConf.shape = &sphere;
        RigidBody bodyS(sConf);
        
        // Box extends y from -1 to 1. Sphere radius 1.
        // Place sphere at y=2.5. Lowest point is 1.5. No overlap.
        bodyS.setPosition(Vec3(0, 2.5f, 0));
        REQUIRE_FALSE(GJK::intersect(&bodyA, &boxShape, &bodyS, &sphere));
        
        // Place sphere at y=1.5. Lowest point is 0.5. Overlap with box max Y=1.
        bodyS.setPosition(Vec3(0, 1.5f, 0));
        REQUIRE(GJK::intersect(&bodyA, &boxShape, &bodyS, &sphere));
    }
    
    SECTION("Diamond case (Rotated box)") {
        // Rotate B by 45 degrees around Z
        // In local space corners are at distance sqrt(2) approx from center in plane.
        // Let's just checking basic rotation handling
        bodyB.setPosition(Vec3(2.5f, 0, 0)); // No overlap
        bodyB.setOrientation(Quat::fromAxisAngle(Vec3(0,0,1), 45.0f * Math::DEG_TO_RAD));
        
        REQUIRE_FALSE(GJK::intersect(&bodyA, &boxShape, &bodyB, &boxShape));
        
        // At 45 deg, the corner of B is pointing at A. 
        // Corner dist from center is sqrt(1^2 + 1^2) = 1.414.
        // A max X is 1.
        // If center is at 2.3. Min X of B is 2.3 - 1.414 = 0.886. Overlap with 1.
        
        bodyB.setPosition(Vec3(2.3f, 0, 0));
        REQUIRE(GJK::intersect(&bodyA, &boxShape, &bodyB, &boxShape));
    }

    SECTION("Sphere vs Sphere") {
        SphereShape s1(1.0f);
        SphereShape s2(1.0f);
        
        RigidBodyConfig c; 
        RigidBody b1(c); b1.setPosition(Vec3(0,0,0));
        RigidBody b2(c); b2.setPosition(Vec3(2.5f,0,0)); // Split
        
        REQUIRE_FALSE(GJK::intersect(&b1, &s1, &b2, &s2));
        
        b2.setPosition(Vec3(1.5f, 0, 0)); // Overlap 0.5
        REQUIRE(GJK::intersect(&b1, &s1, &b2, &s2));
        
        // Deep penetration
        b2.setPosition(Vec3(0.5f, 0, 0)); // Overlap 1.5
        REQUIRE(GJK::intersect(&b1, &s1, &b2, &s2));
    }
}
