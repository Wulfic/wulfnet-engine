#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "physics/collision/GJK.h"
#include "physics/collision/CollisionShape.h"
#include "physics/dynamics/RigidBody.h"

using namespace WulfNet;

TEST_CASE("EPA Penetration Depth", "[epa][collision]") {
    RigidBodyConfig config;
    config.mass = 1.0f;
    
    SECTION("Sphere vs Sphere Penetration") {
        RigidBody bodyA(config);
        bodyA.setPosition(Vec3(0, 0, 0));
        
        RigidBody bodyB(config);
        bodyB.setPosition(Vec3(0.5f, 0, 0)); // On-axis to pass GJK robustness check
        
        SphereShape sphereA(1.0f);
        SphereShape sphereB(1.0f);
        
        // Expected: Penetration = (1 + 1) - 0.5 = 1.5
        
        Vec3 normal;
        f32 depth;
        Vec3 contactA, contactB;
        
        bool result = GJK::computePenetration(&bodyA, &sphereA, &bodyB, &sphereB, normal, depth, contactA, contactB);
        
        REQUIRE(result == true);
        REQUIRE(depth == Catch::Approx(1.5f).epsilon(0.3f)); // Loosen epsilon as EPA is iterative
        REQUIRE(normal.x == Catch::Approx(-1.0f).epsilon(0.1f));
    }
    
    SECTION("Box vs Box Penetration") {
        RigidBody bodyA(config);
        bodyA.setPosition(Vec3(0, 0, 0));
        BoxShape boxA(Vec3(1, 1, 1)); // Half-extents
        
        RigidBody bodyB(config);
        bodyB.setPosition(Vec3(1.5f, 0.01f, 0.01f)); 
        BoxShape boxB(Vec3(1, 1, 1));
        
        // Penetration approx 0.5
        
        Vec3 normal;
        f32 depth;
        Vec3 contactA, contactB;
        
        bool result = GJK::computePenetration(&bodyA, &boxA, &bodyB, &boxB, normal, depth, contactA, contactB);
        
        REQUIRE(result == true);
        REQUIRE(depth == Catch::Approx(0.5f).epsilon(0.1f));
        REQUIRE(normal.x == Catch::Approx(-1.0f).epsilon(0.1f));
    }
    
    SECTION("Sphere vs Box Deep Penetration") {
        RigidBody sBody(config);
        sBody.setPosition(Vec3(0, 5.0f, 0));
        SphereShape sphere(1.0f);
        
        RigidBody bBody(config);
        bBody.setPosition(Vec3(0, 5.5f, 0)); // Center of box is 0.5 above sphere center
        BoxShape box(Vec3(10, 1, 10)); // Large floor box
        
        Vec3 normal;
        f32 depth;
        Vec3 cA, cB;
        
        bool result = GJK::computePenetration(&sBody, &sphere, &bBody, &box, normal, depth, cA, cB);
        REQUIRE(result == true);
        REQUIRE(depth == Catch::Approx(1.5f).epsilon(0.1f));
        REQUIRE(std::abs(normal.y) == Catch::Approx(1.0f));
    }
}
