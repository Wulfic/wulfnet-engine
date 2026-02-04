#include <catch2/catch_all.hpp>
#include "physics/dynamics/RigidBody.h"
#include "core/math/Vec3.h"

using namespace WulfNet;

// =============================================================================
// Sleep System Tests
// =============================================================================

TEST_CASE("RigidBody sleep system", "[physics][sleep]") {
    RigidBodyConfig config;
    config.mass = 1.0f;
    config.type = MotionType::Dynamic;
    config.transform.position = Vec3(0, 0, 0);
    
    RigidBody body(config);
    
    SECTION("Starts awake") {
        REQUIRE(body.isAwake());
        REQUIRE(body.getSleepTimer() == Catch::Approx(0.0f));
    }
    
    SECTION("Goes to sleep when motionless") {
        // Ensure velocity is zero
        body.setLinearVelocity(Vec3::zero());
        body.setAngularVelocity(Vec3::zero());
        
        // Step enough time to trigger sleep (default 0.5s)
        // Note: motion energy smoothing takes time to decay initially
        f32 dt = 0.1f;
        for (int i = 0; i < 20; ++i) { // 2 seconds total, ample time
            body.integrate(dt);
        }
        
        REQUIRE_FALSE(body.isAwake());
        REQUIRE(body.isStatic() == false); // Should still be dynamic
    }
    
    SECTION("Wakes up on force application") {
        body.putToSleep();
        REQUIRE_FALSE(body.isAwake());
        
        body.applyForce(Vec3(10, 0, 0));
        REQUIRE(body.isAwake());
    }
    
    SECTION("Wakes up on velocity change") {
        body.putToSleep();
        REQUIRE_FALSE(body.isAwake());
        
        body.setLinearVelocity(Vec3(0.1f, 0, 0));
        REQUIRE(body.isAwake());
    }
}
