// =============================================================================
// WulfNet Engine - Physics World Tests
// =============================================================================
// Tests for PhysicsWorld initialization, gravity, body creation,
// stepping, contact callbacks, and statistics.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

using namespace WulfNet;
using namespace JPH::literals;

// =============================================================================
// PhysicsWorld Tests
// =============================================================================

void test_PhysicsWorld_Initialize() {
    PhysicsWorld world;
    EXPECT_FALSE(world.IsInitialized());

    PhysicsWorldSettings settings;
    settings.maxBodies = 1024;

    bool result = world.Initialize(settings);
    EXPECT_TRUE(result);
    EXPECT_TRUE(world.IsInitialized());

    world.Shutdown();
    EXPECT_FALSE(world.IsInitialized());
}

void test_PhysicsWorld_DoubleInitialize() {
    PhysicsWorld world;

    PhysicsWorldSettings settings;
    EXPECT_TRUE(world.Initialize(settings));
    EXPECT_FALSE(world.Initialize(settings)); // Should fail

    world.Shutdown();
}

void test_PhysicsWorld_Gravity() {
    PhysicsWorld world;
    world.Initialize();

    JPH::Vec3 gravity(0.0f, -10.0f, 0.0f);
    world.SetGravity(gravity);

    JPH::Vec3 result = world.GetGravity();
    EXPECT_EQ(result.GetY(), -10.0f);

    world.Shutdown();
}

void test_PhysicsWorld_CreateBody() {
    PhysicsWorld world;
    world.Initialize();

    JPH::BodyInterface& bodyInterface = world.GetBodyInterface();

    // Create a sphere
    JPH::BodyCreationSettings settings(
        new JPH::SphereShape(1.0f),
        JPH::RVec3(0.0_r, 0.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );

    JPH::BodyID bodyID = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);
    EXPECT_FALSE(bodyID.IsInvalid());

    EXPECT_GE(world.GetNumBodies(), 1u);

    bodyInterface.RemoveBody(bodyID);
    bodyInterface.DestroyBody(bodyID);

    world.Shutdown();
}

void test_PhysicsWorld_Step() {
    PhysicsWorld world;
    world.Initialize();

    JPH::BodyInterface& bodyInterface = world.GetBodyInterface();

    // Create a falling sphere
    JPH::BodyCreationSettings settings(
        new JPH::SphereShape(0.5f),
        JPH::RVec3(0.0_r, 10.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );

    JPH::BodyID bodyID = bodyInterface.CreateAndAddBody(settings, JPH::EActivation::Activate);

    JPH::RVec3 initialPos = bodyInterface.GetCenterOfMassPosition(bodyID);

    // Step simulation
    for (int i = 0; i < 10; i++) {
        JPH::EPhysicsUpdateError error = world.Step(1.0f / 60.0f);
        EXPECT_EQ(error, JPH::EPhysicsUpdateError::None);
    }

    JPH::RVec3 finalPos = bodyInterface.GetCenterOfMassPosition(bodyID);

    // Sphere should have fallen
    EXPECT_TRUE(finalPos.GetY() < initialPos.GetY());

    bodyInterface.RemoveBody(bodyID);
    bodyInterface.DestroyBody(bodyID);

    world.Shutdown();
}

void test_PhysicsWorld_ContactCallback() {
    PhysicsWorld world;
    world.Initialize();

    bool contactDetected = false;

    world.SetContactAddedCallback([&](const ContactEvent&) {
        contactDetected = true;
    });

    JPH::BodyInterface& bodyInterface = world.GetBodyInterface();

    // Create floor
    JPH::BoxShapeSettings floorShapeSettings(JPH::Vec3(100.0f, 1.0f, 100.0f));
    JPH::ShapeSettings::ShapeResult floorShapeResult = floorShapeSettings.Create();

    JPH::BodyCreationSettings floorSettings(
        floorShapeResult.Get(),
        JPH::RVec3(0.0_r, -1.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Static,
        Layers::NON_MOVING
    );

    JPH::Body* floor = bodyInterface.CreateBody(floorSettings);
    bodyInterface.AddBody(floor->GetID(), JPH::EActivation::DontActivate);

    // Create falling sphere that will hit floor
    JPH::BodyCreationSettings sphereSettings(
        new JPH::SphereShape(0.5f),
        JPH::RVec3(0.0_r, 0.6_r, 0.0_r), // Just above floor
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );

    JPH::BodyID sphereID = bodyInterface.CreateAndAddBody(sphereSettings, JPH::EActivation::Activate);

    world.OptimizeBroadPhase();

    // Step until contact
    for (int i = 0; i < 60 && !contactDetected; i++) {
        world.Step(1.0f / 60.0f);
    }

    EXPECT_TRUE(contactDetected);

    bodyInterface.RemoveBody(sphereID);
    bodyInterface.DestroyBody(sphereID);
    bodyInterface.RemoveBody(floor->GetID());
    bodyInterface.DestroyBody(floor->GetID());

    world.Shutdown();
}

void test_PhysicsWorld_Statistics() {
    PhysicsWorld world;
    world.Initialize();

    world.Step(1.0f / 60.0f);

    const PhysicsWorld::Statistics& stats = world.GetStatistics();
    EXPECT_TRUE(stats.lastStepTimeMs > 0.0f);

    world.Shutdown();
}

// =============================================================================
// Registration
// =============================================================================

void RegisterPhysicsWorldTests() {
    RUN_TEST("PhysicsWorld_Initialize", test_PhysicsWorld_Initialize);
    RUN_TEST("PhysicsWorld_DoubleInitialize", test_PhysicsWorld_DoubleInitialize);
    RUN_TEST("PhysicsWorld_Gravity", test_PhysicsWorld_Gravity);
    RUN_TEST("PhysicsWorld_CreateBody", test_PhysicsWorld_CreateBody);
    RUN_TEST("PhysicsWorld_Step", test_PhysicsWorld_Step);
    RUN_TEST("PhysicsWorld_ContactCallback", test_PhysicsWorld_ContactCallback);
    RUN_TEST("PhysicsWorld_Statistics", test_PhysicsWorld_Statistics);
}
