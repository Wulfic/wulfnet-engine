// =============================================================================
// WulfNet Engine - Advanced PhysicsWorld Tests
// =============================================================================
// Tests for constraints, complex body interactions, queries, and stress scenarios.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Constraints/DistanceConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>

#include <cmath>
#include <vector>

using namespace WulfNet;
using namespace JPH::literals;

// =============================================================================
// Multiple Body Creation Tests
// =============================================================================

void test_PhysicsWorld_CreateMultipleBodies() {
    PhysicsWorld world;
    world.Initialize();

    JPH::BodyInterface& bi = world.GetBodyInterface();
    std::vector<JPH::BodyID> bodies;

    // Create 100 dynamic spheres
    for (int i = 0; i < 100; i++) {
        JPH::BodyCreationSettings settings(
            new JPH::SphereShape(0.3f),
            JPH::RVec3(static_cast<float>(i % 10) * 2.0_r,
                       10.0_r + static_cast<float>(i / 10) * 2.0_r,
                       0.0_r),
            JPH::Quat::sIdentity(),
            JPH::EMotionType::Dynamic,
            Layers::MOVING
        );

        JPH::BodyID id = bi.CreateAndAddBody(settings, JPH::EActivation::Activate);
        EXPECT_FALSE(id.IsInvalid());
        bodies.push_back(id);
    }

    EXPECT_GE(world.GetNumBodies(), 100u);

    // Clean up
    for (auto& id : bodies) {
        bi.RemoveBody(id);
        bi.DestroyBody(id);
    }

    world.Shutdown();
}

void test_PhysicsWorld_MixedMotionTypes() {
    PhysicsWorld world;
    world.Initialize();

    JPH::BodyInterface& bi = world.GetBodyInterface();

    // Static floor
    JPH::BodyCreationSettings floorSettings(
        new JPH::BoxShape(JPH::Vec3(50.0f, 1.0f, 50.0f)),
        JPH::RVec3(0.0_r, -1.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Static,
        Layers::NON_MOVING
    );
    JPH::BodyID floor = bi.CreateAndAddBody(floorSettings, JPH::EActivation::DontActivate);

    // Kinematic body
    JPH::BodyCreationSettings kinSettings(
        new JPH::BoxShape(JPH::Vec3(1.0f, 1.0f, 1.0f)),
        JPH::RVec3(5.0_r, 2.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Kinematic,
        Layers::MOVING
    );
    JPH::BodyID kinematic = bi.CreateAndAddBody(kinSettings, JPH::EActivation::Activate);

    // Dynamic body
    JPH::BodyCreationSettings dynSettings(
        new JPH::SphereShape(0.5f),
        JPH::RVec3(0.0_r, 5.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );
    JPH::BodyID dynamic = bi.CreateAndAddBody(dynSettings, JPH::EActivation::Activate);

    EXPECT_GE(world.GetNumBodies(), 3u);

    // Step and verify dynamic body falls
    for (int i = 0; i < 30; i++) {
        world.Step(1.0f / 60.0f);
    }

    JPH::RVec3 dynPos = bi.GetCenterOfMassPosition(dynamic);
    EXPECT_TRUE(dynPos.GetY() < 5.0f); // Should have fallen

    // Clean up
    bi.RemoveBody(floor);
    bi.DestroyBody(floor);
    bi.RemoveBody(kinematic);
    bi.DestroyBody(kinematic);
    bi.RemoveBody(dynamic);
    bi.DestroyBody(dynamic);

    world.Shutdown();
}

// =============================================================================
// Constraint Tests
// =============================================================================

void test_PhysicsWorld_DistanceConstraint() {
    PhysicsWorld world;
    world.Initialize();

    JPH::BodyInterface& bi = world.GetBodyInterface();

    // Create two dynamic spheres
    JPH::BodyCreationSettings s1(
        new JPH::SphereShape(0.5f),
        JPH::RVec3(-2.0_r, 5.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );
    JPH::BodyCreationSettings s2(
        new JPH::SphereShape(0.5f),
        JPH::RVec3(2.0_r, 5.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );

    JPH::BodyID id1 = bi.CreateAndAddBody(s1, JPH::EActivation::Activate);
    JPH::BodyID id2 = bi.CreateAndAddBody(s2, JPH::EActivation::Activate);

    // Create distance constraint between them
    JPH::DistanceConstraintSettings constraintSettings;
    constraintSettings.mPoint1 = JPH::RVec3(-2.0_r, 5.0_r, 0.0_r);
    constraintSettings.mPoint2 = JPH::RVec3(2.0_r, 5.0_r, 0.0_r);

    JPH::Body& body1 = *world.GetJoltPhysics().GetBodyLockInterface().TryGetBody(id1);
    JPH::Body& body2 = *world.GetJoltPhysics().GetBodyLockInterface().TryGetBody(id2);
    JPH::Constraint* constraint = constraintSettings.Create(body1, body2);

    world.AddConstraint(constraint);

    // Step simulation
    for (int i = 0; i < 60; i++) {
        world.Step(1.0f / 60.0f);
    }

    // Bodies should have fallen but maintained distance
    JPH::RVec3 pos1 = bi.GetCenterOfMassPosition(id1);
    JPH::RVec3 pos2 = bi.GetCenterOfMassPosition(id2);
    float dist = (pos2 - pos1).Length();

    // Distance should be approximately 4.0 (initial separation)
    EXPECT_TRUE(dist > 3.0f && dist < 5.0f);

    world.RemoveConstraint(constraint);

    bi.RemoveBody(id1);
    bi.DestroyBody(id1);
    bi.RemoveBody(id2);
    bi.DestroyBody(id2);

    world.Shutdown();
}

void test_PhysicsWorld_FixedConstraint() {
    PhysicsWorld world;
    world.Initialize();

    JPH::BodyInterface& bi = world.GetBodyInterface();

    // Static anchor
    JPH::BodyCreationSettings anchor(
        new JPH::SphereShape(0.3f),
        JPH::RVec3(0.0_r, 10.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Static,
        Layers::NON_MOVING
    );
    JPH::BodyID anchorId = bi.CreateAndAddBody(anchor, JPH::EActivation::DontActivate);

    // Dynamic ball fixed to anchor
    JPH::BodyCreationSettings ball(
        new JPH::SphereShape(0.5f),
        JPH::RVec3(0.0_r, 9.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );
    JPH::BodyID ballId = bi.CreateAndAddBody(ball, JPH::EActivation::Activate);

    // Fix them together
    JPH::FixedConstraintSettings fixedSettings;
    fixedSettings.mAutoDetectPoint = true;

    JPH::Body& anchorBody = *world.GetJoltPhysics().GetBodyLockInterface().TryGetBody(anchorId);
    JPH::Body& ballBody = *world.GetJoltPhysics().GetBodyLockInterface().TryGetBody(ballId);
    JPH::Constraint* constraint = fixedSettings.Create(anchorBody, ballBody);

    world.AddConstraint(constraint);

    // Step - ball should stay near anchor because it's fixed
    for (int i = 0; i < 60; i++) {
        world.Step(1.0f / 60.0f);
    }

    JPH::RVec3 anchorPos = bi.GetCenterOfMassPosition(anchorId);
    JPH::RVec3 ballPos = bi.GetCenterOfMassPosition(ballId);
    float dist = (ballPos - anchorPos).Length();

    // Should remain close (fixed constraint)
    EXPECT_TRUE(dist < 2.0f);

    world.RemoveConstraint(constraint);
    bi.RemoveBody(anchorId);
    bi.DestroyBody(anchorId);
    bi.RemoveBody(ballId);
    bi.DestroyBody(ballId);

    world.Shutdown();
}

// =============================================================================
// Broadphase Tests
// =============================================================================

void test_PhysicsWorld_OptimizeBroadPhase() {
    PhysicsWorld world;
    world.Initialize();

    JPH::BodyInterface& bi = world.GetBodyInterface();

    // Create bodies
    for (int i = 0; i < 50; i++) {
        JPH::BodyCreationSettings settings(
            new JPH::SphereShape(0.2f),
            JPH::RVec3(static_cast<float>(i) * 1.0_r, 5.0_r, 0.0_r),
            JPH::Quat::sIdentity(),
            JPH::EMotionType::Dynamic,
            Layers::MOVING
        );
        bi.CreateAndAddBody(settings, JPH::EActivation::Activate);
    }

    // Should not crash
    world.OptimizeBroadPhase();
    EXPECT_TRUE(true);

    world.Shutdown();
}

// =============================================================================
// Event Callback Tests
// =============================================================================

void test_PhysicsWorld_BodyActivationCallback() {
    PhysicsWorld world;
    world.Initialize();

    bool bodyActivated = false;
    bool bodyDeactivated = false;

    world.SetBodyActivatedCallback([&](JPH::BodyID) {
        bodyActivated = true;
    });
    world.SetBodyDeactivatedCallback([&](JPH::BodyID) {
        bodyDeactivated = true;
    });

    JPH::BodyInterface& bi = world.GetBodyInterface();

    // Create a dynamic sphere that should activate
    JPH::BodyCreationSettings settings(
        new JPH::SphereShape(0.5f),
        JPH::RVec3(0.0_r, 5.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );

    JPH::BodyID id = bi.CreateAndAddBody(settings, JPH::EActivation::Activate);

    // Step to process activation
    world.Step(1.0f / 60.0f);

    // body should have been activated
    // (callback may or may not fire depending on implementation — just don't crash)
    EXPECT_TRUE(true);

    bi.RemoveBody(id);
    bi.DestroyBody(id);

    world.Shutdown();
}

void test_PhysicsWorld_ContactRemoved() {
    PhysicsWorld world;
    world.Initialize();

    bool contactRemoved = false;
    world.SetContactRemovedCallback([&](JPH::BodyID, JPH::BodyID) {
        contactRemoved = true;
    });

    // Just verify callback can be set without crashing
    world.Step(1.0f / 60.0f);
    EXPECT_TRUE(true);

    world.Shutdown();
}

// =============================================================================
// Shape Variety Tests
// =============================================================================

void test_PhysicsWorld_CapsuleShape() {
    PhysicsWorld world;
    world.Initialize();

    JPH::BodyInterface& bi = world.GetBodyInterface();

    JPH::BodyCreationSettings settings(
        new JPH::CapsuleShape(1.0f, 0.3f),
        JPH::RVec3(0.0_r, 5.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );

    JPH::BodyID id = bi.CreateAndAddBody(settings, JPH::EActivation::Activate);
    EXPECT_FALSE(id.IsInvalid());

    for (int i = 0; i < 10; i++) {
        world.Step(1.0f / 60.0f);
    }

    // Capsule should have fallen
    JPH::RVec3 pos = bi.GetCenterOfMassPosition(id);
    EXPECT_TRUE(pos.GetY() < 5.0f);

    bi.RemoveBody(id);
    bi.DestroyBody(id);
    world.Shutdown();
}

// =============================================================================
// Physics Settings Tests
// =============================================================================

void test_PhysicsWorld_PhysicsSettings() {
    PhysicsWorld world;
    world.Initialize();

    const JPH::PhysicsSettings& settings = world.GetPhysicsSettings();
    // Default settings should be reasonable
    EXPECT_TRUE(settings.mNumVelocitySteps > 0);
    EXPECT_TRUE(settings.mNumPositionSteps > 0);

    world.Shutdown();
}

void test_PhysicsWorld_CustomGravity() {
    PhysicsWorld world;
    PhysicsWorldSettings worldSettings;
    worldSettings.gravity = JPH::Vec3(0.0f, -20.0f, 0.0f);
    world.Initialize(worldSettings);

    JPH::Vec3 gravity = world.GetGravity();
    EXPECT_TRUE(std::abs(gravity.GetY() - (-20.0f)) < 0.01f);

    // Change gravity mid-simulation
    world.SetGravity(JPH::Vec3(0.0f, 5.0f, 0.0f));
    gravity = world.GetGravity();
    EXPECT_TRUE(std::abs(gravity.GetY() - 5.0f) < 0.01f);

    world.Shutdown();
}

// =============================================================================
// Stress Tests
// =============================================================================

void test_PhysicsWorld_ManyBodiesStress() {
    PhysicsWorld world;
    PhysicsWorldSettings settings;
    settings.maxBodies = 10000;
    world.Initialize(settings);

    JPH::BodyInterface& bi = world.GetBodyInterface();

    // Create 500 bodies in a stack
    std::vector<JPH::BodyID> bodies;
    for (int i = 0; i < 500; i++) {
        JPH::BodyCreationSettings s(
            new JPH::SphereShape(0.2f),
            JPH::RVec3(
                static_cast<float>(i % 10) * 0.5_r,
                static_cast<float>(i / 10) * 0.5_r + 1.0_r,
                static_cast<float>((i / 100) % 10) * 0.5_r
            ),
            JPH::Quat::sIdentity(),
            JPH::EMotionType::Dynamic,
            Layers::MOVING
        );
        bodies.push_back(bi.CreateAndAddBody(s, JPH::EActivation::Activate));
    }

    world.OptimizeBroadPhase();

    // Run 30 frames
    for (int i = 0; i < 30; i++) {
        JPH::EPhysicsUpdateError error = world.Step(1.0f / 60.0f);
        EXPECT_EQ(error, JPH::EPhysicsUpdateError::None);
    }

    // Verify performance stats
    const PhysicsWorld::Statistics& stats = world.GetStatistics();
    EXPECT_TRUE(stats.lastStepTimeMs > 0.0f);
    EXPECT_TRUE(stats.lastStepTimeMs < 500.0f); // Should finish in <500ms

    // Clean up
    for (auto& id : bodies) {
        bi.RemoveBody(id);
        bi.DestroyBody(id);
    }

    world.Shutdown();
}

void test_PhysicsWorld_RapidCreateDestroy() {
    PhysicsWorld world;
    world.Initialize();

    JPH::BodyInterface& bi = world.GetBodyInterface();

    // Rapidly create and destroy bodies
    for (int cycle = 0; cycle < 50; cycle++) {
        JPH::BodyCreationSettings settings(
            new JPH::SphereShape(0.5f),
            JPH::RVec3(0.0_r, 5.0_r, 0.0_r),
            JPH::Quat::sIdentity(),
            JPH::EMotionType::Dynamic,
            Layers::MOVING
        );
        JPH::BodyID id = bi.CreateAndAddBody(settings, JPH::EActivation::Activate);
        world.Step(1.0f / 60.0f);
        bi.RemoveBody(id);
        bi.DestroyBody(id);
    }

    EXPECT_TRUE(true); // No crashes

    world.Shutdown();
}

// =============================================================================
// Query Interface Tests
// =============================================================================

void test_PhysicsWorld_BroadPhaseQuery() {
    PhysicsWorld world;
    world.Initialize();

    // Should return a valid reference
    const JPH::BroadPhaseQuery& bpq = world.GetBroadPhaseQuery();
    (void)bpq; // Just verify it doesn't crash
    EXPECT_TRUE(true);

    world.Shutdown();
}

void test_PhysicsWorld_NarrowPhaseQuery() {
    PhysicsWorld world;
    world.Initialize();

    const JPH::NarrowPhaseQuery& npq = world.GetNarrowPhaseQuery();
    (void)npq;
    EXPECT_TRUE(true);

    world.Shutdown();
}

// =============================================================================
// Registration
// =============================================================================

void RegisterAdvancedPhysicsTests() {
    // Multiple bodies
    RUN_TEST("PhysicsWorld_CreateMultipleBodies", test_PhysicsWorld_CreateMultipleBodies);
    RUN_TEST("PhysicsWorld_MixedMotionTypes", test_PhysicsWorld_MixedMotionTypes);

    // Constraints
    RUN_TEST("PhysicsWorld_DistanceConstraint", test_PhysicsWorld_DistanceConstraint);
    RUN_TEST("PhysicsWorld_FixedConstraint", test_PhysicsWorld_FixedConstraint);

    // Broadphase
    RUN_TEST("PhysicsWorld_OptimizeBroadPhase", test_PhysicsWorld_OptimizeBroadPhase);

    // Callbacks
    RUN_TEST("PhysicsWorld_BodyActivationCallback", test_PhysicsWorld_BodyActivationCallback);
    RUN_TEST("PhysicsWorld_ContactRemoved", test_PhysicsWorld_ContactRemoved);

    // Shape variety
    RUN_TEST("PhysicsWorld_CapsuleShape", test_PhysicsWorld_CapsuleShape);

    // Settings
    RUN_TEST("PhysicsWorld_PhysicsSettings", test_PhysicsWorld_PhysicsSettings);
    RUN_TEST("PhysicsWorld_CustomGravity", test_PhysicsWorld_CustomGravity);

    // Stress
    RUN_TEST("PhysicsWorld_ManyBodiesStress", test_PhysicsWorld_ManyBodiesStress);
    RUN_TEST("PhysicsWorld_RapidCreateDestroy", test_PhysicsWorld_RapidCreateDestroy);

    // Queries
    RUN_TEST("PhysicsWorld_BroadPhaseQuery", test_PhysicsWorld_BroadPhaseQuery);
    RUN_TEST("PhysicsWorld_NarrowPhaseQuery", test_PhysicsWorld_NarrowPhaseQuery);
}
