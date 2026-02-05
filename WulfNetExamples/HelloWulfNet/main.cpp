// =============================================================================
// WulfNet Engine - HelloWulfNet Example
// =============================================================================
// A simple example demonstrating the WulfNet Engine physics wrapper.
// =============================================================================

#include <WulfNet/WulfNet.h>

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

#include <iostream>
#include <thread>
#include <chrono>

using namespace WulfNet;
using namespace JPH::literals;

int main(int argc, char** argv) {
    // Configure logging
    Logger::Get().SetMinLevel(LogLevel::Debug);

    WULFNET_INFO("HelloWulfNet", "=== WulfNet Engine Example ===");
    WULFNET_INFO("HelloWulfNet", "Version: " WULFNET_VERSION_STRING);

    // Create and initialize physics world
    PhysicsWorld world;

    PhysicsWorldSettings settings;
    settings.maxBodies = 1024;
    settings.maxBodyPairs = 2048;
    settings.maxContactConstraints = 2048;
    settings.gravity = JPH::Vec3(0.0f, -9.81f, 0.0f);

    if (!world.Initialize(settings)) {
        WULFNET_FATAL("HelloWulfNet", "Failed to initialize physics world!");
        return 1;
    }

    // Set up contact callback
    world.SetContactAddedCallback([](const ContactEvent& event) {
        WULFNET_DEBUG("HelloWulfNet", "Contact detected!");
    });

    // Get body interface
    JPH::BodyInterface& bodyInterface = world.GetBodyInterface();

    // Create floor
    WULFNET_INFO("HelloWulfNet", "Creating floor...");
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
    WULFNET_DEBUG("HelloWulfNet", "Floor created with ID: " + std::to_string(floor->GetID().GetIndexAndSequenceNumber()));

    // Create falling sphere
    WULFNET_INFO("HelloWulfNet", "Creating falling sphere...");
    JPH::BodyCreationSettings sphereSettings(
        new JPH::SphereShape(0.5f),
        JPH::RVec3(0.0_r, 10.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );

    JPH::BodyID sphereID = bodyInterface.CreateAndAddBody(sphereSettings, JPH::EActivation::Activate);
    WULFNET_DEBUG("HelloWulfNet", "Sphere created with ID: " + std::to_string(sphereID.GetIndexAndSequenceNumber()));

    // Give the sphere some initial velocity
    bodyInterface.SetLinearVelocity(sphereID, JPH::Vec3(0.0f, -5.0f, 0.0f));

    // Optimize broadphase
    world.OptimizeBroadPhase();

    // Simulation loop
    WULFNET_INFO("HelloWulfNet", "Starting simulation...");
    const float deltaTime = 1.0f / 60.0f;
    const int numSteps = 300; // 5 seconds at 60 Hz

    for (int step = 0; step < numSteps; ++step) {
        // Step physics
        JPH::EPhysicsUpdateError error = world.Step(deltaTime);
        if (error != JPH::EPhysicsUpdateError::None) {
            WULFNET_ERROR("HelloWulfNet", "Physics update error!");
            break;
        }

        // Get sphere position
        JPH::RVec3 position = bodyInterface.GetCenterOfMassPosition(sphereID);
        JPH::Vec3 velocity = bodyInterface.GetLinearVelocity(sphereID);

        // Print every 30 steps (0.5 seconds)
        if (step % 30 == 0) {
            std::cout << "Step " << step << ": Position = ("
                      << position.GetX() << ", " << position.GetY() << ", " << position.GetZ()
                      << "), Velocity Y = " << velocity.GetY()
                      << ", Active Bodies = " << world.GetNumActiveBodies() << std::endl;
        }

        // Check if sphere has come to rest
        if (!bodyInterface.IsActive(sphereID)) {
            WULFNET_INFO("HelloWulfNet", "Sphere has come to rest at step " + std::to_string(step));
            break;
        }
    }

    // Print final statistics
    const auto& stats = world.GetStatistics();
    WULFNET_INFO("HelloWulfNet", "=== Final Statistics ===");
    WULFNET_INFO("HelloWulfNet", "Total Bodies: " + std::to_string(stats.numBodies));
    WULFNET_INFO("HelloWulfNet", "Active Bodies: " + std::to_string(stats.numActiveBodies));
    WULFNET_INFO("HelloWulfNet", "Last Step Time: " + std::to_string(stats.lastStepTimeMs) + " ms");

    // Cleanup
    bodyInterface.RemoveBody(sphereID);
    bodyInterface.DestroyBody(sphereID);
    bodyInterface.RemoveBody(floor->GetID());
    bodyInterface.DestroyBody(floor->GetID());

    world.Shutdown();

    WULFNET_INFO("HelloWulfNet", "=== Example Complete ===");

    return 0;
}
