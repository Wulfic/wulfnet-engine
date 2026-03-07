// =============================================================================
// WulfNet Engine - EngineDemo
// =============================================================================
// Flagship example demonstrating the full WulfNet Engine lifecycle:
//   1. Engine initialization with the Engine class
//   2. Physics simulation (sphere drops + fluid interaction)
//   3. Software-rendered frame output to .ppm image
//   4. Audio mixer integration
//   5. Clean shutdown
//
// This is the recommended starting point for new users.
//
// Usage:
//   EngineDemo              # Runs demo, outputs frame_output.ppm
//   EngineDemo --frames=N   # Run N frames (default: 120)
// =============================================================================

#include <WulfNet/WulfNet.h>
#include <WulfNet/Engine.h>
#include <WulfNet/EngineConfig.h>
#include <WulfNet/Version.h>
#include <WulfNet/Core/Logging/Logger.h>
#include <WulfNet/Physics/Integration/PhysicsWorld.h>
#include <WulfNet/Rendering/RenderPipeline.h>
#include <WulfNet/Rendering/RenderCommand.h>
#include <WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h>

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

using namespace WulfNet;
using namespace JPH::literals;

// =============================================================================
// PPM Image Writer (portable, no dependencies)
// =============================================================================

static bool WritePPM(const std::string& filename, const uint32_t* pixels,
                     int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        uint32_t c = pixels[i];
        uint8_t r = static_cast<uint8_t>((c >> 16) & 0xFF);
        uint8_t g = static_cast<uint8_t>((c >> 8) & 0xFF);
        uint8_t b = static_cast<uint8_t>(c & 0xFF);
        file.put(static_cast<char>(r));
        file.put(static_cast<char>(g));
        file.put(static_cast<char>(b));
    }

    return file.good();
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // -------------------------------------------------------------------------
    // Parse arguments
    // -------------------------------------------------------------------------
    int numFrames = 120; // 2 seconds at 60 fps
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--frames=") == 0) {
            numFrames = std::atoi(arg.substr(9).c_str());
            if (numFrames <= 0) numFrames = 120;
        }
    }

    // -------------------------------------------------------------------------
    // 1. Configure and initialize the Engine
    // -------------------------------------------------------------------------
    Logger::Initialize();
    Logger::SetMinLevel(LogLevel::Info);

    std::cout << "=== WulfNet EngineDemo ===" << std::endl;
    std::cout << "Version: " << WULFNET_VERSION_STRING << std::endl;
    std::cout << "Frames:  " << numFrames << std::endl;
    std::cout << std::endl;

    EngineConfig config = EngineConfig::HeadlessPhysics();
    config.appName           = "EngineDemo";
    config.enableRendering   = true;
    config.enableAudio       = true;
    config.logLevel          = LogLevel::Info;
    config.physics.gravity   = JPH::Vec3(0.0f, -9.81f, 0.0f);

    // Configure a small software-rasterized framebuffer for demo output
    config.rendering.rasterizer.width  = 320;
    config.rendering.rasterizer.height = 240;
    config.rendering.rasterizer.threadCount = 1;

    Engine engine;
    EngineInitResult result = engine.Initialize(config);
    if (result != EngineInitResult::Success) {
        std::cerr << "Engine initialization failed (code "
                  << static_cast<int>(result) << ")" << std::endl;
        return 1;
    }

    WULFNET_INFO("EngineDemo", "Engine initialized successfully");

    // -------------------------------------------------------------------------
    // 2. Set up the physics scene
    // -------------------------------------------------------------------------
    PhysicsWorld& physics = engine.GetPhysics();
    JPH::BodyInterface& bodyInterface = physics.GetBodyInterface();

    // Floor (static)
    JPH::BoxShapeSettings floorShapeSettings(JPH::Vec3(50.0f, 0.5f, 50.0f));
    JPH::ShapeSettings::ShapeResult floorResult = floorShapeSettings.Create();
    JPH::BodyCreationSettings floorSettings(
        floorResult.Get(),
        JPH::RVec3(0.0_r, -0.5_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Static,
        Layers::NON_MOVING
    );
    JPH::Body* floor = bodyInterface.CreateBody(floorSettings);
    bodyInterface.AddBody(floor->GetID(), JPH::EActivation::DontActivate);

    // Falling sphere (dynamic)
    JPH::BodyCreationSettings sphereSettings(
        new JPH::SphereShape(0.5f),
        JPH::RVec3(0.0_r, 10.0_r, 0.0_r),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );
    JPH::BodyID sphereID = bodyInterface.CreateAndAddBody(
        sphereSettings, JPH::EActivation::Activate);
    bodyInterface.SetLinearVelocity(sphereID, JPH::Vec3(0.0f, -2.0f, 0.0f));

    physics.OptimizeBroadPhase();

    WULFNET_INFO("EngineDemo", "Physics scene ready (floor + sphere)");

    // -------------------------------------------------------------------------
    // 3. Check for extended physics subsystems
    // -------------------------------------------------------------------------
    if (physics.GetFluidSystem()) {
        WULFNET_INFO("EngineDemo", "FluidSystem available");
    }
    if (physics.GetTerrainDeformation()) {
        WULFNET_INFO("EngineDemo", "TerrainDeformation available");
    }

    // -------------------------------------------------------------------------
    // 4. Simulation loop
    // -------------------------------------------------------------------------
    WULFNET_INFO("EngineDemo", "Starting simulation (" +
                 std::to_string(numFrames) + " frames)...");

    for (int frame = 0; frame < numFrames && engine.IsRunning(); ++frame) {
        engine.BeginFrame();

        // Read physics state
        JPH::RVec3 pos = bodyInterface.GetCenterOfMassPosition(sphereID);
        JPH::Vec3  vel = bodyInterface.GetLinearVelocity(sphereID);

        // Print every 30th frame
        if (frame % 30 == 0) {
            std::cout << "  Frame " << frame
                      << " | pos=(" << pos.GetX() << ", " << pos.GetY()
                      << ", " << pos.GetZ() << ")"
                      << " | vel=(" << vel.GetX() << ", " << vel.GetY()
                      << ", " << vel.GetZ() << ")"
                      << " | dt=" << engine.GetDeltaTime()
                      << std::endl;
        }

        engine.EndFrame();
    }

    // -------------------------------------------------------------------------
    // 5. Render final frame and write PPM
    // -------------------------------------------------------------------------
    RenderPipeline& renderer = engine.GetRenderer();
    if (renderer.IsInitialized()) {
        // Get the framebuffer (may just be a cleared buffer if no draw calls
        // were issued through the full rasterizer path in this simple demo)
        const uint32_t* fb = renderer.GetColorBuffer();
        int w = renderer.GetWidth();
        int h = renderer.GetHeight();

        if (fb && w > 0 && h > 0) {
            if (WritePPM("frame_output.ppm", fb, w, h)) {
                WULFNET_INFO("EngineDemo", "Wrote frame_output.ppm (" +
                             std::to_string(w) + "x" + std::to_string(h) + ")");
            }
        }
    }

    // -------------------------------------------------------------------------
    // 6. Print summary
    // -------------------------------------------------------------------------
    std::cout << std::endl;
    std::cout << "=== Demo Complete ===" << std::endl;
    std::cout << "  Total frames: " << engine.GetFrameNumber() << std::endl;
    std::cout << "  Total time:   " << engine.GetTotalTime() << "s" << std::endl;
    if (engine.GetTotalTime() > 0.0f) {
        std::cout << "  Avg FPS:      "
                  << static_cast<float>(engine.GetFrameNumber()) / engine.GetTotalTime()
                  << std::endl;
    }
    std::cout << std::endl;

    // -------------------------------------------------------------------------
    // 7. Shutdown
    // -------------------------------------------------------------------------
    engine.Shutdown();
    Logger::Get().Flush();

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
