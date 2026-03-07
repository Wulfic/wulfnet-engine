// =============================================================================
// WulfNet Engine - Engine Lifecycle Integration Tests
// =============================================================================
// Tests for the Engine class covering:
//   - Initialize/Shutdown cycle
//   - Partial init (physics-only, minimal, headless)
//   - Frame loop (BeginFrame/EndFrame)
//   - Double-init guard
//   - Config validation
//   - Subsystem access safety
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>
#include <WulfNet/Engine.h>
#include <WulfNet/EngineConfig.h>
#include <WulfNet/Version.h>
#include <WulfNet/Core/Logging/Logger.h>

using namespace WulfNet;

// =============================================================================
// Version Header Tests
// =============================================================================

void test_Engine_VersionMacros() {
    EXPECT_EQ(WULFNET_VERSION_MAJOR, 1);
    EXPECT_EQ(WULFNET_VERSION_MINOR, 0);
    EXPECT_EQ(WULFNET_VERSION_PATCH, 0);

    // String should match "1.0.0"
    std::string ver = WULFNET_VERSION_STRING;
    EXPECT_TRUE(ver == "1.0.0");
}

void test_Engine_VersionAtLeast() {
    EXPECT_TRUE(WULFNET_VERSION_AT_LEAST(1, 0, 0));
    EXPECT_TRUE(WULFNET_VERSION_AT_LEAST(0, 99, 99));
    EXPECT_FALSE(WULFNET_VERSION_AT_LEAST(2, 0, 0));
    EXPECT_FALSE(WULFNET_VERSION_AT_LEAST(1, 1, 0));
}

// =============================================================================
// EngineConfig Tests
// =============================================================================

void test_EngineConfig_Defaults() {
    EngineConfig cfg;
    EXPECT_TRUE(cfg.enablePhysics);
    EXPECT_TRUE(cfg.enableRendering);
    EXPECT_TRUE(cfg.enableAudio);
    EXPECT_TRUE(cfg.enableCompute);
    EXPECT_TRUE(cfg.Validate());
}

void test_EngineConfig_MinimalPreset() {
    auto cfg = EngineConfig::Minimal();
    EXPECT_FALSE(cfg.enablePhysics);
    EXPECT_FALSE(cfg.enableRendering);
    EXPECT_FALSE(cfg.enableAudio);
    EXPECT_FALSE(cfg.enableCompute);
    EXPECT_TRUE(cfg.Validate());
}

void test_EngineConfig_HeadlessPreset() {
    auto cfg = EngineConfig::HeadlessPhysics();
    EXPECT_TRUE(cfg.enablePhysics);
    EXPECT_FALSE(cfg.enableRendering);
    EXPECT_FALSE(cfg.enableAudio);
    EXPECT_TRUE(cfg.enableCompute);
    EXPECT_TRUE(cfg.Validate());
}

void test_EngineConfig_FullPreset() {
    auto cfg = EngineConfig::Full();
    EXPECT_TRUE(cfg.enablePhysics);
    EXPECT_TRUE(cfg.enableRendering);
    EXPECT_TRUE(cfg.enableAudio);
    EXPECT_TRUE(cfg.enableCompute);
    EXPECT_TRUE(cfg.Validate());
}

void test_EngineConfig_InvalidTimestep() {
    EngineConfig cfg;
    cfg.physicsTimestep = -1.0f;
    EXPECT_FALSE(cfg.Validate());
}

void test_EngineConfig_InvalidSubsteps() {
    EngineConfig cfg;
    cfg.maxPhysicsSubsteps = 0;
    EXPECT_FALSE(cfg.Validate());
}

void test_EngineConfig_InvalidRenderDimensions() {
    EngineConfig cfg;
    cfg.enableRendering = true;
    cfg.rendering.rasterizer.width = 0;
    EXPECT_FALSE(cfg.Validate());
}

// =============================================================================
// Engine Lifecycle Tests
// =============================================================================

void test_Engine_DefaultState() {
    Engine engine;
    EXPECT_FALSE(engine.IsRunning());
    EXPECT_FALSE(engine.IsInitialized());
    EXPECT_EQ(engine.GetFrameNumber(), static_cast<uint64_t>(0));
}

void test_Engine_MinimalInitShutdown() {
    // Minimal init — no subsystems, just core
    Engine engine;
    auto result = engine.Initialize(EngineConfig::Minimal());
    EXPECT_EQ(result, EngineInitResult::Success);
    EXPECT_TRUE(engine.IsInitialized());
    EXPECT_TRUE(engine.IsRunning());

    engine.Shutdown();
    EXPECT_FALSE(engine.IsInitialized());
    EXPECT_FALSE(engine.IsRunning());
}

void test_Engine_HeadlessPhysicsInit() {
    // Physics + compute, no rendering or audio
    Engine engine;
    auto cfg = EngineConfig::HeadlessPhysics();
    cfg.appName = "TestHeadlessPhysics";
    auto result = engine.Initialize(cfg);
    EXPECT_EQ(result, EngineInitResult::Success);
    EXPECT_TRUE(engine.IsInitialized());

    // Physics subsystem should be accessible
    // (GetPhysics() returns reference — it was initialized)
    PhysicsWorld& physics = engine.GetPhysics();
    EXPECT_TRUE(physics.IsInitialized());

    engine.Shutdown();
    EXPECT_FALSE(engine.IsInitialized());
}

void test_Engine_DoubleInitGuard() {
    Engine engine;
    auto result1 = engine.Initialize(EngineConfig::Minimal());
    EXPECT_EQ(result1, EngineInitResult::Success);

    // Second init should return Success (no-op with warning)
    auto result2 = engine.Initialize(EngineConfig::Minimal());
    EXPECT_EQ(result2, EngineInitResult::Success);

    engine.Shutdown();
}

void test_Engine_DoubleShutdown() {
    Engine engine;
    engine.Initialize(EngineConfig::Minimal());
    engine.Shutdown();

    // Second shutdown should be a no-op (not crash)
    engine.Shutdown();
    EXPECT_FALSE(engine.IsInitialized());
}

void test_Engine_DestructorCleansUp() {
    // Engine should auto-shutdown in destructor
    {
        Engine engine;
        engine.Initialize(EngineConfig::Minimal());
        EXPECT_TRUE(engine.IsInitialized());
        // destructor runs here
    }
    // If we get here without crash, cleanup worked
    EXPECT_TRUE(true);
}

void test_Engine_InvalidConfig() {
    Engine engine;
    EngineConfig cfg;
    cfg.physicsTimestep = -1.0f; // Invalid
    auto result = engine.Initialize(cfg);
    EXPECT_EQ(result, EngineInitResult::ConfigInvalid);
    EXPECT_FALSE(engine.IsInitialized());
}

// =============================================================================
// Frame Loop Tests
// =============================================================================

void test_Engine_FrameLoop_Basic() {
    Engine engine;
    engine.Initialize(EngineConfig::Minimal());

    // Run a few frames
    for (int i = 0; i < 5; ++i) {
        engine.BeginFrame();
        engine.EndFrame();
    }

    EXPECT_EQ(engine.GetFrameNumber(), static_cast<uint64_t>(5));
    EXPECT_TRUE(engine.GetTotalTime() >= 0.0f);

    engine.Shutdown();
}

void test_Engine_FrameLoop_DeltaTime() {
    Engine engine;
    engine.Initialize(EngineConfig::Minimal());

    engine.BeginFrame();
    // Delta time for first frame should be small (near zero)
    float dt = engine.GetDeltaTime();
    EXPECT_TRUE(dt >= 0.0f);
    EXPECT_TRUE(dt < 1.0f); // Certainly less than 1 second
    engine.EndFrame();

    engine.Shutdown();
}

void test_Engine_FrameLoop_WithPhysics() {
    Engine engine;
    auto cfg = EngineConfig::HeadlessPhysics();
    cfg.appName = "TestPhysicsFrameLoop";
    auto result = engine.Initialize(cfg);
    EXPECT_EQ(result, EngineInitResult::Success);

    // Run 10 frames with physics stepping
    for (int i = 0; i < 10; ++i) {
        engine.BeginFrame();
        engine.EndFrame();
    }

    EXPECT_EQ(engine.GetFrameNumber(), static_cast<uint64_t>(10));

    engine.Shutdown();
}

void test_Engine_RequestShutdown() {
    Engine engine;
    engine.Initialize(EngineConfig::Minimal());
    EXPECT_TRUE(engine.IsRunning());

    engine.RequestShutdown();
    EXPECT_FALSE(engine.IsRunning());

    // Engine is still initialized, just not "running"
    EXPECT_TRUE(engine.IsInitialized());

    engine.Shutdown();
}

void test_Engine_FPS() {
    Engine engine;
    engine.Initialize(EngineConfig::Minimal());

    // Before any frame, FPS should be 0 (deltaTime is 0)
    EXPECT_NEAR(engine.GetFPS(), 0.0f, 0.001f);

    engine.BeginFrame();
    engine.EndFrame();

    // After a frame, FPS should be some positive value
    // (or very high if frame was nearly instant)
    float fps = engine.GetFPS();
    EXPECT_TRUE(fps >= 0.0f);

    engine.Shutdown();
}

void test_Engine_ConfigPersisted() {
    Engine engine;
    auto cfg = EngineConfig::Minimal();
    cfg.appName = "PersistenceTest";
    engine.Initialize(cfg);

    const EngineConfig& stored = engine.GetConfig();
    EXPECT_TRUE(stored.appName == "PersistenceTest");
    EXPECT_FALSE(stored.enablePhysics);

    engine.Shutdown();
}

// =============================================================================
// Registration
// =============================================================================

void RegisterEngineLifecycleTests() {
    // Version
    RUN_TEST("Engine_VersionMacros", test_Engine_VersionMacros);
    RUN_TEST("Engine_VersionAtLeast", test_Engine_VersionAtLeast);

    // Config
    RUN_TEST("EngineConfig_Defaults", test_EngineConfig_Defaults);
    RUN_TEST("EngineConfig_MinimalPreset", test_EngineConfig_MinimalPreset);
    RUN_TEST("EngineConfig_HeadlessPreset", test_EngineConfig_HeadlessPreset);
    RUN_TEST("EngineConfig_FullPreset", test_EngineConfig_FullPreset);
    RUN_TEST("EngineConfig_InvalidTimestep", test_EngineConfig_InvalidTimestep);
    RUN_TEST("EngineConfig_InvalidSubsteps", test_EngineConfig_InvalidSubsteps);
    RUN_TEST("EngineConfig_InvalidRenderDimensions", test_EngineConfig_InvalidRenderDimensions);

    // Lifecycle
    RUN_TEST("Engine_DefaultState", test_Engine_DefaultState);
    RUN_TEST("Engine_MinimalInitShutdown", test_Engine_MinimalInitShutdown);
    RUN_TEST("Engine_HeadlessPhysicsInit", test_Engine_HeadlessPhysicsInit);
    RUN_TEST("Engine_DoubleInitGuard", test_Engine_DoubleInitGuard);
    RUN_TEST("Engine_DoubleShutdown", test_Engine_DoubleShutdown);
    RUN_TEST("Engine_DestructorCleansUp", test_Engine_DestructorCleansUp);
    RUN_TEST("Engine_InvalidConfig", test_Engine_InvalidConfig);

    // Frame loop
    RUN_TEST("Engine_FrameLoop_Basic", test_Engine_FrameLoop_Basic);
    RUN_TEST("Engine_FrameLoop_DeltaTime", test_Engine_FrameLoop_DeltaTime);
    RUN_TEST("Engine_FrameLoop_WithPhysics", test_Engine_FrameLoop_WithPhysics);
    RUN_TEST("Engine_RequestShutdown", test_Engine_RequestShutdown);
    RUN_TEST("Engine_FPS", test_Engine_FPS);
    RUN_TEST("Engine_ConfigPersisted", test_Engine_ConfigPersisted);
}
