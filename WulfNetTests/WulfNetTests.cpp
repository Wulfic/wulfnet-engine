// =============================================================================
// WulfNet Engine - Unit Tests
// =============================================================================
// Tests for WulfNet core systems.
// Uses simple return-code based testing (no exceptions).
// =============================================================================

#include <WulfNet/WulfNet.h>

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

#include <iostream>
#include <vector>
#include <string>
#include <cstring>

using namespace WulfNet;
using namespace JPH::literals;

// =============================================================================
// Test Framework (Simple, no exceptions)
// =============================================================================

static int g_testsRun = 0;
static int g_testsPassed = 0;
static int g_testsFailed = 0;
static std::vector<std::string> g_failedTests;
static const char* g_currentTest = nullptr;
static bool g_currentTestPassed = true;
static std::string g_failureReason;

#define EXPECT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            g_currentTestPassed = false; \
            g_failureReason = "Expected true: " #condition; \
            return; \
        } \
    } while(0)

#define EXPECT_FALSE(condition) EXPECT_TRUE(!(condition))

#define EXPECT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            g_currentTestPassed = false; \
            g_failureReason = "Expected equal: " #a " == " #b; \
            return; \
        } \
    } while(0)

#define EXPECT_GE(a, b) \
    do { \
        if ((a) < (b)) { \
            g_currentTestPassed = false; \
            g_failureReason = "Expected >= : " #a " >= " #b; \
            return; \
        } \
    } while(0)

void runTest(const char* name, void (*testFunc)()) {
    g_testsRun++;
    g_currentTest = name;
    g_currentTestPassed = true;
    g_failureReason.clear();

    std::cout << "Running: " << name << "... ";
    std::cout.flush();

    testFunc();

    if (g_currentTestPassed) {
        g_testsPassed++;
        std::cout << "PASSED" << std::endl;
    } else {
        g_testsFailed++;
        g_failedTests.push_back(std::string(name) + ": " + g_failureReason);
        std::cout << "FAILED: " << g_failureReason << std::endl;
    }
}

// =============================================================================
// Logger Tests
// =============================================================================

void test_Logger_Singleton() {
    Logger& logger1 = Logger::Get();
    Logger& logger2 = Logger::Get();
    EXPECT_EQ(&logger1, &logger2);
}

void test_Logger_SetMinLevel() {
    Logger& logger = Logger::Get();
    logger.SetMinLevel(LogLevel::Warning);
    EXPECT_EQ(logger.GetMinLevel(), LogLevel::Warning);
    logger.SetMinLevel(LogLevel::Info); // Reset
}

void test_Logger_Statistics() {
    Logger& logger = Logger::Get();
    logger.ResetStatistics();

    size_t initialCount = logger.GetLogCount();
    logger.SetMinLevel(LogLevel::Debug);
    WULFNET_INFO("Test", "Test message");
    EXPECT_TRUE(logger.GetLogCount() > initialCount);
    logger.SetMinLevel(LogLevel::Error); // Reset
}

void test_Logger_ErrorCount() {
    Logger& logger = Logger::Get();
    logger.ResetStatistics();

    WULFNET_ERROR("Test", "Test error");
    EXPECT_EQ(logger.GetErrorCount(), static_cast<size_t>(1));
}

void test_Logger_WarningCount() {
    Logger& logger = Logger::Get();
    logger.ResetStatistics();
    logger.SetMinLevel(LogLevel::Warning);

    WULFNET_WARNING("Test", "Test warning");
    EXPECT_EQ(logger.GetWarningCount(), static_cast<size_t>(1));
    logger.SetMinLevel(LogLevel::Error); // Reset
}

void test_Logger_CallbackSink() {
    bool callbackCalled = false;
    LogLevel capturedLevel = LogLevel::Off;

    auto callback = [&](const LogEntry& entry) {
        callbackCalled = true;
        capturedLevel = entry.level;
    };

    auto sink = std::make_shared<CallbackLogSink>(callback);
    Logger::Get().AddSink(sink);
    Logger::Get().SetMinLevel(LogLevel::Debug);

    WULFNET_INFO("Test", "Callback test");

    EXPECT_TRUE(callbackCalled);
    EXPECT_EQ(capturedLevel, LogLevel::Info);

    Logger::Get().RemoveSink(sink);
    Logger::Get().SetMinLevel(LogLevel::Error);
}

// =============================================================================
// Profiler Tests
// =============================================================================

void test_ManualTimer_ElapsedTime() {
    ManualTimer timer;
    timer.Start();

    // Do some work
    volatile int sum = 0;
    for (int i = 0; i < 100000; i++) {
        sum += i;
    }
    (void)sum;

    double elapsed = timer.ElapsedMicroseconds();
    EXPECT_TRUE(elapsed > 0.0);
}

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
// Main
// =============================================================================

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    // Suppress logging output during tests
    Logger::Get().SetMinLevel(LogLevel::Error);

    std::cout << "=== WulfNet Engine Unit Tests ===" << std::endl;
    std::cout << std::endl;

    // Logger tests
    runTest("Logger_Singleton", test_Logger_Singleton);
    runTest("Logger_SetMinLevel", test_Logger_SetMinLevel);
    runTest("Logger_Statistics", test_Logger_Statistics);
    runTest("Logger_ErrorCount", test_Logger_ErrorCount);
    runTest("Logger_WarningCount", test_Logger_WarningCount);
    runTest("Logger_CallbackSink", test_Logger_CallbackSink);

    // Profiler tests
    runTest("ManualTimer_ElapsedTime", test_ManualTimer_ElapsedTime);

    // PhysicsWorld tests
    runTest("PhysicsWorld_Initialize", test_PhysicsWorld_Initialize);
    runTest("PhysicsWorld_DoubleInitialize", test_PhysicsWorld_DoubleInitialize);
    runTest("PhysicsWorld_Gravity", test_PhysicsWorld_Gravity);
    runTest("PhysicsWorld_CreateBody", test_PhysicsWorld_CreateBody);
    runTest("PhysicsWorld_Step", test_PhysicsWorld_Step);
    runTest("PhysicsWorld_ContactCallback", test_PhysicsWorld_ContactCallback);
    runTest("PhysicsWorld_Statistics", test_PhysicsWorld_Statistics);

    std::cout << std::endl;
    std::cout << "=== Test Results ===" << std::endl;
    std::cout << "Passed: " << g_testsPassed << "/" << g_testsRun << std::endl;

    if (g_testsFailed > 0) {
        std::cout << "Failed: " << g_testsFailed << std::endl;
        for (const std::string& failure : g_failedTests) {
            std::cout << "  - " << failure << std::endl;
        }
        return 1;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
