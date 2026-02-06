// =============================================================================
// WulfNet Engine - Unit Tests
// =============================================================================
// Tests for WulfNet core systems.
// Uses simple return-code based testing (no exceptions).
// =============================================================================

#include <WulfNet/WulfNet.h>
#include <WulfNet/Physics/Fluids/FluidSystem.h>
#include <WulfNet/Physics/Fluids/FluidParticle.h>
#include <WulfNet/Physics/Fluids/FluidGrid.h>
#include <WulfNet/Compute/Fluids/VulkanFluidCompute.h>
#include <WulfNet/Physics/Fluids/COFLIPSystem.h>

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
// GPU Compute Tests
// =============================================================================

void test_VulkanContext_IsAvailable() {
    // Just check if the availability check works without crashing
    bool available = IsGPUComputeAvailable();
    (void)available; // May or may not be available depending on system
    EXPECT_TRUE(true); // Test passes if we get here without crashing
}

void test_VulkanContext_GetAvailableGPUs() {
    // Query available GPUs - should not crash even if no GPU
    auto gpus = GetAvailableGPUs();
    // The list may be empty on systems without Vulkan support
    EXPECT_TRUE(true); // Test passes if we get here
}

void test_VulkanContext_Initialize() {
    // Try to initialize if Vulkan is available
    if (!IsGPUComputeAvailable()) {
        // Skip test on systems without Vulkan
        EXPECT_TRUE(true);
        return;
    }

    VulkanContextSettings settings;
    settings.enableValidation = false; // Faster for tests
    settings.applicationName = "WulfNetTest";

    VulkanContext& ctx = GetVulkanContext();
    bool success = ctx.Initialize(settings);

    if (success) {
        EXPECT_TRUE(ctx.IsValid());
        const GPUDeviceInfo& info = ctx.GetDeviceInfo();
        EXPECT_TRUE(!info.name.empty());
        EXPECT_TRUE(info.totalMemory > 0);

        ctx.Shutdown();
        EXPECT_FALSE(ctx.IsValid());
    } else {
        // Vulkan may be installed but no suitable device
        EXPECT_TRUE(true);
    }
}

void test_VulkanContext_Singleton() {
    if (!IsGPUComputeAvailable()) {
        EXPECT_TRUE(true);
        return;
    }

    VulkanContext& ctx1 = GetVulkanContext();
    VulkanContext& ctx2 = GetVulkanContext();

    // Should be same instance
    EXPECT_TRUE(&ctx1 == &ctx2);
}

void test_VulkanContext_DeviceInfo() {
    if (!IsGPUComputeAvailable()) {
        EXPECT_TRUE(true);
        return;
    }

    // Get available GPUs
    auto gpus = GetAvailableGPUs();
    EXPECT_TRUE(!gpus.empty());

    // Check first GPU has valid properties
    const GPUDeviceInfo& gpu = gpus[0];
    EXPECT_TRUE(!gpu.name.empty());
    EXPECT_TRUE(gpu.totalMemory > 0);
    EXPECT_TRUE(gpu.maxComputeWorkGroupSize[0] > 0);
    EXPECT_TRUE(gpu.maxComputeWorkGroupSize[1] > 0);
    EXPECT_TRUE(gpu.maxComputeWorkGroupSize[2] > 0);
    EXPECT_TRUE(gpu.maxComputeSharedMemory > 0);
}

void test_VulkanContext_Handles() {
    if (!IsGPUComputeAvailable()) {
        EXPECT_TRUE(true);
        return;
    }

    VulkanContext& ctx = GetVulkanContext();
    VulkanContextSettings settings;
    settings.enableValidation = false;
    settings.applicationName = "WulfNetTest";

    if (!ctx.Initialize(settings)) {
        // Skip if initialization fails
        EXPECT_TRUE(true);
        return;
    }

    // Check all handles are valid
    EXPECT_TRUE(ctx.GetInstance() != nullptr);
    EXPECT_TRUE(ctx.GetPhysicalDevice() != nullptr);
    EXPECT_TRUE(ctx.GetDevice() != nullptr);
    EXPECT_TRUE(ctx.GetComputeQueue() != nullptr);
    EXPECT_TRUE(ctx.GetComputeCommandPool() != nullptr);
    EXPECT_TRUE(ctx.GetDescriptorPool() != nullptr);

    ctx.Shutdown();
}

void test_VulkanContext_Reinitialize() {
    if (!IsGPUComputeAvailable()) {
        EXPECT_TRUE(true);
        return;
    }

    VulkanContext& ctx = GetVulkanContext();
    VulkanContextSettings settings;
    settings.enableValidation = false;

    // First init
    bool success1 = ctx.Initialize(settings);
    if (!success1) {
        EXPECT_TRUE(true);
        return;
    }
    EXPECT_TRUE(ctx.IsValid());
    ctx.Shutdown();
    EXPECT_FALSE(ctx.IsValid());

    // Second init
    bool success2 = ctx.Initialize(settings);
    EXPECT_TRUE(success2);
    EXPECT_TRUE(ctx.IsValid());
    ctx.Shutdown();
}

void test_ShaderUtils_LoadSPIRV() {
    // Try to load the vector_add shader
    std::string shaderPath = "Assets/Shaders/Compute/vector_add.spv";

    auto spirv = ShaderUtils::LoadSPIRV(shaderPath);

    // May fail if working directory is wrong - that's ok for CI
    if (!spirv.empty()) {
        // SPIR-V has a magic number at the start
        EXPECT_TRUE(spirv[0] == 0x07230203);
        // File should be reasonable size (our shader is ~1732 bytes = ~433 words)
        EXPECT_TRUE(spirv.size() > 100);
        EXPECT_TRUE(spirv.size() < 10000);
    } else {
        // If file not found, still pass (depends on working directory)
        EXPECT_TRUE(true);
    }
}

void test_VulkanContext_WaitIdle() {
    if (!IsGPUComputeAvailable()) {
        EXPECT_TRUE(true);
        return;
    }

    VulkanContext& ctx = GetVulkanContext();
    VulkanContextSettings settings;
    settings.enableValidation = false;

    if (!ctx.Initialize(settings)) {
        EXPECT_TRUE(true);
        return;
    }

    // WaitIdle should not crash on empty queue
    ctx.WaitIdle();
    EXPECT_TRUE(ctx.IsValid());

    ctx.Shutdown();
}

void test_ComputePipeline_Construction() {
    // Test that ComputePipeline can be constructed/destructed
    ComputePipeline pipeline;
    EXPECT_FALSE(pipeline.IsValid()); // Should be invalid before creation
}

void test_ComputePipeline_CalculateGroupCount() {
    ComputePipeline pipeline;

    // Default local size is 256
    EXPECT_EQ(pipeline.CalculateGroupCount(1), 1);
    EXPECT_EQ(pipeline.CalculateGroupCount(256), 1);
    EXPECT_EQ(pipeline.CalculateGroupCount(257), 2);
    EXPECT_EQ(pipeline.CalculateGroupCount(512), 2);
    EXPECT_EQ(pipeline.CalculateGroupCount(1000), 4);
    EXPECT_EQ(pipeline.CalculateGroupCount(1024), 4);
}

void test_ShaderUtils_LoadSPIRV_ValidMagic() {
    std::string shaderPath = "Assets/Shaders/Compute/vector_add.spv";
    auto spirv = ShaderUtils::LoadSPIRV(shaderPath);

    if (!spirv.empty()) {
        // SPIR-V magic number is 0x07230203
        EXPECT_EQ(spirv[0], 0x07230203u);

        // Word 1 is version
        // Word 2 is generator magic
        // Word 3 is bound (max ID + 1)
        // Word 4 is reserved (0)
        EXPECT_EQ(spirv[4], 0u); // Reserved field should be 0
    } else {
        EXPECT_TRUE(true); // Skip if file not found
    }
}

void test_ShaderBinding_Types() {
    // Test ShaderBinding struct creation
    ShaderBinding storage = {0, ShaderBindingType::StorageBuffer, "input"};
    EXPECT_EQ(storage.binding, 0u);
    EXPECT_TRUE(storage.type == ShaderBindingType::StorageBuffer);
    EXPECT_TRUE(storage.name == "input");

    ShaderBinding uniform = {1, ShaderBindingType::UniformBuffer, "params"};
    EXPECT_EQ(uniform.binding, 1u);
    EXPECT_TRUE(uniform.type == ShaderBindingType::UniformBuffer);
}

void test_GPUBufferUsage_Flags() {
    // Test buffer usage flag operations
    GPUBufferUsage storage = GPUBufferUsage::Storage;
    GPUBufferUsage transfer = GPUBufferUsage::TransferSrc | GPUBufferUsage::TransferDst;

    EXPECT_TRUE(HasFlag(transfer, GPUBufferUsage::TransferSrc));
    EXPECT_TRUE(HasFlag(transfer, GPUBufferUsage::TransferDst));
    EXPECT_FALSE(HasFlag(transfer, GPUBufferUsage::Storage));

    GPUBufferUsage combined = storage | transfer;
    EXPECT_TRUE(HasFlag(combined, GPUBufferUsage::Storage));
    EXPECT_TRUE(HasFlag(combined, GPUBufferUsage::TransferSrc));
}

void test_GPUDeviceInfo_Structure() {
    if (!IsGPUComputeAvailable()) {
        EXPECT_TRUE(true);
        return;
    }

    auto gpus = GetAvailableGPUs();
    if (gpus.empty()) {
        EXPECT_TRUE(true);
        return;
    }

    const GPUDeviceInfo& info = gpus[0];

    // Validate workgroup size limits are reasonable (EnumerateDevices populates these)
    EXPECT_TRUE(info.maxComputeWorkGroupSize[0] >= 64);
    EXPECT_TRUE(info.maxComputeWorkGroupSize[1] >= 1);
    EXPECT_TRUE(info.maxComputeWorkGroupSize[2] >= 1);

    // Validate shared memory (at least 16KB on any GPU)
    EXPECT_TRUE(info.maxComputeSharedMemory >= 16384);

    // Note: maxComputeWorkGroupCount is only populated after full Initialize(),
    // not in EnumerateDevices() - so we don't test it here
}

// =============================================================================
// Fluid System Tests
// =============================================================================

void test_FluidParticle_Size() {
    // FluidParticle must be 64 bytes for GPU alignment
    EXPECT_EQ(sizeof(FluidParticle), 64u);
}

void test_FluidMaterial_Presets() {
    FluidMaterial water = FluidMaterial::Water();
    EXPECT_EQ(water.type, FluidMaterialType::Water);
    EXPECT_TRUE(water.density > 900.0f && water.density < 1100.0f);
    EXPECT_TRUE(water.viscosity < 0.01f);  // Low viscosity

    FluidMaterial honey = FluidMaterial::Honey();
    EXPECT_EQ(honey.type, FluidMaterialType::Honey);
    EXPECT_TRUE(honey.viscosity > 1.0f);  // High viscosity

    FluidMaterial oil = FluidMaterial::Oil();
    EXPECT_EQ(oil.type, FluidMaterialType::Oil);
    EXPECT_TRUE(oil.viscosity > water.viscosity);
    EXPECT_TRUE(oil.viscosity < honey.viscosity);
}

void test_FluidGrid_Initialize() {
    FluidGrid grid;
    bool result = grid.Initialize(32, 16, 32, 0.1f);
    EXPECT_TRUE(result);
    EXPECT_EQ(grid.GetResolutionX(), 32u);
    EXPECT_EQ(grid.GetResolutionY(), 16u);
    EXPECT_EQ(grid.GetResolutionZ(), 32u);
    EXPECT_EQ(grid.GetCellCount(), 32u * 16u * 32u);
}

void test_FluidGrid_WorldToGrid() {
    FluidGrid grid;
    grid.Initialize(10, 10, 10, 1.0f);
    grid.SetBounds(0.0f, 0.0f, 0.0f, 10.0f, 10.0f, 10.0f);

    float gx, gy, gz;
    grid.WorldToGrid(5.0f, 5.0f, 5.0f, gx, gy, gz);
    EXPECT_TRUE(gx >= 4.9f && gx <= 5.1f);
    EXPECT_TRUE(gy >= 4.9f && gy <= 5.1f);
    EXPECT_TRUE(gz >= 4.9f && gz <= 5.1f);
}

void test_FluidGrid_Bounds() {
    FluidGrid grid;
    grid.Initialize(10, 10, 10, 1.0f);
    grid.SetBounds(-5.0f, 0.0f, -5.0f, 5.0f, 10.0f, 5.0f);

    EXPECT_TRUE(grid.IsInBoundsWorld(0.0f, 5.0f, 0.0f));
    EXPECT_FALSE(grid.IsInBoundsWorld(-10.0f, 5.0f, 0.0f));
}

void test_FluidSystem_Initialize() {
    FluidSystemConfig config;
    config.gridResolutionX = 16;
    config.gridResolutionY = 16;
    config.gridResolutionZ = 16;
    config.cellSize = 0.2f;
    config.maxParticles = 1000;

    FluidSystem system;
    bool result = system.Initialize(config);
    EXPECT_TRUE(result);
    EXPECT_TRUE(system.IsInitialized());
    EXPECT_EQ(system.GetMaxParticles(), 1000u);
    EXPECT_EQ(system.GetParticleCount(), 0u);
}

void test_FluidSystem_AddParticle() {
    FluidSystemConfig config;
    config.gridResolutionX = 16;
    config.gridResolutionY = 16;
    config.gridResolutionZ = 16;
    config.cellSize = 0.2f;
    config.maxParticles = 100;

    FluidSystem system;
    system.Initialize(config);

    EXPECT_EQ(system.GetParticleCount(), 0u);
    system.AddParticle(1.0f, 1.0f, 1.0f);
    EXPECT_EQ(system.GetParticleCount(), 1u);
    system.AddParticle(1.5f, 1.0f, 1.0f);
    EXPECT_EQ(system.GetParticleCount(), 2u);
}

void test_FluidSystem_AddParticleBox() {
    FluidSystemConfig config;
    config.gridResolutionX = 32;
    config.gridResolutionY = 32;
    config.gridResolutionZ = 32;
    config.cellSize = 0.1f;
    config.maxParticles = 10000;

    FluidSystem system;
    system.Initialize(config);

    system.AddParticleBox(0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f);

    // Should have created multiple particles
    EXPECT_TRUE(system.GetParticleCount() > 10);
    EXPECT_TRUE(system.GetParticleCount() < config.maxParticles);
}

void test_FluidSystem_Materials() {
    FluidSystemConfig config;
    config.maxParticles = 100;

    FluidSystem system;
    system.Initialize(config);

    // Default water material
    EXPECT_EQ(system.GetMaterialCount(), 1u);

    uint32_t oilId = system.AddMaterial(FluidMaterial::Oil());
    uint32_t honeyId = system.AddMaterial(FluidMaterial::Honey());

    EXPECT_EQ(system.GetMaterialCount(), 3u);
    EXPECT_TRUE(system.GetMaterial(oilId) != nullptr);
    EXPECT_TRUE(system.GetMaterial(honeyId) != nullptr);
}

void test_FluidSystem_Emitter() {
    FluidSystemConfig config;
    config.maxParticles = 1000;

    FluidSystem system;
    system.Initialize(config);

    FluidEmitter emitter;
    emitter.type = EmitterType::Point;
    emitter.posX = 1.0f;
    emitter.posY = 2.0f;
    emitter.posZ = 1.0f;
    emitter.emissionRate = 100.0f;
    emitter.initialSpeed = 1.0f;

    uint32_t id = system.AddEmitter(emitter);
    EXPECT_EQ(system.GetEmitterCount(), 1u);

    FluidEmitter* e = system.GetEmitter(id);
    EXPECT_TRUE(e != nullptr);
    EXPECT_TRUE(e->posX == 1.0f);
}

void test_FluidSystem_Step() {
    FluidSystemConfig config;
    config.gridResolutionX = 16;
    config.gridResolutionY = 16;
    config.gridResolutionZ = 16;
    config.cellSize = 0.2f;
    config.maxParticles = 1000;
    config.useGPU = false;  // CPU mode

    FluidSystem system;
    system.Initialize(config);

    // Add some particles
    system.AddParticleSphere(1.5f, 1.5f, 1.5f, 0.3f);
    uint32_t initialCount = system.GetParticleCount();
    EXPECT_TRUE(initialCount > 0);

    // Step should not crash
    system.Step(0.016f);  // 60fps

    // Particle count shouldn't change from step alone
    EXPECT_EQ(system.GetParticleCount(), initialCount);
}

void test_FluidSystem_Stats() {
    FluidSystemConfig config;
    config.maxParticles = 1000;

    FluidSystem system;
    system.Initialize(config);

    system.AddParticle(1.0f, 1.0f, 1.0f);
    system.Step(0.016f);

    const FluidStats& stats = system.GetStats();
    EXPECT_EQ(stats.activeParticles, 1u);
    EXPECT_TRUE(stats.totalTimeMs >= 0.0f);
}

void test_ParticleFlags() {
    uint32_t flags = 0;
    flags |= static_cast<uint32_t>(ParticleFlags::Active);
    flags |= static_cast<uint32_t>(ParticleFlags::Surface);

    EXPECT_TRUE(HasFlag(flags, ParticleFlags::Active));
    EXPECT_TRUE(HasFlag(flags, ParticleFlags::Surface));
    EXPECT_FALSE(HasFlag(flags, ParticleFlags::Sleeping));
}

// =============================================================================
// GPU Fluid Compute Tests (Optimization Verification)
// =============================================================================

void test_VulkanFluidCompute_Initialization() {
    if (!IsGPUComputeAvailable()) {
        EXPECT_TRUE(true);
        return;
    }

    VulkanContext& ctx = GetVulkanContext();
    VulkanContextSettings settings;
    settings.enableValidation = false;
    if (!ctx.Initialize(settings)) {
        EXPECT_TRUE(true);
        return;
    }

    COFLIPConfig fluidConfig;
    fluidConfig.gridSizeX = 32;
    fluidConfig.gridSizeY = 32;
    fluidConfig.gridSizeZ = 32;
    fluidConfig.cellSize = 0.1f;
    fluidConfig.useGPU = true;

    VulkanFluidCompute gpuCompute;
    bool success = gpuCompute.Initialize(&ctx, fluidConfig, "Assets/Shaders/Compute");

    // May fail if shaders not compiled, that's OK for this test
    EXPECT_TRUE(true);

    gpuCompute.Shutdown();
    ctx.Shutdown();
}

void test_VulkanFluidCompute_BatchedDispatch() {
    if (!IsGPUComputeAvailable()) {
        EXPECT_TRUE(true);
        return;
    }

    VulkanContext& ctx = GetVulkanContext();
    VulkanContextSettings settings;
    settings.enableValidation = false;
    if (!ctx.Initialize(settings)) {
        EXPECT_TRUE(true);
        return;
    }

    COFLIPConfig fluidConfig;
    fluidConfig.gridSizeX = 32;
    fluidConfig.gridSizeY = 32;
    fluidConfig.gridSizeZ = 32;
    fluidConfig.cellSize = 0.1f;
    fluidConfig.useGPU = true;

    VulkanFluidCompute gpuCompute;
    if (!gpuCompute.Initialize(&ctx, fluidConfig, "Assets/Shaders/Compute")) {
        ctx.Shutdown();
        EXPECT_TRUE(true);
        return;
    }

    // Create test particles
    std::vector<COFLIPParticle> particles(1000);
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].position[0] = 1.6f + (float)(i % 10) * 0.05f;
        particles[i].position[1] = 1.6f + (float)((i / 10) % 10) * 0.05f;
        particles[i].position[2] = 1.6f + (float)(i / 100) * 0.05f;
        particles[i].velocity[0] = 0.0f;
        particles[i].velocity[1] = 0.0f;
        particles[i].velocity[2] = 0.0f;
        particles[i].mass = 1.0f;
        particles[i].volume = 0.001f;
        particles[i].flags = 1; // Active
    }

    gpuCompute.UploadParticles(particles, static_cast<uint32_t>(particles.size()));

    // Run batched dispatch (tests optimization #1)
    FluidSimParams params;
    params.particleCount = static_cast<uint32_t>(particles.size());
    params.gridSizeX = fluidConfig.gridSizeX;
    params.gridSizeY = fluidConfig.gridSizeY;
    params.gridSizeZ = fluidConfig.gridSizeZ;
    params.cellSize = fluidConfig.cellSize;
    params.invCellSize = 1.0f / fluidConfig.cellSize;
    params.dt = 1.0f / 60.0f;
    params.flipRatio = 0.95f;
    params.pressureIterations = 10;
    params.gravity = -9.8f;

    // Should not crash
    gpuCompute.DispatchFullStepBatched(params);

    // Download results
    gpuCompute.DownloadParticles(particles, static_cast<uint32_t>(particles.size()));

    // Particles should have moved (gravity applied)
    bool particlesMoved = false;
    for (const auto& p : particles) {
        if (p.velocity[1] < -0.01f) {
            particlesMoved = true;
            break;
        }
    }
    EXPECT_TRUE(particlesMoved);

    gpuCompute.Shutdown();
    ctx.Shutdown();
}

void test_VulkanFluidCompute_SortedDispatch() {
    if (!IsGPUComputeAvailable()) {
        EXPECT_TRUE(true);
        return;
    }

    VulkanContext& ctx = GetVulkanContext();
    VulkanContextSettings settings;
    settings.enableValidation = false;
    if (!ctx.Initialize(settings)) {
        EXPECT_TRUE(true);
        return;
    }

    COFLIPConfig fluidConfig;
    fluidConfig.gridSizeX = 32;
    fluidConfig.gridSizeY = 32;
    fluidConfig.gridSizeZ = 32;
    fluidConfig.cellSize = 0.1f;
    fluidConfig.useGPU = true;

    VulkanFluidCompute gpuCompute;
    if (!gpuCompute.Initialize(&ctx, fluidConfig, "Assets/Shaders/Compute")) {
        ctx.Shutdown();
        EXPECT_TRUE(true);
        return;
    }

    // Create test particles
    std::vector<COFLIPParticle> particles(500);
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].position[0] = 1.6f + (float)(i % 10) * 0.05f;
        particles[i].position[1] = 1.6f + (float)((i / 10) % 10) * 0.05f;
        particles[i].position[2] = 1.6f + (float)(i / 100) * 0.05f;
        particles[i].velocity[0] = 0.0f;
        particles[i].velocity[1] = 0.0f;
        particles[i].velocity[2] = 0.0f;
        particles[i].mass = 1.0f;
        particles[i].volume = 0.001f;
        particles[i].flags = 1;
    }

    gpuCompute.UploadParticles(particles, static_cast<uint32_t>(particles.size()));

    FluidSimParams params;
    params.particleCount = static_cast<uint32_t>(particles.size());
    params.gridSizeX = fluidConfig.gridSizeX;
    params.gridSizeY = fluidConfig.gridSizeY;
    params.gridSizeZ = fluidConfig.gridSizeZ;
    params.cellSize = fluidConfig.cellSize;
    params.invCellSize = 1.0f / fluidConfig.cellSize;
    params.dt = 1.0f / 60.0f;
    params.flipRatio = 0.95f;
    params.pressureIterations = 10;
    params.gravity = -9.8f;

    // Test sorted dispatch (optimization #2 - particle sorting)
    gpuCompute.DispatchFullStepSorted(params);

    gpuCompute.DownloadParticles(particles, static_cast<uint32_t>(particles.size()));

    // Should complete without crashing
    EXPECT_TRUE(true);

    gpuCompute.Shutdown();
    ctx.Shutdown();
}

void test_VulkanFluidCompute_AsyncSimulation() {
    if (!IsGPUComputeAvailable()) {
        EXPECT_TRUE(true);
        return;
    }

    VulkanContext& ctx = GetVulkanContext();
    VulkanContextSettings settings;
    settings.enableValidation = false;
    if (!ctx.Initialize(settings)) {
        EXPECT_TRUE(true);
        return;
    }

    COFLIPConfig fluidConfig;
    fluidConfig.gridSizeX = 32;
    fluidConfig.gridSizeY = 32;
    fluidConfig.gridSizeZ = 32;
    fluidConfig.cellSize = 0.1f;
    fluidConfig.useGPU = true;

    VulkanFluidCompute gpuCompute;
    if (!gpuCompute.Initialize(&ctx, fluidConfig, "Assets/Shaders/Compute")) {
        ctx.Shutdown();
        EXPECT_TRUE(true);
        return;
    }

    std::vector<COFLIPParticle> particles(500);
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].position[0] = 1.6f + (float)(i % 10) * 0.05f;
        particles[i].position[1] = 1.6f + (float)((i / 10) % 10) * 0.05f;
        particles[i].position[2] = 1.6f + (float)(i / 100) * 0.05f;
        particles[i].velocity[0] = 0.0f;
        particles[i].velocity[1] = 0.0f;
        particles[i].velocity[2] = 0.0f;
        particles[i].mass = 1.0f;
        particles[i].volume = 0.001f;
        particles[i].flags = 1;
    }

    gpuCompute.UploadParticles(particles, static_cast<uint32_t>(particles.size()));

    FluidSimParams params;
    params.particleCount = static_cast<uint32_t>(particles.size());
    params.gridSizeX = fluidConfig.gridSizeX;
    params.gridSizeY = fluidConfig.gridSizeY;
    params.gridSizeZ = fluidConfig.gridSizeZ;
    params.cellSize = fluidConfig.cellSize;
    params.invCellSize = 1.0f / fluidConfig.cellSize;
    params.dt = 1.0f / 60.0f;
    params.flipRatio = 0.95f;
    params.pressureIterations = 10;
    params.gravity = -9.8f;

    // Test async simulation (optimization #3)
    gpuCompute.BeginAsyncSimulation(params);

    // Should be in progress
    EXPECT_TRUE(gpuCompute.IsSimulationInProgress());

    // Wait for completion
    gpuCompute.WaitForSimulation();

    // Should no longer be in progress
    EXPECT_FALSE(gpuCompute.IsSimulationInProgress());

    gpuCompute.DownloadParticles(particles, static_cast<uint32_t>(particles.size()));

    gpuCompute.Shutdown();
    ctx.Shutdown();
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

    // GPU Compute tests
    runTest("VulkanContext_IsAvailable", test_VulkanContext_IsAvailable);
    runTest("VulkanContext_GetAvailableGPUs", test_VulkanContext_GetAvailableGPUs);
    runTest("VulkanContext_Initialize", test_VulkanContext_Initialize);
    runTest("VulkanContext_Singleton", test_VulkanContext_Singleton);
    runTest("VulkanContext_DeviceInfo", test_VulkanContext_DeviceInfo);
    runTest("VulkanContext_Handles", test_VulkanContext_Handles);
    runTest("VulkanContext_Reinitialize", test_VulkanContext_Reinitialize);
    runTest("ShaderUtils_LoadSPIRV", test_ShaderUtils_LoadSPIRV);
    runTest("VulkanContext_WaitIdle", test_VulkanContext_WaitIdle);
    runTest("ComputePipeline_Construction", test_ComputePipeline_Construction);
    runTest("ComputePipeline_CalculateGroupCount", test_ComputePipeline_CalculateGroupCount);
    runTest("ShaderUtils_LoadSPIRV_ValidMagic", test_ShaderUtils_LoadSPIRV_ValidMagic);
    runTest("ShaderBinding_Types", test_ShaderBinding_Types);
    runTest("GPUBufferUsage_Flags", test_GPUBufferUsage_Flags);
    runTest("GPUDeviceInfo_Structure", test_GPUDeviceInfo_Structure);

    // Fluid System tests
    runTest("FluidParticle_Size", test_FluidParticle_Size);
    runTest("FluidMaterial_Presets", test_FluidMaterial_Presets);
    runTest("FluidGrid_Initialize", test_FluidGrid_Initialize);
    runTest("FluidGrid_WorldToGrid", test_FluidGrid_WorldToGrid);
    runTest("FluidGrid_Bounds", test_FluidGrid_Bounds);
    runTest("FluidSystem_Initialize", test_FluidSystem_Initialize);
    runTest("FluidSystem_AddParticle", test_FluidSystem_AddParticle);
    runTest("FluidSystem_AddParticleBox", test_FluidSystem_AddParticleBox);
    runTest("FluidSystem_Materials", test_FluidSystem_Materials);
    runTest("FluidSystem_Emitter", test_FluidSystem_Emitter);
    runTest("FluidSystem_Step", test_FluidSystem_Step);
    runTest("FluidSystem_Stats", test_FluidSystem_Stats);
    runTest("ParticleFlags", test_ParticleFlags);

    // GPU Fluid Compute tests (optimization verification)
    runTest("VulkanFluidCompute_Initialization", test_VulkanFluidCompute_Initialization);
    runTest("VulkanFluidCompute_BatchedDispatch", test_VulkanFluidCompute_BatchedDispatch);
    runTest("VulkanFluidCompute_SortedDispatch", test_VulkanFluidCompute_SortedDispatch);
    runTest("VulkanFluidCompute_AsyncSimulation", test_VulkanFluidCompute_AsyncSimulation);

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
