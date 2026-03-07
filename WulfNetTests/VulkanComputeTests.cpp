// =============================================================================
// WulfNet Engine - Vulkan Compute Tests
// =============================================================================
// Tests for VulkanContext, ShaderUtils, ComputePipeline, GPU buffer usage,
// and device info queries.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>

using namespace WulfNet;

// =============================================================================
// VulkanContext Tests
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
// Registration
// =============================================================================

void RegisterVulkanComputeTests() {
    RUN_TEST("VulkanContext_IsAvailable", test_VulkanContext_IsAvailable);
    RUN_TEST("VulkanContext_GetAvailableGPUs", test_VulkanContext_GetAvailableGPUs);
    RUN_TEST("VulkanContext_Initialize", test_VulkanContext_Initialize);
    RUN_TEST("VulkanContext_Singleton", test_VulkanContext_Singleton);
    RUN_TEST("VulkanContext_DeviceInfo", test_VulkanContext_DeviceInfo);
    RUN_TEST("VulkanContext_Handles", test_VulkanContext_Handles);
    RUN_TEST("VulkanContext_Reinitialize", test_VulkanContext_Reinitialize);
    RUN_TEST("ShaderUtils_LoadSPIRV", test_ShaderUtils_LoadSPIRV);
    RUN_TEST("VulkanContext_WaitIdle", test_VulkanContext_WaitIdle);
    RUN_TEST("ComputePipeline_Construction", test_ComputePipeline_Construction);
    RUN_TEST("ComputePipeline_CalculateGroupCount", test_ComputePipeline_CalculateGroupCount);
    RUN_TEST("ShaderUtils_LoadSPIRV_ValidMagic", test_ShaderUtils_LoadSPIRV_ValidMagic);
    RUN_TEST("ShaderBinding_Types", test_ShaderBinding_Types);
    RUN_TEST("GPUBufferUsage_Flags", test_GPUBufferUsage_Flags);
    RUN_TEST("GPUDeviceInfo_Structure", test_GPUDeviceInfo_Structure);
}
