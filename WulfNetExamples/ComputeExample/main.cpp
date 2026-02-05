// =============================================================================
// WulfNet Engine - GPU Compute Example
// =============================================================================
// Demonstrates GPU compute with vector addition
// =============================================================================

#include <WulfNet/WulfNet.h>

#include <iostream>
#include <vector>
#include <cmath>

using namespace WulfNet;

int main() {
    std::cout << "=== WulfNet GPU Compute Example ===" << std::endl;
    std::cout << std::endl;

    // Add console logging
    Logger::Get().AddSink(std::make_shared<ConsoleLogSink>(true));
    Logger::Get().SetMinLevel(LogLevel::Debug);

    // Check if GPU compute is available
    std::cout << "Checking GPU compute availability..." << std::endl;
    if (!IsGPUComputeAvailable()) {
        std::cout << "ERROR: GPU compute not available (Vulkan not found)" << std::endl;
        return 1;
    }
    std::cout << "GPU compute is available!" << std::endl;
    std::cout << std::endl;

    // List available GPUs
    std::cout << "Available GPUs:" << std::endl;
    auto gpus = GetAvailableGPUs();
    for (size_t i = 0; i < gpus.size(); i++) {
        std::cout << "  [" << i << "] " << gpus[i].name << std::endl;
        std::cout << "       Memory: " << (gpus[i].totalMemory / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "       Max Workgroup: "
                  << gpus[i].maxComputeWorkGroupSize[0] << "x"
                  << gpus[i].maxComputeWorkGroupSize[1] << "x"
                  << gpus[i].maxComputeWorkGroupSize[2] << std::endl;
    }
    std::cout << std::endl;

    // Initialize Vulkan context
    std::cout << "Initializing Vulkan context..." << std::endl;
    VulkanContextSettings settings;
    settings.applicationName = "ComputeExample";
    settings.enableValidation = true;  // Enable for debugging

    VulkanContext& ctx = GetVulkanContext();
    if (!ctx.Initialize(settings)) {
        std::cout << "ERROR: Failed to initialize Vulkan context" << std::endl;
        return 1;
    }

    std::cout << "Using GPU: " << ctx.GetDeviceInfo().name << std::endl;
    std::cout << std::endl;

    // TODO: Load shader and create compute pipeline
    // This requires the ComputeBuffer and ComputePipeline to be fully implemented
    // with proper Vulkan function loading.

    std::cout << "GPU Compute infrastructure initialized successfully!" << std::endl;
    std::cout << std::endl;

    // For now, just demonstrate that initialization works
    std::cout << "Context info:" << std::endl;
    std::cout << "  VkInstance: " << (ctx.GetInstance() ? "valid" : "null") << std::endl;
    std::cout << "  VkDevice: " << (ctx.GetDevice() ? "valid" : "null") << std::endl;
    std::cout << "  VkQueue (compute): " << (ctx.GetComputeQueue() ? "valid" : "null") << std::endl;
    std::cout << "  VkCommandPool: " << (ctx.GetComputeCommandPool() ? "valid" : "null") << std::endl;
    std::cout << "  VkDescriptorPool: " << (ctx.GetDescriptorPool() ? "valid" : "null") << std::endl;
    std::cout << std::endl;

    // Cleanup
    ctx.Shutdown();
    std::cout << "Vulkan context shutdown complete." << std::endl;

    return 0;
}
