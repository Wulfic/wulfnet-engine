// =============================================================================
// WulfNet Engine - GPU Compute Module
// =============================================================================
// Provides GPU acceleration via Vulkan compute shaders.
//
// Features:
//   - Vulkan compute context (headless, no presentation)
//   - GPU buffer management with automatic staging
//   - Compute pipeline abstraction
//   - SPIR-V shader loading
//
// Usage:
//   #include "WulfNet/Compute/Compute.h"
//
//   // Initialize
//   WulfNet::InitializeVulkanContext();
//
//   // Create buffers
//   WulfNet::FloatBuffer input(1000);
//   WulfNet::FloatBuffer output(1000);
//   input.Upload(data);
//
//   // Create and run compute shader
//   WulfNet::ComputePipeline pipeline;
//   pipeline.CreateFromFile("shader.spv", bindings);
//   pipeline.BindBuffer(0, input);
//   pipeline.BindBuffer(1, output);
//   pipeline.DispatchAndWait(pipeline.CalculateGroupCount(1000));
//
//   // Download results
//   output.Download(results);
//
//   // Cleanup
//   WulfNet::ShutdownVulkanContext();
//
// =============================================================================

#pragma once

// Vulkan context
#include "Vulkan/VulkanContext.h"

// GPU memory management
#include "Memory/ComputeBuffer.h"

// Compute pipelines
#include "Shaders/ComputePipeline.h"

namespace WulfNet {

/// Check if GPU compute is available on this system
inline bool IsGPUComputeAvailable() {
    auto devices = VulkanContext::EnumerateDevices();
    return !devices.empty();
}

/// Get information about all available GPUs
inline std::vector<GPUDeviceInfo> GetAvailableGPUs() {
    return VulkanContext::EnumerateDevices();
}

} // namespace WulfNet
