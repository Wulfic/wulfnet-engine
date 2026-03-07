// =============================================================================
// WulfNet Engine - Unit Tests Entry Point
// =============================================================================
// Runs all WulfNet core unit test suites.
// Test implementations are in separate files for maintainability:
//   - CoreTests.cpp              (Logger, Profiler)
//   - PhysicsWorldTests.cpp      (Physics world lifecycle, bodies, contacts)
//   - VulkanComputeTests.cpp     (Vulkan context, shaders, pipelines)
//   - IFSTransformTests.cpp      (Affine transforms, presets, blender, math)
//   - SoftwareRendererTests.cpp  (GBuffer, rasterizer, deferred shading, occlusion)
//   - PipelineIntegrationTests.cpp (IFS chaos game, full pipeline integration)
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>

using namespace WulfNet;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    // Suppress logging output during tests
    Logger::Get().SetMinLevel(LogLevel::Error);

    std::cout << "=== WulfNet Engine Unit Tests ===" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // Core Tests (Logger, Profiler)
    // =========================================================================
    std::cout << "--- Core Tests ---" << std::endl;
    RegisterCoreTests();
    std::cout << std::endl;

    // =========================================================================
    // Physics World Tests
    // =========================================================================
    std::cout << "--- Physics World Tests ---" << std::endl;
    RegisterPhysicsWorldTests();
    std::cout << std::endl;

    // =========================================================================
    // GPU Compute Tests
    // =========================================================================
    std::cout << "--- GPU Compute Tests ---" << std::endl;
    RegisterVulkanComputeTests();
    std::cout << std::endl;

    // =========================================================================
    // IFS Transform Tests
    // =========================================================================
    std::cout << "--- IFS Transform Tests ---" << std::endl;
    RegisterIFSTransformTests();
    std::cout << std::endl;

    // =========================================================================
    // Software Renderer Tests
    // =========================================================================
    std::cout << "--- Software Renderer Tests ---" << std::endl;
    RegisterSoftwareRendererTests();
    std::cout << std::endl;

    // =========================================================================
    // Pipeline Integration Tests
    // =========================================================================
    std::cout << "--- Pipeline Integration Tests ---" << std::endl;
    RegisterPipelineIntegrationTests();
    std::cout << std::endl;

    return PrintTestReport();
}
