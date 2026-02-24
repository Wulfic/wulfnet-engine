// =============================================================================
// WulfNet Engine - IFS Example
// =============================================================================
// Demonstrates GPU-accelerated Iterated Function System fractal generation.
// Initializes Vulkan, creates an IFS system with a preset, runs iterations,
// and prints basic stats.
// =============================================================================

#include "WulfNet/WulfNet.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "=== WulfNet IFS Example ===" << std::endl;
    std::cout << "WulfNet Engine v" << WULFNET_VERSION_STRING << std::endl;

    // Initialize logging
    WulfNet::Logger::Initialize();
    WulfNet::Logger::SetMinLevel(WulfNet::LogLevel::Info);

    // Check GPU availability
    if (!WulfNet::IsGPUComputeAvailable()) {
        std::cerr << "No Vulkan-capable GPU found. IFS requires GPU compute." << std::endl;
        return 1;
    }

    auto gpus = WulfNet::GetAvailableGPUs();
    std::cout << "Found " << gpus.size() << " GPU(s):" << std::endl;
    for (const auto& gpu : gpus) {
        std::cout << "  - " << gpu.name << std::endl;
    }

    // Initialize Vulkan context
    if (!WulfNet::InitializeVulkanContext()) {
        std::cerr << "Failed to initialize Vulkan context." << std::endl;
        return 1;
    }
    std::cout << "Vulkan context initialized." << std::endl;

    // Create IFS system
    WulfNet::IFSConfig config;
    config.cubeResolution = 32;
    config.chaosIterationsPerFrame = 8;
    config.voxelGridSize = 64;
    config.initialPreset = WulfNet::IFSPreset::SierpinskiTriangle3D;

    WulfNet::IFSSystem ifs;
    if (!ifs.Initialize(config)) {
        std::cerr << "Failed to initialize IFS system." << std::endl;
        WulfNet::ShutdownVulkanContext();
        return 1;
    }

    uint32_t particleCount = ifs.GetParticleCount();
    uint32_t transformCount = ifs.GetTransformCount();
    std::cout << "IFS initialized: " << particleCount << " particles, "
              << transformCount << " transforms" << std::endl;

    // Run several frames of iteration
    const int numFrames = 60;
    float dt = 1.0f / 60.0f;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < numFrames; ++frame) {
        ifs.Update(dt);

        if (frame == 0 || frame == numFrames - 1) {
            // Download and verify particles
            std::vector<float> positions;
            if (ifs.DownloadParticles(positions)) {
                // Check bounds of first few particles
                float minX = 1e10f, maxX = -1e10f;
                float minY = 1e10f, maxY = -1e10f;
                for (uint32_t i = 0; i < std::min(particleCount, 100u); ++i) {
                    float px = positions[i * 4 + 0];
                    float py = positions[i * 4 + 1];
                    minX = std::min(minX, px);
                    maxX = std::max(maxX, px);
                    minY = std::min(minY, py);
                    maxY = std::max(maxY, py);
                }
                std::cout << "Frame " << frame << " - Particle bounds: "
                          << "X[" << minX << ", " << maxX << "] "
                          << "Y[" << minY << ", " << maxY << "]" << std::endl;
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << numFrames << " frames in " << elapsed << "ms ("
              << (float(elapsed) / numFrames) << "ms/frame)" << std::endl;

    // Test preset switching
    std::cout << "\nSwitching to Vicsek 3D preset..." << std::endl;
    ifs.SetPreset(WulfNet::IFSPreset::Vicsek3D);
    std::cout << "New transform count: " << ifs.GetTransformCount() << std::endl;

    // Run a few more frames
    for (int frame = 0; frame < 10; ++frame) {
        ifs.Update(dt);
    }

    // Test blending
    std::cout << "Blending to Sierpinski Carpet 3D..." << std::endl;
    ifs.BlendToPreset(WulfNet::IFSPreset::SierpinskiCarpet3D);
    for (int frame = 0; frame < 30; ++frame) {
        ifs.Update(dt);
    }

    // Test voxel grid download
    std::vector<int32_t> voxels;
    if (ifs.DownloadVoxelGrid(voxels)) {
        int occupiedCount = 0;
        for (int32_t v : voxels) occupiedCount += (v > 0 ? 1 : 0);
        std::cout << "Voxel grid: " << occupiedCount << " / " << voxels.size()
                  << " occupied (" << (100.0f * occupiedCount / voxels.size()) << "%)" << std::endl;
    }

    // Cleanup
    ifs.Shutdown();
    WulfNet::ShutdownVulkanContext();
    WulfNet::Logger::Shutdown();

    std::cout << "\nIFS Example completed successfully!" << std::endl;
    return 0;
}
