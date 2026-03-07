// =============================================================================
// WulfNet Engine - Software Rasterizer Example
// =============================================================================
// Demonstrates the CPU software rasterizer with deferred shading.
// Creates test meshes, renders them, applies lighting, and outputs stats.
// =============================================================================

#include "WulfNet/WulfNet.h"
#include <iostream>
#include <chrono>
#include <fstream>

// Write a simple PPM image for verification
static void WritePPM(const char* filename, const uint32_t* pixels, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to write " << filename << std::endl;
        return;
    }

    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        uint32_t c = pixels[i];
        uint8_t rgb[3] = {
            static_cast<uint8_t>(c & 0xFF),
            static_cast<uint8_t>((c >> 8) & 0xFF),
            static_cast<uint8_t>((c >> 16) & 0xFF)
        };
        file.write(reinterpret_cast<const char*>(rgb), 3);
    }

    std::cout << "Wrote " << filename << " (" << width << "x" << height << ")" << std::endl;
}

int main() {
    std::cout << "=== WulfNet Software Rasterizer Example ===" << std::endl;
    std::cout << "WulfNet Engine v" << WULFNET_VERSION_STRING << std::endl;

    // Initialize logging
    WulfNet::Logger::Initialize();
    WulfNet::Logger::SetMinLevel(WulfNet::LogLevel::Info);

    // Create rasterizer
    WulfNet::SoftRasterizerConfig config;
    config.width = 1280;
    config.height = 720;
    config.threadCount = 0;  // Auto-detect

    WulfNet::SoftwareRasterizer rasterizer;
    if (!rasterizer.Initialize(config)) {
        std::cerr << "Failed to initialize rasterizer." << std::endl;
        return 1;
    }
    std::cout << "Rasterizer initialized at " << config.width << "x" << config.height << std::endl;

    // Add test meshes
    auto cube = WulfNet::SoftMeshGen::CreateCube(1.0f);
    cube.material.color = {200, 150, 100, 255};
    int cubeIdx = rasterizer.AddMesh(cube);

    auto sphere = WulfNet::SoftMeshGen::CreateSphere(0.5f, 16, 16);
    sphere.material.color = {100, 150, 200, 255};
    int sphereIdx = rasterizer.AddMesh(sphere);

    std::cout << "Added cube (idx=" << cubeIdx << ", " << cube.vertices.size() << " verts) and "
              << "sphere (idx=" << sphereIdx << ", " << sphere.vertices.size() << " verts)" << std::endl;

    // Create scene objects
    WulfNet::SoftTransform objects[5];

    // Floor (scaled cube)
    objects[0].meshIndex = cubeIdx;
    objects[0].position = {0.0f, -1.5f, 5.0f};
    objects[0].scale = {10.0f, 0.2f, 10.0f};
    objects[0].tint = {150, 150, 150, 255};

    // Center cube
    objects[1].meshIndex = cubeIdx;
    objects[1].position = {0.0f, 0.0f, 5.0f};
    objects[1].rotation = {25.0f, 45.0f, 0.0f};

    // Left sphere
    objects[2].meshIndex = sphereIdx;
    objects[2].position = {-2.0f, 0.0f, 5.0f};

    // Right sphere
    objects[3].meshIndex = sphereIdx;
    objects[3].position = {2.0f, 0.0f, 5.0f};
    objects[3].tint = {255, 200, 100, 255};

    // Background cube
    objects[4].meshIndex = cubeIdx;
    objects[4].position = {0.0f, 0.5f, 8.0f};
    objects[4].scale = {2.0f, 2.0f, 0.5f};
    objects[4].tint = {100, 200, 150, 255};

    // Setup camera
    WulfNet::SoftCamera camera;
    camera.position = {0.0f, 1.0f, 0.0f};
    camera.forward = {0.0f, -0.1f, 1.0f};
    camera.forward = camera.forward.Normalized();
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.right = camera.forward.Cross(camera.up).Normalized();
    camera.up = camera.right.Cross(camera.forward).Normalized();
    camera.fov = 60.0f;
    camera.aspectRatio = static_cast<float>(config.width) / config.height;

    // Render
    std::cout << "\nRendering geometry pass..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    rasterizer.Clear();
    rasterizer.RenderObjects(objects, 5, camera);

    auto geoTime = std::chrono::high_resolution_clock::now();
    auto geoMs = std::chrono::duration_cast<std::chrono::microseconds>(geoTime - startTime).count();
    std::cout << "Geometry pass: " << (geoMs / 1000.0f) << "ms" << std::endl;

    // Save pre-lighting image
    WritePPM("raster_geometry.ppm", rasterizer.GetColorBuffer(), config.width, config.height);

    // Apply deferred shading
    std::cout << "Applying deferred shading..." << std::endl;
    WulfNet::DeferredShadingConfig shadingConfig;
    shadingConfig.sunLight.direction = {-0.5f, -1.0f, 0.5f};
    shadingConfig.sunLight.intensity = 0.8f;

    shadingConfig.pointLights.push_back({
        {-2.0f, 2.0f, 4.0f},   // position
        {1.0f, 0.8f, 0.5f},    // color
        2.0f,                    // intensity
        8.0f                     // range
    });
    shadingConfig.pointLights.push_back({
        {2.0f, 1.0f, 6.0f},
        {0.5f, 0.5f, 1.0f},
        1.5f,
        6.0f
    });

    WulfNet::DeferredShading deferred;
    deferred.Apply(rasterizer.GetGBuffer(), shadingConfig, camera);

    auto shadingTime = std::chrono::high_resolution_clock::now();
    auto shadingMs = std::chrono::duration_cast<std::chrono::microseconds>(shadingTime - geoTime).count();
    std::cout << "Deferred shading: " << (shadingMs / 1000.0f) << "ms" << std::endl;

    // Save final image
    WritePPM("raster_final.ppm", rasterizer.GetColorBuffer(), config.width, config.height);

    // Test occlusion culler
    std::cout << "\nTesting occlusion culler..." << std::endl;
    WulfNet::OcclusionCuller culler;
    culler.Initialize();
    culler.AddMesh(cube);

    // Render a wall as occluder
    WulfNet::SoftTransform wall;
    wall.meshIndex = 0;
    wall.position = {0.0f, 0.0f, 4.0f};
    wall.scale = {5.0f, 3.0f, 0.2f};

    culler.RenderOccluders(&wall, 1, camera);

    // Test visibility
    WulfNet::AABox behindWall = {{-1.0f, -1.0f, 5.0f}, {1.0f, 1.0f, 6.0f}};
    WulfNet::AABox inFrontOfWall = {{-1.0f, -1.0f, 2.0f}, {1.0f, 1.0f, 3.0f}};

    bool behindVisible = culler.IsVisible(behindWall, camera);
    bool frontVisible = culler.IsVisible(inFrontOfWall, camera);

    std::cout << "Object behind wall: " << (behindVisible ? "VISIBLE" : "OCCLUDED") << std::endl;
    std::cout << "Object in front: " << (frontVisible ? "VISIBLE" : "OCCLUDED") << std::endl;

    auto totalTime = std::chrono::high_resolution_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::microseconds>(totalTime - startTime).count();
    std::cout << "\nTotal time: " << (totalMs / 1000.0f) << "ms" << std::endl;

    // Cleanup
    rasterizer.Shutdown();
    WulfNet::Logger::Get().Flush();

    std::cout << "\nSoftware Rasterizer Example completed successfully!" << std::endl;
    return 0;
}
