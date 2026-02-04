#include "../src/core/platform/Window.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h> // For input
#include "../src/rendering/DebugRenderer.h"
#include "../src/rendering/GLLoader.h" // Ensure we include this to init GL
#include "../src/core/math/Mat4.h"
#include "../src/core/math/Vec3.h"
#include "../src/core/math/MathUtils.h"
#include <cmath>

using namespace WulfNet;
using namespace WulfNet::Platform;
using namespace WulfNet::Rendering;

// Simple GL compatible perspective projection for now
Mat4 perspectiveGL(f32 fovY, f32 aspect, f32 nearPlane, f32 farPlane) {
    f32 tanHalfFov = std::tan(fovY * 0.5f);
    
    Mat4 result = Mat4::zero();
    result.columns[0].x = 1.0f / (aspect * tanHalfFov);
    result.columns[1].y = 1.0f / tanHalfFov;
    result.columns[2].z = -(farPlane + nearPlane) / (farPlane - nearPlane);
    result.columns[2].w = -1.0f;
    result.columns[3].z = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
    return result;
}

int main() {
    // 1. Create Window
    WindowConfig config;
    config.title = "WulfNet Test World";
    config.width = 1280;
    config.height = 720;
    config.vsync = true;
    
    Window window(config);
    
    // 2. Initialize OpenGL Loader
    if (!initGL()) {
        return -1;
    }
    
    // 3. Initialize Debug Renderer
    DebugRenderer::init();
    
    // Set clear color (Dark gray background)
    // accessing glClearColor directly requires GL definition. 
    // Since we included GLLoader.h which includes gl.h/windows.h, we should have it.
    glClearColor(0.1f, 0.15f, 0.2f, 1.0f);
    
    // Camera state
    Vec3 cameraPos(10.0f, 10.0f, 10.0f);
    f32 camYaw = -135.0f * Math::DEG_TO_RAD; // Looking at origin from 10,10,10
    f32 camPitch = -35.0f * Math::DEG_TO_RAD;
    
    // Main Loop
    while (!window.shouldClose()) {
        window.pollEvents();
        
        // Input Handling
        GLFWwindow* nativeWin = window.getNativeHandle();
        float speed = 0.1f;
        if (glfwGetKey(nativeWin, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 3.0f;
        
        // Calculate Forward/Right vectors
        Vec3 forward(
            cos(camPitch) * cos(camYaw),
            sin(camPitch),
            cos(camPitch) * sin(camYaw)
        );
        forward = forward.normalized(); // Assuming normalized
        
        Vec3 right = forward.cross(Vec3::unitY()).normalized();
        Vec3 up = Vec3::unitY(); // Global up for movement
        
        // Movement
        if (glfwGetKey(nativeWin, GLFW_KEY_W) == GLFW_PRESS) cameraPos += forward * speed;
        if (glfwGetKey(nativeWin, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= forward * speed;
        if (glfwGetKey(nativeWin, GLFW_KEY_D) == GLFW_PRESS) cameraPos += right * speed;
        if (glfwGetKey(nativeWin, GLFW_KEY_A) == GLFW_PRESS) cameraPos -= right * speed;
        if (glfwGetKey(nativeWin, GLFW_KEY_E) == GLFW_PRESS) cameraPos += Vec3::unitY() * speed;
        if (glfwGetKey(nativeWin, GLFW_KEY_Q) == GLFW_PRESS) cameraPos -= Vec3::unitY() * speed;
        
        // Rotation (Arrow keys)
        float rotSpeed = 0.03f;
        if (glfwGetKey(nativeWin, GLFW_KEY_RIGHT) == GLFW_PRESS) camYaw += rotSpeed;
        if (glfwGetKey(nativeWin, GLFW_KEY_LEFT) == GLFW_PRESS) camYaw -= rotSpeed;
        if (glfwGetKey(nativeWin, GLFW_KEY_UP) == GLFW_PRESS) camPitch += rotSpeed;
        if (glfwGetKey(nativeWin, GLFW_KEY_DOWN) == GLFW_PRESS) camPitch -= rotSpeed;
        
        // Clamp pitch
        if (camPitch > 1.5f) camPitch = 1.5f;
        if (camPitch < -1.5f) camPitch = -1.5f;
        
        // Re-calculate view target based on new orientation
        Vec3 lookDir(
            cos(camPitch) * cos(camYaw),
            sin(camPitch),
            cos(camPitch) * sin(camYaw)
        );
        Vec3 target = cameraPos + lookDir;
        
        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST); // Ensure depth test is on
        
        // Calculate View Projection
        Mat4 view = Mat4::lookAt(cameraPos, target, Vec3::unitY());
        Mat4 proj = perspectiveGL(Math::radians(60.0f), window.getAspectRatio(), 0.1f, 1000.0f);
        Mat4 viewProj = proj * view;
        
        // Begin Debug Rendering
        DebugRenderer::beginFrame(viewProj);
        
        // Draw Grid
        int gridSize = 50;
        float spacing = 1.0f;
        Vec3 gridColor(0.2f, 0.2f, 0.2f);
        Vec3 axisXColor(0.5f, 0.2f, 0.2f);
        Vec3 axisZColor(0.2f, 0.2f, 0.5f);
        
        for (int i = -gridSize; i <= gridSize; i++) {
            // X lines
            Vec3 color = (i == 0) ? axisXColor : gridColor;
            DebugRenderer::drawLine(
                Vec3((float)i * spacing, 0.0f, (float)-gridSize * spacing),
                Vec3((float)i * spacing, 0.0f, (float)gridSize * spacing),
                color
            );
            
            // Z lines
            color = (i == 0) ? axisZColor : gridColor;
            DebugRenderer::drawLine(
                Vec3((float)-gridSize * spacing, 0.0f, (float)i * spacing),
                Vec3((float)gridSize * spacing, 0.0f, (float)i * spacing),
                color
            );
        }
        
        // Draw Axes
        DebugRenderer::drawAxes(Vec3(0,0,0), Quat::identity(), 2.0f);
        
        // Draw some Test Objects
        DebugRenderer::drawSphere(Vec3(0, 2, 0), 1.0f, Vec3(1, 0, 0)); // Red Sphere
        DebugRenderer::drawBox(Vec3(3, 1, 0), Vec3(1, 1, 1), Quat::fromAxisAngle(Vec3::unitY(), (float)glfwGetTime()), Vec3(0, 1, 0)); // Rotating Green Box
        DebugRenderer::drawSphere(Vec3(-3, 1, 0), 1.0f, Vec3(0, 0, 1)); // Blue Sphere
        
        DebugRenderer::endFrame();
        
        window.swapBuffers();
    }
    
    DebugRenderer::shutdown();
    
    return 0;
}
