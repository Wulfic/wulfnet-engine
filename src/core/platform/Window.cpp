#include "Window.h"
#include "../Log.h"
#include <GLFW/glfw3.h>

namespace WulfNet {
namespace Platform {

static void glfwErrorCallback(int error, const char* description) {
    WULFNET_LOG_ERROR("GLFW", "Error {}: {}", error, description);
}

Window::Window(const WindowConfig& config) : m_config(config) {
    glfwSetErrorCallback(glfwErrorCallback);
    
    if (!glfwInit()) {
        WULFNET_LOG_FATAL("Window", "Failed to initialize GLFW");
        return;
    }
    
    // Request OpenGL 4.5 Core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5); 
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    m_handle = glfwCreateWindow(config.width, config.height, config.title.c_str(), nullptr, nullptr);
    if (!m_handle) {
        WULFNET_LOG_FATAL("Window", "Failed to create window. OpenGL 4.5 required.");
        glfwTerminate();
        return;
    }
    
    glfwMakeContextCurrent(m_handle);
    glfwSwapInterval(config.vsync ? 1 : 0);
    
    // Set user pointer for callbacks if needed later
    glfwSetWindowUserPointer(m_handle, this);
    
    WULFNET_LOG_INFO("Window", "Created window '%s' (%dx%d)", config.title.c_str(), config.width, config.height);
}

Window::~Window() {
    if (m_handle) {
        glfwDestroyWindow(m_handle);
    }
    glfwTerminate();
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(m_handle);
}

void Window::pollEvents() {
    glfwPollEvents();
}

void Window::swapBuffers() {
    glfwSwapBuffers(m_handle);
}

} // namespace Platform
} // namespace WulfNet
