#pragma once
#include "../Types.h"
#include <string>

// Forward decl
struct GLFWwindow;

namespace WulfNet {
namespace Platform {

struct WindowConfig {
    std::string title = "WulfNet Engine";
    u32 width = 1280;
    u32 height = 720;
    bool vsync = true;
};

class Window {
public:
    Window(const WindowConfig& config);
    ~Window();

    bool shouldClose() const;
    void pollEvents();
    void swapBuffers();
    
    // Native handle access
    GLFWwindow* getNativeHandle() const { return m_handle; }
    
    u32 getWidth() const { return m_config.width; }
    u32 getHeight() const { return m_config.height; }
    f32 getAspectRatio() const { return (f32)m_config.width / (f32)m_config.height; }

private:
    WindowConfig m_config;
    GLFWwindow* m_handle = nullptr;
};

} // namespace Platform
} // namespace WulfNet
