#include "GLLoader.h"
#include <GLFW/glfw3.h>
#include "../core/Log.h"
#include <type_traits>

namespace WulfNet {
namespace Rendering {

// Define pointers
PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
PFNGLBUFFERDATAPROC glBufferData = nullptr;
PFNGLGENVERTEXARRAYSPROC glGenVertexArrays = nullptr;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray = nullptr;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = nullptr;
PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer = nullptr;
PFNGLDRAWARRAYSPROC glDrawArrays = nullptr;
PFNGLCREATESHADERPROC glCreateShader = nullptr;
PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
PFNGLATTACHSHADERPROC glAttachShader = nullptr;
PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = nullptr;
PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv = nullptr;

bool initGL() {
    bool success = true;
    
    auto load = [&](auto& func, const char* name) {
        using FuncType = std::remove_reference_t<decltype(func)>;
        func = reinterpret_cast<FuncType>(glfwGetProcAddress(name));
        if (!func) {
            WULFNET_LOG_ERROR("GLLoader", "Failed to load GL function: {}", name);
            success = false;
        }
    };
    
    // Load functions
    load(glGenBuffers, "glGenBuffers");
    load(glBindBuffer, "glBindBuffer");
    load(glBufferData, "glBufferData");
    load(glGenVertexArrays, "glGenVertexArrays");
    load(glBindVertexArray, "glBindVertexArray");
    load(glEnableVertexAttribArray, "glEnableVertexAttribArray");
    load(glVertexAttribPointer, "glVertexAttribPointer");
    load(glDrawArrays, "glDrawArrays");
    
    load(glCreateShader, "glCreateShader");
    load(glShaderSource, "glShaderSource");
    load(glCompileShader, "glCompileShader");
    load(glGetShaderiv, "glGetShaderiv");
    load(glGetShaderInfoLog, "glGetShaderInfoLog");
    
    load(glCreateProgram, "glCreateProgram");
    load(glAttachShader, "glAttachShader");
    load(glLinkProgram, "glLinkProgram");
    load(glGetProgramiv, "glGetProgramiv");
    load(glGetProgramInfoLog, "glGetProgramInfoLog");
    load(glUseProgram, "glUseProgram");
    load(glGetUniformLocation, "glGetUniformLocation");
    load(glUniformMatrix4fv, "glUniformMatrix4fv");

    return success;
}

}
}
