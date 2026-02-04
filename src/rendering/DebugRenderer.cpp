#include "DebugRenderer.h"
#include "GLLoader.h"
#include "../core/Log.h"

namespace WulfNet {
namespace Rendering {

std::vector<DebugRenderer::Vertex> DebugRenderer::s_lines;
u32 DebugRenderer::s_shaderProgram = 0;
u32 DebugRenderer::s_vbo = 0;
u32 DebugRenderer::s_vao = 0;
Mat4 DebugRenderer::s_viewProj = Mat4::identity();

static const char* VERTEX_SHADER_SRC = R"(
#version 450 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 vColor;
uniform mat4 uViewProj;

void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    vColor = aColor;
}
)";

static const char* FRAGMENT_SHADER_SRC = R"(
#version 450 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
)";

static u32 compileShader(u32 type, const char* src) {
    u32 shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        WULFNET_LOG_ERROR("DebugRenderer", "Shader Compilation Failed: {}", infoLog);
    }
    return shader;
}

void DebugRenderer::init() {
    initGL(); // Ensure pointers loaded
    
    u32 vs = compileShader(GL_VERTEX_SHADER, VERTEX_SHADER_SRC);
    u32 fs = compileShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);
    
    s_shaderProgram = glCreateProgram();
    glAttachShader(s_shaderProgram, vs);
    glAttachShader(s_shaderProgram, fs);
    glLinkProgram(s_shaderProgram);
    
    // Check link errors
    GLint success;
    glGetProgramiv(s_shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
         char infoLog[512];
         glGetProgramInfoLog(s_shaderProgram, 512, nullptr, infoLog);
         WULFNET_LOG_ERROR("DebugRenderer", "Program Linking Failed: {}", infoLog);
    }
    
    glGenVertexArrays(1, &s_vao);
    glGenBuffers(1, &s_vbo);
    
    glBindVertexArray(s_vao);
    glBindBuffer(GL_ARRAY_BUFFER, s_vbo);
    
    // Pos
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
    // Color
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, color));
    
    glBindVertexArray(0);
}

void DebugRenderer::shutdown() {
    // cleanup
}

void DebugRenderer::beginFrame(const Mat4& viewProj) {
    s_lines.clear();
    s_viewProj = viewProj;
}

void DebugRenderer::endFrame() {
    if (s_lines.empty()) return;
    
    glUseProgram(s_shaderProgram);
    
    GLint loc = glGetUniformLocation(s_shaderProgram, "uViewProj");
    glUniformMatrix4fv(loc, 1, GL_FALSE, s_viewProj.data);
    
    glBindVertexArray(s_vao);
    glBindBuffer(GL_ARRAY_BUFFER, s_vbo);
    glBufferData(GL_ARRAY_BUFFER, s_lines.size() * sizeof(Vertex), s_lines.data(), GL_DYNAMIC_DRAW);
    
    glDrawArrays(GL_LINES, 0, (GLsizei)s_lines.size());
    
    glBindVertexArray(0);
}

void DebugRenderer::drawLine(const Vec3& start, const Vec3& end, const Vec3& color) {
    s_lines.push_back({start, color});
    s_lines.push_back({end, color});
}

void DebugRenderer::drawBox(const Vec3& pos, const Vec3& halfExtents, const Quat& rot, const Vec3& color) {
    Vec3 corners[8];
    Vec3 axisX = rot * Vec3(1,0,0);
    Vec3 axisY = rot * Vec3(0,1,0);
    Vec3 axisZ = rot * Vec3(0,0,1);
    
    Vec3 x = axisX * halfExtents.x;
    Vec3 y = axisY * halfExtents.y;
    Vec3 z = axisZ * halfExtents.z;
    
    corners[0] = pos - x - y - z;
    corners[1] = pos + x - y - z;
    corners[2] = pos + x + y - z;
    corners[3] = pos - x + y - z;
    corners[4] = pos - x - y + z;
    corners[5] = pos + x - y + z;
    corners[6] = pos + x + y + z;
    corners[7] = pos - x + y + z;
    
    // Bottom
    drawLine(corners[0], corners[1], color);
    drawLine(corners[1], corners[2], color);
    drawLine(corners[2], corners[3], color);
    drawLine(corners[3], corners[0], color);
    
    // Top
    drawLine(corners[4], corners[5], color);
    drawLine(corners[5], corners[6], color);
    drawLine(corners[6], corners[7], color);
    drawLine(corners[7], corners[4], color);
    
    // Sides
    drawLine(corners[0], corners[4], color);
    drawLine(corners[1], corners[5], color);
    drawLine(corners[2], corners[6], color);
    drawLine(corners[3], corners[7], color);
}

void DebugRenderer::drawAxes(const Vec3& pos, const Quat& rot, f32 size) {
    Vec3 x = rot * Vec3(1,0,0) * size;
    Vec3 y = rot * Vec3(0,1,0) * size;
    Vec3 z = rot * Vec3(0,0,1) * size;
    
    drawLine(pos, pos + x, Vec3(1,0,0));
    drawLine(pos, pos + y, Vec3(0,1,0));
    drawLine(pos, pos + z, Vec3(0,0,1));
}

void DebugRenderer::drawSphere(const Vec3& pos, f32 radius, const Vec3& color) {
    const int segments = 16;
    const float step = Math::TWO_PI / segments;
    
    // XY plane
    for (int i = 0; i < segments; ++i) {
        float angle1 = i * step;
        float angle2 = (i + 1) * step;
        
        Vec3 p1(std::cos(angle1) * radius, std::sin(angle1) * radius, 0.0f);
        Vec3 p2(std::cos(angle2) * radius, std::sin(angle2) * radius, 0.0f);
        
        drawLine(pos + p1, pos + p2, color);
    }
    
    // YZ plane
    for (int i = 0; i < segments; ++i) {
        float angle1 = i * step;
        float angle2 = (i + 1) * step;
        
        Vec3 p1(0.0f, std::cos(angle1) * radius, std::sin(angle1) * radius);
        Vec3 p2(0.0f, std::cos(angle2) * radius, std::sin(angle2) * radius);
        
        drawLine(pos + p1, pos + p2, color);
    }
    
    // XZ plane
    for (int i = 0; i < segments; ++i) {
        float angle1 = i * step;
        float angle2 = (i + 1) * step;
        
        Vec3 p1(std::cos(angle1) * radius, 0.0f, std::sin(angle1) * radius);
        Vec3 p2(std::cos(angle2) * radius, 0.0f, std::sin(angle2) * radius);
        
        drawLine(pos + p1, pos + p2, color);
    }
}

}
}
