#pragma once
#include <vector>
#include "../core/math/Vec3.h"
#include "../core/math/Vec4.h"
#include "../core/math/Mat4.h"
#include "../core/math/Quat.h"
#include "../core/Types.h"

namespace WulfNet {
namespace Rendering {

class DebugRenderer {
public:
    static void init();
    static void shutdown();
    
    // Per-frame setup
    static void beginFrame(const Mat4& viewProj);
    static void endFrame();
    
    // Primitives
    static void drawLine(const Vec3& start, const Vec3& end, const Vec3& color);
    static void drawBox(const Vec3& pos, const Vec3& halfExtents, const Quat& rot, const Vec3& color);
    static void drawSphere(const Vec3& pos, f32 radius, const Vec3& color);
    static void drawAxes(const Vec3& pos, const Quat& rot, f32 size);
    
private:
    struct Vertex {
        Vec3 pos;
        Vec3 color;
    };
    
    static std::vector<Vertex> s_lines;
    static u32 s_shaderProgram;
    static u32 s_vbo;
    static u32 s_vao;
    static Mat4 s_viewProj;
};

}
}
