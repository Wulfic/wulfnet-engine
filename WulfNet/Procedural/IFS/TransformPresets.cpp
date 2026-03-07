// =============================================================================
// WulfNet Engine - IFS Transform Presets Implementation
// =============================================================================

#include "WulfNet/Procedural/IFS/TransformPresets.h"

namespace WulfNet {
namespace TransformPresets {

static std::vector<TransformInstructions> MakeUniformScalePreset(
    const Vec3& scale, const Vec3* translations, size_t count) {
    std::vector<TransformInstructions> instructions;
    instructions.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        TransformInstructions t;
        t.scale = scale;
        t.translate = translations[i];
        instructions.push_back(t);
    }
    return instructions;
}

static std::vector<TransformInstructions> SierpinskiTriangle2D() {
    Vec3 translations[] = {
        {-0.5f, -0.5f, 0.0f},
        { 0.0f,  0.36f, 0.0f},
        { 0.5f, -0.5f, 0.0f}
    };
    return MakeUniformScalePreset({0.5f, 0.5f, 0.5f}, translations, 3);
}

static std::vector<TransformInstructions> Vicsek2D() {
    Vec3 translations[] = {
        {-0.5f, -0.5f, 0.0f},
        {-0.5f,  0.5f, 0.0f},
        { 0.5f,  0.5f, 0.0f},
        { 0.5f, -0.5f, 0.0f},
        { 0.0f,  0.0f, 0.0f}
    };
    return MakeUniformScalePreset({0.33f, 0.33f, 0.33f}, translations, 5);
}

static std::vector<TransformInstructions> SierpinskiCarpet2D() {
    Vec3 translations[] = {
        {-0.5f, -0.5f, 0.0f}, {-0.5f,  0.5f, 0.0f},
        { 0.5f,  0.5f, 0.0f}, { 0.5f, -0.5f, 0.0f},
        {-0.5f,  0.0f, 0.0f}, { 0.5f,  0.0f, 0.0f},
        { 0.0f,  0.5f, 0.0f}, { 0.0f, -0.5f, 0.0f}
    };
    return MakeUniformScalePreset({0.33f, 0.33f, 0.33f}, translations, 8);
}

static std::vector<TransformInstructions> SierpinskiTriangle3D() {
    Vec3 translations[] = {
        {-0.5f, -0.5f,  0.5f}, {-0.5f, -0.5f, -0.5f},
        { 0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f, -0.5f},
        { 0.0f,  0.36f, 0.0f}
    };
    return MakeUniformScalePreset({0.5f, 0.5f, 0.5f}, translations, 5);
}

static std::vector<TransformInstructions> Vicsek3D() {
    Vec3 translations[] = {
        {-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f,  0.5f},
        { 0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f,  0.5f},
        {-0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f,  0.5f},
        { 0.5f,  0.5f, -0.5f}, { 0.5f,  0.5f,  0.5f},
        { 0.0f,  0.0f,  0.0f}
    };
    return MakeUniformScalePreset({0.33f, 0.33f, 0.33f}, translations, 9);
}

static std::vector<TransformInstructions> SierpinskiCarpet3D() {
    Vec3 translations[] = {
        {-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f,  0.5f},
        { 0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f,  0.5f},
        {-0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f,  0.5f},
        { 0.5f,  0.5f, -0.5f}, { 0.5f,  0.5f,  0.5f},
        {-0.5f,  0.5f,  0.0f}, { 0.5f,  0.5f,  0.0f},
        {-0.5f, -0.5f,  0.0f}, { 0.5f, -0.5f,  0.0f},
        { 0.0f,  0.5f, -0.5f}, { 0.0f,  0.5f,  0.5f},
        { 0.0f, -0.5f, -0.5f}, { 0.0f, -0.5f,  0.5f},
        {-0.5f,  0.0f, -0.5f}, { 0.5f,  0.0f,  0.5f},
        { 0.5f,  0.0f, -0.5f}, {-0.5f,  0.0f,  0.5f}
    };
    return MakeUniformScalePreset({0.33f, 0.33f, 0.33f}, translations, 20);
}

std::vector<TransformInstructions> GetPreset(IFSPreset preset) {
    switch (preset) {
        case IFSPreset::SierpinskiTriangle2D: return SierpinskiTriangle2D();
        case IFSPreset::Vicsek2D:             return Vicsek2D();
        case IFSPreset::SierpinskiCarpet2D:   return SierpinskiCarpet2D();
        case IFSPreset::SierpinskiTriangle3D: return SierpinskiTriangle3D();
        case IFSPreset::Vicsek3D:             return Vicsek3D();
        case IFSPreset::SierpinskiCarpet3D:   return SierpinskiCarpet3D();
        default:                              return SierpinskiTriangle2D();
    }
}

std::vector<TransformInstructions> GenerateProcedural(const ProceduralConfig& config,
                                                       std::mt19937& rng) {
    std::vector<TransformInstructions> instructions;
    instructions.reserve(config.count);

    auto randRange = [&](float mn, float mx) -> float {
        std::uniform_real_distribution<float> dist(mn, mx);
        return dist(rng);
    };

    auto randVec = [&](const Vec3& mn, const Vec3& mx) -> Vec3 {
        return {randRange(mn.x, mx.x), randRange(mn.y, mx.y), randRange(mn.z, mx.z)};
    };

    for (int i = 0; i < config.count; ++i) {
        TransformInstructions t;
        t.scale = randVec(config.scaleMin, config.scaleMax);
        t.shearX = randVec(config.shearMin, config.shearMax);
        t.shearY = randVec(config.shearMin, config.shearMax);
        t.shearZ = randVec(config.shearMin, config.shearMax);
        t.rotate = randVec(config.rotateMin, config.rotateMax);
        t.translate = randVec(config.translateMin, config.translateMax);
        instructions.push_back(t);
    }

    return instructions;
}

std::vector<Mat4> BuildMatrices(const std::vector<TransformInstructions>& instructions) {
    std::vector<Mat4> matrices;
    matrices.reserve(instructions.size());
    for (const auto& inst : instructions) {
        matrices.push_back(AffineTransform::FromInstructions(inst));
    }
    return matrices;
}

} // namespace TransformPresets
} // namespace WulfNet
