// =============================================================================
// WulfNet Engine - IFS Transform Presets
// =============================================================================
// Predefined fractal attractor presets.
// Ported from AttractorPresets.cs / TransformSet.cs (Unity IFS reference).
// =============================================================================

#pragma once

#include "WulfNet/Procedural/IFS/AffineTransform.h"
#include <vector>
#include <random>

namespace WulfNet {

enum class IFSPreset {
    SierpinskiTriangle2D,
    Vicsek2D,
    SierpinskiCarpet2D,
    SierpinskiTriangle3D,
    Vicsek3D,
    SierpinskiCarpet3D,
    Procedural
};

struct ProceduralConfig {
    Vec3 scaleMin = {0.2f, 0.2f, 0.2f};
    Vec3 scaleMax = {0.6f, 0.6f, 0.6f};
    Vec3 shearMin = {-0.1f, -0.1f, -0.1f};
    Vec3 shearMax = {0.1f, 0.1f, 0.1f};
    Vec3 rotateMin = {0.0f, 0.0f, 0.0f};
    Vec3 rotateMax = {360.0f, 360.0f, 360.0f};
    Vec3 translateMin = {-0.5f, -0.5f, -0.5f};
    Vec3 translateMax = {0.5f, 0.5f, 0.5f};
    int count = 8;
};

namespace TransformPresets {

/// Get transform instructions for a given preset
std::vector<TransformInstructions> GetPreset(IFSPreset preset);

/// Generate random procedural transforms
std::vector<TransformInstructions> GenerateProcedural(const ProceduralConfig& config,
                                                       std::mt19937& rng);

/// Build GPU-ready affine matrices from transform instructions
std::vector<Mat4> BuildMatrices(const std::vector<TransformInstructions>& instructions);

} // namespace TransformPresets

} // namespace WulfNet
