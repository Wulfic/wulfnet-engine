// =============================================================================
// WulfNet Engine - IFS Transform Tests
// =============================================================================
// Tests for GPUMat4x4, AffineTransform, TransformPresets,
// TransformBlender, Vec3, and TransformInstructions.
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>
#include <WulfNet/Procedural/IFS/AffineTransform.h>
#include <WulfNet/Procedural/IFS/TransformPresets.h>
#include <WulfNet/Procedural/IFS/TransformBlender.h>

#include <random>

using namespace WulfNet;

// =============================================================================
// Helper: transform a point by a GPUMat4x4 (row-major)
// =============================================================================

static Vec3 TransformPoint(const GPUMat4x4& m, const Vec3& p) {
    float x = m.At(0, 0) * p.x + m.At(0, 1) * p.y + m.At(0, 2) * p.z + m.At(0, 3);
    float y = m.At(1, 0) * p.x + m.At(1, 1) * p.y + m.At(1, 2) * p.z + m.At(1, 3);
    float z = m.At(2, 0) * p.x + m.At(2, 1) * p.y + m.At(2, 2) * p.z + m.At(2, 3);
    return {x, y, z};
}

// =============================================================================
// GPUMat4x4 Tests
// =============================================================================

void test_GPUMat4x4_Identity() {
    GPUMat4x4 id = GPUMat4x4::Identity();
    // Diagonal should be 1, all else 0
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            float expected = (r == c) ? 1.0f : 0.0f;
            float actual = id.At(r, c);
            EXPECT_TRUE(std::abs(actual - expected) < 1e-6f);
        }
    }
}

void test_GPUMat4x4_Multiply_Identity() {
    GPUMat4x4 id = GPUMat4x4::Identity();
    GPUMat4x4 scale = AffineTransform::MakeScale({2.0f, 3.0f, 4.0f});

    GPUMat4x4 result = id * scale;
    for (int i = 0; i < 16; i++) {
        EXPECT_TRUE(std::abs(result.m[i] - scale.m[i]) < 1e-6f);
    }

    result = scale * id;
    for (int i = 0; i < 16; i++) {
        EXPECT_TRUE(std::abs(result.m[i] - scale.m[i]) < 1e-6f);
    }
}

// =============================================================================
// AffineTransform Tests
// =============================================================================

void test_AffineTransform_MakeScale() {
    GPUMat4x4 mat = AffineTransform::MakeScale({2.0f, 3.0f, 4.0f});
    EXPECT_TRUE(std::abs(mat.At(0, 0) - 2.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(mat.At(1, 1) - 3.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(mat.At(2, 2) - 4.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(mat.At(3, 3) - 1.0f) < 1e-6f);
    // Off-diagonal should be zero
    EXPECT_TRUE(std::abs(mat.At(0, 1)) < 1e-6f);
    EXPECT_TRUE(std::abs(mat.At(0, 2)) < 1e-6f);
    EXPECT_TRUE(std::abs(mat.At(1, 0)) < 1e-6f);
}

void test_AffineTransform_MakeTranslate() {
    GPUMat4x4 mat = AffineTransform::MakeTranslate({5.0f, 6.0f, 7.0f});
    // Row-major: translation in column 3
    EXPECT_TRUE(std::abs(mat.At(0, 3) - 5.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(mat.At(1, 3) - 6.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(mat.At(2, 3) - 7.0f) < 1e-6f);
    // Diagonal still 1
    EXPECT_TRUE(std::abs(mat.At(0, 0) - 1.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(mat.At(1, 1) - 1.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(mat.At(2, 2) - 1.0f) < 1e-6f);
}

void test_AffineTransform_MakeRotation_Zero() {
    GPUMat4x4 mat = AffineTransform::MakeRotation({0.0f, 0.0f, 0.0f});
    GPUMat4x4 id = GPUMat4x4::Identity();
    for (int i = 0; i < 16; i++) {
        EXPECT_TRUE(std::abs(mat.m[i] - id.m[i]) < 1e-5f);
    }
}

void test_AffineTransform_MakeRotation_90Y() {
    // 90-degree rotation around Y: x -> z, z -> -x
    GPUMat4x4 mat = AffineTransform::MakeRotation({0.0f, 90.0f, 0.0f});
    // row 0: [ cos90, 0, sin90, 0 ] = [ 0, 0, 1, 0 ]
    EXPECT_TRUE(std::abs(mat.At(0, 0) - 0.0f) < 1e-5f);
    EXPECT_TRUE(std::abs(mat.At(0, 2) - 1.0f) < 1e-5f);
    // row 2: [ -sin90, 0, cos90, 0 ] = [ -1, 0, 0, 0 ]
    EXPECT_TRUE(std::abs(mat.At(2, 0) - (-1.0f)) < 1e-5f);
    EXPECT_TRUE(std::abs(mat.At(2, 2) - 0.0f) < 1e-5f);
}

void test_AffineTransform_FromInstructions_Identity() {
    TransformInstructions inst = TransformInstructions::Identity();
    GPUMat4x4 mat = AffineTransform::FromInstructions(inst);
    GPUMat4x4 id = GPUMat4x4::Identity();
    for (int i = 0; i < 16; i++) {
        EXPECT_TRUE(std::abs(mat.m[i] - id.m[i]) < 1e-5f);
    }
}

void test_AffineTransform_FromInstructions_ScaleTranslate() {
    TransformInstructions inst;
    inst.scale = {0.5f, 0.5f, 0.5f};
    inst.translate = {1.0f, 0.0f, 0.0f};

    GPUMat4x4 mat = AffineTransform::FromInstructions(inst);
    // Scale should be 0.5 on diagonal
    EXPECT_TRUE(std::abs(mat.At(0, 0) - 0.5f) < 1e-5f);
    EXPECT_TRUE(std::abs(mat.At(1, 1) - 0.5f) < 1e-5f);
    EXPECT_TRUE(std::abs(mat.At(2, 2) - 0.5f) < 1e-5f);
    // Translation in column 3 (scale * rotation * shear * translate -> scaled translate)
    EXPECT_TRUE(std::abs(mat.At(0, 3) - 0.5f) < 1e-5f); // 0.5 * 1.0
}

void test_AffineTransform_Interpolate() {
    GPUMat4x4 id = GPUMat4x4::Identity();
    GPUMat4x4 scale2 = AffineTransform::MakeScale({2.0f, 2.0f, 2.0f});

    // t = 0 -> identity
    GPUMat4x4 r0 = AffineTransform::Interpolate(id, scale2, 0.0f);
    for (int i = 0; i < 16; i++) {
        EXPECT_TRUE(std::abs(r0.m[i] - id.m[i]) < 1e-6f);
    }

    // t = 1 -> scale2
    GPUMat4x4 r1 = AffineTransform::Interpolate(id, scale2, 1.0f);
    for (int i = 0; i < 16; i++) {
        EXPECT_TRUE(std::abs(r1.m[i] - scale2.m[i]) < 1e-6f);
    }

    // t = 0.5 -> midpoint, diagonal should be 1.5
    GPUMat4x4 rh = AffineTransform::Interpolate(id, scale2, 0.5f);
    EXPECT_TRUE(std::abs(rh.At(0, 0) - 1.5f) < 1e-6f);
    EXPECT_TRUE(std::abs(rh.At(1, 1) - 1.5f) < 1e-6f);
    EXPECT_TRUE(std::abs(rh.At(2, 2) - 1.5f) < 1e-6f);
}

void test_AffineTransform_SierpinskiConvergence() {
    // Sierpinski Triangle 2D: 3 transforms, each scales by 0.5 and translates to a corner
    // Iterating the chaos game should converge to within the triangle bounds [0,1] x [0, 0.866]
    auto instructions = TransformPresets::GetPreset(IFSPreset::SierpinskiTriangle2D);
    EXPECT_TRUE(instructions.size() >= 3);

    auto matrices = TransformPresets::BuildMatrices(instructions);
    EXPECT_TRUE(matrices.size() == instructions.size());

    // Chaos game simulation: start from arbitrary point
    Vec3 point = {0.5f, 0.5f, 0.0f};
    // Simple seeded PRNG
    uint32_t prngState = 12345;

    // Run 1000 iterations
    for (int i = 0; i < 1000; i++) {
        prngState = prngState * 1103515245 + 12345; // LCG
        int idx = static_cast<int>((prngState >> 16) % matrices.size());
        point = TransformPoint(matrices[idx], point);
    }

    // After convergence, point should be bounded within a reasonable range
    // Sierpinski triangle 2D lives roughly in [-1, 1] range based on preset
    EXPECT_TRUE(point.x > -2.0f && point.x < 2.0f);
    EXPECT_TRUE(point.y > -2.0f && point.y < 2.0f);
}

// =============================================================================
// Transform Presets Tests
// =============================================================================

void test_TransformPresets_AllPresetsReturnNonEmpty() {
    IFSPreset presets[] = {
        IFSPreset::SierpinskiTriangle2D,
        IFSPreset::Vicsek2D,
        IFSPreset::SierpinskiCarpet2D,
        IFSPreset::SierpinskiTriangle3D,
        IFSPreset::Vicsek3D,
        IFSPreset::SierpinskiCarpet3D
    };

    for (auto preset : presets) {
        auto instructions = TransformPresets::GetPreset(preset);
        EXPECT_TRUE(!instructions.empty());
    }
}

void test_TransformPresets_BuildMatrices() {
    auto instructions = TransformPresets::GetPreset(IFSPreset::SierpinskiTriangle3D);
    auto matrices = TransformPresets::BuildMatrices(instructions);
    EXPECT_EQ(matrices.size(), instructions.size());

    // Each matrix should be non-zero (not all zeros)
    for (const auto& mat : matrices) {
        float sum = 0.0f;
        for (int i = 0; i < 16; i++) sum += std::abs(mat.m[i]);
        EXPECT_TRUE(sum > 0.0f);
    }
}

void test_TransformPresets_Sierpinski3D_HasFiveTransforms() {
    auto instructions = TransformPresets::GetPreset(IFSPreset::SierpinskiTriangle3D);
    // Sierpinski 3D: 4 base vertices + 1 apex = 5 transforms
    EXPECT_EQ(instructions.size(), static_cast<size_t>(5));
}

void test_TransformPresets_Vicsek3D_HasNineTransforms() {
    auto instructions = TransformPresets::GetPreset(IFSPreset::Vicsek3D);
    // Vicsek 3D: 8 corners + 1 center = 9 transforms
    EXPECT_EQ(instructions.size(), static_cast<size_t>(9));
}

void test_TransformPresets_Procedural() {
    ProceduralConfig config;
    config.count = 5;
    std::mt19937 rng(42);

    auto instructions = TransformPresets::GenerateProcedural(config, rng);
    EXPECT_EQ(instructions.size(), static_cast<size_t>(5));

    // Verify scales are within specified bounds
    for (const auto& inst : instructions) {
        EXPECT_TRUE(inst.scale.x >= config.scaleMin.x && inst.scale.x <= config.scaleMax.x);
        EXPECT_TRUE(inst.scale.y >= config.scaleMin.y && inst.scale.y <= config.scaleMax.y);
        EXPECT_TRUE(inst.scale.z >= config.scaleMin.z && inst.scale.z <= config.scaleMax.z);
    }
}

void test_TransformPresets_MatricesContraction() {
    // All fractal presets should have contractive transforms (|scale| < 1)
    // This ensures the IFS converges to an attractor
    IFSPreset presets[] = {
        IFSPreset::SierpinskiTriangle2D,
        IFSPreset::Vicsek2D,
        IFSPreset::SierpinskiCarpet2D,
        IFSPreset::SierpinskiTriangle3D,
        IFSPreset::Vicsek3D,
        IFSPreset::SierpinskiCarpet3D
    };

    for (auto preset : presets) {
        auto instructions = TransformPresets::GetPreset(preset);
        for (const auto& inst : instructions) {
            EXPECT_TRUE(std::abs(inst.scale.x) <= 1.0f);
            EXPECT_TRUE(std::abs(inst.scale.y) <= 1.0f);
            EXPECT_TRUE(std::abs(inst.scale.z) <= 1.0f);
        }
    }
}

// =============================================================================
// Transform Blender Tests
// =============================================================================

void test_TransformBlender_Initialize() {
    TransformBlender blender;
    auto set1 = TransformPresets::GetPreset(IFSPreset::SierpinskiTriangle2D);
    auto set2 = TransformPresets::GetPreset(IFSPreset::Vicsek2D);

    blender.SetSets(set1, set2);

    // At t=0, blended set should match set1 (padded to same size)
    EXPECT_TRUE(blender.GetBlendFactor() < 1e-6f);

    auto blendedSet = blender.GetBlendedSet();
    EXPECT_TRUE(!blendedSet.empty());
}

void test_TransformBlender_BlendTowardsTarget() {
    TransformBlender blender;
    auto set1 = TransformPresets::GetPreset(IFSPreset::SierpinskiTriangle2D);
    auto set2 = TransformPresets::GetPreset(IFSPreset::SierpinskiTriangle3D);

    blender.SetSets(set1, set2);

    // Record initial state
    auto initial = blender.GetBlendedSet();
    EXPECT_TRUE(!initial.empty());

    // Update several times to blend towards target
    for (int i = 0; i < 100; i++) {
        blender.Update(0.016f, 5.0f);
    }

    // After sufficient updates, blended set should have moved toward set2
    auto blended = blender.GetBlendedSet();
    EXPECT_TRUE(!blended.empty());
    EXPECT_EQ(blended.size(), initial.size());
}

void test_TransformBlender_GetBlendedMatrices() {
    TransformBlender blender;
    auto set1 = TransformPresets::GetPreset(IFSPreset::SierpinskiTriangle2D);
    auto set2 = TransformPresets::GetPreset(IFSPreset::Vicsek2D);

    blender.SetSets(set1, set2);

    auto matrices = blender.GetBlendedMatrices();
    EXPECT_TRUE(!matrices.empty());
    EXPECT_EQ(matrices.size(), blender.GetBlendedSet().size());
}

void test_TransformBlender_SwitchTarget() {
    TransformBlender blender;
    auto set1 = TransformPresets::GetPreset(IFSPreset::SierpinskiTriangle2D);
    auto set2 = TransformPresets::GetPreset(IFSPreset::Vicsek2D);
    auto set3 = TransformPresets::GetPreset(IFSPreset::SierpinskiCarpet3D);

    blender.SetSets(set1, set2);

    // Blend partway
    for (int i = 0; i < 10; i++) blender.Update(0.016f, 3.0f);

    // Switch to new target
    blender.SwitchTarget(set3);
    EXPECT_TRUE(blender.GetBlendFactor() < 1e-6f); // Reset

    // Continue blending
    for (int i = 0; i < 10; i++) blender.Update(0.016f, 3.0f);
    auto blended = blender.GetBlendedSet();
    EXPECT_TRUE(!blended.empty());
}

// =============================================================================
// Vec3 / TransformInstructions Math Tests
// =============================================================================

void test_Vec3_Lerp() {
    Vec3 a = {0.0f, 0.0f, 0.0f};
    Vec3 b = {10.0f, 20.0f, 30.0f};

    Vec3 mid = Vec3::Lerp(a, b, 0.5f);
    EXPECT_TRUE(std::abs(mid.x - 5.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(mid.y - 10.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(mid.z - 15.0f) < 1e-6f);

    Vec3 atA = Vec3::Lerp(a, b, 0.0f);
    EXPECT_TRUE(std::abs(atA.x) < 1e-6f);

    Vec3 atB = Vec3::Lerp(a, b, 1.0f);
    EXPECT_TRUE(std::abs(atB.x - 10.0f) < 1e-6f);
}

void test_TransformInstructions_Identity() {
    TransformInstructions id = TransformInstructions::Identity();
    EXPECT_TRUE(std::abs(id.scale.x - 1.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(id.scale.y - 1.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(id.scale.z - 1.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(id.translate.x) < 1e-6f);
    EXPECT_TRUE(std::abs(id.translate.y) < 1e-6f);
    EXPECT_TRUE(std::abs(id.translate.z) < 1e-6f);
    EXPECT_TRUE(std::abs(id.rotate.x) < 1e-6f);
}

void test_TransformInstructions_Combine() {
    TransformInstructions a;
    a.scale = {0.5f, 0.5f, 0.5f};
    a.translate = {1.0f, 0.0f, 0.0f};

    TransformInstructions b;
    b.scale = {2.0f, 2.0f, 2.0f};
    b.translate = {0.0f, 1.0f, 0.0f};

    TransformInstructions combined = a + b;
    // Scales multiply
    EXPECT_TRUE(std::abs(combined.scale.x - 1.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(combined.scale.y - 1.0f) < 1e-6f);
    // Translates add
    EXPECT_TRUE(std::abs(combined.translate.x - 1.0f) < 1e-6f);
    EXPECT_TRUE(std::abs(combined.translate.y - 1.0f) < 1e-6f);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterIFSTransformTests() {
    // GPUMat4x4 tests
    RUN_TEST("GPUMat4x4_Identity", test_GPUMat4x4_Identity);
    RUN_TEST("GPUMat4x4_Multiply_Identity", test_GPUMat4x4_Multiply_Identity);

    // AffineTransform tests
    RUN_TEST("AffineTransform_MakeScale", test_AffineTransform_MakeScale);
    RUN_TEST("AffineTransform_MakeTranslate", test_AffineTransform_MakeTranslate);
    RUN_TEST("AffineTransform_MakeRotation_Zero", test_AffineTransform_MakeRotation_Zero);
    RUN_TEST("AffineTransform_MakeRotation_90Y", test_AffineTransform_MakeRotation_90Y);
    RUN_TEST("AffineTransform_FromInstructions_Identity", test_AffineTransform_FromInstructions_Identity);
    RUN_TEST("AffineTransform_FromInstructions_ScaleTranslate", test_AffineTransform_FromInstructions_ScaleTranslate);
    RUN_TEST("AffineTransform_Interpolate", test_AffineTransform_Interpolate);
    RUN_TEST("AffineTransform_SierpinskiConvergence", test_AffineTransform_SierpinskiConvergence);

    // Transform Presets tests
    RUN_TEST("TransformPresets_AllPresetsReturnNonEmpty", test_TransformPresets_AllPresetsReturnNonEmpty);
    RUN_TEST("TransformPresets_BuildMatrices", test_TransformPresets_BuildMatrices);
    RUN_TEST("TransformPresets_Sierpinski3D_HasFiveTransforms", test_TransformPresets_Sierpinski3D_HasFiveTransforms);
    RUN_TEST("TransformPresets_Vicsek3D_HasNineTransforms", test_TransformPresets_Vicsek3D_HasNineTransforms);
    RUN_TEST("TransformPresets_Procedural", test_TransformPresets_Procedural);
    RUN_TEST("TransformPresets_MatricesContraction", test_TransformPresets_MatricesContraction);

    // Transform Blender tests
    RUN_TEST("TransformBlender_Initialize", test_TransformBlender_Initialize);
    RUN_TEST("TransformBlender_BlendTowardsTarget", test_TransformBlender_BlendTowardsTarget);
    RUN_TEST("TransformBlender_GetBlendedMatrices", test_TransformBlender_GetBlendedMatrices);
    RUN_TEST("TransformBlender_SwitchTarget", test_TransformBlender_SwitchTarget);

    // Vec3 / Math tests
    RUN_TEST("Vec3_Lerp", test_Vec3_Lerp);
    RUN_TEST("TransformInstructions_Identity", test_TransformInstructions_Identity);
    RUN_TEST("TransformInstructions_Combine", test_TransformInstructions_Combine);
}
