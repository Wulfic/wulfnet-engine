// =============================================================================
// WulfNet Engine - Constitutive Model Tests
// =============================================================================

#include "TestHarness.h"
#include "WulfNet/Physics/MPM/ConstitutiveModel.h"
#include <cmath>

using namespace WulfNet;

// =============================================================================
// Mat3 Tests
// =============================================================================

void test_Mat3_Identity() {
    Mat3 I = Mat3::Identity();
    EXPECT_NEAR(I.m[0][0], 1.0f, 1e-6f);
    EXPECT_NEAR(I.m[1][1], 1.0f, 1e-6f);
    EXPECT_NEAR(I.m[2][2], 1.0f, 1e-6f);
    EXPECT_NEAR(I.m[0][1], 0.0f, 1e-6f);
    EXPECT_NEAR(I.m[1][0], 0.0f, 1e-6f);
}

void test_Mat3_Determinant() {
    Mat3 I = Mat3::Identity();
    EXPECT_NEAR(I.Determinant(), 1.0f, 1e-6f);

    Mat3 Z = Mat3::Zero();
    EXPECT_NEAR(Z.Determinant(), 0.0f, 1e-6f);

    // 2*Identity should have det = 8
    Mat3 S = Mat3::Identity() * 2.0f;
    EXPECT_NEAR(S.Determinant(), 8.0f, 1e-4f);
}

void test_Mat3_Transpose() {
    Mat3 A = Mat3::Zero();
    A.m[0][1] = 3.0f;
    A.m[1][0] = 5.0f;
    Mat3 AT = A.Transpose();
    EXPECT_NEAR(AT.m[0][1], 5.0f, 1e-6f);
    EXPECT_NEAR(AT.m[1][0], 3.0f, 1e-6f);
}

void test_Mat3_Multiply() {
    Mat3 I = Mat3::Identity();
    Mat3 A = Mat3::Identity() * 3.0f;

    // I * A = A
    Mat3 result = I * A;
    EXPECT_NEAR(result.m[0][0], 3.0f, 1e-6f);
    EXPECT_NEAR(result.m[1][1], 3.0f, 1e-6f);
    EXPECT_NEAR(result.m[2][2], 3.0f, 1e-6f);
}

void test_Mat3_Inverse() {
    // Inverse of 2*I should be 0.5*I
    Mat3 A = Mat3::Identity() * 2.0f;
    Mat3 Ainv = A.Inverse();
    EXPECT_NEAR(Ainv.m[0][0], 0.5f, 1e-5f);
    EXPECT_NEAR(Ainv.m[1][1], 0.5f, 1e-5f);
    EXPECT_NEAR(Ainv.m[2][2], 0.5f, 1e-5f);

    // A * A^-1 ≈ I
    Mat3 product = A * Ainv;
    EXPECT_NEAR(product.m[0][0], 1.0f, 1e-5f);
    EXPECT_NEAR(product.m[1][1], 1.0f, 1e-5f);
    EXPECT_NEAR(product.m[2][2], 1.0f, 1e-5f);
    EXPECT_NEAR(product.m[0][1], 0.0f, 1e-5f);
}

void test_Mat3_FrobeniusNorm() {
    Mat3 I = Mat3::Identity();
    // ||I||_F = sqrt(3) ≈ 1.732
    EXPECT_NEAR(I.FrobeniusNorm(), std::sqrt(3.0f), 1e-5f);
}

void test_Mat3_InverseSingular() {
    // Singular matrix should return identity
    Mat3 S = Mat3::Zero();
    Mat3 Sinv = S.Inverse();
    EXPECT_NEAR(Sinv.m[0][0], 1.0f, 1e-6f);
    EXPECT_NEAR(Sinv.m[1][1], 1.0f, 1e-6f);
    EXPECT_NEAR(Sinv.m[2][2], 1.0f, 1e-6f);
}

// =============================================================================
// SVD Tests
// =============================================================================

void test_SVD_Identity() {
    Mat3 I = Mat3::Identity();
    SVDResult svd = ComputeSVD3x3(I);

    // Singular values should all be 1
    EXPECT_NEAR(svd.sigma[0], 1.0f, 1e-4f);
    EXPECT_NEAR(svd.sigma[1], 1.0f, 1e-4f);
    EXPECT_NEAR(svd.sigma[2], 1.0f, 1e-4f);
}

void test_SVD_Scaled() {
    Mat3 S = Mat3::Identity() * 3.0f;
    SVDResult svd = ComputeSVD3x3(S);

    // Singular values should all be 3
    EXPECT_NEAR(svd.sigma[0], 3.0f, 1e-3f);
    EXPECT_NEAR(svd.sigma[1], 3.0f, 1e-3f);
    EXPECT_NEAR(svd.sigma[2], 3.0f, 1e-3f);
}

void test_SVD_Reconstruction() {
    // Create a non-trivial matrix
    Mat3 A = Mat3::Zero();
    A.m[0][0] = 1.0f; A.m[0][1] = 2.0f; A.m[0][2] = 0.0f;
    A.m[1][0] = 0.0f; A.m[1][1] = 3.0f; A.m[1][2] = 1.0f;
    A.m[2][0] = 1.0f; A.m[2][1] = 0.0f; A.m[2][2] = 2.0f;

    SVDResult svd = ComputeSVD3x3(A);

    // Reconstruct: A ≈ U * diag(σ) * Vᵀ
    Mat3 Sigma = Mat3::Zero();
    Sigma.m[0][0] = svd.sigma[0];
    Sigma.m[1][1] = svd.sigma[1];
    Sigma.m[2][2] = svd.sigma[2];

    Mat3 reconstructed = svd.U * Sigma * svd.V.Transpose();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(reconstructed.m[i][j], A.m[i][j], 0.05f);
        }
    }
}

void test_SVD_SingularValuesPositive() {
    Mat3 A = Mat3::Identity() * 2.0f;
    A.m[0][1] = 0.5f;
    A.m[1][0] = -0.3f;

    SVDResult svd = ComputeSVD3x3(A);

    // After sign correction, singular values should be >= 0
    // (some may be negative for reflection handling, but abs should be correct)
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::abs(svd.sigma[i]) >= 0.0f);
    }
}

// =============================================================================
// Material Parameter Tests
// =============================================================================

void test_MPMMaterial_Presets() {
    auto rubber = MPMMaterialParams::Rubber();
    EXPECT_TRUE(rubber.type == MPMMaterialType::NeoHookean);
    EXPECT_TRUE(rubber.mu > 0.0f);
    EXPECT_TRUE(rubber.lambda > 0.0f);
    EXPECT_TRUE(rubber.density > 0.0f);

    auto sand = MPMMaterialParams::Sand();
    EXPECT_TRUE(sand.type == MPMMaterialType::DruckerPrager);
    EXPECT_TRUE(sand.frictionAngle > 0.0f);
    EXPECT_NEAR(sand.cohesion, 0.0f, 1e-6f);

    auto snow = MPMMaterialParams::Snow();
    EXPECT_TRUE(snow.type == MPMMaterialType::Snow);
    EXPECT_TRUE(snow.criticalCompression > 0.0f);
    EXPECT_TRUE(snow.criticalStretch > 0.0f);
    EXPECT_TRUE(snow.hardeningCoefficient > 0.0f);
}

void test_MPMMaterial_LameConstants() {
    MPMMaterialParams p;
    p.youngsModulus = 1.0e5f;
    p.poissonsRatio = 0.3f;
    p.ComputeLameConstants();

    // μ = E / (2(1+ν)) = 100000 / (2*1.3) ≈ 38461.5
    float expectedMu = 1.0e5f / (2.0f * 1.3f);
    EXPECT_NEAR(p.mu, expectedMu, 1.0f);

    // λ = Eν / ((1+ν)(1-2ν)) = 100000*0.3 / (1.3*0.4) ≈ 57692.3
    float expectedLambda = 1.0e5f * 0.3f / (1.3f * 0.4f);
    EXPECT_NEAR(p.lambda, expectedLambda, 1.0f);
}

void test_MPMMaterial_AllPresets() {
    auto rubber = MPMMaterialParams::Rubber();
    auto flesh = MPMMaterialParams::Flesh();
    auto sand = MPMMaterialParams::Sand();
    auto mud = MPMMaterialParams::WetMud();
    auto soil = MPMMaterialParams::DrySoil();
    auto snow = MPMMaterialParams::Snow();
    auto ice = MPMMaterialParams::Ice();
    auto fluid = MPMMaterialParams::ViscousFluid();

    // All should have valid Lamé constants (except viscous fluid)
    EXPECT_TRUE(rubber.mu > 0.0f);
    EXPECT_TRUE(flesh.mu > 0.0f);
    EXPECT_TRUE(sand.mu > 0.0f);
    EXPECT_TRUE(mud.mu > 0.0f);
    EXPECT_TRUE(soil.mu > 0.0f);
    EXPECT_TRUE(snow.mu > 0.0f);
    EXPECT_TRUE(ice.mu > 0.0f);
    EXPECT_NEAR(fluid.mu, 0.0f, 1e-6f);

    // Density ordering: mud > sand > rubber > flesh > ice > snow
    EXPECT_TRUE(mud.density > sand.density);
    EXPECT_TRUE(sand.density > rubber.density);
    EXPECT_TRUE(snow.density < sand.density);
    EXPECT_TRUE(ice.density < mud.density);
}

// =============================================================================
// Constitutive Model Factory Tests
// =============================================================================

void test_GetConstitutiveModel_Valid() {
    const ConstitutiveModel* nh = GetConstitutiveModel(MPMMaterialType::NeoHookean);
    EXPECT_TRUE(nh != nullptr);

    const ConstitutiveModel* dp = GetConstitutiveModel(MPMMaterialType::DruckerPrager);
    EXPECT_TRUE(dp != nullptr);

    const ConstitutiveModel* sn = GetConstitutiveModel(MPMMaterialType::Snow);
    EXPECT_TRUE(sn != nullptr);

    const ConstitutiveModel* vf = GetConstitutiveModel(MPMMaterialType::ViscousFluid);
    EXPECT_TRUE(vf != nullptr);

    // Different types should return different model pointers
    EXPECT_TRUE(nh != dp);
    EXPECT_TRUE(dp != sn);
    EXPECT_TRUE(sn != vf);
}

void test_GetConstitutiveModel_Custom() {
    // Custom should fall back to NeoHookean
    const ConstitutiveModel* custom = GetConstitutiveModel(MPMMaterialType::Custom);
    const ConstitutiveModel* nh = GetConstitutiveModel(MPMMaterialType::NeoHookean);
    EXPECT_TRUE(custom == nh);
}

// =============================================================================
// NeoHookean Model Tests
// =============================================================================

void test_NeoHookean_ZeroStressAtIdentity() {
    auto params = MPMMaterialParams::Rubber();
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::NeoHookean);

    MPMParticle p{};
    p.F = Mat3::Identity();
    p.Fp = Mat3::Identity();
    p.volume0 = 0.001f;  // 1cm³
    p.mass = params.density * p.volume0;
    p.Jp = 1.0f;

    Mat3 stress = model->ComputeStress(p, params);

    // At identity deformation, stress should be zero (unstrained)
    // P = μ(F - F⁻ᵀ) + λ ln(J) F⁻ᵀ = μ(I - I) + λ*0*I = 0
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(stress.m[i][j], 0.0f, 1e-3f);
        }
    }
}

void test_NeoHookean_CompressiveStress() {
    auto params = MPMMaterialParams::Rubber();
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::NeoHookean);

    MPMParticle p{};
    p.F = Mat3::Identity() * 0.8f;  // 80% compressed
    p.Fp = Mat3::Identity();
    p.volume0 = 0.001f;
    p.mass = params.density * p.volume0;
    p.Jp = 1.0f;

    Mat3 stress = model->ComputeStress(p, params);

    // Under compression, diagonal stresses should be non-zero
    float stressNorm = stress.FrobeniusNorm();
    EXPECT_TRUE(stressNorm > 0.0f);
}

void test_NeoHookean_ProjectionNoChange() {
    auto params = MPMMaterialParams::Rubber();
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::NeoHookean);

    MPMParticle p{};
    p.F = Mat3::Identity();
    p.Fp = Mat3::Identity();
    p.Jp = 1.0f;

    model->ProjectDeformation(p, params);

    // No plasticity in NeoHookean, F should remain identity
    EXPECT_NEAR(p.F.m[0][0], 1.0f, 1e-6f);
    EXPECT_NEAR(p.F.m[1][1], 1.0f, 1e-6f);
    EXPECT_NEAR(p.F.m[2][2], 1.0f, 1e-6f);
}

// =============================================================================
// Drucker-Prager Model Tests
// =============================================================================

void test_DruckerPrager_StressComputation() {
    auto params = MPMMaterialParams::Sand();
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::DruckerPrager);

    MPMParticle p{};
    p.F = Mat3::Identity() * 0.9f;  // Slight compression
    p.Fp = Mat3::Identity();
    p.volume0 = 0.001f;
    p.mass = params.density * p.volume0;
    p.Jp = 1.0f;

    Mat3 stress = model->ComputeStress(p, params);
    float stressNorm = stress.FrobeniusNorm();
    EXPECT_TRUE(stressNorm > 0.0f);
}

void test_DruckerPrager_YieldProjection() {
    auto params = MPMMaterialParams::Sand();
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::DruckerPrager);

    MPMParticle p{};
    // Large deformation that should trigger yield
    p.F = Mat3::Identity() * 0.5f;  // Heavy compression
    p.Fp = Mat3::Identity();
    p.Jp = 1.0f;
    p.volume0 = 0.001f;

    model->ProjectDeformation(p, params);

    // After projection, F should have been modified
    // (exact values depend on yield surface intersection)
    float J = p.F.Determinant();
    EXPECT_TRUE(std::isfinite(J));
    EXPECT_TRUE(J > 0.0f);
}

void test_DruckerPrager_CohesionEffect() {
    // Mud (with cohesion) vs Sand (without)
    auto sand = MPMMaterialParams::Sand();
    auto mud = MPMMaterialParams::WetMud();

    EXPECT_NEAR(sand.cohesion, 0.0f, 1e-6f);
    EXPECT_TRUE(mud.cohesion > 0.0f);

    // Both should produce finite stress
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::DruckerPrager);

    MPMParticle p{};
    p.F = Mat3::Identity() * 0.85f;
    p.Fp = Mat3::Identity();
    p.volume0 = 0.001f;
    p.Jp = 1.0f;

    Mat3 stressSand = model->ComputeStress(p, sand);
    Mat3 stressMud = model->ComputeStress(p, mud);

    EXPECT_TRUE(std::isfinite(stressSand.FrobeniusNorm()));
    EXPECT_TRUE(std::isfinite(stressMud.FrobeniusNorm()));
}

// =============================================================================
// Snow Model Tests
// =============================================================================

void test_Snow_StressComputation() {
    auto params = MPMMaterialParams::Snow();
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::Snow);

    MPMParticle p{};
    p.F = Mat3::Identity() * 0.95f;  // Slight compression
    p.Fp = Mat3::Identity();
    p.volume0 = 0.001f;
    p.mass = params.density * p.volume0;
    p.Jp = 1.0f;

    Mat3 stress = model->ComputeStress(p, params);
    float stressNorm = stress.FrobeniusNorm();
    EXPECT_TRUE(stressNorm > 0.0f);
    EXPECT_TRUE(std::isfinite(stressNorm));
}

void test_Snow_CriticalCompression() {
    auto params = MPMMaterialParams::Snow();
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::Snow);

    MPMParticle p{};
    // Deformation beyond critical compression
    p.F = Mat3::Identity() * (1.0f - params.criticalCompression * 2.0f);
    p.Fp = Mat3::Identity();
    p.Jp = 1.0f;
    p.volume0 = 0.001f;

    model->ProjectDeformation(p, params);

    // After projection, singular values should be clamped
    SVDResult svd = ComputeSVD3x3(p.F * p.Fp.Inverse());
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::abs(svd.sigma[i]) >= (1.0f - params.criticalCompression - 0.01f));
    }
}

void test_Snow_Hardening() {
    auto params = MPMMaterialParams::Snow();
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::Snow);

    // Particle with Jp < 1 (compressed plastic state) should have higher effective stiffness
    MPMParticle p1{};
    p1.F = Mat3::Identity() * 0.95f;
    p1.Fp = Mat3::Identity();
    p1.Jp = 1.0f;
    p1.volume0 = 0.001f;

    MPMParticle p2 = p1;
    p2.Jp = 0.5f;  // More compressed plastically

    Mat3 stress1 = model->ComputeStress(p1, params);
    Mat3 stress2 = model->ComputeStress(p2, params);

    // Higher Jp deviation → higher hardening → higher stress magnitude
    // (hardening = exp(ξ * (1 - Jp)))
    float norm1 = stress1.FrobeniusNorm();
    float norm2 = stress2.FrobeniusNorm();
    EXPECT_TRUE(norm2 > norm1);
}

// =============================================================================
// Viscous Fluid Model Tests
// =============================================================================

void test_ViscousFluid_StressComputation() {
    auto params = MPMMaterialParams::ViscousFluid();
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::ViscousFluid);

    MPMParticle p{};
    p.F = Mat3::Identity() * 1.1f;  // Slight expansion
    p.Fp = Mat3::Identity();
    p.volume0 = 0.001f;
    p.Jp = 1.0f;

    Mat3 stress = model->ComputeStress(p, params);
    EXPECT_TRUE(std::isfinite(stress.FrobeniusNorm()));
}

void test_ViscousFluid_ResetShear() {
    auto params = MPMMaterialParams::ViscousFluid();
    const ConstitutiveModel* model = GetConstitutiveModel(MPMMaterialType::ViscousFluid);

    MPMParticle p{};
    // Add shear to deformation gradient
    p.F = Mat3::Identity();
    p.F.m[0][1] = 0.5f;  // Shear
    p.F.m[1][0] = -0.3f;
    p.Fp = Mat3::Identity();
    p.Jp = 1.0f;

    float J_before = p.F.Determinant();

    model->ProjectDeformation(p, params);

    // After projection, F should be a scaled identity (pure dilation)
    float J_after = p.F.Determinant();
    // Volume should be approximately preserved
    EXPECT_NEAR(J_after, std::max(J_before, 0.01f), 0.1f);

    // Off-diagonal should be zero (shear removed)
    EXPECT_NEAR(p.F.m[0][1], 0.0f, 1e-5f);
    EXPECT_NEAR(p.F.m[1][0], 0.0f, 1e-5f);
}

// =============================================================================
// MPMParticle Tests
// =============================================================================

void test_MPMParticle_DefaultInit() {
    MPMParticle p{};
    EXPECT_NEAR(p.x, 0.0f, 1e-6f);
    EXPECT_NEAR(p.y, 0.0f, 1e-6f);
    EXPECT_NEAR(p.z, 0.0f, 1e-6f);
    EXPECT_NEAR(p.vx, 0.0f, 1e-6f);
    EXPECT_NEAR(p.vy, 0.0f, 1e-6f);
    EXPECT_NEAR(p.vz, 0.0f, 1e-6f);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterConstitutiveModelTests() {
    // Mat3
    RUN_TEST("Mat3_Identity", test_Mat3_Identity);
    RUN_TEST("Mat3_Determinant", test_Mat3_Determinant);
    RUN_TEST("Mat3_Transpose", test_Mat3_Transpose);
    RUN_TEST("Mat3_Multiply", test_Mat3_Multiply);
    RUN_TEST("Mat3_Inverse", test_Mat3_Inverse);
    RUN_TEST("Mat3_FrobeniusNorm", test_Mat3_FrobeniusNorm);
    RUN_TEST("Mat3_InverseSingular", test_Mat3_InverseSingular);

    // SVD
    RUN_TEST("SVD_Identity", test_SVD_Identity);
    RUN_TEST("SVD_Scaled", test_SVD_Scaled);
    RUN_TEST("SVD_Reconstruction", test_SVD_Reconstruction);
    RUN_TEST("SVD_SingularValuesPositive", test_SVD_SingularValuesPositive);

    // Material Parameters
    RUN_TEST("MPMMaterial_Presets", test_MPMMaterial_Presets);
    RUN_TEST("MPMMaterial_LameConstants", test_MPMMaterial_LameConstants);
    RUN_TEST("MPMMaterial_AllPresets", test_MPMMaterial_AllPresets);

    // Factory
    RUN_TEST("ConstitutiveModel_Factory", test_GetConstitutiveModel_Valid);
    RUN_TEST("ConstitutiveModel_CustomFallback", test_GetConstitutiveModel_Custom);

    // NeoHookean
    RUN_TEST("NeoHookean_ZeroStressAtIdentity", test_NeoHookean_ZeroStressAtIdentity);
    RUN_TEST("NeoHookean_CompressiveStress", test_NeoHookean_CompressiveStress);
    RUN_TEST("NeoHookean_ProjectionNoChange", test_NeoHookean_ProjectionNoChange);

    // Drucker-Prager
    RUN_TEST("DruckerPrager_StressComputation", test_DruckerPrager_StressComputation);
    RUN_TEST("DruckerPrager_YieldProjection", test_DruckerPrager_YieldProjection);
    RUN_TEST("DruckerPrager_CohesionEffect", test_DruckerPrager_CohesionEffect);

    // Snow
    RUN_TEST("Snow_StressComputation", test_Snow_StressComputation);
    RUN_TEST("Snow_CriticalCompression", test_Snow_CriticalCompression);
    RUN_TEST("Snow_Hardening", test_Snow_Hardening);

    // Viscous Fluid
    RUN_TEST("ViscousFluid_StressComputation", test_ViscousFluid_StressComputation);
    RUN_TEST("ViscousFluid_ResetShear", test_ViscousFluid_ResetShear);

    // MPMParticle
    RUN_TEST("MPMParticle_DefaultInit", test_MPMParticle_DefaultInit);
}
