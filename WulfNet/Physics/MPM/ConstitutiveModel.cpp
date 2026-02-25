// =============================================================================
// WulfNet Engine - MPM Constitutive Models Implementation
// =============================================================================

#include "ConstitutiveModel.h"
#include <cassert>

namespace WulfNet {

// =============================================================================
// SVD 3x3 — Jacobi Iteration Method
// =============================================================================
// Computes A = U * diag(sigma) * Vᵀ using iterative Jacobi rotations on AᵀA.
// Robust for degenerate/singular matrices. Convergence in ~5-10 sweeps.
// =============================================================================

namespace {

// 2x2 symmetric Schur decomposition: diag(c,s) where G = [c s; -s c]
// zeroes out a[p][q] in the symmetric matrix AᵀA
void JacobiRotation(float app, float aqq, float apq, float& c, float& s) {
    if (std::abs(apq) < 1e-10f) {
        c = 1.0f;
        s = 0.0f;
        return;
    }
    float tau = (aqq - app) / (2.0f * apq);
    float t;
    if (tau >= 0.0f) {
        t = 1.0f / (tau + std::sqrt(1.0f + tau * tau));
    } else {
        t = -1.0f / (-tau + std::sqrt(1.0f + tau * tau));
    }
    c = 1.0f / std::sqrt(1.0f + t * t);
    s = t * c;
}

// Apply Givens rotation G(p,q,theta) to columns of A on the right: A = A * G
void RotateRight(Mat3& A, int p, int q, float c, float s) {
    for (int i = 0; i < 3; ++i) {
        float ap = A.m[i][p];
        float aq = A.m[i][q];
        A.m[i][p] = c * ap - s * aq;
        A.m[i][q] = s * ap + c * aq;
    }
}

// Apply Givens rotation G(p,q,theta) to rows of A on the left: A = Gᵀ * A
void RotateLeft(Mat3& A, int p, int q, float c, float s) {
    for (int j = 0; j < 3; ++j) {
        float ap = A.m[p][j];
        float aq = A.m[q][j];
        A.m[p][j] = c * ap - s * aq;
        A.m[q][j] = s * ap + c * aq;
    }
}

} // anonymous namespace

SVDResult ComputeSVD3x3(const Mat3& A) {
    SVDResult result;

    // Compute S = AᵀA (symmetric)
    Mat3 ATA = A.Transpose() * A;

    // V accumulates right rotations
    Mat3 V = Mat3::Identity();

    // Jacobi eigenvalue decomposition of AᵀA
    // This symmetrically diagonalizes AᵀA = V * D * Vᵀ
    constexpr int maxSweeps = 12;
    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        // Check convergence (off-diagonal elements)
        float offDiag = std::abs(ATA.m[0][1]) + std::abs(ATA.m[0][2]) + std::abs(ATA.m[1][2]);
        if (offDiag < 1e-8f) break;

        // Sweep through all 3 pairs: (0,1), (0,2), (1,2)
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (std::abs(ATA.m[p][q]) < 1e-10f) continue;

                float c, s;
                JacobiRotation(ATA.m[p][p], ATA.m[q][q], ATA.m[p][q], c, s);

                // Apply: ATA = Gᵀ * ATA * G (symmetric update)
                // More efficient to just do the full rotation
                Mat3 G = Mat3::Identity();
                G.m[p][p] = c;  G.m[p][q] = -s;
                G.m[q][p] = s;  G.m[q][q] = c;

                ATA = G.Transpose() * ATA * G;

                // Accumulate V
                RotateRight(V, p, q, c, s);
            }
        }
    }

    // Singular values are sqrt of eigenvalues of AᵀA
    for (int i = 0; i < 3; ++i) {
        result.sigma[i] = std::sqrt(std::max(0.0f, ATA.m[i][i]));
    }

    result.V = V;

    // U = A * V * Σ⁻¹
    Mat3 AV = A * V;
    result.U = Mat3::Identity();
    for (int i = 0; i < 3; ++i) {
        if (result.sigma[i] > 1e-8f) {
            float invSigma = 1.0f / result.sigma[i];
            for (int j = 0; j < 3; ++j) {
                result.U.m[j][i] = AV.m[j][i] * invSigma;
            }
        }
    }

    // Ensure proper rotations (det = +1), not reflections
    float detU = result.U.Determinant();
    float detV = result.V.Determinant();

    if (detU < 0.0f) {
        // Flip the column with smallest singular value
        int minIdx = 0;
        if (result.sigma[1] < result.sigma[minIdx]) minIdx = 1;
        if (result.sigma[2] < result.sigma[minIdx]) minIdx = 2;

        for (int j = 0; j < 3; ++j) {
            result.U.m[j][minIdx] = -result.U.m[j][minIdx];
        }
        result.sigma[minIdx] = -result.sigma[minIdx];
    }

    if (detV < 0.0f) {
        int minIdx = 0;
        if (std::abs(result.sigma[1]) < std::abs(result.sigma[minIdx])) minIdx = 1;
        if (std::abs(result.sigma[2]) < std::abs(result.sigma[minIdx])) minIdx = 2;

        for (int j = 0; j < 3; ++j) {
            result.V.m[j][minIdx] = -result.V.m[j][minIdx];
        }
        result.sigma[minIdx] = -result.sigma[minIdx];
    }

    return result;
}

// =============================================================================
// Neo-Hookean Model
// =============================================================================
// First Piola-Kirchhoff stress:
//   P = μ(F - F⁻ᵀ) + λ ln(J) F⁻ᵀ
// Kirchhoff stress:
//   τ = P * Fᵀ = μ(FFᵀ - I) + λ ln(J) I
// =============================================================================

Mat3 NeoHookeanModel::ComputeStress(const MPMParticle& particle,
                                     const MPMMaterialParams& params) const {
    const Mat3& F = particle.F;
    float J = F.Determinant();

    // Clamp J to prevent numerical issues at extreme compression/extension
    J = std::max(J, 0.01f);

    float logJ = std::log(J);

    // P = μ(F - F⁻ᵀ) + λ ln(J) F⁻ᵀ
    Mat3 FInvT = F.InverseTranspose();
    Mat3 P = (F - FInvT) * params.mu + FInvT * (params.lambda * logJ);

    // Kirchhoff stress: τ = P * Fᵀ (for APIC)
    // But for P2G force transfer, we return volume * P * Fᵀ
    return P * F.Transpose() * (-particle.volume0);
}

void NeoHookeanModel::ProjectDeformation(MPMParticle& particle,
                                          const MPMMaterialParams& /*params*/) const {
    // Neo-Hookean is purely elastic — no plastic correction needed
    // Just clamp J to prevent extreme deformation
    float J = particle.F.Determinant();
    if (J < 0.01f) {
        // Reset to identity to prevent inversion
        particle.F = Mat3::Identity();
    }
}

// =============================================================================
// Drucker-Prager Model
// =============================================================================
// Elastic trial:
//   Fe_trial = F (assume F already contains elastic deformation)
//   Compute SVD: Fe_trial = U * diag(σ) * Vᵀ
//   ε = ln(σ) (Hencky strain in principal space)
//
// Yield function (in principal log-strain space):
//   f(ε) = ||dev(ε)|| + α * tr(ε) + c_yield
//   where α = sqrt(2/3) * 2*sin(φ) / (3 - sin(φ))
//
// Return mapping:
//   If f > 0: project ε back onto yield surface
//   Reconstruct σ = exp(ε_projected)
//   Fe = U * diag(σ_projected) * Vᵀ
// =============================================================================

Mat3 DruckerPragerModel::ComputeStress(const MPMParticle& particle,
                                        const MPMMaterialParams& params) const {
    const Mat3& F = particle.F;
    float J = F.Determinant();
    J = std::max(J, 0.01f);
    float logJ = std::log(J);

    // Use same Neo-Hookean energy but the elastic part only
    Mat3 FInvT = F.InverseTranspose();
    Mat3 P = (F - FInvT) * params.mu + FInvT * (params.lambda * logJ);

    return P * F.Transpose() * (-particle.volume0);
}

void DruckerPragerModel::ProjectDeformation(MPMParticle& particle,
                                             const MPMMaterialParams& params) const {
    // SVD of deformation gradient
    SVDResult svd = ComputeSVD3x3(particle.F);

    // Log strains (Hencky strain in principal space)
    float epsilon[3];
    for (int i = 0; i < 3; ++i) {
        epsilon[i] = std::log(std::max(std::abs(svd.sigma[i]), 1e-8f));
    }

    // Trace and deviatoric part
    float trEps = epsilon[0] + epsilon[1] + epsilon[2];
    float devEps[3];
    float meanEps = trEps / 3.0f;
    for (int i = 0; i < 3; ++i) {
        devEps[i] = epsilon[i] - meanEps;
    }

    float devNorm = std::sqrt(devEps[0]*devEps[0] + devEps[1]*devEps[1] + devEps[2]*devEps[2]);

    // Drucker-Prager yield parameters
    float sinPhi = std::sin(params.frictionAngle * 3.14159265f / 180.0f);
    float alpha = std::sqrt(2.0f / 3.0f) * 2.0f * sinPhi / (3.0f - sinPhi);

    // Cohesion contribution
    float cYield = 0.0f;
    if (params.cohesion > 0.0f) {
        float cosPhi = std::cos(params.frictionAngle * 3.14159265f / 180.0f);
        cYield = params.cohesion * cosPhi / (params.mu + 1e-10f);
    }

    // Yield function
    float f = devNorm + alpha * trEps + cYield;

    if (f > 0.0f) {
        // Project back onto yield surface

        if (trEps > 0.0f) {
            // Tensile: clamp to zero volume change (sand can't sustain tension)
            for (int i = 0; i < 3; ++i) {
                epsilon[i] = 0.0f;
            }
        } else {
            // Compressive: project deviatoric onto cone
            float sinPsi = std::sin(params.dilatancyAngle * 3.14159265f / 180.0f);
            float beta = std::sqrt(2.0f / 3.0f) * 2.0f * sinPsi / (3.0f - sinPsi);

            float delta = devNorm > 1e-10f ?
                (devNorm + alpha * trEps) / (devNorm * (1.0f + alpha * beta)) : 0.0f;
            delta = std::max(0.0f, std::min(delta, 1.0f));

            for (int i = 0; i < 3; ++i) {
                epsilon[i] = epsilon[i] - delta * devEps[i];
            }
        }

        // Reconstruct singular values from projected strains
        for (int i = 0; i < 3; ++i) {
            svd.sigma[i] = std::exp(epsilon[i]);
        }

        // Reconstruct F = U * diag(sigma) * Vᵀ
        Mat3 Sigma = Mat3::Zero();
        Sigma.m[0][0] = svd.sigma[0];
        Sigma.m[1][1] = svd.sigma[1];
        Sigma.m[2][2] = svd.sigma[2];

        particle.F = svd.U * Sigma * svd.V.Transpose();

        // Update plastic determinant
        particle.Jp *= std::exp(trEps - epsilon[0] - epsilon[1] - epsilon[2]);
        particle.Jp = std::max(0.01f, particle.Jp);
    }
}

// =============================================================================
// Snow Model (Stomakhin et al. 2013)
// =============================================================================
// Key ideas:
//   1. Multiplicative decomposition: F = Fe * Fp
//   2. Singular value clamping: σᵢ ∈ [1-θc, 1+θs]
//   3. Exponential hardening: μ̂ = μ₀ exp(ξ(1-Jp))
// =============================================================================

Mat3 SnowModel::ComputeStress(const MPMParticle& particle,
                                const MPMMaterialParams& params) const {
    // Compute elastic part: Fe = F * Fp⁻¹
    Mat3 Fe = particle.F * particle.Fp.Inverse();
    float Je = Fe.Determinant();
    Je = std::max(Je, 0.01f);
    float logJe = std::log(Je);

    // Hardened Lamé constants
    float hardening = std::exp(params.hardeningCoefficient * (1.0f - particle.Jp));
    float mu_hat = params.mu * hardening;
    float lambda_hat = params.lambda * hardening;

    // P = 2μ̂(Fe - Re) + λ̂(Je - 1)Je * Fe⁻ᵀ
    // where Re is the rotation from polar decomposition Fe = Re * Se

    // Polar decomposition via SVD: Fe = U * Σ * Vᵀ, Re = U * Vᵀ
    SVDResult svd = ComputeSVD3x3(Fe);
    Mat3 Re = svd.U * svd.V.Transpose();

    Mat3 FeInvT = Fe.InverseTranspose();
    Mat3 P = (Fe - Re) * (2.0f * mu_hat) + FeInvT * (lambda_hat * (Je - 1.0f) * Je);

    return P * particle.F.Transpose() * (-particle.volume0);
}

void SnowModel::ProjectDeformation(MPMParticle& particle,
                                    const MPMMaterialParams& params) const {
    // Compute elastic part
    Mat3 FpInv = particle.Fp.Inverse();
    Mat3 Fe = particle.F * FpInv;

    // SVD of elastic deformation
    SVDResult svd = ComputeSVD3x3(Fe);

    // Clamp singular values
    float thetaC = params.criticalCompression;
    float thetaS = params.criticalStretch;

    for (int i = 0; i < 3; ++i) {
        float clamped = std::max(1.0f - thetaC, std::min(1.0f + thetaS, svd.sigma[i]));
        svd.sigma[i] = clamped;
    }

    // Reconstruct clamped Fe
    Mat3 SigmaClamped = Mat3::Zero();
    SigmaClamped.m[0][0] = svd.sigma[0];
    SigmaClamped.m[1][1] = svd.sigma[1];
    SigmaClamped.m[2][2] = svd.sigma[2];

    Mat3 FeClamped = svd.U * SigmaClamped * svd.V.Transpose();

    // Update plastic deformation: Fp_new = FeClamped⁻¹ * F
    particle.Fp = FeClamped.Inverse() * particle.F;

    // Update elastic deformation
    particle.F = FeClamped * particle.Fp;

    // IMPORTANT: keep Fe as the clamped version for stress computation
    // Reconstruct full F from elastic * plastic
    particle.F = FeClamped * particle.Fp;

    // Update plastic determinant
    particle.Jp = particle.Fp.Determinant();
    particle.Jp = std::max(0.01f, particle.Jp);
}

// =============================================================================
// Viscous Fluid Model
// =============================================================================
// Treat as very soft Neo-Hookean that always resets plastic deformation.
// This ensures volume preservation while allowing arbitrary shear.
// =============================================================================

Mat3 ViscousFluidModel::ComputeStress(const MPMParticle& particle,
                                       const MPMMaterialParams& params) const {
    const Mat3& F = particle.F;
    float J = F.Determinant();
    J = std::max(J, 0.01f);

    // Pressure-only stress for volume preservation
    // P = -pressure * J * F⁻ᵀ
    // Use bulk modulus from viscosity as a soft pressure
    float bulkModulus = params.viscosity * 1000.0f;  // Derive from viscosity
    float pressure = bulkModulus * (J - 1.0f);

    Mat3 FInvT = F.InverseTranspose();
    Mat3 P = FInvT * (-pressure * J);

    return P * F.Transpose() * (-particle.volume0);
}

void ViscousFluidModel::ProjectDeformation(MPMParticle& particle,
                                            const MPMMaterialParams& /*params*/) const {
    // For fluid: reset F to a pure dilation (remove all shear)
    // Preserve only the volume change (det(F))
    float J = particle.F.Determinant();
    J = std::max(J, 0.01f);

    // Reset to scaled identity: F = J^(1/3) * I
    float cbrtJ = std::cbrt(J);
    particle.F = Mat3::Identity() * cbrtJ;
}

// =============================================================================
// Singleton model instances
// =============================================================================

static NeoHookeanModel    s_neoHookean;
static DruckerPragerModel s_druckerPrager;
static SnowModel          s_snow;
static ViscousFluidModel  s_viscousFluid;

const ConstitutiveModel* GetConstitutiveModel(MPMMaterialType type) {
    switch (type) {
        case MPMMaterialType::NeoHookean:    return &s_neoHookean;
        case MPMMaterialType::DruckerPrager: return &s_druckerPrager;
        case MPMMaterialType::Snow:          return &s_snow;
        case MPMMaterialType::ViscousFluid:  return &s_viscousFluid;
        default:                             return &s_neoHookean;
    }
}

} // namespace WulfNet
