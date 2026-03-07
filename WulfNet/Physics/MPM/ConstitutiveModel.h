// =============================================================================
// WulfNet Engine - MPM Constitutive Models
// =============================================================================
// Material-specific stress computation for the Material Point Method.
// Implements physically-based constitutive models:
//   - Neo-Hookean (rubber, flesh, general elastic)
//   - Drucker-Prager (sand, mud, soil - granular materials)
//   - Disney Snow (snow, ice - plasticity with hardening)
//   - Viscous Fluid (fallback for FluidSystem compatibility)
//
// References:
//   - Stomakhin et al. 2013 "A Material Point Method for Snow Simulation"
//   - Klár et al. 2016 "Drucker-Prager Elastoplasticity for Sand Animation"
//   - Jiang et al. 2016 "The Material Point Method for Simulating Continuum
//     Materials"
// =============================================================================

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

// Use unified math types from Core
#include "WulfNet/Core/Math/MathTypes.h"

namespace WulfNet {

// =============================================================================
// Material Types for MPM constitutive models
// =============================================================================

enum class MPMMaterialType : uint32_t {
    NeoHookean = 0,     // Elastic solid (rubber, flesh)
    DruckerPrager = 1,  // Granular material (sand, mud, soil)
    Snow = 2,           // Disney snow model with hardening
    ViscousFluid = 3,   // Newtonian viscous fluid
    Custom = 255
};

// Mat3 is now defined in Core/Math/MathTypes.h

// =============================================================================
// Singular Value Decomposition (3x3 — Jacobi iteration)
// Used for polar decomposition F = R * S and return-mapping in plasticity
// =============================================================================

struct SVDResult {
    Mat3 U;     // Left orthogonal
    float sigma[3]; // Singular values
    Mat3 V;     // Right orthogonal
};

// Forward declaration (implemented in .cpp)
SVDResult ComputeSVD3x3(const Mat3& A);

// =============================================================================
// MPM Material Parameters
// =============================================================================

struct MPMMaterialParams {
    MPMMaterialType type = MPMMaterialType::NeoHookean;

    // Elastic parameters (Lamé)
    float youngsModulus = 1.4e5f;   // Pa — rubber: ~1e5, sand: ~1e6, snow: ~1e4
    float poissonsRatio = 0.2f;     // Dimensionless — typically 0.2–0.4

    // Derived Lamé constants (computed from E and ν)
    float mu = 0.0f;                // Shear modulus: E / (2(1+ν))
    float lambda = 0.0f;            // First Lamé parameter: Eν / ((1+ν)(1-2ν))

    // Drucker-Prager specific
    float frictionAngle = 30.0f;    // Degrees — sand ~30°, mud ~15°
    float cohesion = 0.0f;          // Pa — sand ~0, mud ~500+
    float dilatancyAngle = 0.0f;    // Degrees — typically 0 for sand

    // Snow specific (Stomakhin et al. 2013)
    float criticalCompression = 2.5e-2f;    // θ_c: ~0.025
    float criticalStretch = 7.5e-3f;        // θ_s: ~0.0075
    float hardeningCoefficient = 10.0f;     // ξ: ~10

    // Density
    float density = 1000.0f;        // kg/m³

    // Viscosity (for ViscousFluid type)
    float viscosity = 0.001f;       // Pa·s

    // Compute Lamé constants from E and ν
    void ComputeLameConstants() {
        mu = youngsModulus / (2.0f * (1.0f + poissonsRatio));
        lambda = youngsModulus * poissonsRatio /
                 ((1.0f + poissonsRatio) * (1.0f - 2.0f * poissonsRatio));
    }

    // =========================================================================
    // Preset Factory Methods
    // =========================================================================

    static MPMMaterialParams Rubber() {
        MPMMaterialParams p;
        p.type = MPMMaterialType::NeoHookean;
        p.youngsModulus = 1.0e5f;
        p.poissonsRatio = 0.45f;
        p.density = 1100.0f;
        p.ComputeLameConstants();
        return p;
    }

    static MPMMaterialParams Flesh() {
        MPMMaterialParams p;
        p.type = MPMMaterialType::NeoHookean;
        p.youngsModulus = 5.0e4f;
        p.poissonsRatio = 0.49f;    // Nearly incompressible
        p.density = 1050.0f;
        p.ComputeLameConstants();
        return p;
    }

    static MPMMaterialParams Sand() {
        MPMMaterialParams p;
        p.type = MPMMaterialType::DruckerPrager;
        p.youngsModulus = 3.537e5f;
        p.poissonsRatio = 0.3f;
        p.frictionAngle = 30.0f;
        p.cohesion = 0.0f;
        p.dilatancyAngle = 0.0f;
        p.density = 1600.0f;
        p.ComputeLameConstants();
        return p;
    }

    static MPMMaterialParams WetMud() {
        MPMMaterialParams p;
        p.type = MPMMaterialType::DruckerPrager;
        p.youngsModulus = 1.0e5f;
        p.poissonsRatio = 0.35f;
        p.frictionAngle = 15.0f;
        p.cohesion = 500.0f;           // Non-zero cohesion
        p.dilatancyAngle = 0.0f;
        p.density = 1800.0f;
        p.ComputeLameConstants();
        return p;
    }

    static MPMMaterialParams DrySoil() {
        MPMMaterialParams p;
        p.type = MPMMaterialType::DruckerPrager;
        p.youngsModulus = 8.0e5f;
        p.poissonsRatio = 0.25f;
        p.frictionAngle = 35.0f;
        p.cohesion = 200.0f;
        p.dilatancyAngle = 5.0f;
        p.density = 1500.0f;
        p.ComputeLameConstants();
        return p;
    }

    static MPMMaterialParams Snow() {
        MPMMaterialParams p;
        p.type = MPMMaterialType::Snow;
        p.youngsModulus = 1.4e5f;
        p.poissonsRatio = 0.2f;
        p.criticalCompression = 2.5e-2f;
        p.criticalStretch = 7.5e-3f;
        p.hardeningCoefficient = 10.0f;
        p.density = 400.0f;
        p.ComputeLameConstants();
        return p;
    }

    static MPMMaterialParams Ice() {
        MPMMaterialParams p;
        p.type = MPMMaterialType::Snow;
        p.youngsModulus = 9.0e6f;       // Much stiffer
        p.poissonsRatio = 0.3f;
        p.criticalCompression = 1.0e-2f;
        p.criticalStretch = 2.0e-3f;
        p.hardeningCoefficient = 15.0f;
        p.density = 917.0f;
        p.ComputeLameConstants();
        return p;
    }

    static MPMMaterialParams ViscousFluid(float viscosity = 0.001f, float density = 1000.0f) {
        MPMMaterialParams p;
        p.type = MPMMaterialType::ViscousFluid;
        p.youngsModulus = 0.0f;
        p.poissonsRatio = 0.0f;
        p.viscosity = viscosity;
        p.density = density;
        p.mu = 0.0f;
        p.lambda = 0.0f;
        return p;
    }
};

// =============================================================================
// MPM Particle (extended with deformation gradient)
// =============================================================================

struct alignas(16) MPMParticle {
    // Position (12 bytes + pad)
    float x, y, z;
    float mass;

    // Velocity (12 bytes + pad)
    float vx, vy, vz;
    float volume0;              // Initial volume

    // Deformation gradient F (36 bytes)
    Mat3 F;                     // Full 3x3 deformation gradient

    // Plastic deformation (for snow/Drucker-Prager)
    Mat3 Fp;                    // Plastic part of deformation gradient

    // APIC affine momentum matrix C (36 bytes)
    Mat3 C;

    // Material
    uint32_t materialId;
    uint32_t flags;
    float Jp;                   // Determinant of plastic deformation gradient
    float padding;
};

// =============================================================================
// Constitutive Model Interface
// =============================================================================

class ConstitutiveModel {
public:
    virtual ~ConstitutiveModel() = default;

    // Compute the Kirchhoff stress tensor τ = P * Fᵀ from the deformation gradient
    // This is the primary output used in P2G transfer
    virtual Mat3 ComputeStress(const MPMParticle& particle,
                               const MPMMaterialParams& params) const = 0;

    // Apply return mapping (plasticity correction) to the deformation gradient
    // Modifies particle.F and particle.Fp in-place
    virtual void ProjectDeformation(MPMParticle& particle,
                                    const MPMMaterialParams& params) const = 0;
};

// =============================================================================
// Neo-Hookean Constitutive Model (Elastic)
// =============================================================================
// Ψ(F) = μ/2 (tr(FᵀF) - d) - μ ln(J) + λ/2 (ln(J))²
// P(F) = μ(F - F⁻ᵀ) + λ ln(J) F⁻ᵀ
// =============================================================================

class NeoHookeanModel : public ConstitutiveModel {
public:
    Mat3 ComputeStress(const MPMParticle& particle,
                       const MPMMaterialParams& params) const override;

    void ProjectDeformation(MPMParticle& particle,
                            const MPMMaterialParams& params) const override;
};

// =============================================================================
// Drucker-Prager Constitutive Model (Sand/Mud/Soil)
// =============================================================================
// Extended Drucker-Prager with:
//   - Elastic trial step (Neo-Hookean)
//   - Yield function: f(σ) = ||dev(σ)|| + α tr(σ)
//   - Return mapping via SVD projection
//
// Reference: Klár et al. 2016
// =============================================================================

class DruckerPragerModel : public ConstitutiveModel {
public:
    Mat3 ComputeStress(const MPMParticle& particle,
                       const MPMMaterialParams& params) const override;

    void ProjectDeformation(MPMParticle& particle,
                            const MPMMaterialParams& params) const override;
};

// =============================================================================
// Disney Snow Constitutive Model
// =============================================================================
// Stomakhin et al. 2013:
//   - Multiplicative decomposition: F = Fe * Fp
//   - Hardening: μ̂ = μ₀ e^(ξ(1-Jp)), λ̂ = λ₀ e^(ξ(1-Jp))
//   - Clamped SVD return mapping for plasticity
// =============================================================================

class SnowModel : public ConstitutiveModel {
public:
    Mat3 ComputeStress(const MPMParticle& particle,
                       const MPMMaterialParams& params) const override;

    void ProjectDeformation(MPMParticle& particle,
                            const MPMMaterialParams& params) const override;
};

// =============================================================================
// Viscous Fluid Constitutive Model
// =============================================================================
// Simple Newtonian fluid: σ = -pI + μ(∇v + ∇vᵀ)
// For MPM: We use a Neo-Hookean model with very low stiffness + volume preservation
// =============================================================================

class ViscousFluidModel : public ConstitutiveModel {
public:
    Mat3 ComputeStress(const MPMParticle& particle,
                       const MPMMaterialParams& params) const override;

    void ProjectDeformation(MPMParticle& particle,
                            const MPMMaterialParams& params) const override;
};

// =============================================================================
// Factory Function
// =============================================================================

/// Creates the appropriate constitutive model for the given material type
const ConstitutiveModel* GetConstitutiveModel(MPMMaterialType type);

} // namespace WulfNet
