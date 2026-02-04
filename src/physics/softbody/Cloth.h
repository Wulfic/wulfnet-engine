// =============================================================================
// WulfNet Engine - XPBD Cloth Simulation (CPU)
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/math/Vec3.h"
#include <vector>

namespace WulfNet {

struct ClothParticle {
    Vec3 position = Vec3::zero();
    Vec3 prevPosition = Vec3::zero();
    Vec3 velocity = Vec3::zero();
    f32 invMass = 1.0f;  // 0 = pinned
};

struct ClothDistanceConstraint {
    u32 particleA = 0;
    u32 particleB = 0;
    f32 restLength = 0.0f;
    f32 compliance = 0.0f;  // XPBD compliance (0 = rigid)
    f32 lambda = 0.0f;      // Lagrange multiplier accumulator
};

class ClothSimulation {
public:
    ClothSimulation() = default;

    void clear();

    // Build a grid of particles with structural constraints
    void initializeGrid(u32 width, u32 height, f32 spacing, const Vec3& origin,
                        f32 invMass = 1.0f, f32 compliance = 0.0f,
                        bool addShear = false, bool addBend = false,
                        f32 shearCompliance = 0.0f, f32 bendCompliance = 0.0f);

    void setGravity(const Vec3& gravity) { m_gravity = gravity; }

    void setParticleInvMass(u32 index, f32 invMass);
    void setParticlePosition(u32 index, const Vec3& position);

    void step(f32 dt, u32 iterations);

    const std::vector<ClothParticle>& getParticles() const { return m_particles; }
    const std::vector<ClothDistanceConstraint>& getConstraints() const { return m_constraints; }

private:
    void solveDistanceConstraints(f32 dt);
    void addDistanceConstraint(u32 a, u32 b, f32 restLength, f32 compliance);

    std::vector<ClothParticle> m_particles;
    std::vector<ClothDistanceConstraint> m_constraints;
    Vec3 m_gravity = Vec3(0.0f, -9.81f, 0.0f);
};

} // namespace WulfNet
