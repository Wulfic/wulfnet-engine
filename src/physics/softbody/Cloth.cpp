// =============================================================================
// WulfNet Engine - XPBD Cloth Simulation (CPU)
// =============================================================================

#include "Cloth.h"
#include "core/Log.h"
#include "core/Assert.h"
#include <cmath>

namespace WulfNet {

void ClothSimulation::clear() {
    m_particles.clear();
    m_constraints.clear();
}

void ClothSimulation::initializeGrid(u32 width, u32 height, f32 spacing, const Vec3& origin,
                                     f32 invMass, f32 compliance,
                                     bool addShear, bool addBend,
                                     f32 shearCompliance, f32 bendCompliance) {
    WULFNET_ASSERT_MSG(width > 0 && height > 0, "Cloth grid dimensions must be positive");

    clear();
    m_particles.reserve(static_cast<size_t>(width) * static_cast<size_t>(height));

    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            Vec3 pos = origin + Vec3(static_cast<f32>(x) * spacing,
                                     0.0f,
                                     static_cast<f32>(y) * spacing);
            ClothParticle p;
            p.position = pos;
            p.prevPosition = pos;
            p.velocity = Vec3::zero();
            p.invMass = invMass;
            m_particles.push_back(p);
        }
    }

    auto index = [width](u32 x, u32 y) { return y * width + x; };

    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            if (x + 1 < width) {
                addDistanceConstraint(index(x, y), index(x + 1, y), spacing, compliance);
            }
            if (y + 1 < height) {
                addDistanceConstraint(index(x, y), index(x, y + 1), spacing, compliance);
            }

            if (addShear) {
                if (x + 1 < width && y + 1 < height) {
                    addDistanceConstraint(index(x, y), index(x + 1, y + 1),
                        spacing * std::sqrt(2.0f), shearCompliance);
                }
                if (x + 1 < width && y > 0) {
                    addDistanceConstraint(index(x, y), index(x + 1, y - 1),
                        spacing * std::sqrt(2.0f), shearCompliance);
                }
            }

            if (addBend) {
                if (x + 2 < width) {
                    addDistanceConstraint(index(x, y), index(x + 2, y),
                        spacing * 2.0f, bendCompliance);
                }
                if (y + 2 < height) {
                    addDistanceConstraint(index(x, y), index(x, y + 2),
                        spacing * 2.0f, bendCompliance);
                }
            }
        }
    }

    WULFNET_LOG_DEBUG("ClothSimulation", "Initialized grid {}x{} (particles={}, constraints={}, shear={}, bend={})",
        width, height, m_particles.size(), m_constraints.size(), addShear, addBend);
}

void ClothSimulation::setParticleInvMass(u32 index, f32 invMass) {
    if (index >= m_particles.size()) {
        WULFNET_LOG_WARN("ClothSimulation", "Invalid particle index {}", index);
        return;
    }
    m_particles[index].invMass = invMass;
}

void ClothSimulation::setParticlePosition(u32 index, const Vec3& position) {
    if (index >= m_particles.size()) {
        WULFNET_LOG_WARN("ClothSimulation", "Invalid particle index {}", index);
        return;
    }
    m_particles[index].position = position;
    m_particles[index].prevPosition = position;
}

void ClothSimulation::step(f32 dt, u32 iterations) {
    if (dt <= 0.0f || m_particles.empty()) {
        return;
    }

    // Predict positions
    for (ClothParticle& p : m_particles) {
        if (p.invMass == 0.0f) {
            p.prevPosition = p.position;
            p.velocity = Vec3::zero();
            continue;
        }

        p.velocity += m_gravity * dt;
        p.prevPosition = p.position;
        p.position += p.velocity * dt;
    }

    // Reset lambdas each step for stability
    for (ClothDistanceConstraint& c : m_constraints) {
        c.lambda = 0.0f;
    }

    // Solve constraints
    for (u32 i = 0; i < iterations; ++i) {
        solveDistanceConstraints(dt);
    }

    // Update velocities from position change
    const f32 invDt = 1.0f / dt;
    for (ClothParticle& p : m_particles) {
        if (p.invMass == 0.0f) {
            p.velocity = Vec3::zero();
            continue;
        }
        p.velocity = (p.position - p.prevPosition) * invDt;
    }
}

void ClothSimulation::solveDistanceConstraints(f32 dt) {
    const f32 eps = 1e-6f;

    for (ClothDistanceConstraint& c : m_constraints) {
        ClothParticle& a = m_particles[c.particleA];
        ClothParticle& b = m_particles[c.particleB];

        f32 wA = a.invMass;
        f32 wB = b.invMass;
        f32 wSum = wA + wB;
        if (wSum <= 0.0f) {
            continue;
        }

        Vec3 delta = a.position - b.position;
        f32 len = delta.length();
        if (len < eps) {
            continue;
        }

        Vec3 n = delta / len;
        f32 C = len - c.restLength;
        f32 alpha = c.compliance / (dt * dt);

        f32 dlambda = (-C - alpha * c.lambda) / (wSum + alpha);
        c.lambda += dlambda;

        Vec3 correction = n * dlambda;
        a.position += correction * wA;
        b.position -= correction * wB;
    }
}

void ClothSimulation::addDistanceConstraint(u32 a, u32 b, f32 restLength, f32 compliance) {
    ClothDistanceConstraint c;
    c.particleA = a;
    c.particleB = b;
    c.restLength = restLength;
    c.compliance = compliance;
    m_constraints.push_back(c);
}

} // namespace WulfNet
