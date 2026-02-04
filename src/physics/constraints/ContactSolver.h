// =============================================================================
// WulfNet Engine - Contact Solver (PGS)
// =============================================================================
// Minimal projected Gauss-Seidel contact solver.
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/math/Vec3.h"
#include "core/math/Math.h"
#include "physics/dynamics/RigidBody.h"
#include "physics/collision/CollisionTypes.h"
#include <functional>
#include <vector>

namespace WulfNet {

struct ContactSolverConfig {
    u32 iterations = 8;
    f32 baumgarte = 0.05f;     // Positional correction factor
    f32 penetrationSlop = 0.01f;
    f32 restitutionVelocityThreshold = 0.0f;
    f32 maxBias = 5.0f;
    f32 positionCorrectionPercent = 0.1f;
    f32 maxPositionCorrection = 0.2f;
};

class ContactSolverPGS {
public:
    explicit ContactSolverPGS(const ContactSolverConfig& config = {})
        : m_config(config) {}

    void solve(std::vector<ContactManifold>& manifolds,
               const std::function<RigidBody*(u32)>& getBody,
               f32 dt) const;

private:
    ContactSolverConfig m_config;
};

} // namespace WulfNet
