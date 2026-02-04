// =============================================================================
// WulfNet Engine - Joint Constraints
// =============================================================================
// Basic joint types (Distance) and solver.
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/math/Vec3.h"
#include "core/math/Quat.h"
#include "physics/dynamics/RigidBody.h"
#include <functional>
#include <vector>

namespace WulfNet {

struct DistanceJointConfig {
    u32 bodyIdA = 0;
    u32 bodyIdB = 0;
    Vec3 localAnchorA = Vec3::zero();
    Vec3 localAnchorB = Vec3::zero();
    f32 restLength = -1.0f; // If < 0, compute from current anchors.
    f32 stiffness = 0.8f;   // 0..1 (bias strength)
    f32 damping = 0.1f;     // 0..1 (velocity damping)
};

struct DistanceJoint {
    u32 id = 0;
    u32 bodyIdA = 0;
    u32 bodyIdB = 0;
    Vec3 localAnchorA = Vec3::zero();
    Vec3 localAnchorB = Vec3::zero();
    f32 restLength = 0.0f;
    f32 stiffness = 0.8f;
    f32 damping = 0.1f;
    f32 accumulatedImpulse = 0.0f;
};

struct BallJointConfig {
    u32 bodyIdA = 0;
    u32 bodyIdB = 0;
    Vec3 localAnchorA = Vec3::zero();
    Vec3 localAnchorB = Vec3::zero();
    f32 stiffness = 0.8f; // 0..1 (bias strength)
    f32 damping = 0.1f;   // 0..1 (velocity damping)
};

struct BallJoint {
    u32 id = 0;
    u32 bodyIdA = 0;
    u32 bodyIdB = 0;
    Vec3 localAnchorA = Vec3::zero();
    Vec3 localAnchorB = Vec3::zero();
    f32 stiffness = 0.8f;
    f32 damping = 0.1f;
    Vec3 accumulatedImpulse = Vec3::zero();
};

struct FixedJointConfig {
    u32 bodyIdA = 0;
    u32 bodyIdB = 0;
    Vec3 localAnchorA = Vec3::zero();
    Vec3 localAnchorB = Vec3::zero();
    Quat restRelativeRotation = Quat::identity();
    bool computeRestRotation = true;
    f32 stiffness = 0.8f; // 0..1 (bias strength)
    f32 damping = 0.1f;   // 0..1 (velocity damping)
};

struct FixedJoint {
    u32 id = 0;
    u32 bodyIdA = 0;
    u32 bodyIdB = 0;
    Vec3 localAnchorA = Vec3::zero();
    Vec3 localAnchorB = Vec3::zero();
    Quat restRelativeRotation = Quat::identity();
    f32 stiffness = 0.8f;
    f32 damping = 0.1f;
    Vec3 accumulatedLinearImpulse = Vec3::zero();
    Vec3 accumulatedAngularImpulse = Vec3::zero();
};

struct HingeJointConfig {
    u32 bodyIdA = 0;
    u32 bodyIdB = 0;
    Vec3 localAnchorA = Vec3::zero();
    Vec3 localAnchorB = Vec3::zero();
    Vec3 localAxisA = Vec3::unitY();
    Vec3 localAxisB = Vec3::unitY();
    f32 stiffness = 0.8f; // 0..1 (bias strength)
    f32 damping = 0.1f;   // 0..1 (velocity damping)
};

struct HingeJoint {
    u32 id = 0;
    u32 bodyIdA = 0;
    u32 bodyIdB = 0;
    Vec3 localAnchorA = Vec3::zero();
    Vec3 localAnchorB = Vec3::zero();
    Vec3 localAxisA = Vec3::unitY();
    Vec3 localAxisB = Vec3::unitY();
    f32 stiffness = 0.8f;
    f32 damping = 0.1f;
    Vec3 accumulatedLinearImpulse = Vec3::zero();
    Vec3 accumulatedAngularImpulse = Vec3::zero();
};

class JointSolverPGS {
public:
    void solveDistanceJoints(std::vector<DistanceJoint*>& joints,
                             const std::function<RigidBody*(u32)>& getBody,
                             f32 dt) const;

    void solveBallJoints(std::vector<BallJoint*>& joints,
                         const std::function<RigidBody*(u32)>& getBody,
                         f32 dt) const;

    void solveFixedJoints(std::vector<FixedJoint*>& joints,
                          const std::function<RigidBody*(u32)>& getBody,
                          f32 dt) const;

    void solveHingeJoints(std::vector<HingeJoint*>& joints,
                          const std::function<RigidBody*(u32)>& getBody,
                          f32 dt) const;
};

} // namespace WulfNet
