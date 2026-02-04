// =============================================================================
// WulfNet Engine - Joint Constraints Implementation
// =============================================================================

#include "Joint.h"
#include "core/math/Math.h"
#include <cmath>

namespace WulfNet {

void JointSolverPGS::solveDistanceJoints(std::vector<DistanceJoint*>& joints,
                                         const std::function<RigidBody*(u32)>& getBody,
                                         f32 dt) const {
    if (dt <= 0.0f || joints.empty()) return;

    for (DistanceJoint* jointPtr : joints) {
        if (!jointPtr) continue;
        DistanceJoint& joint = *jointPtr;
        RigidBody* bodyA = getBody(joint.bodyIdA);
        RigidBody* bodyB = getBody(joint.bodyIdB);
        if (!bodyA || !bodyB) continue;

        const Transform& tA = bodyA->getTransform();
        const Transform& tB = bodyB->getTransform();

        Vec3 worldAnchorA = tA.transformPoint(joint.localAnchorA);
        Vec3 worldAnchorB = tB.transformPoint(joint.localAnchorB);

        Vec3 delta = worldAnchorB - worldAnchorA;
        f32 distSq = delta.lengthSq();
        if (distSq <= Math::EPSILON) continue;

        f32 dist = std::sqrt(distSq);
        Vec3 n = delta / dist;

        Vec3 ra = worldAnchorA - bodyA->getPosition();
        Vec3 rb = worldAnchorB - bodyB->getPosition();

        Vec3 vA = bodyA->getLinearVelocity() + bodyA->getAngularVelocity().cross(ra);
        Vec3 vB = bodyB->getLinearVelocity() + bodyB->getAngularVelocity().cross(rb);
        Vec3 relVel = vB - vA;
        f32 velAlong = relVel.dot(n);

        f32 C = dist - joint.restLength;
        f32 bias = (joint.stiffness * C) / Math::max(dt, Math::EPSILON);

        f32 dampingTerm = joint.damping * velAlong;

        const f32 invMassA = bodyA->getInverseMass();
        const f32 invMassB = bodyB->getInverseMass();
        const Mat4& invInertiaA = bodyA->getInvInertiaTensorWorld();
        const Mat4& invInertiaB = bodyB->getInvInertiaTensorWorld();

        Vec3 rnA = ra.cross(n);
        Vec3 rnB = rb.cross(n);
        f32 denom = invMassA + invMassB +
                    n.dot(invInertiaA.transformDirection(rnA).cross(ra) +
                          invInertiaB.transformDirection(rnB).cross(rb));

        if (denom <= Math::EPSILON) continue;

        f32 lambda = -(velAlong + bias + dampingTerm) / denom;
        joint.accumulatedImpulse += lambda;

        Vec3 impulse = n * lambda;
        bodyA->applyImpulseAtPoint(-impulse, worldAnchorA);
        bodyB->applyImpulseAtPoint(impulse, worldAnchorB);
    }
}

void JointSolverPGS::solveBallJoints(std::vector<BallJoint*>& joints,
                                     const std::function<RigidBody*(u32)>& getBody,
                                     f32 dt) const {
    if (dt <= 0.0f || joints.empty()) return;

    const Vec3 axes[3] = { Vec3::unitX(), Vec3::unitY(), Vec3::unitZ() };

    for (BallJoint* jointPtr : joints) {
        if (!jointPtr) continue;
        BallJoint& joint = *jointPtr;
        RigidBody* bodyA = getBody(joint.bodyIdA);
        RigidBody* bodyB = getBody(joint.bodyIdB);
        if (!bodyA || !bodyB) continue;

        const Transform& tA = bodyA->getTransform();
        const Transform& tB = bodyB->getTransform();

        Vec3 worldAnchorA = tA.transformPoint(joint.localAnchorA);
        Vec3 worldAnchorB = tB.transformPoint(joint.localAnchorB);

        Vec3 ra = worldAnchorA - bodyA->getPosition();
        Vec3 rb = worldAnchorB - bodyB->getPosition();

        Vec3 vA = bodyA->getLinearVelocity() + bodyA->getAngularVelocity().cross(ra);
        Vec3 vB = bodyB->getLinearVelocity() + bodyB->getAngularVelocity().cross(rb);
        Vec3 relVel = vB - vA;

        Vec3 delta = worldAnchorB - worldAnchorA;

        const f32 invMassA = bodyA->getInverseMass();
        const f32 invMassB = bodyB->getInverseMass();
        const Mat4& invInertiaA = bodyA->getInvInertiaTensorWorld();
        const Mat4& invInertiaB = bodyB->getInvInertiaTensorWorld();

        for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
            const Vec3& n = axes[axisIndex];
            f32 C = delta.dot(n);
            f32 velAlong = relVel.dot(n);

            f32 bias = (joint.stiffness * C) / Math::max(dt, Math::EPSILON);
            f32 dampingTerm = joint.damping * velAlong;

            Vec3 rnA = ra.cross(n);
            Vec3 rnB = rb.cross(n);
            f32 denom = invMassA + invMassB +
                        n.dot(invInertiaA.transformDirection(rnA).cross(ra) +
                              invInertiaB.transformDirection(rnB).cross(rb));

            if (denom <= Math::EPSILON) continue;

            f32 lambda = -(velAlong + bias + dampingTerm) / denom;
            joint.accumulatedImpulse[axisIndex] += lambda;

            Vec3 impulse = n * lambda;
            bodyA->applyImpulseAtPoint(-impulse, worldAnchorA);
            bodyB->applyImpulseAtPoint(impulse, worldAnchorB);
        }
    }
}

void JointSolverPGS::solveFixedJoints(std::vector<FixedJoint*>& joints,
                                      const std::function<RigidBody*(u32)>& getBody,
                                      f32 dt) const {
    if (dt <= 0.0f || joints.empty()) return;

    const Vec3 axes[3] = { Vec3::unitX(), Vec3::unitY(), Vec3::unitZ() };

    for (FixedJoint* jointPtr : joints) {
        if (!jointPtr) continue;
        FixedJoint& joint = *jointPtr;
        RigidBody* bodyA = getBody(joint.bodyIdA);
        RigidBody* bodyB = getBody(joint.bodyIdB);
        if (!bodyA || !bodyB) continue;

        const Transform& tA = bodyA->getTransform();
        const Transform& tB = bodyB->getTransform();

        Vec3 worldAnchorA = tA.transformPoint(joint.localAnchorA);
        Vec3 worldAnchorB = tB.transformPoint(joint.localAnchorB);

        Vec3 ra = worldAnchorA - bodyA->getPosition();
        Vec3 rb = worldAnchorB - bodyB->getPosition();

        Vec3 vA = bodyA->getLinearVelocity() + bodyA->getAngularVelocity().cross(ra);
        Vec3 vB = bodyB->getLinearVelocity() + bodyB->getAngularVelocity().cross(rb);
        Vec3 relVel = vB - vA;

        Vec3 delta = worldAnchorB - worldAnchorA;

        const f32 invMassA = bodyA->getInverseMass();
        const f32 invMassB = bodyB->getInverseMass();
        const Mat4& invInertiaA = bodyA->getInvInertiaTensorWorld();
        const Mat4& invInertiaB = bodyB->getInvInertiaTensorWorld();

        for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
            const Vec3& n = axes[axisIndex];
            f32 C = delta.dot(n);
            f32 velAlong = relVel.dot(n);

            f32 bias = (joint.stiffness * C) / Math::max(dt, Math::EPSILON);
            f32 dampingTerm = joint.damping * velAlong;

            Vec3 rnA = ra.cross(n);
            Vec3 rnB = rb.cross(n);
            f32 denom = invMassA + invMassB +
                        n.dot(invInertiaA.transformDirection(rnA).cross(ra) +
                              invInertiaB.transformDirection(rnB).cross(rb));

            if (denom <= Math::EPSILON) continue;

            f32 lambda = -(velAlong + bias + dampingTerm) / denom;
            joint.accumulatedLinearImpulse[axisIndex] += lambda;

            Vec3 impulse = n * lambda;
            bodyA->applyImpulseAtPoint(-impulse, worldAnchorA);
            bodyB->applyImpulseAtPoint(impulse, worldAnchorB);
        }

        Quat qA = bodyA->getOrientation();
        Quat qB = bodyB->getOrientation();
        Quat targetQ = qA * joint.restRelativeRotation;
        Quat error = qB * targetQ.conjugate();
        if (error.w < 0.0f) {
            error = Quat(-error.x, -error.y, -error.z, -error.w);
        }

        Vec3 axis;
        f32 angle = 0.0f;
        error.toAxisAngle(axis, angle);

        if (angle > Math::EPSILON) {
            Vec3 errorVec = axis * angle;
            Vec3 relAngularVel = bodyB->getAngularVelocity() - bodyA->getAngularVelocity();

            for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
                const Vec3& n = axes[axisIndex];
                f32 C = errorVec.dot(n);
                f32 velAlong = relAngularVel.dot(n);
                f32 bias = (joint.stiffness * C) / Math::max(dt, Math::EPSILON);
                f32 dampingTerm = joint.damping * velAlong;

                f32 denom = n.dot(invInertiaA.transformDirection(n) +
                                  invInertiaB.transformDirection(n));
                if (denom <= Math::EPSILON) continue;

                f32 lambda = -(velAlong + bias + dampingTerm) / denom;
                Vec3 angularImpulse = n * lambda;
                joint.accumulatedAngularImpulse += angularImpulse;

                bodyA->applyAngularImpulse(-angularImpulse);
                bodyB->applyAngularImpulse(angularImpulse);
            }

            f32 correctionFactor = Math::clamp(joint.stiffness * 0.2f, 0.0f, 1.0f);
            if (correctionFactor > 0.0f) {
                f32 half = correctionFactor * 0.5f;
                if (bodyA->isDynamic()) {
                    Quat corrA = Quat::fromAxisAngle(axis, angle * half);
                    bodyA->setOrientation((corrA * qA).normalized());
                }
                if (bodyB->isDynamic()) {
                    Quat corrB = Quat::fromAxisAngle(axis, -angle * half);
                    bodyB->setOrientation((corrB * qB).normalized());
                }
            }
        }
    }
}

void JointSolverPGS::solveHingeJoints(std::vector<HingeJoint*>& joints,
                                      const std::function<RigidBody*(u32)>& getBody,
                                      f32 dt) const {
    if (dt <= 0.0f || joints.empty()) return;

    const Vec3 axes[3] = { Vec3::unitX(), Vec3::unitY(), Vec3::unitZ() };

    for (HingeJoint* jointPtr : joints) {
        if (!jointPtr) continue;
        HingeJoint& joint = *jointPtr;
        RigidBody* bodyA = getBody(joint.bodyIdA);
        RigidBody* bodyB = getBody(joint.bodyIdB);
        if (!bodyA || !bodyB) continue;

        const Transform& tA = bodyA->getTransform();
        const Transform& tB = bodyB->getTransform();

        Vec3 worldAnchorA = tA.transformPoint(joint.localAnchorA);
        Vec3 worldAnchorB = tB.transformPoint(joint.localAnchorB);

        Vec3 ra = worldAnchorA - bodyA->getPosition();
        Vec3 rb = worldAnchorB - bodyB->getPosition();

        Vec3 vA = bodyA->getLinearVelocity() + bodyA->getAngularVelocity().cross(ra);
        Vec3 vB = bodyB->getLinearVelocity() + bodyB->getAngularVelocity().cross(rb);
        Vec3 relVel = vB - vA;

        Vec3 delta = worldAnchorB - worldAnchorA;

        const f32 invMassA = bodyA->getInverseMass();
        const f32 invMassB = bodyB->getInverseMass();
        const Mat4& invInertiaA = bodyA->getInvInertiaTensorWorld();
        const Mat4& invInertiaB = bodyB->getInvInertiaTensorWorld();

        for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
            const Vec3& n = axes[axisIndex];
            f32 C = delta.dot(n);
            f32 velAlong = relVel.dot(n);

            f32 bias = (joint.stiffness * C) / Math::max(dt, Math::EPSILON);
            f32 dampingTerm = joint.damping * velAlong;

            Vec3 rnA = ra.cross(n);
            Vec3 rnB = rb.cross(n);
            f32 denom = invMassA + invMassB +
                        n.dot(invInertiaA.transformDirection(rnA).cross(ra) +
                              invInertiaB.transformDirection(rnB).cross(rb));

            if (denom <= Math::EPSILON) continue;

            f32 lambda = -(velAlong + bias + dampingTerm) / denom;
            joint.accumulatedLinearImpulse[axisIndex] += lambda;

            Vec3 impulse = n * lambda;
            bodyA->applyImpulseAtPoint(-impulse, worldAnchorA);
            bodyB->applyImpulseAtPoint(impulse, worldAnchorB);
        }

        Vec3 axisAWorld = (bodyA->getOrientation() * joint.localAxisA).normalized();
        Vec3 axisBWorld = (bodyB->getOrientation() * joint.localAxisB).normalized();
        if (axisAWorld.dot(axisBWorld) < 0.0f) {
            axisBWorld = -axisBWorld;
        }

        Vec3 errorVec = axisAWorld.cross(axisBWorld);
        if (errorVec.lengthSq() <= Math::EPSILON) {
            continue;
        }

        Vec3 perp1;
        if (Math::abs(axisAWorld.x) > Math::abs(axisAWorld.z)) {
            perp1 = Vec3(-axisAWorld.y, axisAWorld.x, 0.0f);
        } else {
            perp1 = Vec3(0.0f, -axisAWorld.z, axisAWorld.y);
        }
        perp1 = perp1.normalized();
        Vec3 perp2 = axisAWorld.cross(perp1).normalized();

        Vec3 relAngularVel = bodyB->getAngularVelocity() - bodyA->getAngularVelocity();

        const Vec3 angularAxes[2] = { perp1, perp2 };
        for (int i = 0; i < 2; ++i) {
            const Vec3& n = angularAxes[i];
            f32 C = errorVec.dot(n);
            f32 velAlong = relAngularVel.dot(n);
                f32 bias = (joint.stiffness * C) / Math::max(dt, Math::EPSILON);
            f32 dampingTerm = joint.damping * velAlong;

            f32 denom = n.dot(invInertiaA.transformDirection(n) +
                              invInertiaB.transformDirection(n));
            if (denom <= Math::EPSILON) continue;

            f32 lambda = -(velAlong + bias + dampingTerm) / denom;
            Vec3 angularImpulse = n * lambda;
            joint.accumulatedAngularImpulse += angularImpulse;

            bodyA->applyAngularImpulse(-angularImpulse);
            bodyB->applyAngularImpulse(angularImpulse);
        }

            f32 errorLen = errorVec.length();
            f32 correctionAngle = std::asin(Math::clamp(errorLen, 0.0f, 1.0f));
            f32 correctionFactor = Math::clamp(joint.stiffness * 0.2f, 0.0f, 1.0f);
            if (correctionAngle > Math::EPSILON && correctionFactor > 0.0f) {
                Vec3 errorAxis = errorVec / errorLen;
                f32 half = correctionFactor * 0.5f;
                if (bodyA->isDynamic()) {
                    Quat corrA = Quat::fromAxisAngle(errorAxis, correctionAngle * half);
                    bodyA->setOrientation((corrA * bodyA->getOrientation()).normalized());
                }
                if (bodyB->isDynamic()) {
                    Quat corrB = Quat::fromAxisAngle(errorAxis, -correctionAngle * half);
                    bodyB->setOrientation((corrB * bodyB->getOrientation()).normalized());
                }
            }
    }
}

} // namespace WulfNet
