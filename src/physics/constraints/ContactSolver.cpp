// =============================================================================
// WulfNet Engine - Contact Solver (PGS)
// =============================================================================

#include "ContactSolver.h"
#include <algorithm>
#include <cmath>

namespace WulfNet {

void ContactSolverPGS::solve(std::vector<ContactManifold>& manifolds,
                             const std::function<RigidBody*(u32)>& getBody,
                             f32 dt) const {
    if (dt <= 0.0f || manifolds.empty()) return;

    // Warm-start: apply cached impulses
    for (ContactManifold& manifold : manifolds) {
        RigidBody* bodyA = getBody(manifold.bodyIdA);
        RigidBody* bodyB = getBody(manifold.bodyIdB);
        if (!bodyA || !bodyB) continue;

        for (u32 i = 0; i < manifold.contactCount; ++i) {
            ContactPoint& cp = manifold.contacts[i];
            if (cp.normalImpulse <= 0.0f && Math::abs(cp.tangentImpulse) <= Math::EPSILON) {
                continue;
            }

            Vec3 impulse = cp.normal * cp.normalImpulse + cp.tangent * cp.tangentImpulse;
            if (impulse.lengthSq() <= Math::EPSILON) continue;

            bodyA->applyImpulseAtPoint(-impulse, cp.positionWorldA);
            bodyB->applyImpulseAtPoint(impulse, cp.positionWorldB);
        }
    }

    for (u32 iter = 0; iter < m_config.iterations; ++iter) {
        for (ContactManifold& manifold : manifolds) {
            RigidBody* bodyA = getBody(manifold.bodyIdA);
            RigidBody* bodyB = getBody(manifold.bodyIdB);
            if (!bodyA || !bodyB) continue;

            const f32 invMassA = bodyA->getInverseMass();
            const f32 invMassB = bodyB->getInverseMass();
            const Mat4& invInertiaA = bodyA->getInvInertiaTensorWorld();
            const Mat4& invInertiaB = bodyB->getInvInertiaTensorWorld();

            for (u32 i = 0; i < manifold.contactCount; ++i) {
                ContactPoint& cp = manifold.contacts[i];
                const Vec3& n = cp.normal;

                Vec3 ra = cp.positionWorldA - bodyA->getPosition();
                Vec3 rb = cp.positionWorldB - bodyB->getPosition();

                Vec3 velA = bodyA->getLinearVelocity() + bodyA->getAngularVelocity().cross(ra);
                Vec3 velB = bodyB->getLinearVelocity() + bodyB->getAngularVelocity().cross(rb);
                Vec3 relVel = velB - velA;

                f32 velAlongNormal = relVel.dot(n);

                f32 restitution = std::max(bodyA->getRestitution(), bodyB->getRestitution());
                f32 restitutionTerm = (velAlongNormal < -m_config.restitutionVelocityThreshold) ? restitution : 0.0f;

                f32 bias = 0.0f;
                if (cp.penetration > m_config.penetrationSlop) {
                    bias = (m_config.baumgarte / dt) * (cp.penetration - m_config.penetrationSlop);
                    bias = Math::min(bias, m_config.maxBias);
                }

                Vec3 rnA = ra.cross(n);
                Vec3 rnB = rb.cross(n);
                f32 denom = invMassA + invMassB +
                            n.dot(invInertiaA.transformDirection(rnA).cross(ra) +
                                  invInertiaB.transformDirection(rnB).cross(rb));

                if (denom <= Math::EPSILON) continue;

                f32 j = -(1.0f + restitutionTerm) * velAlongNormal - bias;
                j /= denom;

                f32 oldNormalImpulse = cp.normalImpulse;
                f32 newNormalImpulse = Math::max(oldNormalImpulse + j, 0.0f);
                f32 deltaNormalImpulse = newNormalImpulse - oldNormalImpulse;

                Vec3 impulse = n * deltaNormalImpulse;
                bodyA->applyImpulseAtPoint(-impulse, cp.positionWorldA);
                bodyB->applyImpulseAtPoint(impulse, cp.positionWorldB);

                // Update cached impulse
                cp.normalImpulse = newNormalImpulse;

                // Friction
                velA = bodyA->getLinearVelocity() + bodyA->getAngularVelocity().cross(ra);
                velB = bodyB->getLinearVelocity() + bodyB->getAngularVelocity().cross(rb);
                relVel = velB - velA;

                Vec3 tangent = relVel - n * relVel.dot(n);
                f32 tangentLenSq = tangent.lengthSq();
                if (tangentLenSq > Math::EPSILON) {
                    tangent /= std::sqrt(tangentLenSq);

                    Vec3 rtA = ra.cross(tangent);
                    Vec3 rtB = rb.cross(tangent);
                    f32 denomT = invMassA + invMassB +
                                 tangent.dot(invInertiaA.transformDirection(rtA).cross(ra) +
                                             invInertiaB.transformDirection(rtB).cross(rb));

                    if (denomT > Math::EPSILON) {
                        f32 jt = -relVel.dot(tangent) / denomT;
                        f32 friction = std::sqrt(bodyA->getFriction() * bodyB->getFriction());
                        f32 maxFriction = friction * newNormalImpulse;

                        f32 oldTangentImpulse = cp.tangentImpulse;
                        f32 newTangentImpulse = Math::clamp(oldTangentImpulse + jt, -maxFriction, maxFriction);
                        f32 deltaTangentImpulse = newTangentImpulse - oldTangentImpulse;

                        Vec3 frictionImpulse = tangent * deltaTangentImpulse;
                        bodyA->applyImpulseAtPoint(-frictionImpulse, cp.positionWorldA);
                        bodyB->applyImpulseAtPoint(frictionImpulse, cp.positionWorldB);

                        // Update cached tangent impulse and direction
                        cp.tangent = tangent;
                        cp.tangentImpulse = newTangentImpulse;
                    }
                }
            }
        }
    }

    // Post-solve restitution pass (ensure rebound for closing contacts)
    for (const ContactManifold& manifold : manifolds) {
        RigidBody* bodyA = getBody(manifold.bodyIdA);
        RigidBody* bodyB = getBody(manifold.bodyIdB);
        if (!bodyA || !bodyB) continue;

        const f32 invMassA = bodyA->getInverseMass();
        const f32 invMassB = bodyB->getInverseMass();
        const Mat4& invInertiaA = bodyA->getInvInertiaTensorWorld();
        const Mat4& invInertiaB = bodyB->getInvInertiaTensorWorld();

        const f32 restitution = std::max(bodyA->getRestitution(), bodyB->getRestitution());
        if (restitution <= 0.0f) continue;

        for (u32 i = 0; i < manifold.contactCount; ++i) {
            const ContactPoint& cp = manifold.contacts[i];
            const Vec3& n = cp.normal;

            Vec3 ra = cp.positionWorldA - bodyA->getPosition();
            Vec3 rb = cp.positionWorldB - bodyB->getPosition();

            Vec3 velA = bodyA->getLinearVelocity() + bodyA->getAngularVelocity().cross(ra);
            Vec3 velB = bodyB->getLinearVelocity() + bodyB->getAngularVelocity().cross(rb);
            Vec3 relVel = velB - velA;

            f32 velAlongNormal = relVel.dot(n);
            if (velAlongNormal > -m_config.restitutionVelocityThreshold) {
                continue; // Not closing fast enough to bounce
            }

            Vec3 rnA = ra.cross(n);
            Vec3 rnB = rb.cross(n);
            f32 denom = invMassA + invMassB +
                        n.dot(invInertiaA.transformDirection(rnA).cross(ra) +
                              invInertiaB.transformDirection(rnB).cross(rb));

            if (denom <= Math::EPSILON) continue;

            f32 j = -(1.0f + restitution) * velAlongNormal / denom;
            Vec3 impulse = n * j;
            bodyA->applyImpulseAtPoint(-impulse, cp.positionWorldA);
            bodyB->applyImpulseAtPoint(impulse, cp.positionWorldB);
        }
    }

    // Positional correction (post-step projection)
    for (const ContactManifold& manifold : manifolds) {
        RigidBody* bodyA = getBody(manifold.bodyIdA);
        RigidBody* bodyB = getBody(manifold.bodyIdB);
        if (!bodyA || !bodyB) continue;

        f32 invMassA = bodyA->getInverseMass();
        f32 invMassB = bodyB->getInverseMass();
        f32 invMassSum = invMassA + invMassB;
        if (invMassSum <= Math::EPSILON) continue;

        for (u32 i = 0; i < manifold.contactCount; ++i) {
            const ContactPoint& cp = manifold.contacts[i];
            if (cp.penetration <= m_config.penetrationSlop) continue;

            f32 correctionMag = (cp.penetration - m_config.penetrationSlop) * m_config.positionCorrectionPercent;
            correctionMag = Math::min(correctionMag, m_config.maxPositionCorrection);

            Vec3 correction = cp.normal * (correctionMag / invMassSum);
            bodyA->applyPositionCorrection(-correction * invMassA);
            bodyB->applyPositionCorrection(correction * invMassB);
        }
    }
}

} // namespace WulfNet
