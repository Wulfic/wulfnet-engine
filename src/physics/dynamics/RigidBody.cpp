// =============================================================================
// WulfNet Engine - Rigid Body Implementation
// =============================================================================

#include "RigidBody.h"
#include <cmath>

namespace WulfNet {

RigidBody::RigidBody(const RigidBodyConfig& config)
    : m_transform(config.transform)
    , m_linearVelocity(Vec3::zero())
    , m_angularVelocity(Vec3::zero())
    , m_mass(config.mass)
    , m_linearDamping(config.linearDamping)
    , m_angularDamping(config.angularDamping)
    , m_friction(config.friction)
    , m_restitution(config.restitution)
    , m_useGravity(config.useGravity)
    , m_isAwake(true)
    , m_canSleep(true)
    , m_motionType(config.type)
    , m_sleepThreshold(DEFAULT_SLEEP_THRESHOLD)
    , m_sleepTimer(0.0f)
    , m_motionEnergy(DEFAULT_SLEEP_THRESHOLD * 2.0f) // Start above threshold
    , m_shape(config.shape)
    , m_collisionLayer(config.collisionLayer)
    , m_forceAccumulator(Vec3::zero())
    , m_torqueAccumulator(Vec3::zero())
{
    calculateMassProperties();
    updateWorldTransform();
}

void RigidBody::setMotionType(MotionType type) {
    if (m_motionType == type) return;
    m_motionType = type;
    calculateMassProperties();
    
    if (m_motionType == MotionType::Static) {
        m_linearVelocity = Vec3::zero();
        m_angularVelocity = Vec3::zero();
        clearForces();
        m_isAwake = false;
        m_sleepTimer = 0.0f;
        m_motionEnergy = 0.0f;
    } else {
        wakeUp();
    }
}

void RigidBody::setAwake(bool awake) {
    if (awake) {
        wakeUp();
    } else {
        putToSleep();
    }
}

void RigidBody::wakeUp() {
    if (m_motionType == MotionType::Static) return;
    
    if (!m_isAwake) {
        m_isAwake = true;
        // Set motion energy slightly above threshold to prevent immediate re-sleep
        m_motionEnergy = m_sleepThreshold * 2.0f;
    }
    m_sleepTimer = 0.0f;
}

void RigidBody::putToSleep() {
    if (m_motionType == MotionType::Static) return;
    
    m_isAwake = false;
    m_linearVelocity = Vec3::zero();
    m_angularVelocity = Vec3::zero();
    m_sleepTimer = 0.0f;
    m_motionEnergy = 0.0f;
    clearForces();
}

void RigidBody::setLinearVelocity(const Vec3& v) {
    if (m_motionType == MotionType::Static) return;
    m_linearVelocity = v;
    if (v.lengthSq() > m_sleepThreshold * 0.1f) {
        wakeUp();
    }
}

void RigidBody::setAngularVelocity(const Vec3& v) {
    if (m_motionType == MotionType::Static) return;
    m_angularVelocity = v;
    if (v.lengthSq() > m_sleepThreshold * 0.1f) {
        wakeUp();
    }
}

void RigidBody::setPosition(const Vec3& pos) {
    m_transform.position = pos;
    if (m_motionType != MotionType::Static) {
        wakeUp();
    }
}

void RigidBody::setOrientation(const Quat& rot) {
    m_transform.rotation = rot;
    if (m_motionType != MotionType::Static) {
        wakeUp();
    }
}

void RigidBody::setTransform(const Transform& vals) {
    m_transform = vals;
    if (m_motionType != MotionType::Static) {
        wakeUp();
    }
}

void RigidBody::setMass(f32 mass) {
    m_mass = mass;
    calculateMassProperties();
}

void RigidBody::applyForce(const Vec3& force) {
    if (m_motionType != MotionType::Dynamic) return;
    m_forceAccumulator += force;
    wakeUp();
}

void RigidBody::applyForceAtPoint(const Vec3& force, const Vec3& point) {
    if (m_motionType != MotionType::Dynamic) return;
    
    m_forceAccumulator += force;
    
    // Torque = r x F
    Vec3 r = point - m_transform.position;
    m_torqueAccumulator += r.cross(force);
    
    wakeUp();
}

void RigidBody::applyTorque(const Vec3& torque) {
    if (m_motionType != MotionType::Dynamic) return;
    m_torqueAccumulator += torque;
    wakeUp();
}

void RigidBody::applyImpulse(const Vec3& impulse) {
    if (m_motionType != MotionType::Dynamic) return;
    m_linearVelocity += impulse * m_invMass;
    wakeUp();
}

void RigidBody::applyImpulseAtPoint(const Vec3& impulse, const Vec3& point) {
    if (m_motionType != MotionType::Dynamic) return;

    m_linearVelocity += impulse * m_invMass;

    Vec3 r = point - m_transform.position;
    Vec3 angularImpulse = r.cross(impulse);
    m_angularVelocity += m_invInertiaTensorWorld.transformDirection(angularImpulse);

    wakeUp();
}

void RigidBody::applyAngularImpulse(const Vec3& impulse) {
    if (m_motionType != MotionType::Dynamic) return;
    m_angularVelocity += m_invInertiaTensorWorld.transformDirection(impulse);
    wakeUp();
}

void RigidBody::applyPositionCorrection(const Vec3& correction) {
    if (m_motionType != MotionType::Dynamic) return;
    m_transform.position += correction;
    wakeUp();
}

void RigidBody::clearForces() {
    m_forceAccumulator = Vec3::zero();
    m_torqueAccumulator = Vec3::zero();
}

const AABB& RigidBody::getWorldAABB() const {
    // If not awake and AABB likely valid, we could skip re-computation (optimization)
    // But transform might have changed manually.
    
    // Update cached world AABB
    if (m_shape) {
        // Create transform matrix
        Mat4 transform = Mat4::translation(m_transform.position) * 
                         Mat4::rotation(m_transform.rotation) * 
                         Mat4::scale(m_transform.scale);
                         
        m_worldAABB = m_shape->getWorldAABB(transform);
    } else {
        // Fallback for no shape
        m_worldAABB = AABB(m_transform.position, m_transform.position);
    }
    
    return m_worldAABB;
}

void RigidBody::integrate(f32 dt) {
    if (m_motionType == MotionType::Static) {
        clearForces();
        return;
    }
    
    if (!m_isAwake) {
        clearForces();
        return;
    }
    
    // Semi-implicit Euler integration
    
    // 1. Update velocities (v += a * dt)
    // Dynamic bodies only
    if (m_motionType == MotionType::Dynamic) {
        // Linear acceleration
        Vec3 linearAcceleration = m_forceAccumulator * m_invMass;
        
        m_linearVelocity += linearAcceleration * dt;
        
        // Linear damping
        m_linearVelocity *= std::pow(1.0f - m_linearDamping, dt);
        
        // Angular acceleration (alpha = I_inv * Torque)
        Vec3 angularAcceleration = m_invInertiaTensorWorld.transformDirection(m_torqueAccumulator);
        m_angularVelocity += angularAcceleration * dt;
        
        // Angular damping
        m_angularVelocity *= std::pow(1.0f - m_angularDamping, dt);
    }
    
    // 2. Update position/orientation (p += v * dt)
    m_transform.position += m_linearVelocity * dt;
    
    // Quaternion integration: q += 0.5 * w * q * dt
    // w is angular velocity as pure quaternion (0, vx, vy, vz)
    Quat q = m_transform.rotation;
    Quat w(0, m_angularVelocity.x, m_angularVelocity.y, m_angularVelocity.z);
    
    // q_dot = 0.5 * w * q
    Quat qDot = w * q; // Assuming standard Hamilton product
    qDot.x *= 0.5f;
    qDot.y *= 0.5f;
    qDot.z *= 0.5f;
    qDot.w *= 0.5f;
    
    q.x += qDot.x * dt;
    q.y += qDot.y * dt;
    q.z += qDot.z * dt;
    q.w += qDot.w * dt;
    
    m_transform.rotation = q.normalized();
    
    // 3. Update derived state
    updateWorldTransform();
    
    // 4. Clear accumulators
    clearForces();
    
    // 5. Sleep detection
    if (m_canSleep && m_motionType == MotionType::Dynamic) {
        // Calculate current motion energy (kinetic energy proxy)
        // Use squared velocities for efficiency
        f32 currentMotion = m_linearVelocity.lengthSq() + m_angularVelocity.lengthSq();
        
        // Exponential moving average for smoothing (prevents jitter from causing wake)
        m_motionEnergy = MOTION_BIAS * m_motionEnergy + (1.0f - MOTION_BIAS) * currentMotion;
        
        if (m_motionEnergy < m_sleepThreshold) {
            m_sleepTimer += dt;
            if (m_sleepTimer >= SLEEP_TIME_REQUIRED) {
                putToSleep();
            }
        } else {
            m_sleepTimer = 0.0f;
        }
    }
}

void RigidBody::updateWorldTransform() {
    // Recompute cached world inverse inertia tensor
    // I_inv_world = R * I_inv_local * R^T
    
    if (m_motionType == MotionType::Dynamic && m_mass > 0.0f) {
        Mat4 rotationMat = Mat4::rotation(m_transform.rotation);
        m_invInertiaTensorWorld = rotationMat * m_invInertiaTensorLocal * rotationMat.transposed();
    } else {
        m_invInertiaTensorWorld = Mat4::zero();
    }
}

void RigidBody::calculateMassProperties() {
    if (m_motionType == MotionType::Static || m_motionType == MotionType::Kinematic) {
        m_invMass = 0.0f;
        m_inertiaTensor = Mat4::zero();
        m_invInertiaTensorLocal = Mat4::zero();
        m_invInertiaTensorWorld = Mat4::zero();
    } else {
        // Dynamic
        m_invMass = m_mass > 0.0f ? 1.0f / m_mass : 0.0f;
        
        if (m_shape && m_mass > 0.0f) {
            m_inertiaTensor = m_shape->getLocalInertiaTensor() * m_mass; // Shape gives normalized inertia usually? 
            // Shape::getLocalInertiaTensor usually returns inertia for mass=1. Check CollisionShape.h?
            // "getLocalInertiaTensor()" implementations like Sphere return (2/5)*r*r. Multiplied by mass is I.
            // Wait, sphere inertia is (2/5)mr^2. If shape returns just geometric part, we invoke mass.
            
            // SphereShape::getLocalInertiaTensor:
            // return Mat4::scale(Vec3(i, i, i)); where i = (2/5)*r*r.
            // So it does not include mass.
            // CORRECT: We multiply by mass.
            // (Assuming existing code matches this logic, let's trust it for now)
            
            
            // We need inverse local.
            // For diagonal matrices (AABB, Sphere), inverse is 1/diagonal.
            // But generic Mat4 inverse is 4x4.
            m_invInertiaTensorLocal = m_inertiaTensor.inverse();
            
        } else {
            // Point mass approximation
            m_inertiaTensor = Mat4::identity();
            m_invInertiaTensorLocal = Mat4::identity();
        }
    }

    updateWorldTransform();
}

} // namespace WulfNet
