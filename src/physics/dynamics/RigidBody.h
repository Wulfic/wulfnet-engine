// =============================================================================
// WulfNet Engine - Rigid Body
// =============================================================================
// Represents a physical object in the simulation with mass and motion.
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/math/Math.h"
#include "core/Assert.h"
#include "physics/collision/CollisionShape.h"
#include "physics/collision/AABB.h"
#include "physics/collision/CollisionTypes.h"

namespace WulfNet {

// =============================================================================
// Motion Type
// =============================================================================

enum class MotionType : u8 {
    Static = 0,    // Infinite mass, zero velocity, does not move (walls, ground)
    Kinematic = 1, // Infinite mass, moves by setting velocity (platforms, characters)
    Dynamic = 2    // Finite mass, moves by forces and collisions
};

// =============================================================================
// Rigid Body
// =============================================================================

struct RigidBodyConfig {
    Transform transform;
    CollisionShape* shape = nullptr;
    CollisionLayer collisionLayer = CollisionLayer::Default;
    MotionType type = MotionType::Dynamic;
    f32 mass = 1.0f;
    f32 linearDamping = 0.05f;
    f32 angularDamping = 0.05f;
    f32 friction = 0.5f;
    f32 restitution = 0.2f; // Bounciness
    bool useGravity = true;
};

class alignas(16) RigidBody {
public:
    RigidBody(const RigidBodyConfig& config);
    ~RigidBody() = default;
    
    // Non-copyable to prevent accidental object slicing or pointer issues
    RigidBody(const RigidBody&) = delete;
    RigidBody& operator=(const RigidBody&) = delete;
    
    // Move-constructible
    RigidBody(RigidBody&&) noexcept = default;
    RigidBody& operator=(RigidBody&&) noexcept = default;
    
    // =========================================================================
    // Properties
    // =========================================================================
    
    MotionType getMotionType() const { return m_motionType; }
    void setMotionType(MotionType type);
    
    // Transform
    const Vec3& getPosition() const { return m_transform.position; }
    const Quat& getOrientation() const { return m_transform.rotation; }
    const Transform& getTransform() const { return m_transform; }
    
    void setPosition(const Vec3& pos);
    void setOrientation(const Quat& rot);
    void setTransform(const Transform& vals);
    
    // Velocity
    const Vec3& getLinearVelocity() const { return m_linearVelocity; }
    const Vec3& getAngularVelocity() const { return m_angularVelocity; }
    const Mat4& getInvInertiaTensorWorld() const { return m_invInertiaTensorWorld; }
    
    void setLinearVelocity(const Vec3& v);
    void setAngularVelocity(const Vec3& v);
    
    // Mass properties
    f32 getMass() const { return m_mass; }
    f32 getInverseMass() const { return m_invMass; }
    void setMass(f32 mass);

    bool usesGravity() const { return m_useGravity; }
    void setUseGravity(bool value) { m_useGravity = value; }

    f32 getFriction() const { return m_friction; }
    f32 getRestitution() const { return m_restitution; }
    
    // Forces
    void applyForce(const Vec3& force);
    void applyForceAtPoint(const Vec3& force, const Vec3& point);
    void applyTorque(const Vec3& torque);
    void applyImpulse(const Vec3& impulse);
    void applyImpulseAtPoint(const Vec3& impulse, const Vec3& point);
    void applyAngularImpulse(const Vec3& impulse);
    void applyPositionCorrection(const Vec3& correction);
    void clearForces();
    
    // Collision
    const AABB& getWorldAABB() const; // cached
    CollisionShape* getShape() const { return m_shape; }
    CollisionLayer getCollisionLayer() const { return m_collisionLayer; }
    void setCollisionLayer(CollisionLayer layer) { m_collisionLayer = layer; }
    
    // =========================================================================
    // Sleep System
    // =========================================================================
    // Bodies are put to sleep when their motion (linear + angular velocity) 
    // remains below a threshold for a sustained period. Sleeping bodies skip
    // integration and collision detection for improved performance.
    
    bool isAwake() const { return m_isAwake; }
    void setAwake(bool awake);
    void wakeUp();      // Force wake and reset sleep timer
    void putToSleep();  // Force sleep
    
    bool canSleep() const { return m_canSleep; }
    void setCanSleep(bool value) { m_canSleep = value; }
    
    f32 getSleepThreshold() const { return m_sleepThreshold; }
    void setSleepThreshold(f32 threshold) { m_sleepThreshold = threshold; }
    
    f32 getMotionEnergy() const { return m_motionEnergy; }
    f32 getSleepTimer() const { return m_sleepTimer; }
    
    bool isStatic() const { return m_motionType == MotionType::Static; }
    bool isDynamic() const { return m_motionType == MotionType::Dynamic; }
    bool isKinematic() const { return m_motionType == MotionType::Kinematic; }
    
    // Simulation step (integrator)
    void integrate(f32 dt);
    void updateWorldTransform();
    
private:
    void calculateMassProperties();
    
    // State
    Transform m_transform;
    Vec3 m_linearVelocity;
    Vec3 m_angularVelocity;
    
    // Mass Data
    f32 m_mass;
    f32 m_invMass;
    Mat4 m_inertiaTensor;      // Local inertia
    Mat4 m_invInertiaTensorLocal; // Local inverse inertia (cached)
    Mat4 m_invInertiaTensorWorld; // World inverse inertia (cached for solver)
    
    // Forces
    Vec3 m_forceAccumulator;
    Vec3 m_torqueAccumulator;
    
    // Properties
    f32 m_linearDamping;
    f32 m_angularDamping;
    f32 m_friction;
    f32 m_restitution;
    bool m_useGravity;
    bool m_isAwake;
    bool m_canSleep;
    MotionType m_motionType;
    
    // Sleep system
    f32 m_sleepThreshold;  // Motion energy below which body can sleep
    f32 m_sleepTimer;      // Time spent below threshold
    f32 m_motionEnergy;    // Smoothed motion energy for sleep detection
    static constexpr f32 DEFAULT_SLEEP_THRESHOLD = 0.01f;
    static constexpr f32 SLEEP_TIME_REQUIRED = 0.5f; // 0.5 seconds of low motion
    static constexpr f32 MOTION_BIAS = 0.8f; // Smoothing factor for motion energy
    
    // Collision
    CollisionShape* m_shape;
    mutable AABB m_worldAABB;
    CollisionLayer m_collisionLayer;
};

} // namespace WulfNet
