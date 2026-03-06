// =============================================================================
// WulfNet Engine - MPM ↔ Rigid Body Coupling Implementation
// =============================================================================

#include "MPMRigidCoupling.h"
#include "ConstitutiveModel.h"

// Jolt includes
#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/MotionType.h>

#include <cmath>
#include <algorithm>
#include <limits>

namespace WulfNet {

// =============================================================================
// Quaternion helpers (rotate vector by quaternion)
// =============================================================================

static void QuatRotate(float qx, float qy, float qz, float qw,
                       float vx, float vy, float vz,
                       float& ox, float& oy, float& oz)
{
    // q * v * q^-1 using the optimized formula
    float tx = 2.0f * (qy * vz - qz * vy);
    float ty = 2.0f * (qz * vx - qx * vz);
    float tz = 2.0f * (qx * vy - qy * vx);
    ox = vx + qw * tx + (qy * tz - qz * ty);
    oy = vy + qw * ty + (qz * tx - qx * tz);
    oz = vz + qw * tz + (qx * ty - qy * tx);
}

static void QuatConjugateRotate(float qx, float qy, float qz, float qw,
                                 float vx, float vy, float vz,
                                 float& ox, float& oy, float& oz)
{
    // Conjugate quaternion rotation: q^-1 * v * q
    QuatRotate(-qx, -qy, -qz, qw, vx, vy, vz, ox, oy, oz);
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

MPMRigidCoupling::MPMRigidCoupling() = default;
MPMRigidCoupling::~MPMRigidCoupling() { Shutdown(); }

// =============================================================================
// Initialization
// =============================================================================

bool MPMRigidCoupling::Initialize(const MPMCouplingConfig& config)
{
    if (m_initialized) return false;

    m_config = config;
    m_bodies.clear();
    m_stats = MPMCouplingStats{};
    m_initialized = true;
    return true;
}

void MPMRigidCoupling::Shutdown()
{
    m_bodies.clear();
    m_stats = MPMCouplingStats{};
    m_initialized = false;
}

// =============================================================================
// Body Registration
// =============================================================================

uint32_t MPMRigidCoupling::AddCoupledBody(
    JPH::BodyID bodyId,
    CoupledRigidBody::ShapeType shape,
    float radius,
    float halfExtentX, float halfExtentY, float halfExtentZ)
{
    CoupledRigidBody body;
    body.bodyId = bodyId;
    body.shapeType = shape;
    body.radius = radius;
    body.halfExtentX = halfExtentX;
    body.halfExtentY = halfExtentY;
    body.halfExtentZ = halfExtentZ;

    if (shape == CoupledRigidBody::ShapeType::Capsule) {
        body.height = halfExtentY * 2.0f; // Use Y extent as half-height
    }

    uint32_t handle = static_cast<uint32_t>(m_bodies.size());
    m_bodies.push_back(body);
    return handle;
}

void MPMRigidCoupling::RemoveCoupledBody(uint32_t handle)
{
    if (handle < m_bodies.size()) {
        m_bodies[handle].enabled = false;
    }
}

CoupledRigidBody* MPMRigidCoupling::GetCoupledBody(uint32_t handle)
{
    if (handle < m_bodies.size()) return &m_bodies[handle];
    return nullptr;
}

const CoupledRigidBody* MPMRigidCoupling::GetCoupledBody(uint32_t handle) const
{
    if (handle < m_bodies.size()) return &m_bodies[handle];
    return nullptr;
}

// =============================================================================
// Body-Local Coordinate Transforms
// =============================================================================

void MPMRigidCoupling::TransformToBodyLocal(
    const CoupledRigidBody& body,
    float wx, float wy, float wz,
    float& lx, float& ly, float& lz) const
{
    // Translate to body center
    float dx = wx - body.posX;
    float dy = wy - body.posY;
    float dz = wz - body.posZ;

    // Rotate by inverse quaternion (conjugate)
    QuatConjugateRotate(body.quatX, body.quatY, body.quatZ, body.quatW,
                        dx, dy, dz, lx, ly, lz);
}

void MPMRigidCoupling::TransformNormalToWorld(
    const CoupledRigidBody& body,
    float lnx, float lny, float lnz,
    float& wnx, float& wny, float& wnz) const
{
    QuatRotate(body.quatX, body.quatY, body.quatZ, body.quatW,
               lnx, lny, lnz, wnx, wny, wnz);
}

// =============================================================================
// Signed Distance Functions (in body-local space)
// =============================================================================

float MPMRigidCoupling::ComputeBodySDF(
    const CoupledRigidBody& body,
    float px, float py, float pz,
    float& nx, float& ny, float& nz) const
{
    // Transform to local coordinates
    float lx, ly, lz;
    TransformToBodyLocal(body, px, py, pz, lx, ly, lz);

    float dist = 0.0f;
    float lnx = 0.0f, lny = 0.0f, lnz = 0.0f;

    switch (body.shapeType) {
    case CoupledRigidBody::ShapeType::Sphere: {
        float r = std::sqrt(lx * lx + ly * ly + lz * lz);
        dist = r - body.radius;
        if (r > 1e-8f) {
            float invR = 1.0f / r;
            lnx = lx * invR;
            lny = ly * invR;
            lnz = lz * invR;
        } else {
            lny = 1.0f; // Default up
        }
        break;
    }

    case CoupledRigidBody::ShapeType::Box: {
        // SDF for axis-aligned box in local space
        float dx = std::abs(lx) - body.halfExtentX;
        float dy = std::abs(ly) - body.halfExtentY;
        float dz = std::abs(lz) - body.halfExtentZ;

        float ex = std::max(dx, 0.0f);
        float ey = std::max(dy, 0.0f);
        float ez = std::max(dz, 0.0f);

        float outsideDist = std::sqrt(ex * ex + ey * ey + ez * ez);
        float insideDist = std::min(std::max(dx, std::max(dy, dz)), 0.0f);
        dist = outsideDist + insideDist;

        // Approximate gradient for normal
        if (dist > 1e-8f) {
            // Outside: normal points away from surface
            if (outsideDist > 1e-8f) {
                float invLen = 1.0f / outsideDist;
                lnx = ex * invLen * (lx >= 0.0f ? 1.0f : -1.0f);
                lny = ey * invLen * (ly >= 0.0f ? 1.0f : -1.0f);
                lnz = ez * invLen * (lz >= 0.0f ? 1.0f : -1.0f);
            } else {
                // On edge — pick dominant axis
                if (dx >= dy && dx >= dz) lnx = (lx >= 0.0f) ? 1.0f : -1.0f;
                else if (dy >= dx && dy >= dz) lny = (ly >= 0.0f) ? 1.0f : -1.0f;
                else lnz = (lz >= 0.0f) ? 1.0f : -1.0f;
            }
        } else {
            // Inside: find closest face
            if (dx > dy && dx > dz) lnx = (lx >= 0.0f) ? 1.0f : -1.0f;
            else if (dy > dx && dy > dz) lny = (ly >= 0.0f) ? 1.0f : -1.0f;
            else lnz = (lz >= 0.0f) ? 1.0f : -1.0f;
        }
        break;
    }

    case CoupledRigidBody::ShapeType::Capsule: {
        // Capsule: Two hemispheres connected by a cylinder along Y axis
        float halfH = body.height * 0.5f;
        float clampedY = std::max(-halfH, std::min(ly, halfH));

        float relX = lx;
        float relY = ly - clampedY;
        float relZ = lz;
        float r = std::sqrt(relX * relX + relY * relY + relZ * relZ);
        dist = r - body.radius;

        if (r > 1e-8f) {
            float invR = 1.0f / r;
            lnx = relX * invR;
            lny = relY * invR;
            lnz = relZ * invR;
        } else {
            lny = 1.0f;
        }
        break;
    }

    default:
        // For custom shapes, default to sphere SDF
        {
            float r = std::sqrt(lx * lx + ly * ly + lz * lz);
            dist = r - body.radius;
            if (r > 1e-8f) {
                float invR = 1.0f / r;
                lnx = lx * invR; lny = ly * invR; lnz = lz * invR;
            } else {
                lny = 1.0f;
            }
        }
        break;
    }

    // Transform normal back to world space
    TransformNormalToWorld(body, lnx, lny, lnz, nx, ny, nz);

    // Normalize
    float nLen = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (nLen > 1e-8f) {
        nx /= nLen;
        ny /= nLen;
        nz /= nLen;
    }

    return dist;
}

// =============================================================================
// SDF Query (nearest body)
// =============================================================================

float MPMRigidCoupling::QuerySDF(
    float x, float y, float z,
    int32_t& bodyIndex,
    float& nx, float& ny, float& nz) const
{
    bodyIndex = -1;
    nx = 0.0f; ny = 1.0f; nz = 0.0f;
    float minDist = std::numeric_limits<float>::max();

    for (uint32_t i = 0; i < m_bodies.size(); ++i) {
        const auto& body = m_bodies[i];
        if (!body.enabled) continue;

        float bnx, bny, bnz;
        float dist = ComputeBodySDF(body, x, y, z, bnx, bny, bnz);

        if (dist < minDist) {
            minDist = dist;
            bodyIndex = static_cast<int32_t>(i);
            nx = bnx;
            ny = bny;
            nz = bnz;
        }
    }

    return minDist;
}

// =============================================================================
// Body Surface Velocity
// =============================================================================

void MPMRigidCoupling::GetBodySurfaceVelocity(
    uint32_t bodyIdx,
    float px, float py, float pz,
    float& vx, float& vy, float& vz) const
{
    if (bodyIdx >= m_bodies.size()) {
        vx = vy = vz = 0.0f;
        return;
    }

    const auto& body = m_bodies[bodyIdx];

    // v_surface = v_linear + ω × r
    float rx = px - body.posX;
    float ry = py - body.posY;
    float rz = pz - body.posZ;

    // Cross product: ω × r
    float crossX = body.angVelY * rz - body.angVelZ * ry;
    float crossY = body.angVelZ * rx - body.angVelX * rz;
    float crossZ = body.angVelX * ry - body.angVelY * rx;

    vx = body.velX + crossX;
    vy = body.velY + crossY;
    vz = body.velZ + crossZ;
}

// =============================================================================
// Force Accumulation with Torque
// =============================================================================

void MPMRigidCoupling::AccumulateForceOnBody(
    CoupledRigidBody& body,
    float fx, float fy, float fz,
    float px, float py, float pz)
{
    // Accumulate linear force (reaction force: Newton's 3rd law)
    // Particles push body, so negate the force applied to particle
    body.accForceX -= fx;
    body.accForceY -= fy;
    body.accForceZ -= fz;

    // Compute torque: τ = r × F
    float rx = px - body.posX;
    float ry = py - body.posY;
    float rz = pz - body.posZ;

    body.accTorqueX -= (ry * fz - rz * fy);
    body.accTorqueY -= (rz * fx - rx * fz);
    body.accTorqueZ -= (rx * fy - ry * fx);

    body.contactCount++;
}

// =============================================================================
// Sync Body States from Jolt
// =============================================================================

void MPMRigidCoupling::SyncBodyStates(JPH::PhysicsSystem& joltPhysics)
{
    JPH::BodyInterface& bi = joltPhysics.GetBodyInterface();

    for (auto& body : m_bodies) {
        if (!body.enabled) continue;

        if (!bi.IsAdded(body.bodyId)) {
            body.enabled = false;
            continue;
        }

        // Read position
        JPH::RVec3 pos = bi.GetCenterOfMassPosition(body.bodyId);
        body.posX = static_cast<float>(pos.GetX());
        body.posY = static_cast<float>(pos.GetY());
        body.posZ = static_cast<float>(pos.GetZ());

        // Read velocity
        JPH::Vec3 vel = bi.GetLinearVelocity(body.bodyId);
        body.velX = vel.GetX();
        body.velY = vel.GetY();
        body.velZ = vel.GetZ();

        // Read angular velocity
        JPH::Vec3 angVel = bi.GetAngularVelocity(body.bodyId);
        body.angVelX = angVel.GetX();
        body.angVelY = angVel.GetY();
        body.angVelZ = angVel.GetZ();

        // Read orientation
        JPH::Quat rot = bi.GetRotation(body.bodyId);
        body.quatX = rot.GetX();
        body.quatY = rot.GetY();
        body.quatZ = rot.GetZ();
        body.quatW = rot.GetW();

        // Read mass via motion type (Jolt doesn't expose inverse mass directly
        // through BodyInterface — we use motion type to detect static bodies)
        JPH::EMotionType motionType = bi.GetMotionType(body.bodyId);
        if (motionType == JPH::EMotionType::Static || motionType == JPH::EMotionType::Kinematic) {
            body.isStatic = (motionType == JPH::EMotionType::Static);
            body.invMass = 0.0f;
            body.mass = 1e10f;
        } else {
            body.isStatic = false;
            // For dynamic bodies, use configured mass (user sets via CoupledRigidBody)
            if (body.mass > 0.0f) {
                body.invMass = 1.0f / body.mass;
            }
        }
    }
}

void MPMRigidCoupling::ClearAccumulators()
{
    for (auto& body : m_bodies) {
        body.ClearAccumulators();
    }
}

// =============================================================================
// Main Coupling: MPMParticle ↔ Rigid Bodies
// =============================================================================

void MPMRigidCoupling::ComputeCoupling(
    MPMParticle* particles,
    uint32_t particleCount,
    const MPMMaterialParams& params,
    JPH::PhysicsSystem& joltPhysics,
    float dt)
{
    if (!m_initialized || !particles || particleCount == 0) return;

    // Reset stats
    m_stats = MPMCouplingStats{};

    // 1. Sync body states from Jolt
    SyncBodyStates(joltPhysics);
    ClearAccumulators();

    // Count active bodies
    uint32_t activeBodies = 0;
    for (const auto& b : m_bodies) {
        if (b.enabled) activeBodies++;
    }
    m_stats.activeBodies = activeBodies;
    if (activeBodies == 0) return;

    float maxForce = 0.0f;
    float totalForce = 0.0f;

    // 2. For each particle, check interaction with each body
    for (uint32_t pi = 0; pi < particleCount; ++pi) {
        MPMParticle& p = particles[pi];

        for (uint32_t bi = 0; bi < m_bodies.size(); ++bi) {
            CoupledRigidBody& body = m_bodies[bi];
            if (!body.enabled) continue;

            // Compute SDF
            float nx, ny, nz;
            float dist = ComputeBodySDF(body, p.x, p.y, p.z, nx, ny, nz);

            // Skip if too far
            if (dist > m_config.interactionRadius) continue;

            m_stats.particleBodyContacts++;

            // === Particle → Body force (penalty contact) ===
            if (m_config.enableParticleToBody && dist < m_config.interactionRadius) {
                float penetration = m_config.interactionRadius - dist;

                // Penalty force: F = k * penetration * normal
                float forceScale = m_config.penaltyStiffness * penetration;

                // Clamp each particle's contribution
                forceScale = std::min(forceScale, m_config.maxCouplingForce);

                float fx = forceScale * nx;
                float fy = forceScale * ny;
                float fz = forceScale * nz;

                // Damping: resist relative velocity along normal
                float surfVx, surfVy, surfVz;
                GetBodySurfaceVelocity(bi, p.x, p.y, p.z, surfVx, surfVy, surfVz);

                float relVx = p.vx - surfVx;
                float relVy = p.vy - surfVy;
                float relVz = p.vz - surfVz;

                float relVn = relVx * nx + relVy * ny + relVz * nz;

                // Normal damping
                float dampForce = m_config.dampingCoefficient * relVn;
                fx += dampForce * nx;
                fy += dampForce * ny;
                fz += dampForce * nz;

                // Friction (Coulomb)
                if (m_config.enableFriction && dist < 0.0f) {
                    float tanVx = relVx - relVn * nx;
                    float tanVy = relVy - relVn * ny;
                    float tanVz = relVz - relVn * nz;

                    float tanSpeed = std::sqrt(tanVx * tanVx + tanVy * tanVy + tanVz * tanVz);
                    if (tanSpeed > 1e-8f) {
                        float frictionMag = std::min(
                            m_config.frictionCoefficient * std::abs(forceScale),
                            m_config.dampingCoefficient * tanSpeed
                        );
                        float invTan = frictionMag / tanSpeed;
                        fx += tanVx * invTan;
                        fy += tanVy * invTan;
                        fz += tanVz * invTan;
                    }
                }

                float forceMag = std::sqrt(fx * fx + fy * fy + fz * fz);
                maxForce = std::max(maxForce, forceMag);
                totalForce += forceMag;

                // Accumulate on body (Newton's 3rd — negated inside)
                AccumulateForceOnBody(body, fx, fy, fz, p.x, p.y, p.z);
            }

            // === Body → Particle velocity correction ===
            if (m_config.enableBodyToParticle && dist < 0.0f) {
                // Particle is inside the body — project it out and match velocity
                float surfVx, surfVy, surfVz;
                GetBodySurfaceVelocity(bi, p.x, p.y, p.z, surfVx, surfVy, surfVz);

                float relVn = (p.vx - surfVx) * nx + (p.vy - surfVy) * ny + (p.vz - surfVz) * nz;

                // Only correct if particle is moving into the body
                if (relVn < 0.0f) {
                    float strength = body.couplingStrength;

                    // Remove inward velocity component
                    p.vx -= strength * relVn * nx;
                    p.vy -= strength * relVn * ny;
                    p.vz -= strength * relVn * nz;

                    // Friction on tangent
                    if (m_config.enableFriction) {
                        float tanVx = (p.vx - surfVx) - ((p.vx - surfVx) * nx + (p.vy - surfVy) * ny + (p.vz - surfVz) * nz) * nx;
                        float tanVy = (p.vy - surfVy) - ((p.vx - surfVx) * nx + (p.vy - surfVy) * ny + (p.vz - surfVz) * nz) * ny;
                        float tanVz = (p.vz - surfVz) - ((p.vx - surfVx) * nx + (p.vy - surfVy) * ny + (p.vz - surfVz) * nz) * nz;

                        float tanSpeed = std::sqrt(tanVx * tanVx + tanVy * tanVy + tanVz * tanVz);
                        if (tanSpeed > 1e-8f) {
                            float frictionScale = std::min(1.0f,
                                body.friction * std::abs(relVn) / tanSpeed);
                            p.vx -= strength * frictionScale * tanVx;
                            p.vy -= strength * frictionScale * tanVy;
                            p.vz -= strength * frictionScale * tanVz;
                        }
                    }

                    // Position correction: push particle to surface
                    float pushDist = -dist * strength;
                    p.x += pushDist * nx;
                    p.y += pushDist * ny;
                    p.z += pushDist * nz;
                }
            }
        }
    }

    m_stats.maxForceApplied = maxForce;
    m_stats.totalForceApplied = totalForce;
}

// =============================================================================
// Apply Forces to Jolt Bodies
// =============================================================================

void MPMRigidCoupling::ApplyForcesToBodies(JPH::PhysicsSystem& joltPhysics)
{
    if (!m_initialized) return;

    JPH::BodyInterface& bi = joltPhysics.GetBodyInterface();

    for (auto& body : m_bodies) {
        if (!body.enabled || body.isStatic || body.contactCount == 0) continue;

        // Clamp total force magnitude
        float fMag = std::sqrt(body.accForceX * body.accForceX +
                               body.accForceY * body.accForceY +
                               body.accForceZ * body.accForceZ);

        float scale = 1.0f;
        if (fMag > m_config.maxBodyForce) {
            scale = m_config.maxBodyForce / fMag;
        }

        JPH::Vec3 force(
            body.accForceX * scale,
            body.accForceY * scale,
            body.accForceZ * scale
        );

        JPH::Vec3 torque(
            body.accTorqueX * scale,
            body.accTorqueY * scale,
            body.accTorqueZ * scale
        );

        // Apply force and torque to Jolt body
        if (bi.IsAdded(body.bodyId) && bi.IsActive(body.bodyId)) {
            bi.AddForce(body.bodyId, force);
            bi.AddTorque(body.bodyId, torque);
        }
    }
}

} // namespace WulfNet
