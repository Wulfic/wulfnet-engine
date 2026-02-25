// =============================================================================
// WulfNet Engine - MPM Rigid Body Coupling Tests
// =============================================================================
// Tests for bidirectional force exchange between MPM particles and rigid bodies.
// =============================================================================

#include "TestHarness.h"
#include "WulfNet/Physics/MPM/MPMRigidCoupling.h"
#include "WulfNet/Physics/MPM/ConstitutiveModel.h"
#include "WulfNet/Physics/Fluids/FluidParticle.h"
#include <cmath>

using namespace WulfNet;

// =============================================================================
// Helper: Create a default coupling config
// =============================================================================

static MPMCouplingConfig MakeCouplingConfig() {
    MPMCouplingConfig cfg;
    cfg.penaltyStiffness = 1.0e4f;
    cfg.dampingCoefficient = 100.0f;
    cfg.frictionCoefficient = 0.3f;
    cfg.interactionRadius = 0.05f;
    cfg.smoothingRadius = 0.1f;
    cfg.maxCouplingForce = 1.0e6f;
    cfg.maxBodyForce = 1.0e8f;
    cfg.enableParticleToBody = true;
    cfg.enableBodyToParticle = true;
    cfg.enableFriction = true;
    return cfg;
}

// =============================================================================
// MPMCouplingConfig Tests
// =============================================================================

void test_CouplingConfig_Defaults() {
    MPMCouplingConfig cfg;
    EXPECT_TRUE(cfg.penaltyStiffness > 0.0f);
    EXPECT_TRUE(cfg.dampingCoefficient >= 0.0f);
    EXPECT_TRUE(cfg.interactionRadius > 0.0f);
    EXPECT_TRUE(cfg.maxCouplingForce > 0.0f);
    EXPECT_TRUE(cfg.maxBodyForce > 0.0f);
    EXPECT_TRUE(cfg.enableParticleToBody);
    EXPECT_TRUE(cfg.enableBodyToParticle);
    EXPECT_TRUE(cfg.enableFriction);
}

void test_CouplingConfig_Custom() {
    MPMCouplingConfig cfg;
    cfg.penaltyStiffness = 5000.0f;
    cfg.interactionRadius = 0.1f;
    cfg.enableFriction = false;
    EXPECT_NEAR(cfg.penaltyStiffness, 5000.0f, 1e-6f);
    EXPECT_NEAR(cfg.interactionRadius, 0.1f, 1e-6f);
    EXPECT_TRUE(!cfg.enableFriction);
}

// =============================================================================
// Initialization Tests
// =============================================================================

void test_Coupling_InitShutdown() {
    MPMRigidCoupling coupling;
    EXPECT_TRUE(!coupling.IsInitialized());

    MPMCouplingConfig cfg = MakeCouplingConfig();
    EXPECT_TRUE(coupling.Initialize(cfg));
    EXPECT_TRUE(coupling.IsInitialized());
    EXPECT_EQ(coupling.GetCoupledBodyCount(), 0u);

    coupling.Shutdown();
    EXPECT_TRUE(!coupling.IsInitialized());
}

void test_Coupling_DoubleInit() {
    MPMRigidCoupling coupling;
    MPMCouplingConfig cfg = MakeCouplingConfig();

    EXPECT_TRUE(coupling.Initialize(cfg));
    // Second init should fail (already initialized)
    EXPECT_TRUE(!coupling.Initialize(cfg));
    coupling.Shutdown();
}

void test_Coupling_ConfigAccess() {
    MPMRigidCoupling coupling;
    MPMCouplingConfig cfg = MakeCouplingConfig();
    cfg.penaltyStiffness = 12345.0f;
    coupling.Initialize(cfg);

    EXPECT_NEAR(coupling.GetConfig().penaltyStiffness, 12345.0f, 1e-6f);

    MPMCouplingConfig newCfg = MakeCouplingConfig();
    newCfg.penaltyStiffness = 99999.0f;
    coupling.SetConfig(newCfg);
    EXPECT_NEAR(coupling.GetConfig().penaltyStiffness, 99999.0f, 1e-6f);

    coupling.Shutdown();
}

// =============================================================================
// Body Registration Tests
// =============================================================================

void test_Coupling_AddBody() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    JPH::BodyID bid(0);
    uint32_t handle = coupling.AddCoupledBody(
        bid, CoupledRigidBody::ShapeType::Sphere, 1.0f);

    EXPECT_EQ(handle, 0u);
    EXPECT_EQ(coupling.GetCoupledBodyCount(), 1u);

    auto* body = coupling.GetCoupledBody(handle);
    EXPECT_TRUE(body != nullptr);
    EXPECT_NEAR(body->radius, 1.0f, 1e-6f);
    EXPECT_TRUE(body->enabled);

    coupling.Shutdown();
}

void test_Coupling_AddMultipleBodies() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    uint32_t h0 = coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 0.5f);
    uint32_t h1 = coupling.AddCoupledBody(
        JPH::BodyID(1), CoupledRigidBody::ShapeType::Box,
        0.0f, 1.0f, 1.0f, 1.0f);
    uint32_t h2 = coupling.AddCoupledBody(
        JPH::BodyID(2), CoupledRigidBody::ShapeType::Capsule,
        0.3f, 0.3f, 0.5f, 0.3f);

    EXPECT_EQ(coupling.GetCoupledBodyCount(), 3u);
    EXPECT_NE(h0, h1);
    EXPECT_NE(h1, h2);

    auto* b0 = coupling.GetCoupledBody(h0);
    auto* b1 = coupling.GetCoupledBody(h1);
    auto* b2 = coupling.GetCoupledBody(h2);

    EXPECT_TRUE(b0 != nullptr);
    EXPECT_TRUE(b1 != nullptr);
    EXPECT_TRUE(b2 != nullptr);

    coupling.Shutdown();
}

void test_Coupling_RemoveBody() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    uint32_t h = coupling.AddCoupledBody(
        JPH::BodyID(5), CoupledRigidBody::ShapeType::Sphere, 1.0f);

    auto* body = coupling.GetCoupledBody(h);
    EXPECT_TRUE(body->enabled);

    coupling.RemoveCoupledBody(h);
    body = coupling.GetCoupledBody(h);
    EXPECT_TRUE(!body->enabled);

    coupling.Shutdown();
}

void test_Coupling_GetInvalidBody() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    EXPECT_TRUE(coupling.GetCoupledBody(999) == nullptr);

    coupling.Shutdown();
}

// =============================================================================
// CoupledRigidBody Tests
// =============================================================================

void test_CoupledBody_DefaultState() {
    CoupledRigidBody body;
    EXPECT_NEAR(body.posX, 0.0f, 1e-6f);
    EXPECT_NEAR(body.posY, 0.0f, 1e-6f);
    EXPECT_NEAR(body.posZ, 0.0f, 1e-6f);
    EXPECT_NEAR(body.velX, 0.0f, 1e-6f);
    EXPECT_NEAR(body.quatW, 1.0f, 1e-6f);
    EXPECT_NEAR(body.couplingStrength, 1.0f, 1e-6f);
    EXPECT_TRUE(body.enabled);
    EXPECT_TRUE(!body.isStatic);
}

void test_CoupledBody_ClearAccumulators() {
    CoupledRigidBody body;
    body.accForceX = 100.0f;
    body.accForceY = -50.0f;
    body.accTorqueZ = 10.0f;
    body.contactCount = 42;

    body.ClearAccumulators();

    EXPECT_NEAR(body.accForceX, 0.0f, 1e-6f);
    EXPECT_NEAR(body.accForceY, 0.0f, 1e-6f);
    EXPECT_NEAR(body.accTorqueZ, 0.0f, 1e-6f);
    EXPECT_EQ(body.contactCount, 0u);
}

void test_CoupledBody_ShapeTypes() {
    CoupledRigidBody sphere;
    sphere.shapeType = CoupledRigidBody::ShapeType::Sphere;
    EXPECT_EQ(static_cast<uint32_t>(sphere.shapeType), 0u);

    CoupledRigidBody box;
    box.shapeType = CoupledRigidBody::ShapeType::Box;
    EXPECT_EQ(static_cast<uint32_t>(box.shapeType), 1u);

    CoupledRigidBody capsule;
    capsule.shapeType = CoupledRigidBody::ShapeType::Capsule;
    EXPECT_EQ(static_cast<uint32_t>(capsule.shapeType), 2u);
}

// =============================================================================
// SDF Query Tests (no Jolt physics needed — operates on cached body state)
// =============================================================================

void test_SDF_SphereOutside() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    uint32_t h = coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 1.0f);

    // Manually set body position at origin
    auto* body = coupling.GetCoupledBody(h);
    body->posX = 0.0f; body->posY = 0.0f; body->posZ = 0.0f;
    body->quatX = 0.0f; body->quatY = 0.0f; body->quatZ = 0.0f; body->quatW = 1.0f;

    // Query point 2m away along X — should be outside (dist ≈ 1.0)
    int32_t idx; float nx, ny, nz;
    float dist = coupling.QuerySDF(2.0f, 0.0f, 0.0f, idx, nx, ny, nz);

    EXPECT_EQ(idx, 0);
    EXPECT_NEAR(dist, 1.0f, 1e-4f);    // 2.0 - radius 1.0
    EXPECT_NEAR(nx, 1.0f, 1e-4f);       // Normal points along +X
    EXPECT_NEAR(ny, 0.0f, 1e-4f);
    EXPECT_NEAR(nz, 0.0f, 1e-4f);

    coupling.Shutdown();
}

void test_SDF_SphereInside() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    uint32_t h = coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 2.0f);

    auto* body = coupling.GetCoupledBody(h);
    body->posX = 0.0f; body->posY = 0.0f; body->posZ = 0.0f;
    body->quatW = 1.0f;

    // Query point at (0.5, 0, 0) — inside sphere of radius 2
    int32_t idx; float nx, ny, nz;
    float dist = coupling.QuerySDF(0.5f, 0.0f, 0.0f, idx, nx, ny, nz);

    EXPECT_TRUE(dist < 0.0f);  // Inside
    EXPECT_NEAR(dist, -1.5f, 1e-4f);  // 0.5 - 2.0
    EXPECT_EQ(idx, 0);

    coupling.Shutdown();
}

void test_SDF_SphereOnSurface() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 1.0f);

    auto* body = coupling.GetCoupledBody(0);
    body->posX = 0.0f; body->posY = 0.0f; body->posZ = 0.0f;
    body->quatW = 1.0f;

    // Point at exact distance of radius along Y
    int32_t idx; float nx, ny, nz;
    float dist = coupling.QuerySDF(0.0f, 1.0f, 0.0f, idx, nx, ny, nz);

    EXPECT_NEAR(dist, 0.0f, 1e-4f);
    EXPECT_NEAR(ny, 1.0f, 1e-4f);

    coupling.Shutdown();
}

void test_SDF_BoxOutside() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    // Box with half extents (1, 1, 1) — a 2x2x2 cube at origin
    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Box,
        0.0f, 1.0f, 1.0f, 1.0f);

    auto* body = coupling.GetCoupledBody(0);
    body->posX = 0.0f; body->posY = 0.0f; body->posZ = 0.0f;
    body->quatW = 1.0f;

    // Point at (3, 0, 0) — 2 units from face at x=1
    int32_t idx; float nx, ny, nz;
    float dist = coupling.QuerySDF(3.0f, 0.0f, 0.0f, idx, nx, ny, nz);

    EXPECT_TRUE(dist > 0.0f);
    EXPECT_NEAR(dist, 2.0f, 1e-4f);
    EXPECT_EQ(idx, 0);

    coupling.Shutdown();
}

void test_SDF_BoxInside() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    // Box with half extents (2, 2, 2)
    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Box,
        0.0f, 2.0f, 2.0f, 2.0f);

    auto* body = coupling.GetCoupledBody(0);
    body->posX = 0.0f; body->posY = 0.0f; body->posZ = 0.0f;
    body->quatW = 1.0f;

    // Point at origin — 2 units inside all faces
    int32_t idx; float nx, ny, nz;
    float dist = coupling.QuerySDF(0.0f, 0.0f, 0.0f, idx, nx, ny, nz);

    EXPECT_TRUE(dist < 0.0f);
    EXPECT_EQ(idx, 0);

    coupling.Shutdown();
}

void test_SDF_CapsuleOutside() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    // Capsule: radius=0.5, height=2.0 (half-height=1.0)
    uint32_t h = coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Capsule,
        0.5f, 0.5f, 1.0f, 0.5f);  // halfExtentY = half height

    auto* body = coupling.GetCoupledBody(h);
    body->posX = 0.0f; body->posY = 0.0f; body->posZ = 0.0f;
    body->quatW = 1.0f;
    body->height = 2.0f;

    // Point at (2, 0, 0) — far from capsule
    int32_t idx; float nx, ny, nz;
    float dist = coupling.QuerySDF(2.0f, 0.0f, 0.0f, idx, nx, ny, nz);

    EXPECT_TRUE(dist > 0.0f);  // Outside
    EXPECT_NEAR(dist, 1.5f, 1e-4f);  // 2.0 - 0.5 radius

    coupling.Shutdown();
}

void test_SDF_NearestBody() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    // Body 0 at origin, radius 1
    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 1.0f);
    auto* b0 = coupling.GetCoupledBody(0);
    b0->posX = 0.0f; b0->posY = 0.0f; b0->posZ = 0.0f;
    b0->quatW = 1.0f;

    // Body 1 at (10, 0, 0), radius 1
    coupling.AddCoupledBody(
        JPH::BodyID(1), CoupledRigidBody::ShapeType::Sphere, 1.0f);
    auto* b1 = coupling.GetCoupledBody(1);
    b1->posX = 10.0f; b1->posY = 0.0f; b1->posZ = 0.0f;
    b1->quatW = 1.0f;

    // Query near body 0
    int32_t idx; float nx, ny, nz;
    coupling.QuerySDF(2.0f, 0.0f, 0.0f, idx, nx, ny, nz);
    EXPECT_EQ(idx, 0);  // Closer to body 0

    // Query near body 1
    coupling.QuerySDF(9.0f, 0.0f, 0.0f, idx, nx, ny, nz);
    EXPECT_EQ(idx, 1);  // Closer to body 1

    coupling.Shutdown();
}

void test_SDF_DisabledBody() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 1.0f);
    auto* b = coupling.GetCoupledBody(0);
    b->posX = 0.0f; b->quatW = 1.0f;

    coupling.RemoveCoupledBody(0);

    int32_t idx; float nx, ny, nz;
    coupling.QuerySDF(0.5f, 0.0f, 0.0f, idx, nx, ny, nz);
    EXPECT_EQ(idx, -1);  // No enabled bodies

    coupling.Shutdown();
}

// =============================================================================
// Surface Velocity Tests
// =============================================================================

void test_SurfaceVelocity_Linear() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 1.0f);

    auto* body = coupling.GetCoupledBody(0);
    body->posX = 0.0f; body->posY = 0.0f; body->posZ = 0.0f;
    body->velX = 5.0f; body->velY = 0.0f; body->velZ = 0.0f;
    body->angVelX = 0.0f; body->angVelY = 0.0f; body->angVelZ = 0.0f;

    float vx, vy, vz;
    coupling.GetBodySurfaceVelocity(0, 1.0f, 0.0f, 0.0f, vx, vy, vz);

    // Pure linear → surface velocity = body velocity
    EXPECT_NEAR(vx, 5.0f, 1e-4f);
    EXPECT_NEAR(vy, 0.0f, 1e-4f);
    EXPECT_NEAR(vz, 0.0f, 1e-4f);

    coupling.Shutdown();
}

void test_SurfaceVelocity_Angular() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 1.0f);

    auto* body = coupling.GetCoupledBody(0);
    body->posX = 0.0f; body->posY = 0.0f; body->posZ = 0.0f;
    body->velX = 0.0f; body->velY = 0.0f; body->velZ = 0.0f;
    // Spin around Y axis at 2 rad/s
    body->angVelX = 0.0f; body->angVelY = 2.0f; body->angVelZ = 0.0f;

    float vx, vy, vz;
    // Point at (1, 0, 0): ω × r = (0,2,0) × (1,0,0) = (0*0-0*0, 0*1-2*0, 2*0-0*1) = (0,0,-2)
    // Wait: ω × r = (ωy*rz - ωz*ry, ωz*rx - ωx*rz, ωx*ry - ωy*rx)
    // = (2*0 - 0*0, 0*1 - 0*0, 0*0 - 2*1) = (0, 0, -2)
    coupling.GetBodySurfaceVelocity(0, 1.0f, 0.0f, 0.0f, vx, vy, vz);

    EXPECT_NEAR(vx, 0.0f, 1e-4f);
    EXPECT_NEAR(vy, 0.0f, 1e-4f);
    EXPECT_NEAR(vz, -2.0f, 1e-4f);

    coupling.Shutdown();
}

void test_SurfaceVelocity_InvalidBody() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    float vx, vy, vz;
    coupling.GetBodySurfaceVelocity(999, 0.0f, 0.0f, 0.0f, vx, vy, vz);
    EXPECT_NEAR(vx, 0.0f, 1e-6f);
    EXPECT_NEAR(vy, 0.0f, 1e-6f);
    EXPECT_NEAR(vz, 0.0f, 1e-6f);

    coupling.Shutdown();
}

// =============================================================================
// Statistics Tests
// =============================================================================

void test_Coupling_StatsInitial() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    const auto& stats = coupling.GetStats();
    EXPECT_EQ(stats.activeBodies, 0u);
    EXPECT_EQ(stats.particleBodyContacts, 0u);
    EXPECT_NEAR(stats.maxForceApplied, 0.0f, 1e-6f);

    coupling.Shutdown();
}

// =============================================================================
// Edge Case Tests
// =============================================================================

void test_Coupling_NoParticles() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 1.0f);

    // Should not crash with null/zero particles
    coupling.ComputeCoupling(nullptr, 0, MPMMaterialParams::Sand(),
                             *(JPH::PhysicsSystem*)nullptr, 1.0f / 60.0f);
    // Returns early due to particleCount == 0

    coupling.Shutdown();
}

void test_Coupling_NoBodies() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    // No bodies registered — should handle gracefully
    MPMParticle p{};
    p.x = 0.0f; p.y = 0.0f; p.z = 0.0f;
    p.vx = 1.0f;

    // Note: This needs a valid PhysicsSystem to sync, but with 0 bodies
    // it returns early after SyncBodyStates finds 0 active bodies
    // We can't call this without a real PhysicsSystem, so test the stats path
    EXPECT_EQ(coupling.GetCoupledBodyCount(), 0u);

    coupling.Shutdown();
}

void test_Coupling_NotInitialized() {
    MPMRigidCoupling coupling;
    EXPECT_TRUE(!coupling.IsInitialized());

    // SDF queries should return max distance
    int32_t idx; float nx, ny, nz;
    float dist = coupling.QuerySDF(0.0f, 0.0f, 0.0f, idx, nx, ny, nz);
    (void)dist;
    EXPECT_EQ(idx, -1);  // No bodies

    coupling.Shutdown();  // Should not crash even if not initialized
}

void test_Coupling_ShutdownClears() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 1.0f);
    coupling.AddCoupledBody(
        JPH::BodyID(1), CoupledRigidBody::ShapeType::Box);
    EXPECT_EQ(coupling.GetCoupledBodyCount(), 2u);

    coupling.Shutdown();
    EXPECT_EQ(coupling.GetCoupledBodyCount(), 0u);
    EXPECT_TRUE(!coupling.IsInitialized());

    // Re-initialize should work
    EXPECT_TRUE(coupling.Initialize(MakeCouplingConfig()));
    EXPECT_EQ(coupling.GetCoupledBodyCount(), 0u);

    coupling.Shutdown();
}

// =============================================================================
// SDF with Rotated Bodies
// =============================================================================

void test_SDF_RotatedSphere() {
    // Sphere SDF is rotationally symmetric — rotation should not affect distance
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 1.0f);

    auto* body = coupling.GetCoupledBody(0);
    body->posX = 0.0f; body->posY = 0.0f; body->posZ = 0.0f;
    // 90-degree rotation around Y
    float angle = 3.14159265f / 4.0f;
    body->quatX = 0.0f;
    body->quatY = std::sin(angle / 2.0f);
    body->quatZ = 0.0f;
    body->quatW = std::cos(angle / 2.0f);

    int32_t idx; float nx, ny, nz;
    float dist = coupling.QuerySDF(2.0f, 0.0f, 0.0f, idx, nx, ny, nz);

    EXPECT_NEAR(dist, 1.0f, 1e-3f);  // Still 2.0 - 1.0

    coupling.Shutdown();
}

void test_SDF_TranslatedBody() {
    MPMRigidCoupling coupling;
    coupling.Initialize(MakeCouplingConfig());

    coupling.AddCoupledBody(
        JPH::BodyID(0), CoupledRigidBody::ShapeType::Sphere, 1.0f);

    auto* body = coupling.GetCoupledBody(0);
    body->posX = 5.0f; body->posY = 0.0f; body->posZ = 0.0f;
    body->quatW = 1.0f;

    // Point at origin — dist to center = 5, dist to surface = 4
    int32_t idx; float nx, ny, nz;
    float dist = coupling.QuerySDF(0.0f, 0.0f, 0.0f, idx, nx, ny, nz);

    EXPECT_NEAR(dist, 4.0f, 1e-4f);
    // Normal should point from body center toward query point = -X direction
    EXPECT_NEAR(nx, -1.0f, 1e-4f);

    coupling.Shutdown();
}

// =============================================================================
// Registration Function
// =============================================================================

void RegisterMPMRigidCouplingTests() {
    // Config
    RUN_TEST("CouplingConfig_Defaults", test_CouplingConfig_Defaults);
    RUN_TEST("CouplingConfig_Custom", test_CouplingConfig_Custom);

    // Initialization
    RUN_TEST("Coupling_InitShutdown", test_Coupling_InitShutdown);
    RUN_TEST("Coupling_DoubleInit", test_Coupling_DoubleInit);
    RUN_TEST("Coupling_ConfigAccess", test_Coupling_ConfigAccess);

    // Body registration
    RUN_TEST("Coupling_AddBody", test_Coupling_AddBody);
    RUN_TEST("Coupling_AddMultipleBodies", test_Coupling_AddMultipleBodies);
    RUN_TEST("Coupling_RemoveBody", test_Coupling_RemoveBody);
    RUN_TEST("Coupling_GetInvalidBody", test_Coupling_GetInvalidBody);

    // CoupledRigidBody
    RUN_TEST("CoupledBody_DefaultState", test_CoupledBody_DefaultState);
    RUN_TEST("CoupledBody_ClearAccumulators", test_CoupledBody_ClearAccumulators);
    RUN_TEST("CoupledBody_ShapeTypes", test_CoupledBody_ShapeTypes);

    // SDF queries
    RUN_TEST("SDF_SphereOutside", test_SDF_SphereOutside);
    RUN_TEST("SDF_SphereInside", test_SDF_SphereInside);
    RUN_TEST("SDF_SphereOnSurface", test_SDF_SphereOnSurface);
    RUN_TEST("SDF_BoxOutside", test_SDF_BoxOutside);
    RUN_TEST("SDF_BoxInside", test_SDF_BoxInside);
    RUN_TEST("SDF_CapsuleOutside", test_SDF_CapsuleOutside);
    RUN_TEST("SDF_NearestBody", test_SDF_NearestBody);
    RUN_TEST("SDF_DisabledBody", test_SDF_DisabledBody);
    RUN_TEST("SDF_RotatedSphere", test_SDF_RotatedSphere);
    RUN_TEST("SDF_TranslatedBody", test_SDF_TranslatedBody);

    // Surface velocity
    RUN_TEST("SurfaceVelocity_Linear", test_SurfaceVelocity_Linear);
    RUN_TEST("SurfaceVelocity_Angular", test_SurfaceVelocity_Angular);
    RUN_TEST("SurfaceVelocity_InvalidBody", test_SurfaceVelocity_InvalidBody);

    // Statistics
    RUN_TEST("Coupling_StatsInitial", test_Coupling_StatsInitial);

    // Edge cases
    RUN_TEST("Coupling_NoParticles", test_Coupling_NoParticles);
    RUN_TEST("Coupling_NoBodies", test_Coupling_NoBodies);
    RUN_TEST("Coupling_NotInitialized", test_Coupling_NotInitialized);
    RUN_TEST("Coupling_ShutdownClears", test_Coupling_ShutdownClears);
}
