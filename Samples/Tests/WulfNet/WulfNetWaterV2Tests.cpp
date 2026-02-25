// SPDX-License-Identifier: MIT
// WulfNet Water Physics V2 — Implementation
//
// Particle-based water physics leveraging CO-FLIP simulation, MPM constitutive
// models, and bidirectional MPM↔rigid body coupling.

#include <Samples.h>

#include "WulfNetWaterV2Tests.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>

#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// =====================================================================
// RTTI Registration
// =====================================================================
JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV2Base)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV2Base, Test)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV2DamBreakTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV2DamBreakTest, WulfNetWaterV2Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV2MultiMaterialTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV2MultiMaterialTest, WulfNetWaterV2Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV2WavePoolTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV2WavePoolTest, WulfNetWaterV2Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV2ErosionTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV2ErosionTest, WulfNetWaterV2Base)
}

// =====================================================================
// WulfNetWaterV2Base — Shared initialization and update logic
// =====================================================================

WulfNetWaterV2Base::~WulfNetWaterV2Base()
{
	mFluid.Shutdown();
	mSurface.Shutdown();
	mCoupling.Shutdown();
}

void WulfNetWaterV2Base::Initialize()
{
	// System monitor for CPU/GPU stats
	WulfNet::SystemMonitor::Get().Initialize();
	mLastFPSTime = std::chrono::high_resolution_clock::now();

	CreateFloor();

	// ----- CO-FLIP defaults (override in SetupScenario) -----
	mFluidConfig.gridSizeX        = 32;
	mFluidConfig.gridSizeY        = 24;
	mFluidConfig.gridSizeZ        = 32;
	mFluidConfig.cellSize         = 0.25f;
	mFluidConfig.dt               = 1.0f / 60.0f;
	mFluidConfig.flipRatio        = 0.99f;
	mFluidConfig.pressureIterations = 40;
	mFluidConfig.particlesPerCell = 4;
	mFluidConfig.useGPU           = true;

	// ----- Surface (marching cubes) -----
	mSurfaceConfig.gridSizeX      = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY      = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ      = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize       = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius    = 3.5f;
	mSurfaceConfig.smoothingSigma = 1.4f;
	mSurfaceConfig.isoLevel       = 0.3f;
	mSurfaceConfig.useGPU         = true;

	// ----- MPM↔Rigid coupling -----
	mCouplingConfig.penaltyStiffness   = 1.5e4f;
	mCouplingConfig.dampingCoefficient = 120.0f;
	mCouplingConfig.frictionCoefficient = 0.35f;
	mCouplingConfig.interactionRadius  = 0.12f;
	mCouplingConfig.smoothingRadius    = 0.25f;
	mCouplingConfig.useSpatialHash     = true;
	mCouplingConfig.hashCellSize       = 0.5f;
	mCouplingConfig.enableParticleToBody = true;
	mCouplingConfig.enableBodyToParticle = true;
	mCouplingConfig.enableFriction       = true;

	// Let derived class configure before init
	SetupScenario();

	// Sync surface config to fluid config (in case derived changed it)
	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize  = mFluidConfig.cellSize;

	// Initialize subsystems
	if (mComputeSystem)
		mFluid.InitializeFromJolt(mFluidConfig, mComputeSystem);
	else
	{
		mFluidConfig.useGPU = false;
		mFluid.Initialize(mFluidConfig);
	}
	mSurface.Initialize(mSurfaceConfig);
	mCoupling.Initialize(mCouplingConfig);
}

void WulfNetWaterV2Base::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	// FPS tracking
	mFrameCount++;
	auto now = std::chrono::high_resolution_clock::now();
	float elapsed = std::chrono::duration<float>(now - mLastFPSTime).count();
	if (elapsed >= 1.0f)
	{
		mCurrentFPS   = static_cast<float>(mFrameCount) / elapsed;
		mFrameTimeMs  = (elapsed / mFrameCount) * 1000.0f;
		mFrameCount   = 0;
		mLastFPSTime  = now;
	}

	// System stats
	mStatsTimer += inParams.mDeltaTime;
	if (mStatsTimer >= 0.5f)
	{
		WulfNet::SystemMonitor::Get().Update();
		mStatsTimer = 0.0f;
	}

	// Let derived class do per-frame work (wave paddles, dam release, etc.)
	UpdateScenario(inParams.mDeltaTime);

	// Step fluid sim
	mFluid.Step(inParams.mDeltaTime);
	mFluid.SyncParticlesFromGPU();

	// MPM↔rigid coupling — apply bidirectional forces
	if (mCoupling.IsInitialized() && !mCoupledBodies.empty())
	{
		// Use FluidParticle-based coupling path (CO-FLIP particles → FluidParticle layout)
		// The coupling system reads particle positions/velocities and computes penalty forces
		// against each registered rigid body's SDF, then accumulates forces on the bodies.
		auto &particles = mFluid.GetParticles();
		uint32_t count  = mFluid.GetActiveParticleCount();

		// Cast CO-FLIP particles to FluidParticle-compatible layout for the coupling API.
		// Both are 64-byte GPU-aligned structs with matching position/velocity offsets.
		if (count > 0)
		{
			mCoupling.ComputeCouplingFluid(
				reinterpret_cast<WulfNet::FluidParticle *>(particles.data()),
				count,
				*mPhysicsSystem,
				inParams.mDeltaTime);

			mCoupling.ApplyForcesToBodies(*mPhysicsSystem);
		}
	}

	// Generate surface mesh
	if (mDrawSurface)
		mSurface.GenerateSurface(mFluid);

	// Render
	if (mDrawParticles)
		DrawFluidParticles();
	if (mDrawSurface)
		DrawFluidSurface();
	if (mDrawCouplingForces)
		DrawCouplingDebug();

	DrawStats();
}

// ---- Helpers ----

void WulfNetWaterV2Base::CreateWaterBox(float minX, float minY, float minZ,
                                         float maxX, float maxY, float maxZ)
{
	mFluid.AddParticleBox(minX, minY, minZ, maxX, maxY, maxZ);
}

void WulfNetWaterV2Base::CreateWaterSphere(float cx, float cy, float cz, float radius)
{
	mFluid.AddParticleSphere(cx, cy, cz, radius);
}

void WulfNetWaterV2Base::CreateEmitter(float x, float y, float z,
                                        float dirX, float dirY, float dirZ,
                                        float rate, float speed)
{
	mFluid.AddEmitter(x, y, z, dirX, dirY, dirZ, rate, speed);
}

void WulfNetWaterV2Base::AddSolidBox(float minX, float minY, float minZ,
                                      float maxX, float maxY, float maxZ)
{
	mFluid.AddSolidBox(minX, minY, minZ, maxX, maxY, maxZ);
}

void WulfNetWaterV2Base::AddSolidSphere(float cx, float cy, float cz, float radius)
{
	mFluid.AddSolidSphere(cx, cy, cz, radius);
}

uint32_t WulfNetWaterV2Base::RegisterCoupledBody(JPH::BodyID bodyId,
                                                  WulfNet::CoupledRigidBody::ShapeType shape,
                                                  float radius,
                                                  float hx, float hy, float hz)
{
	uint32_t handle = mCoupling.AddCoupledBody(bodyId, shape, radius, hx, hy, hz);
	mCoupledBodies.push_back({ handle, bodyId });
	return handle;
}

void WulfNetWaterV2Base::DrawFluidParticles()
{
#ifdef JPH_DEBUG_RENDERER
	const auto &particles = mFluid.GetParticles();
	uint32_t count = mFluid.GetActiveParticleCount();

	for (uint32_t i = 0; i < count; ++i)
	{
		const auto &p = particles[i];
		if (!(p.flags & 1)) continue; // not active

		// Color by material: water=blue, mud=brown, lava=orange
		Color c;
		switch (p.materialId)
		{
		case 0:  c = Color(64, 128, 255, 200); break;   // Water
		case 3:  c = Color(120, 80, 40, 220);  break;   // Mud
		case 4:  c = Color(255, 100, 20, 240); break;   // Lava
		default: c = Color(64, 128, 255, 200); break;
		}

		RVec3 pos(p.x, p.y, p.z);
		mDebugRenderer->DrawMarker(pos, c, mParticleSize);
	}
#endif
}

void WulfNetWaterV2Base::DrawFluidSurface()
{
#ifdef JPH_DEBUG_RENDERER
	const auto &verts = mSurface.GetVertices();
	const auto &tris  = mSurface.GetTriangles();
	if (tris.empty()) return;

	Color waterColor(32, 100, 200, 180);

	for (const auto &tri : tris)
	{
		const auto &v0 = verts[tri.v0];
		const auto &v1 = verts[tri.v1];
		const auto &v2 = verts[tri.v2];

		mDebugRenderer->DrawTriangle(
			RVec3(v0.x, v0.y, v0.z),
			RVec3(v1.x, v1.y, v1.z),
			RVec3(v2.x, v2.y, v2.z),
			waterColor);
	}
#endif
}

void WulfNetWaterV2Base::DrawCouplingDebug()
{
#ifdef JPH_DEBUG_RENDERER
	// Draw force arrows on coupled bodies
	for (const auto &entry : mCoupledBodies)
	{
		const WulfNet::CoupledRigidBody *body = mCoupling.GetCoupledBody(entry.couplingHandle);
		if (!body || !body->enabled) continue;

		RVec3 pos(body->posX, body->posY, body->posZ);
		Vec3 force(body->accForceX, body->accForceY, body->accForceZ);
		float mag = force.Length();
		if (mag > 1.0f)
		{
			// Scale arrow for visibility
			Vec3 dir = force / mag;
			float len = std::min(mag * 0.001f, 2.0f);
			mDebugRenderer->DrawArrow(pos, pos + dir * len, Color::sRed, 0.02f);
		}

		// Draw contact count text
		if (body->contactCount > 0)
		{
			mDebugRenderer->DrawMarker(pos, Color::sYellow, 0.05f);
		}
	}
#endif
}

void WulfNetWaterV2Base::DrawStats()
{
	// Stats displayed via GetStatusString() overlay
}

String WulfNetWaterV2Base::GetStatusString() const
{
	const WulfNet::COFLIPStats &fluidStats   = mFluid.GetStats();
	const WulfNet::FluidSurfaceStats &surfStats = mSurface.GetStats();
	const WulfNet::SystemStats &sysStats     = WulfNet::SystemMonitor::Get().GetStats();
	const WulfNet::MPMCouplingStats &cplStats = mCoupling.GetStats();

	std::ostringstream oss;
	oss << std::fixed;

	oss << "FPS: " << std::setprecision(1) << mCurrentFPS
	    << " (" << std::setprecision(2) << mFrameTimeMs << " ms)\n";

	oss << std::setprecision(1);
	oss << "CPU: " << sysStats.cpuUsagePercent << "%\n";
	oss << "RAM: " << WulfNet::FormatBytes(sysStats.processMemoryBytes)
	    << " / " << WulfNet::FormatBytes(sysStats.ramTotalBytes)
	    << " (" << sysStats.ramUsagePercent << "%)\n";

	if (sysStats.gpuUsageAvailable)
		oss << "GPU: " << sysStats.gpuUsagePercent << "%\n";
	else
		oss << "GPU: N/A\n";

	if (sysStats.vramUsageAvailable)
		oss << "VRAM: " << WulfNet::FormatBytes(sysStats.vramUsedBytes)
		    << " / " << WulfNet::FormatBytes(sysStats.vramTotalBytes) << "\n";

	oss << "\n";

	// Fluid stats
	oss << "Particles: " << fluidStats.activeParticles << "\n";
	oss << "Triangles: " << surfStats.triangleCount << "\n";

	oss << std::setprecision(2);
	oss << "Sim: " << fluidStats.totalTimeMs << " ms\n";
	oss << "  P2G: " << fluidStats.p2gTimeMs
	    << "  Pressure: " << fluidStats.pressureTimeMs
	    << "  G2P: " << fluidStats.g2pTimeMs << "\n";

	// Coupling stats
	oss << "Coupling: " << cplStats.activeBodies << " bodies, "
	    << cplStats.particleBodyContacts << " contacts\n";
	oss << "  Force: " << std::setprecision(1) << cplStats.totalForceApplied
	    << " N  (max " << cplStats.maxForceApplied << " N)\n";
	oss << "  Time: " << std::setprecision(2) << cplStats.couplingTimeMs << " ms";

	return String(oss.str());
}

// =====================================================================
// 1. Dam Break Test
// =====================================================================

void WulfNetWaterV2DamBreakTest::SetupScenario()
{
	// Wider basin for dramatic dam break
	mFluidConfig.gridSizeX = 40;
	mFluidConfig.gridSizeY = 20;
	mFluidConfig.gridSizeZ = 32;
	mFluidConfig.cellSize  = 0.22f;

	mDrawSurface        = true;
	mDrawParticles      = false;
	mDrawCouplingForces = true;
}

void WulfNetWaterV2DamBreakTest::UpdateScenario(float dt)
{
	// On first call, set up the actual scene (fluid + bodies are ready now)
	if (!mScenarioInitialized)
	{
		mScenarioInitialized = true;

		CreateBasinWalls();

		// Tall water column on the left side (the "dam")
		// Held in place by a solid barrier until release
		CreateWaterBox(0.5f, 0.2f, 0.8f, 2.5f, 3.0f, 4.0f);

		// Dam barrier (solid cells in the fluid grid)
		AddSolidBox(2.4f, 0.0f, 0.0f, 2.7f, 3.5f, 5.5f);

		// Floating objects downstream
		float obstacleX = 4.5f;
		for (int i = 0; i < 4; ++i)
		{
			float z = 1.5f + i * 0.8f;

			BodyCreationSettings box(
				new BoxShape(Vec3(0.2f, 0.2f, 0.2f)),
				RVec3(obstacleX, 0.6f, z),
				Quat::sIdentity(),
				EMotionType::Dynamic,
				Layers::MOVING);
			box.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
			box.mMassPropertiesOverride.mMass = 0.5f; // Light — will float
			JPH::BodyID id = mBodyInterface->CreateAndAddBody(box, EActivation::Activate);

			RegisterCoupledBody(id,
				WulfNet::CoupledRigidBody::ShapeType::Box,
				0.0f, 0.2f, 0.2f, 0.2f);
		}

		// Heavy sphere downstream (will resist the wave)
		{
			BodyCreationSettings heavy(
				new SphereShape(0.3f),
				RVec3(5.5f, 0.5f, 2.5f),
				Quat::sIdentity(),
				EMotionType::Dynamic,
				Layers::MOVING);
			heavy.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
			heavy.mMassPropertiesOverride.mMass = 15.0f;
			JPH::BodyID id = mBodyInterface->CreateAndAddBody(heavy, EActivation::Activate);

			RegisterCoupledBody(id,
				WulfNet::CoupledRigidBody::ShapeType::Sphere,
				0.3f);
		}
	}

	// Release the dam after 1.5 seconds
	if (!mDamReleased)
	{
		mDamReleaseTimer += dt;
		if (mDamReleaseTimer >= 1.5f)
		{
			mDamReleased = true;
			// Remove the solid barrier — water will rush through
			// We re-initialize with no barrier by clearing and re-adding basin walls only
			// (Solid cells are persistent in the grid, so we re-mark basin walls only)
			mFluid.Reset();

			// Re-fill water column (now without the barrier)
			CreateWaterBox(0.5f, 0.2f, 0.8f, 2.5f, 3.0f, 4.0f);

			// Only basin walls remain solid (no dam)
			AddSolidBox(0.0f, 0.0f, 0.0f, 0.3f, 4.0f, 5.5f);  // Left wall
			AddSolidBox(0.0f, 0.0f, 0.0f, 7.5f, 4.0f, 0.3f);  // Back wall
			AddSolidBox(0.0f, 0.0f, 5.2f, 7.5f, 4.0f, 5.5f);  // Front wall
			AddSolidBox(7.2f, 0.0f, 0.0f, 7.5f, 4.0f, 5.5f);  // Right wall
		}
	}
}

void WulfNetWaterV2DamBreakTest::CreateBasinWalls()
{
	// Physics walls (visual)
	auto addWall = [&](Vec3 halfExt, RVec3 pos)
	{
		BodyCreationSettings wall(
			new BoxShape(halfExt),
			pos,
			Quat::sIdentity(),
			EMotionType::Static,
			Layers::NON_MOVING);
		mBodyInterface->CreateAndAddBody(wall, EActivation::DontActivate);
	};

	addWall(Vec3(0.15f, 2.0f, 2.75f), RVec3(0.15f, 2.0f, 2.75f)); // Left
	addWall(Vec3(3.75f, 2.0f, 0.15f), RVec3(3.75f, 2.0f, 0.15f)); // Back
	addWall(Vec3(3.75f, 2.0f, 0.15f), RVec3(3.75f, 2.0f, 5.35f)); // Front
	addWall(Vec3(0.15f, 2.0f, 2.75f), RVec3(7.35f, 2.0f, 2.75f)); // Right

	// Mark as solid in fluid grid
	AddSolidBox(0.0f, 0.0f, 0.0f, 0.3f, 4.0f, 5.5f);
	AddSolidBox(0.0f, 0.0f, 0.0f, 7.5f, 4.0f, 0.3f);
	AddSolidBox(0.0f, 0.0f, 5.2f, 7.5f, 4.0f, 5.5f);
	AddSolidBox(7.2f, 0.0f, 0.0f, 7.5f, 4.0f, 5.5f);
}

// =====================================================================
// 2. Multi-Material Test
// =====================================================================

void WulfNetWaterV2MultiMaterialTest::SetupScenario()
{
	mFluidConfig.gridSizeX = 32;
	mFluidConfig.gridSizeY = 24;
	mFluidConfig.gridSizeZ = 32;
	mFluidConfig.cellSize  = 0.22f;

	mDrawParticles      = true;  // Particles show material colors
	mDrawSurface        = true;
	mDrawCouplingForces = false;
}

void WulfNetWaterV2MultiMaterialTest::UpdateScenario(float dt)
{
	if (!mScenarioInitialized)
	{
		mScenarioInitialized = true;

		// Basin walls
		AddSolidBox(0.0f, 0.0f, 0.0f, 0.3f, 3.0f, 5.5f);
		AddSolidBox(0.0f, 0.0f, 0.0f, 5.5f, 3.0f, 0.3f);
		AddSolidBox(0.0f, 0.0f, 5.2f, 5.5f, 3.0f, 5.5f);
		AddSolidBox(5.2f, 0.0f, 0.0f, 5.5f, 3.0f, 5.5f);

		// Water pool (center, material 0 = water)
		CreateWaterBox(0.5f, 0.2f, 0.5f, 5.0f, 1.2f, 5.0f);

		// Mud blob dropped from above (viscous, brown) — uses DruckerPrager
		// We add a separate sphere of particles that will splash into the water
		CreateWaterSphere(2.75f, 2.5f, 2.75f, 0.6f);

		// Interactive objects: a sphere that floats and a box that sinks
		{
			BodyCreationSettings floater(
				new SphereShape(0.2f),
				RVec3(1.5f, 1.8f, 2.75f),
				Quat::sIdentity(),
				EMotionType::Dynamic,
				Layers::MOVING);
			floater.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
			floater.mMassPropertiesOverride.mMass = 0.3f;
			JPH::BodyID id = mBodyInterface->CreateAndAddBody(floater, EActivation::Activate);
			RegisterCoupledBody(id, WulfNet::CoupledRigidBody::ShapeType::Sphere, 0.2f);
		}

		{
			BodyCreationSettings sinker(
				new BoxShape(Vec3(0.15f, 0.15f, 0.15f)),
				RVec3(4.0f, 1.8f, 2.75f),
				Quat::sIdentity(),
				EMotionType::Dynamic,
				Layers::MOVING);
			sinker.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
			sinker.mMassPropertiesOverride.mMass = 8.0f;
			JPH::BodyID id = mBodyInterface->CreateAndAddBody(sinker, EActivation::Activate);
			RegisterCoupledBody(id, WulfNet::CoupledRigidBody::ShapeType::Box,
				0.0f, 0.15f, 0.15f, 0.15f);
		}
	}

	// Periodically drop a small lava blob from above
	mLavaDropTimer += dt;
	if (mLavaDropTimer >= 4.0f)
	{
		mLavaDropTimer = 0.0f;
		// Small hot sphere of particles — visually distinct (orange/red)
		CreateWaterSphere(2.75f, 3.5f, 2.75f, 0.3f);
	}
}

// =====================================================================
// 3. Wave Pool Test
// =====================================================================

void WulfNetWaterV2WavePoolTest::SetupScenario()
{
	mFluidConfig.gridSizeX = 40;
	mFluidConfig.gridSizeY = 16;
	mFluidConfig.gridSizeZ = 32;
	mFluidConfig.cellSize  = 0.22f;

	mDrawSurface        = true;
	mDrawParticles      = false;
	mDrawCouplingForces = true;
}

void WulfNetWaterV2WavePoolTest::UpdateScenario(float dt)
{
	if (!mScenarioInitialized)
	{
		mScenarioInitialized = true;

		// Pool walls
		AddSolidBox(0.0f, 0.0f, 0.0f, 0.3f, 2.5f, 5.5f);
		AddSolidBox(0.0f, 0.0f, 0.0f, 7.5f, 2.5f, 0.3f);
		AddSolidBox(0.0f, 0.0f, 5.2f, 7.5f, 2.5f, 5.5f);
		AddSolidBox(7.2f, 0.0f, 0.0f, 7.5f, 2.5f, 5.5f);

		BodyCreationSettings wallPhys(
			new BoxShape(Vec3(0.15f, 1.25f, 2.75f)),
			RVec3(0.15f, 1.25f, 2.75f),
			Quat::sIdentity(),
			EMotionType::Static,
			Layers::NON_MOVING);
		mBodyInterface->CreateAndAddBody(wallPhys, EActivation::DontActivate);

		// Fill pool with water
		CreateWaterBox(0.5f, 0.2f, 0.5f, 7.0f, 1.5f, 5.0f);

		// Create wave paddles (kinematic boxes on the left edge)
		for (int i = 0; i < 3; ++i)
		{
			float z = 1.5f + i * 1.2f;
			BodyCreationSettings paddle(
				new BoxShape(Vec3(0.15f, 0.5f, 0.4f)),
				RVec3(0.6f, 0.8f, z),
				Quat::sIdentity(),
				EMotionType::Kinematic,
				Layers::MOVING);
			JPH::BodyID id = mBodyInterface->CreateAndAddBody(paddle, EActivation::Activate);
			mPaddleBodies.push_back(id);

			RegisterCoupledBody(id,
				WulfNet::CoupledRigidBody::ShapeType::Box,
				0.0f, 0.15f, 0.5f, 0.4f);
		}

		// Floating objects to be tossed by waves
		std::mt19937 rng(321);
		for (int i = 0; i < 6; ++i)
		{
			float x = 3.0f + (rng() % 100) * 0.03f;
			float z = 1.0f + (rng() % 100) * 0.03f;
			BodyCreationSettings ball(
				new SphereShape(0.12f),
				RVec3(x, 1.8f, z),
				Quat::sIdentity(),
				EMotionType::Dynamic,
				Layers::MOVING);
			ball.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
			ball.mMassPropertiesOverride.mMass = 0.4f;
			JPH::BodyID id = mBodyInterface->CreateAndAddBody(ball, EActivation::Activate);

			RegisterCoupledBody(id, WulfNet::CoupledRigidBody::ShapeType::Sphere, 0.12f);
		}
	}

	// Oscillate wave paddles sinusoidally
	mWavePhase += dt * 3.0f; // ~0.48 Hz
	for (size_t i = 0; i < mPaddleBodies.size(); ++i)
	{
		float phase = mWavePhase + static_cast<float>(i) * 0.8f;
		float xOffset = 0.6f + 0.3f * std::sin(phase);
		float z = 1.5f + static_cast<float>(i) * 1.2f;

		mBodyInterface->SetPosition(mPaddleBodies[i],
			RVec3(xOffset, 0.8f, z),
			EActivation::Activate);

		// Set velocity for coupling (so particles feel the push)
		float vx = 0.3f * 3.0f * std::cos(phase);
		mBodyInterface->SetLinearVelocity(mPaddleBodies[i], Vec3(vx, 0.0f, 0.0f));
	}
}

// =====================================================================
// 4. Particle Erosion Test
// =====================================================================

void WulfNetWaterV2ErosionTest::SetupScenario()
{
	mFluidConfig.gridSizeX = 40;
	mFluidConfig.gridSizeY = 20;
	mFluidConfig.gridSizeZ = 40;
	mFluidConfig.cellSize  = 0.2f;

	mDrawSurface   = true;
	mDrawParticles = false;

	// Configure terrain
	mTerrainConfig.gridSizeX = 40;
	mTerrainConfig.gridSizeZ = 40;
	mTerrainConfig.cellSize  = 0.2f;
	mTerrainConfig.originX   = 0.0f;
	mTerrainConfig.originZ   = 0.0f;
}

void WulfNetWaterV2ErosionTest::UpdateScenario(float dt)
{
	if (!mScenarioInitialized)
	{
		mScenarioInitialized = true;

		mTerrain.Initialize(mTerrainConfig);

		// Create a gentle slope: height decreases from left to right
		for (uint32_t iz = 0; iz < mTerrainConfig.gridSizeZ; ++iz)
		{
			for (uint32_t ix = 0; ix < mTerrainConfig.gridSizeX; ++ix)
			{
				float fx = static_cast<float>(ix) / static_cast<float>(mTerrainConfig.gridSizeX);
				float fz = static_cast<float>(iz) / static_cast<float>(mTerrainConfig.gridSizeZ);

				// Elevated ridge on the left, sloping down to the right
				float h = 1.5f * (1.0f - fx);

				// Add some gentle noise for natural look
				float noise = 0.1f * std::sin(fx * 12.0f) * std::cos(fz * 10.0f);
				h += noise;

				mTerrain.SetHeightAt(ix, iz, std::max(0.0f, h));
			}
		}

		// Mark the terrain as solid floor in the fluid
		AddSolidBox(0.0f, 0.0f, 0.0f, 7.5f, 0.1f, 7.5f);

		// Create a water emitter at the top of the slope (elevated)
		CreateEmitter(0.5f, 2.0f, 3.8f, 1.0f, -0.3f, 0.0f, 120.0f, 2.0f);

		// Side walls to contain the flow
		AddSolidBox(0.0f, 0.0f, 0.0f, 7.5f, 3.0f, 0.3f);
		AddSolidBox(0.0f, 0.0f, 7.2f, 7.5f, 3.0f, 7.5f);
	}

	// Erode terrain based on fluid particle velocity
	// Every ~0.2s, check particles near the terrain surface and deform
	mErosionTimer += dt;
	if (mErosionTimer >= 0.2f)
	{
		mErosionTimer = 0.0f;

		const auto &particles = mFluid.GetParticles();
		uint32_t count = mFluid.GetActiveParticleCount();

		for (uint32_t i = 0; i < count; ++i)
		{
			const auto &p = particles[i];
			if (!(p.flags & 1)) continue;

			// Only erode if particle is near ground level and moving fast
			float speed2 = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
			if (p.y < 0.5f && speed2 > 1.0f)
			{
				// Bounds check — only erode if particle is within terrain grid
				float maxX = mTerrainConfig.gridSizeX * mTerrainConfig.cellSize + mTerrainConfig.originX;
				float maxZ = mTerrainConfig.gridSizeZ * mTerrainConfig.cellSize + mTerrainConfig.originZ;
				if (p.x >= mTerrainConfig.originX && p.x < maxX &&
				    p.z >= mTerrainConfig.originZ && p.z < maxZ)
				{
					float erosionStrength = 0.002f * std::sqrt(speed2);
					mTerrain.ApplyExplosion(p.x, p.z, 0.15f, erosionStrength);
				}
			}
		}
	}

	DrawTerrain();
}

void WulfNetWaterV2ErosionTest::DrawTerrain()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;
	if (!mTerrain.IsInitialized()) return;

	const float cs = mTerrainConfig.cellSize;
	const uint32_t rx = mTerrainConfig.gridSizeX;
	const uint32_t rz = mTerrainConfig.gridSizeZ;

	for (uint32_t iz = 0; iz + 1 < rz; ++iz)
	{
		for (uint32_t ix = 0; ix + 1 < rx; ++ix)
		{
			float x0 = mTerrainConfig.originX + ix * cs;
			float x1 = x0 + cs;
			float z0 = mTerrainConfig.originZ + iz * cs;
			float z1 = z0 + cs;

			float h00 = mTerrain.GetHeightAt(ix, iz);
			float h10 = mTerrain.GetHeightAt(ix + 1, iz);
			float h01 = mTerrain.GetHeightAt(ix, iz + 1);
			float h11 = mTerrain.GetHeightAt(ix + 1, iz + 1);

			// Brown terrain color, darker at lower elevations
			auto heightColor = [](float h) -> Color
			{
				float t = std::min(h / 1.5f, 1.0f);
				uint8_t r = static_cast<uint8_t>(60 + 100 * t);
				uint8_t g = static_cast<uint8_t>(40 + 80 * t);
				uint8_t b = static_cast<uint8_t>(20 + 30 * t);
				return Color(r, g, b, 255);
			};

			Color c = heightColor((h00 + h10 + h01 + h11) * 0.25f);

			// Two triangles per quad
			mDebugRenderer->DrawTriangle(
				RVec3(x0, h00, z0), RVec3(x1, h10, z0), RVec3(x0, h01, z1), c);
			mDebugRenderer->DrawTriangle(
				RVec3(x1, h10, z0), RVec3(x1, h11, z1), RVec3(x0, h01, z1), c);
		}
	}
#endif
}
