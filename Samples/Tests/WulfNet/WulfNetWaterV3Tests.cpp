// SPDX-License-Identifier: MIT
// WulfNet Water Physics V3 — Implementation
//
// MPM FluidSystem as water: every particle is a discrete physical drop
// with mass, velocity, density, temperature, and material properties.

#include <Samples.h>

#include "WulfNetWaterV3Tests.h"

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
JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV3Base)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV3Base, Test)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV3OceanSwellTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV3OceanSwellTest, WulfNetWaterV3Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV3ViscousCascadeTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV3ViscousCascadeTest, WulfNetWaterV3Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV3ThermalConvectionTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV3ThermalConvectionTest, WulfNetWaterV3Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV3SprayFoamTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV3SprayFoamTest, WulfNetWaterV3Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV3ObstacleCourseTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV3ObstacleCourseTest, WulfNetWaterV3Base)
}

// =====================================================================
// Utility
// =====================================================================
static float Clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// =====================================================================
// WulfNetWaterV3Base — Shared initialization
// =====================================================================

WulfNetWaterV3Base::~WulfNetWaterV3Base()
{
	mFluid.Shutdown();
	mSurface.Shutdown();
}

void WulfNetWaterV3Base::Initialize()
{
	WulfNet::SystemMonitor::Get().Initialize();
	mLastFPSTime = std::chrono::high_resolution_clock::now();

	CreateFloor();

	// ----- FluidSystem defaults — optimized for larger particles -----
	mFluidConfig.gridResolutionX = 40;
	mFluidConfig.gridResolutionY = 20;
	mFluidConfig.gridResolutionZ = 40;
	mFluidConfig.cellSize        = 0.18f;
	mFluidConfig.maxParticles    = 150000;
	mFluidConfig.particlesPerCell = 4;
	mFluidConfig.gravity         = -9.81f;
	mFluidConfig.flipRatio       = 0.95f;
	mFluidConfig.pressureIterations = 50;
	mFluidConfig.substeps        = 1;
	mFluidConfig.maxTimestep     = 1.0f / 60.0f;
	mFluidConfig.useGPU          = true;
	mFluidConfig.enableSleeping  = true;
	mFluidConfig.sleepThreshold  = 0.002f;
	mFluidConfig.enableSpatialHash = true;

	// Bounds (computed from grid)
	mFluidConfig.boundsMinX = 0.0f;
	mFluidConfig.boundsMinY = 0.0f;
	mFluidConfig.boundsMinZ = 0.0f;
	mFluidConfig.boundsMaxX = mFluidConfig.gridResolutionX * mFluidConfig.cellSize;
	mFluidConfig.boundsMaxY = mFluidConfig.gridResolutionY * mFluidConfig.cellSize;
	mFluidConfig.boundsMaxZ = mFluidConfig.gridResolutionZ * mFluidConfig.cellSize;

	// ----- Surface -----
	mSurfaceConfig.gridSizeX      = mFluidConfig.gridResolutionX;
	mSurfaceConfig.gridSizeY      = mFluidConfig.gridResolutionY;
	mSurfaceConfig.gridSizeZ      = mFluidConfig.gridResolutionZ;
	mSurfaceConfig.cellSize       = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius    = 3.0f;
	mSurfaceConfig.smoothingSigma = 1.2f;
	mSurfaceConfig.isoLevel       = 0.3f;
	mSurfaceConfig.useGPU         = true;

	// Let derived class configure before init
	SetupScenario();

	// Sync surface config
	mSurfaceConfig.gridSizeX = mFluidConfig.gridResolutionX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridResolutionY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridResolutionZ;
	mSurfaceConfig.cellSize  = mFluidConfig.cellSize;

	// Initialize subsystems
	mFluid.Initialize(mFluidConfig);
	mSurface.Initialize(mSurfaceConfig);

	// Register material palette
	mWaterMaterialId = mFluid.AddMaterial(WulfNet::FluidMaterial::Water());
	mOilMaterialId   = mFluid.AddMaterial(WulfNet::FluidMaterial::Oil());
	mHoneyMaterialId = mFluid.AddMaterial(WulfNet::FluidMaterial::Honey());
	mMudMaterialId   = mFluid.AddMaterial(WulfNet::FluidMaterial::Mud());
	mLavaMaterialId  = mFluid.AddMaterial(WulfNet::FluidMaterial::Lava());
}

void WulfNetWaterV3Base::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	// FPS
	mFrameCount++;
	auto now = std::chrono::high_resolution_clock::now();
	float elapsed = std::chrono::duration<float>(now - mLastFPSTime).count();
	if (elapsed >= 1.0f)
	{
		mCurrentFPS  = static_cast<float>(mFrameCount) / elapsed;
		mFrameTimeMs = (elapsed / mFrameCount) * 1000.0f;
		mFrameCount  = 0;
		mLastFPSTime = now;
	}

	mStatsTimer += inParams.mDeltaTime;
	if (mStatsTimer >= 0.5f)
	{
		WulfNet::SystemMonitor::Get().Update();
		mStatsTimer = 0.0f;
	}

	// Derived per-frame logic
	UpdateScenario(inParams.mDeltaTime);

	// Step the MPM fluid system
	mFluid.Step(inParams.mDeltaTime);

	// Generate surface mesh from particle positions
	// FluidSurface can splat FluidParticle positions directly
	if (mDrawSurface)
	{
		mSurface.ClearDensity();
		const WulfNet::FluidParticle *particles = mFluid.GetParticles();
		uint32_t count = mFluid.GetParticleCount();
		for (uint32_t i = 0; i < count; ++i)
		{
			const auto &p = particles[i];
			if (!WulfNet::HasFlag(p.flags, WulfNet::ParticleFlags::Active))
				continue;
			mSurface.SplatParticle(p.x, p.y, p.z, 1.0f);
		}
		mSurface.SmoothDensity();
		mSurface.ExtractSurface();
	}

	// Render
	if (mDrawParticles)  DrawParticles();
	if (mDrawSurface)    DrawSurfaceMesh();
	if (mDrawGrid)       DrawGridSlice(1.0f);
}

// ---- Helpers ----

void WulfNetWaterV3Base::AddWaterBox(float minX, float minY, float minZ,
                                      float maxX, float maxY, float maxZ,
                                      uint32_t materialId)
{
	mFluid.AddParticleBox(minX, minY, minZ, maxX, maxY, maxZ, materialId);
}

void WulfNetWaterV3Base::AddWaterSphere(float cx, float cy, float cz,
                                         float radius, uint32_t materialId)
{
	mFluid.AddParticleSphere(cx, cy, cz, radius, materialId);
}

uint32_t WulfNetWaterV3Base::AddEmitter(const WulfNet::FluidEmitter &emitter)
{
	return mFluid.AddEmitter(emitter);
}

uint32_t WulfNetWaterV3Base::AddCollider(const WulfNet::FluidCollider &collider)
{
	return mFluid.AddCollider(collider);
}

uint32_t WulfNetWaterV3Base::AddBuoyancyObject(uint32_t bodyId, float density,
                                                float volume, float drag)
{
	WulfNet::BuoyancyObject bo;
	bo.bodyId           = bodyId;
	bo.density          = density;
	bo.volume           = volume;
	bo.dragCoefficient  = drag;
	return mFluid.AddBuoyancyObject(bo);
}

void WulfNetWaterV3Base::DrawParticles()
{
#ifdef JPH_DEBUG_RENDERER
	const WulfNet::FluidParticle *particles = mFluid.GetParticles();
	uint32_t count = mFluid.GetParticleCount();

	for (uint32_t i = 0; i < count; ++i)
	{
		const auto &p = particles[i];
		if (!WulfNet::HasFlag(p.flags, WulfNet::ParticleFlags::Active))
			continue;

		Color c;
		if (mColorByMaterial)
		{
			// Decode RGBA from material color
			const WulfNet::FluidMaterial *mat = mFluid.GetMaterial(p.materialId);
			if (mat)
			{
				uint32_t rgba = mat->color;
				c = Color(
					static_cast<uint8_t>((rgba >> 24) & 0xFF),
					static_cast<uint8_t>((rgba >> 16) & 0xFF),
					static_cast<uint8_t>((rgba >>  8) & 0xFF),
					static_cast<uint8_t>( rgba        & 0xFF));
			}
			else
			{
				c = Color(64, 128, 255, 200); // fallback blue
			}
		}
		else
		{
			c = Color(64, 128, 255, 200);
		}

		// Secondary particles get different visual treatment
		if (WulfNet::HasFlag(p.flags, WulfNet::ParticleFlags::Spray))
			c = Color(255, 255, 255, 160);  // white spray
		else if (WulfNet::HasFlag(p.flags, WulfNet::ParticleFlags::Foam))
			c = Color(220, 240, 255, 180);  // light foam
		else if (WulfNet::HasFlag(p.flags, WulfNet::ParticleFlags::Bubble))
			c = Color(180, 200, 255, 120);  // transparent bubble

		RVec3 pos(p.x, p.y, p.z);
		mDebugRenderer->DrawMarker(pos, c, mParticleSize);

		if (mDrawVelocities)
		{
			float speed2 = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
			if (speed2 > 0.01f)
			{
				Vec3 vel(p.vx, p.vy, p.vz);
				mDebugRenderer->DrawArrow(pos, pos + 0.08f * vel, Color::sYellow, 0.005f);
			}
		}
	}
#endif
}

void WulfNetWaterV3Base::DrawSurfaceMesh()
{
#ifdef JPH_DEBUG_RENDERER
	const auto &verts = mSurface.GetVertices();
	const auto &tris  = mSurface.GetTriangles();
	if (tris.empty()) return;

	Color waterColor(32, 100, 200, 170);

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

void WulfNetWaterV3Base::DrawGridSlice(float ySlice)
{
#ifdef JPH_DEBUG_RENDERER
	const WulfNet::FluidGrid *grid = mFluid.GetGrid();
	if (!grid) return;

	float cs = grid->GetCellSize();
	int jSlice = static_cast<int>(ySlice / cs);
	if (jSlice < 0 || jSlice >= static_cast<int>(grid->GetResolutionY()))
		return;

	for (uint32_t k = 0; k < grid->GetResolutionZ(); ++k)
	{
		for (uint32_t i = 0; i < grid->GetResolutionX(); ++i)
		{
			const auto &cell = grid->GetCell(i, static_cast<uint32_t>(jSlice), k);
			if (cell.particleCount == 0) continue;

			float wx, wy, wz;
			grid->GridToWorld(static_cast<float>(i) + 0.5f,
			                  static_cast<float>(jSlice) + 0.5f,
			                  static_cast<float>(k) + 0.5f,
			                  wx, wy, wz);

			// Color by density
			float d = Clamp01(cell.density / 2000.0f);
			uint8_t b = static_cast<uint8_t>(80 + 175 * d);
			Color c(30, 50, b, 60);

			float hs = cs * 0.5f;
			RVec3 lo(wx - hs, wy, wz - hs);
			RVec3 hi(wx + hs, wy, wz + hs);
			mDebugRenderer->DrawTriangle(
				RVec3(lo.GetX(), wy, lo.GetZ()),
				RVec3(hi.GetX(), wy, lo.GetZ()),
				RVec3(lo.GetX(), wy, hi.GetZ()), c);
			mDebugRenderer->DrawTriangle(
				RVec3(hi.GetX(), wy, lo.GetZ()),
				RVec3(hi.GetX(), wy, hi.GetZ()),
				RVec3(lo.GetX(), wy, hi.GetZ()), c);
		}
	}
#endif
}

String WulfNetWaterV3Base::GetStatusString() const
{
	const WulfNet::FluidStats &fs   = mFluid.GetStats();
	const WulfNet::FluidSurfaceStats &ss = mSurface.GetStats();
	const WulfNet::SystemStats &sys = WulfNet::SystemMonitor::Get().GetStats();

	std::ostringstream oss;
	oss << std::fixed;

	oss << "FPS: " << std::setprecision(1) << mCurrentFPS
	    << " (" << std::setprecision(2) << mFrameTimeMs << " ms)\n";
	oss << std::setprecision(1);
	oss << "CPU: " << sys.cpuUsagePercent << "%  RAM: "
	    << WulfNet::FormatBytes(sys.processMemoryBytes) << "\n";

	if (sys.gpuUsageAvailable)
		oss << "GPU: " << sys.gpuUsagePercent << "%\n";

	oss << "\n";

	// Particle stats
	oss << "Particles: " << fs.activeParticles;
	if (fs.sleepingParticles > 0)
		oss << " (sleeping: " << fs.sleepingParticles << ")";
	oss << "\n";

	if (fs.surfaceParticles > 0 || fs.sprayParticles > 0)
		oss << "Surface: " << fs.surfaceParticles
		    << "  Spray: " << fs.sprayParticles << "\n";

	oss << "Triangles: " << ss.triangleCount << "\n";

	// Velocity / energy
	oss << std::setprecision(2);
	oss << "Avg vel: " << fs.averageVelocity
	    << "  Max vel: " << fs.maxVelocity << "\n";
	oss << "KE: " << std::setprecision(1) << fs.totalKineticEnergy
	    << "  PE: " << fs.totalPotentialEnergy << "\n";

	// Timing
	oss << std::setprecision(2);
	oss << "Sim: " << fs.totalTimeMs << " ms\n";
	oss << "  P2G: " << fs.p2gTimeMs
	    << "  Solve: " << fs.gridSolveTimeMs
	    << "  G2P: " << fs.g2pTimeMs
	    << "  Col: " << fs.collisionTimeMs;

	return String(oss.str());
}

// =====================================================================
// 1. Ocean Swell
// =====================================================================

void WulfNetWaterV3OceanSwellTest::SetupScenario()
{
	// Large domain for open water — coarse cells for perf
	mFluidConfig.gridResolutionX = 48;
	mFluidConfig.gridResolutionY = 16;
	mFluidConfig.gridResolutionZ = 48;
	mFluidConfig.cellSize        = 0.2f;
	mFluidConfig.maxParticles    = 200000;
	mFluidConfig.flipRatio       = 0.97f;

	mFluidConfig.boundsMaxX = 48 * 0.2f;   // 9.6 m
	mFluidConfig.boundsMaxY = 16 * 0.2f;   // 3.2 m
	mFluidConfig.boundsMaxZ = 48 * 0.2f;   // 9.6 m

	mDrawParticles   = false;
	mDrawSurface     = true;
	mColorByMaterial = true;
}

void WulfNetWaterV3OceanSwellTest::UpdateScenario(float dt)
{
	if (!mScenarioSetup)
	{
		mScenarioSetup = true;

		// Fill a large water body
		AddWaterBox(0.5f, 0.1f, 0.5f, 7.5f, 1.0f, 7.5f, mWaterMaterialId);

		// Wind-driven emitters along the left edge, pointing right
		for (int i = 0; i < 4; ++i)
		{
			WulfNet::FluidEmitter em;
			em.type          = WulfNet::EmitterType::Box;
			em.posX          = 0.3f;
			em.posY          = 0.7f;
			em.posZ          = 1.5f + i * 1.4f;
			em.dirX          = 1.0f;
			em.dirY          = -0.1f;
			em.dirZ          = 0.0f;
			em.sizeX         = 0.1f;
			em.sizeY         = 0.3f;
			em.sizeZ         = 0.6f;
			em.emissionRate  = 75.0f;
			em.initialSpeed  = 1.2f;
			em.speedVariance = 0.15f;
			em.materialId    = mWaterMaterialId;
			em.enabled       = true;
			mEmitterIds.push_back(AddEmitter(em));
		}

		// Buoyant crates
		for (int r = 0; r < 3; ++r)
		{
			for (int c = 0; c < 3; ++c)
			{
				float x = 3.0f + r * 1.2f;
				float z = 3.0f + c * 1.2f;

				BodyCreationSettings crate(
					new BoxShape(Vec3(0.18f, 0.12f, 0.18f)),
					RVec3(x, 1.3f, z),
					Quat::sIdentity(),
					EMotionType::Dynamic,
					Layers::MOVING);
				crate.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
				crate.mMassPropertiesOverride.mMass = 0.6f;
				crate.mLinearDamping = 0.2f;
				JPH::BodyID id = mBodyInterface->CreateAndAddBody(crate, EActivation::Activate);

				// Register buoyancy — density 500 kg/m³ → floats in water (1000)
				AddBuoyancyObject(id.GetIndexAndSequenceNumber(), 500.0f,
				                  0.18f * 0.12f * 0.18f * 8.0f, 0.6f);
			}
		}

		// Basin border colliders
		WulfNet::FluidCollider wallL;
		wallL.type = WulfNet::ColliderType::Box;
		wallL.posX = 0.0f; wallL.posY = 1.0f; wallL.posZ = 4.0f;
		wallL.scaleX = 0.2f; wallL.scaleY = 2.5f; wallL.scaleZ = 8.0f;
		AddCollider(wallL);

		WulfNet::FluidCollider wallR = wallL;
		wallR.posX = 8.0f;
		AddCollider(wallR);

		WulfNet::FluidCollider wallB = wallL;
		wallB.posX = 4.0f; wallB.posZ = 0.0f;
		wallB.scaleX = 8.0f; wallB.scaleZ = 0.2f;
		AddCollider(wallB);

		WulfNet::FluidCollider wallF = wallB;
		wallF.posZ = 8.0f;
		AddCollider(wallF);
	}

	// Oscillate emitter speed to create swell pulses
	mWindPhase += dt * 1.5f;
	for (size_t i = 0; i < mEmitterIds.size(); ++i)
	{
		WulfNet::FluidEmitter *em = mFluid.GetEmitter(mEmitterIds[i]);
		if (em)
		{
			float phase = mWindPhase + static_cast<float>(i) * 0.7f;
			em->initialSpeed = 1.2f + 0.6f * std::sin(phase);
		}
	}
}

// =====================================================================
// 2. Viscous Cascade
// =====================================================================

void WulfNetWaterV3ViscousCascadeTest::SetupScenario()
{
	mFluidConfig.gridResolutionX = 48;
	mFluidConfig.gridResolutionY = 32;
	mFluidConfig.gridResolutionZ = 24;
	mFluidConfig.cellSize        = 0.1f;
	mFluidConfig.maxParticles    = 120000;

	mFluidConfig.boundsMaxX = 48 * 0.1f;  // 4.8 m
	mFluidConfig.boundsMaxY = 32 * 0.1f;  // 3.2 m
	mFluidConfig.boundsMaxZ = 24 * 0.1f;  // 2.4 m

	mDrawParticles   = true;
	mDrawSurface     = false;   // Particles show material colors better
	mColorByMaterial = true;
	mParticleSize    = 0.025f;
}

void WulfNetWaterV3ViscousCascadeTest::UpdateScenario(float dt)
{
	if (!mScenarioSetup)
	{
		mScenarioSetup = true;

		// Three elevated columns side-by-side
		float colWidth  = 0.6f;
		float colDepth  = 1.4f;
		float colHeight = 1.2f;
		float baseY     = 1.5f;  // elevated
		float gap       = 0.3f;

		// Column 1: Water (low viscosity — flows fast)
		float x0 = 0.6f;
		AddWaterBox(x0, baseY, 0.3f,
		            x0 + colWidth, baseY + colHeight, 0.3f + colDepth,
		            mWaterMaterialId);

		// Column 2: Oil (medium viscosity)
		float x1 = x0 + colWidth + gap;
		AddWaterBox(x1, baseY, 0.3f,
		            x1 + colWidth, baseY + colHeight, 0.3f + colDepth,
		            mOilMaterialId);

		// Column 3: Honey (high viscosity — flows very slowly)
		float x2 = x1 + colWidth + gap;
		AddWaterBox(x2, baseY, 0.3f,
		            x2 + colWidth, baseY + colHeight, 0.3f + colDepth,
		            mHoneyMaterialId);

		// Ramp below each column (angled box collider directing flow forward)
		for (int i = 0; i < 3; ++i)
		{
			float cx = x0 + static_cast<float>(i) * (colWidth + gap) + colWidth * 0.5f;

			WulfNet::FluidCollider ramp;
			ramp.type   = WulfNet::ColliderType::Box;
			ramp.posX   = cx;
			ramp.posY   = baseY - 0.1f;
			ramp.posZ   = 0.3f + colDepth * 0.5f;
			ramp.scaleX = colWidth * 0.5f + 0.05f;
			ramp.scaleY = 0.05f;
			ramp.scaleZ = colDepth * 0.5f;
			// Slight tilt (we approximate by nudging position; real rotation handled internally)
			ramp.rotX   = -0.05f;
			ramp.rotW   = 1.0f;
			AddCollider(ramp);
		}

		// Basin at the bottom to collect everything
		WulfNet::FluidCollider basin;
		basin.type   = WulfNet::ColliderType::Box;
		basin.posX   = 2.16f;
		basin.posY   = 0.15f;
		basin.posZ   = 0.96f;
		basin.scaleX = 2.5f;
		basin.scaleY = 0.15f;
		basin.scaleZ = 1.2f;
		AddCollider(basin);
	}

	// Nothing dynamic to do — physics runs automatically
}

// =====================================================================
// 3. Thermal Convection
// =====================================================================

void WulfNetWaterV3ThermalConvectionTest::SetupScenario()
{
	mFluidConfig.gridResolutionX = 32;
	mFluidConfig.gridResolutionY = 32;
	mFluidConfig.gridResolutionZ = 32;
	mFluidConfig.cellSize        = 0.14f;
	mFluidConfig.maxParticles    = 150000;
	mFluidConfig.flipRatio       = 0.92f;

	mFluidConfig.boundsMaxX = 32 * 0.14f; // 4.48 m
	mFluidConfig.boundsMaxY = 32 * 0.14f;
	mFluidConfig.boundsMaxZ = 32 * 0.14f;

	mDrawParticles   = true;
	mDrawSurface     = true;
	mColorByMaterial = true;
	mParticleSize    = 0.03f;
}

void WulfNetWaterV3ThermalConvectionTest::UpdateScenario(float dt)
{
	if (!mScenarioSetup)
	{
		mScenarioSetup = true;

		// Cold water pool (bottom half)
		AddWaterBox(0.3f, 0.1f, 0.3f, 3.5f, 1.5f, 3.5f, mWaterMaterialId);

		// A hot lava pocket at the bottom center
		AddWaterSphere(1.9f, 0.4f, 1.9f, 0.5f, mLavaMaterialId);

		// Container walls
		WulfNet::FluidCollider wall;
		wall.type = WulfNet::ColliderType::Box;

		// Left wall
		wall.posX = 0.1f; wall.posY = 1.5f; wall.posZ = 1.9f;
		wall.scaleX = 0.1f; wall.scaleY = 1.8f; wall.scaleZ = 2.0f;
		AddCollider(wall);

		// Right wall
		wall.posX = 3.7f;
		AddCollider(wall);

		// Back wall
		wall.posX = 1.9f; wall.posZ = 0.1f;
		wall.scaleX = 2.0f; wall.scaleZ = 0.1f;
		AddCollider(wall);

		// Front wall
		wall.posZ = 3.7f;
		AddCollider(wall);

		// Floating light sphere (rides convection)
		{
			BodyCreationSettings ball(
				new SphereShape(0.12f),
				RVec3(1.9f, 1.2f, 1.9f),
				Quat::sIdentity(),
				EMotionType::Dynamic,
				Layers::MOVING);
			ball.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
			ball.mMassPropertiesOverride.mMass = 0.15f;
			JPH::BodyID id = mBodyInterface->CreateAndAddBody(ball, EActivation::Activate);
			AddBuoyancyObject(id.GetIndexAndSequenceNumber(), 300.0f, 0.007f, 0.8f);
		}
	}

	// Periodically inject more lava from below
	mLavaInjectTimer += dt;
	if (mLavaInjectTimer >= 3.0f)
	{
		mLavaInjectTimer = 0.0f;
		AddWaterSphere(1.9f, 0.2f, 1.9f, 0.25f, mLavaMaterialId);
	}
}

// =====================================================================
// 4. Spray & Foam
// =====================================================================

void WulfNetWaterV3SprayFoamTest::SetupScenario()
{
	mFluidConfig.gridResolutionX = 36;
	mFluidConfig.gridResolutionY = 28;
	mFluidConfig.gridResolutionZ = 36;
	mFluidConfig.cellSize        = 0.12f;
	mFluidConfig.maxParticles    = 150000;
	mFluidConfig.flipRatio       = 0.98f;

	mFluidConfig.boundsMaxX = 36 * 0.12f; // 4.32 m
	mFluidConfig.boundsMaxY = 28 * 0.12f; // 3.36 m
	mFluidConfig.boundsMaxZ = 36 * 0.12f; // 4.32 m

	mDrawParticles   = true;  // Shows spray/foam/bubble classification
	mDrawSurface     = true;
	mColorByMaterial = false; // Color by particle type instead
	mParticleSize    = 0.028f;
}

void WulfNetWaterV3SprayFoamTest::UpdateScenario(float dt)
{
	if (!mScenarioSetup)
	{
		mScenarioSetup = true;

		// Shallow basin
		AddWaterBox(0.5f, 0.1f, 0.5f, 3.4f, 0.5f, 3.4f, mWaterMaterialId);

		// High-pressure jet emitter pointing down at the basin
		WulfNet::FluidEmitter jet;
		jet.type          = WulfNet::EmitterType::Sphere;
		jet.posX          = 1.95f;
		jet.posY          = 2.4f;
		jet.posZ          = 1.95f;
		jet.radius        = 0.08f;
		jet.dirX          = 0.0f;
		jet.dirY          = -1.0f;
		jet.dirZ          = 0.0f;
		jet.emissionRate  = 200.0f;
		jet.initialSpeed  = 5.0f;
		jet.speedVariance = 0.3f;
		jet.materialId    = mWaterMaterialId;
		jet.enabled       = true;
		AddEmitter(jet);

		// Basin walls
		WulfNet::FluidCollider wall;
		wall.type = WulfNet::ColliderType::Box;

		wall.posX = 0.3f; wall.posY = 0.5f; wall.posZ = 1.95f;
		wall.scaleX = 0.1f; wall.scaleY = 0.6f; wall.scaleZ = 1.8f;
		AddCollider(wall);

		wall.posX = 3.6f;
		AddCollider(wall);

		wall.posX = 1.95f; wall.posZ = 0.3f;
		wall.scaleX = 1.8f; wall.scaleZ = 0.1f;
		AddCollider(wall);

		wall.posZ = 3.6f;
		AddCollider(wall);

		// Obstacle sphere in the basin for splash deflection
		WulfNet::FluidCollider obstacle;
		obstacle.type   = WulfNet::ColliderType::Sphere;
		obstacle.posX   = 1.95f;
		obstacle.posY   = 0.4f;
		obstacle.posZ   = 1.95f;
		obstacle.radius = 0.25f;
		AddCollider(obstacle);

		// Visual body for the obstacle
		BodyCreationSettings obsBall(
			new SphereShape(0.25f),
			RVec3(1.95f, 0.4f, 1.95f),
			Quat::sIdentity(),
			EMotionType::Static,
			Layers::NON_MOVING);
		mBodyInterface->CreateAndAddBody(obsBall, EActivation::DontActivate);
	}

	// Slowly sweep the jet angle for variety
	mJetAngle += dt * 0.4f;
	WulfNet::FluidEmitter *jet = mFluid.GetEmitter(0);
	if (jet)
	{
		float sweep = 0.3f * std::sin(mJetAngle);
		jet->dirX = sweep;
		jet->dirY = -1.0f;
		jet->dirZ = 0.15f * std::cos(mJetAngle * 0.7f);

		// Normalize
		float len = std::sqrt(jet->dirX * jet->dirX + jet->dirY * jet->dirY + jet->dirZ * jet->dirZ);
		if (len > 0.001f)
		{
			jet->dirX /= len;
			jet->dirY /= len;
			jet->dirZ /= len;
		}
	}
}

// =====================================================================
// 5. Obstacle Course
// =====================================================================

void WulfNetWaterV3ObstacleCourseTest::SetupScenario()
{
	mFluidConfig.gridResolutionX = 48;
	mFluidConfig.gridResolutionY = 24;
	mFluidConfig.gridResolutionZ = 24;
	mFluidConfig.cellSize        = 0.14f;
	mFluidConfig.maxParticles    = 150000;

	mFluidConfig.boundsMaxX = 48 * 0.14f; // 6.72 m
	mFluidConfig.boundsMaxY = 24 * 0.14f; // 3.36 m
	mFluidConfig.boundsMaxZ = 24 * 0.14f; // 3.36 m

	mDrawParticles   = true;
	mDrawSurface     = true;
	mColorByMaterial = true;
	mParticleSize    = 0.025f;
}

void WulfNetWaterV3ObstacleCourseTest::UpdateScenario(float dt)
{
	if (!mScenarioSetup)
	{
		mScenarioSetup = true;

		// Entry reservoir (elevated, left side)
		AddWaterBox(0.3f, 1.5f, 0.5f, 1.2f, 2.2f, 2.3f, mWaterMaterialId);

		// Continuous emitter feeding the reservoir
		WulfNet::FluidEmitter feed;
		feed.type         = WulfNet::EmitterType::Box;
		feed.posX         = 0.2f;
		feed.posY         = 2.0f;
		feed.posZ         = 1.4f;
		feed.dirX         = 1.0f;
		feed.dirY         = -0.2f;
		feed.dirZ         = 0.0f;
		feed.sizeX        = 0.1f;
		feed.sizeY        = 0.2f;
		feed.sizeZ        = 0.6f;
		feed.emissionRate = 100.0f;
		feed.initialSpeed = 1.5f;
		feed.materialId   = mWaterMaterialId;
		feed.enabled      = true;
		AddEmitter(feed);

		// ----- Obstacle gauntlet (left → right) -----

		// Stage 1: Sphere obstacles
		for (int i = 0; i < 3; ++i)
		{
			WulfNet::FluidCollider sphere;
			sphere.type   = WulfNet::ColliderType::Sphere;
			sphere.posX   = 1.8f + i * 0.4f;
			sphere.posY   = 0.4f + i * 0.15f;
			sphere.posZ   = 1.0f + (i % 2) * 0.8f;
			sphere.radius = 0.15f;
			AddCollider(sphere);

			// Visual body
			BodyCreationSettings vis(
				new SphereShape(0.15f),
				RVec3(sphere.posX, sphere.posY, sphere.posZ),
				Quat::sIdentity(),
				EMotionType::Static,
				Layers::NON_MOVING);
			mBodyInterface->CreateAndAddBody(vis, EActivation::DontActivate);
		}

		// Stage 2: Box chicane
		for (int i = 0; i < 2; ++i)
		{
			WulfNet::FluidCollider box;
			box.type   = WulfNet::ColliderType::Box;
			box.posX   = 3.0f;
			box.posY   = 0.5f;
			box.posZ   = 0.7f + i * 1.4f;
			box.scaleX = 0.15f;
			box.scaleY = 0.6f;
			box.scaleZ = 0.3f;
			AddCollider(box);

			BodyCreationSettings vis(
				new BoxShape(Vec3(0.15f, 0.6f, 0.3f)),
				RVec3(box.posX, box.posY, box.posZ),
				Quat::sIdentity(),
				EMotionType::Static,
				Layers::NON_MOVING);
			mBodyInterface->CreateAndAddBody(vis, EActivation::DontActivate);
		}

		// Stage 3: Capsule pillars
		for (int i = 0; i < 2; ++i)
		{
			WulfNet::FluidCollider cap;
			cap.type   = WulfNet::ColliderType::Capsule;
			cap.posX   = 3.8f + i * 0.5f;
			cap.posY   = 0.5f;
			cap.posZ   = 1.1f + i * 0.6f;
			cap.radius = 0.1f;
			cap.height = 0.5f;
			AddCollider(cap);

			BodyCreationSettings vis(
				new CapsuleShape(0.25f, 0.1f),
				RVec3(cap.posX, cap.posY, cap.posZ),
				Quat::sIdentity(),
				EMotionType::Static,
				Layers::NON_MOVING);
			mBodyInterface->CreateAndAddBody(vis, EActivation::DontActivate);
		}

		// Stage 4: Narrow channel with opening
		{
			WulfNet::FluidCollider leftWall;
			leftWall.type   = WulfNet::ColliderType::Box;
			leftWall.posX   = 4.6f;
			leftWall.posY   = 0.5f;
			leftWall.posZ   = 0.5f;
			leftWall.scaleX = 0.08f;
			leftWall.scaleY = 0.8f;
			leftWall.scaleZ = 0.5f;
			AddCollider(leftWall);

			WulfNet::FluidCollider rightWall = leftWall;
			rightWall.posZ = 2.3f;
			AddCollider(rightWall);

			// Visual walls
			BodyCreationSettings visL(
				new BoxShape(Vec3(0.08f, 0.8f, 0.5f)),
				RVec3(leftWall.posX, leftWall.posY, leftWall.posZ),
				Quat::sIdentity(),
				EMotionType::Static,
				Layers::NON_MOVING);
			mBodyInterface->CreateAndAddBody(visL, EActivation::DontActivate);

			BodyCreationSettings visR(
				new BoxShape(Vec3(0.08f, 0.8f, 0.5f)),
				RVec3(rightWall.posX, rightWall.posY, rightWall.posZ),
				Quat::sIdentity(),
				EMotionType::Static,
				Layers::NON_MOVING);
			mBodyInterface->CreateAndAddBody(visR, EActivation::DontActivate);
		}

		// Collection basin (right side, lower)
		WulfNet::FluidCollider basinFloor;
		basinFloor.type   = WulfNet::ColliderType::Box;
		basinFloor.posX   = 5.0f;
		basinFloor.posY   = 0.05f;
		basinFloor.posZ   = 1.4f;
		basinFloor.scaleX = 0.8f;
		basinFloor.scaleY = 0.05f;
		basinFloor.scaleZ = 1.4f;
		AddCollider(basinFloor);

		// Side channel walls
		WulfNet::FluidCollider side;
		side.type   = WulfNet::ColliderType::Box;
		side.posX   = 2.5f;
		side.posY   = 0.4f;
		side.posZ   = 0.2f;
		side.scaleX = 3.0f;
		side.scaleY = 0.5f;
		side.scaleZ = 0.1f;
		AddCollider(side);

		side.posZ = 2.6f;
		AddCollider(side);
	}

	mTime += dt;
}
