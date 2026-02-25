// SPDX-License-Identifier: MIT
// WulfNet Water Physics V4 — Implementation
//
// Elastic ball water: each particle is a visible sphere, nearby spheres
// are seamlessly connected by midpoint bridge spheres.  Physics tuned for
// bouncy / elastic behaviour using high surface tension and stiffness.

#include <Samples.h>

#include "WulfNetWaterV4Tests.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>

#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// =====================================================================
// RTTI Registration
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV4Base)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV4Base, Test)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV4BallPoolTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV4BallPoolTest, WulfNetWaterV4Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV4ElasticCascadeTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV4ElasticCascadeTest, WulfNetWaterV4Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV4BallSplashTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV4BallSplashTest, WulfNetWaterV4Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV4BallWaterfallTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV4BallWaterfallTest, WulfNetWaterV4Base)
}

// =====================================================================
// WulfNetWaterV4Base — Shared initialisation and update
// =====================================================================

WulfNetWaterV4Base::~WulfNetWaterV4Base()
{
	mFluid.Shutdown();
}

void WulfNetWaterV4Base::Initialize()
{
	WulfNet::SystemMonitor::Get().Initialize();
	mLastFPSTime = std::chrono::high_resolution_clock::now();

	CreateFloor();

	// ----- FluidSystem defaults: tuned for "elastic ball" water -----
	mFluidConfig.gridResolutionX   = 40;
	mFluidConfig.gridResolutionY   = 24;
	mFluidConfig.gridResolutionZ   = 40;
	mFluidConfig.cellSize          = 0.14f;
	mFluidConfig.maxParticles      = 80000;
	mFluidConfig.particlesPerCell  = 4;
	mFluidConfig.gravity           = -9.81f;
	mFluidConfig.flipRatio         = 0.98f;   // High FLIP → preserves velocity → bouncy
	mFluidConfig.pressureIterations = 50;
	mFluidConfig.substeps          = 1;
	mFluidConfig.maxTimestep       = 1.0f / 60.0f;
	mFluidConfig.useGPU            = true;
	mFluidConfig.enableSleeping    = true;
	mFluidConfig.sleepThreshold    = 0.002f;
	mFluidConfig.enableSpatialHash = true;

	// Bounds
	mFluidConfig.boundsMinX = 0.0f;
	mFluidConfig.boundsMinY = 0.0f;
	mFluidConfig.boundsMinZ = 0.0f;
	mFluidConfig.boundsMaxX = mFluidConfig.gridResolutionX * mFluidConfig.cellSize;
	mFluidConfig.boundsMaxY = mFluidConfig.gridResolutionY * mFluidConfig.cellSize;
	mFluidConfig.boundsMaxZ = mFluidConfig.gridResolutionZ * mFluidConfig.cellSize;

	// Let derived class override config before init
	SetupScenario();

	// Recompute bounds after override
	mFluidConfig.boundsMaxX = mFluidConfig.gridResolutionX * mFluidConfig.cellSize;
	mFluidConfig.boundsMaxY = mFluidConfig.gridResolutionY * mFluidConfig.cellSize;
	mFluidConfig.boundsMaxZ = mFluidConfig.gridResolutionZ * mFluidConfig.cellSize;

	// Initialise the fluid system
	mFluid.Initialize(mFluidConfig);

	// Register the "elastic water" material:
	//   High surface tension → keeps blobs spherical
	//   High stiffness       → strong pressure response → bounce
	//   Slightly higher viscosity → cohesion without over-damping
	WulfNet::FluidMaterial elasticWater = WulfNet::FluidMaterial::Water();
	elasticWater.surfaceTension = 0.25f;     // 3.5× normal water
	elasticWater.stiffness      = 150000.0f; // 3× normal — bouncy
	elasticWater.viscosity      = 0.005f;    // 5× normal — slight cohesion
	mElasticWaterId = mFluid.AddMaterial(elasticWater);

	// Default visual palette (16 slots, safe for any material ID)
	mMaterialVisuals.assign(16, { Color(60, 150, 230, 255), Color(45, 125, 210, 255) });

	// Pre-allocate spatial hash buckets
	mSpatialHash.reserve(4096);
}

void WulfNetWaterV4Base::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	// FPS tracking
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

	// Derived per-frame logic (emitter toggling, etc.)
	UpdateScenario(inParams.mDeltaTime);

	// Step the fluid sim
	mFluid.Step(inParams.mDeltaTime);

	// Render: balls + bridge fill (no marching-cubes surface at all)
	DrawBalls();
	DrawBridges();
}

// =====================================================================
// Rendering — Draw each active particle as a coloured sphere
// =====================================================================

void WulfNetWaterV4Base::DrawBalls()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	const WulfNet::FluidParticle *particles = mFluid.GetParticles();
	uint32_t count = mFluid.GetParticleCount();

	for (uint32_t i = 0; i < count; ++i)
	{
		const auto &p = particles[i];
		if (!WulfNet::HasFlag(p.flags, WulfNet::ParticleFlags::Active))
			continue;

		// Material colour
		uint32_t matIdx = p.materialId;
		if (matIdx >= static_cast<uint32_t>(mMaterialVisuals.size()))
			matIdx = 0;
		Color c = mMaterialVisuals[matIdx].ballColor;

		// Surface particles get a brighter tint (light-catching effect)
		if (WulfNet::HasFlag(p.flags, WulfNet::ParticleFlags::Surface))
		{
			c = Color(
				static_cast<uint8_t>(std::min(255, c.r + 40)),
				static_cast<uint8_t>(std::min(255, c.g + 40)),
				static_cast<uint8_t>(std::min(255, c.b + 20)),
				c.a);
		}

		mDebugRenderer->DrawSphere(RVec3(p.x, p.y, p.z), mBallRadius, c);
	}
#endif
}

// =====================================================================
// Spatial hash — used by DrawBridges for fast neighbour lookup
// =====================================================================

uint64_t WulfNetWaterV4Base::HashCell(int cx, int cy, int cz)
{
	// Standard spatial-hash primes (handles negative coordinates naturally)
	return static_cast<uint64_t>(
		static_cast<uint32_t>(cx * 73856093) ^
		static_cast<uint32_t>(cy * 19349663) ^
		static_cast<uint32_t>(cz * 83492791));
}

void WulfNetWaterV4Base::BuildSpatialHash(
	const WulfNet::FluidParticle *particles, uint32_t count)
{
	mSpatialHash.clear();

	const float invCell = 1.0f / mBridgeThreshold;

	for (uint32_t i = 0; i < count; ++i)
	{
		const auto &p = particles[i];
		if (!WulfNet::HasFlag(p.flags, WulfNet::ParticleFlags::Active))
			continue;

		int cx = static_cast<int>(std::floor(p.x * invCell));
		int cy = static_cast<int>(std::floor(p.y * invCell));
		int cz = static_cast<int>(std::floor(p.z * invCell));

		mSpatialHash[HashCell(cx, cy, cz)].push_back(i);
	}
}

// =====================================================================
// Rendering — Draw bridge spheres at the midpoint of nearby pairs
// =====================================================================

void WulfNetWaterV4Base::DrawBridges()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	const WulfNet::FluidParticle *particles = mFluid.GetParticles();
	uint32_t count = mFluid.GetParticleCount();

	BuildSpatialHash(particles, count);

	const float thresh2 = mBridgeThreshold * mBridgeThreshold;
	const float invCell = 1.0f / mBridgeThreshold;
	uint32_t bridgesDraw = 0;

	// For every active particle, check its 27-cell neighbourhood.
	// We only process pairs where b > a to avoid double-counting.
	for (uint32_t a = 0; a < count && bridgesDraw < mMaxBridges; ++a)
	{
		const auto &pa = particles[a];
		if (!WulfNet::HasFlag(pa.flags, WulfNet::ParticleFlags::Active))
			continue;

		int cx = static_cast<int>(std::floor(pa.x * invCell));
		int cy = static_cast<int>(std::floor(pa.y * invCell));
		int cz = static_cast<int>(std::floor(pa.z * invCell));

		for (int dx = -1; dx <= 1 && bridgesDraw < mMaxBridges; ++dx)
		{
			for (int dy = -1; dy <= 1 && bridgesDraw < mMaxBridges; ++dy)
			{
				for (int dz = -1; dz <= 1 && bridgesDraw < mMaxBridges; ++dz)
				{
					uint64_t nkey = HashCell(cx + dx, cy + dy, cz + dz);
					auto nit = mSpatialHash.find(nkey);
					if (nit == mSpatialHash.end()) continue;

					for (uint32_t b : nit->second)
					{
						if (b <= a) continue;  // unique pairs only (b > a)

						const auto &pb = particles[b];
						// Active check not needed — only active particles are in the hash

						float ddx = pa.x - pb.x;
						float ddy = pa.y - pb.y;
						float ddz = pa.z - pb.z;
						float d2  = ddx * ddx + ddy * ddy + ddz * ddz;

						if (d2 >= thresh2 || d2 < 1e-6f)
							continue;

						float d = std::sqrt(d2);
						float t = d / mBridgeThreshold;   // 0 → touching, 1 → threshold

						// Bridge sphere radius: large when balls are close, tapers to zero
						float br = mBallRadius * mBridgeSizeFactor * (1.0f - t * t);
						if (br < 0.004f)
							continue;

						// Midpoint position
						float mx = (pa.x + pb.x) * 0.5f;
						float my = (pa.y + pb.y) * 0.5f;
						float mz = (pa.z + pb.z) * 0.5f;

						// Average the two particles' bridge colours
						uint32_t mA = std::min(pa.materialId,
							static_cast<uint32_t>(mMaterialVisuals.size() - 1));
						uint32_t mB = std::min(pb.materialId,
							static_cast<uint32_t>(mMaterialVisuals.size() - 1));
						Color ca = mMaterialVisuals[mA].bridgeColor;
						Color cb = mMaterialVisuals[mB].bridgeColor;
						Color bc(
							static_cast<uint8_t>((ca.r + cb.r) / 2),
							static_cast<uint8_t>((ca.g + cb.g) / 2),
							static_cast<uint8_t>((ca.b + cb.b) / 2),
							255);

						mDebugRenderer->DrawSphere(RVec3(mx, my, mz), br, bc);
						++bridgesDraw;
						if (bridgesDraw >= mMaxBridges) break;
					}
				}
			}
		}
	}

	mBridgeCount = bridgesDraw;
#endif
}

// =====================================================================
// Helpers
// =====================================================================

void WulfNetWaterV4Base::AddWaterBox(float minX, float minY, float minZ,
                                      float maxX, float maxY, float maxZ,
                                      uint32_t materialId)
{
	mFluid.AddParticleBox(minX, minY, minZ, maxX, maxY, maxZ, materialId);
}

void WulfNetWaterV4Base::AddWaterSphere(float cx, float cy, float cz,
                                         float radius, uint32_t materialId)
{
	mFluid.AddParticleSphere(cx, cy, cz, radius, materialId);
}

uint32_t WulfNetWaterV4Base::AddEmitter(const WulfNet::FluidEmitter &emitter)
{
	return mFluid.AddEmitter(emitter);
}

uint32_t WulfNetWaterV4Base::AddCollider(const WulfNet::FluidCollider &collider)
{
	return mFluid.AddCollider(collider);
}

// =====================================================================
// Status overlay
// =====================================================================

String WulfNetWaterV4Base::GetStatusString() const
{
	const WulfNet::FluidStats &fs  = mFluid.GetStats();
	const WulfNet::SystemStats &sys = WulfNet::SystemMonitor::Get().GetStats();

	std::ostringstream oss;
	oss << std::fixed;

	oss << "FPS: " << std::setprecision(1) << mCurrentFPS
	    << "  (" << std::setprecision(2) << mFrameTimeMs << " ms)\n";

	oss << std::setprecision(1);
	oss << "CPU: " << sys.cpuUsagePercent << "%  RAM: "
	    << WulfNet::FormatBytes(sys.processMemoryBytes) << "\n";
	if (sys.gpuUsageAvailable)
		oss << "GPU: " << sys.gpuUsagePercent << "%\n";

	oss << "\n";
	oss << "Balls: " << fs.activeParticles << "\n";
	oss << "Bridges drawn: " << mBridgeCount;
	if (mBridgeCount >= mMaxBridges) oss << " (capped)";
	oss << "\n";
	oss << "Ball radius: " << std::setprecision(3) << mBallRadius
	    << " m   Threshold: " << mBridgeThreshold << " m\n";

	oss << std::setprecision(2);
	oss << "Avg vel: " << fs.averageVelocity
	    << "  Max vel: " << fs.maxVelocity << "\n";
	oss << "KE: " << std::setprecision(1) << fs.totalKineticEnergy << "\n";

	oss << std::setprecision(2);
	oss << "Sim: " << fs.totalTimeMs << " ms\n";
	oss << "  P2G: " << fs.p2gTimeMs
	    << "  Solve: " << fs.gridSolveTimeMs
	    << "  G2P: " << fs.g2pTimeMs;

	return String(oss.str());
}

// =====================================================================
// 1. Ball Pool — Emitter fills a basin with bouncy balls
// =====================================================================

void WulfNetWaterV4BallPoolTest::SetupScenario()
{
	mFluidConfig.gridResolutionX = 36;
	mFluidConfig.gridResolutionY = 24;
	mFluidConfig.gridResolutionZ = 36;
	mFluidConfig.cellSize        = 0.16f;
	mFluidConfig.maxParticles    = 60000;

	mBallRadius       = 0.05f;
	mBridgeThreshold  = 0.115f;
}

void WulfNetWaterV4BallPoolTest::UpdateScenario(float dt)
{
	if (!mScenarioSetup)
	{
		mScenarioSetup = true;

		// Pre-fill a shallow layer of ball-water
		AddWaterBox(0.8f, 0.15f, 0.8f, 4.9f, 0.6f, 4.9f, mElasticWaterId);

		// Continuous emitter pouring from above
		WulfNet::FluidEmitter pour;
		pour.type          = WulfNet::EmitterType::Box;
		pour.posX          = 2.85f;
		pour.posY          = 3.2f;
		pour.posZ          = 2.85f;
		pour.dirX          = 0.0f;
		pour.dirY          = -1.0f;
		pour.dirZ          = 0.0f;
		pour.sizeX         = 0.4f;
		pour.sizeY         = 0.05f;
		pour.sizeZ         = 0.4f;
		pour.emissionRate  = 80.0f;
		pour.initialSpeed  = 0.8f;
		pour.speedVariance = 0.1f;
		pour.materialId    = mElasticWaterId;
		pour.enabled       = true;
		AddEmitter(pour);

		// Basin walls (fluid colliders)
		WulfNet::FluidCollider wall;
		wall.type = WulfNet::ColliderType::Box;

		wall.posX = 0.5f; wall.posY = 1.5f; wall.posZ = 2.85f;
		wall.scaleX = 0.1f; wall.scaleY = 1.8f; wall.scaleZ = 2.5f;
		AddCollider(wall);   // Left

		wall.posX = 5.2f;
		AddCollider(wall);   // Right

		wall.posX = 2.85f; wall.posZ = 0.5f;
		wall.scaleX = 2.5f; wall.scaleZ = 0.1f;
		AddCollider(wall);   // Back

		wall.posZ = 5.2f;
		AddCollider(wall);   // Front

		// Visual Jolt walls so the user can see the basin
		auto addVisualWall = [&](Vec3 halfExt, RVec3 pos)
		{
			BodyCreationSettings w(new BoxShape(halfExt), pos,
				Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
			mBodyInterface->CreateAndAddBody(w, EActivation::DontActivate);
		};

		addVisualWall(Vec3(0.1f, 1.5f, 2.5f), RVec3(0.5f, 1.5f, 2.85f));
		addVisualWall(Vec3(0.1f, 1.5f, 2.5f), RVec3(5.2f, 1.5f, 2.85f));
		addVisualWall(Vec3(2.5f, 1.5f, 0.1f), RVec3(2.85f, 1.5f, 0.5f));
		addVisualWall(Vec3(2.5f, 1.5f, 0.1f), RVec3(2.85f, 1.5f, 5.2f));
	}
}

// =====================================================================
// 2. Elastic Cascade — Three columns at different elasticity levels
// =====================================================================

void WulfNetWaterV4ElasticCascadeTest::SetupScenario()
{
	mFluidConfig.gridResolutionX = 48;
	mFluidConfig.gridResolutionY = 32;
	mFluidConfig.gridResolutionZ = 24;
	mFluidConfig.cellSize        = 0.12f;
	mFluidConfig.maxParticles    = 80000;

	mBallRadius       = 0.04f;
	mBridgeThreshold  = 0.095f;
}

void WulfNetWaterV4ElasticCascadeTest::UpdateScenario(float dt)
{
	if (!mScenarioSetup)
	{
		mScenarioSetup = true;

		// ------ Register three materials with varying elasticity ------

		// Soft: low surface tension, low stiffness → splashy, loose balls
		WulfNet::FluidMaterial soft = WulfNet::FluidMaterial::Water();
		soft.surfaceTension = 0.05f;
		soft.stiffness      = 50000.0f;
		soft.viscosity      = 0.002f;
		mSoftMatId = mFluid.AddMaterial(soft);

		// Medium: balanced
		WulfNet::FluidMaterial medium = WulfNet::FluidMaterial::Water();
		medium.surfaceTension = 0.15f;
		medium.stiffness      = 100000.0f;
		medium.viscosity      = 0.005f;
		mMediumMatId = mFluid.AddMaterial(medium);

		// Firm: high surface tension, high stiffness → cohesive, bouncy blobs
		WulfNet::FluidMaterial firm = WulfNet::FluidMaterial::Water();
		firm.surfaceTension = 0.35f;
		firm.stiffness      = 200000.0f;
		firm.viscosity      = 0.01f;
		mFirmMatId = mFluid.AddMaterial(firm);

		// Assign distinct colours per elasticity level
		if (mSoftMatId   < mMaterialVisuals.size())
			mMaterialVisuals[mSoftMatId]   = { Color(100, 200, 255, 255), Color(80, 170, 230, 255) };
		if (mMediumMatId < mMaterialVisuals.size())
			mMaterialVisuals[mMediumMatId] = { Color(60, 150, 230, 255),  Color(45, 125, 210, 255) };
		if (mFirmMatId   < mMaterialVisuals.size())
			mMaterialVisuals[mFirmMatId]   = { Color(30, 90, 200, 255),   Color(20, 70, 180, 255) };

		// ------ Three elevated columns side-by-side ------
		float colWidth  = 0.8f;
		float colDepth  = 2.0f;
		float colHeight = 1.0f;
		float baseY     = 2.0f;
		float gap       = 0.4f;

		// Column 1 — Soft (light blue)
		float x0 = 0.6f;
		AddWaterBox(x0, baseY, 0.3f,
		            x0 + colWidth, baseY + colHeight, 0.3f + colDepth,
		            mSoftMatId);

		// Column 2 — Medium (blue)
		float x1 = x0 + colWidth + gap;
		AddWaterBox(x1, baseY, 0.3f,
		            x1 + colWidth, baseY + colHeight, 0.3f + colDepth,
		            mMediumMatId);

		// Column 3 — Firm (deep blue)
		float x2 = x1 + colWidth + gap;
		AddWaterBox(x2, baseY, 0.3f,
		            x2 + colWidth, baseY + colHeight, 0.3f + colDepth,
		            mFirmMatId);

		// Tilted ramp underneath to guide the draining flow
		BodyCreationSettings ramp(
			new BoxShape(Vec3(2.5f, 0.06f, 1.2f)),
			RVec3(2.85f, 1.2f, 1.3f),
			Quat::sRotation(Vec3::sAxisZ(), -0.15f),
			EMotionType::Static,
			Layers::NON_MOVING);
		mBodyInterface->CreateAndAddBody(ramp, EActivation::DontActivate);

		// Ramp as fluid collider
		WulfNet::FluidCollider rampCol;
		rampCol.type   = WulfNet::ColliderType::Box;
		rampCol.posX   = 2.85f;
		rampCol.posY   = 1.2f;
		rampCol.posZ   = 1.3f;
		rampCol.scaleX = 2.5f;
		rampCol.scaleY = 0.06f;
		rampCol.scaleZ = 1.2f;
		AddCollider(rampCol);
	}
}

// =====================================================================
// 3. Ball Splash — Burst of fast particles impacts a calm pool
// =====================================================================

void WulfNetWaterV4BallSplashTest::SetupScenario()
{
	mFluidConfig.gridResolutionX = 40;
	mFluidConfig.gridResolutionY = 28;
	mFluidConfig.gridResolutionZ = 40;
	mFluidConfig.cellSize        = 0.14f;
	mFluidConfig.maxParticles    = 80000;

	mBallRadius       = 0.045f;
	mBridgeThreshold  = 0.105f;
}

void WulfNetWaterV4BallSplashTest::UpdateScenario(float dt)
{
	if (!mScenarioSetup)
	{
		mScenarioSetup = true;

		// Calm pool of ball-water
		AddWaterBox(0.6f, 0.15f, 0.6f, 5.0f, 1.2f, 5.0f, mElasticWaterId);

		// Basin walls
		WulfNet::FluidCollider wall;
		wall.type = WulfNet::ColliderType::Box;

		wall.posX = 0.3f; wall.posY = 1.2f; wall.posZ = 2.8f;
		wall.scaleX = 0.1f; wall.scaleY = 1.4f; wall.scaleZ = 2.6f;
		AddCollider(wall);                  // Left

		wall.posX = 5.3f;
		AddCollider(wall);                  // Right

		wall.posX = 2.8f; wall.posZ = 0.3f;
		wall.scaleX = 2.6f; wall.scaleZ = 0.1f;
		AddCollider(wall);                  // Back

		wall.posZ = 5.3f;
		AddCollider(wall);                  // Front

		// Visual walls
		auto addWall = [&](Vec3 he, RVec3 pos)
		{
			BodyCreationSettings w(new BoxShape(he), pos,
				Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
			mBodyInterface->CreateAndAddBody(w, EActivation::DontActivate);
		};
		addWall(Vec3(0.1f, 1.2f, 2.6f), RVec3(0.3f, 1.2f, 2.8f));
		addWall(Vec3(0.1f, 1.2f, 2.6f), RVec3(5.3f, 1.2f, 2.8f));
		addWall(Vec3(2.6f, 1.2f, 0.1f), RVec3(2.8f, 1.2f, 0.3f));
		addWall(Vec3(2.6f, 1.2f, 0.1f), RVec3(2.8f, 1.2f, 5.3f));

		// High-speed burst emitter (initially disabled)
		WulfNet::FluidEmitter burst;
		burst.type          = WulfNet::EmitterType::Sphere;
		burst.posX          = 2.8f;
		burst.posY          = 3.5f;
		burst.posZ          = 2.8f;
		burst.radius        = 0.15f;
		burst.dirX          = 0.0f;
		burst.dirY          = -1.0f;
		burst.dirZ          = 0.0f;
		burst.emissionRate  = 600.0f;   // Brief intense burst
		burst.initialSpeed  = 6.0f;
		burst.speedVariance = 0.5f;
		burst.materialId    = mElasticWaterId;
		burst.enabled       = false;    // Enabled after 2 s
		mBurstEmitterId = AddEmitter(burst);
	}

	// Trigger the splash at 2 s, disable at 2.3 s
	mSplashTimer += dt;
	if (!mSplashTriggered && mSplashTimer >= 2.0f)
	{
		mSplashTriggered = true;
		WulfNet::FluidEmitter *em = mFluid.GetEmitter(mBurstEmitterId);
		if (em) em->enabled = true;
	}
	if (mSplashTriggered && mSplashTimer >= 2.3f)
	{
		WulfNet::FluidEmitter *em = mFluid.GetEmitter(mBurstEmitterId);
		if (em) em->enabled = false;
	}
}

// =====================================================================
// 4. Ball Waterfall — Balls pour off a cliff edge into a pool below
// =====================================================================

void WulfNetWaterV4BallWaterfallTest::SetupScenario()
{
	mFluidConfig.gridResolutionX = 40;
	mFluidConfig.gridResolutionY = 32;
	mFluidConfig.gridResolutionZ = 32;
	mFluidConfig.cellSize        = 0.14f;
	mFluidConfig.maxParticles    = 80000;

	mBallRadius       = 0.045f;
	mBridgeThreshold  = 0.105f;
}

void WulfNetWaterV4BallWaterfallTest::UpdateScenario(float dt)
{
	if (!mScenarioSetup)
	{
		mScenarioSetup = true;

		// ---- Cliff ledge at ~2.8 m ----
		BodyCreationSettings cliff(
			new BoxShape(Vec3(1.2f, 0.15f, 1.8f)),
			RVec3(1.5f, 2.8f, 2.24f),
			Quat::sIdentity(),
			EMotionType::Static,
			Layers::NON_MOVING);
		mBodyInterface->CreateAndAddBody(cliff, EActivation::DontActivate);

		WulfNet::FluidCollider cliffCol;
		cliffCol.type   = WulfNet::ColliderType::Box;
		cliffCol.posX   = 1.5f;
		cliffCol.posY   = 2.8f;
		cliffCol.posZ   = 2.24f;
		cliffCol.scaleX = 1.2f;
		cliffCol.scaleY = 0.15f;
		cliffCol.scaleZ = 1.8f;
		AddCollider(cliffCol);

		// ---- Pool at bottom ----
		AddWaterBox(2.0f, 0.15f, 0.5f, 5.0f, 0.8f, 4.0f, mElasticWaterId);

		// Pool back wall
		WulfNet::FluidCollider poolBack;
		poolBack.type   = WulfNet::ColliderType::Box;
		poolBack.posX   = 5.2f;
		poolBack.posY   = 1.0f;
		poolBack.posZ   = 2.24f;
		poolBack.scaleX = 0.1f;
		poolBack.scaleY = 1.2f;
		poolBack.scaleZ = 2.0f;
		AddCollider(poolBack);

		BodyCreationSettings backWall(
			new BoxShape(Vec3(0.1f, 1.2f, 2.0f)),
			RVec3(5.2f, 1.0f, 2.24f),
			Quat::sIdentity(),
			EMotionType::Static,
			Layers::NON_MOVING);
		mBodyInterface->CreateAndAddBody(backWall, EActivation::DontActivate);

		// Pool side walls
		WulfNet::FluidCollider side;
		side.type   = WulfNet::ColliderType::Box;
		side.posX   = 3.5f;
		side.posY   = 1.0f;
		side.posZ   = 0.3f;
		side.scaleX = 2.0f;
		side.scaleY = 1.2f;
		side.scaleZ = 0.1f;
		AddCollider(side);

		side.posZ = 4.2f;
		AddCollider(side);

		// ---- Emitter on the cliff edge ----
		WulfNet::FluidEmitter pour;
		pour.type          = WulfNet::EmitterType::Box;
		pour.posX          = 0.5f;
		pour.posY          = 3.2f;
		pour.posZ          = 2.24f;
		pour.dirX          = 1.0f;
		pour.dirY          = 0.0f;
		pour.dirZ          = 0.0f;
		pour.sizeX         = 0.1f;
		pour.sizeY         = 0.15f;
		pour.sizeZ         = 1.0f;
		pour.emissionRate  = 100.0f;
		pour.initialSpeed  = 1.2f;
		pour.speedVariance = 0.1f;
		pour.materialId    = mElasticWaterId;
		pour.enabled       = true;
		AddEmitter(pour);
	}
}
