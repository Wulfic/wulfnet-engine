// SPDX-License-Identifier: MIT
// WulfNet Advanced Visual Tests — Implementation (v2, fully fixed)
//
// Integrates Phase 6-8 systems into the Jolt Samples 3D viewer:
//   1. Gaseous dynamics — fire with pre-seeded density  (FIXED: emitter position, pre-seed)
//   2. Voronoi destruction — house demolished by cannonball (FIXED: EvaluateImpact + fragments)
//   3. Terrain deformation — flat plain with vehicle tire tracks (FIXED: flat ground, gentle deform)
//   4. Volumetric cloud — pre-filled density cloud (FIXED: gaussian fill, slow dissipation)
//   5. Spatial audio & acoustic ray visualization

#include <Samples.h>
#include "WulfNetAdvancedTests.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Renderer/DebugRendererImp.h>
#include <Layers.h>

#include <cmath>
#include <algorithm>

// ======================================================================
//  Utility
// ======================================================================
static float Clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// =====================================================================
//  1.  GASEOUS DYNAMICS TEST — Fire / Smoke
//
//  Previous bugs:
//    - Emitter at world Y=0.3 mapped to grid row j=0 (boundary cell).
//      AdvectFields() skips boundary rows (loop starts at j=1),
//      so injected density never advected upward.
//    - No pre-seeded density → nothing visible for several seconds.
//
//  Fixes:
//    - Larger cells (0.25m), emitter at Y=0.85 → grid row 3 (safely interior).
//    - Pre-seed a sphere of density around the emitter for instant visibility.
//    - Much higher densityRate (40) and temperatureRate (1000).
//    - Very slow dissipation (0.999) so density accumulates.
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetGasTest)
{
	JPH_ADD_BASE_CLASS(WulfNetGasTest, Test)
}

void WulfNetGasTest::Initialize()
{
	CreateFloor();

	// 16x24x16 grid, 0.25m cells → 4m x 6m x 4m domain
	mGasConfig.resolutionX = 16;
	mGasConfig.resolutionY = 24;
	mGasConfig.resolutionZ = 16;
	mGasConfig.cellSize    = 0.25f;
	// Center grid around X=0, Z=0; bottom at Y=0.1
	mGasConfig.originX = -(16 * 0.25f) / 2.0f; // -2.0
	mGasConfig.originY = 0.1f;
	mGasConfig.originZ = -(16 * 0.25f) / 2.0f; // -2.0

	// Strong buoyancy for dramatic fire plume
	mGasConfig.buoyancyAlpha       = 0.05f;
	mGasConfig.buoyancyBeta        = 1.0f;
	mGasConfig.vorticityStrength   = 0.5f;

	// Very slow dissipation — density and heat accumulate
	mGasConfig.densityDissipation     = 0.999f;
	mGasConfig.temperatureDissipation = 0.99f;
	mGasConfig.velocityDissipation    = 0.995f;
	mGasConfig.fuelDissipation        = 0.999f;

	// Fast pressure solve for real-time
	mGasConfig.pressureIterations = 15;
	mGasConfig.substeps           = 1;
	mGasConfig.maxTimestep        = 1.0f / 30.0f;

	// Combustion: ignite easily
	mGasConfig.ignitionTemperature = 400.0f;
	mGasConfig.burnRate            = 3.0f;
	mGasConfig.burnTemperature     = 1200.0f;
	mGasConfig.sootGeneration      = 1.0f;

	mGas.Initialize(mGasConfig);

	// Fire emitter centered well inside the grid
	// World (0, 0.85, 0) → grid (8, 3, 8) — safely 3 rows above bottom boundary
	WulfNet::GasEmitter fireEmitter;
	fireEmitter.type            = WulfNet::GasEmitterType::Sphere;
	fireEmitter.posX            = 0.0f;
	fireEmitter.posY            = 0.85f;
	fireEmitter.posZ            = 0.0f;
	fireEmitter.radius          = 0.6f;   // ~2.4 cells → covers a 5x5x5 region
	fireEmitter.densityRate     = 40.0f;
	fireEmitter.temperatureRate = 1000.0f;
	fireEmitter.fuelRate        = 5.0f;
	fireEmitter.velocityX       = 0.0f;
	fireEmitter.velocityY       = 0.8f;   // gentle upward push (m/s)
	fireEmitter.velocityZ       = 0.0f;
	fireEmitter.enabled         = true;
	mGas.AddEmitter(fireEmitter);

	// Pre-seed density in a sphere around the emitter so there's something
	// visible from frame 1 even before the sim accumulates.
	for (uint32_t j = 2; j < 10; ++j)
	{
		for (uint32_t k = 5; k < 11; ++k)
		{
			for (uint32_t i = 5; i < 11; ++i)
			{
				float dx = static_cast<float>(i) - 8.0f;
				float dy = static_cast<float>(j) - 3.0f;
				float dz = static_cast<float>(k) - 8.0f;
				float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
				if (dist < 4.0f)
				{
					float falloff = 1.0f - dist / 4.0f;
					mGas.SetDensity(i, j, k, 3.0f * falloff);
					mGas.SetTemperature(i, j, k, 500.0f * falloff);
					mGas.SetFuel(i, j, k, 0.8f * falloff);
				}
			}
		}
	}

	mTime = 0.0f;
}

void WulfNetGasTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = std::min(inParams.mDeltaTime, 1.0f / 30.0f);
	mTime += dt;

	mGas.Step(dt);
	DrawGasField();
}

void WulfNetGasTest::DrawGasField()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	const float cellSize = mGas.GetCellSize();
	const uint32_t rx = mGas.GetResolutionX();
	const uint32_t ry = mGas.GetResolutionY();
	const uint32_t rz = mGas.GetResolutionZ();

	for (uint32_t j = 0; j < ry; ++j)
	{
		for (uint32_t k = 0; k < rz; ++k)
		{
			for (uint32_t i = 0; i < rx; ++i)
			{
				const auto &cell = mGas.GetCell(i, j, k);
				// Draw if there's any meaningful density OR temperature
				if (cell.density < 0.01f && cell.temperature < 10.0f)
					continue;

				float wx, wy, wz;
				mGas.GridToWorld(static_cast<float>(i) + 0.5f,
				                 static_cast<float>(j) + 0.5f,
				                 static_cast<float>(k) + 0.5f,
				                 wx, wy, wz);

				float t = Clamp01(cell.temperature / 800.0f);
				float d = Clamp01(cell.density / 5.0f);
				// Ensure hot cells are visible even with low density
				float vis = std::max(d, t * 0.5f);

				uint8_t r, g, b, a;
				if (t > 0.6f)
				{
					// Hot core: white-yellow
					float f = (t - 0.6f) / 0.4f;
					r = 255;
					g = (uint8_t)(200 + 55 * f);
					b = (uint8_t)(50 + 180 * f);
					a = (uint8_t)(200 * vis);
				}
				else if (t > 0.2f)
				{
					// Flame: orange-red
					float f = (t - 0.2f) / 0.4f;
					r = (uint8_t)(150 + 105 * f);
					g = (uint8_t)(30 + 100 * f);
					b = 10;
					a = (uint8_t)(180 * vis);
				}
				else
				{
					// Cool smoke: gray
					uint8_t gray = (uint8_t)(40 + 60 * t / 0.2f);
					r = gray;
					g = gray;
					b = (uint8_t)(gray + 10);
					a = (uint8_t)(140 * vis);
				}

				if (a < 5) continue;

				Color c(r, g, b, a);
				// Larger sphere radius for solid visual appearance
				float sphereR = cellSize * 0.55f * (0.6f + 0.4f * vis);
				mDebugRenderer->DrawSphere(RVec3(wx, wy, wz), sphereR, c,
					DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
			}
		}
	}

	const auto &stats = mGas.GetStats();
	char buf[256];
	snprintf(buf, sizeof(buf), "Gas: active=%u  density=%.1f  maxT=%.0fK  maxV=%.1f",
	         stats.activeCells, stats.totalDensity, stats.maxTemperature, stats.maxVelocity);
	mDebugRenderer->DrawText3D(RVec3(0, 7, 0), buf, Color::sYellow, 0.25f);
	mDebugRenderer->DrawText3D(RVec3(0, 7.5, 0), "Gaseous Dynamics — Fire Simulation", Color::sWhite, 0.4f);
#endif
}

// =====================================================================
//  2.  DESTRUCTION TEST — House demolished by cannonball
//
//  Previous bugs:
//    - EvaluateImpact() was NEVER called → no actual fracture.
//    - Even if fracture occurred, no fragment Jolt bodies were created.
//    - Projectiles too light and slow to wake up the walls.
//
//  Fixes:
//    - Build a simple house (4 walls + roof) from dynamic boxes.
//    - Heavy cannonball (50 kg, 30 m/s) aimed at front wall.
//    - Detect wall activation (collision) → use velocity to estimate impulse.
//    - Call EvaluateImpact → on fracture, remove intact body, create real
//      fragment bodies from VoronoiCell AABBs.
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetDestructionVisualTest)
{
	JPH_ADD_BASE_CLASS(WulfNetDestructionVisualTest, Test)
}

void WulfNetDestructionVisualTest::Initialize()
{
	CreateFloor();

	WulfNet::DestructionConfig cfg;
	cfg.defaultCellCount       = 12;
	cfg.fragmentEjectionSpeed  = 5.0f;
	cfg.fragmentAngularSpeed   = 8.0f;
	cfg.fragmentLifetime       = 30.0f;
	cfg.enableSecondaryFracture = false;
	mDestruction.Initialize(cfg);

	// ---- Build a simple house: 4 walls + flat roof ----
	float wallThick = 0.2f;
	float wallH     = 2.0f;
	float houseW    = 3.0f; // X width
	float houseD    = 3.0f; // Z depth
	float roofThick = 0.15f;

	struct WallSpec {
		Vec3  halfExt;
		RVec3 pos;
	};

	WallSpec specs[] = {
		// Front wall (faces -Z, cannonball comes from -Z)
		{ Vec3(houseW / 2, wallH / 2, wallThick / 2), RVec3(0, wallH / 2, -houseD / 2) },
		// Back wall
		{ Vec3(houseW / 2, wallH / 2, wallThick / 2), RVec3(0, wallH / 2,  houseD / 2) },
		// Left wall
		{ Vec3(wallThick / 2, wallH / 2, houseD / 2), RVec3(-houseW / 2, wallH / 2, 0) },
		// Right wall
		{ Vec3(wallThick / 2, wallH / 2, houseD / 2), RVec3( houseW / 2, wallH / 2, 0) },
		// Roof
		{ Vec3(houseW / 2 + 0.2f, roofThick / 2, houseD / 2 + 0.2f), RVec3(0, wallH + roofThick / 2, 0) },
	};

	for (int si = 0; si < 5; ++si)
	{
		auto &s = specs[si];

		BodyCreationSettings bcs(
			new BoxShape(s.halfExt),
			s.pos,
			Quat::sIdentity(),
			EMotionType::Dynamic,
			Layers::MOVING);
		bcs.mRestitution = 0.1f;
		bcs.mFriction    = 0.6f;
		// Heavy walls so they don't drift from gravity alone
		bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		bcs.mMassPropertiesOverride.mMass = 200.0f;
		bcs.mLinearDamping  = 0.2f;
		bcs.mAngularDamping = 0.2f;

		BodyID bodyId = mBodyInterface->CreateAndAddBody(bcs, EActivation::DontActivate);

		// Register as destructible with a low threshold (cannonball delivers ~1500 N·s)
		uint32_t handle = mDestruction.AddDestructible(bodyId, 150.0f, 12);
		auto *dbody = mDestruction.GetDestructible(handle);
		if (dbody)
			dbody->pattern = WulfNet::DestructionSystem::GenerateBoxPattern(
				s.halfExt.GetX(), s.halfExt.GetY(), s.halfExt.GetZ(), 12, 2000.0f);

		DestrWall w;
		w.bodyId         = bodyId;
		w.destructHandle = handle;
		w.halfX = s.halfExt.GetX();
		w.halfY = s.halfExt.GetY();
		w.halfZ = s.halfExt.GetZ();
		w.pos   = s.pos;
		mWalls.push_back(w);
	}

	mLaunchTimer = mLaunchInterval * 0.8f; // first shot arrives quickly
}

void WulfNetDestructionVisualTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = inParams.mDeltaTime;

	// Periodically launch cannonball
	mLaunchTimer += dt;
	if (mLaunchTimer >= mLaunchInterval)
	{
		mLaunchTimer = 0.0f;
		LaunchProjectile();
	}

	// ---- Check for collisions and trigger fracture ----
	for (size_t wi = 0; wi < mWalls.size(); ++wi)
	{
		auto &w = mWalls[wi];
		auto *dbody = mDestruction.GetDestructible(w.destructHandle);
		if (!dbody || dbody->fractured) continue;

		// Skip if the body is no longer in the simulation
		if (!mBodyInterface->IsAdded(w.bodyId)) continue;

		// Skip sleeping bodies — they haven't been smashed yet
		if (!mBodyInterface->IsActive(w.bodyId)) continue;

		// If the wall gained significant velocity, it was hit by a projectile
		Vec3 vel = mBodyInterface->GetLinearVelocity(w.bodyId);
		float speed = vel.Length();
		if (speed < 1.0f) continue; // not hit hard enough

		// Estimate impulse = mass * deltaV ≈ mass * speed
		RVec3 bodyPos = mBodyInterface->GetCenterOfMassPosition(w.bodyId);
		float impulse = speed * 200.0f; // wall mass = 200 kg

		bool fractured = mDestruction.EvaluateImpact(
			w.destructHandle,
			(float)bodyPos.GetX(), (float)bodyPos.GetY(), (float)bodyPos.GetZ(),
			impulse);

		if (fractured)
		{
			// Remove intact wall body
			mBodyInterface->RemoveBody(w.bodyId);
			mBodyInterface->DestroyBody(w.bodyId);

			// Create real Jolt fragment bodies from Voronoi cells
			for (size_t ci = 0; ci < dbody->pattern.cells.size(); ++ci)
			{
				const auto &cell = dbody->pattern.cells[ci];
				if (cell.mass < 0.1f) continue;

				float fragHX = (cell.maxX - cell.minX) * 0.5f;
				float fragHY = (cell.maxY - cell.minY) * 0.5f;
				float fragHZ = (cell.maxZ - cell.minZ) * 0.5f;
				if (fragHX < 0.02f || fragHY < 0.02f || fragHZ < 0.02f) continue;

				RVec3 fragPos = bodyPos + RVec3(cell.centerX, cell.centerY, cell.centerZ);

				BodyCreationSettings fragBCS(
					new BoxShape(Vec3(fragHX, fragHY, fragHZ)),
					fragPos,
					Quat::sIdentity(),
					EMotionType::Dynamic,
					Layers::MOVING);
				fragBCS.mRestitution = 0.2f;
				fragBCS.mFriction    = 0.5f;
				fragBCS.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
				fragBCS.mMassPropertiesOverride.mMass = cell.mass;

				// Eject fragments outward from impact point
				Vec3 ejectDir(cell.centerX, cell.centerY + 0.5f, cell.centerZ);
				float ejectLen = ejectDir.Length();
				if (ejectLen > 0.01f) ejectDir /= ejectLen;
				fragBCS.mLinearVelocity  = vel + ejectDir * 5.0f;
				fragBCS.mAngularVelocity = Vec3(
					(float)((int)ci % 3) - 1.0f,
					(float)((int)ci % 5) - 2.0f,
					(float)((int)ci % 7) - 3.0f) * 3.0f;

				mBodyInterface->CreateAndAddBody(fragBCS, EActivation::Activate);
			}
		}
	}

	DrawFracturePatterns();
}

void WulfNetDestructionVisualTest::LaunchProjectile()
{
	// Remove oldest projectile if over limit
	while ((int)mProjectiles.size() >= mMaxProjectiles)
	{
		if (mBodyInterface->IsAdded(mProjectiles.front()))
		{
			mBodyInterface->RemoveBody(mProjectiles.front());
			mBodyInterface->DestroyBody(mProjectiles.front());
		}
		mProjectiles.erase(mProjectiles.begin());
	}

	// Aim at the front wall with some random spread
	std::uniform_real_distribution<float> xDist(-0.8f, 0.8f);
	std::uniform_real_distribution<float> yDist(0.3f, 1.8f);
	float tx = xDist(mRng);
	float ty = yDist(mRng);

	RVec3 launchPos(tx, ty, -10.0);
	Vec3  launchVel(0.0f, 1.0f, 30.0f); // fast & high arc

	BodyCreationSettings bcs(
		new SphereShape(0.4f), // big cannonball
		launchPos,
		Quat::sIdentity(),
		EMotionType::Dynamic,
		Layers::MOVING);
	bcs.mRestitution    = 0.3f;
	bcs.mFriction       = 0.3f;
	bcs.mLinearVelocity = launchVel;
	bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
	bcs.mMassPropertiesOverride.mMass = 50.0f; // 50 kg cannonball

	BodyID ball = mBodyInterface->CreateAndAddBody(bcs, EActivation::Activate);
	mProjectiles.push_back(ball);
}

void WulfNetDestructionVisualTest::DrawFracturePatterns()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	// Draw Voronoi cell wireframes for intact walls
	for (auto &w : mWalls)
	{
		auto *dbody = mDestruction.GetDestructible(w.destructHandle);
		if (!dbody || dbody->pattern.cells.empty()) continue;

		// If already fractured, skip wireframe (fragments are their own bodies now)
		if (dbody->fractured) continue;

		RVec3 bodyPos = mBodyInterface->IsAdded(w.bodyId)
			? mBodyInterface->GetCenterOfMassPosition(w.bodyId)
			: w.pos;

		for (size_t ci = 0; ci < dbody->pattern.cells.size(); ++ci)
		{
			const auto &cell = dbody->pattern.cells[ci];
			Color col = Color::sGetDistinctColor((int)ci);
			col = Color(col, 100);

			RVec3 mn = bodyPos + RVec3(cell.minX, cell.minY, cell.minZ);
			RVec3 mx = bodyPos + RVec3(cell.maxX, cell.maxY, cell.maxZ);

			RVec3 corners[8] = {
				{mn.GetX(), mn.GetY(), mn.GetZ()},
				{mx.GetX(), mn.GetY(), mn.GetZ()},
				{mx.GetX(), mx.GetY(), mn.GetZ()},
				{mn.GetX(), mx.GetY(), mn.GetZ()},
				{mn.GetX(), mn.GetY(), mx.GetZ()},
				{mx.GetX(), mn.GetY(), mx.GetZ()},
				{mx.GetX(), mx.GetY(), mx.GetZ()},
				{mn.GetX(), mx.GetY(), mx.GetZ()},
			};

			int edges[12][2] = {
				{0, 1}, {1, 2}, {2, 3}, {3, 0},
				{4, 5}, {5, 6}, {6, 7}, {7, 4},
				{0, 4}, {1, 5}, {2, 6}, {3, 7}
			};
			for (auto &e : edges)
				mDebugRenderer->DrawArrow(corners[e[0]], corners[e[1]], col, 0.005f);
		}
	}

	// Highlight cannonballs
	for (auto &pid : mProjectiles)
	{
		if (mBodyInterface->IsAdded(pid) && mBodyInterface->IsActive(pid))
		{
			RVec3 pos = mBodyInterface->GetCenterOfMassPosition(pid);
			mDebugRenderer->DrawSphere(pos, 0.42f, Color(255, 60, 20, 240),
				DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
		}
	}

	mDebugRenderer->DrawText3D(RVec3(0, 5, 0), "Voronoi Destruction — Cannonball vs House", Color::sWhite, 0.35f);
#endif
}

// =====================================================================
//  3.  TERRAIN DEFORMATION TEST — Flat plain with tire tracks
//
//  Previous bugs:
//    - Sinusoidal terrain + constant high-force cratering every 1.8s
//      completely shredded the surface within seconds.
//    - Material was mixed rock/soil, rocks barely deformed → inconsistent.
//
//  Fixes:
//    - Perfectly flat terrain (all heights = 0).
//    - All SoftSoil material for uniform, visible deformation.
//    - Vehicle proxy orbits in a circle leaving tire tracks every frame.
//    - Craters are rare (every 4 s), small radius (1.0), low force (0.4).
//    - Pre-apply one lap of tire tracks so there's immediate visible content.
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetTerrainVisualTest)
{
	JPH_ADD_BASE_CLASS(WulfNetTerrainVisualTest, Test)
}

void WulfNetTerrainVisualTest::Initialize()
{
	// No CreateFloor() — the terrain IS the ground surface

	// 80x80 grid, 0.25m cells = 20m x 20m terrain
	mTerrainConfig.gridSizeX  = 80;
	mTerrainConfig.gridSizeZ  = 80;
	mTerrainConfig.cellSize   = 0.25f;
	mTerrainConfig.originX    = -(80 * 0.25f) / 2.0f; // -10.0
	mTerrainConfig.originZ    = -(80 * 0.25f) / 2.0f; // -10.0
	mTerrainConfig.originY    = 0.0f;
	mTerrainConfig.maxDeformDepth = 1.0f;
	mTerrainConfig.maxDeformRaise = 0.5f;

	mTerrain.Initialize(mTerrainConfig);

	// Perfectly flat ground at Y=0
	std::vector<float> heights(80 * 80, 0.0f);
	mTerrain.SetHeightField(heights.data(), 80, 80);

	// All soft soil for easy, visible deformation
	WulfNet::TerrainMaterial soil = WulfNet::TerrainMaterial::SoftSoil();
	mTerrain.SetMaterialRegion(0, 0, 79, 79, soil);

	mTime       = 0.0f;
	mDropTimer  = 0.0f;
	mTrackAngle = 0.0f;

	// Pre-apply one full lap of tire tracks for immediate visibility
	for (float a = 0.02f; a < 6.28f; a += 0.02f)
	{
		float r = 5.0f;
		float x0 = cosf(a - 0.02f) * r;
		float z0 = sinf(a - 0.02f) * r;
		float x1 = cosf(a) * r;
		float z1 = sinf(a) * r;
		mTerrain.ApplyTireTrack(x0, z0, x1, z1, 0.3f, 0.04f, 0.4f);
	}
}

void WulfNetTerrainVisualTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = inParams.mDeltaTime;
	mTime += dt;

	// Vehicle drives in a circle, leaving tire tracks
	float speed = 0.5f; // radians per second
	float prevAngle = mTrackAngle;
	mTrackAngle += speed * dt;

	// Outer track
	float trackR = 5.0f;
	float tx0 = cosf(prevAngle) * trackR;
	float tz0 = sinf(prevAngle) * trackR;
	float tx1 = cosf(mTrackAngle) * trackR;
	float tz1 = sinf(mTrackAngle) * trackR;
	mTerrain.ApplyTireTrack(tx0, tz0, tx1, tz1, 0.3f, 0.04f, 0.4f);

	// Inner track (simulates 4 wheels)
	float trackR2 = 4.4f;
	float ix0 = cosf(prevAngle) * trackR2;
	float iz0 = sinf(prevAngle) * trackR2;
	float ix1 = cosf(mTrackAngle) * trackR2;
	float iz1 = sinf(mTrackAngle) * trackR2;
	mTerrain.ApplyTireTrack(ix0, iz0, ix1, iz1, 0.3f, 0.04f, 0.4f);

	// Occasional small crater (every 4 seconds)
	mDropTimer += dt;
	if (mDropTimer >= 4.0f)
	{
		mDropTimer = 0.0f;
		std::uniform_real_distribution<float> posDist(-6.0f, 6.0f);
		float wx = posDist(mRng);
		float wz = posDist(mRng);
		mTerrain.ApplyExplosion(wx, wz, 1.0f, 0.4f);
	}

	DrawTerrain();
}

void WulfNetTerrainVisualTest::DrawTerrain()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	const float cs = mTerrainConfig.cellSize;
	const uint32_t sx = mTerrainConfig.gridSizeX;
	const uint32_t sz = mTerrainConfig.gridSizeZ;
	const float ox = mTerrainConfig.originX;
	const float oz = mTerrainConfig.originZ;

	for (uint32_t z = 0; z < sz - 1; ++z)
	{
		for (uint32_t x = 0; x < sx - 1; ++x)
		{
			float h00 = mTerrain.GetHeightAt(x, z);
			float h10 = mTerrain.GetHeightAt(x + 1, z);
			float h01 = mTerrain.GetHeightAt(x, z + 1);
			float h11 = mTerrain.GetHeightAt(x + 1, z + 1);

			float wx0 = ox + x * cs;
			float wx1 = ox + (x + 1) * cs;
			float wz0 = oz + z * cs;
			float wz1 = oz + (z + 1) * cs;

			RVec3 p00(wx0, h00, wz0);
			RVec3 p10(wx1, h10, wz0);
			RVec3 p01(wx0, h01, wz1);
			RVec3 p11(wx1, h11, wz1);

			// Color: green for flat, brown for depressions (tire tracks/craters)
			float avgDelta = (h00 + h10 + h01 + h11) * 0.25f;
			uint8_t r, g, b;
			if (avgDelta < -0.005f)
			{
				// Deformed: brown dirt
				float f = Clamp01(-avgDelta / 0.3f);
				r = (uint8_t)(100 + 80 * f);
				g = (uint8_t)(80 + 20 * f);
				b = 40;
			}
			else if (avgDelta > 0.005f)
			{
				// Raised rim: lighter brown
				float f = Clamp01(avgDelta / 0.2f);
				r = (uint8_t)(120 + 60 * f);
				g = (uint8_t)(100 + 40 * f);
				b = (uint8_t)(50 + 20 * f);
			}
			else
			{
				// Flat grass
				r = 60; g = 130; b = 40;
			}

			Color col(r, g, b, 255);

			// Two triangles per quad
			mDebugRenderer->DrawTriangle(p00, p10, p01, col);
			mDebugRenderer->DrawTriangle(p10, p11, p01, col);
		}
	}

	// Draw a "vehicle" marker at current track position
	float vx = cosf(mTrackAngle) * 5.0f;
	float vz = sinf(mTrackAngle) * 5.0f;
	float vh = mTerrain.SampleHeight(vx, vz);
	RVec3 vehiclePos(vx, vh + 0.4f, vz);
	mDebugRenderer->DrawSphere(vehiclePos, 0.3f, Color(200, 200, 50, 230),
		DebugRenderer::ECastShadow::On, DebugRenderer::EDrawMode::Solid);
	mDebugRenderer->DrawText3D(vehiclePos + RVec3(0, 0.5, 0), "Vehicle", Color::sYellow, 0.2f);

	// Forward direction arrow
	float fwdX = -sinf(mTrackAngle) * 1.0f;
	float fwdZ =  cosf(mTrackAngle) * 1.0f;
	mDebugRenderer->DrawArrow(vehiclePos, vehiclePos + RVec3(fwdX, 0, fwdZ), Color::sGreen, 0.04f);

	const auto &stats = mTerrain.GetStats();
	char buf[256];
	snprintf(buf, sizeof(buf), "Terrain 20x20m: %u deformations, %u cells modified",
	         stats.totalDeformations, stats.cellsModified);
	mDebugRenderer->DrawText3D(RVec3(0, 4, 0), buf, Color::sYellow, 0.2f);
	mDebugRenderer->DrawText3D(RVec3(0, 4.5, 0), "Terrain Deformation — Vehicle Tire Tracks", Color::sWhite, 0.35f);
#endif
}

// =====================================================================
//  4.  VOLUMETRIC CLOUD — pre-filled density cloud
//
//  Previous bugs:
//    - Relied solely on emitters near grid boundaries → sparse dots.
//    - No pre-filled density → took forever to build up.
//    - Small cell size + fast dissipation → dots flew apart.
//
//  Fixes:
//    - Pre-fill the entire grid with a smooth gaussian cloud blob.
//    - Extremely slow dissipation (0.9998) preserves shape.
//    - One gentle center emitter to replenish.
//    - No gravity, minimal buoyancy → cloud hangs in place.
//    - Larger cells (0.3m), larger render spheres → cohesive volume.
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetVolumetricVisualTest)
{
	JPH_ADD_BASE_CLASS(WulfNetVolumetricVisualTest, Test)
}

void WulfNetVolumetricVisualTest::Initialize()
{
	CreateFloor();

	// 20x12x20 grid, 0.3m cells → 6m x 3.6m x 6m cloud volume
	mGasConfig.resolutionX = 20;
	mGasConfig.resolutionY = 12;
	mGasConfig.resolutionZ = 20;
	mGasConfig.cellSize    = 0.3f;
	mGasConfig.originX     = -(20 * 0.3f) / 2.0f; // -3.0
	mGasConfig.originY     = 2.0f;                  // floating above ground
	mGasConfig.originZ     = -(20 * 0.3f) / 2.0f;

	// Cloud behavior: almost no movement, very slow fade
	mGasConfig.buoyancyAlpha       = 0.0f;
	mGasConfig.buoyancyBeta        = 0.05f;   // tiny buoyancy
	mGasConfig.vorticityStrength   = 0.1f;
	mGasConfig.gravityY            = 0.0f;     // clouds don't fall

	mGasConfig.densityDissipation     = 0.9998f; // very slow fade
	mGasConfig.temperatureDissipation = 0.999f;
	mGasConfig.velocityDissipation    = 0.98f;   // velocities dampen quickly

	mGasConfig.pressureIterations = 10;
	mGasConfig.substeps           = 1;
	mGasConfig.maxTimestep        = 1.0f / 30.0f;

	mGas.Initialize(mGasConfig);

	// Pre-fill the grid with a smooth gaussian-like cloud blob
	float centerI = 10.0f, centerJ = 6.0f, centerK = 10.0f;
	float cloudRadiusI = 7.0f, cloudRadiusJ = 4.0f, cloudRadiusK = 7.0f;

	for (uint32_t j = 0; j < 12; ++j)
	{
		for (uint32_t k = 0; k < 20; ++k)
		{
			for (uint32_t i = 0; i < 20; ++i)
			{
				float di = (static_cast<float>(i) - centerI) / cloudRadiusI;
				float dj = (static_cast<float>(j) - centerJ) / cloudRadiusJ;
				float dk = (static_cast<float>(k) - centerK) / cloudRadiusK;
				float dist2 = di * di + dj * dj + dk * dk;

				if (dist2 < 1.0f)
				{
					float falloff = (1.0f - dist2);
					falloff = falloff * falloff; // sharper edges

					// Add noise for natural cloud appearance
					float noise = 0.7f + 0.3f * sinf(i * 2.1f + k * 3.7f + j * 1.3f);
					float density = 3.0f * falloff * noise;

					if (density > 0.05f)
					{
						mGas.SetDensity(i, j, k, density);
						mGas.SetTemperature(i, j, k, 30.0f * falloff);
					}
				}
			}
		}
	}

	// One gentle emitter at cloud center to keep it alive
	WulfNet::GasEmitter em;
	em.type          = WulfNet::GasEmitterType::Sphere;
	em.posX          = 0.0f;
	em.posY          = 3.8f; // middle of the cloud volume
	em.posZ          = 0.0f;
	em.radius        = 1.0f;
	em.densityRate   = 2.0f;     // gentle replenishment
	em.temperatureRate = 10.0f;
	em.fuelRate      = 0.0f;
	em.velocityX     = 0.0f;
	em.velocityY     = 0.0f;
	em.velocityZ     = 0.0f;
	em.enabled       = true;
	mGas.AddEmitter(em);

	mTime = 0.0f;
}

void WulfNetVolumetricVisualTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = std::min(inParams.mDeltaTime, 1.0f / 30.0f);
	mTime += dt;
	mGas.Step(dt);
	DrawVolumetricField();
}

void WulfNetVolumetricVisualTest::DrawVolumetricField()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	const float cellSize = mGas.GetCellSize();
	const uint32_t rx = mGas.GetResolutionX();
	const uint32_t ry = mGas.GetResolutionY();
	const uint32_t rz = mGas.GetResolutionZ();

	for (uint32_t j = 0; j < ry; ++j)
	{
		for (uint32_t k = 0; k < rz; ++k)
		{
			for (uint32_t i = 0; i < rx; ++i)
			{
				const auto &cell = mGas.GetCell(i, j, k);
				if (cell.density < 0.02f) continue;

				float wx, wy, wz;
				mGas.GridToWorld(i + 0.5f, j + 0.5f, k + 0.5f, wx, wy, wz);

				float d = Clamp01(cell.density / 3.0f);

				// Cloud coloring: white to light gray with soft transparency
				uint8_t brightness = (uint8_t)(200 + 55 * d);
				uint8_t a = (uint8_t)(40 + 160 * d);

				Color c(brightness, brightness, (uint8_t)(brightness - 5), a);
				float sphereR = cellSize * 0.6f * (0.5f + 0.5f * d);
				mDebugRenderer->DrawSphere(RVec3(wx, wy, wz), sphereR, c,
					DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
			}
		}
	}

	const auto &stats = mGas.GetStats();
	char buf[128];
	snprintf(buf, sizeof(buf), "Cloud: %u cells, density=%.1f", stats.activeCells, stats.totalDensity);
	mDebugRenderer->DrawText3D(RVec3(0, 7, 0), buf, Color::sYellow, 0.25f);
	mDebugRenderer->DrawText3D(RVec3(0, 7.5, 0), "Volumetric Cloud / Fog", Color::sWhite, 0.4f);
#endif
}

// =====================================================================
//  5.  SPATIAL AUDIO & ACOUSTIC VISUALIZATION TEST
//      (This test worked correctly — kept largely unchanged)
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetAudioVisualTest)
{
	JPH_ADD_BASE_CLASS(WulfNetAudioVisualTest, Test)
}

void WulfNetAudioVisualTest::Initialize()
{
	CreateFloor();

	// Build a room from static walls
	float wallThick = 0.15f;
	float roomH     = 5.0f;
	float roomHX    = 6.0f, roomHZ = 6.0f;

	struct WallDef { Vec3 halfExt; RVec3 pos; };
	WallDef walls[] = {
		{ Vec3(roomHX, wallThick, roomHZ), RVec3(0, roomH, 0) },          // ceiling
		{ Vec3(roomHX, roomH / 2, wallThick), RVec3(0, roomH / 2, -roomHZ) }, // back
		{ Vec3(roomHX, roomH / 2, wallThick), RVec3(0, roomH / 2,  roomHZ) }, // front
		{ Vec3(wallThick, roomH / 2, roomHZ), RVec3(-roomHX, roomH / 2, 0) }, // left
		{ Vec3(wallThick, roomH / 2, roomHZ), RVec3( roomHX, roomH / 2, 0) }, // right
	};

	for (auto &w : walls)
	{
		BodyCreationSettings bcs(
			new BoxShape(w.halfExt), w.pos,
			Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
		BodyID bid = mBodyInterface->CreateAndAddBody(bcs, EActivation::DontActivate);
		mRoomWalls.push_back(bid);
	}

	mSources.push_back({ RVec3(-3, 1.5, -3), Color::sRed,    8.0f, "Music" });
	mSources.push_back({ RVec3( 4, 2.0,  2), Color::sGreen,  6.0f, "Speech" });
	mSources.push_back({ RVec3( 0, 0.5, -4), Color(255, 165, 0, 255), 10.0f, "Footsteps" });

	mSpatialAudio.Initialize(44100);

	WulfNet::AcousticConfig acfg;
	acfg.maxBounces    = 4;
	acfg.numRays       = 32;
	acfg.maxDistance    = 20.0f;
	acfg.roomProbeRays = 16;
	mAcoustics.Initialize(acfg);

	// Ray-cast callback for the box-shaped room
	const RVec3 rmin = mRoomMin;
	const RVec3 rmax = mRoomMax;
	mAcoustics.SetRayCastFunction(
		[rmin, rmax](float ox, float oy, float oz,
		             float dx, float dy, float dz,
		             float maxDist) -> WulfNet::AcousticRayHit
		{
			WulfNet::AcousticRayHit result;
			result.hit      = false;
			result.distance = maxDist;

			float planes[][4] = {
				{ 1, 0, 0, (float)-rmin.GetX()},
				{-1, 0, 0, (float) rmax.GetX()},
				{ 0, 1, 0, (float)-rmin.GetY()},
				{ 0,-1, 0, (float) rmax.GetY()},
				{ 0, 0, 1, (float)-rmin.GetZ()},
				{ 0, 0,-1, (float) rmax.GetZ()},
			};

			for (auto &p : planes)
			{
				float denom = p[0] * dx + p[1] * dy + p[2] * dz;
				if (fabsf(denom) < 1e-6f) continue;
				float t = -(p[0] * ox + p[1] * oy + p[2] * oz + p[3]) / denom;
				if (t > 0.001f && t < result.distance)
				{
					float hx = ox + dx * t;
					float hy = oy + dy * t;
					float hz = oz + dz * t;
					if (hx >= (float)rmin.GetX() - 0.01f && hx <= (float)rmax.GetX() + 0.01f &&
					    hy >= (float)rmin.GetY() - 0.01f && hy <= (float)rmax.GetY() + 0.01f &&
					    hz >= (float)rmin.GetZ() - 0.01f && hz <= (float)rmax.GetZ() + 0.01f)
					{
						result.hit      = true;
						result.distance = t;
						result.normalX  = p[0];
						result.normalY  = p[1];
						result.normalZ  = p[2];
						result.materialId = 0;
					}
				}
			}
			return result;
		}
	);

	mAcoustics.AddMaterial(WulfNet::AcousticMaterial::Concrete());

	mAcousticInfo.resize(mSources.size());
	mUpdateTimer = 0.0f;
	mTime        = 0.0f;
}

void WulfNetAudioVisualTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = inParams.mDeltaTime;
	mTime += dt;

	// Slowly orbit the listener
	float lx = cosf(mTime * 0.3f) * 2.0f;
	float lz = sinf(mTime * 0.3f) * 2.0f;
	mListenerPos = RVec3(lx, 1.7, lz);
	mListenerFwd = RVec3(-lx, 0, -lz).Normalized();

	// Update acoustic data periodically (expensive)
	mUpdateTimer += dt;
	if (mUpdateTimer >= 0.3f)
	{
		mUpdateTimer = 0.0f;
		for (size_t i = 0; i < mSources.size(); ++i)
		{
			auto &src  = mSources[i];
			auto &info = mAcousticInfo[i];

			float sx = (float)src.position.GetX();
			float sy = (float)src.position.GetY();
			float sz = (float)src.position.GetZ();

			info.occlusion   = mAcoustics.ComputeOcclusion(
				sx, sy, sz,
				(float)mListenerPos.GetX(), (float)mListenerPos.GetY(), (float)mListenerPos.GetZ());
			info.obstruction = mAcoustics.ComputeObstruction(
				sx, sy, sz,
				(float)mListenerPos.GetX(), (float)mListenerPos.GetY(), (float)mListenerPos.GetZ(), 8);

			float dx = sx - (float)mListenerPos.GetX();
			float dy = sy - (float)mListenerPos.GetY();
			float dz = sz - (float)mListenerPos.GetZ();
			float dist = sqrtf(dx * dx + dy * dy + dz * dz);
			info.distGain = mSpatialAudio.ComputeDistanceGain(dist);

			info.ir = mAcoustics.TraceImpulseResponse(
				sx, sy, sz,
				(float)mListenerPos.GetX(), (float)mListenerPos.GetY(), (float)mListenerPos.GetZ());
		}
	}

	DrawAudioSources();
	DrawAcousticRays();
	DrawAttenuationRadii();
}

void WulfNetAudioVisualTest::DrawAudioSources()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	mDebugRenderer->DrawSphere(mListenerPos, 0.15f, Color::sWhite,
		DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
	mDebugRenderer->DrawText3D(mListenerPos + RVec3(0, 0.3, 0), "Listener", Color::sWhite, 0.2f);

	RVec3 fwdEnd = mListenerPos + mListenerFwd * 0.6;
	mDebugRenderer->DrawArrow(mListenerPos, fwdEnd, Color::sGreen, 0.03f);

	for (size_t i = 0; i < mSources.size(); ++i)
	{
		auto &src  = mSources[i];
		auto &info = mAcousticInfo[i];

		float pulse = 0.8f + 0.2f * sinf(mTime * 4.0f + (float)i * 2.0f);
		mDebugRenderer->DrawSphere(src.position, 0.12f * pulse, src.color,
			DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
		mDebugRenderer->DrawText3D(src.position + RVec3(0, 0.35, 0), src.label, src.color, 0.2f);

		char buf[128];
		snprintf(buf, sizeof(buf), "occ=%.2f obstr=%.2f gain=%.2f",
		         info.occlusion, info.obstruction, info.distGain);
		mDebugRenderer->DrawText3D(src.position + RVec3(0, 0.55, 0), buf, Color::sYellow, 0.15f);

		uint8_t occA = (uint8_t)(info.occlusion * 200);
		Color lineCol(info.occlusion > 0.5f ? (uint8_t)0 : (uint8_t)255,
		              info.occlusion > 0.5f ? (uint8_t)200 : (uint8_t)50,
		              (uint8_t)0, occA);
		mDebugRenderer->DrawArrow(src.position, mListenerPos, lineCol, 0.015f);
	}
#endif
}

void WulfNetAudioVisualTest::DrawAcousticRays()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	for (size_t i = 0; i < mSources.size(); ++i)
	{
		auto &src  = mSources[i];
		auto &info = mAcousticInfo[i];

		int maxTaps = std::min((int)info.ir.taps.size(), 8);
		for (int t = 0; t < maxTaps; ++t)
		{
			const auto &tap = info.ir.taps[t];
			if (tap.energy < 0.01f) continue;

			float reflDist = tap.time * 343.0f * 0.5f;
			RVec3 reflPoint = mListenerPos + RVec3(
				tap.direction[0] * reflDist,
				tap.direction[1] * reflDist,
				tap.direction[2] * reflDist);

			float energyF = Clamp01(tap.energy * 3.0f);
			uint8_t ea = (uint8_t)(100 * energyF);
			Color reflCol(200, 200, 255, ea);

			mDebugRenderer->DrawArrow(src.position, reflPoint, reflCol, 0.008f);
			mDebugRenderer->DrawArrow(reflPoint, mListenerPos, reflCol, 0.008f);
		}
	}
#endif
}

void WulfNetAudioVisualTest::DrawAttenuationRadii()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	for (auto &src : mSources)
	{
		Color faded(src.color, 60);
		mDebugRenderer->DrawWireSphere(src.position, src.radius, faded, 1);
		mDebugRenderer->DrawWireSphere(src.position, src.radius * 0.5f,
			Color(src.color, 40), 1);
	}

	auto room = mAcoustics.EstimateRoom(
		(float)mListenerPos.GetX(),
		(float)mListenerPos.GetY(),
		(float)mListenerPos.GetZ());

	char roomBuf[256];
	snprintf(roomBuf, sizeof(roomBuf), "Room: V=%.0f m3  SA=%.0f m2  RT60=%.2fs",
	         room.volume, room.surfaceArea, room.rt60);
	mDebugRenderer->DrawText3D(RVec3(0, 5.5, 0), roomBuf, Color::sYellow, 0.25f);
	mDebugRenderer->DrawText3D(RVec3(0, 6.0, 0), "Spatial Audio & Acoustics", Color::sWhite, 0.4f);
#endif
}
