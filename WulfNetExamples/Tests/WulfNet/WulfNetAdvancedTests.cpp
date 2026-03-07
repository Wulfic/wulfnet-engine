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

#include <WulfNet/Jolt/Physics/Body/BodyCreationSettings.h>
#include <WulfNet/Jolt/Physics/Collision/Shape/BoxShape.h>
#include <WulfNet/Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Renderer/DebugRendererImp.h>
#include <SamplesLayers.h>

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
