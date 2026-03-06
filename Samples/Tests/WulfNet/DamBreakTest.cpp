// WulfNet Dam Break Test — Destructible dam + large-scale SWE flood
// 512x512 grid at 5m cell size = 2560x2560m (~6.5 km²) terrain.
// Physical destructible dam wall (Voronoi fracture) with auto-launched ball.
// When the ball shatters the wall, a 75m-deep reservoir floods the valley.

#include <TestFramework.h>
#include <Tests/WulfNet/DamBreakTest.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>

JPH_IMPLEMENT_RTTI_VIRTUAL(DamBreakTest)
{
	JPH_ADD_BASE_CLASS(DamBreakTest, Test)
}

void DamBreakTest::Initialize()
{
	// 512x512 grid at 5m cells = 2.56km × 2.56km (~6.5 km²)
	WulfNet::Physics::WaterSystemV3Config config;
	config.width       = cGridW;
	config.height      = cGridH;
	config.gridSize    = cGridSize;
	config.gravity     = 9.81f;
	config.fluxDamping = 0.5f;    // Baseline from WaterBoxTest — realistic settling & flow
	config.dtMax       = 0.016f;  // Standard timestep (CFL stable at 5m cells for depths < 100m)
	config.originX     = -(cGridW * cGridSize) / 2.0f;
	config.originZ     = -(cGridH * cGridSize) / 2.0f;

	mWaterSystem = new WulfNet::Physics::WaterSystemV3(config, mPhysicsSystem);
	auto &state = mWaterSystem->GetCPUState();

	const float centerX = cGridW / 2.0f;  // 256
	const float gs = cGridSize;
	const float wOX = config.originX;      // -1280
	const float wOZ = config.originZ;      // -1280

	// ===================================================================
	// TERRAIN — Alpine valley, 100-600m elevation
	//
	// The dam ridge is built into the heightmap to contain water.
	// Physical destructible wall segments sit on top as the visible dam.
	// ===================================================================
	for (uint32_t gy = 0; gy < cGridH; ++gy) {
		for (uint32_t gx = 0; gx < cGridW; ++gx) {
			uint32_t idx = gy * cGridW + gx;
			float fx = (float)gx;
			float fy = (float)gy;

			// Base slope: north=high, south=low
			float base = 480.0f - fy * 0.75f;

			// Mountain ridge along north edge
			float ridgeT = std::max(0.0f, 1.0f - fy / 60.0f);
			float ridge = 120.0f * ridgeT * ridgeT;

			// Mountain flanks enclosing reservoir (strong enough to contain 385m water)
			float armLT = std::max(0.0f, 1.0f - fx / 160.0f);
			float armLF = std::max(0.4f, std::min(1.0f, 1.0f - (fy - 200.0f) / 100.0f));
			float armL  = 200.0f * armLT * armLT * armLF;

			float armRT = std::max(0.0f, (fx - 352.0f) / 160.0f);
			float armRF = std::max(0.4f, std::min(1.0f, 1.0f - (fy - 200.0f) / 100.0f));
			float armR  = 200.0f * armRT * armRT * armRF;

			// Reservoir basin: parabolic depression
			float ldx = fx - centerX, ldy = fy - 120.0f;
			float lakeDist = std::sqrt(ldx * ldx + ldy * ldy);
			float lakeD = 0.0f;
			if (lakeDist < 100.0f) {
				float t = lakeDist / 100.0f;
				lakeD = 80.0f * (1.0f - t * t);
			}

			// Dam ridge in terrain — flat-topped with steep ramps so water can't leak
			// Covers full width between mountain flanks. Peak adds 100m.
			float damH = 0.0f;
			if (fy >= 193.0f && fy <= 220.0f && fx >= 50.0f && fx <= 462.0f) {
				// Steep ramp at Y edges (2-cell transition), flat top in center
				float yDist = std::min(fy - 193.0f, 220.0f - fy);
				float prof  = (yDist < 3.0f) ? yDist / 3.0f : 1.0f;
				// Taper at X edges to blend with mountain flanks
				float xDist = std::min(fx - 50.0f, 462.0f - fx);
				float xProf = (xDist < 20.0f) ? xDist / 20.0f : 1.0f;
				damH = 100.0f * prof * xProf;
			}

			// River gorge below dam
			float rvDist  = std::abs(fx - centerX);
			float rvWidth = 25.0f + std::max(0.0f, (fy - 220.0f)) * 0.08f;
			float rvDepth = 0.0f;
			if (fy > 218.0f && rvDist < rvWidth) {
				float t = rvDist / rvWidth;
				rvDepth = 20.0f * (1.0f - t * t);
			}

			// Natural noise
			float noise = 8.0f * std::sin(fx * 0.065f) * std::cos(fy * 0.048f)
			            + 5.0f * std::sin(fx * 0.14f  + fy * 0.11f)
			            + 3.0f * std::cos(fx * 0.03f  - fy * 0.085f);

			// Border containment walls
			float bDist  = std::min({fx, fy, (float)(cGridW - 1 - gx), (float)(cGridH - 1 - gy)});
			float border = (bDist < 8.0f) ? 200.0f * (1.0f - bDist / 8.0f) : 0.0f;

			float h = base + ridge + armL + armR - lakeD + damH - rvDepth + noise;
			state.terrainHeight[idx] = std::max(h, 10.0f) + border;
		}
	}

	// ===================================================================
	// FILL RESERVOIR — Water surface at 385m (dam crest ~410m)
	// ===================================================================
	const float waterSurface = 385.0f;
	for (uint32_t gy = 10; gy < 190; ++gy)
		for (uint32_t gx = 10; gx < cGridW - 10; ++gx) {
			uint32_t idx = gy * cGridW + gx;
			float terrH = state.terrainHeight[idx];
			if (terrH < waterSurface)
				mWaterSystem->AddWater(gx, gy, (waterSurface - terrH) * gs * gs);
		}

	// ===================================================================
	// DESTRUCTIBLE DAM WALL — Physical box segments with Voronoi fracture
	// Placed along the dam crest; gravity=0 so they hover in place.
	// When the ball hits them, they shatter and terrain drops underneath.
	// ===================================================================
	{
		WulfNet::DestructionConfig dcfg;
		dcfg.defaultCellCount       = 10;
		dcfg.fragmentEjectionSpeed  = 8.0f;
		dcfg.fragmentAngularSpeed   = 6.0f;
		dcfg.fragmentLifetime       = 30.0f;
		dcfg.maxTotalFragments      = 500;
		dcfg.maxFragmentsPerFrame   = 80;
		dcfg.enableSecondaryFracture = false;
		mDestruction.Initialize(dcfg);

		// Dam spans gx=150-362 → 10 segments
		const int numSegments = 10;
		const float damStartGX = 150.0f;
		const float damEndGX   = 362.0f;
		const float segWidthGX = (damEndGX - damStartGX) / (float)numSegments;
		const float segHalfW   = segWidthGX * gs / 2.0f + 3.0f;  // +3m overlap to seal gaps
		const float segHalfH   = 30.0f;                    // 60m tall wall
		const float segHalfT   = 20.0f;                    // 40m thick
		const float damY       = 206.5f;                    // grid Y center of dam
		const float wallBaseH  = 360.0f;                    // terrain base at dam center

		for (int i = 0; i < numSegments; ++i) {
			float segCenterGX = damStartGX + segWidthGX * (i + 0.5f);
			float worldX = wOX + segCenterGX * gs;
			float worldZ = wOZ + damY * gs;
			float worldY = wallBaseH + segHalfH;

			JPH::RefConst<JPH::Shape> wallShape = new JPH::BoxShape(
				JPH::Vec3(segHalfW, segHalfH, segHalfT));

			JPH::BodyCreationSettings bcs(wallShape,
				JPH::RVec3(worldX, worldY, worldZ),
				JPH::Quat::sIdentity(),
				JPH::EMotionType::Dynamic, Layers::MOVING);
			bcs.mRestitution    = 0.1f;
			bcs.mFriction       = 0.8f;
			bcs.mGravityFactor  = 0.0f;   // Hover in place until smashed
			bcs.mLinearDamping  = 0.3f;
			bcs.mAngularDamping = 0.3f;
			bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
			bcs.mMassPropertiesOverride.mMass = 80000.0f; // 80 tonnes per segment

			JPH::BodyID bodyId = mBodyInterface->CreateAndAddBody(bcs, JPH::EActivation::DontActivate);

			// Register with destruction system (low threshold — ball delivers huge impulse)
			uint32_t handle = mDestruction.AddDestructible(bodyId, 300.0f, 10);
			auto *dbody = mDestruction.GetDestructible(handle);
			if (dbody)
				dbody->pattern = WulfNet::DestructionSystem::GenerateBoxPattern(
					segHalfW, segHalfH, segHalfT, 10, 2400.0f);

			DamSegment seg;
			seg.bodyId         = bodyId;
			seg.destructHandle = handle;
			seg.halfX          = segHalfW;
			seg.halfY          = segHalfH;
			seg.halfZ          = segHalfT;
			seg.gxCenter       = segCenterGX;
			mDamSegments.push_back(seg);
		}
	}

	// ===================================================================
	// STATIC OBSTACLES — Village buildings, bridge, trees
	// ===================================================================
	auto terrainAt = [&](float gxf, float gyf) -> float {
		uint32_t xi = std::min((uint32_t)std::max(0.0f, gxf), cGridW - 1);
		uint32_t yi = std::min((uint32_t)std::max(0.0f, gyf), cGridH - 1);
		return state.terrainHeight[yi * cGridW + xi];
	};

	auto addStatic = [&](JPH::RefConst<JPH::Shape> shape, float gxf, float gyf, float yOff) {
		JPH::BodyCreationSettings bcs(shape,
			JPH::RVec3(wOX + gxf * gs, terrainAt(gxf, gyf) + yOff, wOZ + gyf * gs),
			JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::NON_MOVING);
		mBodyInterface->CreateAndAddBody(bcs, JPH::EActivation::DontActivate);
	};

	JPH::RefConst<JPH::Shape> bigBldg    = new JPH::BoxShape(JPH::Vec3(12.0f, 14.0f, 10.0f));
	JPH::RefConst<JPH::Shape> medBldg    = new JPH::BoxShape(JPH::Vec3(8.0f, 10.0f, 7.0f));
	JPH::RefConst<JPH::Shape> smallHouse = new JPH::BoxShape(JPH::Vec3(5.0f, 6.0f, 5.0f));
	JPH::RefConst<JPH::Shape> tallTower  = new JPH::BoxShape(JPH::Vec3(4.0f, 22.0f, 4.0f));
	JPH::RefConst<JPH::Shape> treeShape  = new JPH::CylinderShape(8.0f, 1.5f);
	JPH::RefConst<JPH::Shape> pillar     = new JPH::CylinderShape(10.0f, 3.0f);
	JPH::RefConst<JPH::Shape> wallSeg    = new JPH::BoxShape(JPH::Vec3(25.0f, 4.0f, 1.5f));

	// Left village
	addStatic(bigBldg,    180.0f, 280.0f, 14.0f);
	addStatic(medBldg,    200.0f, 270.0f, 10.0f);
	addStatic(smallHouse, 175.0f, 300.0f,  6.0f);
	addStatic(smallHouse, 195.0f, 310.0f,  6.0f);
	addStatic(smallHouse, 210.0f, 295.0f,  6.0f);
	addStatic(tallTower,  220.0f, 275.0f, 22.0f);

	// Right village
	addStatic(bigBldg,    300.0f, 275.0f, 14.0f);
	addStatic(medBldg,    320.0f, 285.0f, 10.0f);
	addStatic(smallHouse, 290.0f, 300.0f,  6.0f);
	addStatic(smallHouse, 310.0f, 310.0f,  6.0f);
	addStatic(smallHouse, 330.0f, 295.0f,  6.0f);
	addStatic(medBldg,    285.0f, 265.0f, 10.0f);

	// Bridge pillars
	addStatic(pillar, 240.0f, 300.0f, 10.0f);
	addStatic(pillar, 255.0f, 300.0f, 10.0f);
	addStatic(pillar, 270.0f, 300.0f, 10.0f);

	// Downstream settlement
	addStatic(bigBldg,    230.0f, 370.0f, 14.0f);
	addStatic(bigBldg,    280.0f, 380.0f, 14.0f);
	addStatic(medBldg,    200.0f, 390.0f, 10.0f);
	addStatic(medBldg,    310.0f, 400.0f, 10.0f);
	addStatic(smallHouse, 250.0f, 410.0f,  6.0f);
	addStatic(smallHouse, 270.0f, 395.0f,  6.0f);

	// Retaining walls
	addStatic(wallSeg, 200.0f, 340.0f, 4.0f);
	addStatic(wallSeg, 310.0f, 340.0f, 4.0f);

	// Scattered trees
	std::uniform_real_distribution<float> treeGX(140.0f, 370.0f);
	std::uniform_real_distribution<float> treeGY(230.0f, 440.0f);
	for (int i = 0; i < 30; ++i) {
		float tgx = treeGX(mRng);
		float tgy = treeGY(mRng);
		if (std::abs(tgx - centerX) < 35.0f) continue;
		addStatic(treeShape, tgx, tgy, 8.0f);
	}

	// ===================================================================
	// DYNAMIC DEBRIS — Swept away by the flood
	// ===================================================================
	JPH::RefConst<JPH::Shape> crateShape   = new JPH::BoxShape(JPH::Vec3(2.0f, 2.0f, 2.0f));
	JPH::RefConst<JPH::Shape> barrelShape  = new JPH::CylinderShape(1.5f, 1.0f);
	JPH::RefConst<JPH::Shape> boulderShape = new JPH::SphereShape(3.0f);
	JPH::RefConst<JPH::Shape> logShape     = new JPH::BoxShape(JPH::Vec3(0.6f, 0.6f, 5.0f));
	JPH::RefConst<JPH::Shape> bigBoulder   = new JPH::SphereShape(5.0f);

	std::uniform_real_distribution<float> dGX(170.0f, 340.0f);
	std::uniform_real_distribution<float> dGY(225.0f, 420.0f);

	auto addDynamic = [&](JPH::RefConst<JPH::Shape> shape, float mass, float gxf, float gyf, float yOff) {
		JPH::BodyCreationSettings bcs(shape,
			JPH::RVec3(wOX + gxf * gs, terrainAt(gxf, gyf) + yOff, wOZ + gyf * gs),
			JPH::Quat::sRandom(mRng), JPH::EMotionType::Dynamic, Layers::MOVING);
		bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateMassAndInertia;
		bcs.mMassPropertiesOverride.mMass = mass;
		JPH::Body *body = mBodyInterface->CreateBody(bcs);
		if (body) {
			mBodyInterface->AddBody(body->GetID(), JPH::EActivation::Activate);
			mFloatingBodies.push_back(body->GetID());
		}
	};

	for (int i = 0; i < 50; ++i) {
		float dgx = dGX(mRng), dgy = dGY(mRng);
		switch (i % 4) {
			case 0:  addDynamic(crateShape,   200.0f, dgx, dgy, 4.0f); break;
			case 1:  addDynamic(barrelShape,  100.0f, dgx, dgy, 4.0f); break;
			case 2:  addDynamic(boulderShape, 800.0f, dgx, dgy, 5.0f); break;
			default: addDynamic(logShape,     120.0f, dgx, dgy, 4.0f); break;
		}
	}

	for (int i = 0; i < 6; ++i)
		addDynamic(bigBoulder, 4000.0f, centerX - 30.0f + i * 12.0f, 225.0f, 6.0f);

	// Initialize system resource monitor
	WulfNet::SystemMonitor::Get().Initialize();
}

void DamBreakTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	if (!mWaterSystem) return;

	mTime += inParams.mDeltaTime;

	// Update performance stats
	mFrameCount++;
	mStatsTimer += inParams.mDeltaTime;
	if (mStatsTimer >= 0.5f) {
		mCurrentFPS   = (float)mFrameCount / mStatsTimer;
		mFrameTimeMs  = mStatsTimer / (float)mFrameCount * 1000.0f;
		mFrameCount   = 0;
		mStatsTimer   = 0.0f;
		WulfNet::SystemMonitor::Get().Update();
	}

	const auto &config = mWaterSystem->GetConfig();
	const float gs  = config.gridSize;
	const float wOX = config.originX;
	const float wOZ = config.originZ;

	// ---------------------------------------------------------------
	// AUTO-LAUNCH BALL — Fire a massive sphere at the dam center
	// ---------------------------------------------------------------
	if (!mProjectileLaunched && mTime >= mLaunchTime) {
		mProjectileLaunched = true;

		// Dam center in world space
		float damCenterX = wOX + 256.0f * gs;   // ~0
		float damCenterZ = wOZ + 206.0f * gs;   // ~-250
		float damCrestY  = 390.0f;

		// Launch from upstream (reservoir side, lower Z), arcing into the dam
		JPH::RVec3 launchPos(damCenterX, damCrestY + 80.0f, damCenterZ - 250.0f);
		JPH::Vec3  launchVel(0.0f, -15.0f, 70.0f);

		JPH::BodyCreationSettings bcs(
			new JPH::SphereShape(12.0f),   // 24m diameter wrecking ball
			launchPos,
			JPH::Quat::sIdentity(),
			JPH::EMotionType::Dynamic, Layers::MOVING);
		bcs.mRestitution    = 0.3f;
		bcs.mFriction       = 0.3f;
		bcs.mLinearVelocity = launchVel;
		bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
		bcs.mMassPropertiesOverride.mMass = 200000.0f; // 200 tonnes

		mProjectile = mBodyInterface->CreateAndAddBody(bcs, JPH::EActivation::Activate);
		mFloatingBodies.push_back(mProjectile);
	}

	// ---------------------------------------------------------------
	// FRACTURE DETECTION — Check if dam segments were hit
	// ---------------------------------------------------------------
	for (auto &seg : mDamSegments) {
		if (seg.broken) continue;

		auto *dbody = mDestruction.GetDestructible(seg.destructHandle);
		if (!dbody || dbody->fractured) continue;
		if (!mBodyInterface->IsAdded(seg.bodyId)) continue;
		if (!mBodyInterface->IsActive(seg.bodyId)) continue;

		// If the segment gained velocity, it was struck by the ball
		JPH::Vec3 vel = mBodyInterface->GetLinearVelocity(seg.bodyId);
		float speed = vel.Length();
		if (speed < 0.5f) continue;

		JPH::RVec3 bodyPos = mBodyInterface->GetCenterOfMassPosition(seg.bodyId);
		float impulse = speed * 80000.0f; // mass × speed

		bool fractured = mDestruction.EvaluateImpact(
			seg.destructHandle,
			(float)bodyPos.GetX(), (float)bodyPos.GetY(), (float)bodyPos.GetZ(),
			impulse);

		if (fractured) {
			// Remove intact wall segment
			mBodyInterface->RemoveBody(seg.bodyId);
			mBodyInterface->DestroyBody(seg.bodyId);
			seg.broken = true;

			// Create Voronoi fragment bodies
			for (size_t ci = 0; ci < dbody->pattern.cells.size(); ++ci) {
				const auto &cell = dbody->pattern.cells[ci];
				if (cell.mass < 1.0f) continue;

				float fhx = (cell.maxX - cell.minX) * 0.5f;
				float fhy = (cell.maxY - cell.minY) * 0.5f;
				float fhz = (cell.maxZ - cell.minZ) * 0.5f;
				if (fhx < 0.1f || fhy < 0.1f || fhz < 0.1f) continue;

				JPH::RVec3 fragPos = bodyPos + JPH::RVec3(cell.centerX, cell.centerY, cell.centerZ);

				JPH::BodyCreationSettings fragBCS(
					new JPH::BoxShape(JPH::Vec3(fhx, fhy, fhz)),
					fragPos, JPH::Quat::sIdentity(),
					JPH::EMotionType::Dynamic, Layers::MOVING);
				fragBCS.mRestitution = 0.2f;
				fragBCS.mFriction    = 0.5f;
				fragBCS.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
				fragBCS.mMassPropertiesOverride.mMass = cell.mass;

				// Eject fragments outward + add ball's velocity
				JPH::Vec3 ejectDir(cell.centerX, cell.centerY + 1.0f, cell.centerZ);
				float elen = ejectDir.Length();
				if (elen > 0.01f) ejectDir /= elen;
				fragBCS.mLinearVelocity  = vel * 0.3f + ejectDir * 8.0f;
				fragBCS.mAngularVelocity = JPH::Vec3(
					(float)((int)ci % 3) - 1.0f,
					(float)((int)ci % 5) - 2.0f,
					(float)((int)ci % 7) - 3.0f) * 4.0f;

				JPH::BodyID fragId = mBodyInterface->CreateAndAddBody(fragBCS, JPH::EActivation::Activate);
				mFloatingBodies.push_back(fragId);
			}

			// Lower terrain under this broken segment → water flows through
			auto &wstate = mWaterSystem->GetCPUState();
			float segHalfGX = seg.halfX / gs + 2.0f; // extra margin
			for (uint32_t gy = 193; gy < 221; ++gy)
				for (uint32_t gx = 0; gx < cGridW; ++gx) {
					float dist = std::abs((float)gx - seg.gxCenter);
					if (dist > segHalfGX) continue;
					float factor = std::max(0.0f, 1.0f - dist / segHalfGX);
					float drop = 150.0f * factor;
					uint32_t idx = gy * cGridW + gx;
					wstate.terrainHeight[idx] = std::max(wstate.terrainHeight[idx] - drop, 250.0f);
				}

			// Immediately fill breach cells with water up to reservoir surface.
			// Without this, the SWE solver propagates through dry cells at
			// millimeters per frame — far too slow for a realistic breach.
			const float resSurface = 385.0f;
			for (uint32_t gy = 185; gy < 230; ++gy)
				for (uint32_t gx = 0; gx < cGridW; ++gx) {
					float dist = std::abs((float)gx - seg.gxCenter);
					if (dist > segHalfGX) continue;
					uint32_t idx = gy * cGridW + gx;
					float tH = wstate.terrainHeight[idx];
					if (tH < resSurface) {
						float targetDepth = resSurface - tH;
						if (wstate.waterDepth[idx] < targetDepth)
							wstate.waterDepth[idx] = targetDepth;
					}
				}

			// Seed physically-motivated flux through the breach.
			// Torricelli outflow: v = sqrt(2*g*h) ≈ 51 m/s for 135m head.
			// Volume flux per pipe = v * dx ≈ 260 m³/s.
			for (uint32_t gy = 193; gy < 225; ++gy)
				for (uint32_t gx = 0; gx < cGridW; ++gx) {
					float dist = std::abs((float)gx - seg.gxCenter);
					if (dist > segHalfGX) continue;
					uint32_t idx = gy * cGridW + gx;
					float depth = wstate.waterDepth[idx];
					if (depth < 0.1f) continue;
					// Seed flux proportional to sqrt(2*g*depth) — real physics
					float v = std::sqrt(2.0f * 9.81f * depth);
					float fluxSeed = v * gs;  // volume flux = velocity × cell width
					float factor = std::max(0.0f, 1.0f - dist / segHalfGX);
					wstate.flux[idx].B += fluxSeed * factor;  // B = downstream (+Y)
				}
		}
	}

	// Step water simulation — single step per frame, matching WaterBox baseline.
	// The SWE solver's internal CFL subdivision handles stability.
	mWaterSystem->StepSimulationCPU(inParams.mDeltaTime);

	// Apply buoyancy forces
	auto &context = mWaterSystem->GetJobContextForTesting();
	context.interactingBodies = mFloatingBodies;
	mWaterSystem->ApplyBuoyancyForces(mJobSystem);

	// ---------------------------------------------------------------
	// RENDER TERRAIN
	// ---------------------------------------------------------------
	const auto &state = mWaterSystem->GetCPUState();
	const uint32_t W  = config.width;
	const uint32_t H  = config.height;
	const uint32_t S  = cRenderStep;

	{
		uint32_t trW = (W + S - 1) / S;
		uint32_t trH = (H + S - 1) / S;
		JPH::Vec3 tOff(config.originX, 0.0f, config.originZ);

		std::vector<JPH::DebugRenderer::Vertex> tVerts;
		std::vector<uint32> tIdx;
		tVerts.reserve(trW * trH);
		std::vector<int> tMap(trW * trH, -1);

		std::vector<float> tH(trW * trH);
		for (uint32_t ry = 0; ry < trH; ++ry)
			for (uint32_t rx = 0; rx < trW; ++rx) {
				uint32_t gx2 = std::min(rx * S, W - 1);
				uint32_t gy2 = std::min(ry * S, H - 1);
				tH[ry * trW + rx] = state.terrainHeight[gy2 * W + gx2];
			}

		std::vector<JPH::Vec3> tNorm(trW * trH, JPH::Vec3::sZero());
		for (uint32_t ry = 0; ry < trH - 1; ++ry)
			for (uint32_t rx = 0; rx < trW - 1; ++rx) {
				uint32_t i00 = ry * trW + rx, i10 = i00 + 1, i01 = i00 + trW, i11 = i01 + 1;
				float sgs = S * gs;
				JPH::Vec3 p00 = tOff + JPH::Vec3(rx * sgs, tH[i00], ry * sgs);
				JPH::Vec3 p10 = tOff + JPH::Vec3((rx+1)*sgs, tH[i10], ry * sgs);
				JPH::Vec3 p01 = tOff + JPH::Vec3(rx * sgs, tH[i01], (ry+1)*sgs);
				JPH::Vec3 p11 = tOff + JPH::Vec3((rx+1)*sgs, tH[i11], (ry+1)*sgs);
				JPH::Vec3 n1 = (p10 - p00).Cross(p01 - p00);
				tNorm[i00] += n1; tNorm[i10] += n1; tNorm[i01] += n1;
				JPH::Vec3 n2 = (p11 - p10).Cross(p01 - p10);
				tNorm[i10] += n2; tNorm[i11] += n2; tNorm[i01] += n2;
			}
		for (uint32_t i = 0; i < trW * trH; ++i) {
			float len = tNorm[i].Length();
			tNorm[i] = len > 1e-6f ? tNorm[i] / len : JPH::Vec3::sAxisY();
		}

		for (uint32_t ry = 0; ry < trH - 1; ++ry)
			for (uint32_t rx = 0; rx < trW - 1; ++rx) {
				uint32_t i00 = ry * trW + rx, i10 = i00 + 1, i01 = i00 + trW, i11 = i01 + 1;

				auto emitT = [&](uint32_t vrx, uint32_t vry, uint32_t vi) -> uint32 {
					if (tMap[vi] >= 0) return (uint32)tMap[vi];
					JPH::DebugRenderer::Vertex v;
					float sgs = S * gs;
					JPH::Vec3 pos = tOff + JPH::Vec3(vrx * sgs, tH[vi], vry * sgs);
					v.mPosition = { pos.GetX(), pos.GetY(), pos.GetZ() };
					v.mNormal   = { tNorm[vi].GetX(), tNorm[vi].GetY(), tNorm[vi].GetZ() };
					v.mUV       = { float(vrx) / float(trW), float(vry) / float(trH) };
					float h = tH[vi];
					float slope = 1.0f - std::max(0.0f, tNorm[vi].GetY());
					uint8_t cr, cg, cb;
					if (h < 200.0f) {
						float t2 = h / 200.0f;
						cr = (uint8_t)(40 + t2 * 40);  cg = (uint8_t)(80 + t2 * 30);  cb = (uint8_t)(25 + t2 * 15);
					} else if (h < 400.0f) {
						float t2 = (h - 200.0f) / 200.0f;
						cr = (uint8_t)(80 + t2 * 50);  cg = (uint8_t)(110 - t2 * 30); cb = (uint8_t)(40 + t2 * 20);
					} else {
						float t2 = std::min((h - 400.0f) / 200.0f, 1.0f);
						cr = (uint8_t)(130 + t2 * 60); cg = (uint8_t)(80 + t2 * 70);  cb = (uint8_t)(60 + t2 * 70);
					}
					float sf = 1.0f - slope * 0.4f;
					v.mColor = JPH::Color((uint8_t)(cr * sf), (uint8_t)(cg * sf), (uint8_t)(cb * sf), 255);
					uint32 idx2 = (uint32)tVerts.size();
					tVerts.push_back(v);
					tMap[vi] = (int)idx2;
					return idx2;
				};

				uint32 v00 = emitT(rx, ry, i00), v10 = emitT(rx+1, ry, i10);
				uint32 v01 = emitT(rx, ry+1, i01), v11 = emitT(rx+1, ry+1, i11);
				tIdx.push_back(v00); tIdx.push_back(v10); tIdx.push_back(v01);
				tIdx.push_back(v10); tIdx.push_back(v11); tIdx.push_back(v01);
			}

		if (!tVerts.empty()) {
			JPH::DebugRenderer::Batch batch = mDebugRenderer->CreateTriangleBatch(
				tVerts.data(), (int)tVerts.size(), tIdx.data(), (int)tIdx.size());
			JPH::AABox bounds;
			for (auto &v : tVerts)
				bounds.Encapsulate(JPH::Vec3(v.mPosition.x, v.mPosition.y, v.mPosition.z));
			JPH::DebugRenderer::GeometryRef geom = new JPH::DebugRenderer::Geometry(batch, bounds);
			mDebugRenderer->DrawGeometry(JPH::RMat44::sIdentity(), JPH::Color(255, 255, 255, 255), geom,
				JPH::DebugRenderer::ECullMode::Off, JPH::DebugRenderer::ECastShadow::On, JPH::DebugRenderer::EDrawMode::Solid);
		}
	}

	// ---------------------------------------------------------------
	// RENDER WATER SURFACE
	// ---------------------------------------------------------------
	uint32_t rW = (W + S - 1) / S;
	uint32_t rH = (H + S - 1) / S;

	JPH::Vec3 offset(config.originX, 0.0f, config.originZ);

	std::vector<float> surfaceH(rW * rH);
	std::vector<float> terrainH(rW * rH);
	std::vector<bool>  wet(rW * rH, false);

	for (uint32_t ry = 0; ry < rH; ++ry)
		for (uint32_t rx = 0; rx < rW; ++rx) {
			uint32_t gx2 = std::min(rx * S, W - 1);
			uint32_t gy2 = std::min(ry * S, H - 1);
			uint32_t gi = gy2 * W + gx2;
			uint32_t ri = ry * rW + rx;
			terrainH[ri] = state.terrainHeight[gi];
			surfaceH[ri] = terrainH[ri] + state.waterDepth[gi];
			if (state.waterDepth[gi] > 0.01f) wet[ri] = true;
		}

	// Expand wet mask by 3 cells
	std::vector<bool> wetExp = wet;
	for (int expand = 0; expand < 3; ++expand) {
		std::vector<bool> prev = wetExp;
		for (uint32_t ry = 0; ry < rH; ++ry)
			for (uint32_t rx = 0; rx < rW; ++rx)
				if (prev[ry * rW + rx]) {
					if (rx > 0)    wetExp[ry * rW + (rx - 1)] = true;
					if (rx < rW-1) wetExp[ry * rW + (rx + 1)] = true;
					if (ry > 0)    wetExp[(ry-1) * rW + rx]   = true;
					if (ry < rH-1) wetExp[(ry+1) * rW + rx]   = true;
				}
	}

	// Smooth surface heights (3 passes, matching V3 test)
	for (int pass = 0; pass < 3; ++pass) {
		std::vector<float> tmp = surfaceH;
		for (uint32_t ry = 1; ry < rH - 1; ++ry)
			for (uint32_t rx = 1; rx < rW - 1; ++rx) {
				uint32_t ri = ry * rW + rx;
				if (!wetExp[ri]) continue;
				float sum = 0.0f, wt = 0.0f;
				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx) {
						uint32_t ni = (ry + dy) * rW + (rx + dx);
						if (wetExp[ni]) {
							float w = (dx == 0 && dy == 0) ? 4.0f : 1.0f;
							sum += tmp[ni] * w;
							wt  += w;
						}
					}
				if (wt > 0.0f) surfaceH[ri] = sum / wt;
			}
	}

	// Per-vertex normals
	std::vector<JPH::Vec3> normals(rW * rH, JPH::Vec3::sZero());
	for (uint32_t ry = 0; ry < rH - 1; ++ry)
		for (uint32_t rx = 0; rx < rW - 1; ++rx) {
			uint32_t i00 = ry * rW + rx, i10 = i00 + 1;
			uint32_t i01 = i00 + rW,     i11 = i01 + 1;
			if (!wetExp[i00] && !wetExp[i10] && !wetExp[i01] && !wetExp[i11]) continue;

			float sgs = S * gs;
			JPH::Vec3 p00 = offset + JPH::Vec3(rx * sgs,       surfaceH[i00], ry * sgs);
			JPH::Vec3 p10 = offset + JPH::Vec3((rx+1) * sgs,   surfaceH[i10], ry * sgs);
			JPH::Vec3 p01 = offset + JPH::Vec3(rx * sgs,       surfaceH[i01], (ry+1) * sgs);
			JPH::Vec3 p11 = offset + JPH::Vec3((rx+1) * sgs,   surfaceH[i11], (ry+1) * sgs);

			JPH::Vec3 n1 = (p10 - p00).Cross(p01 - p00);
			normals[i00] += n1; normals[i10] += n1; normals[i01] += n1;
			JPH::Vec3 n2 = (p11 - p10).Cross(p01 - p10);
			normals[i10] += n2; normals[i11] += n2; normals[i01] += n2;
		}
	for (uint32_t i = 0; i < rW * rH; ++i) {
		float len = normals[i].Length();
		normals[i] = len > 1e-6f ? normals[i] / len : JPH::Vec3::sAxisY();
	}

	// Build indexed water surface mesh
	std::vector<JPH::DebugRenderer::Vertex> vertices;
	std::vector<uint32> indices;
	vertices.reserve(rW * rH / 2);
	std::vector<int> vtxMap(rW * rH, -1);

	for (uint32_t ry = 0; ry < rH - 1; ++ry)
		for (uint32_t rx = 0; rx < rW - 1; ++rx) {
			uint32_t i00 = ry * rW + rx, i10 = i00 + 1;
			uint32_t i01 = i00 + rW,     i11 = i01 + 1;
			if (!wetExp[i00] && !wetExp[i10] && !wetExp[i01] && !wetExp[i11]) continue;

			auto emitVtx = [&](uint32_t vrx, uint32_t vry, uint32_t vi) -> uint32 {
				if (vtxMap[vi] >= 0) return (uint32)vtxMap[vi];

				uint32_t gx2 = std::min(vrx * S, W - 1);
				uint32_t gy2 = std::min(vry * S, H - 1);
				uint32_t gi = gy2 * W + gx2;

				JPH::DebugRenderer::Vertex v;
				float sgs = S * gs;
				JPH::Vec3 pos = offset + JPH::Vec3(vrx * sgs, surfaceH[vi], vry * sgs);
				v.mPosition = { pos.GetX(), pos.GetY(), pos.GetZ() };
				v.mNormal   = { normals[vi].GetX(), normals[vi].GetY(), normals[vi].GetZ() };
				v.mUV       = { float(vrx) / float(rW), float(vry) / float(rH) };

				// Depth-based water color — WaterBox baseline scaled for dam depths (~0-80m)
				float depth = state.waterDepth[gi];
				float t = std::min(depth / 40.0f, 1.0f);
				uint8_t cr = (uint8_t)(100 - t * 90);
				uint8_t cg = (uint8_t)(200 - t * 160);
				uint8_t cb = (uint8_t)(220 - t * 120);
				uint8_t ca = (uint8_t)(230 + t * 25);
				v.mColor = JPH::Color(cr, cg, cb, ca);

				uint32 idx2 = (uint32)vertices.size();
				vertices.push_back(v);
				vtxMap[vi] = (int)idx2;
				return idx2;
			};

			uint32 v00 = emitVtx(rx, ry, i00),   v10 = emitVtx(rx+1, ry, i10);
			uint32 v01 = emitVtx(rx, ry+1, i01), v11 = emitVtx(rx+1, ry+1, i11);
			indices.push_back(v00); indices.push_back(v10); indices.push_back(v01);
			indices.push_back(v10); indices.push_back(v11); indices.push_back(v01);
		}

	// Bottom mesh: terrain-following underside
	for (uint32_t ry = 0; ry < rH - 1; ++ry)
		for (uint32_t rx = 0; rx < rW - 1; ++rx) {
			uint32_t i00 = ry * rW + rx, i10 = i00 + 1;
			uint32_t i01 = i00 + rW,     i11 = i01 + 1;
			if (!wetExp[i00] && !wetExp[i10] && !wetExp[i01] && !wetExp[i11]) continue;

			auto emitBtm = [&](uint32_t vrx, uint32_t vry, uint32_t vi) -> uint32 {
				JPH::DebugRenderer::Vertex v;
				float sgs = S * gs;
				JPH::Vec3 pos = offset + JPH::Vec3(vrx * sgs, terrainH[vi], vry * sgs);
				v.mPosition = { pos.GetX(), pos.GetY(), pos.GetZ() };
				v.mNormal   = { 0.0f, -1.0f, 0.0f };
				v.mUV       = { 0.0f, 0.0f };
				v.mColor    = JPH::Color(8, 20, 55, 255);
				uint32 idx2 = (uint32)vertices.size();
				vertices.push_back(v);
				return idx2;
			};

			uint32 b00 = emitBtm(rx, ry, i00), b10 = emitBtm(rx+1, ry, i10);
			uint32 b01 = emitBtm(rx, ry+1, i01), b11 = emitBtm(rx+1, ry+1, i11);
			indices.push_back(b00); indices.push_back(b01); indices.push_back(b10);
			indices.push_back(b10); indices.push_back(b01); indices.push_back(b11);
		}

	// Edge skirts
	{
		uint32_t qW = rW - 1, qH = rH - 1;
		std::vector<bool> qR(qW * qH, false);
		for (uint32_t qy = 0; qy < qH; ++qy)
			for (uint32_t qx = 0; qx < qW; ++qx) {
				uint32_t i = qy * rW + qx;
				if (wetExp[i] || wetExp[i+1] || wetExp[i+rW] || wetExp[i+rW+1])
					qR[qy * qW + qx] = true;
			}

		auto skirtVtx = [&](float wx, float wy, float wz) -> uint32 {
			JPH::DebugRenderer::Vertex v;
			v.mPosition = { wx, wy, wz };
			v.mNormal   = { 0.0f, -1.0f, 0.0f };
			v.mUV       = { 0.0f, 0.0f };
			v.mColor    = JPH::Color(15, 40, 100, 230);
			uint32 vi2  = (uint32)vertices.size();
			vertices.push_back(v);
			return vi2;
		};

		auto skirtQuad = [&](float ax, float atop, float abot, float az,
		                      float bx, float btop, float bbot, float bz) {
			uint32 t0 = skirtVtx(ax, atop, az), b0 = skirtVtx(ax, abot, az);
			uint32 t1 = skirtVtx(bx, btop, bz), b1 = skirtVtx(bx, bbot, bz);
			indices.push_back(t0); indices.push_back(t1); indices.push_back(b0);
			indices.push_back(t1); indices.push_back(b1); indices.push_back(b0);
		};

		float oX = offset.GetX(), oZ = offset.GetZ();
		float sgs = S * gs;
		for (uint32_t qy = 0; qy < qH; ++qy)
			for (uint32_t qx = 0; qx < qW; ++qx) {
				if (!qR[qy * qW + qx]) continue;
				uint32_t i00 = qy*rW + qx, i10 = i00+1, i01 = i00+rW, i11 = i01+1;
				float x0 = oX + qx * sgs, x1 = oX + (qx+1) * sgs;
				float z0 = oZ + qy * sgs, z1 = oZ + (qy+1) * sgs;
				if (qx == 0 || !qR[qy*qW + qx-1])
					skirtQuad(x0, surfaceH[i00], terrainH[i00], z0,
					          x0, surfaceH[i01], terrainH[i01], z1);
				if (qx+1 >= qW || !qR[qy*qW + qx+1])
					skirtQuad(x1, surfaceH[i10], terrainH[i10], z0,
					          x1, surfaceH[i11], terrainH[i11], z1);
				if (qy == 0 || !qR[(qy-1)*qW + qx])
					skirtQuad(x0, surfaceH[i00], terrainH[i00], z0,
					          x1, surfaceH[i10], terrainH[i10], z0);
				if (qy+1 >= qH || !qR[(qy+1)*qW + qx])
					skirtQuad(x0, surfaceH[i01], terrainH[i01], z1,
					          x1, surfaceH[i11], terrainH[i11], z1);
			}
	}

	// Submit water to debug renderer
	if (!vertices.empty() && !indices.empty()) {
		JPH::DebugRenderer::Batch batch = mDebugRenderer->CreateTriangleBatch(
			vertices.data(), (int)vertices.size(),
			indices.data(), (int)indices.size());

		JPH::AABox bounds;
		for (auto &v : vertices)
			bounds.Encapsulate(JPH::Vec3(v.mPosition.x, v.mPosition.y, v.mPosition.z));

		JPH::DebugRenderer::GeometryRef geom = new JPH::DebugRenderer::Geometry(batch, bounds);
		mDebugRenderer->DrawGeometry(
			JPH::RMat44::sIdentity(),
			JPH::Color(255, 255, 255, 255),
			geom,
			JPH::DebugRenderer::ECullMode::Off,
			JPH::DebugRenderer::ECastShadow::Off,
			JPH::DebugRenderer::EDrawMode::Solid);
	}
}

void DamBreakTest::GetInitialCamera(CameraState &ioState) const
{
	// View from downstream looking up at the dam — sees the ball impact and flood
	ioState.mPos = JPH::RVec3(600.0f, 500.0f, 200.0f);
	JPH::Vec3 target(0.0f, 350.0f, -250.0f);
	ioState.mForward = (target - JPH::Vec3(ioState.mPos)).Normalized();
}

String DamBreakTest::GetStatusString() const
{
	const WulfNet::SystemStats &sys = WulfNet::SystemMonitor::Get().GetStats();

	std::ostringstream oss;
	oss << std::fixed;

	oss << "FPS: " << std::setprecision(1) << mCurrentFPS
	    << "  (" << std::setprecision(2) << mFrameTimeMs << " ms)\n";

	oss << std::setprecision(1);
	oss << "CPU: " << sys.cpuUsagePercent << "%  "
	    << "RAM: " << WulfNet::FormatBytes(sys.processMemoryBytes)
	    << " / " << WulfNet::FormatBytes(sys.ramTotalBytes)
	    << " (" << sys.ramUsagePercent << "%)\n";

	if (sys.gpuUsageAvailable) {
		oss << "GPU: " << sys.gpuUsagePercent << "%";
		if (!sys.gpuName.empty())
			oss << " (" << sys.gpuName << ")";
		oss << "\n";
	}

	if (sys.vramUsageAvailable) {
		oss << "VRAM: " << WulfNet::FormatBytes(sys.vramUsedBytes)
		    << " / " << WulfNet::FormatBytes(sys.vramTotalBytes)
		    << " (" << sys.vramUsagePercent << "%)\n";
	}

	// Simulation info
	oss << "\n";
	int brokenCount = 0;
	for (const auto &seg : mDamSegments)
		if (seg.broken) brokenCount++;
	oss << "Dam: " << brokenCount << "/" << mDamSegments.size() << " segments broken\n";
	oss << "Bodies: " << mFloatingBodies.size() << "\n";
	oss << "Time: " << std::setprecision(1) << mTime << "s\n";

	// Water diagnostics — depth at reservoir, dam gap, and downstream
	if (mWaterSystem) {
		const auto &st = mWaterSystem->GetCPUState();
		uint32_t mid = cGridW / 2;
		float dRes = st.waterDepth[100 * cGridW + mid];    // upstream reservoir
		float dGap = st.waterDepth[206 * cGridW + mid];    // dam center
		float dDown = st.waterDepth[300 * cGridW + mid];   // downstream valley
		oss << std::setprecision(1);
		oss << "Depth [res/gap/down]: " << dRes << " / " << dGap << " / " << dDown << " m\n";
	}

	return String(oss.str().c_str());
}
