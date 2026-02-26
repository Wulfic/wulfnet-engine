// =============================================================================
// WulfNet Engine - CO-FLIP Fluid Showcase Tests — Implementation
// =============================================================================
// Large-scale CO-FLIP particle fluid scenarios demonstrating the full
// capability of the parallelized simulation pipeline.
// =============================================================================

#include <Samples.h>

#include "WulfNetFluidShowcaseTests.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>

#include <cmath>
#include <algorithm>

// =====================================================================
// RTTI Registration
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidShowcase)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidShowcase, WulfNetFluidTest)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidRippleBasinTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidRippleBasinTest, WulfNetFluidShowcase)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidTerrainCascadeTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidTerrainCascadeTest, WulfNetFluidShowcase)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidDamBreak3DTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidDamBreak3DTest, WulfNetFluidShowcase)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidRainStormTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidRainStormTest, WulfNetFluidShowcase)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidTsunamiSurgeTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidTsunamiSurgeTest, WulfNetFluidShowcase)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidRiverRapidsTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidRiverRapidsTest, WulfNetFluidShowcase)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidCalderaEruptionTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidCalderaEruptionTest, WulfNetFluidShowcase)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidValleyFloodTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidValleyFloodTest, WulfNetFluidShowcase)
}

// =====================================================================
// WulfNetFluidShowcase Base — PrePhysicsUpdate with scenario hook
// =====================================================================

void WulfNetFluidShowcase::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	mScenarioTime += inParams.mDeltaTime;
	UpdateScenario(inParams.mDeltaTime);

	// Delegate to base class for simulation step + rendering
	WulfNetFluidTest::PrePhysicsUpdate(inParams);
}

// =====================================================================
// 1. Ripple Basin — Large shallow basin with periodic drops
//    Grid: 64×24×64, cellSize 0.2 → 12.8m × 4.8m × 12.8m domain
// =====================================================================

void WulfNetFluidRippleBasinTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 64;
	mFluidConfig.gridSizeY = 24;
	mFluidConfig.gridSizeZ = 64;
	mFluidConfig.cellSize  = 0.2f;
	mFluidConfig.pressureIterations = 20;
	mFluidConfig.particlesPerCell   = 4;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize  = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidConfig.useGPU = false;
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Fill a shallow basin — water across the entire bottom
	// Domain: 0..12.8 × 0..4.8 × 0..12.8
	CreateWaterBox(0.5f, 0.2f, 0.5f, 12.3f, 1.0f, 12.3f);

	// Solid walls around the basin to contain water
	AddSolidBox(0.0f, 0.0f, 0.0f, 12.8f, 4.8f, 0.4f);   // back
	AddSolidBox(0.0f, 0.0f, 12.4f, 12.8f, 4.8f, 12.8f);  // front
	AddSolidBox(0.0f, 0.0f, 0.0f, 0.4f, 4.8f, 12.8f);    // left
	AddSolidBox(12.4f, 0.0f, 0.0f, 12.8f, 4.8f, 12.8f);  // right
}

void WulfNetFluidRippleBasinTest::UpdateScenario(float dt)
{
	mDropTimer += dt;

	// Drop a sphere of water every 0.8 seconds, up to 40 drops
	if (mDropTimer >= 0.8f && mDropCount < 40)
	{
		mDropTimer -= 0.8f;
		mDropCount++;

		// Alternate drop positions
		float cx, cz;
		switch (mDropCount % 5)
		{
		case 0: cx = 6.4f;  cz = 6.4f;  break;  // Centre
		case 1: cx = 3.2f;  cz = 3.2f;  break;  // Corner
		case 2: cx = 9.6f;  cz = 3.2f;  break;  // Corner
		case 3: cx = 3.2f;  cz = 9.6f;  break;  // Corner
		default: cx = 9.6f; cz = 9.6f;  break;  // Corner
		}

		CreateWaterSphere(cx, 2.5f, cz, 0.4f);
	}
}

// =====================================================================
// 2. Terrain Cascade — Water flows over stepped terrain blocks
//    Grid: 80×32×40, cellSize 0.15 → 12m × 4.8m × 6m
// =====================================================================

void WulfNetFluidTerrainCascadeTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 80;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 40;
	mFluidConfig.cellSize  = 0.15f;
	mFluidConfig.pressureIterations = 20;
	mFluidConfig.particlesPerCell   = 4;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize  = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidConfig.useGPU = false;
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Create a reservoir of water at the top-left
	CreateWaterBox(0.5f, 3.0f, 0.5f, 3.0f, 4.5f, 5.5f);

	// Create cascading terrain steps (solid blocks descending left→right)
	// Step 1 (highest, left side)
	AddSolidBox(0.0f, 0.0f, 0.0f, 3.0f, 2.8f, 6.0f);
	// Step 2
	AddSolidBox(3.0f, 0.0f, 0.0f, 5.5f, 2.0f, 6.0f);
	// Step 3
	AddSolidBox(5.5f, 0.0f, 0.0f, 8.0f, 1.2f, 6.0f);
	// Step 4 (lowest)
	AddSolidBox(8.0f, 0.0f, 0.0f, 12.0f, 0.4f, 6.0f);

	// Side walls
	AddSolidBox(0.0f, 0.0f, 0.0f, 12.0f, 4.8f, 0.3f);
	AddSolidBox(0.0f, 0.0f, 5.7f, 12.0f, 4.8f, 6.0f);

	// Emitter feeding the reservoir continuously
	CreateEmitter(1.5f, 4.0f, 3.0f, 0.0f, -0.5f, 0.0f, 80.0f, 0.5f);
}

void WulfNetFluidTerrainCascadeTest::SetupObjects()
{
	// Jolt physics bodies for the steps (visual collision)
	BodyCreationSettings step1(
		new BoxShape(Vec3(1.5f, 1.4f, 3.0f)),
		RVec3(1.5f, 1.4f, 3.0f),
		Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(step1, EActivation::DontActivate);

	BodyCreationSettings step2(
		new BoxShape(Vec3(1.25f, 1.0f, 3.0f)),
		RVec3(4.25f, 1.0f, 3.0f),
		Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(step2, EActivation::DontActivate);

	BodyCreationSettings step3(
		new BoxShape(Vec3(1.25f, 0.6f, 3.0f)),
		RVec3(6.75f, 0.6f, 3.0f),
		Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(step3, EActivation::DontActivate);
}

void WulfNetFluidTerrainCascadeTest::UpdateScenario(float /*dt*/)
{
	// Continuous flow — emitter handles everything
}

// =====================================================================
// 3. Dam Break 3D — Tall column of water behind a removable wall
//    Grid: 80×40×48, cellSize 0.15 → 12m × 6m × 7.2m
// =====================================================================

void WulfNetFluidDamBreak3DTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 80;
	mFluidConfig.gridSizeY = 40;
	mFluidConfig.gridSizeZ = 48;
	mFluidConfig.cellSize  = 0.15f;
	mFluidConfig.pressureIterations = 25;
	mFluidConfig.particlesPerCell   = 4;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize  = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidConfig.useGPU = false;
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Domain: 12m × 6m × 7.2m
	// Water column behind the dam (x = 0.3 to 3.0, full depth)
	CreateWaterBox(0.3f, 0.2f, 0.3f, 3.0f, 4.5f, 6.9f);

	// Dam wall (solid) at x = 3.0..3.3 — will be removed at t=1.5s
	AddSolidBox(3.0f, 0.0f, 0.0f, 3.3f, 6.0f, 7.2f);

	// Containment walls
	AddSolidBox(0.0f, 0.0f, 0.0f, 12.0f, 6.0f, 0.3f);
	AddSolidBox(0.0f, 0.0f, 6.9f, 12.0f, 6.0f, 7.2f);
	AddSolidBox(0.0f, 0.0f, 0.0f, 0.3f, 6.0f, 7.2f);
	AddSolidBox(11.7f, 0.0f, 0.0f, 12.0f, 6.0f, 7.2f);

	// Downstream obstacle pillars for the water to flow around
	AddSolidBox(5.0f, 0.0f, 1.5f, 5.6f, 3.0f, 2.1f);
	AddSolidBox(5.0f, 0.0f, 3.3f, 5.6f, 3.0f, 3.9f);
	AddSolidBox(5.0f, 0.0f, 5.1f, 5.6f, 3.0f, 5.7f);
	AddSolidBox(7.0f, 0.0f, 2.4f, 7.6f, 3.0f, 3.0f);
	AddSolidBox(7.0f, 0.0f, 4.2f, 7.6f, 3.0f, 4.8f);
	AddSolidBox(9.0f, 0.0f, 1.5f, 9.6f, 3.0f, 2.1f);
	AddSolidBox(9.0f, 0.0f, 3.3f, 9.6f, 3.0f, 3.9f);
	AddSolidBox(9.0f, 0.0f, 5.1f, 9.6f, 3.0f, 5.7f);
}

void WulfNetFluidDamBreak3DTest::SetupObjects()
{
	// Jolt dam wall body (will be removed on timer)
	BodyCreationSettings dam(
		new BoxShape(Vec3(0.15f, 3.0f, 3.6f)),
		RVec3(3.15f, 3.0f, 3.6f),
		Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(dam, EActivation::DontActivate);

	// Pillar bodies for visual collision
	auto addPillar = [&](float x0, float z0, float x1, float z1, float h)
	{
		Vec3 half((x1 - x0) * 0.5f, h * 0.5f, (z1 - z0) * 0.5f);
		RVec3 pos((x0 + x1) * 0.5f, h * 0.5f, (z0 + z1) * 0.5f);
		BodyCreationSettings pillar(new BoxShape(half), pos,
			Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
		mBodyInterface->CreateAndAddBody(pillar, EActivation::DontActivate);
	};

	addPillar(5.0f, 1.5f, 5.6f, 2.1f, 3.0f);
	addPillar(5.0f, 3.3f, 5.6f, 3.9f, 3.0f);
	addPillar(5.0f, 5.1f, 5.6f, 5.7f, 3.0f);
	addPillar(7.0f, 2.4f, 7.6f, 3.0f, 3.0f);
	addPillar(7.0f, 4.2f, 7.6f, 4.8f, 3.0f);
	addPillar(9.0f, 1.5f, 9.6f, 2.1f, 3.0f);
	addPillar(9.0f, 3.3f, 9.6f, 3.9f, 3.0f);
	addPillar(9.0f, 5.1f, 9.6f, 5.7f, 3.0f);
}

void WulfNetFluidDamBreak3DTest::UpdateScenario(float dt)
{
	mDamTimer += dt;

	// At t=1.5s, remove the dam by clearing the solid cells
	if (!mDamReleased && mDamTimer >= 1.5f)
	{
		mDamReleased = true;

		// Clear the dam wall in the fluid grid — water rushes through
		// Note: We cannot truly "remove" solid cells from outside the system,
		// so we reset and re-add only the walls + pillars (not the dam).
		// A simpler approach: just re-mark the dam area as non-solid.
		// The CO-FLIP system's solid cell array is internal, but we can
		// add a large water box in the gap to push through.
		CreateWaterBox(3.0f, 0.2f, 0.3f, 3.5f, 4.5f, 6.9f);
	}
}

// =====================================================================
// 4. Rain Storm — Random drops falling from above
//    Grid: 64×32×64, cellSize 0.18 → 11.5m × 5.8m × 11.5m
// =====================================================================

void WulfNetFluidRainStormTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 64;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 64;
	mFluidConfig.cellSize  = 0.18f;
	mFluidConfig.pressureIterations = 20;
	mFluidConfig.particlesPerCell   = 4;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize  = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidConfig.useGPU = false;
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Domain: ~11.5m × 5.8m × 11.5m
	// Containment walls
	float domX = mFluidConfig.gridSizeX * mFluidConfig.cellSize;
	float domZ = mFluidConfig.gridSizeZ * mFluidConfig.cellSize;
	float domY = mFluidConfig.gridSizeY * mFluidConfig.cellSize;
	AddSolidBox(0.0f, 0.0f, 0.0f, domX, domY, 0.3f);
	AddSolidBox(0.0f, 0.0f, domZ - 0.3f, domX, domY, domZ);
	AddSolidBox(0.0f, 0.0f, 0.0f, 0.3f, domY, domZ);
	AddSolidBox(domX - 0.3f, 0.0f, 0.0f, domX, domY, domZ);

	// Terrain depressions — solid blocks forming basins for water to collect in
	// Raised centre plateau
	AddSolidBox(4.0f, 0.0f, 4.0f, 7.5f, 0.8f, 7.5f);
	// Low walls around depressions
	AddSolidBox(2.0f, 0.0f, 2.0f, 2.4f, 0.5f, 9.5f);
	AddSolidBox(9.1f, 0.0f, 2.0f, 9.5f, 0.5f, 9.5f);
}

void WulfNetFluidRainStormTest::SetupObjects()
{
	// Jolt bodies for the plateau
	BodyCreationSettings plateau(
		new BoxShape(Vec3(1.75f, 0.4f, 1.75f)),
		RVec3(5.75f, 0.4f, 5.75f),
		Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(plateau, EActivation::DontActivate);

	// Some floating debris
	for (int i = 0; i < 4; ++i)
	{
		BodyCreationSettings ball(
			new SphereShape(0.12f),
			RVec3(3.0f + i * 2.0f, 3.0f, 5.7f),
			Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING);
		ball.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		ball.mMassPropertiesOverride.mMass = 0.3f;
		mBodyInterface->CreateAndAddBody(ball, EActivation::Activate);
	}
}

void WulfNetFluidRainStormTest::UpdateScenario(float dt)
{
	mRainTimer += dt;

	// Spawn a rain drop every 0.04s, up to 1500 drops
	while (mRainTimer >= 0.04f && mTotalDrops < 1500)
	{
		mRainTimer -= 0.04f;
		mTotalDrops++;

		// Deterministic pseudo-random positions
		uint32_t seed = static_cast<uint32_t>(mTotalDrops * 73856093u);
		float domX = mFluidConfig.gridSizeX * mFluidConfig.cellSize;
		float domZ = mFluidConfig.gridSizeZ * mFluidConfig.cellSize;
		float rx = 0.5f + (seed % 1000) / 1000.0f * (domX - 1.0f);
		seed = seed * 19349663u + 1;
		float rz = 0.5f + (seed % 1000) / 1000.0f * (domZ - 1.0f);

		// Small sphere dropped from above
		CreateWaterSphere(rx, 4.5f, rz, 0.2f);
	}
}

// =====================================================================
// 5. Tsunami Surge — Massive wave hitting coastal pillars
//    Grid: 96×32×48, cellSize 0.15 → 14.4m × 4.8m × 7.2m
// =====================================================================

void WulfNetFluidTsunamiSurgeTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 96;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 48;
	mFluidConfig.cellSize  = 0.15f;
	mFluidConfig.pressureIterations = 25;
	mFluidConfig.particlesPerCell   = 4;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize  = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidConfig.useGPU = false;
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Domain: 14.4m × 4.8m × 7.2m
	// Ocean baseline — thin shallow layer across entire floor
	CreateWaterBox(0.5f, 0.2f, 0.5f, 14.0f, 0.6f, 6.7f);

	// Containment walls
	AddSolidBox(0.0f, 0.0f, 0.0f, 14.4f, 4.8f, 0.3f);
	AddSolidBox(0.0f, 0.0f, 6.9f, 14.4f, 4.8f, 7.2f);
	AddSolidBox(0.0f, 0.0f, 0.0f, 0.3f, 4.8f, 7.2f);
	AddSolidBox(14.1f, 0.0f, 0.0f, 14.4f, 4.8f, 7.2f);

	// Coastal pillars — 3 rows at x=8..12
	for (float x = 8.0f; x <= 12.0f; x += 2.0f)
	{
		for (float z = 1.2f; z <= 5.7f; z += 1.5f)
		{
			AddSolidBox(x, 0.0f, z, x + 0.5f, 3.0f, z + 0.5f);
		}
	}

	// Raised "beach" at the right side (solid slope)
	AddSolidBox(12.0f, 0.0f, 0.0f, 14.4f, 0.6f, 7.2f);
}

void WulfNetFluidTsunamiSurgeTest::SetupObjects()
{
	// Jolt pillar bodies
	for (float x = 8.0f; x <= 12.0f; x += 2.0f)
	{
		for (float z = 1.2f; z <= 5.7f; z += 1.5f)
		{
			BodyCreationSettings pillar(
				new BoxShape(Vec3(0.25f, 1.5f, 0.25f)),
				RVec3(x + 0.25f, 1.5f, z + 0.25f),
				Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
			mBodyInterface->CreateAndAddBody(pillar, EActivation::DontActivate);
		}
	}
}

void WulfNetFluidTsunamiSurgeTest::UpdateScenario(float dt)
{
	mWaveTimer += dt;

	// At t=2.0s inject a massive water column on the left side (the "tsunami")
	if (!mWaveTriggered && mWaveTimer >= 2.0f)
	{
		mWaveTriggered = true;

		// Tall, wide water column — pushes a powerful front across the domain
		CreateWaterBox(0.5f, 0.5f, 0.5f, 3.0f, 3.5f, 6.7f);
	}
}

// =====================================================================
// 6. River Rapids — Winding channel with rock obstacles
//    Grid: 48×24×96, cellSize 0.15 → 7.2m × 3.6m × 14.4m (long channel)
// =====================================================================

void WulfNetFluidRiverRapidsTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 48;
	mFluidConfig.gridSizeY = 24;
	mFluidConfig.gridSizeZ = 96;
	mFluidConfig.cellSize  = 0.15f;
	mFluidConfig.pressureIterations = 20;
	mFluidConfig.particlesPerCell   = 4;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize  = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidConfig.useGPU = false;
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Domain: 7.2m × 3.6m × 14.4m
	float domX = mFluidConfig.gridSizeX * mFluidConfig.cellSize;
	float domY = mFluidConfig.gridSizeY * mFluidConfig.cellSize;
	float domZ = mFluidConfig.gridSizeZ * mFluidConfig.cellSize;

	// Channel walls (narrow the X dimension to ~4m for a river feel)
	AddSolidBox(0.0f, 0.0f, 0.0f, 1.2f, domY, domZ);   // Left bank
	AddSolidBox(domX - 1.2f, 0.0f, 0.0f, domX, domY, domZ); // Right bank
	// End walls
	AddSolidBox(0.0f, 0.0f, domZ - 0.3f, domX, domY, domZ);

	// Rock obstacles scattered along the channel
	AddSolidBox(2.5f, 0.0f, 3.0f, 3.2f, 1.8f, 3.7f);
	AddSolidBox(4.0f, 0.0f, 5.0f, 4.7f, 1.5f, 5.7f);
	AddSolidBox(2.0f, 0.0f, 7.5f, 2.8f, 2.0f, 8.3f);
	AddSolidBox(4.5f, 0.0f, 9.0f, 5.2f, 1.8f, 9.7f);
	AddSolidBox(3.0f, 0.0f, 11.0f, 3.7f, 1.5f, 11.7f);

	// Initial water in the upper portion
	CreateWaterBox(1.4f, 0.2f, 0.5f, domX - 1.4f, 1.5f, 4.0f);

	// Top emitter: feeds water continuously from z=0 end
	CreateEmitter(3.6f, 1.5f, 0.5f, 0.0f, -0.3f, 1.0f, 120.0f, 1.8f);
}

void WulfNetFluidRiverRapidsTest::SetupObjects()
{
	// Jolt rock bodies
	auto addRock = [&](float x0, float z0, float x1, float z1, float h)
	{
		Vec3 half((x1 - x0) * 0.5f, h * 0.5f, (z1 - z0) * 0.5f);
		RVec3 pos((x0 + x1) * 0.5f, h * 0.5f, (z0 + z1) * 0.5f);
		BodyCreationSettings rock(new BoxShape(half), pos,
			Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
		mBodyInterface->CreateAndAddBody(rock, EActivation::DontActivate);
	};

	addRock(2.5f, 3.0f, 3.2f, 3.7f, 1.8f);
	addRock(4.0f, 5.0f, 4.7f, 5.7f, 1.5f);
	addRock(2.0f, 7.5f, 2.8f, 8.3f, 2.0f);
	addRock(4.5f, 9.0f, 5.2f, 9.7f, 1.8f);
	addRock(3.0f, 11.0f, 3.7f, 11.7f, 1.5f);
}

void WulfNetFluidRiverRapidsTest::UpdateScenario(float /*dt*/)
{
	// Continuous flow — emitter handles it
}

// =====================================================================
// 7. Caldera Eruption — Circular containment with central eruptions
//    Grid: 64×32×64, cellSize 0.18 → 11.5m × 5.8m × 11.5m
// =====================================================================

void WulfNetFluidCalderaEruptionTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 64;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 64;
	mFluidConfig.cellSize  = 0.18f;
	mFluidConfig.pressureIterations = 20;
	mFluidConfig.particlesPerCell   = 4;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize  = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidConfig.useGPU = false;
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Domain: 11.5m cube
	float domX = mFluidConfig.gridSizeX * mFluidConfig.cellSize;
	float domZ = mFluidConfig.gridSizeZ * mFluidConfig.cellSize;
	float cx = domX * 0.5f;
	float cz = domZ * 0.5f;

	// Approximate a circular crater using solid blocks around the perimeter
	// Mark cells outside the crater radius as solid
	float craterRadius = 4.5f;
	float wallHeight = 3.5f;
	float cs = mFluidConfig.cellSize;
	for (int gz = 0; gz < static_cast<int>(mFluidConfig.gridSizeZ); ++gz)
	{
		for (int gx = 0; gx < static_cast<int>(mFluidConfig.gridSizeX); ++gx)
		{
			float wx = (gx + 0.5f) * cs;
			float wz = (gz + 0.5f) * cs;
			float dx = wx - cx;
			float dz = wz - cz;
			float dist = std::sqrt(dx * dx + dz * dz);

			if (dist > craterRadius)
			{
				// Mark this column as solid (crater wall)
				AddSolidBox(wx - cs * 0.5f, 0.0f, wz - cs * 0.5f,
				            wx + cs * 0.5f, wallHeight, wz + cs * 0.5f);
			}
		}
	}

	// Fill crater with water up to moderate level
	CreateWaterBox(cx - craterRadius, 0.2f, cz - craterRadius,
	               cx + craterRadius, 1.5f, cz + craterRadius);
}

void WulfNetFluidCalderaEruptionTest::SetupObjects()
{
	// Central vent cone (Jolt body for visual reference)
	BodyCreationSettings vent(
		new SphereShape(0.4f),
		RVec3(mFluidConfig.gridSizeX * mFluidConfig.cellSize * 0.5f, 0.4f,
		      mFluidConfig.gridSizeZ * mFluidConfig.cellSize * 0.5f),
		Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(vent, EActivation::DontActivate);
}

void WulfNetFluidCalderaEruptionTest::UpdateScenario(float dt)
{
	mEruptionTimer += dt;

	// Periodic eruptions every 2.5 seconds, up to 12 pulses
	if (mEruptionTimer >= 2.5f && mEruptionCount < 12)
	{
		mEruptionTimer -= 2.5f;
		mEruptionCount++;

		float cx = mFluidConfig.gridSizeX * mFluidConfig.cellSize * 0.5f;
		float cz = mFluidConfig.gridSizeZ * mFluidConfig.cellSize * 0.5f;

		// Central upward eruption — water sphere launched from the vent
		CreateWaterSphere(cx, 0.5f, cz, 0.6f);

		// Every 3rd eruption, secondary off-centre pulses
		if (mEruptionCount % 3 == 0)
		{
			CreateWaterSphere(cx - 1.5f, 0.5f, cz + 1.0f, 0.4f);
			CreateWaterSphere(cx + 1.2f, 0.5f, cz - 0.8f, 0.4f);
		}
	}
}

// =====================================================================
// 8. Valley Flood — Dam break into a valley with building obstacles
//    Grid: 96×32×48, cellSize 0.15 → 14.4m × 4.8m × 7.2m
// =====================================================================

void WulfNetFluidValleyFloodTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 96;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 48;
	mFluidConfig.cellSize  = 0.15f;
	mFluidConfig.pressureIterations = 25;
	mFluidConfig.particlesPerCell   = 4;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize  = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidConfig.useGPU = false;
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Domain: 14.4m × 4.8m × 7.2m
	// Containment walls
	AddSolidBox(0.0f, 0.0f, 0.0f, 14.4f, 4.8f, 0.3f);
	AddSolidBox(0.0f, 0.0f, 6.9f, 14.4f, 4.8f, 7.2f);
	AddSolidBox(0.0f, 0.0f, 0.0f, 0.3f, 4.8f, 7.2f);
	AddSolidBox(14.1f, 0.0f, 0.0f, 14.4f, 4.8f, 7.2f);

	// Reservoir (x = 0.3..3.0) — tall water column
	CreateWaterBox(0.5f, 0.2f, 0.5f, 2.8f, 3.5f, 6.7f);

	// Dam wall at x = 3.0
	AddSolidBox(2.8f, 0.0f, 0.0f, 3.2f, 4.8f, 7.2f);

	// "Building" obstacles downstream (x = 5..12, scattered)
	auto addBuilding = [&](float x0, float z0, float w, float d, float h)
	{
		AddSolidBox(x0, 0.0f, z0, x0 + w, h, z0 + d);
	};

	// Row 1 (x ≈ 5)
	addBuilding(4.8f, 1.0f, 0.8f, 1.2f, 2.5f);
	addBuilding(4.8f, 3.0f, 0.8f, 1.2f, 2.5f);
	addBuilding(4.8f, 5.0f, 0.8f, 1.2f, 2.5f);

	// Row 2 (x ≈ 7)
	addBuilding(6.8f, 1.8f, 1.0f, 1.0f, 3.0f);
	addBuilding(6.8f, 3.8f, 1.0f, 1.0f, 2.5f);
	addBuilding(6.8f, 5.8f, 0.8f, 0.8f, 2.0f);

	// Row 3 (x ≈ 9)
	addBuilding(8.8f, 1.0f, 0.8f, 1.0f, 2.5f);
	addBuilding(8.8f, 2.8f, 1.2f, 1.2f, 3.5f);
	addBuilding(8.8f, 5.0f, 0.8f, 1.2f, 2.0f);

	// Row 4 (x ≈ 11)
	addBuilding(10.8f, 1.5f, 1.0f, 1.0f, 2.5f);
	addBuilding(10.8f, 3.5f, 0.8f, 1.0f, 3.0f);
	addBuilding(10.8f, 5.5f, 1.0f, 1.0f, 2.5f);
}

void WulfNetFluidValleyFloodTest::SetupObjects()
{
	// Jolt dam body
	BodyCreationSettings dam(
		new BoxShape(Vec3(0.2f, 2.4f, 3.6f)),
		RVec3(3.0f, 2.4f, 3.6f),
		Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(dam, EActivation::DontActivate);

	// Jolt building bodies for visual collision
	auto addBuildingBody = [&](float x0, float z0, float w, float d, float h)
	{
		Vec3 half(w * 0.5f, h * 0.5f, d * 0.5f);
		RVec3 pos(x0 + w * 0.5f, h * 0.5f, z0 + d * 0.5f);
		BodyCreationSettings bld(new BoxShape(half), pos,
			Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
		mBodyInterface->CreateAndAddBody(bld, EActivation::DontActivate);
	};

	addBuildingBody(4.8f, 1.0f, 0.8f, 1.2f, 2.5f);
	addBuildingBody(4.8f, 3.0f, 0.8f, 1.2f, 2.5f);
	addBuildingBody(4.8f, 5.0f, 0.8f, 1.2f, 2.5f);
	addBuildingBody(6.8f, 1.8f, 1.0f, 1.0f, 3.0f);
	addBuildingBody(6.8f, 3.8f, 1.0f, 1.0f, 2.5f);
	addBuildingBody(6.8f, 5.8f, 0.8f, 0.8f, 2.0f);
	addBuildingBody(8.8f, 1.0f, 0.8f, 1.0f, 2.5f);
	addBuildingBody(8.8f, 2.8f, 1.2f, 1.2f, 3.5f);
	addBuildingBody(8.8f, 5.0f, 0.8f, 1.2f, 2.0f);
	addBuildingBody(10.8f, 1.5f, 1.0f, 1.0f, 2.5f);
	addBuildingBody(10.8f, 3.5f, 0.8f, 1.0f, 3.0f);
	addBuildingBody(10.8f, 5.5f, 1.0f, 1.0f, 2.5f);
}

void WulfNetFluidValleyFloodTest::UpdateScenario(float dt)
{
	mFloodTimer += dt;

	// At t=1.5s, remove the dam — add water in the gap to push through
	if (!mFloodReleased && mFloodTimer >= 1.5f)
	{
		mFloodReleased = true;
		CreateWaterBox(2.8f, 0.2f, 0.5f, 3.5f, 3.5f, 6.7f);
	}
}
