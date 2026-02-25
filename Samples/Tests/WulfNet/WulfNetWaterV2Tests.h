// SPDX-License-Identifier: MIT
// WulfNet Water Physics V2 — Particle-Based Tests
//
// V2 water tests leverage the full particle physics pipeline:
//   - CO-FLIP fluid simulation with vorticity preservation
//   - MPM constitutive models (ViscousFluid, DruckerPrager, Snow)
//   - MPM↔Rigid coupling for bidirectional fluid-body interaction
//   - Marching cubes surface extraction
//   - Multi-material particle support
//
// Each test demonstrates a distinct real-world water scenario that
// requires tight integration between particle physics and rigid bodies.

#pragma once

#include <Tests/Test.h>

// WulfNet particle physics subsystems
#include <WulfNet/Physics/Fluids/COFLIPSystem.h>
#include <WulfNet/Physics/Fluids/FluidSurface.h>
#include <WulfNet/Physics/Fluids/FluidParticle.h>
#include <WulfNet/Physics/MPM/ConstitutiveModel.h>
#include <WulfNet/Physics/MPM/MPMRigidCoupling.h>
#include <WulfNet/Physics/Terrain/TerrainDeformation.h>
#include <WulfNet/Core/System/SystemMonitor.h>

#include <vector>
#include <random>
#include <chrono>

// ===========================================================================
// Base class for V2 particle water tests
// ===========================================================================
class WulfNetWaterV2Base : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV2Base)

public:
	virtual ~WulfNetWaterV2Base() override;

	void Initialize() override;
	void PrePhysicsUpdate(const PreUpdateParams &inParams) override;
	String GetStatusString() const override;

	// Derived classes override these
	virtual void SetupScenario() {}
	virtual void UpdateScenario(float dt) {}

protected:
	// ---- CO-FLIP fluid system ----
	WulfNet::COFLIPSystem		mFluid;
	WulfNet::COFLIPConfig		mFluidConfig;

	// ---- Marching cubes surface ----
	WulfNet::FluidSurface		mSurface;
	WulfNet::FluidSurfaceConfig	mSurfaceConfig;

	// ---- MPM rigid coupling ----
	WulfNet::MPMRigidCoupling	mCoupling;
	WulfNet::MPMCouplingConfig	mCouplingConfig;

	// Tracked coupled bodies (handle → BodyID)
	struct CoupledBodyEntry
	{
		uint32_t		couplingHandle = 0;
		JPH::BodyID		bodyId;
	};
	std::vector<CoupledBodyEntry> mCoupledBodies;

	// ---- Scenario init flag (must NOT be static!) ----
	bool mScenarioInitialized	= false;

	// ---- Rendering options ----
	bool mDrawParticles			= false;
	bool mDrawSurface			= true;
	bool mDrawCouplingForces	= false;
	float mParticleSize			= 0.02f;

	// ---- FPS tracking ----
	float mCurrentFPS			= 0.0f;
	float mFrameTimeMs			= 0.0f;
	int   mFrameCount			= 0;
	float mStatsTimer			= 0.0f;
	std::chrono::high_resolution_clock::time_point mLastFPSTime;

	// ---- Helpers ----

	/// Create a water box and return the particle count before/after for tracking.
	void CreateWaterBox(float minX, float minY, float minZ,
	                    float maxX, float maxY, float maxZ);

	void CreateWaterSphere(float cx, float cy, float cz, float radius);

	void CreateEmitter(float x, float y, float z,
	                   float dirX, float dirY, float dirZ,
	                   float rate, float speed);

	void AddSolidBox(float minX, float minY, float minZ,
	                 float maxX, float maxY, float maxZ);

	void AddSolidSphere(float cx, float cy, float cz, float radius);

	/// Register a Jolt rigid body for bidirectional MPM coupling.
	uint32_t RegisterCoupledBody(JPH::BodyID bodyId,
	                             WulfNet::CoupledRigidBody::ShapeType shape,
	                             float radius = 0.5f,
	                             float hx = 0.5f, float hy = 0.5f, float hz = 0.5f);

	void DrawFluidParticles();
	void DrawFluidSurface();
	void DrawCouplingDebug();
	void DrawStats();
};

// ===========================================================================
// 1. Dam Break — Classic dam break with rigid body coupling
//    Water column collapses into a basin, interacting with floating/sinking
//    rigid bodies via MPM↔rigid bidirectional forces.
// ===========================================================================
class WulfNetWaterV2DamBreakTest : public WulfNetWaterV2Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV2DamBreakTest)

public:
	const char *GetDescription() const override
	{
		return "Dam break: water column collapses into rigid bodies via MPM coupling.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	void CreateBasinWalls();

	float mDamReleaseTimer = 0.0f;
	bool  mDamReleased     = false;
};

// ===========================================================================
// 2. Multi-Material — Water, mud, and lava particles interacting
//    Demonstrates per-particle material properties using different
//    constitutive models (ViscousFluid, DruckerPrager) in the same scene.
// ===========================================================================
class WulfNetWaterV2MultiMaterialTest : public WulfNetWaterV2Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV2MultiMaterialTest)

public:
	const char *GetDescription() const override
	{
		return "Multi-material: water, mud, and lava particles with constitutive models.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	// Track emitter timing for periodic lava drops
	float mLavaDropTimer = 0.0f;
};

// ===========================================================================
// 3. Wave Pool — Emitter-driven waves with coupled floating objects
//    Oscillating emitters generate waves that push and toss rigid bodies
//    around the pool. MPM coupling provides realistic two-way interaction.
// ===========================================================================
class WulfNetWaterV2WavePoolTest : public WulfNetWaterV2Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV2WavePoolTest)

public:
	const char *GetDescription() const override
	{
		return "Wave pool: oscillating emitters drive waves through coupled rigid bodies.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	float mWavePhase = 0.0f;

	// Body IDs of wave paddles (kinematic)
	std::vector<JPH::BodyID> mPaddleBodies;
};

// ===========================================================================
// 4. Particle Erosion — Water erodes a terrain heightfield
//    Streams of water particles flow over elevated terrain, deforming the
//    heightfield via TerrainDeformation, carving channels and depositing
//    sediment downstream.
// ===========================================================================
class WulfNetWaterV2ErosionTest : public WulfNetWaterV2Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV2ErosionTest)

public:
	const char *GetDescription() const override
	{
		return "Erosion: water particles carve terrain heightfield channels.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	void DrawTerrain();

	WulfNet::TerrainDeformation		mTerrain;
	WulfNet::TerrainDeformConfig	mTerrainConfig;
	float mErosionTimer = 0.0f;
};

