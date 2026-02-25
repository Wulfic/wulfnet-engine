// SPDX-License-Identifier: MIT
// WulfNet Water Physics V3 — MPM FluidSystem as Water
//
// V3 water tests use the original MPM FluidSystem pipeline directly:
//   - FluidParticle (64-byte GPU-aligned) with per-particle material
//   - FluidMaterial presets (Water, Oil, Honey, Mud, Lava)
//   - FluidEmitter / FluidCollider / BuoyancyObject subsystems
//   - MAC staggered grid with P2G → pressure solve → G2P transfer
//   - FluidSurface marching cubes for smooth rendering
//
// The particle system IS the fluid — every drop of water is a discrete
// physical particle carrying mass, velocity, density, and temperature.
// This contrasts with V1 (CO-FLIP high-level) and V2 (CO-FLIP + rigid coupling).

#pragma once

#include <Tests/Test.h>

// MPM FluidSystem — the particle-is-the-water pipeline
#include <WulfNet/Physics/Fluids/FluidSystem.h>
#include <WulfNet/Physics/Fluids/FluidParticle.h>
#include <WulfNet/Physics/Fluids/FluidGrid.h>
#include <WulfNet/Physics/Fluids/FluidSurface.h>
#include <WulfNet/Physics/Fluids/COFLIPSystem.h>   // for FluidSurface interop
#include <WulfNet/Core/System/SystemMonitor.h>

#include <vector>
#include <random>
#include <chrono>

// ===========================================================================
//  Base class — wraps FluidSystem for the Samples viewer
// ===========================================================================
class WulfNetWaterV3Base : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV3Base)

public:
	virtual ~WulfNetWaterV3Base() override;

	void Initialize() override;
	void PrePhysicsUpdate(const PreUpdateParams &inParams) override;
	String GetStatusString() const override;

	// Override in derived tests
	virtual void SetupScenario() {}
	virtual void UpdateScenario(float dt) {}

protected:
	// ---- MPM particle fluid system ----
	WulfNet::FluidSystem		mFluid;
	WulfNet::FluidSystemConfig	mFluidConfig;

	// ---- Marching cubes surface ----
	WulfNet::FluidSurface		mSurface;
	WulfNet::FluidSurfaceConfig	mSurfaceConfig;

	// ---- Material palette ----
	uint32_t mWaterMaterialId	= 0;
	uint32_t mOilMaterialId		= 0;
	uint32_t mHoneyMaterialId	= 0;
	uint32_t mMudMaterialId		= 0;
	uint32_t mLavaMaterialId	= 0;

	// ---- Scenario init flag (must NOT be static!) ----
	bool mScenarioSetup			= false;

	// ---- Render options ----
	bool mDrawParticles			= true;
	bool mDrawSurface			= true;
	bool mDrawGrid				= false;
	bool mDrawVelocities		= false;
	bool mColorByMaterial		= true;
	float mParticleSize			= 0.018f;

	// ---- FPS / stats ----
	float mCurrentFPS			= 0.0f;
	float mFrameTimeMs			= 0.0f;
	int   mFrameCount			= 0;
	float mStatsTimer			= 0.0f;
	std::chrono::high_resolution_clock::time_point mLastFPSTime;

	// ---- Helpers ----
	void AddWaterBox(float minX, float minY, float minZ,
	                 float maxX, float maxY, float maxZ,
	                 uint32_t materialId = 0);

	void AddWaterSphere(float cx, float cy, float cz, float radius,
	                    uint32_t materialId = 0);

	uint32_t AddEmitter(const WulfNet::FluidEmitter &emitter);
	uint32_t AddCollider(const WulfNet::FluidCollider &collider);
	uint32_t AddBuoyancyObject(uint32_t bodyId, float density, float volume,
	                           float drag = 0.5f);

	void DrawParticles();
	void DrawSurfaceMesh();
	void DrawGridSlice(float ySlice);
};

// ===========================================================================
// 1. Ocean Swell — Large open-water body with wind emitters and gravity waves.
//    Demonstrates FluidSystem at scale with FluidMaterial::Water, multiple
//    directional emitters simulating wind-driven surface current, and
//    BuoyancyObjects (crates, barrels) bobbing on the surface.
// ===========================================================================
class WulfNetWaterV3OceanSwellTest : public WulfNetWaterV3Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV3OceanSwellTest)

public:
	const char *GetDescription() const override
	{
		return "Ocean swell: MPM FluidSystem water body with wind emitters and buoyant crates.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	std::vector<uint32_t> mEmitterIds;
	float mWindPhase = 0.0f;
};

// ===========================================================================
// 2. Viscous Cascade — Side-by-side columns of water, oil, and honey
//    released simultaneously. Each material flows at a different rate,
//    producing a clear visual comparison of FluidMaterial viscosity presets.
// ===========================================================================
class WulfNetWaterV3ViscousCascadeTest : public WulfNetWaterV3Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV3ViscousCascadeTest)

public:
	const char *GetDescription() const override
	{
		return "Viscous cascade: water vs oil vs honey particles, side-by-side viscosity race.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	float mReleaseTimer = 0.0f;
	bool  mReleased     = false;
};

// ===========================================================================
// 3. Thermal Convection — Hot lava particles meet cold water, driving
//    convection currents. Per-particle temperature + density changes produce
//    visible convection cells. BuoyancyObjects sink in lava, float in water.
// ===========================================================================
class WulfNetWaterV3ThermalConvectionTest : public WulfNetWaterV3Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV3ThermalConvectionTest)

public:
	const char *GetDescription() const override
	{
		return "Thermal convection: hot lava meets cold water with per-particle temperature.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	float mLavaInjectTimer = 0.0f;
};

// ===========================================================================
// 4. Spray & Foam — High-speed jet of water into a shallow basin producing
//    spray, foam, and bubble secondary particles classified by ParticleFlags.
//    Render modes distinguish surface / spray / foam / bubble particles.
// ===========================================================================
class WulfNetWaterV3SprayFoamTest : public WulfNetWaterV3Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV3SprayFoamTest)

public:
	const char *GetDescription() const override
	{
		return "Spray & foam: high-speed jet producing spray/foam/bubble particles.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	float mJetAngle = 0.0f;
};

// ===========================================================================
// 5. Obstacle Course — Flow of water particles through a gauntlet of
//    FluidColliders (spheres, boxes, capsules) and a heightfield ramp,
//    collecting in a basin at the bottom. Shows the full collider pipeline.
// ===========================================================================
class WulfNetWaterV3ObstacleCourseTest : public WulfNetWaterV3Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV3ObstacleCourseTest)

public:
	const char *GetDescription() const override
	{
		return "Obstacle course: water flows through sphere/box/capsule FluidColliders.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	float mTime = 0.0f;
};

