// SPDX-License-Identifier: MIT
// WulfNet Water Physics V4 — Elastic Ball Water
//
// V4 renders water as visible spheres ("balls") with seamless gap filling:
//   - Each active particle is drawn as a small solid sphere
//   - Nearby particle pairs are connected by midpoint "bridge" spheres
//     that fill the empty space between touching balls seamlessly
//   - Physics tuned for elastic / bouncy ball behaviour: high surface tension,
//     high stiffness, low viscosity, high FLIP ratio
//   - No marching-cubes surface — pure sphere + bridge rendering
//   - Spatial hash for O(n) neighbour-pair detection each frame
//
// The result: water looks like a collection of elastic balls that merge
// smoothly when touching and bounce apart when disturbed.

#pragma once

#include <Tests/Test.h>

// MPM FluidSystem for particle dynamics
#include <WulfNet/Physics/Fluids/FluidSystem.h>
#include <WulfNet/Physics/Fluids/FluidParticle.h>
#include <WulfNet/Core/System/SystemMonitor.h>

#include <vector>
#include <unordered_map>
#include <chrono>

// ===========================================================================
// Base class — elastic ball water renderer + physics
// ===========================================================================
class WulfNetWaterV4Base : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV4Base)

public:
	virtual ~WulfNetWaterV4Base() override;

	void Initialize() override;
	void PrePhysicsUpdate(const PreUpdateParams &inParams) override;
	String GetStatusString() const override;

	// Derived tests override these
	virtual void SetupScenario() {}
	virtual void UpdateScenario(float dt) {}

protected:
	// ---- MPM particle fluid system ----
	WulfNet::FluidSystem		mFluid;
	WulfNet::FluidSystemConfig	mFluidConfig;

	// ---- Material IDs ----
	uint32_t mElasticWaterId	= 0;

	// ---- Per-material visual style ----
	struct MaterialVisual
	{
		Color ballColor;
		Color bridgeColor;
	};
	std::vector<MaterialVisual> mMaterialVisuals;

	// ---- Ball rendering parameters ----
	float    mBallRadius		= 0.045f;	// Drawn sphere radius
	float    mBridgeThreshold	= 0.105f;	// Max centre-to-centre for a bridge
	float    mBridgeSizeFactor	= 0.75f;	// Bridge sphere size relative to ball
	uint32_t mMaxBridges		= 60000;	// Per-frame draw cap

	// ---- Scenario init flag ----
	bool mScenarioSetup			= false;

	// ---- FPS / stats ----
	float    mCurrentFPS		= 0.0f;
	float    mFrameTimeMs		= 0.0f;
	int      mFrameCount		= 0;
	float    mStatsTimer		= 0.0f;
	uint32_t mBridgeCount		= 0;
	std::chrono::high_resolution_clock::time_point mLastFPSTime;

	// ---- Helpers (fluid setup) ----
	void AddWaterBox(float minX, float minY, float minZ,
	                 float maxX, float maxY, float maxZ,
	                 uint32_t materialId = 0);

	void AddWaterSphere(float cx, float cy, float cz, float radius,
	                    uint32_t materialId = 0);

	uint32_t AddEmitter(const WulfNet::FluidEmitter &emitter);
	uint32_t AddCollider(const WulfNet::FluidCollider &collider);

	// ---- Rendering ----
	void DrawBalls();
	void DrawBridges();

private:
	// Spatial hash for O(n) bridge-neighbour detection
	std::unordered_map<uint64_t, std::vector<uint32_t>> mSpatialHash;

	static uint64_t HashCell(int cx, int cy, int cz);
	void BuildSpatialHash(const WulfNet::FluidParticle *particles,
	                       uint32_t count);
};

// ===========================================================================
// 1. Ball Pool — Emitter pours elastic balls into a walled basin.
//    Balls bounce off walls and each other, settling into a pool.
// ===========================================================================
class WulfNetWaterV4BallPoolTest : public WulfNetWaterV4Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV4BallPoolTest)

public:
	const char *GetDescription() const override
	{
		return "Ball pool: elastic water balls pour and settle in a basin.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;
};

// ===========================================================================
// 2. Elastic Cascade — Three side-by-side elevated columns of water balls
//    with soft / medium / firm elasticity drain simultaneously, showing how
//    surface tension and stiffness affect the ball-water look.
// ===========================================================================
class WulfNetWaterV4ElasticCascadeTest : public WulfNetWaterV4Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV4ElasticCascadeTest)

public:
	const char *GetDescription() const override
	{
		return "Elastic cascade: soft / medium / firm water balls drain side by side.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	uint32_t mSoftMatId   = 0;
	uint32_t mMediumMatId = 0;
	uint32_t mFirmMatId   = 0;
};

// ===========================================================================
// 3. Ball Splash — A calm basin of ball-water is hit by a brief high-speed
//    burst of particles, scattering balls outward and upward.
// ===========================================================================
class WulfNetWaterV4BallSplashTest : public WulfNetWaterV4Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV4BallSplashTest)

public:
	const char *GetDescription() const override
	{
		return "Ball splash: high-speed ball burst impacts a calm pool.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	float    mSplashTimer      = 0.0f;
	bool     mSplashTriggered  = false;
	uint32_t mBurstEmitterId   = UINT32_MAX;
};

// ===========================================================================
// 4. Ball Waterfall — Elastic balls pour off a cliff ledge into a pool
//    below, demonstrating free-fall bridging and splash collection.
// ===========================================================================
class WulfNetWaterV4BallWaterfallTest : public WulfNetWaterV4Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV4BallWaterfallTest)

public:
	const char *GetDescription() const override
	{
		return "Ball waterfall: elastic balls cascade over a cliff into a pool.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;
};
