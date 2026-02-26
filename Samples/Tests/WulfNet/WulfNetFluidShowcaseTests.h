// =============================================================================
// WulfNet Engine - CO-FLIP Fluid Showcase Tests
// =============================================================================
// Large-scale, visually impressive CO-FLIP particle fluid test scenarios.
// Mirrors the V5 sheet water tests but uses full 3D particle simulation
// with marching cubes surface extraction for smooth water rendering.
//
// Each test demonstrates a different aspect of the CO-FLIP solver:
//   1. Ripple Basin      — Periodic drops into a calm basin (3D ripple waves)
//   2. Terrain Cascade   — Water flowing over terrain obstacles
//   3. Dam Break 3D      — Classic particle dam break with obstacles
//   4. Rain Storm        — Random rain drops accumulate in terrain
//   5. Tsunami Surge     — Massive wave hits coastal obstacles
//   6. River Rapids      — Channeled flow through a winding river
//   7. Caldera Eruption  — Crater lake with periodic eruption pulses
//   8. Valley Flood      — Dam break floods through building obstacles
// =============================================================================

#pragma once

#include <Tests/WulfNet/WulfNetFluidTest.h>
#include <chrono>

// =============================================================================
// Base class for showcase tests — extends WulfNetFluidTest with per-frame
// scenario update hook and larger default grids
// =============================================================================
class WulfNetFluidShowcase : public WulfNetFluidTest
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFluidShowcase)

public:
	virtual ~WulfNetFluidShowcase() override = default;

	// Override PrePhysicsUpdate to call UpdateScenario each frame
	void PrePhysicsUpdate(const PreUpdateParams &inParams) override;

	// Derived tests implement these
	virtual void UpdateScenario(float dt) {}

protected:
	float mScenarioTime = 0.0f;   // Accumulated time since test start
};

// =============================================================================
// 1. Ripple Basin — Large basin, periodic drops create 3D surface ripples
// =============================================================================
class WulfNetFluidRippleBasinTest : public WulfNetFluidShowcase
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFluidRippleBasinTest)

public:
	const char *GetDescription() const override
	{
		return "CO-FLIP ripple basin: periodic drops create expanding 3D surface waves.";
	}

	void SetupFluid() override;
	void UpdateScenario(float dt) override;

private:
	float    mDropTimer  = 0.0f;
	int      mDropCount  = 0;
};

// =============================================================================
// 2. Terrain Cascade — Water released on high ground cascades down steps
// =============================================================================
class WulfNetFluidTerrainCascadeTest : public WulfNetFluidShowcase
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFluidTerrainCascadeTest)

public:
	const char *GetDescription() const override
	{
		return "CO-FLIP terrain cascade: water flows over stepped terrain and pools at bottom.";
	}

	void SetupFluid() override;
	void SetupObjects() override;
	void UpdateScenario(float dt) override;
};

// =============================================================================
// 3. Dam Break 3D — Tall column of water released against obstacles
// =============================================================================
class WulfNetFluidDamBreak3DTest : public WulfNetFluidShowcase
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFluidDamBreak3DTest)

public:
	const char *GetDescription() const override
	{
		return "CO-FLIP 3D dam break: massive particle column smashes through obstacles.";
	}

	void SetupFluid() override;
	void SetupObjects() override;
	void UpdateScenario(float dt) override;

private:
	bool  mDamReleased = false;
	float mDamTimer    = 0.0f;
};

// =============================================================================
// 4. Rain Storm — Random rain drops accumulate on terrain with obstacles
// =============================================================================
class WulfNetFluidRainStormTest : public WulfNetFluidShowcase
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFluidRainStormTest)

public:
	const char *GetDescription() const override
	{
		return "CO-FLIP rain storm: random drops accumulate in terrain depressions.";
	}

	void SetupFluid() override;
	void SetupObjects() override;
	void UpdateScenario(float dt) override;

private:
	float    mRainTimer   = 0.0f;
	int      mTotalDrops  = 0;
};

// =============================================================================
// 5. Tsunami Surge — Massive wave impacts a row of coastal pillars
// =============================================================================
class WulfNetFluidTsunamiSurgeTest : public WulfNetFluidShowcase
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFluidTsunamiSurgeTest)

public:
	const char *GetDescription() const override
	{
		return "CO-FLIP tsunami surge: massive wave crashes into coastal pillars.";
	}

	void SetupFluid() override;
	void SetupObjects() override;
	void UpdateScenario(float dt) override;

private:
	bool  mWaveTriggered = false;
	float mWaveTimer     = 0.0f;
};

// =============================================================================
// 6. River Rapids — Continuous flow through a winding channel with rocks
// =============================================================================
class WulfNetFluidRiverRapidsTest : public WulfNetFluidShowcase
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFluidRiverRapidsTest)

public:
	const char *GetDescription() const override
	{
		return "CO-FLIP river rapids: continuous flow through a rock-strewn channel.";
	}

	void SetupFluid() override;
	void SetupObjects() override;
	void UpdateScenario(float dt) override;

private:
	float mSourceTimer = 0.0f;
};

// =============================================================================
// 7. Caldera Eruption — Crater lake with periodic central eruption pulses
// =============================================================================
class WulfNetFluidCalderaEruptionTest : public WulfNetFluidShowcase
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFluidCalderaEruptionTest)

public:
	const char *GetDescription() const override
	{
		return "CO-FLIP caldera eruption: crater lake with periodic eruption pulses.";
	}

	void SetupFluid() override;
	void SetupObjects() override;
	void UpdateScenario(float dt) override;

private:
	float mEruptionTimer = 0.0f;
	int   mEruptionCount = 0;
};

// =============================================================================
// 8. Valley Flood — Dam release floods a valley with building obstacles
// =============================================================================
class WulfNetFluidValleyFloodTest : public WulfNetFluidShowcase
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFluidValleyFloodTest)

public:
	const char *GetDescription() const override
	{
		return "CO-FLIP valley flood: dam break sends water through a building grid.";
	}

	void SetupFluid() override;
	void SetupObjects() override;
	void UpdateScenario(float dt) override;

private:
	bool  mFloodReleased = false;
	float mFloodTimer    = 0.0f;
};
