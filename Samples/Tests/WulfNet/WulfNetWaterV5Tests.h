// SPDX-License-Identifier: MIT
// WulfNet Water Physics V5 — Sheet Water (Shallow Water Equations)
//
// V5 is completely different from V1–V4: water is a 2D height-field sheet
// that sits on top of surfaces and flows under gravity.  No particles.
//
//   - 2D grid of water columns: each cell stores height h, velocity (vx, vz)
//   - Underlying terrain height-field supports hills, basins, slopes
//   - Shallow Water Equations (SWE) solver: conservation of mass and momentum
//     with gravity-driven flux, viscous damping, and surface tension
//   - Rendered as a triangle mesh via DrawTriangle — colour / alpha vary
//     with depth for a natural transparent-shallow / opaque-deep look
//   - Ripples propagate naturally through the wave equation component
//
// The result: a thin sheet of water that flows over terrain, pools in
// valleys, ripples when disturbed, and travels across surfaces.

#pragma once

#include <Tests/Test.h>
#include <WulfNet/Core/System/SystemMonitor.h>

#include <vector>
#include <chrono>
#include <cstdint>

// ===========================================================================
// Water cell — one column in the 2D grid
// ===========================================================================
struct WaterCell
{
	float waterHeight = 0.0f;   // Depth of water above terrain (metres)
	float terrainHeight = 0.0f; // Ground elevation
	float vx = 0.0f;           // Velocity X (horizontal)
	float vz = 0.0f;           // Velocity Z (horizontal)
};

// ===========================================================================
// Sheet water configuration
// ===========================================================================
struct SheetWaterConfig
{
	uint32_t gridSizeX        = 80;     // Number of cells in X
	uint32_t gridSizeZ        = 80;     // Number of cells in Z
	float    cellSize         = 0.1f;   // Metres per cell edge

	float    gravity          = 9.81f;  // m/s² (positive — downward acceleration)
	float    damping          = 0.002f; // Per-step velocity damping (low → long ripples)
	float    viscosity        = 0.0001f;// Velocity diffusion (Laplacian smoothing of vel)
	float    minWaterDraw     = 0.001f; // Cells below this depth are not rendered

	// World-space origin of the grid corner (0,0)
	float    originX          = 0.0f;
	float    originY          = 0.0f;   // Baseline Y — terrain heights are relative to this
	float    originZ          = 0.0f;

	// Source / sink helpers
	uint32_t substeps         = 2;      // SWE substeps per frame for stability

	// Colour ramp
	Color    shallowColor     = Color(120, 200, 240, 100);  // Transparent light blue
	Color    deepColor        = Color(20, 60, 160, 230);     // Opaque dark blue
	float    depthColorScale  = 1.0f;   // Depth at which colour is fully "deep"
};

// ===========================================================================
// Base class — shared shallow-water solver and triangle renderer
// ===========================================================================
class WulfNetWaterV5Base : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV5Base)

public:
	virtual ~WulfNetWaterV5Base() override = default;

	void Initialize() override;
	void PrePhysicsUpdate(const PreUpdateParams &inParams) override;
	String GetStatusString() const override;

	// Derived tests override these
	virtual void SetupScenario() {}
	virtual void UpdateScenario(float dt) {}

protected:
	// ---- Configuration (set in SetupScenario before base Initialize runs) ----
	SheetWaterConfig mConfig;

	// ---- Grid data ----
	std::vector<WaterCell> mGrid;            // gridSizeX * gridSizeZ
	std::vector<WaterCell> mGridTemp;        // Temporary buffer for update

	// ---- Access helpers ----
	inline uint32_t CellIndex(uint32_t x, uint32_t z) const
	{
		return z * mConfig.gridSizeX + x;
	}
	inline WaterCell &Cell(uint32_t x, uint32_t z)         { return mGrid[CellIndex(x, z)]; }
	inline const WaterCell &Cell(uint32_t x, uint32_t z) const { return mGrid[CellIndex(x, z)]; }

	// ---- Terrain helpers — derived classes call these in SetupScenario ----
	void SetTerrainFlat(float height);
	void SetTerrainSlope(float startHeight, float endHeight, bool alongX = true);
	void SetTerrainBowl(float rimHeight, float centerHeight);
	void SetTerrainHills(float baseHeight, float amplitude, float frequencyX, float frequencyZ);
	void SetTerrainAt(uint32_t x, uint32_t z, float height);

	// ---- Water source helpers ----
	void AddWaterRect(uint32_t x0, uint32_t z0, uint32_t x1, uint32_t z1, float depth);
	void AddWaterDisk(uint32_t cx, uint32_t cz, uint32_t radius, float depth);
	void AddWaterDrop(uint32_t cx, uint32_t cz, uint32_t radius, float peakDepth);

	// ---- Scenario init flag ----
	bool mScenarioSetup = false;

	// ---- FPS / stats ----
	float mCurrentFPS   = 0.0f;
	float mFrameTimeMs  = 0.0f;
	int   mFrameCount   = 0;
	float mStatsTimer   = 0.0f;
	float mSimTimeMs    = 0.0f;
	float mTotalWater   = 0.0f;  // Sum of all water heights (conservation check)
	std::chrono::high_resolution_clock::time_point mLastFPSTime;

private:
	// ---- Solver ----
	void StepSWE(float dt);
	void ApplyBoundary();

	// ---- Rendering ----
	void DrawSheet();
	Color DepthColor(float depth) const;
};

// ===========================================================================
// 1. Ripple Pond — Flat basin with periodic point disturbances that create
//    expanding ring ripples.  Demonstrates natural wave propagation.
// ===========================================================================
class WulfNetWaterV5RipplePondTest : public WulfNetWaterV5Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV5RipplePondTest)

public:
	const char *GetDescription() const override
	{
		return "Ripple pond: point disturbances create expanding ring waves on a sheet.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	float mDropTimer   = 0.0f;
	int   mDropCount   = 0;
};

// ===========================================================================
// 2. Terrain Flow — Water released on a sloped / hilly terrain flows
//    downhill and pools in valleys.
// ===========================================================================
class WulfNetWaterV5TerrainFlowTest : public WulfNetWaterV5Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV5TerrainFlowTest)

public:
	const char *GetDescription() const override
	{
		return "Terrain flow: water sheet flows downhill and pools in valleys.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;
};

// ===========================================================================
// 3. Sheet Dam Break — A tall wall of sheet water is released to rush
//    across a flat surface, showing the characteristic front wave.
// ===========================================================================
class WulfNetWaterV5DamBreakTest : public WulfNetWaterV5Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV5DamBreakTest)

public:
	const char *GetDescription() const override
	{
		return "Sheet dam break: wall of water rushes across a flat surface.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	bool mDamReleased = false;
	float mDamTimer   = 0.0f;
};

// ===========================================================================
// 4. Rain on Hills — Random rain drops fall on hilly terrain, water
//    collects in valleys over time, creating natural drainage patterns.
// ===========================================================================
class WulfNetWaterV5RainHillsTest : public WulfNetWaterV5Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV5RainHillsTest)

public:
	const char *GetDescription() const override
	{
		return "Rain on hills: random drops on terrain, water collects in valleys.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	float mRainTimer     = 0.0f;
	float mRainInterval  = 0.05f;  // Seconds between drops
	int   mTotalDrops    = 0;
};

// ===========================================================================
// 5. Tsunami Coast — Massive wave generated offshore crashes onto a
//    sloped coastline with hills.  Shows wave shoaling and flooding.
// ===========================================================================
class WulfNetWaterV5TsunamiTest : public WulfNetWaterV5Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV5TsunamiTest)

public:
	const char *GetDescription() const override
	{
		return "Tsunami: massive wave crashes onto a coastal landscape.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	bool  mWaveTriggered = false;
	float mWaveTimer     = 0.0f;
};

// ===========================================================================
// 6. River Canyon — Water flows through a winding canyon carved into
//    an elevated plateau.  Demonstrates channeled flow and rapids.
// ===========================================================================
class WulfNetWaterV5RiverCanyonTest : public WulfNetWaterV5Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV5RiverCanyonTest)

public:
	const char *GetDescription() const override
	{
		return "River canyon: water flows through a winding canyon carved in rock.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	float mSourceTimer = 0.0f;
};

// ===========================================================================
// 7. Caldera Lake — Volcanic crater lake with a central vent.  Periodic
//    eruption pulses create standing waves and interference patterns.
// ===========================================================================
class WulfNetWaterV5CalderaLakeTest : public WulfNetWaterV5Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV5CalderaLakeTest)

public:
	const char *GetDescription() const override
	{
		return "Caldera lake: volcanic crater lake with periodic eruption pulses.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	float mEruptionTimer = 0.0f;
	int   mEruptionCount = 0;
};

// ===========================================================================
// 8. Flood Valley — A reservoir dam breaks, sending a massive flood wave
//    through a valley filled with obstacle buildings.
// ===========================================================================
class WulfNetWaterV5FloodValleyTest : public WulfNetWaterV5Base
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV5FloodValleyTest)

public:
	const char *GetDescription() const override
	{
		return "Flood valley: dam break sends water rushing through a village.";
	}

	void SetupScenario() override;
	void UpdateScenario(float dt) override;

private:
	bool  mFloodReleased = false;
	float mFloodTimer    = 0.0f;
};
