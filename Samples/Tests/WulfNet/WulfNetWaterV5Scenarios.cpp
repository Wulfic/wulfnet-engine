// =============================================================================
// WulfNet Water V5 — Scenario Implementations
// =============================================================================
// 8 scenario subclasses: RipplePond, TerrainFlow, DamBreak, RainHills,
// Tsunami, RiverCanyon, CalderaLake, FloodValley.
// Extracted from WulfNetWaterV5Tests.cpp for maintainability.
// =============================================================================

#include <Samples.h>
#include "WulfNetWaterV5Tests.h"
#include "WaterDiagnostics.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

// Register RTTI for scenario subclasses
JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV5RipplePondTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV5RipplePondTest, WulfNetWaterV5Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV5TerrainFlowTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV5TerrainFlowTest, WulfNetWaterV5Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV5DamBreakTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV5DamBreakTest, WulfNetWaterV5Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV5RainHillsTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV5RainHillsTest, WulfNetWaterV5Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV5TsunamiTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV5TsunamiTest, WulfNetWaterV5Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV5RiverCanyonTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV5RiverCanyonTest, WulfNetWaterV5Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV5CalderaLakeTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV5CalderaLakeTest, WulfNetWaterV5Base)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV5FloodValleyTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV5FloodValleyTest, WulfNetWaterV5Base)
}

// =====================================================================
// WulfNetWaterV5Base — Initialization

// =====================================================================
// 1. Ripple Pond — Point disturbances on a massive flat basin
//    Grid: 400×400, cellSize 0.2 → 80m × 80m world (was 8m × 8m → 10x)
// =====================================================================

void WulfNetWaterV5RipplePondTest::SetupScenario()
{
	SWE_LOG_INFO("[SCENARIO] RipplePond — 400x400 grid, periodic drops on shallow basin");
	mConfig.gridSizeX       = 400;
	mConfig.gridSizeZ       = 400;
	mConfig.cellSize        = 0.2f;
	mConfig.damping         = 0.0008f;   // Very low → long-lived ripples
	mConfig.viscosity       = 0.00005f;
	mConfig.substeps        = 2;
	mConfig.depthColorScale = 0.6f;
	mConfig.originX         = -(400 * 0.2f) / 2.0f;  // Centre on origin
	mConfig.originY         = 0.2f;
	mConfig.originZ         = -(400 * 0.2f) / 2.0f;

	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	mGrid.resize(total);

	// Large bowl terrain — depression in centre for natural pooling
	SetTerrainBowl(0.3f, 0.0f);

	// Pre-fill a thin uniform layer of water across the whole basin
	AddWaterRect(0, 0, mConfig.gridSizeX, mConfig.gridSizeZ, 0.2f);
}

void WulfNetWaterV5RipplePondTest::UpdateScenario(float dt)
{
	mDropTimer += dt;

	// Drop a ripple every 0.6 seconds, up to 60 drops total
	if (mDropTimer >= 0.6f && mDropCount < 60)
	{
		mDropTimer -= 0.6f;
		mDropCount++;

		// Five varied drop positions for interesting interference patterns
		uint32_t cx, cz;
		switch (mDropCount % 5)
		{
		case 0: cx = mConfig.gridSizeX / 2;     cz = mConfig.gridSizeZ / 2;     break;
		case 1: cx = mConfig.gridSizeX / 4;     cz = mConfig.gridSizeZ / 4;     break;
		case 2: cx = mConfig.gridSizeX * 3 / 4; cz = mConfig.gridSizeZ / 3;     break;
		case 3: cx = mConfig.gridSizeX / 3;     cz = mConfig.gridSizeZ * 3 / 4; break;
		default:cx = mConfig.gridSizeX * 2 / 3; cz = mConfig.gridSizeZ * 2 / 3; break;
		}

		// Larger radius drops (scaled up from 4 to 12) for visible ripples
		AddWaterDrop(cx, cz, 12, 0.15f);
		WaterDiagnostics::LogEvent(SWE_LOG_CAT,
			"Drop #" + std::to_string(mDropCount) +
			" at (" + std::to_string(cx) + "," + std::to_string(cz) + ")");
	}
}

// =====================================================================
// 2. Terrain Flow — Water on a large hilly landscape
//    Grid: 350×350, cellSize 0.25 → 87.5m × 87.5m world (was 8m → ~11x)
// =====================================================================

void WulfNetWaterV5TerrainFlowTest::SetupScenario()
{
	SWE_LOG_INFO("[SCENARIO] TerrainFlow — 350x350 grid, hilly landscape with ridge");
	mConfig.gridSizeX       = 350;
	mConfig.gridSizeZ       = 350;
	mConfig.cellSize        = 0.25f;
	mConfig.damping         = 0.003f;
	mConfig.viscosity       = 0.0003f;
	mConfig.substeps        = 2;
	mConfig.depthColorScale = 1.0f;
	mConfig.originX         = -(350 * 0.25f) / 2.0f;
	mConfig.originY         = 0.2f;
	mConfig.originZ         = -(350 * 0.25f) / 2.0f;

	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	mGrid.resize(total);

	// Complex multi-octave hilly landscape with a ridge line
	const float pi = 3.14159265f;
	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float fx = static_cast<float>(x) / mConfig.gridSizeX;
			float fz = static_cast<float>(z) / mConfig.gridSizeZ;

			// Base slope: higher on left, lower on right
			float slope = 3.0f * (1.0f - fx);

			// Multiple overlapping sinusoidal hills at different scales
			float hills1 = 1.0f  * std::sin(fx * 3.0f * pi)  * std::sin(fz * 2.0f * pi);
			float hills2 = 0.5f  * std::sin(fx * 7.0f * pi + 0.5f) * std::sin(fz * 5.0f * pi + 1.0f);
			float hills3 = 0.3f  * std::sin(fx * 11.0f * pi + 2.0f) * std::sin(fz * 9.0f * pi + 1.5f);

			// Ridge line across the middle — water has to flow around it
			float ridge = 0.8f * std::exp(-((fz - 0.5f) * (fz - 0.5f)) / 0.01f);

			mGrid[CellIndex(x, z)].terrainHeight = slope + hills1 + hills2 + hills3 + ridge;
		}
	}

	// Deposit water at the high left side — large uphill reservoir
	AddWaterRect(5, 20, 60, mConfig.gridSizeZ - 20, 1.2f);

	// Additional water pool on a plateau in the middle
	AddWaterRect(100, 80, 160, 150, 0.8f);
}

void WulfNetWaterV5TerrainFlowTest::UpdateScenario(float /*dt*/)
{
	// Static scenario — water flows downhill on its own
}

// =====================================================================
// 3. Sheet Dam Break — Massive wall of water rushes across terrain
//    Grid: 500×250, cellSize 0.2 → 100m × 50m world (was 9.6m × 4.8m → ~10x)
// =====================================================================

void WulfNetWaterV5DamBreakTest::SetupScenario()
{
	SWE_LOG_INFO("[SCENARIO] DamBreak — 500x250 grid, tall water column behind dam wall");
	mConfig.gridSizeX       = 500;
	mConfig.gridSizeZ       = 250;
	mConfig.cellSize        = 0.2f;
	mConfig.damping         = 0.0015f;
	mConfig.viscosity       = 0.00015f;
	mConfig.substeps        = 2;
	mConfig.depthColorScale = 1.5f;
	mConfig.originX         = -(500 * 0.2f) / 2.0f;
	mConfig.originY         = 0.2f;
	mConfig.originZ         = -(250 * 0.2f) / 2.0f;

	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	mGrid.resize(total);

	// Terrain: gently undulating floor with end wall
	const float pi = 3.14159265f;
	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float fx = static_cast<float>(x) / mConfig.gridSizeX;
			float fz = static_cast<float>(z) / mConfig.gridSizeZ;
			float h = 0.1f * std::sin(fx * 2.0f * pi) * std::sin(fz * 2.0f * pi);
			mGrid[CellIndex(x, z)].terrainHeight = h;
		}
	}

	// End walls to contain water
	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		mGrid[CellIndex(mConfig.gridSizeX - 1, z)].terrainHeight = 1.0f;
		mGrid[CellIndex(mConfig.gridSizeX - 2, z)].terrainHeight = 0.5f;
	}

	// Tall column of water behind the dam
	AddWaterRect(5, 10, 80, mConfig.gridSizeZ - 10, 2.5f);

	// Dam wall at x=80 (thick, 3 cells wide)
	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		mGrid[CellIndex(80, z)].terrainHeight = 5.0f;
		mGrid[CellIndex(81, z)].terrainHeight = 4.0f;
		mGrid[CellIndex(82, z)].terrainHeight = 2.5f;
	}

	// Obstacle pillars downstream for water to flow around
	for (uint32_t pz = 50; pz < 200; pz += 50)
	{
		for (uint32_t px = 150; px < 400; px += 80)
		{
			for (uint32_t dz = 0; dz < 8; ++dz)
				for (uint32_t dx = 0; dx < 8; ++dx)
					if (px + dx < mConfig.gridSizeX && pz + dz < mConfig.gridSizeZ)
						mGrid[CellIndex(px + dx, pz + dz)].terrainHeight = 3.0f;
		}
	}
}

void WulfNetWaterV5DamBreakTest::UpdateScenario(float dt)
{
	mDamTimer += dt;

	// At t=1.5s, remove the dam wall by lowering the terrain
	if (!mDamReleased && mDamTimer >= 1.5f)
	{
		mDamReleased = true;
		WaterDiagnostics::LogEvent(SWE_LOG_CAT, "DAM BREAK at t=1.5s \u2014 terrain lowered at x=80-82");
		for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
		{
			mGrid[CellIndex(80, z)].terrainHeight = 0.0f;
			mGrid[CellIndex(81, z)].terrainHeight = 0.0f;
			mGrid[CellIndex(82, z)].terrainHeight = 0.0f;
		}
	}
}

// =====================================================================
// 4. Rain on Hills — Random drops on a large hilly terrain
//    Grid: 400×400, cellSize 0.22 → 88m × 88m world (was 8.1m → ~10.9x)
// =====================================================================

void WulfNetWaterV5RainHillsTest::SetupScenario()
{
	SWE_LOG_INFO("[SCENARIO] RainHills — 400x400 grid, random drops on hilly terrain");
	mConfig.gridSizeX       = 400;
	mConfig.gridSizeZ       = 400;
	mConfig.cellSize        = 0.22f;
	mConfig.damping         = 0.002f;
	mConfig.viscosity       = 0.0002f;
	mConfig.substeps        = 2;
	mConfig.depthColorScale = 0.6f;
	mConfig.originX         = -(400 * 0.22f) / 2.0f;
	mConfig.originY         = 0.2f;
	mConfig.originZ         = -(400 * 0.22f) / 2.0f;
	mRainInterval           = 0.02f;   // Very frequent rain for the larger grid

	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	mGrid.resize(total);

	// Complex multi-octave hillscape
	const float pi = 3.14159265f;
	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float fx = static_cast<float>(x) / mConfig.gridSizeX;
			float fz = static_cast<float>(z) / mConfig.gridSizeZ;

			float h = 0.5f;
			h += 1.2f * std::sin(fx * 3.0f * pi) * std::sin(fz * 3.0f * pi);
			h += 0.6f * std::sin(fx * 7.0f * pi + 1.0f) * std::sin(fz * 5.0f * pi + 0.7f);
			h += 0.3f * std::sin(fx * 13.0f * pi + 2.5f) * std::sin(fz * 11.0f * pi + 1.2f);

			mGrid[CellIndex(x, z)].terrainHeight = h;
		}
	}

	// Create several deep valleys (Gaussian depressions) to collect water
	struct Depression { float cx; float cz; float radius; float depth; };
	Depression deps[] = {
		{ 0.25f, 0.35f, 25.0f, 0.8f },
		{ 0.60f, 0.50f, 30.0f, 1.0f },
		{ 0.40f, 0.75f, 20.0f, 0.6f },
		{ 0.80f, 0.20f, 22.0f, 0.7f },
		{ 0.15f, 0.80f, 18.0f, 0.5f },
	};

	for (const auto &dep : deps)
	{
		for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
		{
			for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
			{
				float dx = (static_cast<float>(x) - dep.cx * mConfig.gridSizeX) / dep.radius;
				float dz = (static_cast<float>(z) - dep.cz * mConfig.gridSizeZ) / dep.radius;
				float pit = -dep.depth * std::exp(-(dx * dx + dz * dz));
				if (pit < 0.0f)
					mGrid[CellIndex(x, z)].terrainHeight += pit;
			}
		}
	}
}

void WulfNetWaterV5RainHillsTest::UpdateScenario(float dt)
{
	mRainTimer += dt;

	// Spawn random drops — more drops for the larger grid
	while (mRainTimer >= mRainInterval && mTotalDrops < 3000)
	{
		mRainTimer -= mRainInterval;
		mTotalDrops++;

		// Deterministic-ish random from drop count
		uint32_t seed = static_cast<uint32_t>(mTotalDrops * 73856093u);
		uint32_t rx = (seed % (mConfig.gridSizeX - 16)) + 8;
		seed = seed * 19349663u + 1;
		uint32_t rz = (seed % (mConfig.gridSizeZ - 16)) + 8;

		// Larger drops (radius 5 instead of 2) for visible impact
		AddWaterDrop(rx, rz, 5, 0.08f);
	}
}

// =====================================================================
// 5. Tsunami Coast — Massive wave crashes onto a coastal landscape
//    Grid: 500×300, cellSize 0.3 → 150m × 90m world
// =====================================================================

void WulfNetWaterV5TsunamiTest::SetupScenario()
{
	SWE_LOG_INFO("[SCENARIO] Tsunami — 500x300 grid, deep ocean to coastline");
	mConfig.gridSizeX       = 500;
	mConfig.gridSizeZ       = 300;
	mConfig.cellSize        = 0.3f;
	mConfig.damping         = 0.001f;
	mConfig.viscosity       = 0.0001f;
	mConfig.substeps        = 2;
	mConfig.depthColorScale = 2.0f;
	mConfig.originX         = -(500 * 0.3f) / 2.0f;
	mConfig.originY         = 0.0f;
	mConfig.originZ         = -(300 * 0.3f) / 2.0f;

	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	mGrid.resize(total);

	// Terrain: deep ocean → shallow shelf → beach → coastal hills
	const float pi = 3.14159265f;
	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float fx = static_cast<float>(x) / mConfig.gridSizeX;
			float fz = static_cast<float>(z) / mConfig.gridSizeZ;

			float h;
			if (fx < 0.6f)
			{
				// Ocean floor: deep on left, rising towards beach
				h = -3.0f + fx * 5.0f;
				// Underwater ridges for wave refraction
				h += 0.2f * std::sin(fx * 8.0f * pi) * std::sin(fz * 4.0f * pi);
			}
			else if (fx < 0.7f)
			{
				// Beach: gentle slope from sea level to low coast
				float t = (fx - 0.6f) / 0.1f;
				h = t * 0.5f;
			}
			else
			{
				// Coastal hills — varied elevation
				float t = (fx - 0.7f) / 0.3f;
				h = 0.5f + t * 2.0f;
				h += 0.5f * std::sin(fz * 4.0f * pi) * std::sin(t * 3.0f * pi);
			}

			mGrid[CellIndex(x, z)].terrainHeight = h;
		}
	}

	// Fill ocean with water up to sea level (h = 0)
	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float terrain = mGrid[CellIndex(x, z)].terrainHeight;
			if (terrain < 0.0f)
				mGrid[CellIndex(x, z)].waterHeight = -terrain;
		}
}

void WulfNetWaterV5TsunamiTest::UpdateScenario(float dt)
{
	mWaveTimer += dt;

	// At t=2s generate the tsunami \u2014 a massive water pulse on the deep ocean side
	if (!mWaveTriggered && mWaveTimer >= 2.0f)
	{
		mWaveTriggered = true;
		WaterDiagnostics::LogEvent(SWE_LOG_CAT, "TSUNAMI triggered at t=2.0s \u2014 3m pulse injected");
		const float pi = 3.14159265f;
		for (uint32_t z = 20; z < mConfig.gridSizeZ - 20; ++z)
		{
			for (uint32_t x = 10; x < 60; ++x)
			{
				float dist = static_cast<float>(x - 10) / 50.0f;
				float profile = 0.5f * (1.0f + std::cos(dist * pi));
				mGrid[CellIndex(x, z)].waterHeight += 3.0f * profile;
			}
		}
	}
}

// =====================================================================
// 6. River Canyon — Water flows through a long winding canyon
//    Grid: 150×600, cellSize 0.2 → 30m × 120m world
// =====================================================================

void WulfNetWaterV5RiverCanyonTest::SetupScenario()
{
	SWE_LOG_INFO("[SCENARIO] RiverCanyon — 150x600 grid, winding canyon with reservoir");
	mConfig.gridSizeX       = 150;
	mConfig.gridSizeZ       = 600;
	mConfig.cellSize        = 0.2f;
	mConfig.damping         = 0.003f;
	mConfig.viscosity       = 0.0002f;
	mConfig.substeps        = 2;
	mConfig.depthColorScale = 0.8f;
	mConfig.originX         = -(150 * 0.2f) / 2.0f;
	mConfig.originY         = 0.0f;
	mConfig.originZ         = -(600 * 0.2f) / 2.0f;

	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	mGrid.resize(total);

	// Terrain: elevated plateau with a winding canyon carved through it
	const float pi = 3.14159265f;
	float halfX = mConfig.gridSizeX * 0.5f;

	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		float fz = static_cast<float>(z) / mConfig.gridSizeZ;

		// Canyon centre-line wanders sinusoidally
		float canyonCentre = halfX + 25.0f * std::sin(fz * 3.0f * pi)
		                           + 15.0f * std::sin(fz * 7.0f * pi + 1.0f);
		float canyonWidth  = 20.0f + 5.0f * std::sin(fz * 5.0f * pi);

		// Gentle downhill slope (water flows from z=0 towards z=max)
		float baseSlope = 2.0f * (1.0f - fz);

		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float dist = std::abs(static_cast<float>(x) - canyonCentre);

			float h;
			if (dist < canyonWidth * 0.5f)
			{
				// Canyon floor
				h = baseSlope;
			}
			else if (dist < canyonWidth)
			{
				// Canyon walls — steep parabolic rise
				float t = (dist - canyonWidth * 0.5f) / (canyonWidth * 0.5f);
				h = baseSlope + t * t * 3.0f;
			}
			else
			{
				// Plateau above canyon
				h = baseSlope + 3.0f + 0.3f * std::sin(static_cast<float>(x) * 0.1f);
			}

			mGrid[CellIndex(x, z)].terrainHeight = h;
		}
	}

	// Fill the top of the canyon with a large water reservoir
	for (uint32_t z = 0; z < 40; ++z)
	{
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float terrain = mGrid[CellIndex(x, z)].terrainHeight;
			if (terrain < 3.5f) // Only inside the canyon
				mGrid[CellIndex(x, z)].waterHeight = 3.5f - terrain;
		}
	}
}

void WulfNetWaterV5RiverCanyonTest::UpdateScenario(float dt)
{
	mSourceTimer += dt;

	// Continuously feed water at the canyon entrance (z ≈ 0)
	if (mSourceTimer >= 0.1f)
	{
		mSourceTimer -= 0.1f;
		float halfX = mConfig.gridSizeX * 0.5f;
		for (uint32_t z = 0; z < 5; ++z)
		{
			for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
			{
				float dist = std::abs(static_cast<float>(x) - halfX);
				if (dist < 15.0f)
					mGrid[CellIndex(x, z)].waterHeight += 0.05f;
			}
		}
	}
}

// =====================================================================
// 7. Caldera Lake — Volcanic crater lake with eruption pulses
//    Grid: 400×400, cellSize 0.25 → 100m × 100m world
// =====================================================================

void WulfNetWaterV5CalderaLakeTest::SetupScenario()
{
	SWE_LOG_INFO("[SCENARIO] CalderaLake — 400x400 grid, volcanic crater lake");
	mConfig.gridSizeX       = 400;
	mConfig.gridSizeZ       = 400;
	mConfig.cellSize        = 0.25f;
	mConfig.damping         = 0.0015f;
	mConfig.viscosity       = 0.0001f;
	mConfig.substeps        = 2;
	mConfig.depthColorScale = 1.5f;
	mConfig.originX         = -(400 * 0.25f) / 2.0f;
	mConfig.originY         = 0.0f;
	mConfig.originZ         = -(400 * 0.25f) / 2.0f;

	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	mGrid.resize(total);

	// Terrain: circular volcanic caldera with steep walls and central vent
	float cx = mConfig.gridSizeX * 0.5f;
	float cz = mConfig.gridSizeZ * 0.5f;
	float outerRadius = mConfig.gridSizeX * 0.45f;
	float innerRadius = mConfig.gridSizeX * 0.35f;
	float rimHeight   = 4.0f;
	float floorHeight = 0.0f;

	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float dx = static_cast<float>(x) - cx;
			float dz = static_cast<float>(z) - cz;
			float dist = std::sqrt(dx * dx + dz * dz);

			float h;
			if (dist > outerRadius)
			{
				// Outside crater — slopes down from rim
				float t = (dist - outerRadius) / (mConfig.gridSizeX * 0.05f);
				h = rimHeight * std::max(0.0f, 1.0f - t);
			}
			else if (dist > innerRadius)
			{
				// Crater wall — steep parabolic slope
				float t = (dist - innerRadius) / (outerRadius - innerRadius);
				h = floorHeight + (rimHeight - floorHeight) * t * t;
			}
			else
			{
				// Crater floor — mostly flat with a central volcanic cone
				float centralDist = dist / innerRadius;
				h = floorHeight;
				if (centralDist < 0.15f)
					h += 0.8f * (1.0f - centralDist / 0.15f);
				// Subtle radial undulations
				float angle = std::atan2(dz, dx);
				h += 0.1f * std::sin(angle * 5.0f) * (1.0f - centralDist);
			}

			mGrid[CellIndex(x, z)].terrainHeight = h;
		}
	}

	// Fill crater with water up to a moderate level
	float waterLevel = 1.5f;
	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float terrain = mGrid[CellIndex(x, z)].terrainHeight;
			if (terrain < waterLevel)
				mGrid[CellIndex(x, z)].waterHeight = waterLevel - terrain;
		}
}

void WulfNetWaterV5CalderaLakeTest::UpdateScenario(float dt)
{
	mEruptionTimer += dt;

	// Periodic eruptions every 3 seconds \u2014 up to 15 pulses
	if (mEruptionTimer >= 3.0f && mEruptionCount < 15)
	{
		mEruptionTimer -= 3.0f;
		mEruptionCount++;
		WaterDiagnostics::LogEvent(SWE_LOG_CAT,
			"Eruption #" + std::to_string(mEruptionCount) + " \u2014 central pulse r=20 h=1.5");

		// Central eruption: massive water pulse at the vent
		uint32_t cx = mConfig.gridSizeX / 2;
		uint32_t cz = mConfig.gridSizeZ / 2;
		AddWaterDrop(cx, cz, 20, 1.5f);

		// Every 3rd eruption, add secondary off-centre pulses
		if (mEruptionCount % 3 == 0)
		{
			AddWaterDrop(cx - 30, cz + 20, 12, 0.8f);
			AddWaterDrop(cx + 25, cz - 15, 12, 0.8f);
		}
	}
}

// =====================================================================
// 8. Flood Valley — Dam break floods a valley with obstacles
//    Grid: 500×300, cellSize 0.2 → 100m × 60m world
// =====================================================================

void WulfNetWaterV5FloodValleyTest::SetupScenario()
{
	SWE_LOG_INFO("[SCENARIO] FloodValley — 500x300 grid, dam break into building grid");
	mConfig.gridSizeX       = 500;
	mConfig.gridSizeZ       = 300;
	mConfig.cellSize        = 0.2f;
	mConfig.damping         = 0.002f;
	mConfig.viscosity       = 0.00015f;
	mConfig.substeps        = 2;
	mConfig.depthColorScale = 1.5f;
	mConfig.originX         = -(500 * 0.2f) / 2.0f;
	mConfig.originY         = 0.0f;
	mConfig.originZ         = -(300 * 0.2f) / 2.0f;

	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	mGrid.resize(total);

	// Terrain: V-shaped valley with gently sloped floor
	float centreZ = mConfig.gridSizeZ * 0.5f;
	const float pi = 3.14159265f;

	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float fx = static_cast<float>(x) / mConfig.gridSizeX;
			float distFromCentre = std::abs(static_cast<float>(z) - centreZ) / centreZ;

			// Valley floor slopes gently downhill from left to right
			float slope = 2.0f * (1.0f - fx);

			// Valley walls rise steeply away from centre
			float walls = 3.0f * distFromCentre * distFromCentre;

			// Natural undulation
			float noise = 0.1f * std::sin(fx * 10.0f * pi)
			                    * std::sin(distFromCentre * 6.0f * pi);

			mGrid[CellIndex(x, z)].terrainHeight = slope + walls + noise;
		}
	}

	// "Buildings" — raised terrain blocks scattered in the valley floor
	auto addBuilding = [&](uint32_t bx, uint32_t bz, uint32_t w, uint32_t d, float h)
	{
		for (uint32_t dz = 0; dz < d && bz + dz < mConfig.gridSizeZ; ++dz)
			for (uint32_t dx = 0; dx < w && bx + dx < mConfig.gridSizeX; ++dx)
				mGrid[CellIndex(bx + dx, bz + dz)].terrainHeight = h;
	};

	// Rows of buildings in the valley — the flood must flow around them
	addBuilding(150, 120, 10, 15, 4.0f);
	addBuilding(150, 160, 10, 15, 4.0f);
	addBuilding(180, 110, 12, 20, 3.5f);
	addBuilding(180, 155,  8, 12, 3.5f);
	addBuilding(180, 175, 12, 18, 3.5f);
	addBuilding(210, 125, 10, 10, 4.0f);
	addBuilding(210, 145, 15, 15, 4.0f);
	addBuilding(210, 170, 10, 12, 3.5f);
	addBuilding(240, 115,  8, 18, 4.5f);
	addBuilding(240, 150, 10, 10, 3.5f);
	addBuilding(240, 175, 12, 14, 4.0f);
	addBuilding(270, 130, 10, 15, 3.5f);
	addBuilding(270, 160, 12, 12, 4.0f);
	addBuilding(300, 120, 10, 20, 3.8f);
	addBuilding(300, 155,  8, 15, 3.5f);
	addBuilding(300, 180, 10, 10, 4.0f);
	addBuilding(330, 135, 12, 12, 3.5f);
	addBuilding(330, 165, 10, 15, 4.0f);

	// Reservoir behind a thick dam at x=60
	AddWaterRect(5, 100, 60, 200, 4.0f);
	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		mGrid[CellIndex(60, z)].terrainHeight = 6.0f;
		mGrid[CellIndex(61, z)].terrainHeight = 5.0f;
		mGrid[CellIndex(62, z)].terrainHeight = 3.5f;
	}
}

void WulfNetWaterV5FloodValleyTest::UpdateScenario(float dt)
{
	mFloodTimer += dt;

	// At t=1.5s, break the dam \u2014 water rushes through the valley
	if (!mFloodReleased && mFloodTimer >= 1.5f)
	{
		mFloodReleased = true;
		WaterDiagnostics::LogEvent(SWE_LOG_CAT, "FLOOD RELEASE at t=1.5s \u2014 dam at x=60-62 removed");
		for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
		{
			mGrid[CellIndex(60, z)].terrainHeight = 0.0f;
			mGrid[CellIndex(61, z)].terrainHeight = 0.0f;
			mGrid[CellIndex(62, z)].terrainHeight = 0.0f;
		}
	}
}
