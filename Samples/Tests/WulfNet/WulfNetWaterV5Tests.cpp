// SPDX-License-Identifier: MIT
// WulfNet Water Physics V5 — Implementation
//
// 2D Shallow Water Equations (SWE) solver rendered as a triangle mesh.
// Water is a height-field sheet that flows over terrain under gravity.

#include <Samples.h>

#include "WulfNetWaterV5Tests.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>

#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

// =====================================================================
// RTTI Registration
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV5Base)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV5Base, Test)
}

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

void WulfNetWaterV5Base::Initialize()
{
	WulfNet::SystemMonitor::Get().Initialize();
	mLastFPSTime = std::chrono::high_resolution_clock::now();

	CreateFloor();

	// Let derived test configure grid size, terrain, initial water, etc.
	SetupScenario();

	// Allocate grid
	uint32_t totalCells = mConfig.gridSizeX * mConfig.gridSizeZ;
	mGrid.resize(totalCells);
	mGridTemp.resize(totalCells);

	// If derived hasn't filled terrain yet, default to flat
	bool anyTerrain = false;
	for (uint32_t i = 0; i < totalCells && !anyTerrain; ++i)
		anyTerrain = mGrid[i].terrainHeight != 0.0f;
	if (!anyTerrain)
		SetTerrainFlat(0.0f);
}

// =====================================================================
// Terrain helpers
// =====================================================================

void WulfNetWaterV5Base::SetTerrainFlat(float height)
{
	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	if (mGrid.size() < total)
		mGrid.resize(total);
	for (uint32_t i = 0; i < total; ++i)
		mGrid[i].terrainHeight = height;
}

void WulfNetWaterV5Base::SetTerrainSlope(float startH, float endH, bool alongX)
{
	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	if (mGrid.size() < total)
		mGrid.resize(total);

	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float t = alongX
				? static_cast<float>(x) / std::max(1u, mConfig.gridSizeX - 1)
				: static_cast<float>(z) / std::max(1u, mConfig.gridSizeZ - 1);
			mGrid[CellIndex(x, z)].terrainHeight = startH + (endH - startH) * t;
		}
	}
}

void WulfNetWaterV5Base::SetTerrainBowl(float rimH, float centerH)
{
	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	if (mGrid.size() < total)
		mGrid.resize(total);

	float cx = static_cast<float>(mConfig.gridSizeX - 1) * 0.5f;
	float cz = static_cast<float>(mConfig.gridSizeZ - 1) * 0.5f;
	float maxDist = std::sqrt(cx * cx + cz * cz);

	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float dx = static_cast<float>(x) - cx;
			float dz = static_cast<float>(z) - cz;
			float d = std::sqrt(dx * dx + dz * dz) / maxDist; // 0..1
			mGrid[CellIndex(x, z)].terrainHeight = centerH + (rimH - centerH) * d * d;
		}
	}
}

void WulfNetWaterV5Base::SetTerrainHills(float baseH, float amp, float freqX, float freqZ)
{
	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	if (mGrid.size() < total)
		mGrid.resize(total);

	const float pi = 3.14159265f;
	for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
	{
		for (uint32_t x = 0; x < mConfig.gridSizeX; ++x)
		{
			float fx = std::sin(static_cast<float>(x) * freqX * pi / mConfig.gridSizeX);
			float fz = std::sin(static_cast<float>(z) * freqZ * pi / mConfig.gridSizeZ);
			mGrid[CellIndex(x, z)].terrainHeight = baseH + amp * fx * fz;
		}
	}
}

void WulfNetWaterV5Base::SetTerrainAt(uint32_t x, uint32_t z, float height)
{
	if (x < mConfig.gridSizeX && z < mConfig.gridSizeZ)
	{
		uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
		if (mGrid.size() < total)
			mGrid.resize(total);
		mGrid[CellIndex(x, z)].terrainHeight = height;
	}
}

// =====================================================================
// Water source helpers
// =====================================================================

void WulfNetWaterV5Base::AddWaterRect(uint32_t x0, uint32_t z0,
                                       uint32_t x1, uint32_t z1, float depth)
{
	x1 = std::min(x1, mConfig.gridSizeX);
	z1 = std::min(z1, mConfig.gridSizeZ);
	for (uint32_t z = z0; z < z1; ++z)
		for (uint32_t x = x0; x < x1; ++x)
			mGrid[CellIndex(x, z)].waterHeight += depth;
}

void WulfNetWaterV5Base::AddWaterDisk(uint32_t cx, uint32_t cz,
                                       uint32_t radius, float depth)
{
	uint32_t r2 = radius * radius;
	uint32_t xMin = (cx >= radius) ? cx - radius : 0;
	uint32_t xMax = std::min(cx + radius + 1, mConfig.gridSizeX);
	uint32_t zMin = (cz >= radius) ? cz - radius : 0;
	uint32_t zMax = std::min(cz + radius + 1, mConfig.gridSizeZ);

	for (uint32_t z = zMin; z < zMax; ++z)
	{
		for (uint32_t x = xMin; x < xMax; ++x)
		{
			uint32_t dx = (x > cx) ? x - cx : cx - x;
			uint32_t dz = (z > cz) ? z - cz : cz - z;
			if (dx * dx + dz * dz <= r2)
				mGrid[CellIndex(x, z)].waterHeight += depth;
		}
	}
}

void WulfNetWaterV5Base::AddWaterDrop(uint32_t cx, uint32_t cz,
                                       uint32_t radius, float peakDepth)
{
	uint32_t r2 = radius * radius;
	float fr = static_cast<float>(radius);
	uint32_t xMin = (cx >= radius) ? cx - radius : 0;
	uint32_t xMax = std::min(cx + radius + 1, mConfig.gridSizeX);
	uint32_t zMin = (cz >= radius) ? cz - radius : 0;
	uint32_t zMax = std::min(cz + radius + 1, mConfig.gridSizeZ);

	for (uint32_t z = zMin; z < zMax; ++z)
	{
		for (uint32_t x = xMin; x < xMax; ++x)
		{
			uint32_t dx = (x > cx) ? x - cx : cx - x;
			uint32_t dz = (z > cz) ? z - cz : cz - z;
			uint32_t d2 = dx * dx + dz * dz;
			if (d2 <= r2)
			{
				// Smooth cosine-bell drop profile
				float d = std::sqrt(static_cast<float>(d2)) / fr;
				float profile = 0.5f * (1.0f + std::cos(d * 3.14159265f));
				mGrid[CellIndex(x, z)].waterHeight += peakDepth * profile;
			}
		}
	}
}

// =====================================================================
// Shallow Water Equations — Pipe-model solver
// =====================================================================
//
// We use the "virtual pipe" formulation which is stable and simple:
//   For each cell, compute flux to each of 4 neighbours based on the
//   difference in water surface heights (terrain + water).
//   Flux is accelerated by gravity proportional to the height difference.
//   Then update water heights from net flux and velocities from flux.
//
// This naturally handles: flow over terrain, pooling, wave propagation,
// and ripples.  The pipe model is equivalent to the linearised SWE for
// small displacements and correctly captures dam-break fronts.

void WulfNetWaterV5Base::StepSWE(float dt)
{
	const uint32_t NX = mConfig.gridSizeX;
	const uint32_t NZ = mConfig.gridSizeZ;
	const float g     = mConfig.gravity;
	const float cs    = mConfig.cellSize;
	const float damp  = 1.0f - mConfig.damping;
	const float visc  = mConfig.viscosity;
	const uint32_t total = NX * NZ;

	// Ensure temp buffer is correctly sized (reused across frames — no allocation)
	if (mGridTemp.size() != total)
		mGridTemp.resize(total);

	// Snapshot current state into temp for read-only access during Phase 1.
	// We use memcpy for POD data — significantly faster than vector assignment
	// which invokes element-wise copy constructors.
	std::memcpy(mGridTemp.data(), mGrid.data(), total * sizeof(WaterCell));

	// ---------- Phase 1: Compute flux-driven velocity update ----------
	// Interior cells only (1..NX-2, 1..NZ-2) — boundary is handled separately.
	for (uint32_t z = 1; z + 1 < NZ; ++z)
	{
		const uint32_t rowOff = z * NX;
		for (uint32_t x = 1; x + 1 < NX; ++x)
		{
			const uint32_t idx = rowOff + x;
			const WaterCell &src = mGridTemp[idx];
			float h = src.waterHeight;

			if (h < 1e-6f)
			{
				mGrid[idx].vx = 0.0f;
				mGrid[idx].vz = 0.0f;
				continue;
			}

			float surfH = src.terrainHeight + h;

			// Direct neighbour access (no bounds check needed — we skip edges)
			const WaterCell &nL = mGridTemp[idx - 1];
			const WaterCell &nR = mGridTemp[idx + 1];
			const WaterCell &nB = mGridTemp[idx - NX];
			const WaterCell &nF = mGridTemp[idx + NX];

			float fluxL = g * (surfH - (nL.terrainHeight + nL.waterHeight)) / cs;
			float fluxR = g * (surfH - (nR.terrainHeight + nR.waterHeight)) / cs;
			float fluxB = g * (surfH - (nB.terrainHeight + nB.waterHeight)) / cs;
			float fluxF = g * (surfH - (nF.terrainHeight + nF.waterHeight)) / cs;

			float newVx = (src.vx + (fluxR - fluxL) * 0.5f * dt) * damp;
			float newVz = (src.vz + (fluxF - fluxB) * 0.5f * dt) * damp;

			// Viscosity: blend toward neighbour average (Laplacian smoothing)
			if (visc > 0.0f)
			{
				float avgVx = (nL.vx + nR.vx + nB.vx + nF.vx) * 0.25f;
				float avgVz = (nL.vz + nR.vz + nB.vz + nF.vz) * 0.25f;
				newVx += visc * (avgVx - newVx);
				newVz += visc * (avgVz - newVz);
			}

			mGrid[idx].vx = newVx;
			mGrid[idx].vz = newVz;
		}
	}

	// Handle edge rows/columns: zero velocity for boundary cells
	for (uint32_t x = 0; x < NX; ++x)
	{
		mGrid[x].vx = 0.0f;            mGrid[x].vz = 0.0f;
		mGrid[(NZ - 1) * NX + x].vx = 0.0f; mGrid[(NZ - 1) * NX + x].vz = 0.0f;
	}
	for (uint32_t z = 1; z + 1 < NZ; ++z)
	{
		mGrid[z * NX].vx = 0.0f;         mGrid[z * NX].vz = 0.0f;
		mGrid[z * NX + NX - 1].vx = 0.0f; mGrid[z * NX + NX - 1].vz = 0.0f;
	}

	// ---------- Phase 1.5: Perlin noise velocity perturbation ----------
	if (mConfig.noiseEnabled && mConfig.noiseVelStrength > 0.0f)
	{
		const float freq  = mConfig.noiseFrequency;
		const float speed = mConfig.noiseSpeed;
		const float str   = mConfig.noiseVelStrength * dt;
		const int   oct   = mConfig.noiseOctaves;
		const float lac   = mConfig.noiseLacunarity;
		const float per   = mConfig.noisePersistence;
		const float t     = mNoiseTime;

		constexpr uint32_t DS = 4;
		uint32_t dsNX = (NX + DS - 1) / DS + 1;
		uint32_t dsNZ = (NZ + DS - 1) / DS + 1;

		std::vector<float> noiseBufVx(dsNX * dsNZ);
		std::vector<float> noiseBufVz(dsNX * dsNZ);

		for (uint32_t dz = 0; dz < dsNZ; ++dz)
		{
			float fz = static_cast<float>(dz * DS) * freq / NZ;
			for (uint32_t dx = 0; dx < dsNX; ++dx)
			{
				float fx = static_cast<float>(dx * DS) * freq / NX;
				uint32_t idx = dz * dsNX + dx;
				noiseBufVx[idx] = mNoise.FBM3D(fx, fz, t * speed,
				                                oct, lac, per, str, 1.0f);
				noiseBufVz[idx] = mNoise.FBM3D(fx + 17.3f, fz + 31.7f, t * speed + 5.0f,
				                                oct, lac, per, str, 1.0f);
			}
		}

		float invDS = 1.0f / static_cast<float>(DS);
		for (uint32_t z = 1; z + 1 < NZ; ++z)
		{
			float cz = static_cast<float>(z) * invDS;
			uint32_t cz0 = static_cast<uint32_t>(cz);
			uint32_t cz1 = std::min(cz0 + 1, dsNZ - 1);
			float fz = cz - cz0;

			for (uint32_t x = 1; x + 1 < NX; ++x)
			{
				WaterCell &cell = mGrid[CellIndex(x, z)];
				if (cell.waterHeight < 1e-5f) continue;

				float cx = static_cast<float>(x) * invDS;
				uint32_t cx0 = static_cast<uint32_t>(cx);
				uint32_t cx1 = std::min(cx0 + 1, dsNX - 1);
				float fx = cx - cx0;

				float nVx = (1-fx)*(1-fz) * noiseBufVx[cz0*dsNX+cx0]
				          + fx*(1-fz)     * noiseBufVx[cz0*dsNX+cx1]
				          + (1-fx)*fz     * noiseBufVx[cz1*dsNX+cx0]
				          + fx*fz         * noiseBufVx[cz1*dsNX+cx1];

				float nVz = (1-fx)*(1-fz) * noiseBufVz[cz0*dsNX+cx0]
				          + fx*(1-fz)     * noiseBufVz[cz0*dsNX+cx1]
				          + (1-fx)*fz     * noiseBufVz[cz1*dsNX+cx0]
				          + fx*fz         * noiseBufVz[cz1*dsNX+cx1];

				float depthScale = std::min(1.0f, cell.waterHeight * 10.0f);
				cell.vx += nVx * depthScale;
				cell.vz += nVz * depthScale;
			}
		}
	}

	// ---------- Phase 2: Conservative water transport ----------
	// Snapshot the velocity-updated grid for stable reads during transport
	std::memcpy(mGridTemp.data(), mGrid.data(), total * sizeof(WaterCell));

	for (uint32_t z = 1; z + 1 < NZ; ++z)
	{
		const uint32_t rowOff = z * NX;
		for (uint32_t x = 1; x + 1 < NX; ++x)
		{
			const uint32_t idx = rowOff + x;
			const WaterCell &src = mGridTemp[idx];
			if (src.waterHeight < 1e-6f) continue;

			float surfH = src.terrainHeight + src.waterHeight;

			// Outflow to 4 neighbours based on surface height difference
			float outflow[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			const uint32_t nbIdx[4] = { idx - 1, idx + 1, idx - NX, idx + NX };
			float totalOut = 0.0f;

			for (int d = 0; d < 4; ++d)
			{
				const WaterCell &nb = mGridTemp[nbIdx[d]];
				float dh = surfH - (nb.terrainHeight + nb.waterHeight);
				if (dh > 0.0f)
				{
					outflow[d] = dh * 0.25f;
					totalOut += outflow[d];
				}
			}

			// Clamp to avoid removing more water than exists
			if (totalOut > src.waterHeight)
			{
				float scale = src.waterHeight / totalOut;
				for (int d = 0; d < 4; ++d)
					outflow[d] *= scale;
				totalOut = src.waterHeight;
			}

			mGrid[idx].waterHeight -= totalOut;
			for (int d = 0; d < 4; ++d)
			{
				if (outflow[d] > 0.0f)
					mGrid[nbIdx[d]].waterHeight += outflow[d];
			}
		}
	}

	ApplyBoundary();
}

void WulfNetWaterV5Base::ApplyBoundary()
{
	const uint32_t NX = mConfig.gridSizeX;
	const uint32_t NZ = mConfig.gridSizeZ;

	// Reflective boundaries — zero normal velocity, keep water
	for (uint32_t x = 0; x < NX; ++x)
	{
		mGrid[CellIndex(x, 0)].vz       = std::min(0.0f, mGrid[CellIndex(x, 0)].vz);
		mGrid[CellIndex(x, NZ - 1)].vz  = std::max(0.0f, mGrid[CellIndex(x, NZ - 1)].vz);
	}
	for (uint32_t z = 0; z < NZ; ++z)
	{
		mGrid[CellIndex(0, z)].vx       = std::min(0.0f, mGrid[CellIndex(0, z)].vx);
		mGrid[CellIndex(NX - 1, z)].vx  = std::max(0.0f, mGrid[CellIndex(NX - 1, z)].vx);
	}

	// Clamp negative water heights (numerical safety)
	uint32_t total = NX * NZ;
	for (uint32_t i = 0; i < total; ++i)
	{
		if (mGrid[i].waterHeight < 0.0f)
			mGrid[i].waterHeight = 0.0f;
	}
}

// =====================================================================
// PrePhysicsUpdate — Solver + Render
// =====================================================================

void WulfNetWaterV5Base::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	// FPS tracking
	mFrameCount++;
	auto now = std::chrono::high_resolution_clock::now();
	float elapsed = std::chrono::duration<float>(now - mLastFPSTime).count();
	if (elapsed >= 1.0f)
	{
		mCurrentFPS  = static_cast<float>(mFrameCount) / elapsed;
		mFrameTimeMs = (elapsed / mFrameCount) * 1000.0f;
		mFrameCount  = 0;
		mLastFPSTime = now;
	}

	mStatsTimer += inParams.mDeltaTime;
	if (mStatsTimer >= 0.5f)
	{
		WulfNet::SystemMonitor::Get().Update();
		mStatsTimer = 0.0f;
	}

	// Derived per-frame logic (adding drops, releasing dams, etc.)
	UpdateScenario(inParams.mDeltaTime);

	// Advance noise time for animated ripples
	mNoiseTime += inParams.mDeltaTime;

	// Step the SWE solver
	auto simStart = std::chrono::high_resolution_clock::now();
	float subDt = inParams.mDeltaTime / static_cast<float>(mConfig.substeps);
	for (uint32_t s = 0; s < mConfig.substeps; ++s)
		StepSWE(subDt);
	auto simEnd = std::chrono::high_resolution_clock::now();
	mSimTimeMs = std::chrono::duration<float, std::milli>(simEnd - simStart).count();

	// Compute total water volume (for conservation display)
	mTotalWater = 0.0f;
	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	for (uint32_t i = 0; i < total; ++i)
		mTotalWater += mGrid[i].waterHeight;

	// Render the water sheet
	DrawSheet();
}

// =====================================================================
// Rendering — Draw the water surface as a coloured triangle mesh
// =====================================================================

Color WulfNetWaterV5Base::DepthColor(float depth) const
{
	float t = std::min(1.0f, depth / mConfig.depthColorScale);
	// Use a non-linear ramp: shallow water transitions faster to show depth
	t = t * t * (3.0f - 2.0f * t); // smoothstep for perceptually nicer gradient

	auto lerp = [](uint8_t a, uint8_t b, float t) -> uint8_t
	{
		return static_cast<uint8_t>(a + (b - a) * t);
	};

	return Color(
		lerp(mConfig.shallowColor.r, mConfig.deepColor.r, t),
		lerp(mConfig.shallowColor.g, mConfig.deepColor.g, t),
		lerp(mConfig.shallowColor.b, mConfig.deepColor.b, t),
		lerp(mConfig.shallowColor.a, mConfig.deepColor.a, t));
}

void WulfNetWaterV5Base::DrawSheet()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	const uint32_t NX = mConfig.gridSizeX;
	const uint32_t NZ = mConfig.gridSizeZ;
	const float cs     = mConfig.cellSize;
	const float ox     = mConfig.originX;
	const float oy     = mConfig.originY;
	const float oz     = mConfig.originZ;
	const float minH   = mConfig.minWaterDraw;

	// Adaptive render stride for large grids — reduces triangle count
	// while preserving visual quality.  Terrain can be coarser than water.
	uint32_t maxDim = std::max(NX, NZ);
	uint32_t tS = 1;  // Terrain stride
	uint32_t wS = 1;  // Water stride
	if (maxDim > 250) { tS = 2; }
	if (maxDim > 450) { tS = 3; wS = 2; }

	// Directional light for shading (from upper-right, normalised)
	const Vec3 lightDir = Vec3(0.4f, 0.8f, 0.3f).Normalized();
	const float ambientTerrain = 0.35f;   // Minimum brightness for terrain
	const float ambientWater   = 0.40f;   // Minimum brightness for water

	// ---- Terrain mesh with normal-based lighting ----
	for (uint32_t z = 0; z + tS < NZ; z += tS)
	{
		for (uint32_t x = 0; x + tS < NX; x += tS)
		{
			uint32_t x1 = std::min(x + tS, NX - 1);
			uint32_t z1 = std::min(z + tS, NZ - 1);

			const WaterCell &c00 = mGrid[CellIndex(x,  z)];
			const WaterCell &c10 = mGrid[CellIndex(x1, z)];
			const WaterCell &c01 = mGrid[CellIndex(x,  z1)];
			const WaterCell &c11 = mGrid[CellIndex(x1, z1)];

			float maxWater = std::max({ c00.waterHeight, c10.waterHeight,
			                             c01.waterHeight, c11.waterHeight });

			float wx0 = ox + x  * cs;
			float wx1 = ox + x1 * cs;
			float wz0 = oz + z  * cs;
			float wz1 = oz + z1 * cs;

			RVec3 t00(wx0, oy + c00.terrainHeight, wz0);
			RVec3 t10(wx1, oy + c10.terrainHeight, wz0);
			RVec3 t01(wx0, oy + c01.terrainHeight, wz1);
			RVec3 t11(wx1, oy + c11.terrainHeight, wz1);

			// Base terrain color varies with height for visual depth
			float avgTerrainH = (c00.terrainHeight + c10.terrainHeight +
			                     c01.terrainHeight + c11.terrainHeight) * 0.25f;
			// Height-based color: low = dark earthy brown, high = bright green
			float ht = std::min(1.0f, std::max(0.0f, avgTerrainH * 0.15f));
			uint8_t tR, tG, tB;
			if (maxWater > minH) {
				// Submerged terrain: dark blue-green tint
				tR = static_cast<uint8_t>(30  + ht * 15);
				tG = static_cast<uint8_t>(60  + ht * 30);
				tB = static_cast<uint8_t>(35  + ht * 15);
			} else {
				// Dry terrain: earthy green-brown with height variation
				tR = static_cast<uint8_t>(55  + ht * 35);
				tG = static_cast<uint8_t>(100 + ht * 60);
				tB = static_cast<uint8_t>(35  + ht * 25);
			}

			// Compute face normals and apply directional lighting
			// Triangle 1: t00, t10, t01
			Vec3 e1_a = Vec3(t10 - t00);
			Vec3 e2_a = Vec3(t01 - t00);
			Vec3 n1 = e1_a.Cross(e2_a);
			if (n1.LengthSq() > 1e-12f) n1 = n1.Normalized();
			float shade1 = ambientTerrain + (1.0f - ambientTerrain) * std::max(0.0f, n1.Dot(lightDir));

			// Triangle 2: t10, t11, t01
			Vec3 e1_b = Vec3(t11 - t10);
			Vec3 e2_b = Vec3(t01 - t10);
			Vec3 n2 = e1_b.Cross(e2_b);
			if (n2.LengthSq() > 1e-12f) n2 = n2.Normalized();
			float shade2 = ambientTerrain + (1.0f - ambientTerrain) * std::max(0.0f, n2.Dot(lightDir));

			Color tCol1(static_cast<uint8_t>(tR * shade1),
			            static_cast<uint8_t>(tG * shade1),
			            static_cast<uint8_t>(tB * shade1), 255);
			Color tCol2(static_cast<uint8_t>(tR * shade2),
			            static_cast<uint8_t>(tG * shade2),
			            static_cast<uint8_t>(tB * shade2), 255);

			mDebugRenderer->DrawTriangle(t00, t10, t01, tCol1);
			mDebugRenderer->DrawTriangle(t10, t11, t01, tCol2);
		}
	}

	// ---- Water surface mesh with normal-based lighting ----
	for (uint32_t z = 0; z + wS < NZ; z += wS)
	{
		for (uint32_t x = 0; x + wS < NX; x += wS)
		{
			uint32_t x1 = std::min(x + wS, NX - 1);
			uint32_t z1 = std::min(z + wS, NZ - 1);

			const WaterCell &c00 = mGrid[CellIndex(x,  z)];
			const WaterCell &c10 = mGrid[CellIndex(x1, z)];
			const WaterCell &c01 = mGrid[CellIndex(x,  z1)];
			const WaterCell &c11 = mGrid[CellIndex(x1, z1)];

			float maxWater = std::max({ c00.waterHeight, c10.waterHeight,
			                             c01.waterHeight, c11.waterHeight });
			if (maxWater <= minH) continue;

			float wx0 = ox + x  * cs;
			float wx1 = ox + x1 * cs;
			float wz0 = oz + z  * cs;
			float wz1 = oz + z1 * cs;

			float s00 = c00.terrainHeight + c00.waterHeight;
			float s10 = c10.terrainHeight + c10.waterHeight;
			float s01 = c01.terrainHeight + c01.waterHeight;
			float s11 = c11.terrainHeight + c11.waterHeight;

			// Perlin noise visual displacement — makes the surface shimmer
			// and ripple even when the underlying simulation is calm.
			if (mConfig.noiseEnabled && mConfig.noiseAmplitude > 0.0f)
			{
				float nFreq = mConfig.noiseFrequency * 1.5f;
				float nTime = mNoiseTime * mConfig.noiseSpeed;
				int   nOct  = mConfig.noiseOctaves;
				float nLac  = mConfig.noiseLacunarity;
				float nPer  = mConfig.noisePersistence;

				auto sampleNoise = [&](float wx, float wz, float depth) -> float
				{
					if (depth < minH) return 0.0f;
					float depthScale = std::min(1.0f, depth * 5.0f);
					float n = mNoise.FBM3D(wx * nFreq, wz * nFreq, nTime,
					                        nOct, nLac, nPer, mConfig.noiseAmplitude, 1.0f);
					return n * depthScale;
				};

				s00 += sampleNoise(wx0, wz0, c00.waterHeight);
				s10 += sampleNoise(wx1, wz0, c10.waterHeight);
				s01 += sampleNoise(wx0, wz1, c01.waterHeight);
				s11 += sampleNoise(wx1, wz1, c11.waterHeight);
			}

			RVec3 w00(wx0, oy + s00, wz0);
			RVec3 w10(wx1, oy + s10, wz0);
			RVec3 w01(wx0, oy + s01, wz1);
			RVec3 w11(wx1, oy + s11, wz1);

			float avgDepth = (c00.waterHeight + c10.waterHeight +
			                  c01.waterHeight + c11.waterHeight) * 0.25f;
			Color baseCol = DepthColor(avgDepth);

			// Compute face normal for specular highlight / directional shade
			Vec3 we1 = Vec3(w10 - w00);
			Vec3 we2 = Vec3(w01 - w00);
			Vec3 wn = we1.Cross(we2);
			if (wn.LengthSq() > 1e-12f) wn = wn.Normalized();
			float nDotL = std::max(0.0f, wn.Dot(lightDir));
			float shade = ambientWater + (1.0f - ambientWater) * nDotL;

			// Fresnel-like rim brightening: steeper viewing = more opaque
			float specPower = nDotL * nDotL * nDotL;  // Cheap specular approximation
			float specBoost = specPower * 40.0f;        // Bright highlight

			uint8_t sR = static_cast<uint8_t>(std::min(255.0f, baseCol.r * shade + specBoost));
			uint8_t sG = static_cast<uint8_t>(std::min(255.0f, baseCol.g * shade + specBoost));
			uint8_t sB = static_cast<uint8_t>(std::min(255.0f, baseCol.b * shade + specBoost));

			Color wCol(sR, sG, sB, baseCol.a);

			// Front and back faces for double-sided water rendering
			mDebugRenderer->DrawTriangle(w00, w10, w01, wCol);
			mDebugRenderer->DrawTriangle(w10, w11, w01, wCol);
			mDebugRenderer->DrawTriangle(w01, w10, w00, wCol);
			mDebugRenderer->DrawTriangle(w01, w11, w10, wCol);
		}
	}
#endif
}

// =====================================================================
// Status overlay
// =====================================================================

String WulfNetWaterV5Base::GetStatusString() const
{
	const WulfNet::SystemStats &sys = WulfNet::SystemMonitor::Get().GetStats();

	// Count wet cells — use v² comparison to avoid sqrt per cell
	uint32_t wetCells = 0;
	float maxDepth = 0.0f;
	float maxVel2 = 0.0f;
	uint32_t total = mConfig.gridSizeX * mConfig.gridSizeZ;
	for (uint32_t i = 0; i < total; ++i)
	{
		if (mGrid[i].waterHeight > mConfig.minWaterDraw)
			++wetCells;
		maxDepth = std::max(maxDepth, mGrid[i].waterHeight);
		float vel2 = mGrid[i].vx * mGrid[i].vx + mGrid[i].vz * mGrid[i].vz;
		if (vel2 > maxVel2) maxVel2 = vel2;
	}
	float maxVel = std::sqrt(maxVel2);  // Single sqrt at the end

	std::ostringstream oss;
	oss << std::fixed;

	oss << "FPS: " << std::setprecision(1) << mCurrentFPS
	    << "  (" << std::setprecision(2) << mFrameTimeMs << " ms)\n";

	oss << std::setprecision(1);
	oss << "CPU: " << sys.cpuUsagePercent << "%  RAM: "
	    << WulfNet::FormatBytes(sys.processMemoryBytes) << "\n";
	if (sys.gpuUsageAvailable)
		oss << "GPU: " << sys.gpuUsagePercent << "%\n";

	oss << "\n";
	oss << "Grid: " << mConfig.gridSizeX << "x" << mConfig.gridSizeZ
	    << " (" << total << " cells)\n";
	oss << "Wet cells: " << wetCells
	    << " (" << std::setprecision(1) << (100.0f * wetCells / total) << "%)\n";
	oss << "Total water: " << std::setprecision(2) << mTotalWater
	    << " (volume units)\n";
	oss << "Max depth: " << std::setprecision(3) << maxDepth << " m\n";
	oss << "Max velocity: " << std::setprecision(2) << maxVel << " m/s\n";
	oss << "Substeps: " << mConfig.substeps << "\n";
	oss << "Sim: " << std::setprecision(2) << mSimTimeMs << " ms";

	return String(oss.str());
}

// =====================================================================
// 1. Ripple Pond — Point disturbances on a massive flat basin
//    Grid: 400×400, cellSize 0.2 → 80m × 80m world (was 8m × 8m → 10x)
// =====================================================================

void WulfNetWaterV5RipplePondTest::SetupScenario()
{
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
	}
}

// =====================================================================
// 2. Terrain Flow — Water on a large hilly landscape
//    Grid: 350×350, cellSize 0.25 → 87.5m × 87.5m world (was 8m → ~11x)
// =====================================================================

void WulfNetWaterV5TerrainFlowTest::SetupScenario()
{
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

	// At t=2s generate the tsunami — a massive water pulse on the deep ocean side
	if (!mWaveTriggered && mWaveTimer >= 2.0f)
	{
		mWaveTriggered = true;
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

	// Periodic eruptions every 3 seconds — up to 15 pulses
	if (mEruptionTimer >= 3.0f && mEruptionCount < 15)
	{
		mEruptionTimer -= 3.0f;
		mEruptionCount++;

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

	// At t=1.5s, break the dam — water rushes through the valley
	if (!mFloodReleased && mFloodTimer >= 1.5f)
	{
		mFloodReleased = true;
		for (uint32_t z = 0; z < mConfig.gridSizeZ; ++z)
		{
			mGrid[CellIndex(60, z)].terrainHeight = 0.0f;
			mGrid[CellIndex(61, z)].terrainHeight = 0.0f;
			mGrid[CellIndex(62, z)].terrainHeight = 0.0f;
		}
	}
}
