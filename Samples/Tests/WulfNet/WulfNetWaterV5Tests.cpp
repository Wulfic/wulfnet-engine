// SPDX-License-Identifier: MIT
// WulfNet Water Physics V5 — Implementation
//
// 2D Shallow Water Equations (SWE) solver rendered as a triangle mesh.
// Water is a height-field sheet that flows over terrain under gravity.

#include <Samples.h>

#include "WulfNetWaterV5Tests.h"
#include "WaterDiagnostics.h"

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

#ifdef _OPENMP
#include <omp.h>
#endif

// =====================================================================
// RTTI Registration
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV5Base)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterV5Base, Test)
}

// Subclass RTTI + scenario implementations --> WulfNetWaterV5Scenarios.cpp

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

	// Initialize water diagnostics logging
	WaterDiagnostics::Initialize(GetRTTI()->GetName());
	WaterDiagnostics::LogSWEConfig(mConfig);

	// Log initial water volume
	float initWater = 0.0f;
	for (uint32_t i = 0; i < totalCells; ++i)
		initWater += mGrid[i].waterHeight;
	SWE_LOG_INFO("[INIT] Setup complete — TotalWater: " +
	             std::to_string(initWater) + " | WetCells initial scan done");

	// --- GPU Compute Initialization ---
	// Attempt to initialize GPU SWE compute if config requests it.
	// WaterCell is { waterHeight, terrainHeight, vx, vz } = 4 floats = vec4,
	// which maps directly to the GPU buffer layout.
	mGPUEnabled = false;
	if (mConfig.useGPU)
	{
		mGPUCompute = std::make_unique<WulfNet::SWEComputeGPU>();
		if (mGPUCompute->Initialize(mConfig.gridSizeX, mConfig.gridSizeZ))
		{
			// Upload initial grid state to GPU
			static_assert(sizeof(WaterCell) == 4 * sizeof(float),
			              "WaterCell must be 4 contiguous floats for GPU upload");
			if (mGPUCompute->UploadGrid(reinterpret_cast<const float*>(mGrid.data()), totalCells))
			{
				mGPUEnabled = true;
				SWE_LOG_INFO("[GPU] SWE GPU compute initialized — " +
				             std::to_string(totalCells) + " cells on GPU");
			}
			else
			{
				SWE_LOG_INFO("[GPU] Failed to upload grid — falling back to CPU");
				mGPUCompute.reset();
			}
		}
		else
		{
			SWE_LOG_INFO("[GPU] SWE GPU compute unavailable — falling back to CPU");
			mGPUCompute.reset();
		}
	}
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
	const float g_over_cs = g / cs;
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
	// Each cell reads only from mGridTemp (snapshot) and writes only its own velocity,
	// so rows can be processed in parallel without any data races.
	#pragma omp parallel for schedule(static)
	for (int32_t z = 1; z < static_cast<int32_t>(NZ) - 1; ++z)
	{
		const uint32_t rowOff = static_cast<uint32_t>(z) * NX;
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

			float fluxL = g_over_cs * (surfH - (nL.terrainHeight + nL.waterHeight));
			float fluxR = g_over_cs * (surfH - (nR.terrainHeight + nR.waterHeight));
			float fluxB = g_over_cs * (surfH - (nB.terrainHeight + nB.waterHeight));
			float fluxF = g_over_cs * (surfH - (nF.terrainHeight + nF.waterHeight));

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

		const size_t noiseBufSize = static_cast<size_t>(dsNX) * dsNZ;
		if (mNoiseBufVx.size() != noiseBufSize)
		{
			mNoiseBufVx.resize(noiseBufSize);
			mNoiseBufVz.resize(noiseBufSize);
		}

		#pragma omp parallel for collapse(2) schedule(static)
		for (int32_t dz = 0; dz < static_cast<int32_t>(dsNZ); ++dz)
		{
			for (int32_t dx = 0; dx < static_cast<int32_t>(dsNX); ++dx)
			{
				float fz = static_cast<float>(static_cast<uint32_t>(dz) * DS) * freq / NZ;
				float fx = static_cast<float>(static_cast<uint32_t>(dx) * DS) * freq / NX;
				uint32_t idx = static_cast<uint32_t>(dz) * dsNX + static_cast<uint32_t>(dx);
				mNoiseBufVx[idx] = mNoise.FBM3D(fx, fz, t * speed,
				                                 oct, lac, per, str, 1.0f);
				mNoiseBufVz[idx] = mNoise.FBM3D(fx + 17.3f, fz + 31.7f, t * speed + 5.0f,
				                                 oct, lac, per, str, 1.0f);
			}
		}

		float invDS = 1.0f / static_cast<float>(DS);
		#pragma omp parallel for schedule(static)
		for (int32_t z = 1; z < static_cast<int32_t>(NZ) - 1; ++z)
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

				float nVx = (1-fx)*(1-fz) * mNoiseBufVx[cz0*dsNX+cx0]
				          + fx*(1-fz)     * mNoiseBufVx[cz0*dsNX+cx1]
				          + (1-fx)*fz     * mNoiseBufVx[cz1*dsNX+cx0]
				          + fx*fz         * mNoiseBufVx[cz1*dsNX+cx1];

				float nVz = (1-fx)*(1-fz) * mNoiseBufVz[cz0*dsNX+cx0]
				          + fx*(1-fz)     * mNoiseBufVz[cz0*dsNX+cx1]
				          + (1-fx)*fz     * mNoiseBufVz[cz1*dsNX+cx0]
				          + fx*fz         * mNoiseBufVz[cz1*dsNX+cx1];

				float depthScale = std::min(1.0f, cell.waterHeight * 10.0f);
				cell.vx += nVx * depthScale;
				cell.vz += nVz * depthScale;
			}
		}
	}

	// ---------- Phase 2: Conservative water transport (two-pass gather) ----------
	// Restructured from scatter to gather for thread-safe parallelism:
	//   Pass 1: Compute directional outflows per cell (no race — each cell writes its own)
	//   Pass 2: Each cell gathers inflow from neighbours' outflow toward it
	// NOTE: No second memcpy needed — Phase 1/1.5 only modified .vx/.vz,
	// and Phase 2 reads only .waterHeight/.terrainHeight from mGridTemp (unchanged).

	// Ensure outflow buffer is correctly sized (reused across frames — no allocation)
	const size_t outflowSize = static_cast<size_t>(total) * 4;
	if (mOutflowBuf.size() != outflowSize)
		mOutflowBuf.assign(outflowSize, 0.0f);
	else
		std::memset(mOutflowBuf.data(), 0, outflowSize * sizeof(float));

	// Pass 1: Compute per-cell directional outflows [Left, Right, Back, Front]
	#pragma omp parallel for schedule(static)
	for (int32_t z = 1; z < static_cast<int32_t>(NZ) - 1; ++z)
	{
		const uint32_t rowOff = static_cast<uint32_t>(z) * NX;
		for (uint32_t x = 1; x + 1 < NX; ++x)
		{
			const uint32_t idx = rowOff + x;
			const WaterCell &src = mGridTemp[idx];
			if (src.waterHeight < 1e-6f) continue;

			float surfH = src.terrainHeight + src.waterHeight;

			float outL = 0.0f, outR = 0.0f, outB = 0.0f, outF = 0.0f;
			float totalOut = 0.0f;

			float dh = surfH - (mGridTemp[idx - 1].terrainHeight + mGridTemp[idx - 1].waterHeight);
			if (dh > 0.0f) { outL = dh * 0.25f; totalOut += outL; }

			dh = surfH - (mGridTemp[idx + 1].terrainHeight + mGridTemp[idx + 1].waterHeight);
			if (dh > 0.0f) { outR = dh * 0.25f; totalOut += outR; }

			dh = surfH - (mGridTemp[idx - NX].terrainHeight + mGridTemp[idx - NX].waterHeight);
			if (dh > 0.0f) { outB = dh * 0.25f; totalOut += outB; }

			dh = surfH - (mGridTemp[idx + NX].terrainHeight + mGridTemp[idx + NX].waterHeight);
			if (dh > 0.0f) { outF = dh * 0.25f; totalOut += outF; }

			// Clamp to available water
			if (totalOut > src.waterHeight)
			{
				float scale = src.waterHeight / totalOut;
				outL *= scale; outR *= scale; outB *= scale; outF *= scale;
			}

			const size_t base = static_cast<size_t>(idx) * 4;
			mOutflowBuf[base + 0] = outL;
			mOutflowBuf[base + 1] = outR;
			mOutflowBuf[base + 2] = outB;
			mOutflowBuf[base + 3] = outF;
		}
	}

	// Pass 2: Gather — each cell subtracts its own outflows, adds inflow from neighbours
	#pragma omp parallel for schedule(static)
	for (int32_t z = 1; z < static_cast<int32_t>(NZ) - 1; ++z)
	{
		const uint32_t rowOff = static_cast<uint32_t>(z) * NX;
		for (uint32_t x = 1; x + 1 < NX; ++x)
		{
			const uint32_t idx = rowOff + x;
			const size_t base = static_cast<size_t>(idx) * 4;
			float selfOut = mOutflowBuf[base + 0] + mOutflowBuf[base + 1]
			              + mOutflowBuf[base + 2] + mOutflowBuf[base + 3];

			// Inflow: left neighbour's RIGHT, right neighbour's LEFT,
			//         back neighbour's FRONT, front neighbour's BACK
			float inflow = mOutflowBuf[static_cast<size_t>(idx - 1) * 4 + 1]
			             + mOutflowBuf[static_cast<size_t>(idx + 1) * 4 + 0]
			             + mOutflowBuf[static_cast<size_t>(idx - NX) * 4 + 3]
			             + mOutflowBuf[static_cast<size_t>(idx + NX) * 4 + 2];

			mGrid[idx].waterHeight = mGridTemp[idx].waterHeight - selfOut + inflow;
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
	int32_t total = static_cast<int32_t>(NX * NZ);
	#pragma omp parallel for schedule(static)
	for (int32_t i = 0; i < total; ++i)
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

	// Step the SWE solver with adaptive CFL-based substepping.
	// Instead of a fixed substep count, we compute the maximum wave speed
	// (v_max + sqrt(g*h_max)) and choose dt so CFL = v*dt/dx < CFL_TARGET.
	// This avoids both instability (CFL>>1) and wasted work when slow.
	auto simStart = std::chrono::high_resolution_clock::now();
	{
		constexpr float CFL_TARGET  = 0.45f;  // Conservative target
		constexpr float MIN_SUB_DT  = 1e-6f;  // Safety floor
		constexpr uint32_t MAX_SUBS = 200;     // Hard cap to prevent runaway

		const float cs = mConfig.cellSize;
		const float g  = mConfig.gravity;
		float remaining = inParams.mDeltaTime;
		uint32_t subCount = 0;

		// Build GPU params struct once (dt updated per substep)
		WulfNet::SWESimParams gpuParams{};
		gpuParams.gridSizeX     = mConfig.gridSizeX;
		gpuParams.gridSizeZ     = mConfig.gridSizeZ;
		gpuParams.gravity_over_cs = g / cs;
		gpuParams.damping       = 1.0f - mConfig.damping;
		gpuParams.viscosity     = mConfig.viscosity;
		gpuParams.dt            = 0.0f;

		if (mGPUEnabled && mGPUCompute && mGPUCompute->IsInitialized())
		{
			// ===== GPU PATH =====
			// Upload current CPU grid to GPU (includes any scenario updates)
			mGPUCompute->UploadGrid(reinterpret_cast<const float*>(mGrid.data()),
			                         mConfig.gridSizeX * mConfig.gridSizeZ);

			while (remaining > MIN_SUB_DT && subCount < MAX_SUBS)
			{
				// CFL scan (CPU — fast on the local grid copy)
				float maxSpeed2 = 0.0f;
				float maxDepth  = 0.0f;
				const int32_t total = static_cast<int32_t>(mConfig.gridSizeX * mConfig.gridSizeZ);
				#pragma omp parallel for schedule(static) reduction(max:maxSpeed2,maxDepth)
				for (int32_t i = 0; i < total; ++i)
				{
					float s2 = mGrid[i].vx * mGrid[i].vx + mGrid[i].vz * mGrid[i].vz;
					if (s2 > maxSpeed2) maxSpeed2 = s2;
					if (mGrid[i].waterHeight > maxDepth) maxDepth = mGrid[i].waterHeight;
				}

				float maxVel = std::sqrt(maxSpeed2);
				float waveSpeed = maxVel + std::sqrt(g * maxDepth);

				float subDt;
				if (waveSpeed > 1e-6f)
					subDt = CFL_TARGET * cs / waveSpeed;
				else
					subDt = remaining;

				subDt = std::min(subDt, remaining);
				subDt = std::max(subDt, MIN_SUB_DT);

				// Dispatch GPU SWE step (batched: snapshot→vel→outflow→gather→boundary)
				gpuParams.dt = subDt;
				mGPUCompute->StepSWE(gpuParams);

				remaining -= subDt;
				++subCount;

				// Download after each substep for CFL scan accuracy
				// (GPU→CPU transfer is ~150KB for 80×80 grid, very fast)
				mGPUCompute->DownloadGrid(reinterpret_cast<float*>(mGrid.data()),
				                           mConfig.gridSizeX * mConfig.gridSizeZ);
			}
		}
		else
		{
			// ===== CPU PATH (fallback) =====
			while (remaining > MIN_SUB_DT && subCount < MAX_SUBS)
			{
				float maxSpeed2 = 0.0f;
				float maxDepth  = 0.0f;
				const int32_t total = static_cast<int32_t>(mConfig.gridSizeX * mConfig.gridSizeZ);
				#pragma omp parallel for schedule(static) reduction(max:maxSpeed2,maxDepth)
				for (int32_t i = 0; i < total; ++i)
				{
					float s2 = mGrid[i].vx * mGrid[i].vx + mGrid[i].vz * mGrid[i].vz;
					if (s2 > maxSpeed2) maxSpeed2 = s2;
					if (mGrid[i].waterHeight > maxDepth) maxDepth = mGrid[i].waterHeight;
				}

				float maxVel = std::sqrt(maxSpeed2);
				float waveSpeed = maxVel + std::sqrt(g * maxDepth);

				float subDt;
				if (waveSpeed > 1e-6f)
					subDt = CFL_TARGET * cs / waveSpeed;
				else
					subDt = remaining;

				subDt = std::min(subDt, remaining);
				subDt = std::max(subDt, MIN_SUB_DT);

				StepSWE(subDt);
				remaining -= subDt;
				++subCount;
			}
		}
		mActualSubsteps = subCount;
	}
	auto simEnd = std::chrono::high_resolution_clock::now();
	mSimTimeMs = std::chrono::duration<float, std::milli>(simEnd - simStart).count();

	// Compute total water volume (for conservation display)
	float totalWater = 0.0f;
	int32_t total = static_cast<int32_t>(mConfig.gridSizeX * mConfig.gridSizeZ);
	#pragma omp parallel for schedule(static) reduction(+:totalWater)
	for (int32_t i = 0; i < total; ++i)
		totalWater += mGrid[i].waterHeight;
	mTotalWater = totalWater;

	// Log per-frame SWE diagnostics
	WaterDiagnostics::LogSWEFrame(mGrid, mConfig.gridSizeX, mConfig.gridSizeZ,
	                               mConfig.cellSize, mSimTimeMs, mCurrentFPS,
	                               mTotalWater);

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
	int32_t wetCells = 0;
	float maxDepth = 0.0f;
	float maxVel2 = 0.0f;
	int32_t total = static_cast<int32_t>(mConfig.gridSizeX * mConfig.gridSizeZ);
	float minDraw = mConfig.minWaterDraw;
	#pragma omp parallel for schedule(static) reduction(+:wetCells) reduction(max:maxDepth) reduction(max:maxVel2)
	for (int32_t i = 0; i < total; ++i)
	{
		float wh = mGrid[i].waterHeight;
		if (wh > minDraw)
			++wetCells;
		if (wh > maxDepth) maxDepth = wh;
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

	// CFL number for display (wave speed * dt / dx)
	float waveSpeed = maxVel + std::sqrt(mConfig.gravity * maxDepth);
	float cflNum = (waveSpeed > 0.0f && mActualSubsteps > 0)
	             ? waveSpeed * (mFrameTimeMs * 0.001f / mActualSubsteps) / mConfig.cellSize
	             : 0.0f;
	oss << "Substeps: " << mActualSubsteps << " (adaptive CFL)\n";
	oss << "CFL: " << std::setprecision(2) << cflNum << "\n";
	oss << "Sim: " << std::setprecision(2) << mSimTimeMs << " ms"
	    << (mGPUEnabled ? " [GPU]" : " [CPU]");

	return String(oss.str());
}

