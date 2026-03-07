// =============================================================================
// WulfNet Advanced Visual Tests — Part 2
// =============================================================================
// 3. Terrain deformation test
// 4. Volumetric cloud test
// 5. Spatial audio test
// Extracted from WulfNetAdvancedTests.cpp for maintainability.
// =============================================================================

#include <Samples.h>
#include "WulfNetAdvancedTests.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Renderer/DebugRendererImp.h>
#include <Layers.h>

#include <cmath>
#include <algorithm>

static float Clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetTerrainVisualTest)
{
	JPH_ADD_BASE_CLASS(WulfNetTerrainVisualTest, Test)
}

void WulfNetTerrainVisualTest::Initialize()
{
	// No CreateFloor() — the terrain IS the ground surface

	// 80x80 grid, 0.25m cells = 20m x 20m terrain
	mTerrainConfig.gridSizeX  = 80;
	mTerrainConfig.gridSizeZ  = 80;
	mTerrainConfig.cellSize   = 0.25f;
	mTerrainConfig.originX    = -(80 * 0.25f) / 2.0f; // -10.0
	mTerrainConfig.originZ    = -(80 * 0.25f) / 2.0f; // -10.0
	mTerrainConfig.originY    = 0.0f;
	mTerrainConfig.maxDeformDepth = 1.0f;
	mTerrainConfig.maxDeformRaise = 0.5f;

	mTerrain.Initialize(mTerrainConfig);

	// Perfectly flat ground at Y=0
	std::vector<float> heights(80 * 80, 0.0f);
	mTerrain.SetHeightField(heights.data(), 80, 80);

	// All soft soil for easy, visible deformation
	WulfNet::TerrainMaterial soil = WulfNet::TerrainMaterial::SoftSoil();
	mTerrain.SetMaterialRegion(0, 0, 79, 79, soil);

	mTime       = 0.0f;
	mDropTimer  = 0.0f;
	mTrackAngle = 0.0f;

	// Pre-apply one full lap of tire tracks for immediate visibility
	for (float a = 0.02f; a < 6.28f; a += 0.02f)
	{
		float r = 5.0f;
		float x0 = cosf(a - 0.02f) * r;
		float z0 = sinf(a - 0.02f) * r;
		float x1 = cosf(a) * r;
		float z1 = sinf(a) * r;
		mTerrain.ApplyTireTrack(x0, z0, x1, z1, 0.3f, 0.04f, 0.4f);
	}
}

void WulfNetTerrainVisualTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = inParams.mDeltaTime;
	mTime += dt;

	// Vehicle drives in a circle, leaving tire tracks
	float speed = 0.5f; // radians per second
	float prevAngle = mTrackAngle;
	mTrackAngle += speed * dt;

	// Outer track
	float trackR = 5.0f;
	float tx0 = cosf(prevAngle) * trackR;
	float tz0 = sinf(prevAngle) * trackR;
	float tx1 = cosf(mTrackAngle) * trackR;
	float tz1 = sinf(mTrackAngle) * trackR;
	mTerrain.ApplyTireTrack(tx0, tz0, tx1, tz1, 0.3f, 0.04f, 0.4f);

	// Inner track (simulates 4 wheels)
	float trackR2 = 4.4f;
	float ix0 = cosf(prevAngle) * trackR2;
	float iz0 = sinf(prevAngle) * trackR2;
	float ix1 = cosf(mTrackAngle) * trackR2;
	float iz1 = sinf(mTrackAngle) * trackR2;
	mTerrain.ApplyTireTrack(ix0, iz0, ix1, iz1, 0.3f, 0.04f, 0.4f);

	// Occasional small crater (every 4 seconds)
	mDropTimer += dt;
	if (mDropTimer >= 4.0f)
	{
		mDropTimer = 0.0f;
		std::uniform_real_distribution<float> posDist(-6.0f, 6.0f);
		float wx = posDist(mRng);
		float wz = posDist(mRng);
		mTerrain.ApplyExplosion(wx, wz, 1.0f, 0.4f);
	}

	DrawTerrain();
}

void WulfNetTerrainVisualTest::DrawTerrain()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	const float cs = mTerrainConfig.cellSize;
	const uint32_t sx = mTerrainConfig.gridSizeX;
	const uint32_t sz = mTerrainConfig.gridSizeZ;
	const float ox = mTerrainConfig.originX;
	const float oz = mTerrainConfig.originZ;

	for (uint32_t z = 0; z < sz - 1; ++z)
	{
		for (uint32_t x = 0; x < sx - 1; ++x)
		{
			float h00 = mTerrain.GetHeightAt(x, z);
			float h10 = mTerrain.GetHeightAt(x + 1, z);
			float h01 = mTerrain.GetHeightAt(x, z + 1);
			float h11 = mTerrain.GetHeightAt(x + 1, z + 1);

			float wx0 = ox + x * cs;
			float wx1 = ox + (x + 1) * cs;
			float wz0 = oz + z * cs;
			float wz1 = oz + (z + 1) * cs;

			RVec3 p00(wx0, h00, wz0);
			RVec3 p10(wx1, h10, wz0);
			RVec3 p01(wx0, h01, wz1);
			RVec3 p11(wx1, h11, wz1);

			// Color: green for flat, brown for depressions (tire tracks/craters)
			float avgDelta = (h00 + h10 + h01 + h11) * 0.25f;
			uint8_t r, g, b;
			if (avgDelta < -0.005f)
			{
				// Deformed: brown dirt
				float f = Clamp01(-avgDelta / 0.3f);
				r = (uint8_t)(100 + 80 * f);
				g = (uint8_t)(80 + 20 * f);
				b = 40;
			}
			else if (avgDelta > 0.005f)
			{
				// Raised rim: lighter brown
				float f = Clamp01(avgDelta / 0.2f);
				r = (uint8_t)(120 + 60 * f);
				g = (uint8_t)(100 + 40 * f);
				b = (uint8_t)(50 + 20 * f);
			}
			else
			{
				// Flat grass
				r = 60; g = 130; b = 40;
			}

			Color col(r, g, b, 255);

			// Two triangles per quad
			mDebugRenderer->DrawTriangle(p00, p10, p01, col);
			mDebugRenderer->DrawTriangle(p10, p11, p01, col);
		}
	}

	// Draw a "vehicle" marker at current track position
	float vx = cosf(mTrackAngle) * 5.0f;
	float vz = sinf(mTrackAngle) * 5.0f;
	float vh = mTerrain.SampleHeight(vx, vz);
	RVec3 vehiclePos(vx, vh + 0.4f, vz);
	mDebugRenderer->DrawSphere(vehiclePos, 0.3f, Color(200, 200, 50, 230),
		DebugRenderer::ECastShadow::On, DebugRenderer::EDrawMode::Solid);
	mDebugRenderer->DrawText3D(vehiclePos + RVec3(0, 0.5, 0), "Vehicle", Color::sYellow, 0.2f);

	// Forward direction arrow
	float fwdX = -sinf(mTrackAngle) * 1.0f;
	float fwdZ =  cosf(mTrackAngle) * 1.0f;
	mDebugRenderer->DrawArrow(vehiclePos, vehiclePos + RVec3(fwdX, 0, fwdZ), Color::sGreen, 0.04f);

	const auto &stats = mTerrain.GetStats();
	char buf[256];
	snprintf(buf, sizeof(buf), "Terrain 20x20m: %u deformations, %u cells modified",
	         stats.totalDeformations, stats.cellsModified);
	mDebugRenderer->DrawText3D(RVec3(0, 4, 0), buf, Color::sYellow, 0.2f);
	mDebugRenderer->DrawText3D(RVec3(0, 4.5, 0), "Terrain Deformation — Vehicle Tire Tracks", Color::sWhite, 0.35f);
#endif
}

// =====================================================================
//  4.  VOLUMETRIC CLOUD — pre-filled density cloud
//
//  Previous bugs:
//    - Relied solely on emitters near grid boundaries → sparse dots.
//    - No pre-filled density → took forever to build up.
//    - Small cell size + fast dissipation → dots flew apart.
//
//  Fixes:
//    - Pre-fill the entire grid with a smooth gaussian cloud blob.
//    - Extremely slow dissipation (0.9998) preserves shape.
//    - One gentle center emitter to replenish.
//    - No gravity, minimal buoyancy → cloud hangs in place.
//    - Larger cells (0.3m), larger render spheres → cohesive volume.
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetVolumetricVisualTest)
{
	JPH_ADD_BASE_CLASS(WulfNetVolumetricVisualTest, Test)
}

void WulfNetVolumetricVisualTest::Initialize()
{
	CreateFloor();

	// 20x12x20 grid, 0.3m cells → 6m x 3.6m x 6m cloud volume
	mGasConfig.resolutionX = 20;
	mGasConfig.resolutionY = 12;
	mGasConfig.resolutionZ = 20;
	mGasConfig.cellSize    = 0.3f;
	mGasConfig.originX     = -(20 * 0.3f) / 2.0f; // -3.0
	mGasConfig.originY     = 2.0f;                  // floating above ground
	mGasConfig.originZ     = -(20 * 0.3f) / 2.0f;

	// Cloud behavior: almost no movement, very slow fade
	mGasConfig.buoyancyAlpha       = 0.0f;
	mGasConfig.buoyancyBeta        = 0.05f;   // tiny buoyancy
	mGasConfig.vorticityStrength   = 0.1f;
	mGasConfig.gravityY            = 0.0f;     // clouds don't fall

	mGasConfig.densityDissipation     = 0.9998f; // very slow fade
	mGasConfig.temperatureDissipation = 0.999f;
	mGasConfig.velocityDissipation    = 0.98f;   // velocities dampen quickly

	mGasConfig.pressureIterations = 10;
	mGasConfig.substeps           = 1;
	mGasConfig.maxTimestep        = 1.0f / 30.0f;

	mGas.Initialize(mGasConfig);

	// Pre-fill the grid with a smooth gaussian-like cloud blob
	float centerI = 10.0f, centerJ = 6.0f, centerK = 10.0f;
	float cloudRadiusI = 7.0f, cloudRadiusJ = 4.0f, cloudRadiusK = 7.0f;

	for (uint32_t j = 0; j < 12; ++j)
	{
		for (uint32_t k = 0; k < 20; ++k)
		{
			for (uint32_t i = 0; i < 20; ++i)
			{
				float di = (static_cast<float>(i) - centerI) / cloudRadiusI;
				float dj = (static_cast<float>(j) - centerJ) / cloudRadiusJ;
				float dk = (static_cast<float>(k) - centerK) / cloudRadiusK;
				float dist2 = di * di + dj * dj + dk * dk;

				if (dist2 < 1.0f)
				{
					float falloff = (1.0f - dist2);
					falloff = falloff * falloff; // sharper edges

					// Add noise for natural cloud appearance
					float noise = 0.7f + 0.3f * sinf(i * 2.1f + k * 3.7f + j * 1.3f);
					float density = 3.0f * falloff * noise;

					if (density > 0.05f)
					{
						mGas.SetDensity(i, j, k, density);
						mGas.SetTemperature(i, j, k, 30.0f * falloff);
					}
				}
			}
		}
	}

	// One gentle emitter at cloud center to keep it alive
	WulfNet::GasEmitter em;
	em.type          = WulfNet::GasEmitterType::Sphere;
	em.posX          = 0.0f;
	em.posY          = 3.8f; // middle of the cloud volume
	em.posZ          = 0.0f;
	em.radius        = 1.0f;
	em.densityRate   = 2.0f;     // gentle replenishment
	em.temperatureRate = 10.0f;
	em.fuelRate      = 0.0f;
	em.velocityX     = 0.0f;
	em.velocityY     = 0.0f;
	em.velocityZ     = 0.0f;
	em.enabled       = true;
	mGas.AddEmitter(em);

	mTime = 0.0f;
}

void WulfNetVolumetricVisualTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = std::min(inParams.mDeltaTime, 1.0f / 30.0f);
	// Guard: skip sim step if dt is invalid
	if (!std::isfinite(dt) || dt <= 0.0f)
	{
		DrawVolumetricField();
		return;
	}
	mTime += dt;
	mGas.Step(dt);
	DrawVolumetricField();
}

void WulfNetVolumetricVisualTest::DrawVolumetricField()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	const float cellSize = mGas.GetCellSize();
	const uint32_t rx = mGas.GetResolutionX();
	const uint32_t ry = mGas.GetResolutionY();
	const uint32_t rz = mGas.GetResolutionZ();

	// Safety: clamp max spheres to prevent debug renderer overflow
	constexpr uint32_t cMaxDrawSpheres = 2000;
	uint32_t drawnSpheres = 0;

	for (uint32_t j = 0; j < ry && drawnSpheres < cMaxDrawSpheres; ++j)
	{
		for (uint32_t k = 0; k < rz && drawnSpheres < cMaxDrawSpheres; ++k)
		{
			for (uint32_t i = 0; i < rx && drawnSpheres < cMaxDrawSpheres; ++i)
			{
				const auto &cell = mGas.GetCell(i, j, k);
				if (cell.density < 0.02f) continue;

				// Guard against NaN/Inf from numerical instability
				if (!std::isfinite(cell.density)) continue;

				float wx, wy, wz;
				mGas.GridToWorld(i + 0.5f, j + 0.5f, k + 0.5f, wx, wy, wz);
				if (!std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz))
					continue;

				float d = Clamp01(cell.density / 3.0f);

				// Cloud coloring: white to light gray with soft transparency
				uint8_t brightness = (uint8_t)(200 + 55 * d);
				uint8_t a = (uint8_t)(40 + 160 * d);

				Color c(brightness, brightness, (uint8_t)(brightness - 5), a);
				float sphereR = cellSize * 0.6f * (0.5f + 0.5f * d);
				mDebugRenderer->DrawSphere(RVec3(wx, wy, wz), sphereR, c,
					DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
				++drawnSpheres;
			}
		}
	}

	const auto &stats = mGas.GetStats();
	char buf[128];
	snprintf(buf, sizeof(buf), "Cloud: %u cells, density=%.1f (drawn: %u)",
	         stats.activeCells, stats.totalDensity, drawnSpheres);
	mDebugRenderer->DrawText3D(RVec3(0, 7, 0), buf, Color::sYellow, 0.25f);
	mDebugRenderer->DrawText3D(RVec3(0, 7.5, 0), "Volumetric Cloud / Fog", Color::sWhite, 0.4f);
#endif
}

// =====================================================================
//  5.  SPATIAL AUDIO & ACOUSTIC VISUALIZATION TEST
//      (This test worked correctly — kept largely unchanged)
// =====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetAudioVisualTest)
{
	JPH_ADD_BASE_CLASS(WulfNetAudioVisualTest, Test)
}

void WulfNetAudioVisualTest::Initialize()
{
	CreateFloor();

	// Build a room from static walls
	float wallThick = 0.15f;
	float roomH     = 5.0f;
	float roomHX    = 6.0f, roomHZ = 6.0f;

	struct WallDef { Vec3 halfExt; RVec3 pos; };
	WallDef walls[] = {
		{ Vec3(roomHX, wallThick, roomHZ), RVec3(0, roomH, 0) },          // ceiling
		{ Vec3(roomHX, roomH / 2, wallThick), RVec3(0, roomH / 2, -roomHZ) }, // back
		{ Vec3(roomHX, roomH / 2, wallThick), RVec3(0, roomH / 2,  roomHZ) }, // front
		{ Vec3(wallThick, roomH / 2, roomHZ), RVec3(-roomHX, roomH / 2, 0) }, // left
		{ Vec3(wallThick, roomH / 2, roomHZ), RVec3( roomHX, roomH / 2, 0) }, // right
	};

	for (auto &w : walls)
	{
		BodyCreationSettings bcs(
			new BoxShape(w.halfExt), w.pos,
			Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);
		BodyID bid = mBodyInterface->CreateAndAddBody(bcs, EActivation::DontActivate);
		mRoomWalls.push_back(bid);
	}

	mSources.push_back({ RVec3(-3, 1.5, -3), Color::sRed,    8.0f, "Music" });
	mSources.push_back({ RVec3( 4, 2.0,  2), Color::sGreen,  6.0f, "Speech" });
	mSources.push_back({ RVec3( 0, 0.5, -4), Color(255, 165, 0, 255), 10.0f, "Footsteps" });

	mSpatialAudio.Initialize(44100);

	WulfNet::AcousticConfig acfg;
	acfg.maxBounces    = 4;
	acfg.numRays       = 32;
	acfg.maxDistance    = 20.0f;
	acfg.roomProbeRays = 16;
	mAcoustics.Initialize(acfg);

	// Ray-cast callback for the box-shaped room
	const RVec3 rmin = mRoomMin;
	const RVec3 rmax = mRoomMax;
	mAcoustics.SetRayCastFunction(
		[rmin, rmax](float ox, float oy, float oz,
		             float dx, float dy, float dz,
		             float maxDist) -> WulfNet::AcousticRayHit
		{
			WulfNet::AcousticRayHit result;
			result.hit      = false;
			result.distance = maxDist;

			float planes[][4] = {
				{ 1, 0, 0, (float)-rmin.GetX()},
				{-1, 0, 0, (float) rmax.GetX()},
				{ 0, 1, 0, (float)-rmin.GetY()},
				{ 0,-1, 0, (float) rmax.GetY()},
				{ 0, 0, 1, (float)-rmin.GetZ()},
				{ 0, 0,-1, (float) rmax.GetZ()},
			};

			for (auto &p : planes)
			{
				float denom = p[0] * dx + p[1] * dy + p[2] * dz;
				if (fabsf(denom) < 1e-6f) continue;
				float t = -(p[0] * ox + p[1] * oy + p[2] * oz + p[3]) / denom;
				if (t > 0.001f && t < result.distance)
				{
					float hx = ox + dx * t;
					float hy = oy + dy * t;
					float hz = oz + dz * t;
					if (hx >= (float)rmin.GetX() - 0.01f && hx <= (float)rmax.GetX() + 0.01f &&
					    hy >= (float)rmin.GetY() - 0.01f && hy <= (float)rmax.GetY() + 0.01f &&
					    hz >= (float)rmin.GetZ() - 0.01f && hz <= (float)rmax.GetZ() + 0.01f)
					{
						result.hit      = true;
						result.distance = t;
						result.normalX  = p[0];
						result.normalY  = p[1];
						result.normalZ  = p[2];
						result.materialId = 0;
					}
				}
			}
			return result;
		}
	);

	mAcoustics.AddMaterial(WulfNet::AcousticMaterial::Concrete());

	mAcousticInfo.resize(mSources.size());
	mUpdateTimer = 0.0f;
	mTime        = 0.0f;
}

void WulfNetAudioVisualTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = inParams.mDeltaTime;
	mTime += dt;

	// Slowly orbit the listener
	float lx = cosf(mTime * 0.3f) * 2.0f;
	float lz = sinf(mTime * 0.3f) * 2.0f;
	mListenerPos = RVec3(lx, 1.7, lz);
	mListenerFwd = RVec3(-lx, 0, -lz).Normalized();

	// Update acoustic data periodically (expensive)
	mUpdateTimer += dt;
	if (mUpdateTimer >= 0.3f)
	{
		mUpdateTimer = 0.0f;
		for (size_t i = 0; i < mSources.size(); ++i)
		{
			auto &src  = mSources[i];
			auto &info = mAcousticInfo[i];

			float sx = (float)src.position.GetX();
			float sy = (float)src.position.GetY();
			float sz = (float)src.position.GetZ();

			info.occlusion   = mAcoustics.ComputeOcclusion(
				sx, sy, sz,
				(float)mListenerPos.GetX(), (float)mListenerPos.GetY(), (float)mListenerPos.GetZ());
			info.obstruction = mAcoustics.ComputeObstruction(
				sx, sy, sz,
				(float)mListenerPos.GetX(), (float)mListenerPos.GetY(), (float)mListenerPos.GetZ(), 8);

			float dx = sx - (float)mListenerPos.GetX();
			float dy = sy - (float)mListenerPos.GetY();
			float dz = sz - (float)mListenerPos.GetZ();
			float dist = sqrtf(dx * dx + dy * dy + dz * dz);
			info.distGain = mSpatialAudio.ComputeDistanceGain(dist);

			info.ir = mAcoustics.TraceImpulseResponse(
				sx, sy, sz,
				(float)mListenerPos.GetX(), (float)mListenerPos.GetY(), (float)mListenerPos.GetZ());
		}
	}

	DrawAudioSources();
	DrawAcousticRays();
	DrawAttenuationRadii();
}

void WulfNetAudioVisualTest::DrawAudioSources()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	mDebugRenderer->DrawSphere(mListenerPos, 0.15f, Color::sWhite,
		DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
	mDebugRenderer->DrawText3D(mListenerPos + RVec3(0, 0.3, 0), "Listener", Color::sWhite, 0.2f);

	RVec3 fwdEnd = mListenerPos + mListenerFwd * 0.6;
	mDebugRenderer->DrawArrow(mListenerPos, fwdEnd, Color::sGreen, 0.03f);

	for (size_t i = 0; i < mSources.size(); ++i)
	{
		auto &src  = mSources[i];
		auto &info = mAcousticInfo[i];

		float pulse = 0.8f + 0.2f * sinf(mTime * 4.0f + (float)i * 2.0f);
		mDebugRenderer->DrawSphere(src.position, 0.12f * pulse, src.color,
			DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
		mDebugRenderer->DrawText3D(src.position + RVec3(0, 0.35, 0), src.label, src.color, 0.2f);

		char buf[128];
		snprintf(buf, sizeof(buf), "occ=%.2f obstr=%.2f gain=%.2f",
		         info.occlusion, info.obstruction, info.distGain);
		mDebugRenderer->DrawText3D(src.position + RVec3(0, 0.55, 0), buf, Color::sYellow, 0.15f);

		uint8_t occA = (uint8_t)(info.occlusion * 200);
		Color lineCol(info.occlusion > 0.5f ? (uint8_t)0 : (uint8_t)255,
		              info.occlusion > 0.5f ? (uint8_t)200 : (uint8_t)50,
		              (uint8_t)0, occA);
		mDebugRenderer->DrawArrow(src.position, mListenerPos, lineCol, 0.015f);
	}
#endif
}

void WulfNetAudioVisualTest::DrawAcousticRays()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	for (size_t i = 0; i < mSources.size(); ++i)
	{
		auto &src  = mSources[i];
		auto &info = mAcousticInfo[i];

		int maxTaps = std::min((int)info.ir.taps.size(), 8);
		for (int t = 0; t < maxTaps; ++t)
		{
			const auto &tap = info.ir.taps[t];
			if (tap.energy < 0.01f) continue;

			float reflDist = tap.time * 343.0f * 0.5f;
			RVec3 reflPoint = mListenerPos + RVec3(
				tap.direction[0] * reflDist,
				tap.direction[1] * reflDist,
				tap.direction[2] * reflDist);

			float energyF = Clamp01(tap.energy * 3.0f);
			uint8_t ea = (uint8_t)(100 * energyF);
			Color reflCol(200, 200, 255, ea);

			mDebugRenderer->DrawArrow(src.position, reflPoint, reflCol, 0.008f);
			mDebugRenderer->DrawArrow(reflPoint, mListenerPos, reflCol, 0.008f);
		}
	}
#endif
}

void WulfNetAudioVisualTest::DrawAttenuationRadii()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer) return;

	for (auto &src : mSources)
	{
		Color faded(src.color, 60);
		mDebugRenderer->DrawWireSphere(src.position, src.radius, faded, 1);
		mDebugRenderer->DrawWireSphere(src.position, src.radius * 0.5f,
			Color(src.color, 40), 1);
	}

	auto room = mAcoustics.EstimateRoom(
		(float)mListenerPos.GetX(),
		(float)mListenerPos.GetY(),
		(float)mListenerPos.GetZ());

	char roomBuf[256];
	snprintf(roomBuf, sizeof(roomBuf), "Room: V=%.0f m3  SA=%.0f m2  RT60=%.2fs",
	         room.volume, room.surfaceArea, room.rt60);
	mDebugRenderer->DrawText3D(RVec3(0, 5.5, 0), roomBuf, Color::sYellow, 0.25f);
	mDebugRenderer->DrawText3D(RVec3(0, 6.0, 0), "Spatial Audio & Acoustics", Color::sWhite, 0.4f);
#endif
}
