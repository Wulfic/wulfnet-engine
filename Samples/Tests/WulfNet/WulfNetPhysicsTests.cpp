// SPDX-License-Identifier: MIT
// WulfNet Physics Integration Visual Tests — Implementation

#include <Samples.h>

#include "WulfNetPhysicsTests.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Renderer/DebugRendererImp.h>
#include <Layers.h>

#include <cmath>
#include <algorithm>

// ====================================================================
// Helper: apply a GPUMat4x4 to a 3D point (homogeneous, row-major)
// ====================================================================
static void TransformPoint(const WulfNet::GPUMat4x4 &mat, float inX, float inY, float inZ,
						   float &outX, float &outY, float &outZ)
{
	// GPUMat4x4 is row-major: m[row*4 + col]
	outX = mat.m[0] * inX + mat.m[1] * inY + mat.m[2]  * inZ + mat.m[3];
	outY = mat.m[4] * inX + mat.m[5] * inY + mat.m[6]  * inZ + mat.m[7];
	outZ = mat.m[8] * inX + mat.m[9] * inY + mat.m[10] * inZ + mat.m[11];
}

// ====================================================================
// Simple value noise for turbulence
// ====================================================================
static float HashFloat(float x, float y, float z)
{
	// Fast deterministic hash → float in [-1,1]
	// Use unsigned to avoid signed integer overflow (UB)
	uint32_t ix = (uint32_t)(int)(fmodf(x, 97.0f) * 73856.0f)
	            ^ (uint32_t)(int)(fmodf(y, 97.0f) * 19349.0f)
	            ^ (uint32_t)(int)(fmodf(z, 97.0f) * 83492.0f);
	ix = (ix ^ (ix >> 13)) * 1274126177u;
	return (float)(ix & 0x7FFFFFFFu) / (float)0x7FFFFFFFu * 2.0f - 1.0f;
}

static float SmoothNoise(float x, float y, float z)
{
	float fx = x - floorf(x);
	float fy = y - floorf(y);
	float fz = z - floorf(z);
	float ix = floorf(x);
	float iy = floorf(y);
	float iz = floorf(z);

	// Trilinear interpolation of 8 corner hash values
	float c000 = HashFloat(ix, iy, iz);
	float c100 = HashFloat(ix + 1, iy, iz);
	float c010 = HashFloat(ix, iy + 1, iz);
	float c110 = HashFloat(ix + 1, iy + 1, iz);
	float c001 = HashFloat(ix, iy, iz + 1);
	float c101 = HashFloat(ix + 1, iy, iz + 1);
	float c011 = HashFloat(ix, iy + 1, iz + 1);
	float c111 = HashFloat(ix + 1, iy + 1, iz + 1);

	// Smoothstep interpolation
	float sx = fx * fx * (3.0f - 2.0f * fx);
	float sy = fy * fy * (3.0f - 2.0f * fy);
	float sz = fz * fz * (3.0f - 2.0f * fz);

	float x00 = c000 + sx * (c100 - c000);
	float x10 = c010 + sx * (c110 - c010);
	float x01 = c001 + sx * (c101 - c001);
	float x11 = c011 + sx * (c111 - c011);

	float xy0 = x00 + sy * (x10 - x00);
	float xy1 = x01 + sy * (x11 - x01);

	return xy0 + sz * (xy1 - xy0);
}

static float FBMNoise(float x, float y, float z, int octaves = 3)
{
	float value = 0.0f;
	float amplitude = 1.0f;
	float frequency = 1.0f;
	for (int i = 0; i < octaves; ++i)
	{
		value += amplitude * SmoothNoise(x * frequency, y * frequency, z * frequency);
		amplitude *= 0.5f;
		frequency *= 2.0f;
	}
	return value;
}

// ====================================================================
//                    1. IFS FRACTAL TEST
// ====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetIFSFractalTest)
{
	JPH_ADD_BASE_CLASS(WulfNetIFSFractalTest, Test)
}

void WulfNetIFSFractalTest::Initialize()
{
	// Start with Sierpinski Triangle 3D
	auto instructions = WulfNet::TransformPresets::GetPreset(WulfNet::IFSPreset::SierpinskiTriangle3D);
	mCurrentMatrices = WulfNet::TransformPresets::BuildMatrices(instructions);

	// Set up blender for morphing
	auto targetInstr = WulfNet::TransformPresets::GetPreset(WulfNet::IFSPreset::Vicsek3D);
	mBlender.SetSets(instructions, targetInstr);
	mCurrentPreset = 0;

	// Seed the chaos game with origin
	mPointX = 0.1f;
	mPointY = 0.1f;
	mPointZ = 0.1f;
	mPoints.clear();
	mPoints.reserve(mNumPoints);

	// Run a warm-up pass so points converge to the attractor
	RunChaosGame(mNumPoints + 200);

	// Create a floor for visual reference
	CreateFloor();
}

void WulfNetIFSFractalTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = inParams.mDeltaTime;

	// Morph between presets over time
	mMorphTimer += dt;
	if (mMorphTimer >= mMorphInterval)
	{
		mMorphTimer = 0.0f;
		mCurrentPreset = (mCurrentPreset + 1) % 5;

		WulfNet::IFSPreset presets[] = {
			WulfNet::IFSPreset::SierpinskiTriangle3D,
			WulfNet::IFSPreset::Vicsek3D,
			WulfNet::IFSPreset::SierpinskiCarpet3D,
			WulfNet::IFSPreset::SierpinskiTriangle2D,
			WulfNet::IFSPreset::Vicsek2D
		};

		auto target = WulfNet::TransformPresets::GetPreset(presets[mCurrentPreset]);
		mBlender.SwitchTarget(target);
	}

	// Smoothly blend transforms
	mBlender.Update(dt, 2.0f);
	mCurrentMatrices = mBlender.GetBlendedMatrices();

	// Re-run chaos game with current matrices so fractal morphs
	mPoints.clear();
	RunChaosGame(mNumPoints + 100);

	DrawFractal();
}

void WulfNetIFSFractalTest::RunChaosGame(int inIterations)
{
	if (mCurrentMatrices.empty())
		return;

	std::uniform_int_distribution<int> dist(0, (int)mCurrentMatrices.size() - 1);
	int warmup = 100; // skip first N points (haven't converged yet)

	for (int i = 0; i < inIterations; ++i)
	{
		int idx = dist(mRng);
		float nx, ny, nz;
		TransformPoint(mCurrentMatrices[idx], mPointX, mPointY, mPointZ, nx, ny, nz);
		mPointX = nx;
		mPointY = ny;
		mPointZ = nz;

		if (i >= warmup && (int)mPoints.size() < mNumPoints)
		{
			FractalPoint fp;
			fp.x = nx;
			fp.y = ny;
			fp.z = nz;
			fp.transformIndex = idx;
			mPoints.push_back(fp);
		}
	}
}

void WulfNetIFSFractalTest::DrawFractal()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer)
		return;

	for (const auto &p : mPoints)
	{
		// Color based on which transform created this point
		Color c = Color::sGetDistinctColor(p.transformIndex);

		// Make points glow — distinct bright colors with alpha
		c = Color(c, 220);

		// Position the fractal above the floor, scaled up for visibility
		float scale = 3.0f;
		RVec3 pos(p.x * scale, p.y * scale + 3.0f, p.z * scale);

		// Use small spheres for volumetric particle look
		mDebugRenderer->DrawSphere(pos, 0.025f, c, DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
	}

	// Title text
	mDebugRenderer->DrawText3D(RVec3(0, 6.5, 0), "IFS Fractal (CPU Chaos Game)", Color::sWhite, 0.4f);
#endif
}

// ====================================================================
//                    2. SMOKE & FIRE TEST
// ====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetSmokeTest)
{
	JPH_ADD_BASE_CLASS(WulfNetSmokeTest, Test)
}

void WulfNetSmokeTest::Initialize()
{
	mParticles.clear();
	mParticles.reserve(2000);
	mTime = 0.0f;
	mEmitAccum = 0.0f;

	// Create floor and a fire pit (visual only)
	CreateFloor();

	// Create a fire pit (dark box)
	BodyCreationSettings pitSettings(
		new BoxShape(Vec3(0.5f, 0.1f, 0.5f)),
		RVec3(0.0, 0.1, 0.0),
		Quat::sIdentity(),
		EMotionType::Static,
		Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(pitSettings, EActivation::DontActivate);
}

void WulfNetSmokeTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = inParams.mDeltaTime;
	mTime += dt;

	EmitParticles(dt);
	UpdateParticles(dt);
	DrawSmoke();
}

void WulfNetSmokeTest::EmitParticles(float inDt)
{
	// Clamp dt to prevent burst spawning on long frames (e.g. during loading)
	float clampedDt = std::min(inDt, 0.05f);
	mEmitAccum += mEmitRate * clampedDt;

	// Cap accumulator to prevent spawning too many particles at once
	mEmitAccum = std::min(mEmitAccum, 10.0f);

	std::uniform_real_distribution<float> unitDist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> lifeDist(2.5f, 5.0f);

	const int maxParticles = 800;
	while (mEmitAccum >= 1.0f && (int)mParticles.size() < maxParticles)
	{
		mEmitAccum -= 1.0f;

		SmokeParticle p;

		// Emit from a circular area
		float angle = unitDist(mRng) * 3.14159f;
		float radius = std::abs(unitDist(mRng)) * mEmitterRadius;

		p.x = (float)mEmitterPos.GetX() + cosf(angle) * radius;
		p.y = (float)mEmitterPos.GetY();
		p.z = (float)mEmitterPos.GetZ() + sinf(angle) * radius;

		// Initial velocity: mostly upward with a bit of spread
		p.vx = unitDist(mRng) * 0.3f;
		p.vy = 1.5f + std::abs(unitDist(mRng)) * 0.8f;
		p.vz = unitDist(mRng) * 0.3f;

		p.temperature = 1.0f;      // starts as fire
		p.maxLife = lifeDist(mRng);
		p.life = p.maxLife;
		p.size = 0.04f;            // starts small
		p.turbSeed = unitDist(mRng) * 10.0f;  // Keep small to avoid hash overflow

		mParticles.push_back(p);
	}

	// If we hit the cap, drain the accumulator to prevent stalling
	if ((int)mParticles.size() >= maxParticles)
		mEmitAccum = 0.0f;
}

void WulfNetSmokeTest::UpdateParticles(float inDt)
{
	for (int i = (int)mParticles.size() - 1; i >= 0; --i)
	{
		SmokeParticle &p = mParticles[i];

		// Age the particle
		p.life -= inDt;
		if (p.life <= 0.0f)
		{
			// Remove by swap with last
			mParticles[i] = mParticles.back();
			mParticles.pop_back();
			continue;
		}

		float lifeRatio = p.life / p.maxLife;  // 1 = newborn, 0 = dying

		// Cool down: temperature decays over time
		p.temperature = std::max(0.0f, p.temperature - mCoolingRate * inDt);

		// Buoyancy: hot particles rise faster
		float buoyancyForce = mBuoyancy * (0.3f + p.temperature * 0.7f);
		p.vy += buoyancyForce * inDt;

		// Turbulence: 3D noise displaces the particle
		float turbScale = 1.5f;
		float noiseX = FBMNoise(p.x * turbScale + p.turbSeed, p.y * turbScale, mTime * 0.5f);
		float noiseZ = FBMNoise(p.x * turbScale, p.y * turbScale + p.turbSeed, mTime * 0.5f + 50.0f);
		float noiseY = FBMNoise(p.x * turbScale + 100.0f, mTime * 0.3f, p.z * turbScale);

		float turbStrength = mTurbulence * (1.0f - p.temperature * 0.5f); // smoke gets more turbulent
		p.vx += noiseX * turbStrength * inDt;
		p.vy += noiseY * turbStrength * 0.3f * inDt;
		p.vz += noiseZ * turbStrength * inDt;

		// Drag: slow down over time
		float dragFactor = 1.0f / (1.0f + mDrag * inDt);
		p.vx *= dragFactor;
		p.vy *= dragFactor;
		p.vz *= dragFactor;

		// Integrate position
		p.x += p.vx * inDt;
		p.y += p.vy * inDt;
		p.z += p.vz * inDt;

		// Kill particles that have gone NaN or flown too far
		if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)
		    || std::abs(p.x) > 50.0f || p.y > 50.0f || p.y < -1.0f)
		{
			mParticles[i] = mParticles.back();
			mParticles.pop_back();
			continue;
		}

		// Grow: smoke expands as it rises and cools
		float targetSize;
		if (p.temperature > 0.6f)
			targetSize = 0.05f + (1.0f - p.temperature) * 0.1f;      // fire: small
		else if (p.temperature > 0.2f)
			targetSize = 0.1f + (0.6f - p.temperature) * 0.4f;       // ember/transition
		else
			targetSize = 0.2f + (1.0f - lifeRatio) * 0.25f;          // smoke: big billowy

		p.size += (targetSize - p.size) * 3.0f * inDt;  // smooth lerp
	}
}

void WulfNetSmokeTest::DrawSmoke()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer)
		return;

	// Sort particles back-to-front would be ideal for alpha blending,
	// but for debug rendering we just draw in order — still looks good

	for (const auto &p : mParticles)
	{
		float lifeRatio = p.life / p.maxLife;

		uint8 r, g, b, a;

		if (p.temperature > 0.7f)
		{
			// HOT FIRE CORE: bright yellow-white
			float t = (p.temperature - 0.7f) / 0.3f;  // 0..1
			r = (uint8)(255);
			g = (uint8)(200 + (int)(55 * t));
			b = (uint8)(50 + (int)(150 * t));
			a = (uint8)(220);
		}
		else if (p.temperature > 0.4f)
		{
			// FLAME: orange to dark orange
			float t = (p.temperature - 0.4f) / 0.3f;  // 0..1
			r = (uint8)(200 + (int)(55 * t));
			g = (uint8)(80 + (int)(120 * t));
			b = (uint8)(10 + (int)(40 * t));
			a = (uint8)(200);
		}
		else if (p.temperature > 0.15f)
		{
			// EMBERS: dark red/brown
			float t = (p.temperature - 0.15f) / 0.25f;
			r = (uint8)(80 + (int)(120 * t));
			g = (uint8)(30 + (int)(50 * t));
			b = (uint8)(10 + (int)(10 * t));
			a = (uint8)(160 + (int)(40 * t));
		}
		else
		{
			// SMOKE: dark gray, fading out
			float fade = lifeRatio;  // fade as life decreases
			r = (uint8)(40 + (int)(30 * p.temperature / 0.15f));
			g = (uint8)(40 + (int)(25 * p.temperature / 0.15f));
			b = (uint8)(45 + (int)(20 * p.temperature / 0.15f));
			a = (uint8)(std::max(0.0f, 140.0f * fade));
		}

		Color c(r, g, b, a);
		RVec3 pos(p.x, p.y, p.z);

		mDebugRenderer->DrawSphere(pos, p.size, c, DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
	}

	// Draw emitter glow (pulsing bright spot at the fire base)
	float pulse = 0.7f + 0.3f * sinf(mTime * 8.0f);
	uint8 glowA = (uint8)(180 * pulse);
	Color glowColor(255, 180, 30, glowA);
	mDebugRenderer->DrawSphere(mEmitterPos, 0.3f * pulse, glowColor, DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);

	// Secondary inner glow
	Color coreColor(255, 255, 200, (uint8)(120 * pulse));
	mDebugRenderer->DrawSphere(mEmitterPos + RVec3(0, 0.1, 0), 0.15f * pulse, coreColor, DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);

	mDebugRenderer->DrawText3D(RVec3(0, 7, 0), "Smoke & Fire", Color::sWhite, 0.4f);
#endif
}

// ====================================================================
//                    3. OCCLUSION CULLING TEST
// ====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetOcclusionTest)
{
	JPH_ADD_BASE_CLASS(WulfNetOcclusionTest, Test)
}

void WulfNetOcclusionTest::Initialize()
{
	CreateFloor();

	// Create an occluding wall
	BodyCreationSettings wallSettings(
		new BoxShape(Vec3(3.0f, 2.0f, 0.2f)),
		RVec3(0, 2, 0),
		Quat::sIdentity(),
		EMotionType::Static,
		Layers::NON_MOVING);
	mWallBody = mBodyInterface->CreateAndAddBody(wallSettings, EActivation::DontActivate);

	// Create objects behind the wall (negative Z side)
	std::mt19937 rng(456);
	std::uniform_real_distribution<float> posDist(-2.5f, 2.5f);
	std::uniform_real_distribution<float> heightDist(0.5f, 3.5f);

	for (int i = 0; i < 12; ++i)
	{
		RVec3 pos(posDist(rng), heightDist(rng), -2.0f - std::abs(posDist(rng)));
		mOccludeePositions.push_back(pos);

		BodyCreationSettings sphereSettings(
			new SphereShape(0.3f),
			pos,
			Quat::sIdentity(),
			EMotionType::Static,
			Layers::NON_MOVING);
		BodyID id = mBodyInterface->CreateAndAddBody(sphereSettings, EActivation::DontActivate);
		mOccludees.push_back(id);
	}

	// Create some objects in FRONT of the wall (visible, positive Z)
	for (int i = 0; i < 6; ++i)
	{
		RVec3 pos(posDist(rng), heightDist(rng), 2.0f + std::abs(posDist(rng)));
		mOccludeePositions.push_back(pos);

		BodyCreationSettings sphereSettings(
			new SphereShape(0.3f),
			pos,
			Quat::sIdentity(),
			EMotionType::Static,
			Layers::NON_MOVING);
		BodyID id = mBodyInterface->CreateAndAddBody(sphereSettings, EActivation::DontActivate);
		mOccludees.push_back(id);
	}
}

void WulfNetOcclusionTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer)
		return;

	mCameraPos = RVec3(inParams.mCameraState.mPos);

	// Simple CPU occlusion test: check if object center is behind the wall
	// relative to camera. The wall is at z=0, thickness 0.4.
	// If camera is at +Z and object is at -Z, it's occluded.
	float wallZ = 0.0f;
	float wallHalfThick = 0.2f;
	float camZ = (float)mCameraPos.GetZ();

	for (size_t i = 0; i < mOccludeePositions.size(); ++i)
	{
		RVec3 pos = mOccludeePositions[i];
		float objZ = (float)pos.GetZ();
		float objX = (float)pos.GetX();

		// Check if object is on opposite side of wall from camera
		// and within the wall's X extent (-3..3)
		bool behindWall = false;
		if (std::abs(objX) < 3.0f)
		{
			if (camZ > wallZ + wallHalfThick && objZ < wallZ - wallHalfThick)
				behindWall = true;
			else if (camZ < wallZ - wallHalfThick && objZ > wallZ + wallHalfThick)
				behindWall = true;
		}

		if (behindWall)
		{
			// Occluded — draw as red wireframe
			mDebugRenderer->DrawWireSphere(pos, 0.35f, Color::sRed, 2);
			mDebugRenderer->DrawText3D(pos + RVec3(0, 0.5, 0), "OCCLUDED", Color::sRed, 0.2f);
		}
		else
		{
			// Visible — draw as green solid
			mDebugRenderer->DrawSphere(pos, 0.3f, Color::sGreen, DebugRenderer::ECastShadow::On, DebugRenderer::EDrawMode::Solid);
			mDebugRenderer->DrawText3D(pos + RVec3(0, 0.5, 0), "VISIBLE", Color::sGreen, 0.2f);
		}
	}

	// Label the wall
	mDebugRenderer->DrawText3D(RVec3(0, 4.5, 0), "Occlusion Wall", Color::sYellow, 0.4f);
#endif
}

// ====================================================================
//                    4. FALLING BODIES + FRACTAL
// ====================================================================

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFallingFractalTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFallingFractalTest, Test)
}

void WulfNetFallingFractalTest::Initialize()
{
	CreateFloor();

	// Build Vicsek 3D fractal matrices
	auto instructions = WulfNet::TransformPresets::GetPreset(WulfNet::IFSPreset::Vicsek3D);
	mMatrices = WulfNet::TransformPresets::BuildMatrices(instructions);

	// Seed the chaos game
	mPointX = 0.1f;
	mPointY = 0.1f;
	mPointZ = 0.1f;
	mPoints.clear();
	mPoints.reserve(mNumPoints);

	// Warm up the chaos game
	RunChaosGame(mNumPoints + 200);
}

void WulfNetFallingFractalTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	float dt = inParams.mDeltaTime;

	// Spawn falling bodies periodically
	mSpawnTimer += dt;
	if (mSpawnTimer >= mSpawnInterval && (int)mBodies.size() < mMaxBodies)
	{
		mSpawnTimer = 0.0f;
		SpawnBody();
	}

	// Regenerate fractal points each frame (they orbit slightly)
	mPoints.clear();
	RunChaosGame(mNumPoints + 100);

	DrawFractal();
}

void WulfNetFallingFractalTest::RunChaosGame(int inIterations)
{
	if (mMatrices.empty())
		return;

	std::uniform_int_distribution<int> dist(0, (int)mMatrices.size() - 1);
	int warmup = 100;

	for (int i = 0; i < inIterations; ++i)
	{
		int idx = dist(mRng);
		float nx, ny, nz;
		TransformPoint(mMatrices[idx], mPointX, mPointY, mPointZ, nx, ny, nz);
		mPointX = nx;
		mPointY = ny;
		mPointZ = nz;

		if (i >= warmup && (int)mPoints.size() < mNumPoints)
		{
			FractalPoint fp;
			fp.x = nx;
			fp.y = ny;
			fp.z = nz;
			fp.transformIndex = idx;
			mPoints.push_back(fp);
		}
	}
}

void WulfNetFallingFractalTest::DrawFractal()
{
#ifdef JPH_DEBUG_RENDERER
	if (!mDebugRenderer)
		return;

	for (const auto &p : mPoints)
	{
		Color c = Color::sGetDistinctColor(p.transformIndex);
		c = Color(c, 160);  // semi-transparent so bodies show through

		float scale = 4.0f;
		RVec3 pos(p.x * scale, p.y * scale + 3.0f, p.z * scale);

		mDebugRenderer->DrawSphere(pos, 0.02f, c, DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
	}

	mDebugRenderer->DrawText3D(RVec3(0, 8, 0), "Falling Bodies + Vicsek Fractal", Color::sWhite, 0.35f);
#endif
}

void WulfNetFallingFractalTest::SpawnBody()
{
	std::uniform_real_distribution<float> posDist(-1.5f, 1.5f);
	std::uniform_real_distribution<float> sizeDist(0.15f, 0.4f);
	std::uniform_int_distribution<int> shapeDist(0, 1);

	float x = posDist(mRng);
	float z = posDist(mRng);
	float sz = sizeDist(mRng);

	BodyCreationSettings settings(
		shapeDist(mRng) == 0
			? static_cast<const Shape *>(new SphereShape(sz))
			: static_cast<const Shape *>(new BoxShape(Vec3(sz, sz, sz))),
		RVec3(x, 10.0f, z),
		Quat::sIdentity(),
		EMotionType::Dynamic,
		Layers::MOVING);
	settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
	settings.mMassPropertiesOverride.mMass = 1.0f;

	BodyID id = mBodyInterface->CreateAndAddBody(settings, EActivation::Activate);
	mBodies.push_back(id);
}
