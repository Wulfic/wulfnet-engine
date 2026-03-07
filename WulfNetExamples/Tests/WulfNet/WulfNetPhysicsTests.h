// SPDX-License-Identifier: MIT
// WulfNet Physics Integration Visual Tests
// - CPU Chaos Game fractals (no GPU shaders required)
// - Volumetric smoke & fire simulation
// - Occlusion culling visualization
// - Falling bodies with fractal background

#pragma once

#include <Tests/Test.h>
#include <WulfNet/Procedural/IFS/AffineTransform.h>
#include <WulfNet/Procedural/IFS/TransformPresets.h>
#include <WulfNet/Procedural/IFS/TransformBlender.h>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// A single point in the fractal attractor (position + color weight)
// ---------------------------------------------------------------------------
struct FractalPoint
{
	float x = 0.0f, y = 0.0f, z = 0.0f;
	int   transformIndex = 0;   // which transform last touched this point
};

// ===========================================================================
// 1. IFS Fractal Test — pure CPU chaos game, no GPU/.spv dependency
// ===========================================================================
class WulfNetIFSFractalTest : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetIFSFractalTest)

public:
	const char *			GetDescription() const override
	{
		return "CPU chaos-game IFS fractal. Morphs between presets using TransformBlender.";
	}

	void					Initialize() override;
	void					PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
	void					RunChaosGame(int inIterations);
	void					DrawFractal();

	// Chaos game state
	std::vector<FractalPoint>	mPoints;
	int							mNumPoints		= 10000;
	float						mPointX			= 0.0f;
	float						mPointY			= 0.0f;
	float						mPointZ			= 0.0f;

	// Transform matrices (built from presets on CPU)
	std::vector<WulfNet::Mat4>		mCurrentMatrices;
	WulfNet::TransformBlender			mBlender;
	float						mMorphTimer		= 0.0f;
	float						mMorphInterval	= 4.0f;     // seconds between preset switches
	int							mCurrentPreset	= 0;

	std::mt19937				mRng{ 42 };
};

// ===========================================================================
// 2. Smoke & Fire Test — volumetric particle simulation
// ===========================================================================
struct SmokeParticle
{
	float x, y, z;              // position
	float vx, vy, vz;          // velocity
	float temperature;          // 1.0 = hot fire core, 0.0 = cold smoke
	float life;                 // remaining lifetime [0..maxLife]
	float maxLife;              // initial lifetime
	float size;                 // rendered radius
	float turbSeed;             // per-particle turbulence offset
};

class WulfNetSmokeTest : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetSmokeTest)

public:
	const char *			GetDescription() const override
	{
		return "Volumetric smoke & fire with buoyancy, turbulence, and alpha-blended spheres.";
	}

	void					Initialize() override;
	void					PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
	void					EmitParticles(float inDt);
	void					UpdateParticles(float inDt);
	void					DrawSmoke();

	std::vector<SmokeParticle>	mParticles;

	// Emitter settings
	float					mEmitRate		= 120.0f;    // particles/sec
	float					mEmitAccum		= 0.0f;
	RVec3					mEmitterPos		= RVec3(0.0, 0.5, 0.0);
	float					mEmitterRadius	= 0.25f;

	// Physics tuning
	float					mBuoyancy		= 2.8f;
	float					mCoolingRate	= 0.35f;
	float					mTurbulence		= 1.2f;
	float					mDrag			= 1.0f;

	float					mTime			= 0.0f;
	std::mt19937			mRng{ 123 };
};

// ===========================================================================
// 3. Occlusion Culling Visualization Test
// ===========================================================================
class WulfNetOcclusionTest : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetOcclusionTest)

public:
	const char *			GetDescription() const override
	{
		return "CPU occlusion culling demo: objects behind the wall are drawn in wireframe.";
	}

	void					Initialize() override;
	void					PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
	// Wall body
	BodyID					mWallBody;

	// Occludee bodies (small spheres behind the wall)
	std::vector<BodyID>		mOccludees;
	std::vector<RVec3>		mOccludeePositions;

	// Camera for occlusion queries
	RVec3					mCameraPos = RVec3(0, 3, 10);
};

// ===========================================================================
// 4. Falling Bodies + Fractal Background
// ===========================================================================
class WulfNetFallingFractalTest : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFallingFractalTest)

public:
	const char *			GetDescription() const override
	{
		return "Dynamic rigid bodies falling through a CPU-rendered 3D fractal.";
	}

	void					Initialize() override;
	void					PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
	void					RunChaosGame(int inIterations);
	void					DrawFractal();
	void					SpawnBody();

	// Fractal
	std::vector<FractalPoint>	mPoints;
	int							mNumPoints		= 6000;
	float						mPointX			= 0.0f;
	float						mPointY			= 0.0f;
	float						mPointZ			= 0.0f;
	std::vector<WulfNet::Mat4>		mMatrices;

	// Falling bodies
	std::vector<BodyID>			mBodies;
	float						mSpawnTimer		= 0.0f;
	float						mSpawnInterval	= 0.4f;
	int							mMaxBodies		= 30;

	std::mt19937				mRng{ 77 };
};
