// SPDX-License-Identifier: MIT
// WulfNet Advanced Visual Tests
// - Gaseous dynamics (fire / smoke volumetric grid)
// - Voronoi destruction of rigid bodies
// - Terrain deformation (craters, tire tracks)
// - Volumetric fog / cloud rendering
// - Spatial audio & acoustic ray visualization

#pragma once

#include <Tests/Test.h>

// WulfNet subsystems
#include <WulfNet/Physics/Gaseous/GaseousSystem.h>
#include <WulfNet/Physics/Destruction/DestructionSystem.h>
#include <WulfNet/Physics/Terrain/TerrainDeformation.h>
#include <WulfNet/Audio/Acoustics/AcousticSystem.h>
#include <WulfNet/Audio/Spatial/SpatialAudio.h>

#include <vector>
#include <random>

// ===========================================================================
// 1. Gaseous Dynamics — real-time smoke / fire grid visualization
// ===========================================================================
class WulfNetGasTest : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetGasTest)

public:
	const char *GetDescription() const override
	{
		return "Gaseous dynamics: fire/smoke Euler grid with density/temperature visualization.";
	}

	void Initialize() override;
	void PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
	void DrawGasField();

	WulfNet::GaseousSystem		mGas;
	WulfNet::GaseousSystemConfig mGasConfig;
	float						mTime = 0.0f;
};

// ===========================================================================
// 2. Destruction — Voronoi fracture of rigid bodies
// ===========================================================================
class WulfNetDestructionVisualTest : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetDestructionVisualTest)

public:
	const char *GetDescription() const override
	{
		return "Voronoi fracture: launch projectiles at destructible walls.";
	}

	void Initialize() override;
	void PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
	void DrawFracturePatterns();
	void LaunchProjectile();

	WulfNet::DestructionSystem	mDestruction;
	struct DestrWall
	{
		JPH::BodyID			bodyId;
		uint32_t			destructHandle = 0;
		float				halfX = 0, halfY = 0, halfZ = 0;
		RVec3				pos;
	};
	std::vector<DestrWall>		mWalls;
	std::vector<JPH::BodyID>	mProjectiles;
	float						mLaunchTimer	= 0.0f;
	float						mLaunchInterval	= 2.5f;
	int							mMaxProjectiles	= 8;
	std::mt19937				mRng{ 99 };
};

// ===========================================================================
// 3. Terrain Deformation — craters, tracks, explosions
// ===========================================================================
class WulfNetTerrainVisualTest : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetTerrainVisualTest)

public:
	const char *GetDescription() const override
	{
		return "Terrain heightfield deformation: craters and tire tracks.";
	}

	void Initialize() override;
	void PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
	void DrawTerrain();

	WulfNet::TerrainDeformation	mTerrain;
	WulfNet::TerrainDeformConfig mTerrainConfig;
	float						mDropTimer		= 0.0f;
	float						mTrackAngle		= 0.0f;
	float						mTime			= 0.0f;
	std::mt19937				mRng{ 55 };
};

// ===========================================================================
// 4. Volumetric Clouds — gaseous grid + volumetric ray march rendering
// ===========================================================================
class WulfNetVolumetricVisualTest : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetVolumetricVisualTest)

public:
	const char *GetDescription() const override
	{
		return "Volumetric rendering: density-field fog / cloud with emission coloring.";
	}

	void Initialize() override;
	void PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
	void DrawVolumetricField();

	WulfNet::GaseousSystem		mGas;
	WulfNet::GaseousSystemConfig mGasConfig;
	float						mTime = 0.0f;
};

// ===========================================================================
// 5. Spatial Audio & Acoustics — ray-traced reflections visualization
// ===========================================================================
class WulfNetAudioVisualTest : public Test
{
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetAudioVisualTest)

public:
	const char *GetDescription() const override
	{
		return "Acoustic ray tracing & spatial audio: reflection paths, HRTF cones, attenuation.";
	}

	void Initialize() override;
	void PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
	void DrawAudioSources();
	void DrawAcousticRays();
	void DrawAttenuationRadii();

	WulfNet::AcousticSystem		mAcoustics;
	WulfNet::SpatialAudio		mSpatialAudio;

	// Room walls (physics bodies for visual + ray cast)
	std::vector<JPH::BodyID>	mRoomWalls;
	RVec3						mRoomMin = RVec3(-6, 0, -6);
	RVec3						mRoomMax = RVec3( 6, 5,  6);

	// Sound sources
	struct AudioSource
	{
		RVec3		position;
		Color		color;
		float		radius; // max audible distance
		const char *label;
	};
	std::vector<AudioSource>	mSources;

	// Listener
	RVec3						mListenerPos  = RVec3(0, 1.7, 0);
	RVec3						mListenerFwd  = RVec3(0, 0, -1);
	RVec3						mListenerUp   = RVec3(0, 1, 0);

	// Cached acoustic data for drawing
	struct SourceAcousticInfo
	{
		float occlusion		= 1.0f;
		float obstruction	= 1.0f;
		float distGain		= 1.0f;
		WulfNet::ImpulseResponse ir;
	};
	std::vector<SourceAcousticInfo>	mAcousticInfo;
	float						mUpdateTimer  = 0.0f;
	float						mTime		  = 0.0f;
};
