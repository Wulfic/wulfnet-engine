#pragma once

#include <Tests/Test.h>
#include <WulfNet/Physics/WaterSystemV3.h>
#include <WulfNet/Physics/Destruction/DestructionSystem.h>
#include <WulfNet/Core/System/SystemMonitor.h>
#include <random>

// Large-scale dam break simulation using WulfNet V3 SWE physics
// and WulfNet DestructionSystem for Voronoi-fracture dam walls.
// A massive ball auto-launches into the dam wall, shattering it,
// and releasing a 75m-deep reservoir into a 2.5km alpine valley.
class DamBreakTest : public Test
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, DamBreakTest)

	virtual const char *	GetDescription() const override
	{
		return "Destructible dam wall + 6.5km² SWE flood. Ball auto-launches to shatter the dam.";
	}

	virtual void			Initialize() override;
	virtual void			PrePhysicsUpdate(const PreUpdateParams &inParams) override;
	virtual void			GetInitialCamera(CameraState &ioState) const override;
	virtual String			GetStatusString() const override;

private:
	// Water system
	WulfNet::Physics::WaterSystemV3 *	mWaterSystem = nullptr;
	float								mTime = 0.0f;
	std::vector<JPH::BodyID>			mFloatingBodies;

	// Destruction system for dam wall
	WulfNet::DestructionSystem			mDestruction;

	// Physical dam wall segments
	struct DamSegment
	{
		JPH::BodyID		bodyId;
		uint32_t		destructHandle = 0;
		float			halfX = 0, halfY = 0, halfZ = 0;
		float			gxCenter = 0;   // grid-space X center of this segment
		bool			broken = false;
	};
	std::vector<DamSegment>				mDamSegments;

	// Auto-launched projectile
	JPH::BodyID							mProjectile;
	bool								mProjectileLaunched = false;
	float								mLaunchTime = 3.0f;

	std::mt19937						mRng{ 42 };

	// Performance monitoring
	float								mStatsTimer = 0.0f;
	float								mCurrentFPS = 0.0f;
	float								mFrameTimeMs = 0.0f;
	int									mFrameCount = 0;

	// Grid constants — true large scale
	static constexpr uint32_t			cGridW = 512;
	static constexpr uint32_t			cGridH = 512;
	static constexpr float				cGridSize = 5.0f;  // 5m per cell → 2.56km × 2.56km

	// Rendering subsample (render every Nth cell for performance)
	static constexpr uint32_t			cRenderStep = 2;
};
