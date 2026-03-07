#pragma once

#include <Tests/Test.h>
#include <WulfNet/Physics/Fluids/FluidSystem.h>
#include <WulfNet/Core/System/SystemMonitor.h>

// Isolated water observation test: a visible glass box with a water emitter.
// Designed to debug/verify water depth rendering, SWE physics, and buoyancy
// in a controlled environment.
class WaterBoxTest : public Test
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WaterBoxTest)

	virtual const char *	GetDescription() const override
	{
		return "Glass box with water emitter — observe depth, waves, and buoyancy up close.";
	}

	virtual void			Initialize() override;
	virtual void			PrePhysicsUpdate(const PreUpdateParams &inParams) override;
	virtual void			GetInitialCamera(CameraState &ioState) const override;
	virtual String			GetStatusString() const override;

private:
	// Water system (small, high-res grid)
	WulfNet::FluidSystem *	mWaterSystem = nullptr;
	float								mTime = 0.0f;
	std::vector<JPH::BodyID>			mFloatingBodies;

	// Glass box wall bodies (static, for containment)
	std::vector<JPH::BodyID>			mWallBodies;

	// Emitter state
	float								mEmitterRate = 8.0f;    // volume units per second
	float								mEmitterAccum = 0.0f;   // accumulator for fractional adds
	bool								mEmitterActive = true;

	// Performance monitoring
	float								mStatsTimer = 0.0f;
	float								mCurrentFPS = 0.0f;
	float								mFrameTimeMs = 0.0f;
	int									mFrameCount = 0;

	// Grid constants — small box, high resolution
	static constexpr uint32_t			cGridW = 128;
	static constexpr uint32_t			cGridH = 128;
	static constexpr float				cGridSize = 0.25f;  // 0.25m per cell → 32×32 m area

	// Glass box dimensions (in world meters, centered at origin)
	static constexpr float				cBoxHalfExtent = 10.0f;  // 20×20 m inner area
	static constexpr float				cBoxWallHeight = 8.0f;   // 8m tall walls
	static constexpr float				cBoxWallThick  = 0.15f;   // thin glass walls

	// Helper: render the glass box walls
	void RenderGlassBox();
};
