// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2024 Jorrit Rouwe
// SPDX-License-Identifier: MIT
// Modified: WulfNet V3 Water Physics Integration

#pragma once

#include <Tests/Test.h>
#include <WulfNet/Physics/WaterSystemV3.h>

class BoatTest : public Test
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, BoatTest)

	// Description of the test
	virtual const char *		GetDescription() const override
	{
		return "Shows how a boat can be driven on WulfNet V3 SWE water with dynamic waves.";
	}

	// See: Test
	virtual void				Initialize() override;
	virtual void				ProcessInput(const ProcessInputParams &inParams) override;
	virtual void				PrePhysicsUpdate(const PreUpdateParams &inParams) override;
	virtual void				SaveInputState(StateRecorder &inStream) const override;
	virtual void				RestoreInputState(StateRecorder &inStream) override;
	virtual void				GetInitialCamera(CameraState &ioState) const override;
	virtual RMat44				GetCameraPivot(float inCameraHeading, float inCameraPitch) const override { return mCameraPivot; }

private:
	void						UpdateCameraPivot();

	// Configuration
	static constexpr float		cHalfBoatLength = 4.0f;
	static constexpr float		cHalfBoatTopWidth = 1.5f;
	static constexpr float		cHalfBoatBottomWidth = 1.2f;
	static constexpr float		cBoatBowLength = 2.0f;
	static constexpr float		cHalfBoatHeight = 0.75f;

	static constexpr float		cBoatMass = 1000.0f;
	static constexpr float		cBarrelMass = 50.0f;

	static constexpr float		cForwardAcceleration = 15.0f;
	static constexpr float		cSteerAcceleration = 1.5f;

	// WulfNet V3 Water System
	WulfNet::Physics::WaterSystemV3 *	mWaterSystem = nullptr;

	// The boat
	Body *						mBoatBody = nullptr;

	// All floating bodies (boat + barrels)
	std::vector<BodyID>			mFloatingBodies;

	// The camera pivot, recorded before the physics update to align with the drawn world
	RMat44						mCameraPivot = RMat44::sIdentity();

	// Time
	float						mTime = 0.0f;

	// Player input
	float						mForward = 0.0f;
	float						mRight = 0.0f;
};
