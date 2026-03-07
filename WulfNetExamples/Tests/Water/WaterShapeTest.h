// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT
// Modified: WulfNet V3 Water Physics Integration

#pragma once

#include <Tests/Test.h>
#include <WulfNet/Physics/Fluids/FluidSystem.h>

class WaterShapeTest : public Test
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WaterShapeTest)

	// Description of the test
	virtual const char *	GetDescription() const override
	{
		return "Shows buoyancy of various shapes using WulfNet V3 SWE water physics.";
	}

	// Initialize the test
	virtual void			Initialize() override;

	// Update the test, called before the physics update
	virtual void			PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
	WulfNet::FluidSystem *	mWaterSystem = nullptr;
	float								mTime = 0.0f;
	std::vector<JPH::BodyID>			mFloatingBodies;
};
