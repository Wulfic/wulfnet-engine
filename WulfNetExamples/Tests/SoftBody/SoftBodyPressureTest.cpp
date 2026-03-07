// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2023 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Samples.h>

#include <Tests/SoftBody/SoftBodyPressureTest.h>
#include <WulfNet/Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Utils/SoftBodyCreator.h>
#include <SamplesLayers.h>

JPH_IMPLEMENT_RTTI_VIRTUAL(SoftBodyPressureTest)
{
	JPH_ADD_BASE_CLASS(SoftBodyPressureTest, Test)
}

void SoftBodyPressureTest::Initialize()
{
	// Floor
	CreateFloor();

	// Bodies with increasing pressure
	SoftBodyCreationSettings sphere(SoftBodyCreator::CreateSphere(2.0f), RVec3::sZero(), Quat::sIdentity(), Layers::MOVING);

	for (int i = 0; i <= 10; ++i)
	{
		sphere.mPosition = RVec3(-50.0f + i * 10.0f, 10.0f, 0);
		sphere.mPressure = 1000.0f * i;
		BodyID id = mBodyInterface->CreateAndAddSoftBody(sphere, EActivation::Activate);
		SetBodyLabel(id, StringFormat("Pressure: %g", double(sphere.mPressure)));
	}
}
