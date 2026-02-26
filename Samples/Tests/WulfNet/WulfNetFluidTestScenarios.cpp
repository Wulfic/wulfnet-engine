// =============================================================================
// WulfNet Engine - CO-FLIP Fluid Test Scenarios
// =============================================================================
// River, Waterfall, Puddle, Lake, Viscosity, Buoyancy, Ragdoll Swim,
// and Cloth Water test scenario implementations.
// Extracted from WulfNetFluidTest.cpp for maintainability.
// =============================================================================

#include <Samples.h>
#include <Tests/WulfNet/WulfNetFluidTest.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>

// Register RTTI for scenario subclasses
JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetRiverTest)
{
	JPH_ADD_BASE_CLASS(WulfNetRiverTest, WulfNetFluidTest)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterfallTest)
{
	JPH_ADD_BASE_CLASS(WulfNetWaterfallTest, WulfNetFluidTest)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetPuddleTest)
{
	JPH_ADD_BASE_CLASS(WulfNetPuddleTest, WulfNetFluidTest)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetLakeTest)
{
	JPH_ADD_BASE_CLASS(WulfNetLakeTest, WulfNetFluidTest)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetViscosityTest)
{
	JPH_ADD_BASE_CLASS(WulfNetViscosityTest, WulfNetFluidTest)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetBuoyancyTest)
{
	JPH_ADD_BASE_CLASS(WulfNetBuoyancyTest, WulfNetFluidTest)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetRagdollSwimTest)
{
	JPH_ADD_BASE_CLASS(WulfNetRagdollSwimTest, WulfNetFluidTest)
}

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetClothWaterTest)
{
	JPH_ADD_BASE_CLASS(WulfNetClothWaterTest, WulfNetFluidTest)
}

// =============================================================================
// River Test Implementation
// =============================================================================

void WulfNetRiverTest::SetupFluid()
{
	// River — elongated domain, moderate resolution
	mFluidConfig.gridSizeX = 40;
	mFluidConfig.gridSizeY = 16;
	mFluidConfig.gridSizeZ = 20;
	mFluidConfig.cellSize = 0.22f;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize = mFluidConfig.cellSize;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Create initial water body
	CreateWaterBox(0.5f, 0.2f, 0.5f, 5.0f, 0.8f, 2.5f);

	// Emitter — halved rate from 200→100 (larger particles fill same volume)
	CreateEmitter(0.3f, 0.5f, 1.5f, 1.0f, 0.0f, 0.0f, 100.0f, 1.5f);

	CreateRiverChannel();
}

void WulfNetRiverTest::SetupObjects()
{
	// Add some floating debris
	for (int i = 0; i < 3; ++i)
	{
		BodyCreationSettings settings(
			new BoxShape(Vec3(0.15f, 0.08f, 0.15f)),
			RVec3(1.5f + i * 1.0f, 1.2f, 1.5f),
			Quat::sIdentity(),
			EMotionType::Dynamic,
			Layers::MOVING);
		settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		settings.mMassPropertiesOverride.mMass = 0.3f;  // Light for floating
		mBodyInterface->CreateAndAddBody(settings, EActivation::Activate);
	}
}

void WulfNetRiverTest::CreateRiverChannel()
{
	// Create banks with Jolt bodies
	BodyCreationSettings leftBank(
		new BoxShape(Vec3(4.0f, 0.5f, 0.2f)),
		RVec3(3.5f, 0.5f, 0.2f),
		Quat::sIdentity(),
		EMotionType::Static,
		Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(leftBank, EActivation::DontActivate);

	BodyCreationSettings rightBank(
		new BoxShape(Vec3(4.0f, 0.5f, 0.2f)),
		RVec3(3.5f, 0.5f, 2.8f),
		Quat::sIdentity(),
		EMotionType::Static,
		Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(rightBank, EActivation::DontActivate);

	// Mark banks as solid in fluid sim
	AddSolidBox(0.0f, 0.0f, 0.0f, 7.5f, 1.0f, 0.4f);
	AddSolidBox(0.0f, 0.0f, 2.6f, 7.5f, 1.0f, 3.0f);
}

// =============================================================================
// Waterfall Test Implementation
// =============================================================================

void WulfNetWaterfallTest::SetupFluid()
{
	// Waterfall — needs vertical space, moderate XZ
	mFluidConfig.gridSizeX = 28;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 28;
	mFluidConfig.cellSize = 0.18f;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize = mFluidConfig.cellSize;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Create pool at bottom
	CreateWaterBox(1.0f, 0.2f, 1.0f, 3.5f, 0.6f, 3.5f);

	// Emitter — halved from 300→150
	CreateEmitter(2.3f, 3.5f, 2.3f, 0.0f, -1.0f, 0.0f, 150.0f, 0.5f);
}

void WulfNetWaterfallTest::SetupObjects()
{
	// Add cliff/ledge
	BodyCreationSettings cliff(
		new BoxShape(Vec3(1.5f, 0.2f, 1.5f)),
		RVec3(2.3f, 3.2f, 2.3f),
		Quat::sIdentity(),
		EMotionType::Static,
		Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(cliff, EActivation::DontActivate);
	AddSolidBox(0.8f, 3.0f, 0.8f, 3.8f, 3.4f, 3.8f);

	// Pool walls
	BodyCreationSettings poolWall(
		new BoxShape(Vec3(0.1f, 0.5f, 1.5f)),
		RVec3(0.9f, 0.4f, 2.3f),
		Quat::sIdentity(),
		EMotionType::Static,
		Layers::NON_MOVING);
	mBodyInterface->CreateAndAddBody(poolWall, EActivation::DontActivate);
}

// =============================================================================
// Puddle Test Implementation
// =============================================================================

void WulfNetPuddleTest::SetupFluid()
{
	// Small puddle — tiny domain, moderate cells
	mFluidConfig.gridSizeX = 20;
	mFluidConfig.gridSizeY = 12;
	mFluidConfig.gridSizeZ = 20;
	mFluidConfig.cellSize = 0.14f;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize = mFluidConfig.cellSize;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Create small puddle
	CreateWaterBox(0.8f, 0.1f, 0.8f, 1.8f, 0.3f, 1.8f);

	// Rain drops — reduced from 50→30
	CreateEmitter(1.3f, 1.2f, 1.3f, 0.0f, -1.0f, 0.0f, 30.0f, 0.2f);
}

// =============================================================================
// Lake Test Implementation
// =============================================================================

void WulfNetLakeTest::SetupFluid()
{
	// Large lake — 10× volume vs original, coarser cells for massive scale
	mFluidConfig.gridSizeX = 80;
	mFluidConfig.gridSizeY = 24;
	mFluidConfig.gridSizeZ = 80;
	mFluidConfig.cellSize = 0.25f;
	mFluidConfig.pressureIterations = 20;  // SOR converges fast

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;  // Reduced from 3.5 for performance

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Create large body of water (~10× original volume)
	// Original: 7.5 × 1.3 × 7.5 = 73.125 m³
	// New:      16.0 × 2.8 × 16.0 = 716.8 m³ (~9.8× larger)
	CreateWaterBox(1.0f, 0.2f, 1.0f, 17.0f, 3.0f, 17.0f);
}

void WulfNetLakeTest::SetupObjects()
{
	// Add boat
	BodyCreationSettings boat(
		new BoxShape(Vec3(0.6f, 0.15f, 0.25f)),
		RVec3(4.0f, 1.8f, 4.0f),
		Quat::sIdentity(),
		EMotionType::Dynamic,
		Layers::MOVING);
	boat.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
	boat.mMassPropertiesOverride.mMass = 2.0f;
	mBodyInterface->CreateAndAddBody(boat, EActivation::Activate);

	// Add some balls to splash
	for (int i = 0; i < 3; ++i)
	{
		BodyCreationSettings ball(
			new SphereShape(0.15f),
			RVec3(3.0f + i * 0.8f, 2.5f, 4.5f),
			Quat::sIdentity(),
			EMotionType::Dynamic,
			Layers::MOVING);
		ball.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		ball.mMassPropertiesOverride.mMass = 0.5f;
		mBodyInterface->CreateAndAddBody(ball, EActivation::Activate);
	}
}

// =============================================================================
// Viscosity Test Implementation
// =============================================================================

void WulfNetViscosityTest::SetupFluid()
{
	// Viscosity comparison — moderate grid
	mFluidConfig.gridSizeX = 36;
	mFluidConfig.gridSizeY = 20;
	mFluidConfig.gridSizeZ = 20;
	mFluidConfig.cellSize = 0.2f;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize = mFluidConfig.cellSize;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Create water blob (will flow quickly)
	CreateWaterSphere(2.0f, 1.5f, 1.5f, 0.5f);

	// TODO: Add different viscosity materials when CO-FLIP supports per-particle viscosity
}

// =============================================================================
// Buoyancy Test Implementation
// =============================================================================

void WulfNetBuoyancyTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 28;
	mFluidConfig.gridSizeY = 20;
	mFluidConfig.gridSizeZ = 28;
	mFluidConfig.cellSize = 0.22f;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize = mFluidConfig.cellSize;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Create pool
	CreateWaterBox(1.0f, 0.2f, 1.0f, 4.5f, 2.0f, 4.5f);
}

void WulfNetBuoyancyTest::SetupObjects()
{
	// Light object (wood - floats)
	{
		BodyCreationSettings settings(
			new BoxShape(Vec3(0.2f, 0.2f, 0.2f)),
			RVec3(2.0f, 2.5f, 2.0f),
			Quat::sIdentity(),
			EMotionType::Dynamic,
			Layers::MOVING);
		settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		settings.mMassPropertiesOverride.mMass = 0.2f;  // Light
		mBodyInterface->CreateAndAddBody(settings, EActivation::Activate);
	}

	// Medium object (plastic - partially submerged)
	{
		BodyCreationSettings settings(
			new SphereShape(0.15f),
			RVec3(2.7f, 2.5f, 2.7f),
			Quat::sIdentity(),
			EMotionType::Dynamic,
			Layers::MOVING);
		settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		settings.mMassPropertiesOverride.mMass = 0.8f;  // Medium
		mBodyInterface->CreateAndAddBody(settings, EActivation::Activate);
	}

	// Heavy object (metal - sinks)
	{
		BodyCreationSettings settings(
			new SphereShape(0.12f),
			RVec3(3.4f, 2.5f, 2.0f),
			Quat::sIdentity(),
			EMotionType::Dynamic,
			Layers::MOVING);
		settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		settings.mMassPropertiesOverride.mMass = 5.0f;  // Heavy
		mBodyInterface->CreateAndAddBody(settings, EActivation::Activate);
	}

	// Very heavy (stone - sinks fast)
	{
		BodyCreationSettings settings(
			new BoxShape(Vec3(0.15f, 0.15f, 0.15f)),
			RVec3(3.4f, 2.5f, 3.0f),
			Quat::sIdentity(),
			EMotionType::Dynamic,
			Layers::MOVING);
		settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		settings.mMassPropertiesOverride.mMass = 10.0f;  // Very heavy
		mBodyInterface->CreateAndAddBody(settings, EActivation::Activate);
	}
}

// =============================================================================
// Ragdoll Swimming Test Implementation
// =============================================================================

void WulfNetRagdollSwimTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 28;
	mFluidConfig.gridSizeY = 20;
	mFluidConfig.gridSizeZ = 28;
	mFluidConfig.cellSize = 0.22f;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize = mFluidConfig.cellSize;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Create swimming pool
	CreateWaterBox(1.0f, 0.2f, 1.0f, 4.5f, 2.5f, 4.5f);
}

void WulfNetRagdollSwimTest::SetupObjects()
{
	// Create simple ragdoll
	CreateRagdoll(2.75f, 3.0f, 2.75f);
}

void WulfNetRagdollSwimTest::CreateRagdoll(float x, float y, float z)
{
	// Simplified ragdoll using capsules
	// Torso
	BodyCreationSettings torso(
		new CapsuleShape(0.15f, 0.3f),
		RVec3(x, y, z),
		Quat::sRotation(Vec3::sAxisZ(), 0.0f),
		EMotionType::Dynamic,
		Layers::MOVING);
	torso.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
	torso.mMassPropertiesOverride.mMass = 3.0f;
	mBodyInterface->CreateAndAddBody(torso, EActivation::Activate);

	// Head
	BodyCreationSettings head(
		new SphereShape(0.1f),
		RVec3(x, y + 0.4f, z),
		Quat::sIdentity(),
		EMotionType::Dynamic,
		Layers::MOVING);
	head.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
	head.mMassPropertiesOverride.mMass = 0.5f;
	mBodyInterface->CreateAndAddBody(head, EActivation::Activate);

	// Arms (simplified as capsules)
	for (float side : {-0.25f, 0.25f})
	{
		BodyCreationSettings arm(
			new CapsuleShape(0.05f, 0.2f),
			RVec3(x + side, y + 0.1f, z),
			Quat::sRotation(Vec3::sAxisZ(), 1.57f),
			EMotionType::Dynamic,
			Layers::MOVING);
		arm.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		arm.mMassPropertiesOverride.mMass = 0.4f;
		mBodyInterface->CreateAndAddBody(arm, EActivation::Activate);
	}

	// Legs
	for (float side : {-0.08f, 0.08f})
	{
		BodyCreationSettings leg(
			new CapsuleShape(0.06f, 0.25f),
			RVec3(x + side, y - 0.4f, z),
			Quat::sIdentity(),
			EMotionType::Dynamic,
			Layers::MOVING);
		leg.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		leg.mMassPropertiesOverride.mMass = 0.8f;
		mBodyInterface->CreateAndAddBody(leg, EActivation::Activate);
	}
}

// =============================================================================
// Cloth Water Test Implementation
// =============================================================================

void WulfNetClothWaterTest::SetupFluid()
{
	mFluidConfig.gridSizeX = 28;
	mFluidConfig.gridSizeY = 20;
	mFluidConfig.gridSizeZ = 28;
	mFluidConfig.cellSize = 0.2f;

	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize = mFluidConfig.cellSize;

	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);

	// Create water pool
	CreateWaterBox(1.0f, 0.2f, 1.0f, 4.0f, 1.8f, 4.0f);
}

void WulfNetClothWaterTest::SetupObjects()
{
	// Create a grid of spheres to simulate cloth (placeholder)
	// TODO: Use proper soft body cloth when available
	const int gridSize = 6;
	const float spacing = 0.12f;
	const float startX = 2.0f;
	const float startZ = 2.0f;
	const float height = 2.5f;

	for (int i = 0; i < gridSize; ++i)
	{
		for (int j = 0; j < gridSize; ++j)
		{
			BodyCreationSettings clothNode(
				new SphereShape(0.03f),
				RVec3(startX + i * spacing, height, startZ + j * spacing),
				Quat::sIdentity(),
				EMotionType::Dynamic,
				Layers::MOVING);
			clothNode.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
			clothNode.mMassPropertiesOverride.mMass = 0.02f;  // Very light
			clothNode.mLinearDamping = 0.3f;  // Some drag
			mBodyInterface->CreateAndAddBody(clothNode, EActivation::Activate);
		}
	}
}
