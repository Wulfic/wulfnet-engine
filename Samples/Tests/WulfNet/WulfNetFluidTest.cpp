// =============================================================================
// WulfNet Engine - CO-FLIP Fluid Test Implementation
// =============================================================================

#include <Samples.h>

#include <Tests/WulfNet/WulfNetFluidTest.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>
#include <sstream>
#include <iomanip>

// Register RTTI for factory
JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidTest, Test)
}

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
// WulfNetFluidTest Base Implementation
// =============================================================================

WulfNetFluidTest::~WulfNetFluidTest()
{
	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
}

void WulfNetFluidTest::Initialize()
{
	// Initialize system monitor
	WulfNet::SystemMonitor::Get().Initialize();
	mLastFPSTime = std::chrono::high_resolution_clock::now();

	// Create ground floor
	CreateFloor();

	// Default CO-FLIP configuration (low resolution works well with CO-FLIP!)
	mFluidConfig.gridSizeX = 48;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 48;
	mFluidConfig.cellSize = 0.15f;  // 15cm cells
	mFluidConfig.dt = 1.0f / 60.0f;
	mFluidConfig.flipRatio = 0.99f;
	mFluidConfig.pressureIterations = 30;
	mFluidConfig.particlesPerCell = 8;
	mFluidConfig.useGPU = true;  // GPU accelerated via Jolt compute system

	// Surface configuration (marching cubes)
	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;
	mSurfaceConfig.smoothingSigma = 1.2f;
	mSurfaceConfig.isoLevel = 0.4f;
	mSurfaceConfig.useGPU = false;

	// Initialize fluid system with Jolt's compute system for GPU acceleration
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		// Fallback to CPU if no compute system available
		mFluidConfig.useGPU = false;
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);;

	// Let derived class set up specific fluid scenario
	SetupFluid();
	SetupObjects();
}

void WulfNetFluidTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	// Update FPS tracking
	mFrameCount++;
	auto now = std::chrono::high_resolution_clock::now();
	float elapsed = std::chrono::duration<float>(now - mLastFPSTime).count();
	if (elapsed >= 1.0f) {
		mCurrentFPS = static_cast<float>(mFrameCount) / elapsed;
		mFrameTimeMs = (elapsed / mFrameCount) * 1000.0f;
		mFrameCount = 0;
		mLastFPSTime = now;
	}

	// Update system stats periodically
	mStatsUpdateTimer += inParams.mDeltaTime;
	if (mStatsUpdateTimer >= cStatsUpdateInterval) {
		WulfNet::SystemMonitor::Get().Update();
		mStatsUpdateTimer = 0.0f;
	}

	// Step fluid simulation
	mFluidSystem.Step(inParams.mDeltaTime);

	// Sync particles from GPU for rendering and surface generation
	// (no-op if running on CPU)
	mFluidSystem.SyncParticlesFromGPU();

	// Generate surface mesh from particles
	if (mDrawSurface)
	{
		mFluidSurface.GenerateSurface(mFluidSystem);
	}

	// Draw fluid
	if (mDrawParticles || mRenderMode == RenderMode::Particles || mRenderMode == RenderMode::Both)
	{
		DrawFluid();
	}

	if (mDrawSurface || mRenderMode == RenderMode::Surface || mRenderMode == RenderMode::Both)
	{
		DrawSurface();
	}

	if (mShowStats)
	{
		DrawStats();
	}
}

void WulfNetFluidTest::CreateWaterBox(float minX, float minY, float minZ,
                                       float maxX, float maxY, float maxZ)
{
	mFluidSystem.AddParticleBox(minX, minY, minZ, maxX, maxY, maxZ);
}

void WulfNetFluidTest::CreateWaterSphere(float cx, float cy, float cz, float radius)
{
	mFluidSystem.AddParticleSphere(cx, cy, cz, radius);
}

void WulfNetFluidTest::CreateEmitter(float x, float y, float z,
                                      float dirX, float dirY, float dirZ,
                                      float rate, float speed)
{
	mFluidSystem.AddEmitter(x, y, z, dirX, dirY, dirZ, rate, speed);
}

void WulfNetFluidTest::AddSolidBox(float minX, float minY, float minZ,
                                    float maxX, float maxY, float maxZ)
{
	mFluidSystem.AddSolidBox(minX, minY, minZ, maxX, maxY, maxZ);
}

void WulfNetFluidTest::AddSolidSphere(float cx, float cy, float cz, float radius)
{
	mFluidSystem.AddSolidSphere(cx, cy, cz, radius);
}

void WulfNetFluidTest::DrawFluid()
{
#ifdef JPH_DEBUG_RENDERER
	const auto& particles = mFluidSystem.GetParticles();
	uint32_t count = mFluidSystem.GetActiveParticleCount();

	for (uint32_t i = 0; i < count; ++i)
	{
		const WulfNet::COFLIPParticle& p = particles[i];
		if (!(p.flags & 1)) continue;  // Not active

		// Blue water color
		Color drawColor(64, 128, 255, 200);

		// Draw particle as small marker
		RVec3 pos(p.x, p.y, p.z);
		mDebugRenderer->DrawMarker(pos, drawColor, mParticleSize);

		// Optionally draw velocity
		if (mDrawVelocities && (p.vx*p.vx + p.vy*p.vy + p.vz*p.vz) > 0.01f)
		{
			Vec3 vel(p.vx, p.vy, p.vz);
			mDebugRenderer->DrawArrow(pos, pos + 0.1f * vel, Color::sYellow, 0.01f);
		}
	}
#endif
}

void WulfNetFluidTest::DrawSurface()
{
#ifdef JPH_DEBUG_RENDERER
	const auto& vertices = mFluidSurface.GetVertices();
	const auto& triangles = mFluidSurface.GetTriangles();

	if (triangles.empty()) return;

	// Water color with transparency
	Color waterColor(32, 100, 200, 180);

	// Draw each triangle
	for (const auto& tri : triangles)
	{
		const auto& v0 = vertices[tri.v0];
		const auto& v1 = vertices[tri.v1];
		const auto& v2 = vertices[tri.v2];

		RVec3 p0(v0.x, v0.y, v0.z);
		RVec3 p1(v1.x, v1.y, v1.z);
		RVec3 p2(v2.x, v2.y, v2.z);

		mDebugRenderer->DrawTriangle(p0, p1, p2, waterColor);
	}
#endif
}

void WulfNetFluidTest::DrawStats()
{
	// Stats are now displayed via GetStatusString() overlay
}

String WulfNetFluidTest::GetStatusString() const
{
	if (!mShowStats)
		return String();

	const WulfNet::COFLIPStats& stats = mFluidSystem.GetStats();
	const WulfNet::FluidSurfaceStats& surfStats = mFluidSurface.GetStats();
	const WulfNet::SystemStats& sysStats = WulfNet::SystemMonitor::Get().GetStats();

	std::ostringstream oss;
	oss << std::fixed;

	// Performance stats
	oss << "FPS: " << std::setprecision(1) << mCurrentFPS
	    << " (" << std::setprecision(2) << mFrameTimeMs << " ms)\n";

	oss << std::setprecision(1);
	oss << "CPU: " << sysStats.cpuUsagePercent << "%\n";

	oss << "RAM: " << WulfNet::FormatBytes(sysStats.processMemoryBytes)
	    << " / " << WulfNet::FormatBytes(sysStats.ramTotalBytes)
	    << " (" << sysStats.ramUsagePercent << "%)\n";

	if (sysStats.gpuUsageAvailable) {
		oss << "GPU: " << sysStats.gpuUsagePercent << "%";
		if (!sysStats.gpuName.empty()) {
			oss << " (" << sysStats.gpuName << ")";
		}
		oss << "\n";
	} else {
		oss << "GPU: N/A\n";
	}

	if (sysStats.vramUsageAvailable) {
		oss << "VRAM: " << WulfNet::FormatBytes(sysStats.vramUsedBytes)
		    << " / " << WulfNet::FormatBytes(sysStats.vramTotalBytes)
		    << " (" << sysStats.vramUsagePercent << "%)\n";
	} else {
		oss << "VRAM: N/A\n";
	}

	oss << "\n";  // Separator

	// Fluid simulation stats
	oss << "Particles: " << stats.activeParticles << "\n";
	oss << "Triangles: " << surfStats.triangleCount << "\n";

	oss << std::setprecision(2);
	oss << "Sim: " << stats.totalTimeMs << " ms (P2G: " << stats.p2gTimeMs
	    << ", Pressure: " << stats.pressureTimeMs << ", G2P: " << stats.g2pTimeMs << ")";

	return String(oss.str());
}

// =============================================================================
// River Test Implementation
// =============================================================================

void WulfNetRiverTest::SetupFluid()
{
	// Configure for river simulation (CO-FLIP works great at lower resolutions)
	mFluidConfig.gridSizeX = 64;
	mFluidConfig.gridSizeY = 24;
	mFluidConfig.gridSizeZ = 32;
	mFluidConfig.cellSize = 0.12f;

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

	// Create emitter at one end
	CreateEmitter(0.3f, 0.5f, 1.5f, 1.0f, 0.0f, 0.0f, 200.0f, 1.5f);

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
	// Configure for waterfall (needs vertical space)
	mFluidConfig.gridSizeX = 48;
	mFluidConfig.gridSizeY = 48;
	mFluidConfig.gridSizeZ = 48;
	mFluidConfig.cellSize = 0.1f;

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

	// Create falling water emitter at top
	CreateEmitter(2.3f, 3.5f, 2.3f, 0.0f, -1.0f, 0.0f, 300.0f, 0.5f);
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
	// Small puddle configuration
	mFluidConfig.gridSizeX = 32;
	mFluidConfig.gridSizeY = 16;
	mFluidConfig.gridSizeZ = 32;
	mFluidConfig.cellSize = 0.08f;

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

	// Rain drops falling into puddle
	CreateEmitter(1.3f, 1.2f, 1.3f, 0.0f, -1.0f, 0.0f, 50.0f, 0.2f);
}

// =============================================================================
// Lake Test Implementation
// =============================================================================

void WulfNetLakeTest::SetupFluid()
{
	// Large lake configuration
	mFluidConfig.gridSizeX = 64;
	mFluidConfig.gridSizeY = 24;
	mFluidConfig.gridSizeZ = 64;
	mFluidConfig.cellSize = 0.15f;

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

	// Create large body of water
	CreateWaterBox(1.0f, 0.2f, 1.0f, 8.5f, 1.5f, 8.5f);
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
	// Standard grid for comparison
	mFluidConfig.gridSizeX = 64;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 32;
	mFluidConfig.cellSize = 0.1f;

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
	mFluidConfig.gridSizeX = 48;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 48;
	mFluidConfig.cellSize = 0.12f;

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
	mFluidConfig.gridSizeX = 48;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 48;
	mFluidConfig.cellSize = 0.12f;

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
	mFluidConfig.gridSizeX = 48;
	mFluidConfig.gridSizeY = 32;
	mFluidConfig.gridSizeZ = 48;
	mFluidConfig.cellSize = 0.1f;

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
