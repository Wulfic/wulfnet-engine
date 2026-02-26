// =============================================================================
// WulfNet Engine - CO-FLIP Fluid Test Implementation
// =============================================================================

#include <Samples.h>

#include <Tests/WulfNet/WulfNetFluidTest.h>
#include "WaterDiagnostics.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cfloat>

// Register RTTI for factory
JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetFluidTest)
{
	JPH_ADD_BASE_CLASS(WulfNetFluidTest, Test)
}


// Subclass RTTI + scenario implementations --> WulfNetFluidTestScenarios.cpp


// =============================================================================
// WulfNetFluidTest Base Implementation
// =============================================================================

WulfNetFluidTest::~WulfNetFluidTest()
{
	// Wait for any in-flight async surface generation before teardown
	if (mSurfaceFuture.valid()) mSurfaceFuture.get();
	mFluidSystem.Shutdown();
	mFluidSurface.Shutdown();
	WaterDiagnostics::Shutdown();
}

void WulfNetFluidTest::Initialize()
{
	// Initialize system monitor
	WulfNet::SystemMonitor::Get().Initialize();
	mLastFPSTime = std::chrono::high_resolution_clock::now();

	// Create ground floor
	CreateFloor();

	// Default CO-FLIP configuration — optimized for larger particles / fewer total
	mFluidConfig.gridSizeX = 32;
	mFluidConfig.gridSizeY = 24;
	mFluidConfig.gridSizeZ = 32;
	mFluidConfig.cellSize = 0.25f;  // 25cm cells — 4× fewer particles vs 0.15
	mFluidConfig.dt = 1.0f / 60.0f;
	mFluidConfig.flipRatio = 0.99f;
	mFluidConfig.pressureIterations = 20;  // SOR converges fast
	mFluidConfig.particlesPerCell = 4;
	mFluidConfig.useGPU = true;  // GPU accelerated via Jolt compute system

	// Surface configuration (marching cubes)
	mSurfaceConfig.gridSizeX = mFluidConfig.gridSizeX;
	mSurfaceConfig.gridSizeY = mFluidConfig.gridSizeY;
	mSurfaceConfig.gridSizeZ = mFluidConfig.gridSizeZ;
	mSurfaceConfig.cellSize = mFluidConfig.cellSize;
	mSurfaceConfig.splatRadius = 2.5f;  // Reduced from 3.5 for performance
	mSurfaceConfig.smoothingSigma = 1.4f;
	mSurfaceConfig.isoLevel = 0.3f;
	mSurfaceConfig.useGPU = true;

	// Initialize fluid system with Jolt's compute system for GPU acceleration
	if (mComputeSystem) {
		mFluidSystem.InitializeFromJolt(mFluidConfig, mComputeSystem);
	} else {
		// Fallback to CPU if no compute system available
		mFluidConfig.useGPU = false;
		mFluidSystem.Initialize(mFluidConfig);
	}
	mFluidSurface.Initialize(mSurfaceConfig);;

	// Initialize water diagnostics logging
	WaterDiagnostics::Initialize(GetRTTI()->GetName());
	WaterDiagnostics::LogCOFLIPConfig(mFluidConfig);

	// Let derived class set up specific fluid scenario
	SetupFluid();
	SetupObjects();

	COFLIP_LOG_INFO("[INIT] Fluid setup complete — Particles: " +
	                std::to_string(mFluidSystem.GetActiveParticleCount()));
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

	// Log per-frame water physics diagnostics
	WaterDiagnostics::LogCOFLIPFrame(mFluidSystem, mCurrentFPS);

	// Generate surface mesh from particles (throttled for performance).
	// When async is enabled, launch on a background thread so MC extraction
	// overlaps with particle drawing and the next physics step dispatch.
	if (mDrawSurface)
	{
		mSurfaceFrameCounter++;
		if (mSurfaceFrameCounter >= mSurfaceUpdateInterval) {
			mSurfaceFrameCounter = 0;
			if (mAsyncSurface) {
				// Wait for any prior async surface gen before starting a new one
				if (mSurfaceFuture.valid()) mSurfaceFuture.get();
				mSurfaceFuture = std::async(std::launch::async, [this]() {
					mFluidSurface.GenerateSurface(mFluidSystem);
				});
			} else {
				mFluidSurface.GenerateSurface(mFluidSystem);
			}
		}
	}

	// Draw fluid particles (can proceed while async surface gen runs)
	if (mDrawParticles || mRenderMode == RenderMode::Particles || mRenderMode == RenderMode::Both)
	{
		DrawFluid();
	}

	// Wait for async surface generation to complete before drawing the mesh
	if (mSurfaceFuture.valid()) mSurfaceFuture.get();

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

	// Use cached vertical extents from UpdateStats() (already computed with OMP)
	// instead of a redundant single-threaded scan every frame.
	const WulfNet::COFLIPStats& stats = mFluidSystem.GetStats();
	float minY = stats.minParticleY;
	float maxY = stats.maxParticleY;
	float yRange = std::max(0.01f, maxY - minY);

	for (uint32_t i = 0; i < count; ++i)
	{
		const WulfNet::COFLIPParticle& p = particles[i];
		if (!(p.flags & 1)) continue;

		// Depth-dependent color: shallow (near surface) = light cyan, deep = dark blue
		float depthT = 1.0f - std::min(1.0f, (p.y - minY) / yRange);
		depthT = depthT * depthT; // Non-linear for better depth perception
		uint8_t r = static_cast<uint8_t>(100 - depthT * 60);
		uint8_t g = static_cast<uint8_t>(180 - depthT * 80);
		uint8_t b = static_cast<uint8_t>(255 - depthT * 40);
		Color drawColor(r, g, b, 200);

		RVec3 pos(p.x, p.y, p.z);
		mDebugRenderer->DrawMarker(pos, drawColor, mParticleSize);

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

	// Directional light for shading (from upper-right, matches V5)
	Vec3 lightDir = Vec3(0.4f, 0.8f, 0.3f).Normalized();
	const float ambient = 0.30f;

	// Use cached vertical extents from GenerateSurface() (already computed with OMP)
	// instead of a redundant single-threaded scan every frame.
	const WulfNet::FluidSurfaceStats& surfStats = mFluidSurface.GetStats();
	float minY = surfStats.minVertexY;
	float maxY = surfStats.maxVertexY;
	float yRange = std::max(0.01f, maxY - minY);

	for (const auto& tri : triangles)
	{
		const auto& v0 = vertices[tri.v0];
		const auto& v1 = vertices[tri.v1];
		const auto& v2 = vertices[tri.v2];

		RVec3 p0(v0.x, v0.y, v0.z);
		RVec3 p1(v1.x, v1.y, v1.z);
		RVec3 p2(v2.x, v2.y, v2.z);

		// Average normal from vertices (marching cubes generates smooth normals)
		Vec3 normal(v0.nx + v1.nx + v2.nx,
		            v0.ny + v1.ny + v2.ny,
		            v0.nz + v1.nz + v2.nz);
		if (normal.LengthSq() > 1e-12f)
			normal = normal.Normalized();

		// Directional lighting
		float nDotL = std::max(0.0f, normal.Dot(lightDir));
		float shade = ambient + (1.0f - ambient) * nDotL;

		// Specular highlight (cheap Blinn-Phong approximation)
		// Assume view roughly from above-forward
		Vec3 halfVec = (lightDir + Vec3(0.0f, 1.0f, 0.0f)).Normalized();
		float spec = std::pow(std::max(0.0f, normal.Dot(halfVec)), 16.0f);

		// Depth-based base color: top of fluid = light translucent blue,
		// bottom = deep opaque blue
		float avgY = (v0.y + v1.y + v2.y) / 3.0f;
		float depthT = 1.0f - std::min(1.0f, (avgY - minY) / yRange);
		depthT = depthT * depthT;

		// Shallow: (100, 180, 240, 160), Deep: (20, 60, 180, 220)
		uint8_t baseR = static_cast<uint8_t>(100 - depthT * 80);
		uint8_t baseG = static_cast<uint8_t>(180 - depthT * 120);
		uint8_t baseB = static_cast<uint8_t>(240 - depthT * 60);
		uint8_t baseA = static_cast<uint8_t>(160 + depthT * 60);

		uint8_t r = static_cast<uint8_t>(std::min(255.0f, baseR * shade + spec * 50.0f));
		uint8_t g = static_cast<uint8_t>(std::min(255.0f, baseG * shade + spec * 50.0f));
		uint8_t b = static_cast<uint8_t>(std::min(255.0f, baseB * shade + spec * 50.0f));

		Color waterColor(r, g, b, baseA);
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

