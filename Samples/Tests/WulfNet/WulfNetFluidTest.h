// =============================================================================
// WulfNet Engine - CO-FLIP Fluid Test Base Class
// =============================================================================
// Base test class for WulfNet fluid physics tests using CO-FLIP algorithm.
// Features marching cubes surface rendering for smooth water visualization.
// =============================================================================

#pragma once

#include <Tests/Test.h>
#include <WulfNet/Physics/Fluids/COFLIPSystem.h>
#include <WulfNet/Physics/Fluids/FluidSurface.h>
#include <WulfNet/Core/System/SystemMonitor.h>
#include <chrono>

class WulfNetFluidTest : public Test
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetFluidTest)

	// Destructor
	virtual ~WulfNetFluidTest() override;

	// Initialize the test
	virtual void Initialize() override;

	// Update the test
	virtual void PrePhysicsUpdate(const PreUpdateParams &inParams) override;

	// Return status string for overlay display
	virtual String GetStatusString() const override;

	// Optional: custom fluid setup (override in derived classes)
	virtual void SetupFluid() {}

	// Optional: add custom objects (boats, balls, etc.)
	virtual void SetupObjects() {}

protected:
	// CO-FLIP fluid system
	WulfNet::COFLIPSystem mFluidSystem;

	// Surface mesh generator (marching cubes)
	WulfNet::FluidSurface mFluidSurface;

	// Configuration - subclasses can override
	WulfNet::COFLIPConfig mFluidConfig;
	WulfNet::FluidSurfaceConfig mSurfaceConfig;

	// Rendering mode
	enum class RenderMode {
		Particles,      // Debug: draw individual particles
		Surface,        // Smooth marching cubes surface
		Both            // Both for debugging
	};
	RenderMode mRenderMode = RenderMode::Surface;

	// Debug rendering
	bool mDrawParticles = false;
	bool mDrawSurface = true;
	bool mDrawGrid = false;
	bool mDrawVelocities = false;
	float mParticleSize = 0.02f;

	// Stats display
	bool mShowStats = true;

	// System monitoring
	float mStatsUpdateTimer = 0.0f;
	static constexpr float cStatsUpdateInterval = 0.5f;  // Update every 0.5 seconds
	float mCurrentFPS = 0.0f;
	float mFrameTimeMs = 0.0f;
	int mFrameCount = 0;
	std::chrono::high_resolution_clock::time_point mLastFPSTime;

	// Helper methods
	void CreateWaterBox(float minX, float minY, float minZ,
	                    float maxX, float maxY, float maxZ);

	void CreateWaterSphere(float cx, float cy, float cz, float radius);

	void CreateEmitter(float x, float y, float z,
	                   float dirX, float dirY, float dirZ,
	                   float rate, float speed);

	void AddSolidBox(float minX, float minY, float minZ,
	                 float maxX, float maxY, float maxZ);

	void AddSolidSphere(float cx, float cy, float cz, float radius);

	void DrawFluid();
	void DrawSurface();
	void DrawStats();
};

// =============================================================================
// River Test - Flowing water in a channel
// =============================================================================

class WulfNetRiverTest : public WulfNetFluidTest
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetRiverTest)

	virtual const char* GetDescription() const override {
		return "MPM fluid simulation of flowing water in a river channel.";
	}

	virtual void SetupFluid() override;
	virtual void SetupObjects() override;

private:
	void CreateRiverChannel();
};

// =============================================================================
// Waterfall Test - Falling water with splashing
// =============================================================================

class WulfNetWaterfallTest : public WulfNetFluidTest
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterfallTest)

	virtual const char* GetDescription() const override {
		return "MPM fluid simulation of a waterfall with splashing effects.";
	}

	virtual void SetupFluid() override;
	virtual void SetupObjects() override;
};

// =============================================================================
// Puddle Test - Small contained water
// =============================================================================

class WulfNetPuddleTest : public WulfNetFluidTest
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetPuddleTest)

	virtual const char* GetDescription() const override {
		return "MPM fluid simulation of a small puddle.";
	}

	virtual void SetupFluid() override;
};

// =============================================================================
// Lake Test - Large body of water
// =============================================================================

class WulfNetLakeTest : public WulfNetFluidTest
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetLakeTest)

	virtual const char* GetDescription() const override {
		return "MPM fluid simulation of a large lake with waves.";
	}

	virtual void SetupFluid() override;
	virtual void SetupObjects() override;
};

// =============================================================================
// Viscosity Test - Different fluid viscosities
// =============================================================================

class WulfNetViscosityTest : public WulfNetFluidTest
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetViscosityTest)

	virtual const char* GetDescription() const override {
		return "Comparison of water, oil, and honey viscosities.";
	}

	virtual void SetupFluid() override;
};

// =============================================================================
// Buoyancy Test - Floating and sinking objects
// =============================================================================

class WulfNetBuoyancyTest : public WulfNetFluidTest
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetBuoyancyTest)

	virtual const char* GetDescription() const override {
		return "Objects floating and sinking in water based on density.";
	}

	virtual void SetupFluid() override;
	virtual void SetupObjects() override;
};

// =============================================================================
// Ragdoll Swimming Test - Ragdoll in water
// =============================================================================

class WulfNetRagdollSwimTest : public WulfNetFluidTest
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetRagdollSwimTest)

	virtual const char* GetDescription() const override {
		return "Ragdoll physics interacting with fluid (swimming/drowning).";
	}

	virtual void SetupFluid() override;
	virtual void SetupObjects() override;

private:
	void CreateRagdoll(float x, float y, float z);
};

// =============================================================================
// Cloth in Water Test - Soft body cloth in fluid
// =============================================================================

class WulfNetClothWaterTest : public WulfNetFluidTest
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetClothWaterTest)

	virtual const char* GetDescription() const override {
		return "Soft body cloth interacting with water.";
	}

	virtual void SetupFluid() override;
	virtual void SetupObjects() override;
};
