// =============================================================================
// WulfNet Engine - Engine Class
// =============================================================================
// The Engine class is the single entry point for the WulfNet runtime.
// It owns all subsystems, manages their lifecycle in dependency order,
// and provides BeginFrame()/EndFrame() for the game loop.
//
// Usage:
//   WulfNet::Engine engine;
//   engine.Initialize(config);
//   while (engine.IsRunning()) {
//       engine.BeginFrame();
//       // ... user logic ...
//       engine.EndFrame();
//   }
//   engine.Shutdown();
// =============================================================================

#pragma once

#include "API.h"
#include "EngineConfig.h"
#include "Core/Profiling/Profiler.h"
#include "Core/System/SystemMonitor.h"
#include <memory>
#include <chrono>

namespace WulfNet {

// =============================================================================
// Initialization Result
// =============================================================================

enum class EngineInitResult {
    Success,
    ConfigInvalid,
    LoggerFailed,
    ComputeFailed,
    PhysicsFailed,
    RenderingFailed,
    AudioFailed
};

// =============================================================================
// Engine Class
// =============================================================================

class WULFNET_API Engine {
public:
    Engine() = default;
    ~Engine();

    // Non-copyable, non-movable (owns singleton-like resources)
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// Initialize all enabled subsystems in dependency order.
    /// Returns Success on success; on failure, returns the failing module
    /// and all previously initialized modules are cleaned up.
    EngineInitResult Initialize(const EngineConfig& config = {});

    /// Shut down all subsystems in reverse dependency order.
    void Shutdown();

    // =========================================================================
    // Frame Loop
    // =========================================================================

    /// Call at the start of each frame.
    /// Updates delta time, frame counter, system monitoring.
    void BeginFrame();

    /// Call at the end of each frame.
    /// Steps physics (fixed timestep), profiles, marks frame for Tracy.
    void EndFrame();

    // =========================================================================
    // Subsystem Access
    // =========================================================================

    /// Returns the PhysicsWorld. Only valid if enablePhysics was true.
    PhysicsWorld&    GetPhysics();

    /// Returns the RenderPipeline. Only valid if enableRendering was true.
    RenderPipeline&  GetRenderer();

    /// Returns the AudioMixer. Only valid if enableAudio was true.
    AudioMixer&      GetAudio();

    // =========================================================================
    // State Queries
    // =========================================================================

    bool IsRunning() const { return m_running; }
    bool IsInitialized() const { return m_initialized; }

    /// Signal the engine to stop (IsRunning() will return false).
    void RequestShutdown() { m_running = false; }

    uint64_t GetFrameNumber() const { return m_frameNumber; }
    float    GetDeltaTime() const { return m_deltaTime; }
    float    GetTotalTime() const { return m_totalTime; }
    float    GetFPS() const { return m_deltaTime > 0.0f ? 1.0f / m_deltaTime : 0.0f; }

    const EngineConfig& GetConfig() const { return m_config; }

private:
    // Subsystem initialization helpers (return true on success)
    bool InitLogger();
    bool InitCompute();
    bool InitPhysics();
    bool InitRendering();
    bool InitAudio();

    // Configuration
    EngineConfig m_config;
    bool m_initialized = false;
    bool m_running     = false;

    // Owned subsystems (unique_ptr for optional ownership)
    std::unique_ptr<PhysicsWorld>   m_physics;
    std::unique_ptr<RenderPipeline> m_renderer;
    std::unique_ptr<AudioMixer>     m_audio;
    bool m_computeInitialized = false;

    // Frame timing
    uint64_t m_frameNumber = 0;
    float    m_deltaTime   = 0.0f;
    float    m_totalTime   = 0.0f;
    float    m_physicsAccumulator = 0.0f;
    int      m_gpuFrameIndex = 0;  // Double-buffered frame-in-flight index (10.3)

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point m_lastFrameTime;
};

} // namespace WulfNet
