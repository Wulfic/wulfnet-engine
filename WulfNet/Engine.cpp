// =============================================================================
// WulfNet Engine - Engine Implementation
// =============================================================================

#include "Engine.h"
#include "Core/Logging/Logger.h"
#include "Core/Profiling/Profiler.h"
#include "Core/System/SystemMonitor.h"
#include "Compute/Vulkan/VulkanContext.h"
#include "Physics/Integration/PhysicsWorld.h"
#include "Physics/Fluids/FluidSystem.h"
#include "Core/Memory/FrameAllocator.h"
#include <algorithm>

namespace WulfNet {

static constexpr const char* LOG_CAT = "Engine";

// =============================================================================
// Destructor — ensure clean shutdown even if user forgets
// =============================================================================

Engine::~Engine() {
    if (m_initialized) {
        Shutdown();
    }
}

// =============================================================================
// Lifecycle: Initialize
// =============================================================================

EngineInitResult Engine::Initialize(const EngineConfig& config) {
    if (m_initialized) {
        WULFNET_WARNING(LOG_CAT, "Initialize called on already-initialized engine. Ignoring.");
        return EngineInitResult::Success;
    }

    m_config = config;

    // Validate configuration before proceeding
    if (!m_config.Validate()) {
        WULFNET_ERROR(LOG_CAT, "Configuration validation failed.");
        return EngineInitResult::ConfigInvalid;
    }

    WULFNET_INFO(LOG_CAT, "=== WulfNet Engine v" WULFNET_VERSION_STRING " ===");
    WULFNET_INFO(LOG_CAT, "Application: " + m_config.appName);

    // --- Phase 1: Core systems (always initialized) ---

    if (!InitLogger()) {
        return EngineInitResult::LoggerFailed;
    }

    // SystemMonitor (best-effort, non-critical)
    auto& sysmon = SystemMonitor::Get();
    if (sysmon.Initialize()) {
        WULFNET_INFO(LOG_CAT, "SystemMonitor initialized");
    } else {
        WULFNET_WARNING(LOG_CAT, "SystemMonitor initialization failed (non-critical, continuing)");
    }

    // --- Phase 2: GPU Compute (before physics, since GPU fluids depend on it) ---

    if (m_config.enableCompute) {
        if (!InitCompute()) {
            WULFNET_WARNING(LOG_CAT, "Vulkan compute initialization failed (non-critical, GPU compute disabled)");
            m_computeInitialized = false;
            // This is non-critical — CPU fallbacks exist for all compute paths
        }
    }

    // --- Phase 3: Physics ---

    if (m_config.enablePhysics) {
        if (!InitPhysics()) {
            WULFNET_ERROR(LOG_CAT, "Physics initialization failed");
            Shutdown();
            return EngineInitResult::PhysicsFailed;
        }
    }

    // --- Phase 4: Rendering ---

    if (m_config.enableRendering) {
        if (!InitRendering()) {
            WULFNET_ERROR(LOG_CAT, "Rendering initialization failed");
            Shutdown();
            return EngineInitResult::RenderingFailed;
        }
    }

    // --- Phase 5: Audio ---

    if (m_config.enableAudio) {
        if (!InitAudio()) {
            WULFNET_WARNING(LOG_CAT, "Audio initialization failed (non-critical, audio disabled)");
            m_audio.reset();
            // Audio is non-critical — continue without it
        }
    }

    // --- Done ---

    m_initialized = true;
    m_running = true;
    m_lastFrameTime = Clock::now();
    m_frameNumber = 0;
    m_deltaTime = 0.0f;
    m_totalTime = 0.0f;
    m_physicsAccumulator = 0.0f;

    // Initialize per-frame linear allocator (10.6.2)
    FrameAllocator::Get().Initialize();

    WULFNET_INFO(LOG_CAT, "Engine initialized successfully"
        + std::string(m_config.enablePhysics   ? " [Physics]"   : "")
        + std::string(m_config.enableRendering ? " [Rendering]" : "")
        + std::string(m_config.enableAudio     ? " [Audio]"     : "")
        + std::string(m_computeInitialized     ? " [Compute]"   : ""));

    return EngineInitResult::Success;
}

// =============================================================================
// Lifecycle: Shutdown
// =============================================================================

void Engine::Shutdown() {
    if (!m_initialized) {
        return;
    }

    WULFNET_INFO(LOG_CAT, "Shutting down engine...");
    m_running = false;

    // Reverse order of initialization

    // Phase 5: Audio
    if (m_audio) {
        m_audio->Shutdown();
        m_audio.reset();
        WULFNET_DEBUG(LOG_CAT, "Audio shut down");
    }

    // Phase 4: Rendering
    if (m_renderer) {
        m_renderer->Shutdown();
        m_renderer.reset();
        WULFNET_DEBUG(LOG_CAT, "Rendering shut down");
    }

    // Phase 3: Physics
    if (m_physics) {
        m_physics->Shutdown();
        m_physics.reset();
        WULFNET_DEBUG(LOG_CAT, "Physics shut down");
    }

    // Phase 2: Compute
    if (m_computeInitialized) {
        ShutdownVulkanContext();
        m_computeInitialized = false;
        WULFNET_DEBUG(LOG_CAT, "Vulkan compute shut down");
    }

    // Phase 1: Core systems
    FrameAllocator::Get().Shutdown();
    SystemMonitor::Get().Shutdown();

    WULFNET_INFO(LOG_CAT, "Engine shut down successfully (frame " + std::to_string(m_frameNumber) + ")");

    // Logger stays alive (singleton) but we flush it
    Logger::Get().Flush();

    m_initialized = false;
}

// =============================================================================
// Frame Loop
// =============================================================================

void Engine::BeginFrame() {
    WULFNET_FRAME_MARK();
    WULFNET_ZONE_NAMED("Engine::BeginFrame");

    // Reset per-frame linear allocator (all transient allocations invalidated)
    FrameAllocator::Get().BeginFrame();

    // Calculate delta time
    auto now = Clock::now();
    std::chrono::duration<float> elapsed = now - m_lastFrameTime;
    m_deltaTime = elapsed.count();
    m_lastFrameTime = now;

    // Clamp delta time to prevent spiral of death
    // (e.g., after a breakpoint or long pause)
    m_deltaTime = std::min(m_deltaTime, 0.25f);

    m_totalTime += m_deltaTime;
    ++m_frameNumber;

    // --- GPU compute dispatch (10.3) ---
    // Kick off GPU work early so it runs concurrently with CPU physics.
    // The CPU will collect results in EndFrame() after physics completes.
#ifdef WULFNET_HAS_VULKAN
    if (m_computeInitialized && m_physics) {
        auto* fluid = m_physics->GetFluidSystem();
        if (fluid) {
            WULFNET_ZONE_NAMED("Engine::GPUDispatch");
            fluid->DispatchCompute(m_deltaTime);
            fluid->RequestAsyncReadback();
        }
    }
#endif
}

void Engine::EndFrame() {
    WULFNET_ZONE_NAMED("Engine::EndFrame");

    // --- Fixed timestep physics ---
    if (m_physics) {
        m_physicsAccumulator += m_deltaTime;
        int substeps = 0;
        while (m_physicsAccumulator >= m_config.physicsTimestep &&
               substeps < m_config.maxPhysicsSubsteps) {
            m_physics->Step(m_config.physicsTimestep);
            m_physicsAccumulator -= m_config.physicsTimestep;
            ++substeps;
        }

        // If accumulator is still very large, drain it to prevent runaway
        if (m_physicsAccumulator > m_config.physicsTimestep * 2.0f) {
            WULFNET_WARNING(LOG_CAT, "Physics can't keep up — dropping accumulated time ("
                + std::to_string(m_physicsAccumulator) + "s)");
            m_physicsAccumulator = 0.0f;
        }
    }

    // --- Periodic system monitor update (every ~60 frames) ---
    if ((m_frameNumber % 60) == 0) {
        SystemMonitor::Get().Update();
    }
}

// =============================================================================
// Subsystem Access
// =============================================================================

PhysicsWorld& Engine::GetPhysics() {
    return *m_physics;
}

RenderPipeline& Engine::GetRenderer() {
    return *m_renderer;
}

AudioMixer& Engine::GetAudio() {
    return *m_audio;
}

// =============================================================================
// Private: Subsystem init helpers
// =============================================================================

bool Engine::InitLogger() {
    auto& logger = Logger::Get();
    logger.SetMinLevel(m_config.logLevel);

    if (m_config.logToConsole) {
        logger.AddSink(std::make_shared<ConsoleLogSink>(true));
    }
    if (m_config.logToFile) {
        logger.AddSink(std::make_shared<FileLogSink>(m_config.logFilePath));
    }

    WULFNET_INFO(LOG_CAT, "Logger configured (level="
        + std::to_string(static_cast<int>(m_config.logLevel)) + ")");
    return true;
}

bool Engine::InitCompute() {
    WULFNET_INFO(LOG_CAT, "Initializing Vulkan compute...");
    m_config.compute.applicationName = m_config.appName;

    bool result = InitializeVulkanContext(m_config.compute);
    if (result) {
        m_computeInitialized = true;
        auto& ctx = GetVulkanContext();
        auto info = ctx.GetDeviceInfo();
        WULFNET_INFO(LOG_CAT, "Vulkan compute ready: " + info.name);
    }
    return result;
}

bool Engine::InitPhysics() {
    WULFNET_INFO(LOG_CAT, "Initializing physics...");
    m_physics = std::make_unique<PhysicsWorld>();

    if (!m_physics->Initialize(m_config.physics)) {
        m_physics.reset();
        return false;
    }
    WULFNET_INFO(LOG_CAT, "Physics initialized (maxBodies="
        + std::to_string(m_config.physics.maxBodies) + ")");
    return true;
}

bool Engine::InitRendering() {
    WULFNET_INFO(LOG_CAT, "Initializing rendering...");
    m_renderer = std::make_unique<RenderPipeline>();

    if (!m_renderer->Initialize(m_config.rendering)) {
        m_renderer.reset();
        return false;
    }
    WULFNET_INFO(LOG_CAT, "Rendering initialized ("
        + std::to_string(m_renderer->GetWidth()) + "x"
        + std::to_string(m_renderer->GetHeight()) + ")");
    return true;
}

bool Engine::InitAudio() {
    WULFNET_INFO(LOG_CAT, "Initializing audio...");
    m_audio = std::make_unique<AudioMixer>();

    if (!m_audio->Initialize(m_config.audio)) {
        m_audio.reset();
        return false;
    }
    WULFNET_INFO(LOG_CAT, "Audio initialized (sampleRate="
        + std::to_string(m_config.audio.sampleRate)
        + " bufferSize=" + std::to_string(m_config.audio.bufferSize) + ")");
    return true;
}

} // namespace WulfNet
