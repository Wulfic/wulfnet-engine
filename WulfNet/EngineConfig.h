// =============================================================================
// WulfNet Engine - Centralized Engine Configuration
// =============================================================================
// Single aggregate struct that collects all module configs with sensible
// defaults. Zero-config (default constructor) produces a valid configuration.
//
// Preset factory methods provide common setups:
//   EngineConfig::Full()            — everything enabled
//   EngineConfig::Minimal()         — core + logging only
//   EngineConfig::HeadlessPhysics() — physics + compute, no rendering/audio
// =============================================================================

#pragma once

#include "Version.h"
#include "Physics/Integration/PhysicsWorld.h"
#include "Rendering/RenderPipeline.h"
#include "Audio/Core/AudioMixer.h"
#include "Compute/Vulkan/VulkanContext.h"
#include "Core/Logging/Logger.h"
#include <string>

namespace WulfNet {

// =============================================================================
// Engine Configuration
// =============================================================================

struct EngineConfig {
    // --- Application identity ---
    std::string appName = "WulfNet Application";

    // --- Feature flags ---
    bool enablePhysics   = true;
    bool enableRendering = true;
    bool enableAudio     = true;
    bool enableCompute   = true;    // Vulkan GPU compute

    // --- Logging ---
    LogLevel logLevel       = LogLevel::Info;
    bool logToConsole       = true;
    bool logToFile          = false;
    std::string logFilePath = "wulfnet.log";

    // --- Module configs (only used if feature is enabled) ---
    PhysicsWorldSettings  physics;
    RenderPipelineConfig  rendering;
    AudioMixerConfig      audio;
    VulkanContextSettings compute;

    // --- Fixed timestep settings ---
    float physicsTimestep   = 1.0f / 60.0f;  // 60 Hz physics
    int   maxPhysicsSubsteps = 4;             // Max catch-up steps per frame

    // =========================================================================
    // Preset Factories
    // =========================================================================

    /// Full engine — all systems enabled with defaults
    static EngineConfig Full() {
        EngineConfig cfg;
        cfg.enablePhysics   = true;
        cfg.enableRendering = true;
        cfg.enableAudio     = true;
        cfg.enableCompute   = true;
        return cfg;
    }

    /// Minimal — core systems only (logging, profiling)
    static EngineConfig Minimal() {
        EngineConfig cfg;
        cfg.enablePhysics   = false;
        cfg.enableRendering = false;
        cfg.enableAudio     = false;
        cfg.enableCompute   = false;
        return cfg;
    }

    /// Headless physics — physics + compute, no rendering or audio
    /// Useful for dedicated servers, simulation tools, etc.
    static EngineConfig HeadlessPhysics() {
        EngineConfig cfg;
        cfg.enablePhysics   = true;
        cfg.enableRendering = false;
        cfg.enableAudio     = false;
        cfg.enableCompute   = true;
        return cfg;
    }

    /// Validate configuration and return true if valid.
    /// Logs warnings for questionable values, errors for invalid ones.
    bool Validate() const {
        bool valid = true;
        if (physicsTimestep <= 0.0f) {
            valid = false;
        }
        if (maxPhysicsSubsteps <= 0) {
            valid = false;
        }
        if (enableRendering && rendering.rasterizer.width <= 0) {
            valid = false;
        }
        if (enableRendering && rendering.rasterizer.height <= 0) {
            valid = false;
        }
        return valid;
    }
};

} // namespace WulfNet
