// =============================================================================
// WulfNet Engine - IRenderer Interface
// =============================================================================
// Abstract rendering backend interface. The engine interacts with rendering
// only through this interface, allowing different backends (software rasterizer,
// Vulkan, etc.) to be swapped without changing consumer code.
//
// The current default implementation is RenderPipeline (software rasterizer).
// =============================================================================

#pragma once

#include "WulfNet/API.h"
#include <cstdint>

namespace WulfNet {

// Forward declarations
struct RenderPipelineConfig;
struct RenderStats;
struct SoftCamera;
struct RenderCommand;
class  RenderableList;

// =============================================================================
// Renderer Interface
// =============================================================================

class WULFNET_API IRenderer {
public:
    virtual ~IRenderer() = default;

    /// Initialize the renderer with configuration.
    virtual bool Initialize(const RenderPipelineConfig& config) = 0;

    /// Shut down and release all resources.
    virtual void Shutdown() = 0;

    /// Prepare for a new frame (clear buffers, update state).
    virtual void BeginFrame() = 0;

    /// Submit a batch of render commands for the current frame.
    virtual void Submit(const RenderableList& renderables, const SoftCamera& camera) = 0;

    /// Finalize the frame (present, post-process).
    virtual void EndFrame() = 0;

    /// Get the final color buffer for display (RGBA8, row-major).
    /// Returns nullptr if not available (e.g., GPU-only backend).
    virtual const uint32_t* GetColorBuffer() const = 0;

    /// Get frame statistics from the last completed frame.
    virtual const RenderStats& GetStats() const = 0;

    /// Get the render target dimensions.
    virtual int GetWidth() const = 0;
    virtual int GetHeight() const = 0;

    /// Check if the renderer is initialized and ready.
    virtual bool IsInitialized() const = 0;
};

} // namespace WulfNet
