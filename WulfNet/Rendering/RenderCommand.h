// =============================================================================
// WulfNet Engine - RenderCommand.h
// =============================================================================
// Decouples "what to render" from "how to render." Physics subsystems,
// procedural generators, and user code produce RenderCommands; the active
// IRenderer consumes them.
//
// RenderableList is a lightweight collection of render commands submitted
// each frame. The renderer interprets them according to its backend.
// =============================================================================

#pragma once

#include "WulfNet/API.h"
#include "WulfNet/Rendering/Types/RenderTypes.h"
#include <vector>
#include <cstdint>

namespace WulfNet {

// =============================================================================
// Render Command Types
// =============================================================================

/// Identifies the kind of render command.
enum class RenderCommandType : uint8_t {
    MeshInstance,       ///< Draw a mesh at a transform
    Volume,             ///< Volumetric region (gas, fluid surface, etc.)
    DebugLine,          ///< Debug visualization line
    DebugAABB,          ///< Debug axis-aligned bounding box
};

// =============================================================================
// Render Command
// =============================================================================

/// A single render command: a mesh instance, a volume, or a debug primitive.
struct RenderCommand {
    RenderCommandType type = RenderCommandType::MeshInstance;

    // --- Mesh Instance data (type == MeshInstance) ---
    int meshIndex    = -1;    ///< Index into the renderer's mesh list
    int textureIndex = -1;    ///< Index into the renderer's texture list
    SoftTransform transform;  ///< World transform for this instance

    // --- Volume data (type == Volume) ---
    // The VolumeSampler is referenced by pointer (owned externally).
    const VolumeSampler* volumeSampler = nullptr;

    // --- Debug data (type == DebugLine or DebugAABB) ---
    float debugP0[3] = {0, 0, 0};  ///< Start point / min corner
    float debugP1[3] = {0, 0, 0};  ///< End point / max corner
    uint32_t debugColor = 0xFFFFFFFF; ///< RGBA8 color

    // =========================================================================
    // Factory methods for convenience
    // =========================================================================

    /// Create a mesh instance command.
    static RenderCommand Mesh(int meshIdx, int texIdx, const SoftTransform& xform) {
        RenderCommand cmd;
        cmd.type = RenderCommandType::MeshInstance;
        cmd.meshIndex = meshIdx;
        cmd.textureIndex = texIdx;
        cmd.transform = xform;
        return cmd;
    }

    /// Create a volumetric render command.
    static RenderCommand VolumetricRegion(const VolumeSampler* sampler) {
        RenderCommand cmd;
        cmd.type = RenderCommandType::Volume;
        cmd.volumeSampler = sampler;
        return cmd;
    }
};

// =============================================================================
// Renderable List
// =============================================================================

/// Collection of render commands for a single frame.
/// Typically built fresh each frame, submitted to IRenderer::Submit().
class WULFNET_API RenderableList {
public:
    RenderableList() = default;

    /// Reserve space for expected number of commands.
    void Reserve(size_t count) { m_commands.reserve(count); }

    /// Add a render command.
    void Add(const RenderCommand& cmd) { m_commands.push_back(cmd); }
    void Add(RenderCommand&& cmd) { m_commands.push_back(std::move(cmd)); }

    /// Clear all commands (call at start of frame).
    void Clear() { m_commands.clear(); }

    /// Get the command list.
    const std::vector<RenderCommand>& GetCommands() const { return m_commands; }

    /// Get the number of commands.
    size_t GetCount() const { return m_commands.size(); }

    /// Check if the list is empty.
    bool IsEmpty() const { return m_commands.empty(); }

private:
    std::vector<RenderCommand> m_commands;
};

} // namespace WulfNet
