// =============================================================================
// WulfNet Engine - Transform Set Blender
// =============================================================================
// Interpolates between two sets of transform instructions for smooth
// morphing between fractal presets.
// Ported from SetBlender.cs (Unity IFS reference).
// =============================================================================

#pragma once

#include "WulfNet/Procedural/IFS/AffineTransform.h"
#include <vector>

namespace WulfNet {

class TransformBlender {
public:
    TransformBlender() = default;

    /// Set the two instruction sets to blend between
    void SetSets(const std::vector<TransformInstructions>& set1,
                 const std::vector<TransformInstructions>& set2);

    /// Update the blender (call each frame)
    /// Uses exponential decay smoothing toward the target set
    void Update(float dt, float speed);

    /// Get the current blended instruction set
    const std::vector<TransformInstructions>& GetBlendedSet() const { return m_blendedSet; }

    /// Get GPU-ready matrices from the current blended set
    std::vector<Mat4> GetBlendedMatrices() const;

    /// Directly set blend factor (0 = set1, 1 = set2)
    void SetBlendFactor(float t) { m_t = t; }
    float GetBlendFactor() const { return m_t; }

    /// Switch target to set2 (triggers new blend)
    void SwitchTarget(const std::vector<TransformInstructions>& newTarget);

private:
    static TransformInstructions LerpInstructions(const TransformInstructions& a,
                                                   const TransformInstructions& b,
                                                   float t);

    std::vector<TransformInstructions> m_set1;
    std::vector<TransformInstructions> m_set2;
    std::vector<TransformInstructions> m_currentState;  // exponential decay state
    std::vector<TransformInstructions> m_blendedSet;

    float m_t = 0.0f;
    bool m_useExpDecay = true;
};

} // namespace WulfNet
