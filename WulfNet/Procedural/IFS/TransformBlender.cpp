// =============================================================================
// WulfNet Engine - Transform Set Blender Implementation
// =============================================================================

#include "WulfNet/Procedural/IFS/TransformBlender.h"
#include <algorithm>
#include <cmath>

namespace WulfNet {

void TransformBlender::SetSets(const std::vector<TransformInstructions>& set1,
                                const std::vector<TransformInstructions>& set2) {
    m_set1 = set1;
    m_set2 = set2;

    // Pad the smaller set with identity transforms to match sizes
    size_t maxSize = std::max(m_set1.size(), m_set2.size());
    while (m_set1.size() < maxSize) m_set1.push_back(TransformInstructions::Identity());
    while (m_set2.size() < maxSize) m_set2.push_back(TransformInstructions::Identity());

    // Initialize current state to set1
    m_currentState = m_set1;
    m_blendedSet = m_set1;
    m_t = 0.0f;
}

void TransformBlender::Update(float dt, float speed) {
    if (m_currentState.empty()) return;

    float decay = std::min(speed * dt, 1.0f);

    m_blendedSet.resize(m_currentState.size());
    for (size_t i = 0; i < m_currentState.size(); ++i) {
        m_currentState[i] = LerpInstructions(m_currentState[i], m_set2[i], decay);
        m_blendedSet[i] = m_currentState[i];
    }

    m_t += dt * speed;
}

std::vector<Mat4> TransformBlender::GetBlendedMatrices() const {
    std::vector<Mat4> matrices;
    matrices.reserve(m_blendedSet.size());
    for (const auto& inst : m_blendedSet) {
        matrices.push_back(AffineTransform::FromInstructions(inst));
    }
    return matrices;
}

void TransformBlender::SwitchTarget(const std::vector<TransformInstructions>& newTarget) {
    m_set2 = newTarget;

    // Pad to match current state size
    size_t maxSize = std::max(m_currentState.size(), m_set2.size());
    while (m_currentState.size() < maxSize) m_currentState.push_back(TransformInstructions::Identity());
    while (m_set2.size() < maxSize) m_set2.push_back(TransformInstructions::Identity());

    m_t = 0.0f;
}

TransformInstructions TransformBlender::LerpInstructions(const TransformInstructions& a,
                                                          const TransformInstructions& b,
                                                          float t) {
    TransformInstructions result;
    result.scale = Vec3::Lerp(a.scale, b.scale, t);
    result.shearX = Vec3::Lerp(a.shearX, b.shearX, t);
    result.shearY = Vec3::Lerp(a.shearY, b.shearY, t);
    result.shearZ = Vec3::Lerp(a.shearZ, b.shearZ, t);
    result.translate = Vec3::Lerp(a.translate, b.translate, t);

    // For rotation, use simple lerp on euler angles (slerp would be better but this matches reference)
    result.rotate = Vec3::Lerp(a.rotate, b.rotate, t);

    return result;
}

} // namespace WulfNet
