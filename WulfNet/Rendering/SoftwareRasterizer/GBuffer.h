// =============================================================================
// WulfNet Engine - GBuffer (Framebuffer Management)
// =============================================================================
// Manages color, normal, and depth buffers for deferred rendering.
// SIMD-accelerated clear with sky gradient (AVX2 with SSE fallback).
// =============================================================================

#pragma once

#include "WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h"
#include <vector>
#include <cstdint>

namespace WulfNet {

class GBuffer {
public:
    GBuffer() = default;
    ~GBuffer() = default;

    /// Initialize with resolution
    bool Initialize(int width, int height);

    /// Clear all buffers (sky gradient on color, zero normals, max depth)
    void Clear(const SoftVec3& skyTop = {0.4f, 0.6f, 0.9f},
               const SoftVec3& skyBottom = {0.8f, 0.85f, 0.95f});

    /// Access individual pixels
    void SetColor(int x, int y, SoftColorRGBA8 color);
    void SetNormal(int x, int y, SoftColorRGBA8 packedNormal);
    void SetDepth(int x, int y, float depth);

    SoftColorRGBA8 GetColor(int x, int y) const;
    SoftColorRGBA8 GetNormal(int x, int y) const;
    float GetDepth(int x, int y) const;

    /// Test and set depth (returns true if pixel passes depth test)
    bool DepthTest(int x, int y, float depth);

    /// Buffer access
    uint32_t* GetColorBuffer() { return m_colorBuffer.data(); }
    const uint32_t* GetColorBuffer() const { return m_colorBuffer.data(); }
    uint32_t* GetNormalBuffer() { return m_normalBuffer.data(); }
    const uint32_t* GetNormalBuffer() const { return m_normalBuffer.data(); }
    float* GetDepthBuffer() { return m_depthBuffer.data(); }
    const float* GetDepthBuffer() const { return m_depthBuffer.data(); }

    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    int GetPixelCount() const { return m_width * m_height; }

private:
    int m_width = 0;
    int m_height = 0;

    std::vector<uint32_t> m_colorBuffer;   // RGBA8 packed as uint32
    std::vector<uint32_t> m_normalBuffer;  // Normal encoded as RGBA8
    std::vector<float> m_depthBuffer;      // Float depth values
};

} // namespace WulfNet
