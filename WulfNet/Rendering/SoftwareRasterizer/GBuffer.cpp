// =============================================================================
// WulfNet Engine - GBuffer Implementation
// =============================================================================

#include "WulfNet/Rendering/SoftwareRasterizer/GBuffer.h"
#include <cstring>
#include <algorithm>
#include <limits>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef __SSE2__
#include <immintrin.h>
#define HAS_SSE2 1
#else
#ifdef _MSC_VER
#include <immintrin.h>
#define HAS_SSE2 1
#endif
#endif

namespace WulfNet {

// Runtime AVX2 detection
static bool HasAVX2() {
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuidex(cpuInfo, 7, 0);
    return (cpuInfo[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

bool GBuffer::Initialize(int width, int height) {
    m_width = width;
    m_height = height;

    int pixelCount = width * height;
    m_colorBuffer.resize(pixelCount);
    m_normalBuffer.resize(pixelCount);
    m_depthBuffer.resize(pixelCount);

    Clear();
    return true;
}

void GBuffer::Clear(const SoftVec3& skyTop, const SoftVec3& skyBottom) {
    int pixelCount = m_width * m_height;

    // Clear depth to max
    std::fill(m_depthBuffer.begin(), m_depthBuffer.end(), std::numeric_limits<float>::max());

    // Clear normals to zero
    std::memset(m_normalBuffer.data(), 0, pixelCount * sizeof(uint32_t));

    // Clear color with sky gradient
#if HAS_SSE2
    if (HasAVX2()) {
        // AVX2 path: clear 8 pixels at a time
        for (int y = 0; y < m_height; ++y) {
            float t = static_cast<float>(y) / static_cast<float>(m_height);
            SoftVec3 skyColor = SoftVec3::Lerp(skyTop, skyBottom, t);
            uint32_t packed = SoftColorRGBA8::FromFloat(skyColor.x, skyColor.y, skyColor.z).ToUint32();

            int rowStart = y * m_width;
            int x = 0;

#ifdef __AVX2__
            __m256i val = _mm256_set1_epi32(static_cast<int>(packed));
            for (; x + 8 <= m_width; x += 8) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&m_colorBuffer[rowStart + x]), val);
            }
#endif
            // Scalar remainder
            for (; x < m_width; ++x) {
                m_colorBuffer[rowStart + x] = packed;
            }
        }
    } else {
        // SSE2 path: clear 4 pixels at a time
        for (int y = 0; y < m_height; ++y) {
            float t = static_cast<float>(y) / static_cast<float>(m_height);
            SoftVec3 skyColor = SoftVec3::Lerp(skyTop, skyBottom, t);
            uint32_t packed = SoftColorRGBA8::FromFloat(skyColor.x, skyColor.y, skyColor.z).ToUint32();

            int rowStart = y * m_width;
            int x = 0;

            __m128i val = _mm_set1_epi32(static_cast<int>(packed));
            for (; x + 4 <= m_width; x += 4) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&m_colorBuffer[rowStart + x]), val);
            }

            for (; x < m_width; ++x) {
                m_colorBuffer[rowStart + x] = packed;
            }
        }
    }
#else
    // Scalar fallback
    for (int y = 0; y < m_height; ++y) {
        float t = static_cast<float>(y) / static_cast<float>(m_height);
        SoftVec3 skyColor = SoftVec3::Lerp(skyTop, skyBottom, t);
        uint32_t packed = SoftColorRGBA8::FromFloat(skyColor.x, skyColor.y, skyColor.z).ToUint32();

        int rowStart = y * m_width;
        for (int x = 0; x < m_width; ++x) {
            m_colorBuffer[rowStart + x] = packed;
        }
    }
#endif
}

void GBuffer::SetColor(int x, int y, SoftColorRGBA8 color) {
    m_colorBuffer[y * m_width + x] = color.ToUint32();
}

void GBuffer::SetNormal(int x, int y, SoftColorRGBA8 packedNormal) {
    m_normalBuffer[y * m_width + x] = packedNormal.ToUint32();
}

void GBuffer::SetDepth(int x, int y, float depth) {
    m_depthBuffer[y * m_width + x] = depth;
}

SoftColorRGBA8 GBuffer::GetColor(int x, int y) const {
    uint32_t val = m_colorBuffer[y * m_width + x];
    return {
        static_cast<uint8_t>(val & 0xFF),
        static_cast<uint8_t>((val >> 8) & 0xFF),
        static_cast<uint8_t>((val >> 16) & 0xFF),
        static_cast<uint8_t>((val >> 24) & 0xFF)
    };
}

SoftColorRGBA8 GBuffer::GetNormal(int x, int y) const {
    uint32_t val = m_normalBuffer[y * m_width + x];
    return {
        static_cast<uint8_t>(val & 0xFF),
        static_cast<uint8_t>((val >> 8) & 0xFF),
        static_cast<uint8_t>((val >> 16) & 0xFF),
        static_cast<uint8_t>((val >> 24) & 0xFF)
    };
}

float GBuffer::GetDepth(int x, int y) const {
    return m_depthBuffer[y * m_width + x];
}

bool GBuffer::DepthTest(int x, int y, float depth) {
    int idx = y * m_width + x;
    if (depth < m_depthBuffer[idx]) {
        m_depthBuffer[idx] = depth;
        return true;
    }
    return false;
}

} // namespace WulfNet
