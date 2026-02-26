// =============================================================================
// WulfNet Engine - Perlin Noise Generator
// =============================================================================
// Classic Perlin noise (improved, 2002) with multi-octave fBm helper.
// Header-only. Thread-safe (stateless after construction).
// =============================================================================

#pragma once

#include <cmath>
#include <cstdint>
#include <array>
#include <numeric>
#include <algorithm>

namespace WulfNet {

class PerlinNoise {
public:
    // Default permutation table (Ken Perlin's reference)
    PerlinNoise()
    {
        // Fill with 0..255
        for (int i = 0; i < 256; ++i)
            m_perm[i] = static_cast<uint8_t>(i);

        // Shuffle with a fixed seed for deterministic results
        // (Knuth shuffle seeded from a simple LCG)
        uint32_t seed = 2166136261u;
        for (int i = 255; i > 0; --i) {
            seed = seed * 1103515245u + 12345u;
            int j = static_cast<int>((seed >> 16) % static_cast<uint32_t>(i + 1));
            std::swap(m_perm[i], m_perm[j]);
        }

        // Duplicate for seamless wrapping
        for (int i = 0; i < 256; ++i)
            m_perm[256 + i] = m_perm[i];
    }

    // Seedable constructor for varied patterns
    explicit PerlinNoise(uint32_t seed)
    {
        for (int i = 0; i < 256; ++i)
            m_perm[i] = static_cast<uint8_t>(i);

        for (int i = 255; i > 0; --i) {
            seed = seed * 1103515245u + 12345u;
            int j = static_cast<int>((seed >> 16) % static_cast<uint32_t>(i + 1));
            std::swap(m_perm[i], m_perm[j]);
        }

        for (int i = 0; i < 256; ++i)
            m_perm[256 + i] = m_perm[i];
    }

    // -----------------------------------------------------------------
    // 2D Perlin noise — returns value in approximately [-1, 1]
    // -----------------------------------------------------------------
    float Noise2D(float x, float y) const
    {
        // Grid cell coordinates
        int xi = FloorInt(x);
        int yi = FloorInt(y);

        // Fractional part within cell
        float xf = x - static_cast<float>(xi);
        float yf = y - static_cast<float>(yi);

        // Wrap grid coordinates
        int X = xi & 255;
        int Y = yi & 255;

        // Fade curves for smooth interpolation
        float u = Fade(xf);
        float v = Fade(yf);

        // Hash corners
        int aa = m_perm[m_perm[X    ] + Y    ];
        int ab = m_perm[m_perm[X    ] + Y + 1];
        int ba = m_perm[m_perm[X + 1] + Y    ];
        int bb = m_perm[m_perm[X + 1] + Y + 1];

        // Gradient dot products
        float g00 = Grad2D(aa, xf,       yf);
        float g10 = Grad2D(ba, xf - 1.0f, yf);
        float g01 = Grad2D(ab, xf,       yf - 1.0f);
        float g11 = Grad2D(bb, xf - 1.0f, yf - 1.0f);

        // Bilinear interpolation
        float x0 = Lerp(g00, g10, u);
        float x1 = Lerp(g01, g11, u);
        return Lerp(x0, x1, v);
    }

    // -----------------------------------------------------------------
    // 3D Perlin noise — returns value in approximately [-1, 1]
    // -----------------------------------------------------------------
    float Noise3D(float x, float y, float z) const
    {
        int xi = FloorInt(x);
        int yi = FloorInt(y);
        int zi = FloorInt(z);

        float xf = x - static_cast<float>(xi);
        float yf = y - static_cast<float>(yi);
        float zf = z - static_cast<float>(zi);

        int X = xi & 255;
        int Y = yi & 255;
        int Z = zi & 255;

        float u = Fade(xf);
        float v = Fade(yf);
        float w = Fade(zf);

        int aaa = m_perm[m_perm[m_perm[X    ] + Y    ] + Z    ];
        int aba = m_perm[m_perm[m_perm[X    ] + Y + 1] + Z    ];
        int aab = m_perm[m_perm[m_perm[X    ] + Y    ] + Z + 1];
        int abb = m_perm[m_perm[m_perm[X    ] + Y + 1] + Z + 1];
        int baa = m_perm[m_perm[m_perm[X + 1] + Y    ] + Z    ];
        int bba = m_perm[m_perm[m_perm[X + 1] + Y + 1] + Z    ];
        int bab = m_perm[m_perm[m_perm[X + 1] + Y    ] + Z + 1];
        int bbb = m_perm[m_perm[m_perm[X + 1] + Y + 1] + Z + 1];

        float g000 = Grad3D(aaa, xf,       yf,       zf);
        float g100 = Grad3D(baa, xf - 1.0f, yf,       zf);
        float g010 = Grad3D(aba, xf,       yf - 1.0f, zf);
        float g110 = Grad3D(bba, xf - 1.0f, yf - 1.0f, zf);
        float g001 = Grad3D(aab, xf,       yf,       zf - 1.0f);
        float g101 = Grad3D(bab, xf - 1.0f, yf,       zf - 1.0f);
        float g011 = Grad3D(abb, xf,       yf - 1.0f, zf - 1.0f);
        float g111 = Grad3D(bbb, xf - 1.0f, yf - 1.0f, zf - 1.0f);

        float x0y0 = Lerp(g000, g100, u);
        float x1y0 = Lerp(g010, g110, u);
        float x0y1 = Lerp(g001, g101, u);
        float x1y1 = Lerp(g011, g111, u);

        float y0 = Lerp(x0y0, x1y0, v);
        float y1 = Lerp(x0y1, x1y1, v);

        return Lerp(y0, y1, w);
    }

    // -----------------------------------------------------------------
    // Fractal Brownian Motion (fBm) — multi-octave layered noise
    // Returns value in approximately [-amplitude, amplitude]
    // -----------------------------------------------------------------
    float FBM2D(float x, float y,
                int octaves = 4,
                float lacunarity = 2.0f,   // Frequency multiplier per octave
                float persistence = 0.5f,  // Amplitude multiplier per octave
                float amplitude = 1.0f,
                float frequency = 1.0f) const
    {
        float sum = 0.0f;
        float amp = amplitude;
        float freq = frequency;

        for (int i = 0; i < octaves; ++i) {
            sum += amp * Noise2D(x * freq, y * freq);
            freq *= lacunarity;
            amp  *= persistence;
        }
        return sum;
    }

    float FBM3D(float x, float y, float z,
                int octaves = 4,
                float lacunarity = 2.0f,
                float persistence = 0.5f,
                float amplitude = 1.0f,
                float frequency = 1.0f) const
    {
        float sum = 0.0f;
        float amp = amplitude;
        float freq = frequency;

        for (int i = 0; i < octaves; ++i) {
            sum += amp * Noise3D(x * freq, y * freq, z * freq);
            freq *= lacunarity;
            amp  *= persistence;
        }
        return sum;
    }

private:
    uint8_t m_perm[512];

    static int FloorInt(float x)
    {
        int xi = static_cast<int>(x);
        return (x < static_cast<float>(xi)) ? xi - 1 : xi;
    }

    // Improved Perlin fade curve: 6t⁵ - 15t⁴ + 10t³
    static float Fade(float t)
    {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }

    static float Lerp(float a, float b, float t)
    {
        return a + t * (b - a);
    }

    // 2D gradient — 8 directions (unit vectors at 45° intervals)
    static float Grad2D(int hash, float x, float y)
    {
        int h = hash & 7;
        float u = (h < 4) ? x : y;
        float v = (h < 4) ? y : x;
        return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
    }

    // 3D gradient — 12 directions (edges of a cube)
    static float Grad3D(int hash, float x, float y, float z)
    {
        int h = hash & 15;
        float u = (h < 8) ? x : y;
        float v = (h < 4) ? y : ((h == 12 || h == 14) ? x : z);
        return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
    }
};

} // namespace WulfNet
