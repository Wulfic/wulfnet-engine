// =============================================================================
// WulfNet Engine - CO-FLIP Interpolation & Grid Utilities
// =============================================================================
// B-spline basis functions, grid helpers, and divergence-free interpolation.
// Extracted from COFLIPSystem.cpp for maintainability.
// =============================================================================

#include "COFLIPSystem.h"
#include <cmath>

namespace WulfNet {

// =============================================================================
// B-Spline Basis Functions
// =============================================================================
// BSpline/BSplineDerivative now defined as __forceinline in COFLIPSystem.h
// for cross-TU inlining.  QuadraticBSpline remains file-local.

// Quadratic B-spline (faster than cubic, 3x3x3=27 vs 4x4x4=64 samples)
// Centered at 0, support [-1.5, 1.5]
static __forceinline float QuadraticBSpline(float x) {
    float ax = std::abs(x);
    if (ax < 0.5f) {
        return 0.75f - ax * ax;
    } else if (ax < 1.5f) {
        float t = 1.5f - ax;
        return 0.5f * t * t;
    }
    return 0.0f;
}

// =============================================================================
// Grid Helpers — now defined inline in COFLIPSystem.h
// =============================================================================

// =============================================================================
// Divergence-Free Interpolation (Key CO-FLIP Innovation)
// =============================================================================

void COFLIPSystem::InterpolateDivergenceFree(float x, float y, float z, float& vx, float& vy, float& vz) const {
    // Convert to grid coordinates
    float gx, gy, gz;
    WorldToGrid(x, y, z, gx, gy, gz);

    // MAC grid: u is at (i+0.5, j, k), v is at (i, j+0.5, k), w is at (i, j, k+0.5)
    // Cubic B-spline interpolation with factored 1D weights:
    // Precompute 4 weights per dimension (12 BSpline calls total)
    // instead of evaluating BSpline 3x per grid point (192 calls total).

    vx = 0; vy = 0; vz = 0;
    float totalWeightU = 0, totalWeightV = 0, totalWeightW = 0;

    const int NX = static_cast<int>(m_config.gridSizeX);
    const int NY = static_cast<int>(m_config.gridSizeY);
    const int NZ = static_cast<int>(m_config.gridSizeZ);

    // --- Interpolate u (at face centers offset by 0.5 in x) ---
    {
        float ux = gx - 0.5f, uy = gy, uz = gz;
        int i0 = static_cast<int>(std::floor(ux)) - 1;
        int j0 = static_cast<int>(std::floor(uy)) - 1;
        int k0 = static_cast<int>(std::floor(uz)) - 1;

        float wx[4], wy[4], wz[4];
        for (int d = 0; d < 4; ++d) {
            wx[d] = BSpline(ux - (i0 + d));
            wy[d] = BSpline(uy - (j0 + d));
            wz[d] = BSpline(uz - (k0 + d));
        }

        for (int dk = 0; dk < 4; ++dk) {
            int k = k0 + dk;
            if (k < 0 || k >= NZ) continue;
            float wk = wz[dk];
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0 + dj;
                if (j < 0 || j >= NY) continue;
                float wjk = wy[dj] * wk;
                for (int di = 0; di < 4; ++di) {
                    int i = i0 + di;
                    if (i < 0 || i >= NX) continue;
                    float w = wx[di] * wjk;
                    vx += w * m_grid[GridIndex(i, j, k)].u;
                    totalWeightU += w;
                }
            }
        }
    }

    // --- Interpolate v (at face centers offset by 0.5 in y) ---
    {
        float vxg = gx, vyg = gy - 0.5f, vzg = gz;
        int i0 = static_cast<int>(std::floor(vxg)) - 1;
        int j0 = static_cast<int>(std::floor(vyg)) - 1;
        int k0 = static_cast<int>(std::floor(vzg)) - 1;

        float wx[4], wy[4], wz[4];
        for (int d = 0; d < 4; ++d) {
            wx[d] = BSpline(vxg - (i0 + d));
            wy[d] = BSpline(vyg - (j0 + d));
            wz[d] = BSpline(vzg - (k0 + d));
        }

        for (int dk = 0; dk < 4; ++dk) {
            int k = k0 + dk;
            if (k < 0 || k >= NZ) continue;
            float wk = wz[dk];
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0 + dj;
                if (j < 0 || j >= NY) continue;
                float wjk = wy[dj] * wk;
                for (int di = 0; di < 4; ++di) {
                    int i = i0 + di;
                    if (i < 0 || i >= NX) continue;
                    float w = wx[di] * wjk;
                    vy += w * m_grid[GridIndex(i, j, k)].v;
                    totalWeightV += w;
                }
            }
        }
    }

    // --- Interpolate w (at face centers offset by 0.5 in z) ---
    {
        float wxg = gx, wyg = gy, wzg = gz - 0.5f;
        int i0 = static_cast<int>(std::floor(wxg)) - 1;
        int j0 = static_cast<int>(std::floor(wyg)) - 1;
        int k0 = static_cast<int>(std::floor(wzg)) - 1;

        float wx[4], wy[4], wz[4];
        for (int d = 0; d < 4; ++d) {
            wx[d] = BSpline(wxg - (i0 + d));
            wy[d] = BSpline(wyg - (j0 + d));
            wz[d] = BSpline(wzg - (k0 + d));
        }

        for (int dk = 0; dk < 4; ++dk) {
            int k = k0 + dk;
            if (k < 0 || k >= NZ) continue;
            float wk = wz[dk];
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0 + dj;
                if (j < 0 || j >= NY) continue;
                float wjk = wy[dj] * wk;
                for (int di = 0; di < 4; ++di) {
                    int i = i0 + di;
                    if (i < 0 || i >= NX) continue;
                    float w = wx[di] * wjk;
                    vz += w * m_grid[GridIndex(i, j, k)].w;
                    totalWeightW += w;
                }
            }
        }
    }

    // Normalize
    if (totalWeightU > 0) vx /= totalWeightU;
    if (totalWeightV > 0) vy /= totalWeightV;
    if (totalWeightW > 0) vz /= totalWeightW;
}

// Optimized version using quadratic B-spline with pre-factored 1D weights
// (9 QuadraticBSpline calls total instead of 81 per-sample calls)
void COFLIPSystem::InterpolateDivergenceFreeQuadratic(float x, float y, float z, float& vx, float& vy, float& vz) const {
    float gx, gy, gz;
    WorldToGrid(x, y, z, gx, gy, gz);

    vx = 0; vy = 0; vz = 0;
    float totalWeightU = 0, totalWeightV = 0, totalWeightW = 0;

    const int NX = static_cast<int>(m_config.gridSizeX);
    const int NY = static_cast<int>(m_config.gridSizeY);
    const int NZ = static_cast<int>(m_config.gridSizeZ);

    // --- Interpolate u (face offset 0.5 in x) ---
    {
        float ux = gx - 0.5f, uy = gy, uz = gz;
        int i0 = static_cast<int>(std::floor(ux + 0.5f)) - 1;
        int j0 = static_cast<int>(std::floor(uy + 0.5f)) - 1;
        int k0 = static_cast<int>(std::floor(uz + 0.5f)) - 1;

        float wx[3], wy[3], wz[3];
        for (int d = 0; d < 3; ++d) {
            wx[d] = QuadraticBSpline(ux - (i0 + d));
            wy[d] = QuadraticBSpline(uy - (j0 + d));
            wz[d] = QuadraticBSpline(uz - (k0 + d));
        }

        for (int dk = 0; dk < 3; ++dk) {
            int k = k0 + dk;
            if (k < 0 || k >= NZ) continue;
            float wk = wz[dk];
            for (int dj = 0; dj < 3; ++dj) {
                int j = j0 + dj;
                if (j < 0 || j >= NY) continue;
                float wjk = wy[dj] * wk;
                for (int di = 0; di < 3; ++di) {
                    int i = i0 + di;
                    if (i < 0 || i >= NX) continue;
                    float w = wx[di] * wjk;
                    vx += w * m_grid[GridIndex(i, j, k)].u;
                    totalWeightU += w;
                }
            }
        }
    }

    // --- Interpolate v (face offset 0.5 in y) ---
    {
        float vxg = gx, vyg = gy - 0.5f, vzg = gz;
        int i0 = static_cast<int>(std::floor(vxg + 0.5f)) - 1;
        int j0 = static_cast<int>(std::floor(vyg + 0.5f)) - 1;
        int k0 = static_cast<int>(std::floor(vzg + 0.5f)) - 1;

        float wx[3], wy[3], wz[3];
        for (int d = 0; d < 3; ++d) {
            wx[d] = QuadraticBSpline(vxg - (i0 + d));
            wy[d] = QuadraticBSpline(vyg - (j0 + d));
            wz[d] = QuadraticBSpline(vzg - (k0 + d));
        }

        for (int dk = 0; dk < 3; ++dk) {
            int k = k0 + dk;
            if (k < 0 || k >= NZ) continue;
            float wk = wz[dk];
            for (int dj = 0; dj < 3; ++dj) {
                int j = j0 + dj;
                if (j < 0 || j >= NY) continue;
                float wjk = wy[dj] * wk;
                for (int di = 0; di < 3; ++di) {
                    int i = i0 + di;
                    if (i < 0 || i >= NX) continue;
                    float w = wx[di] * wjk;
                    vy += w * m_grid[GridIndex(i, j, k)].v;
                    totalWeightV += w;
                }
            }
        }
    }

    // --- Interpolate w (face offset 0.5 in z) ---
    {
        float wxg = gx, wyg = gy, wzg = gz - 0.5f;
        int i0 = static_cast<int>(std::floor(wxg + 0.5f)) - 1;
        int j0 = static_cast<int>(std::floor(wyg + 0.5f)) - 1;
        int k0 = static_cast<int>(std::floor(wzg + 0.5f)) - 1;

        float wx[3], wy[3], wz[3];
        for (int d = 0; d < 3; ++d) {
            wx[d] = QuadraticBSpline(wxg - (i0 + d));
            wy[d] = QuadraticBSpline(wyg - (j0 + d));
            wz[d] = QuadraticBSpline(wzg - (k0 + d));
        }

        for (int dk = 0; dk < 3; ++dk) {
            int k = k0 + dk;
            if (k < 0 || k >= NZ) continue;
            float wk = wz[dk];
            for (int dj = 0; dj < 3; ++dj) {
                int j = j0 + dj;
                if (j < 0 || j >= NY) continue;
                float wjk = wy[dj] * wk;
                for (int di = 0; di < 3; ++di) {
                    int i = i0 + di;
                    if (i < 0 || i >= NX) continue;
                    float w = wx[di] * wjk;
                    vz += w * m_grid[GridIndex(i, j, k)].w;
                    totalWeightW += w;
                }
            }
        }
    }

    // Normalize
    if (totalWeightU > 0) vx /= totalWeightU;
    if (totalWeightV > 0) vy /= totalWeightV;
    if (totalWeightW > 0) vz /= totalWeightW;
}

void COFLIPSystem::InterpolateVelocityGradient(float x, float y, float z, float grad[9]) const {
    // Compute velocity gradient tensor using analytical B-spline derivatives.
    // This replaces the old 6 x InterpolateDivergenceFree finite-difference
    // approach (~1152 BSpline evals) with a single pass per velocity component
    // using BSplineDerivative (~192 evals + ~192 derivative evals = ~384 total).

    float gx, gy, gz;
    WorldToGrid(x, y, z, gx, gy, gz);
    float invDx = 1.0f / m_config.cellSize;

    for (int n = 0; n < 9; ++n) grad[n] = 0;

    // --- du/dx, du/dy, du/dz (u lives at face offset 0.5 in x) ---
    {
        float ux = gx - 0.5f, uy = gy, uz = gz;
        int i0 = static_cast<int>(std::floor(ux)) - 1;
        int j0 = static_cast<int>(std::floor(uy)) - 1;
        int k0 = static_cast<int>(std::floor(uz)) - 1;

        for (int dk = 0; dk < 4; ++dk) {
            int k = k0 + dk;
            if (k < 0 || k >= (int)m_config.gridSizeZ) continue;
            float wz  = BSpline(uz - k);
            float dwz = BSplineDerivative(uz - k);
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0 + dj;
                if (j < 0 || j >= (int)m_config.gridSizeY) continue;
                float wy  = BSpline(uy - j);
                float dwy = BSplineDerivative(uy - j);
                for (int di = 0; di < 4; ++di) {
                    int i = i0 + di;
                    if (i < 0 || i >= (int)m_config.gridSizeX) continue;
                    float wx  = BSpline(ux - i);
                    float dwx = BSplineDerivative(ux - i);
                    float uVal = m_grid[GridIndex(i, j, k)].u;
                    grad[0] += dwx * wy  * wz  * uVal; // du/dx
                    grad[1] += wx  * dwy * wz  * uVal; // du/dy
                    grad[2] += wx  * wy  * dwz * uVal; // du/dz
                }
            }
        }
    }

    // --- dv/dx, dv/dy, dv/dz (v lives at face offset 0.5 in y) ---
    {
        float vxg = gx, vyg = gy - 0.5f, vzg = gz;
        int i0 = static_cast<int>(std::floor(vxg)) - 1;
        int j0 = static_cast<int>(std::floor(vyg)) - 1;
        int k0 = static_cast<int>(std::floor(vzg)) - 1;

        for (int dk = 0; dk < 4; ++dk) {
            int k = k0 + dk;
            if (k < 0 || k >= (int)m_config.gridSizeZ) continue;
            float wz  = BSpline(vzg - k);
            float dwz = BSplineDerivative(vzg - k);
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0 + dj;
                if (j < 0 || j >= (int)m_config.gridSizeY) continue;
                float wy  = BSpline(vyg - j);
                float dwy = BSplineDerivative(vyg - j);
                for (int di = 0; di < 4; ++di) {
                    int i = i0 + di;
                    if (i < 0 || i >= (int)m_config.gridSizeX) continue;
                    float wx  = BSpline(vxg - i);
                    float dwx = BSplineDerivative(vxg - i);
                    float vVal = m_grid[GridIndex(i, j, k)].v;
                    grad[3] += dwx * wy  * wz  * vVal; // dv/dx
                    grad[4] += wx  * dwy * wz  * vVal; // dv/dy
                    grad[5] += wx  * wy  * dwz * vVal; // dv/dz
                }
            }
        }
    }

    // --- dw/dx, dw/dy, dw/dz (w lives at face offset 0.5 in z) ---
    {
        float wxg = gx, wyg = gy, wzg = gz - 0.5f;
        int i0 = static_cast<int>(std::floor(wxg)) - 1;
        int j0 = static_cast<int>(std::floor(wyg)) - 1;
        int k0 = static_cast<int>(std::floor(wzg)) - 1;

        for (int dk = 0; dk < 4; ++dk) {
            int k = k0 + dk;
            if (k < 0 || k >= (int)m_config.gridSizeZ) continue;
            float wz  = BSpline(wzg - k);
            float dwz = BSplineDerivative(wzg - k);
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0 + dj;
                if (j < 0 || j >= (int)m_config.gridSizeY) continue;
                float wy  = BSpline(wyg - j);
                float dwy = BSplineDerivative(wyg - j);
                for (int di = 0; di < 4; ++di) {
                    int i = i0 + di;
                    if (i < 0 || i >= (int)m_config.gridSizeX) continue;
                    float wx  = BSpline(wxg - i);
                    float dwx = BSplineDerivative(wxg - i);
                    float wVal = m_grid[GridIndex(i, j, k)].w;
                    grad[6] += dwx * wy  * wz  * wVal; // dw/dx
                    grad[7] += wx  * dwy * wz  * wVal; // dw/dy
                    grad[8] += wx  * wy  * dwz * wVal; // dw/dz
                }
            }
        }
    }

    // Scale derivatives from grid-space to world-space
    for (int n = 0; n < 9; ++n) grad[n] *= invDx;
}

} // namespace WulfNet
