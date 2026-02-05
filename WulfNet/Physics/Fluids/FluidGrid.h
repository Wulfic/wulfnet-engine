// =============================================================================
// WulfNet Engine - Fluid Grid for MPM/FLIP
// =============================================================================
// MAC (Marker-And-Cell) staggered grid for fluid simulation.
// Velocities stored at face centers for accurate pressure solve.
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

namespace WulfNet {

// =============================================================================
// Grid Cell State
// =============================================================================

enum class CellState : uint8_t {
    Empty = 0,      // No fluid
    Fluid = 1,      // Contains fluid
    Solid = 2,      // Solid boundary
    Air = 3         // Air (for free surface)
};

// =============================================================================
// MAC Grid Cell (staggered velocities)
// =============================================================================

struct MACCell {
    // Face velocities (staggered grid)
    float u = 0.0f;     // X-velocity at (i-0.5, j, k)
    float v = 0.0f;     // Y-velocity at (i, j-0.5, k)
    float w = 0.0f;     // Z-velocity at (i, j, k-0.5)

    // Face weights (for FLIP transfer)
    float uWeight = 0.0f;
    float vWeight = 0.0f;
    float wWeight = 0.0f;

    // Cell-centered values
    float pressure = 0.0f;
    float density = 0.0f;
    float divergence = 0.0f;

    // State
    CellState state = CellState::Empty;
    uint8_t layer = 0;          // Distance from fluid (for extrapolation)
    uint16_t particleCount = 0;

    // Temporary for pressure solve
    float rhs = 0.0f;           // Right-hand side
    float Adiag = 0.0f;         // Diagonal coefficient
    float Ax = 0.0f, Ay = 0.0f, Az = 0.0f;  // Off-diagonal coefficients

    void Reset() {
        u = v = w = 0.0f;
        uWeight = vWeight = wWeight = 0.0f;
        pressure = 0.0f;
        density = 0.0f;
        divergence = 0.0f;
        state = CellState::Empty;
        layer = 255;
        particleCount = 0;
    }
};

// =============================================================================
// Fluid Grid
// =============================================================================

class FluidGrid {
public:
    FluidGrid() = default;
    ~FluidGrid() = default;

    // Initialization
    bool Initialize(uint32_t resX, uint32_t resY, uint32_t resZ, float cellSize);
    void SetBounds(float minX, float minY, float minZ,
                   float maxX, float maxY, float maxZ);
    void Reset();

    // Accessors
    uint32_t GetResolutionX() const { return m_resX; }
    uint32_t GetResolutionY() const { return m_resY; }
    uint32_t GetResolutionZ() const { return m_resZ; }
    float GetCellSize() const { return m_cellSize; }
    float GetInvCellSize() const { return m_invCellSize; }

    // Cell access
    MACCell& GetCell(uint32_t i, uint32_t j, uint32_t k);
    const MACCell& GetCell(uint32_t i, uint32_t j, uint32_t k) const;
    MACCell& GetCellClamped(int i, int j, int k);

    // Index conversion
    uint32_t GetIndex(uint32_t i, uint32_t j, uint32_t k) const {
        return i + j * m_resX + k * m_resX * m_resY;
    }

    void GetIJK(uint32_t index, uint32_t& i, uint32_t& j, uint32_t& k) const {
        i = index % m_resX;
        j = (index / m_resX) % m_resY;
        k = index / (m_resX * m_resY);
    }

    // World <-> Grid conversion
    void WorldToGrid(float wx, float wy, float wz,
                     float& gx, float& gy, float& gz) const {
        gx = (wx - m_minX) * m_invCellSize;
        gy = (wy - m_minY) * m_invCellSize;
        gz = (wz - m_minZ) * m_invCellSize;
    }

    void GridToWorld(float gx, float gy, float gz,
                     float& wx, float& wy, float& wz) const {
        wx = gx * m_cellSize + m_minX;
        wy = gy * m_cellSize + m_minY;
        wz = gz * m_cellSize + m_minZ;
    }

    void WorldToCell(float wx, float wy, float wz,
                     int& i, int& j, int& k) const {
        i = static_cast<int>((wx - m_minX) * m_invCellSize);
        j = static_cast<int>((wy - m_minY) * m_invCellSize);
        k = static_cast<int>((wz - m_minZ) * m_invCellSize);
    }

    // Interpolation (trilinear)
    float InterpolateU(float gx, float gy, float gz) const;
    float InterpolateV(float gx, float gy, float gz) const;
    float InterpolateW(float gx, float gy, float gz) const;
    void InterpolateVelocity(float gx, float gy, float gz,
                             float& vx, float& vy, float& vz) const;

    // Gradient (for APIC)
    void VelocityGradient(float gx, float gy, float gz,
                          float& dudx, float& dudy, float& dudz,
                          float& dvdx, float& dvdy, float& dvdz,
                          float& dwdx, float& dwdy, float& dwdz) const;

    // Bounds checking
    bool IsInBounds(int i, int j, int k) const {
        return i >= 0 && i < static_cast<int>(m_resX) &&
               j >= 0 && j < static_cast<int>(m_resY) &&
               k >= 0 && k < static_cast<int>(m_resZ);
    }

    bool IsInBoundsWorld(float wx, float wy, float wz) const {
        return wx >= m_minX && wx <= m_maxX &&
               wy >= m_minY && wy <= m_maxY &&
               wz >= m_minZ && wz <= m_maxZ;
    }

    // For GPU transfer
    MACCell* GetData() { return m_cells.data(); }
    const MACCell* GetData() const { return m_cells.data(); }
    size_t GetCellCount() const { return m_cells.size(); }
    size_t GetDataSize() const { return m_cells.size() * sizeof(MACCell); }

    // Pressure solve helpers
    void MarkCellStates();
    void ExtrapolateVelocity(int layers);
    float ComputeDivergence();
    void ApplyPressureGradient();

private:
    // Grid dimensions
    uint32_t m_resX = 0, m_resY = 0, m_resZ = 0;
    float m_cellSize = 1.0f;
    float m_invCellSize = 1.0f;

    // World bounds
    float m_minX = 0.0f, m_minY = 0.0f, m_minZ = 0.0f;
    float m_maxX = 1.0f, m_maxY = 1.0f, m_maxZ = 1.0f;

    // Cell storage
    std::vector<MACCell> m_cells;

    // Kernel weight function (quadratic B-spline)
    static float Weight(float x) {
        x = std::abs(x);
        if (x < 0.5f) {
            return 0.75f - x * x;
        } else if (x < 1.5f) {
            return 0.5f * (1.5f - x) * (1.5f - x);
        }
        return 0.0f;
    }

    static float WeightGradient(float x) {
        float absX = std::abs(x);
        if (absX < 0.5f) {
            return -2.0f * x;
        } else if (absX < 1.5f) {
            return x > 0 ? x - 1.5f : 1.5f + x;
        }
        return 0.0f;
    }
};

// =============================================================================
// Inline implementations
// =============================================================================

inline MACCell& FluidGrid::GetCell(uint32_t i, uint32_t j, uint32_t k) {
    return m_cells[GetIndex(i, j, k)];
}

inline const MACCell& FluidGrid::GetCell(uint32_t i, uint32_t j, uint32_t k) const {
    return m_cells[GetIndex(i, j, k)];
}

inline MACCell& FluidGrid::GetCellClamped(int i, int j, int k) {
    i = std::max(0, std::min(i, static_cast<int>(m_resX) - 1));
    j = std::max(0, std::min(j, static_cast<int>(m_resY) - 1));
    k = std::max(0, std::min(k, static_cast<int>(m_resZ) - 1));
    return m_cells[GetIndex(static_cast<uint32_t>(i),
                            static_cast<uint32_t>(j),
                            static_cast<uint32_t>(k))];
}

} // namespace WulfNet
