// =============================================================================
// WulfNet Engine - Fluid Grid Implementation
// =============================================================================

#include "FluidGrid.h"
#include <cstring>

namespace WulfNet {

bool FluidGrid::Initialize(uint32_t resX, uint32_t resY, uint32_t resZ, float cellSize) {
    m_resX = resX;
    m_resY = resY;
    m_resZ = resZ;
    m_cellSize = cellSize;
    m_invCellSize = 1.0f / cellSize;

    size_t totalCells = static_cast<size_t>(resX) * resY * resZ;
    m_cells.resize(totalCells);

    Reset();
    return true;
}

void FluidGrid::SetBounds(float minX, float minY, float minZ,
                          float maxX, float maxY, float maxZ) {
    m_minX = minX; m_minY = minY; m_minZ = minZ;
    m_maxX = maxX; m_maxY = maxY; m_maxZ = maxZ;
}

void FluidGrid::Reset() {
    for (auto& cell : m_cells) {
        cell.Reset();
    }
}

float FluidGrid::InterpolateU(float gx, float gy, float gz) const {
    // U is stored at (i+0.5, j, k), so offset appropriately
    float fx = gx - 0.5f;
    float fy = gy;
    float fz = gz;

    int i0 = static_cast<int>(std::floor(fx));
    int j0 = static_cast<int>(std::floor(fy - 0.5f));
    int k0 = static_cast<int>(std::floor(fz - 0.5f));

    float result = 0.0f;

    for (int di = 0; di <= 1; ++di) {
        for (int dj = 0; dj <= 1; ++dj) {
            for (int dk = 0; dk <= 1; ++dk) {
                int i = i0 + di;
                int j = j0 + dj;
                int k = k0 + dk;

                if (!IsInBounds(i, j, k)) continue;

                float wx = 1.0f - std::abs(fx - i);
                float wy = 1.0f - std::abs(fy - 0.5f - j);
                float wz = 1.0f - std::abs(fz - 0.5f - k);
                float w = wx * wy * wz;

                result += w * GetCell(i, j, k).u;
            }
        }
    }

    return result;
}

float FluidGrid::InterpolateV(float gx, float gy, float gz) const {
    // V is stored at (i, j+0.5, k)
    float fx = gx;
    float fy = gy - 0.5f;
    float fz = gz;

    int i0 = static_cast<int>(std::floor(fx - 0.5f));
    int j0 = static_cast<int>(std::floor(fy));
    int k0 = static_cast<int>(std::floor(fz - 0.5f));

    float result = 0.0f;

    for (int di = 0; di <= 1; ++di) {
        for (int dj = 0; dj <= 1; ++dj) {
            for (int dk = 0; dk <= 1; ++dk) {
                int i = i0 + di;
                int j = j0 + dj;
                int k = k0 + dk;

                if (!IsInBounds(i, j, k)) continue;

                float wx = 1.0f - std::abs(fx - 0.5f - i);
                float wy = 1.0f - std::abs(fy - j);
                float wz = 1.0f - std::abs(fz - 0.5f - k);
                float w = wx * wy * wz;

                result += w * GetCell(i, j, k).v;
            }
        }
    }

    return result;
}

float FluidGrid::InterpolateW(float gx, float gy, float gz) const {
    // W is stored at (i, j, k+0.5)
    float fx = gx;
    float fy = gy;
    float fz = gz - 0.5f;

    int i0 = static_cast<int>(std::floor(fx - 0.5f));
    int j0 = static_cast<int>(std::floor(fy - 0.5f));
    int k0 = static_cast<int>(std::floor(fz));

    float result = 0.0f;

    for (int di = 0; di <= 1; ++di) {
        for (int dj = 0; dj <= 1; ++dj) {
            for (int dk = 0; dk <= 1; ++dk) {
                int i = i0 + di;
                int j = j0 + dj;
                int k = k0 + dk;

                if (!IsInBounds(i, j, k)) continue;

                float wx = 1.0f - std::abs(fx - 0.5f - i);
                float wy = 1.0f - std::abs(fy - 0.5f - j);
                float wz = 1.0f - std::abs(fz - k);
                float w = wx * wy * wz;

                result += w * GetCell(i, j, k).w;
            }
        }
    }

    return result;
}

void FluidGrid::InterpolateVelocity(float gx, float gy, float gz,
                                    float& vx, float& vy, float& vz) const {
    vx = InterpolateU(gx, gy, gz);
    vy = InterpolateV(gx, gy, gz);
    vz = InterpolateW(gx, gy, gz);
}

void FluidGrid::VelocityGradient(float gx, float gy, float gz,
                                 float& dudx, float& dudy, float& dudz,
                                 float& dvdx, float& dvdy, float& dvdz,
                                 float& dwdx, float& dwdy, float& dwdz) const {
    // Finite differences for gradient
    float eps = 0.5f;

    float uP, uM, vP, vM, wP, wM;

    // du/dx, dv/dx, dw/dx
    uP = InterpolateU(gx + eps, gy, gz);
    uM = InterpolateU(gx - eps, gy, gz);
    vP = InterpolateV(gx + eps, gy, gz);
    vM = InterpolateV(gx - eps, gy, gz);
    wP = InterpolateW(gx + eps, gy, gz);
    wM = InterpolateW(gx - eps, gy, gz);

    dudx = (uP - uM) / (2.0f * eps);
    dvdx = (vP - vM) / (2.0f * eps);
    dwdx = (wP - wM) / (2.0f * eps);

    // du/dy, dv/dy, dw/dy
    uP = InterpolateU(gx, gy + eps, gz);
    uM = InterpolateU(gx, gy - eps, gz);
    vP = InterpolateV(gx, gy + eps, gz);
    vM = InterpolateV(gx, gy - eps, gz);
    wP = InterpolateW(gx, gy + eps, gz);
    wM = InterpolateW(gx, gy - eps, gz);

    dudy = (uP - uM) / (2.0f * eps);
    dvdy = (vP - vM) / (2.0f * eps);
    dwdy = (wP - wM) / (2.0f * eps);

    // du/dz, dv/dz, dw/dz
    uP = InterpolateU(gx, gy, gz + eps);
    uM = InterpolateU(gx, gy, gz - eps);
    vP = InterpolateV(gx, gy, gz + eps);
    vM = InterpolateV(gx, gy, gz - eps);
    wP = InterpolateW(gx, gy, gz + eps);
    wM = InterpolateW(gx, gy, gz - eps);

    dudz = (uP - uM) / (2.0f * eps);
    dvdz = (vP - vM) / (2.0f * eps);
    dwdz = (wP - wM) / (2.0f * eps);
}

void FluidGrid::MarkCellStates() {
    // Mark cells based on particle count
    for (size_t idx = 0; idx < m_cells.size(); ++idx) {
        auto& cell = m_cells[idx];
        if (cell.particleCount > 0) {
            cell.state = CellState::Fluid;
        } else {
            cell.state = CellState::Air;
        }
        cell.layer = (cell.state == CellState::Fluid) ? 0 : 255;
    }

    // Mark boundary cells as solid
    for (uint32_t k = 0; k < m_resZ; ++k) {
        for (uint32_t j = 0; j < m_resY; ++j) {
            for (uint32_t i = 0; i < m_resX; ++i) {
                if (i == 0 || i == m_resX - 1 ||
                    j == 0 || j == m_resY - 1 ||
                    k == 0 || k == m_resZ - 1) {
                    GetCell(i, j, k).state = CellState::Solid;
                }
            }
        }
    }
}

void FluidGrid::ExtrapolateVelocity(int layers) {
    // Simple wavefront extrapolation
    for (int layer = 0; layer < layers; ++layer) {
        for (uint32_t k = 1; k < m_resZ - 1; ++k) {
            for (uint32_t j = 1; j < m_resY - 1; ++j) {
                for (uint32_t i = 1; i < m_resX - 1; ++i) {
                    auto& cell = GetCell(i, j, k);
                    if (cell.layer != 255) continue;

                    // Check neighbors
                    float sumU = 0, sumV = 0, sumW = 0;
                    int count = 0;

                    auto checkNeighbor = [&](int ni, int nj, int nk) {
                        if (!IsInBounds(ni, nj, nk)) return;
                        const auto& neighbor = GetCell(ni, nj, nk);
                        if (neighbor.layer == static_cast<uint8_t>(layer)) {
                            sumU += neighbor.u;
                            sumV += neighbor.v;
                            sumW += neighbor.w;
                            count++;
                        }
                    };

                    checkNeighbor(i - 1, j, k);
                    checkNeighbor(i + 1, j, k);
                    checkNeighbor(i, j - 1, k);
                    checkNeighbor(i, j + 1, k);
                    checkNeighbor(i, j, k - 1);
                    checkNeighbor(i, j, k + 1);

                    if (count > 0) {
                        cell.u = sumU / count;
                        cell.v = sumV / count;
                        cell.w = sumW / count;
                        cell.layer = static_cast<uint8_t>(layer + 1);
                    }
                }
            }
        }
    }
}

float FluidGrid::ComputeDivergence() {
    float maxDiv = 0.0f;

    for (uint32_t k = 1; k < m_resZ - 1; ++k) {
        for (uint32_t j = 1; j < m_resY - 1; ++j) {
            for (uint32_t i = 1; i < m_resX - 1; ++i) {
                auto& cell = GetCell(i, j, k);
                if (cell.state != CellState::Fluid) {
                    cell.divergence = 0.0f;
                    continue;
                }

                // Divergence = du/dx + dv/dy + dw/dz
                float div = 0.0f;
                div += GetCell(i + 1, j, k).u - cell.u;
                div += GetCell(i, j + 1, k).v - cell.v;
                div += GetCell(i, j, k + 1).w - cell.w;
                div *= m_invCellSize;

                cell.divergence = div;
                maxDiv = std::max(maxDiv, std::abs(div));
            }
        }
    }

    return maxDiv;
}

void FluidGrid::ApplyPressureGradient() {
    float scale = 1.0f / m_cellSize;

    for (uint32_t k = 1; k < m_resZ - 1; ++k) {
        for (uint32_t j = 1; j < m_resY - 1; ++j) {
            for (uint32_t i = 1; i < m_resX - 1; ++i) {
                auto& cell = GetCell(i, j, k);

                // Update face velocities
                if (cell.state == CellState::Fluid ||
                    GetCell(i - 1, j, k).state == CellState::Fluid) {
                    cell.u -= scale * (cell.pressure - GetCell(i - 1, j, k).pressure);
                }

                if (cell.state == CellState::Fluid ||
                    GetCell(i, j - 1, k).state == CellState::Fluid) {
                    cell.v -= scale * (cell.pressure - GetCell(i, j - 1, k).pressure);
                }

                if (cell.state == CellState::Fluid ||
                    GetCell(i, j, k - 1).state == CellState::Fluid) {
                    cell.w -= scale * (cell.pressure - GetCell(i, j, k - 1).pressure);
                }
            }
        }
    }
}

} // namespace WulfNet
