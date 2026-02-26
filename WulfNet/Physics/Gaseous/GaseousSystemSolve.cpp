// =============================================================================
// WulfNet Engine - Gaseous System Solver Functions
// =============================================================================
// Obstacle marking, buoyancy, combustion, vorticity, divergence, pressure
// solve, advection, dissipation, and stats.
// Extracted from GaseousSystem.cpp for maintainability.
// =============================================================================

#include "GaseousSystem.h"
#include <cstring>
#include <cmath>

#ifdef WULFNET_HAS_OPENMP
#include <omp.h>
#endif

namespace WulfNet {
// =============================================================================
// Mark Obstacles
// =============================================================================

void GaseousSystem::MarkObstacles() {
    // Reset all cells to Air first (unless they have density)
    for (auto& c : m_cells) {
        if (c.state == GasCell::State::Solid) {
            c.state = GasCell::State::Air;
        }
    }

    for (const auto& obs : m_obstacles) {
        if (!obs.enabled) continue;

        float gx, gy, gz;
        WorldToGrid(obs.posX, obs.posY, obs.posZ, gx, gy, gz);

        if (obs.shape == GasObstacle::Shape::Sphere) {
            float rCells = obs.radius * m_invCellSize;
            int r = static_cast<int>(std::ceil(rCells));
            int ci = static_cast<int>(gx);
            int cj = static_cast<int>(gy);
            int ck = static_cast<int>(gz);

            for (int dk = -r; dk <= r; ++dk) {
                for (int dj = -r; dj <= r; ++dj) {
                    for (int di = -r; di <= r; ++di) {
                        float dist = std::sqrt(static_cast<float>(
                            di * di + dj * dj + dk * dk));
                        if (dist > rCells) continue;

                        int gi = ci + di, gj = cj + dj, gk = ck + dk;
                        if (!InBounds(gi, gj, gk)) continue;

                        auto& cell = m_cells[CellIndex(
                            static_cast<uint32_t>(gi),
                            static_cast<uint32_t>(gj),
                            static_cast<uint32_t>(gk))];
                        cell.state = GasCell::State::Solid;
                        cell.u = cell.v = cell.w = 0.0f;
                    }
                }
            }
        } else {
            // Box obstacle
            float hx = obs.halfExtentX * m_invCellSize;
            float hy = obs.halfExtentY * m_invCellSize;
            float hz = obs.halfExtentZ * m_invCellSize;

            int minI = std::max(0, static_cast<int>(gx - hx));
            int maxI = std::min(static_cast<int>(m_resX) - 1, static_cast<int>(gx + hx));
            int minJ = std::max(0, static_cast<int>(gy - hy));
            int maxJ = std::min(static_cast<int>(m_resY) - 1, static_cast<int>(gy + hy));
            int minK = std::max(0, static_cast<int>(gz - hz));
            int maxK = std::min(static_cast<int>(m_resZ) - 1, static_cast<int>(gz + hz));

            for (int k = minK; k <= maxK; ++k) {
                for (int j = minJ; j <= maxJ; ++j) {
                    for (int i = minI; i <= maxI; ++i) {
                        auto& cell = m_cells[CellIndex(
                            static_cast<uint32_t>(i),
                            static_cast<uint32_t>(j),
                            static_cast<uint32_t>(k))];
                        cell.state = GasCell::State::Solid;
                        cell.u = cell.v = cell.w = 0.0f;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Buoyancy Forces
// =============================================================================

void GaseousSystem::ApplyBuoyancy(float dt) {
    float alpha = m_config.buoyancyAlpha;
    float beta = m_config.buoyancyBeta;

    for (uint32_t k = 0; k < m_resZ; ++k) {
        for (uint32_t j = 0; j < m_resY; ++j) {
            for (uint32_t i = 0; i < m_resX; ++i) {
                auto& cell = m_cells[CellIndex(i, j, k)];
                if (cell.state == GasCell::State::Solid) continue;

                // Buoyancy: f_y = -alpha * density + beta * temperature
                // Density pulls down, temperature pushes up
                float buoyancy = -alpha * cell.density + beta * cell.temperature;
                cell.v += buoyancy * dt;
            }
        }
    }
}

// =============================================================================
// Combustion (Fire Simulation)
// =============================================================================

void GaseousSystem::ApplyCombustion(float dt) {
    for (uint32_t k = 0; k < m_resZ; ++k) {
        for (uint32_t j = 0; j < m_resY; ++j) {
            for (uint32_t i = 0; i < m_resX; ++i) {
                auto& cell = m_cells[CellIndex(i, j, k)];
                if (cell.state == GasCell::State::Solid) continue;
                if (cell.fuel <= 0.0f) continue;

                // Check ignition
                float totalTemp = m_config.ambientTemperature + cell.temperature;
                if (totalTemp < m_config.ignitionTemperature) continue;

                // Burn fuel
                float burned = std::min(cell.fuel, m_config.burnRate * dt);
                cell.fuel -= burned;

                // Release heat
                cell.temperature += burned * m_config.burnTemperature;

                // Generate soot/smoke
                cell.density += burned * m_config.sootGeneration;

                // Update reaction progress
                cell.reaction = std::min(1.0f, cell.reaction + burned);
            }
        }
    }
}

// =============================================================================
// Vorticity Confinement
// =============================================================================

void GaseousSystem::ComputeVorticity() {
    // ω = ∇ × v
    for (uint32_t k = 1; k < m_resZ - 1; ++k) {
        for (uint32_t j = 1; j < m_resY - 1; ++j) {
            for (uint32_t i = 1; i < m_resX - 1; ++i) {
                auto& cell = m_cells[CellIndex(i, j, k)];
                if (cell.state == GasCell::State::Solid) continue;

                // Central differences
                float dwdy = (m_cells[CellIndex(i, j + 1, k)].w -
                              m_cells[CellIndex(i, j - 1, k)].w) * 0.5f * m_invCellSize;
                float dvdz = (m_cells[CellIndex(i, j, k + 1)].v -
                              m_cells[CellIndex(i, j, k - 1)].v) * 0.5f * m_invCellSize;

                float dudz = (m_cells[CellIndex(i, j, k + 1)].u -
                              m_cells[CellIndex(i, j, k - 1)].u) * 0.5f * m_invCellSize;
                float dwdx = (m_cells[CellIndex(i + 1, j, k)].w -
                              m_cells[CellIndex(i - 1, j, k)].w) * 0.5f * m_invCellSize;

                float dvdx = (m_cells[CellIndex(i + 1, j, k)].v -
                              m_cells[CellIndex(i - 1, j, k)].v) * 0.5f * m_invCellSize;
                float dudy = (m_cells[CellIndex(i, j + 1, k)].u -
                              m_cells[CellIndex(i, j - 1, k)].u) * 0.5f * m_invCellSize;

                cell.vorticityX = dwdy - dvdz;
                cell.vorticityY = dudz - dwdx;
                cell.vorticityZ = dvdx - dudy;
            }
        }
    }
}

void GaseousSystem::ApplyVorticityConfinement(float dt) {
    if (m_config.vorticityStrength <= 0.0f) return;

    float epsilon = m_config.vorticityStrength;

    for (uint32_t k = 2; k < m_resZ - 2; ++k) {
        for (uint32_t j = 2; j < m_resY - 2; ++j) {
            for (uint32_t i = 2; i < m_resX - 2; ++i) {
                auto& cell = m_cells[CellIndex(i, j, k)];
                if (cell.state == GasCell::State::Solid) continue;

                // |ω| at neighbors (central difference of vorticity magnitude)
                auto vorMag = [this](uint32_t ci, uint32_t cj, uint32_t ck) {
                    const auto& c = m_cells[CellIndex(ci, cj, ck)];
                    return std::sqrt(c.vorticityX * c.vorticityX +
                                    c.vorticityY * c.vorticityY +
                                    c.vorticityZ * c.vorticityZ);
                };

                // η = ∇|ω|
                float etaX = (vorMag(i + 1, j, k) - vorMag(i - 1, j, k)) * 0.5f;
                float etaY = (vorMag(i, j + 1, k) - vorMag(i, j - 1, k)) * 0.5f;
                float etaZ = (vorMag(i, j, k + 1) - vorMag(i, j, k - 1)) * 0.5f;

                float etaLen = std::sqrt(etaX * etaX + etaY * etaY + etaZ * etaZ);
                if (etaLen < 1e-10f) continue;

                // N = η / |η|
                float invLen = 1.0f / etaLen;
                float Nx = etaX * invLen;
                float Ny = etaY * invLen;
                float Nz = etaZ * invLen;

                // f_conf = ε * h * (N × ω)
                float scale = epsilon * m_cellSize;
                float fx = scale * (Ny * cell.vorticityZ - Nz * cell.vorticityY);
                float fy = scale * (Nz * cell.vorticityX - Nx * cell.vorticityZ);
                float fz = scale * (Nx * cell.vorticityY - Ny * cell.vorticityX);

                cell.u += fx * dt;
                cell.v += fy * dt;
                cell.w += fz * dt;
            }
        }
    }
}

// =============================================================================
// Pressure Projection
// =============================================================================

void GaseousSystem::ComputeDivergence() {
    float scale = -m_invCellSize;

    for (uint32_t k = 1; k < m_resZ - 1; ++k) {
        for (uint32_t j = 1; j < m_resY - 1; ++j) {
            for (uint32_t i = 1; i < m_resX - 1; ++i) {
                auto& cell = m_cells[CellIndex(i, j, k)];
                if (cell.state == GasCell::State::Solid) {
                    cell.divergence = 0.0f;
                    continue;
                }

                // ∇·v at cell center
                float du = m_cells[CellIndex(i + 1, j, k)].u - cell.u;
                float dv = m_cells[CellIndex(i, j + 1, k)].v - cell.v;
                float dw = m_cells[CellIndex(i, j, k + 1)].w - cell.w;

                cell.divergence = scale * (du + dv + dw);
            }
        }
    }
}

void GaseousSystem::PressureSolve() {
    // Jacobi iteration
    float invH2 = m_invCellSize * m_invCellSize;
    (void)invH2;

    for (uint32_t iter = 0; iter < m_config.pressureIterations; ++iter) {
        for (uint32_t k = 1; k < m_resZ - 1; ++k) {
            for (uint32_t j = 1; j < m_resY - 1; ++j) {
                for (uint32_t i = 1; i < m_resX - 1; ++i) {
                    auto& cell = m_cells[CellIndex(i, j, k)];
                    if (cell.state == GasCell::State::Solid) continue;

                    float pL = m_cells[CellIndex(i - 1, j, k)].pressure;
                    float pR = m_cells[CellIndex(i + 1, j, k)].pressure;
                    float pD = m_cells[CellIndex(i, j - 1, k)].pressure;
                    float pU = m_cells[CellIndex(i, j + 1, k)].pressure;
                    float pB = m_cells[CellIndex(i, j, k - 1)].pressure;
                    float pF = m_cells[CellIndex(i, j, k + 1)].pressure;

                    // Count valid neighbors (non-solid)
                    float neighbors = 6.0f;
                    float sum = pL + pR + pD + pU + pB + pF;

                    if (m_cells[CellIndex(i - 1, j, k)].state == GasCell::State::Solid) {
                        neighbors -= 1.0f; sum -= pL;
                    }
                    if (m_cells[CellIndex(i + 1, j, k)].state == GasCell::State::Solid) {
                        neighbors -= 1.0f; sum -= pR;
                    }
                    if (m_cells[CellIndex(i, j - 1, k)].state == GasCell::State::Solid) {
                        neighbors -= 1.0f; sum -= pD;
                    }
                    if (m_cells[CellIndex(i, j + 1, k)].state == GasCell::State::Solid) {
                        neighbors -= 1.0f; sum -= pU;
                    }
                    if (m_cells[CellIndex(i, j, k - 1)].state == GasCell::State::Solid) {
                        neighbors -= 1.0f; sum -= pB;
                    }
                    if (m_cells[CellIndex(i, j, k + 1)].state == GasCell::State::Solid) {
                        neighbors -= 1.0f; sum -= pF;
                    }

                    if (neighbors > 0.0f) {
                        cell.pressure = (sum - cell.divergence) / neighbors;
                    }
                }
            }
        }
    }
}

void GaseousSystem::ApplyPressureGradient() {
    float scale = m_invCellSize;

    for (uint32_t k = 1; k < m_resZ - 1; ++k) {
        for (uint32_t j = 1; j < m_resY - 1; ++j) {
            for (uint32_t i = 1; i < m_resX - 1; ++i) {
                auto& cell = m_cells[CellIndex(i, j, k)];
                if (cell.state == GasCell::State::Solid) continue;

                float pC = cell.pressure;

                // Subtract pressure gradient from velocity
                if (i > 0) {
                    cell.u -= scale * (pC - m_cells[CellIndex(i - 1, j, k)].pressure);
                }
                if (j > 0) {
                    cell.v -= scale * (pC - m_cells[CellIndex(i, j - 1, k)].pressure);
                }
                if (k > 0) {
                    cell.w -= scale * (pC - m_cells[CellIndex(i, j, k - 1)].pressure);
                }
            }
        }
    }
}

// =============================================================================
// Semi-Lagrangian Advection
// =============================================================================

void GaseousSystem::AdvectFields(float dt) {
    uint32_t total = m_resX * m_resY * m_resZ;

    // Copy current fields to temp buffers
    for (uint32_t idx = 0; idx < total; ++idx) {
        m_densityTemp[idx] = m_cells[idx].density;
        m_temperatureTemp[idx] = m_cells[idx].temperature;
        m_fuelTemp[idx] = m_cells[idx].fuel;
        m_reactionTemp[idx] = m_cells[idx].reaction;
        m_uTemp[idx] = m_cells[idx].u;
        m_vTemp[idx] = m_cells[idx].v;
        m_wTemp[idx] = m_cells[idx].w;
    }

    for (uint32_t k = 1; k < m_resZ - 1; ++k) {
        for (uint32_t j = 1; j < m_resY - 1; ++j) {
            for (uint32_t i = 1; i < m_resX - 1; ++i) {
                auto& cell = m_cells[CellIndex(i, j, k)];
                if (cell.state == GasCell::State::Solid) continue;

                // Trace back (semi-Lagrangian)
                float gx = static_cast<float>(i) + 0.5f;
                float gy = static_cast<float>(j) + 0.5f;
                float gz = static_cast<float>(k) + 0.5f;

                // Use current velocity to trace back in time
                float backX = gx - cell.u * m_invCellSize * dt;
                float backY = gy - cell.v * m_invCellSize * dt;
                float backZ = gz - cell.w * m_invCellSize * dt;

                // Clamp to grid interior
                backX = std::max(0.5f, std::min(backX, static_cast<float>(m_resX) - 0.5f));
                backY = std::max(0.5f, std::min(backY, static_cast<float>(m_resY) - 0.5f));
                backZ = std::max(0.5f, std::min(backZ, static_cast<float>(m_resZ) - 0.5f));

                // Interpolate fields at backtraced position
                cell.density = InterpolateCellField(m_densityTemp, backX, backY, backZ);
                cell.temperature = InterpolateCellField(m_temperatureTemp, backX, backY, backZ);
                cell.fuel = InterpolateCellField(m_fuelTemp, backX, backY, backZ);
                cell.reaction = InterpolateCellField(m_reactionTemp, backX, backY, backZ);
                cell.u = InterpolateCellField(m_uTemp, backX, backY, backZ);
                cell.v = InterpolateCellField(m_vTemp, backX, backY, backZ);
                cell.w = InterpolateCellField(m_wTemp, backX, backY, backZ);
            }
        }
    }
}

// =============================================================================
// Dissipation
// =============================================================================

void GaseousSystem::ApplyDissipation(float dt) {
    float densityDecay = std::pow(m_config.densityDissipation, dt);
    float tempDecay = std::pow(m_config.temperatureDissipation, dt);
    float velDecay = std::pow(m_config.velocityDissipation, dt);
    float fuelDecay = std::pow(m_config.fuelDissipation, dt);

    for (auto& cell : m_cells) {
        if (cell.state == GasCell::State::Solid) continue;
        cell.density *= densityDecay;
        cell.temperature *= tempDecay;
        cell.fuel *= fuelDecay;
        cell.u *= velDecay;
        cell.v *= velDecay;
        cell.w *= velDecay;

        // Clamp small values to zero to avoid noise
        if (cell.density < 1e-6f) cell.density = 0.0f;
        if (cell.temperature < 1e-4f) cell.temperature = 0.0f;
        if (cell.fuel < 1e-6f) cell.fuel = 0.0f;
    }
}

// =============================================================================
// Statistics
// =============================================================================

void GaseousSystem::UpdateStats() {
    m_stats.activeCells = 0;
    m_stats.solidCells = 0;
    m_stats.totalDensity = 0.0f;
    m_stats.maxDensity = 0.0f;
    m_stats.maxTemperature = 0.0f;
    m_stats.maxVelocity = 0.0f;
    m_stats.totalFuel = 0.0f;

    for (const auto& cell : m_cells) {
        if (cell.state == GasCell::State::Solid) {
            m_stats.solidCells++;
            continue;
        }

        if (cell.density > 1e-6f || cell.temperature > 1e-4f) {
            m_stats.activeCells++;
        }

        m_stats.totalDensity += cell.density;
        m_stats.maxDensity = std::max(m_stats.maxDensity, cell.density);
        m_stats.maxTemperature = std::max(m_stats.maxTemperature, cell.temperature);
        m_stats.totalFuel += cell.fuel;

        float speed = std::sqrt(cell.u * cell.u + cell.v * cell.v + cell.w * cell.w);
        m_stats.maxVelocity = std::max(m_stats.maxVelocity, speed);
    }
}

} // namespace WulfNet
