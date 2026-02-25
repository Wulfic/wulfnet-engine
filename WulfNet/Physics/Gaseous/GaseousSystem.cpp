// =============================================================================
// WulfNet Engine - Gaseous Simulation System Implementation
// =============================================================================

#include "GaseousSystem.h"
#include <cstring>
#include <chrono>

namespace WulfNet {

// =============================================================================
// Constructor / Destructor
// =============================================================================

GaseousSystem::GaseousSystem() = default;

GaseousSystem::~GaseousSystem() {
    Shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool GaseousSystem::Initialize(const GaseousSystemConfig& config) {
    if (m_initialized) return false;
    if (config.resolutionX == 0 || config.resolutionY == 0 || config.resolutionZ == 0)
        return false;
    if (config.cellSize <= 0.0f)
        return false;

    m_config = config;
    m_resX = config.resolutionX;
    m_resY = config.resolutionY;
    m_resZ = config.resolutionZ;
    m_cellSize = config.cellSize;
    m_invCellSize = 1.0f / config.cellSize;

    uint32_t totalCells = m_resX * m_resY * m_resZ;

    // Allocate primary grid
    m_cells.resize(totalCells);
    for (auto& c : m_cells) c.Reset();

    // Allocate scratch buffers for semi-Lagrangian advection
    m_densityTemp.resize(totalCells, 0.0f);
    m_temperatureTemp.resize(totalCells, 0.0f);
    m_fuelTemp.resize(totalCells, 0.0f);
    m_reactionTemp.resize(totalCells, 0.0f);
    m_uTemp.resize(totalCells, 0.0f);
    m_vTemp.resize(totalCells, 0.0f);
    m_wTemp.resize(totalCells, 0.0f);

    m_stats = GaseousStats{};
    m_initialized = true;
    return true;
}

void GaseousSystem::Shutdown() {
    m_cells.clear();
    m_densityTemp.clear();
    m_temperatureTemp.clear();
    m_fuelTemp.clear();
    m_reactionTemp.clear();
    m_uTemp.clear();
    m_vTemp.clear();
    m_wTemp.clear();
    m_emitters.clear();
    m_obstacles.clear();
    m_stats = GaseousStats{};
    m_initialized = false;
    m_gpuEnabled = false;
    m_resX = m_resY = m_resZ = 0;
}

bool GaseousSystem::InitializeGPU(VulkanContext* context) {
    if (!m_initialized || !context) return false;
    m_vulkanContext = context;
    m_gpuEnabled = true;
    return true;
}

// =============================================================================
// Emitters
// =============================================================================

uint32_t GaseousSystem::AddEmitter(const GasEmitter& emitter) {
    uint32_t id = static_cast<uint32_t>(m_emitters.size());
    m_emitters.push_back(emitter);
    return id;
}

GasEmitter* GaseousSystem::GetEmitter(uint32_t id) {
    if (id < m_emitters.size()) return &m_emitters[id];
    return nullptr;
}

void GaseousSystem::RemoveEmitter(uint32_t id) {
    if (id < m_emitters.size()) m_emitters[id].enabled = false;
}

// =============================================================================
// Obstacles
// =============================================================================

uint32_t GaseousSystem::AddObstacle(const GasObstacle& obstacle) {
    uint32_t id = static_cast<uint32_t>(m_obstacles.size());
    m_obstacles.push_back(obstacle);
    return id;
}

GasObstacle* GaseousSystem::GetObstacle(uint32_t id) {
    if (id < m_obstacles.size()) return &m_obstacles[id];
    return nullptr;
}

void GaseousSystem::RemoveObstacle(uint32_t id) {
    if (id < m_obstacles.size()) m_obstacles[id].enabled = false;
}

// =============================================================================
// Index Helpers
// =============================================================================

uint32_t GaseousSystem::CellIndex(uint32_t i, uint32_t j, uint32_t k) const {
    return i + j * m_resX + k * m_resX * m_resY;
}

bool GaseousSystem::InBounds(int i, int j, int k) const {
    return i >= 0 && i < static_cast<int>(m_resX) &&
           j >= 0 && j < static_cast<int>(m_resY) &&
           k >= 0 && k < static_cast<int>(m_resZ);
}

// =============================================================================
// Coordinate Conversion
// =============================================================================

void GaseousSystem::WorldToGrid(float wx, float wy, float wz,
                                float& gx, float& gy, float& gz) const {
    gx = (wx - m_config.originX) * m_invCellSize;
    gy = (wy - m_config.originY) * m_invCellSize;
    gz = (wz - m_config.originZ) * m_invCellSize;
}

void GaseousSystem::GridToWorld(float gx, float gy, float gz,
                                float& wx, float& wy, float& wz) const {
    wx = gx * m_cellSize + m_config.originX;
    wy = gy * m_cellSize + m_config.originY;
    wz = gz * m_cellSize + m_config.originZ;
}

// =============================================================================
// Cell Access
// =============================================================================

const GasCell& GaseousSystem::GetCell(uint32_t i, uint32_t j, uint32_t k) const {
    return m_cells[CellIndex(i, j, k)];
}

GasCell& GaseousSystem::GetCellMut(uint32_t i, uint32_t j, uint32_t k) {
    return m_cells[CellIndex(i, j, k)];
}

void GaseousSystem::SetDensity(uint32_t i, uint32_t j, uint32_t k, float density) {
    if (InBounds(static_cast<int>(i), static_cast<int>(j), static_cast<int>(k))) {
        m_cells[CellIndex(i, j, k)].density = density;
    }
}

void GaseousSystem::SetTemperature(uint32_t i, uint32_t j, uint32_t k, float temp) {
    if (InBounds(static_cast<int>(i), static_cast<int>(j), static_cast<int>(k))) {
        m_cells[CellIndex(i, j, k)].temperature = temp;
    }
}

void GaseousSystem::SetFuel(uint32_t i, uint32_t j, uint32_t k, float fuel) {
    if (InBounds(static_cast<int>(i), static_cast<int>(j), static_cast<int>(k))) {
        m_cells[CellIndex(i, j, k)].fuel = fuel;
    }
}

// =============================================================================
// Trilinear Interpolation (cell-centered field)
// =============================================================================

float GaseousSystem::InterpolateCellField(const std::vector<float>& field,
                                          float gx, float gy, float gz) const {
    // Cell-centered values live at (i+0.5, j+0.5, k+0.5)
    // Shift to cell-center coordinates
    float cx = gx - 0.5f;
    float cy = gy - 0.5f;
    float cz = gz - 0.5f;

    int i0 = static_cast<int>(std::floor(cx));
    int j0 = static_cast<int>(std::floor(cy));
    int k0 = static_cast<int>(std::floor(cz));

    float fx = cx - static_cast<float>(i0);
    float fy = cy - static_cast<float>(j0);
    float fz = cz - static_cast<float>(k0);

    auto sample = [&](int i, int j, int k) -> float {
        i = std::max(0, std::min(i, static_cast<int>(m_resX) - 1));
        j = std::max(0, std::min(j, static_cast<int>(m_resY) - 1));
        k = std::max(0, std::min(k, static_cast<int>(m_resZ) - 1));
        return field[static_cast<uint32_t>(i) + static_cast<uint32_t>(j) * m_resX +
                     static_cast<uint32_t>(k) * m_resX * m_resY];
    };

    // Trilinear
    float c000 = sample(i0, j0, k0);
    float c100 = sample(i0 + 1, j0, k0);
    float c010 = sample(i0, j0 + 1, k0);
    float c110 = sample(i0 + 1, j0 + 1, k0);
    float c001 = sample(i0, j0, k0 + 1);
    float c101 = sample(i0 + 1, j0, k0 + 1);
    float c011 = sample(i0, j0 + 1, k0 + 1);
    float c111 = sample(i0 + 1, j0 + 1, k0 + 1);

    float c00 = c000 * (1.0f - fx) + c100 * fx;
    float c10 = c010 * (1.0f - fx) + c110 * fx;
    float c01 = c001 * (1.0f - fx) + c101 * fx;
    float c11 = c011 * (1.0f - fx) + c111 * fx;

    float c0 = c00 * (1.0f - fy) + c10 * fy;
    float c1 = c01 * (1.0f - fy) + c11 * fy;

    return c0 * (1.0f - fz) + c1 * fz;
}

// =============================================================================
// Sampling (world space)
// =============================================================================

float GaseousSystem::SampleDensity(float wx, float wy, float wz) const {
    if (!m_initialized) return 0.0f;

    // Build a flat density array for interpolation
    float gx, gy, gz;
    WorldToGrid(wx, wy, wz, gx, gy, gz);

    // Direct cell lookup (nearest cell for simple sampling)
    int ci = static_cast<int>(gx);
    int cj = static_cast<int>(gy);
    int ck = static_cast<int>(gz);
    if (!InBounds(ci, cj, ck)) return 0.0f;
    return m_cells[CellIndex(static_cast<uint32_t>(ci),
                             static_cast<uint32_t>(cj),
                             static_cast<uint32_t>(ck))].density;
}

float GaseousSystem::SampleTemperature(float wx, float wy, float wz) const {
    if (!m_initialized) return 0.0f;

    float gx, gy, gz;
    WorldToGrid(wx, wy, wz, gx, gy, gz);

    int ci = static_cast<int>(gx);
    int cj = static_cast<int>(gy);
    int ck = static_cast<int>(gz);
    if (!InBounds(ci, cj, ck)) return 0.0f;
    return m_cells[CellIndex(static_cast<uint32_t>(ci),
                             static_cast<uint32_t>(cj),
                             static_cast<uint32_t>(ck))].temperature;
}

void GaseousSystem::SampleVelocity(float wx, float wy, float wz,
                                   float& vx, float& vy, float& vz) const {
    if (!m_initialized) {
        vx = vy = vz = 0.0f;
        return;
    }

    float gx, gy, gz;
    WorldToGrid(wx, wy, wz, gx, gy, gz);

    int ci = static_cast<int>(gx);
    int cj = static_cast<int>(gy);
    int ck = static_cast<int>(gz);
    if (!InBounds(ci, cj, ck)) {
        vx = vy = vz = 0.0f;
        return;
    }

    const auto& cell = m_cells[CellIndex(static_cast<uint32_t>(ci),
                                         static_cast<uint32_t>(cj),
                                         static_cast<uint32_t>(ck))];
    vx = cell.u;
    vy = cell.v;
    vz = cell.w;
}

// =============================================================================
// Simulation Step
// =============================================================================

void GaseousSystem::Step(float deltaTime) {
    if (!m_initialized) return;

    float dt = std::min(deltaTime, m_config.maxTimestep);

    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t sub = 0; sub < m_config.substeps; ++sub) {
        float subDt = dt / static_cast<float>(m_config.substeps);

        // 1. Mark obstacle cells
        MarkObstacles();

        // 2. Apply emitters (inject density, temperature, fuel)
        ApplyEmitters(subDt);

        // 3. Apply body forces (buoyancy)
        ApplyBuoyancy(subDt);

        // 4. Combustion (fire: fuel → heat + soot)
        ApplyCombustion(subDt);

        // 5. Compute vorticity and apply confinement
        ComputeVorticity();
        ApplyVorticityConfinement(subDt);

        // 6. Pressure projection (divergence-free velocity)
        ComputeDivergence();
        PressureSolve();
        ApplyPressureGradient();

        // 7. Advect all fields (semi-Lagrangian)
        AdvectFields(subDt);

        // 8. Dissipation
        ApplyDissipation(subDt);
    }

    auto end = std::chrono::high_resolution_clock::now();
    m_stats.totalTimeMs = std::chrono::duration<float, std::milli>(end - start).count();

    UpdateStats();
}

void GaseousSystem::Reset() {
    if (!m_initialized) return;
    for (auto& c : m_cells) c.Reset();
    m_stats = GaseousStats{};
}

// =============================================================================
// Apply Emitters
// =============================================================================

void GaseousSystem::ApplyEmitters(float dt) {
    for (auto& emitter : m_emitters) {
        if (!emitter.enabled) continue;

        float gx, gy, gz;
        WorldToGrid(emitter.posX, emitter.posY, emitter.posZ, gx, gy, gz);

        int ci = static_cast<int>(gx);
        int cj = static_cast<int>(gy);
        int ck = static_cast<int>(gz);

        // Determine radius in cells
        float radiusCells = 1.0f;
        if (emitter.type == GasEmitterType::Sphere) {
            radiusCells = emitter.radius * m_invCellSize;
        } else if (emitter.type == GasEmitterType::Box) {
            radiusCells = std::max({emitter.sizeX, emitter.sizeY, emitter.sizeZ}) *
                          0.5f * m_invCellSize;
        }

        int r = static_cast<int>(std::ceil(radiusCells));

        for (int dk = -r; dk <= r; ++dk) {
            for (int dj = -r; dj <= r; ++dj) {
                for (int di = -r; di <= r; ++di) {
                    int gi = ci + di;
                    int gj = cj + dj;
                    int gk = ck + dk;
                    if (!InBounds(gi, gj, gk)) continue;

                    // Distance check for sphere
                    if (emitter.type == GasEmitterType::Sphere) {
                        float dist = std::sqrt(
                            static_cast<float>(di * di + dj * dj + dk * dk));
                        if (dist > radiusCells) continue;
                    }

                    // Box bounds check
                    if (emitter.type == GasEmitterType::Box) {
                        float halfX = emitter.sizeX * 0.5f * m_invCellSize;
                        float halfY = emitter.sizeY * 0.5f * m_invCellSize;
                        float halfZ = emitter.sizeZ * 0.5f * m_invCellSize;
                        if (std::abs(static_cast<float>(di)) > halfX ||
                            std::abs(static_cast<float>(dj)) > halfY ||
                            std::abs(static_cast<float>(dk)) > halfZ)
                            continue;
                    }

                    auto& cell = m_cells[CellIndex(
                        static_cast<uint32_t>(gi),
                        static_cast<uint32_t>(gj),
                        static_cast<uint32_t>(gk))];

                    if (cell.state == GasCell::State::Solid) continue;

                    cell.density += emitter.densityRate * dt;
                    cell.temperature += emitter.temperatureRate * dt;
                    cell.fuel += emitter.fuelRate * dt;

                    cell.u += emitter.velocityX * dt;
                    cell.v += emitter.velocityY * dt;
                    cell.w += emitter.velocityZ * dt;

                    cell.state = GasCell::State::Gas;
                }
            }
        }
    }
}

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
