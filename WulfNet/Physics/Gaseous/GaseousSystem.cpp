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
// Solver functions (MarkObstacles, Buoyancy, Combustion, Vorticity, Divergence,
// PressureSolve, PressureGradient, Advection, Dissipation, Stats)
//     --> GaseousSystemSolve.cpp
// =============================================================================

} // namespace WulfNet
