// =============================================================================
// WulfNet Engine - CO-FLIP (Coadjoint Orbit FLIP) Implementation
// Based on "Fluid Implicit Particles on Coadjoint Orbits" (SIGGRAPH Asia 2024)
// =============================================================================

#include "COFLIPSystem.h"
#include "WulfNet/Compute/Fluids/VulkanFluidCompute.h"
#include "WulfNet/Compute/Vulkan/VulkanContext.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

#ifdef WULFNET_HAS_OPENMP
#include <omp.h>
#endif

namespace WulfNet {

// =============================================================================
// Constructor / Destructor
// =============================================================================

COFLIPSystem::COFLIPSystem() = default;

COFLIPSystem::~COFLIPSystem() {
    Shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool COFLIPSystem::Initialize(const COFLIPConfig& config, VulkanContext* vulkan) {
    if (m_initialized) {
        return false;
    }

    m_config = config;
    m_vulkanContext = vulkan;
    m_gpuEnabled = (vulkan != nullptr) && config.useGPU;

    // Allocate grid
    m_gridTotalCells = config.gridSizeX * config.gridSizeY * config.gridSizeZ;
    m_grid.resize(m_gridTotalCells);
    m_solidCells.resize(m_gridTotalCells, false);

    // Previous velocity storage for FLIP
    m_prevU.resize(m_gridTotalCells, 0.0f);
    m_prevV.resize(m_gridTotalCells, 0.0f);
    m_prevW.resize(m_gridTotalCells, 0.0f);

    // Reserve particles
    m_particles.reserve(config.gridSizeX * config.gridSizeY * config.gridSizeZ * config.particlesPerCell);

    // Mark boundary cells as solid
    for (uint32_t k = 0; k < config.gridSizeZ; ++k) {
        for (uint32_t j = 0; j < config.gridSizeY; ++j) {
            for (uint32_t i = 0; i < config.gridSizeX; ++i) {
                if (i == 0 || i == config.gridSizeX - 1 ||
                    j == 0 || // Only bottom boundary (let top be open)
                    k == 0 || k == config.gridSizeZ - 1) {
                    m_solidCells[GridIndex(i, j, k)] = true;
                    m_grid[GridIndex(i, j, k)].type = 2; // Solid
                }
            }
        }
    }

    // TODO: Initialize GPU resources when available
    if (m_gpuEnabled && vulkan && vulkan->IsValid()) {
        m_gpuCompute = std::make_unique<VulkanFluidCompute>();
        if (m_gpuCompute->Initialize(vulkan, config)) {
            // Upload initial grid state
            m_gpuCompute->UploadGrid(m_grid);
        } else {
            // GPU init failed, fall back to CPU
            m_gpuCompute.reset();
            m_gpuEnabled = false;
        }
    } else {
        m_gpuEnabled = false;
    }

    m_initialized = true;
    return true;
}

void COFLIPSystem::Shutdown() {
    m_particles.clear();
    m_grid.clear();
    m_solidCells.clear();
    m_prevU.clear();
    m_prevV.clear();
    m_prevW.clear();
    m_emitters.clear();

    m_vulkanContext = nullptr;
    m_gpuEnabled = false;
    m_initialized = false;
    m_activeParticles = 0;
}

void COFLIPSystem::Reset() {
    m_particles.clear();
    m_activeParticles = 0;

    // Reset grid
    for (auto& cell : m_grid) {
        cell = COFLIPCell{};
    }

    // Restore solid boundary markers
    for (uint32_t k = 0; k < m_config.gridSizeZ; ++k) {
        for (uint32_t j = 0; j < m_config.gridSizeY; ++j) {
            for (uint32_t i = 0; i < m_config.gridSizeX; ++i) {
                if (m_solidCells[GridIndex(i, j, k)]) {
                    m_grid[GridIndex(i, j, k)].type = 2;
                }
            }
        }
    }
}

// =============================================================================
// Main Simulation Step
// =============================================================================

void COFLIPSystem::Step(float dt) {
    if (!m_initialized || m_activeParticles == 0) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Process emitters
    for (auto& emitter : m_emitters) {
        emitter.accumulator += emitter.rate * dt;
        while (emitter.accumulator >= 1.0f) {
            emitter.accumulator -= 1.0f;
            uint32_t idx = AddParticle(
                emitter.x, emitter.y, emitter.z,
                emitter.dirX * emitter.speed,
                emitter.dirY * emitter.speed,
                emitter.dirZ * emitter.speed
            );
            if (idx != UINT32_MAX) {
                m_particles[idx].wx = 0;
                m_particles[idx].wy = 0;
                m_particles[idx].wz = 0;
            }
        }
    }

    if (m_gpuEnabled) {
        // GPU path
        ParticleToGrid_GPU();
        ApplyExternalForces_GPU(dt);
        PressureSolve_GPU();
        GridToParticle_GPU();
    } else {
        // CPU path
        auto p2gStart = std::chrono::high_resolution_clock::now();
        ParticleToGrid_CPU();
        auto p2gEnd = std::chrono::high_resolution_clock::now();
        m_stats.p2gTimeMs = std::chrono::duration<float, std::milli>(p2gEnd - p2gStart).count();

        // Store previous velocities for FLIP update
        for (uint32_t idx = 0; idx < m_gridTotalCells; ++idx) {
            m_prevU[idx] = m_grid[idx].u;
            m_prevV[idx] = m_grid[idx].v;
            m_prevW[idx] = m_grid[idx].w;
        }

        ApplyExternalForces_CPU(dt);

        auto pressureStart = std::chrono::high_resolution_clock::now();
        ComputeDivergence_CPU();
        PressureSolve_CPU();
        ApplyPressureGradient_CPU();
        auto pressureEnd = std::chrono::high_resolution_clock::now();
        m_stats.pressureTimeMs = std::chrono::duration<float, std::milli>(pressureEnd - pressureStart).count();

        auto g2pStart = std::chrono::high_resolution_clock::now();
        GridToParticle_CPU();
        auto g2pEnd = std::chrono::high_resolution_clock::now();
        m_stats.g2pTimeMs = std::chrono::duration<float, std::milli>(g2pEnd - g2pStart).count();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    m_stats.totalTimeMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    UpdateStats();
}

// =============================================================================
// Particle Management
// =============================================================================

uint32_t COFLIPSystem::AddParticle(float x, float y, float z, float vx, float vy, float vz) {
    if (m_activeParticles >= m_particles.capacity()) {
        m_particles.reserve(m_particles.capacity() * 2 + 1024);
    }

    COFLIPParticle p{};
    p.x = x; p.y = y; p.z = z;
    p.vx = vx; p.vy = vy; p.vz = vz;
    p.wx = 0; p.wy = 0; p.wz = 0;  // Initial vorticity
    p.mass = m_config.restDensity * m_config.cellSize * m_config.cellSize * m_config.cellSize / m_config.particlesPerCell;
    p.volume = m_config.cellSize * m_config.cellSize * m_config.cellSize / m_config.particlesPerCell;
    p.materialId = 0;
    p.flags = 1;  // Active

    if (m_activeParticles < m_particles.size()) {
        m_particles[m_activeParticles] = p;
    } else {
        m_particles.push_back(p);
    }

    return m_activeParticles++;
}

void COFLIPSystem::AddParticleBox(float minX, float minY, float minZ, float maxX, float maxY, float maxZ) {
    float spacing = m_config.cellSize / std::cbrt(static_cast<float>(m_config.particlesPerCell));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> jitter(-spacing * 0.25f, spacing * 0.25f);

    for (float z = minZ + spacing * 0.5f; z < maxZ; z += spacing) {
        for (float y = minY + spacing * 0.5f; y < maxY; y += spacing) {
            for (float x = minX + spacing * 0.5f; x < maxX; x += spacing) {
                AddParticle(x + jitter(gen), y + jitter(gen), z + jitter(gen));
            }
        }
    }
}

void COFLIPSystem::AddParticleSphere(float cx, float cy, float cz, float radius) {
    float spacing = m_config.cellSize / std::cbrt(static_cast<float>(m_config.particlesPerCell));
    float r2 = radius * radius;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> jitter(-spacing * 0.25f, spacing * 0.25f);

    for (float z = cz - radius; z <= cz + radius; z += spacing) {
        for (float y = cy - radius; y <= cy + radius; y += spacing) {
            for (float x = cx - radius; x <= cx + radius; x += spacing) {
                float dx = x - cx, dy = y - cy, dz = z - cz;
                if (dx*dx + dy*dy + dz*dz <= r2) {
                    AddParticle(x + jitter(gen), y + jitter(gen), z + jitter(gen));
                }
            }
        }
    }
}

void COFLIPSystem::AddEmitter(float x, float y, float z, float dirX, float dirY, float dirZ, float rate, float speed) {
    Emitter e{};
    e.x = x; e.y = y; e.z = z;
    float len = std::sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ);
    if (len > 0) {
        e.dirX = dirX / len;
        e.dirY = dirY / len;
        e.dirZ = dirZ / len;
    } else {
        e.dirX = 0; e.dirY = -1; e.dirZ = 0;
    }
    e.rate = rate;
    e.speed = speed;
    e.accumulator = 0;
    m_emitters.push_back(e);
}

void COFLIPSystem::AddSolidBox(float minX, float minY, float minZ, float maxX, float maxY, float maxZ) {
    float cs = m_config.cellSize;

    int iMin = std::max(0, static_cast<int>(minX / cs));
    int iMax = std::min(static_cast<int>(m_config.gridSizeX) - 1, static_cast<int>(maxX / cs));
    int jMin = std::max(0, static_cast<int>(minY / cs));
    int jMax = std::min(static_cast<int>(m_config.gridSizeY) - 1, static_cast<int>(maxY / cs));
    int kMin = std::max(0, static_cast<int>(minZ / cs));
    int kMax = std::min(static_cast<int>(m_config.gridSizeZ) - 1, static_cast<int>(maxZ / cs));

    for (int k = kMin; k <= kMax; ++k) {
        for (int j = jMin; j <= jMax; ++j) {
            for (int i = iMin; i <= iMax; ++i) {
                int idx = GridIndex(i, j, k);
                m_solidCells[idx] = true;
                m_grid[idx].type = 2;
            }
        }
    }
}

void COFLIPSystem::AddSolidSphere(float cx, float cy, float cz, float radius) {
    float cs = m_config.cellSize;
    float r2 = radius * radius;

    int iMin = std::max(0, static_cast<int>((cx - radius) / cs));
    int iMax = std::min(static_cast<int>(m_config.gridSizeX) - 1, static_cast<int>((cx + radius) / cs));
    int jMin = std::max(0, static_cast<int>((cy - radius) / cs));
    int jMax = std::min(static_cast<int>(m_config.gridSizeY) - 1, static_cast<int>((cy + radius) / cs));
    int kMin = std::max(0, static_cast<int>((cz - radius) / cs));
    int kMax = std::min(static_cast<int>(m_config.gridSizeZ) - 1, static_cast<int>((cz + radius) / cs));

    for (int k = kMin; k <= kMax; ++k) {
        for (int j = jMin; j <= jMax; ++j) {
            for (int i = iMin; i <= iMax; ++i) {
                float wx = (i + 0.5f) * cs;
                float wy = (j + 0.5f) * cs;
                float wz = (k + 0.5f) * cs;
                float dx = wx - cx, dy = wy - cy, dz = wz - cz;
                if (dx*dx + dy*dy + dz*dz <= r2) {
                    int idx = GridIndex(i, j, k);
                    m_solidCells[idx] = true;
                    m_grid[idx].type = 2;
                }
            }
        }
    }
}

// =============================================================================
// B-Spline Basis Functions (for high-order interpolation)
// =============================================================================

float COFLIPSystem::BSpline(float x) const {
    // Cubic B-spline (centered at 0, support [-2, 2])
    float ax = std::abs(x);
    if (ax < 1.0f) {
        return 0.5f * ax * ax * ax - ax * ax + 2.0f / 3.0f;
    } else if (ax < 2.0f) {
        float t = 2.0f - ax;
        return t * t * t / 6.0f;
    }
    return 0.0f;
}

float COFLIPSystem::BSplineDerivative(float x) const {
    float ax = std::abs(x);
    float sign = (x >= 0) ? 1.0f : -1.0f;

    if (ax < 1.0f) {
        return sign * (1.5f * ax * ax - 2.0f * ax);
    } else if (ax < 2.0f) {
        float t = 2.0f - ax;
        return -sign * 0.5f * t * t;
    }
    return 0.0f;
}

// =============================================================================
// Grid Helpers
// =============================================================================

int COFLIPSystem::GridIndex(int i, int j, int k) const {
    return i + j * m_config.gridSizeX + k * m_config.gridSizeX * m_config.gridSizeY;
}

void COFLIPSystem::WorldToGrid(float wx, float wy, float wz, float& gx, float& gy, float& gz) const {
    gx = wx / m_config.cellSize;
    gy = wy / m_config.cellSize;
    gz = wz / m_config.cellSize;
}

void COFLIPSystem::GridToWorld(float gx, float gy, float gz, float& wx, float& wy, float& wz) const {
    wx = gx * m_config.cellSize;
    wy = gy * m_config.cellSize;
    wz = gz * m_config.cellSize;
}

bool COFLIPSystem::InBounds(int i, int j, int k) const {
    return i >= 0 && i < static_cast<int>(m_config.gridSizeX) &&
           j >= 0 && j < static_cast<int>(m_config.gridSizeY) &&
           k >= 0 && k < static_cast<int>(m_config.gridSizeZ);
}

// =============================================================================
// Divergence-Free Interpolation (Key CO-FLIP Innovation)
// =============================================================================

void COFLIPSystem::InterpolateDivergenceFree(float x, float y, float z, float& vx, float& vy, float& vz) const {
    // Convert to grid coordinates
    float gx, gy, gz;
    WorldToGrid(x, y, z, gx, gy, gz);

    // MAC grid: u is at (i+0.5, j, k), v is at (i, j+0.5, k), w is at (i, j, k+0.5)
    // Use cubic B-spline interpolation for high-order accuracy

    vx = 0; vy = 0; vz = 0;
    float totalWeightU = 0, totalWeightV = 0, totalWeightW = 0;

    // Interpolate u (at face centers offset by 0.5 in x)
    float ux = gx - 0.5f, uy = gy, uz = gz;
    int i0 = static_cast<int>(std::floor(ux)) - 1;
    int j0 = static_cast<int>(std::floor(uy)) - 1;
    int k0 = static_cast<int>(std::floor(uz)) - 1;

    for (int dk = 0; dk < 4; ++dk) {
        for (int dj = 0; dj < 4; ++dj) {
            for (int di = 0; di < 4; ++di) {
                int i = i0 + di, j = j0 + dj, k = k0 + dk;
                if (InBounds(i, j, k)) {
                    float w = BSpline(ux - i) * BSpline(uy - j) * BSpline(uz - k);
                    vx += w * m_grid[GridIndex(i, j, k)].u;
                    totalWeightU += w;
                }
            }
        }
    }

    // Interpolate v (at face centers offset by 0.5 in y)
    float vxg = gx, vyg = gy - 0.5f, vzg = gz;
    i0 = static_cast<int>(std::floor(vxg)) - 1;
    j0 = static_cast<int>(std::floor(vyg)) - 1;
    k0 = static_cast<int>(std::floor(vzg)) - 1;

    for (int dk = 0; dk < 4; ++dk) {
        for (int dj = 0; dj < 4; ++dj) {
            for (int di = 0; di < 4; ++di) {
                int i = i0 + di, j = j0 + dj, k = k0 + dk;
                if (InBounds(i, j, k)) {
                    float w = BSpline(vxg - i) * BSpline(vyg - j) * BSpline(vzg - k);
                    vy += w * m_grid[GridIndex(i, j, k)].v;
                    totalWeightV += w;
                }
            }
        }
    }

    // Interpolate w (at face centers offset by 0.5 in z)
    float wxg = gx, wyg = gy, wzg = gz - 0.5f;
    i0 = static_cast<int>(std::floor(wxg)) - 1;
    j0 = static_cast<int>(std::floor(wyg)) - 1;
    k0 = static_cast<int>(std::floor(wzg)) - 1;

    for (int dk = 0; dk < 4; ++dk) {
        for (int dj = 0; dj < 4; ++dj) {
            for (int di = 0; di < 4; ++di) {
                int i = i0 + di, j = j0 + dj, k = k0 + dk;
                if (InBounds(i, j, k)) {
                    float w = BSpline(wxg - i) * BSpline(wyg - j) * BSpline(wzg - k);
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
    // Compute velocity gradient tensor using B-spline derivatives
    // grad = [du/dx, du/dy, du/dz; dv/dx, dv/dy, dv/dz; dw/dx, dw/dy, dw/dz]

    float gx, gy, gz;
    WorldToGrid(x, y, z, gx, gy, gz);
    float invDx = 1.0f / m_config.cellSize;

    for (int n = 0; n < 9; ++n) grad[n] = 0;

    // Simplified: use central differences on interpolated velocities
    float eps = m_config.cellSize * 0.5f;
    float vxp, vyp, vzp, vxm, vym, vzm;

    InterpolateDivergenceFree(x + eps, y, z, vxp, vyp, vzp);
    InterpolateDivergenceFree(x - eps, y, z, vxm, vym, vzm);
    grad[0] = (vxp - vxm) / (2 * eps); // du/dx
    grad[3] = (vyp - vym) / (2 * eps); // dv/dx
    grad[6] = (vzp - vzm) / (2 * eps); // dw/dx

    InterpolateDivergenceFree(x, y + eps, z, vxp, vyp, vzp);
    InterpolateDivergenceFree(x, y - eps, z, vxm, vym, vzm);
    grad[1] = (vxp - vxm) / (2 * eps); // du/dy
    grad[4] = (vyp - vym) / (2 * eps); // dv/dy
    grad[7] = (vzp - vzm) / (2 * eps); // dw/dy

    InterpolateDivergenceFree(x, y, z + eps, vxp, vyp, vzp);
    InterpolateDivergenceFree(x, y, z - eps, vxm, vym, vzm);
    grad[2] = (vxp - vxm) / (2 * eps); // du/dz
    grad[5] = (vyp - vym) / (2 * eps); // dv/dz
    grad[8] = (vzp - vzm) / (2 * eps); // dw/dz
}

// =============================================================================
// CPU Simulation Steps
// =============================================================================

void COFLIPSystem::ParticleToGrid_CPU() {
    // Reset grid
    for (auto& cell : m_grid) {
        cell.u = cell.v = cell.w = 0;
        cell.weightU = cell.weightV = cell.weightW = 0;
        cell.pressure = 0;
        cell.divergence = 0;
        if (!m_solidCells[&cell - &m_grid[0]]) {
            cell.type = 0; // Air by default
        }
    }

    // Transfer particle velocities to grid using B-spline weights
    for (uint32_t p = 0; p < m_activeParticles; ++p) {
        const COFLIPParticle& part = m_particles[p];
        if (!(part.flags & 1)) continue;

        float gx, gy, gz;
        WorldToGrid(part.x, part.y, part.z, gx, gy, gz);

        // Mark nearby cells as fluid
        int ci = static_cast<int>(gx);
        int cj = static_cast<int>(gy);
        int ck = static_cast<int>(gz);
        if (InBounds(ci, cj, ck) && !m_solidCells[GridIndex(ci, cj, ck)]) {
            m_grid[GridIndex(ci, cj, ck)].type = 1; // Fluid
        }

        // Transfer to u-faces (staggered)
        float ux = gx - 0.5f, uy = gy, uz = gz;
        int i0 = static_cast<int>(std::floor(ux)) - 1;
        int j0 = static_cast<int>(std::floor(uy)) - 1;
        int k0 = static_cast<int>(std::floor(uz)) - 1;

        for (int dk = 0; dk < 4; ++dk) {
            for (int dj = 0; dj < 4; ++dj) {
                for (int di = 0; di < 4; ++di) {
                    int i = i0 + di, j = j0 + dj, k = k0 + dk;
                    if (InBounds(i, j, k)) {
                        float w = BSpline(ux - i) * BSpline(uy - j) * BSpline(uz - k);
                        int idx = GridIndex(i, j, k);
                        m_grid[idx].u += w * part.mass * part.vx;
                        m_grid[idx].weightU += w * part.mass;
                    }
                }
            }
        }

        // Transfer to v-faces
        float vxg = gx, vyg = gy - 0.5f, vzg = gz;
        i0 = static_cast<int>(std::floor(vxg)) - 1;
        j0 = static_cast<int>(std::floor(vyg)) - 1;
        k0 = static_cast<int>(std::floor(vzg)) - 1;

        for (int dk = 0; dk < 4; ++dk) {
            for (int dj = 0; dj < 4; ++dj) {
                for (int di = 0; di < 4; ++di) {
                    int i = i0 + di, j = j0 + dj, k = k0 + dk;
                    if (InBounds(i, j, k)) {
                        float w = BSpline(vxg - i) * BSpline(vyg - j) * BSpline(vzg - k);
                        int idx = GridIndex(i, j, k);
                        m_grid[idx].v += w * part.mass * part.vy;
                        m_grid[idx].weightV += w * part.mass;
                    }
                }
            }
        }

        // Transfer to w-faces
        float wxg = gx, wyg = gy, wzg = gz - 0.5f;
        i0 = static_cast<int>(std::floor(wxg)) - 1;
        j0 = static_cast<int>(std::floor(wyg)) - 1;
        k0 = static_cast<int>(std::floor(wzg)) - 1;

        for (int dk = 0; dk < 4; ++dk) {
            for (int dj = 0; dj < 4; ++dj) {
                for (int di = 0; di < 4; ++di) {
                    int i = i0 + di, j = j0 + dj, k = k0 + dk;
                    if (InBounds(i, j, k)) {
                        float w = BSpline(wxg - i) * BSpline(wyg - j) * BSpline(wzg - k);
                        int idx = GridIndex(i, j, k);
                        m_grid[idx].w += w * part.mass * part.vz;
                        m_grid[idx].weightW += w * part.mass;
                    }
                }
            }
        }
    }

    // Normalize velocities
    for (auto& cell : m_grid) {
        if (cell.weightU > 1e-8f) cell.u /= cell.weightU;
        if (cell.weightV > 1e-8f) cell.v /= cell.weightV;
        if (cell.weightW > 1e-8f) cell.w /= cell.weightW;
    }
}

void COFLIPSystem::ApplyExternalForces_CPU(float dt) {
    // Apply gravity to v-component (y-velocity)
    for (uint32_t k = 0; k < m_config.gridSizeZ; ++k) {
        for (uint32_t j = 0; j < m_config.gridSizeY; ++j) {
            for (uint32_t i = 0; i < m_config.gridSizeX; ++i) {
                int idx = GridIndex(i, j, k);
                if (m_grid[idx].type == 1) { // Fluid cell
                    m_grid[idx].u += m_config.gravityX * dt;
                    m_grid[idx].v += m_config.gravityY * dt;
                    m_grid[idx].w += m_config.gravityZ * dt;
                }
            }
        }
    }
}

void COFLIPSystem::ComputeDivergence_CPU() {
    float invDx = 1.0f / m_config.cellSize;

    for (uint32_t k = 1; k < m_config.gridSizeZ - 1; ++k) {
        for (uint32_t j = 1; j < m_config.gridSizeY - 1; ++j) {
            for (uint32_t i = 1; i < m_config.gridSizeX - 1; ++i) {
                int idx = GridIndex(i, j, k);
                if (m_grid[idx].type != 1) continue; // Only fluid cells

                // Divergence = du/dx + dv/dy + dw/dz
                float uRight = m_grid[GridIndex(i + 1, j, k)].u;
                float uLeft = m_grid[idx].u;
                float vTop = m_grid[GridIndex(i, j + 1, k)].v;
                float vBottom = m_grid[idx].v;
                float wFront = m_grid[GridIndex(i, j, k + 1)].w;
                float wBack = m_grid[idx].w;

                // Handle solid boundaries (enforce no-slip: velocity = 0)
                if (m_solidCells[GridIndex(i + 1, j, k)]) uRight = 0;
                if (m_solidCells[GridIndex(i - 1, j, k)]) uLeft = 0;
                if (m_solidCells[GridIndex(i, j + 1, k)]) vTop = 0;
                if (m_solidCells[GridIndex(i, j - 1, k)]) vBottom = 0;
                if (m_solidCells[GridIndex(i, j, k + 1)]) wFront = 0;
                if (m_solidCells[GridIndex(i, j, k - 1)]) wBack = 0;

                m_grid[idx].divergence = invDx * ((uRight - uLeft) + (vTop - vBottom) + (wFront - wBack));
            }
        }
    }
}

void COFLIPSystem::PressureSolve_CPU() {
    // Jacobi iteration for pressure Poisson equation
    // Laplacian(p) = divergence / dt

    float dx2 = m_config.cellSize * m_config.cellSize;
    float scale = m_config.dt * m_config.restDensity;

    std::vector<float> pressureNew(m_gridTotalCells, 0.0f);

    for (uint32_t iter = 0; iter < m_config.pressureIterations; ++iter) {
#ifdef WULFNET_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int32_t k = 1; k < static_cast<int32_t>(m_config.gridSizeZ) - 1; ++k) {
            for (int32_t j = 1; j < static_cast<int32_t>(m_config.gridSizeY) - 1; ++j) {
                for (int32_t i = 1; i < static_cast<int32_t>(m_config.gridSizeX) - 1; ++i) {
                    int idx = GridIndex(i, j, k);
                    if (m_grid[idx].type != 1) continue;

                    float pSum = 0;
                    int neighbors = 0;

                    // Check each neighbor (inline for OpenMP compatibility)
                    // Left
                    {
                        int nidx = GridIndex(i - 1, j, k);
                        if (!m_solidCells[nidx]) {
                            if (m_grid[nidx].type == 1) pSum += m_grid[nidx].pressure;
                            neighbors++;
                        }
                    }
                    // Right
                    {
                        int nidx = GridIndex(i + 1, j, k);
                        if (!m_solidCells[nidx]) {
                            if (m_grid[nidx].type == 1) pSum += m_grid[nidx].pressure;
                            neighbors++;
                        }
                    }
                    // Bottom
                    {
                        int nidx = GridIndex(i, j - 1, k);
                        if (!m_solidCells[nidx]) {
                            if (m_grid[nidx].type == 1) pSum += m_grid[nidx].pressure;
                            neighbors++;
                        }
                    }
                    // Top
                    {
                        int nidx = GridIndex(i, j + 1, k);
                        if (!m_solidCells[nidx]) {
                            if (m_grid[nidx].type == 1) pSum += m_grid[nidx].pressure;
                            neighbors++;
                        }
                    }
                    // Back
                    {
                        int nidx = GridIndex(i, j, k - 1);
                        if (!m_solidCells[nidx]) {
                            if (m_grid[nidx].type == 1) pSum += m_grid[nidx].pressure;
                            neighbors++;
                        }
                    }
                    // Front
                    {
                        int nidx = GridIndex(i, j, k + 1);
                        if (!m_solidCells[nidx]) {
                            if (m_grid[nidx].type == 1) pSum += m_grid[nidx].pressure;
                            neighbors++;
                        }
                    }

                    if (neighbors > 0) {
                        pressureNew[idx] = (pSum - dx2 * m_grid[idx].divergence * scale) / neighbors;
                    }
                }
            }
        }

        // Copy back (parallel)
#ifdef WULFNET_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int32_t idx = 0; idx < static_cast<int32_t>(m_gridTotalCells); ++idx) {
            if (m_grid[idx].type == 1) {
                m_grid[idx].pressure = pressureNew[idx];
            }
        }
    }
}

void COFLIPSystem::ApplyPressureGradient_CPU() {
    float invDx = 1.0f / m_config.cellSize;
    float scale = m_config.dt / m_config.restDensity;

    for (uint32_t k = 1; k < m_config.gridSizeZ - 1; ++k) {
        for (uint32_t j = 1; j < m_config.gridSizeY - 1; ++j) {
            for (uint32_t i = 1; i < m_config.gridSizeX - 1; ++i) {
                int idx = GridIndex(i, j, k);

                // Update u (pressure gradient in x)
                if (!m_solidCells[idx] && !m_solidCells[GridIndex(i - 1, j, k)]) {
                    float pLeft = m_grid[GridIndex(i - 1, j, k)].pressure;
                    float pRight = m_grid[idx].pressure;
                    m_grid[idx].u -= scale * invDx * (pRight - pLeft);
                } else if (m_solidCells[idx]) {
                    m_grid[idx].u = 0;
                }

                // Update v (pressure gradient in y)
                if (!m_solidCells[idx] && !m_solidCells[GridIndex(i, j - 1, k)]) {
                    float pBottom = m_grid[GridIndex(i, j - 1, k)].pressure;
                    float pTop = m_grid[idx].pressure;
                    m_grid[idx].v -= scale * invDx * (pTop - pBottom);
                } else if (m_solidCells[idx]) {
                    m_grid[idx].v = 0;
                }

                // Update w (pressure gradient in z)
                if (!m_solidCells[idx] && !m_solidCells[GridIndex(i, j, k - 1)]) {
                    float pBack = m_grid[GridIndex(i, j, k - 1)].pressure;
                    float pFront = m_grid[idx].pressure;
                    m_grid[idx].w -= scale * invDx * (pFront - pBack);
                } else if (m_solidCells[idx]) {
                    m_grid[idx].w = 0;
                }
            }
        }
    }
}

void COFLIPSystem::GridToParticle_CPU() {
    float flipRatio = m_config.flipRatio;
    float picRatio = 1.0f - flipRatio;
    float dt = m_config.dt;

    // Clamp bounds
    float margin = m_config.cellSize * 1.5f;
    float maxX = m_config.gridSizeX * m_config.cellSize - margin;
    float maxY = m_config.gridSizeY * m_config.cellSize - margin;
    float maxZ = m_config.gridSizeZ * m_config.cellSize - margin;

#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int32_t p = 0; p < static_cast<int32_t>(m_activeParticles); ++p) {
        COFLIPParticle& part = m_particles[p];
        if (!(part.flags & 1)) continue;

        // Interpolate new grid velocity (PIC)
        float picVx, picVy, picVz;
        InterpolateDivergenceFree(part.x, part.y, part.z, picVx, picVy, picVz);

        // Interpolate old grid velocity for FLIP delta
        float gx, gy, gz;
        WorldToGrid(part.x, part.y, part.z, gx, gy, gz);

        float oldVx = 0, oldVy = 0, oldVz = 0;
        float totalWeight = 0;

        // Simplified: use trilinear for old velocity lookup
        int i0 = static_cast<int>(std::floor(gx));
        int j0 = static_cast<int>(std::floor(gy));
        int k0 = static_cast<int>(std::floor(gz));
        float fx = gx - i0, fy = gy - j0, fz = gz - k0;

        for (int dk = 0; dk <= 1; ++dk) {
            for (int dj = 0; dj <= 1; ++dj) {
                for (int di = 0; di <= 1; ++di) {
                    int i = i0 + di, j = j0 + dj, k = k0 + dk;
                    if (InBounds(i, j, k)) {
                        float w = ((di == 0) ? (1 - fx) : fx) *
                                  ((dj == 0) ? (1 - fy) : fy) *
                                  ((dk == 0) ? (1 - fz) : fz);
                        int idx = GridIndex(i, j, k);
                        oldVx += w * m_prevU[idx];
                        oldVy += w * m_prevV[idx];
                        oldVz += w * m_prevW[idx];
                        totalWeight += w;
                    }
                }
            }
        }

        if (totalWeight > 0) {
            oldVx /= totalWeight;
            oldVy /= totalWeight;
            oldVz /= totalWeight;
        }

        // FLIP: v_new = v_old + (v_grid_new - v_grid_old)
        float flipVx = part.vx + (picVx - oldVx);
        float flipVy = part.vy + (picVy - oldVy);
        float flipVz = part.vz + (picVz - oldVz);

        // Blend PIC and FLIP
        part.vx = flipRatio * flipVx + picRatio * picVx;
        part.vy = flipRatio * flipVy + picRatio * picVy;
        part.vz = flipRatio * flipVz + picRatio * picVz;

        // Update vorticity (for visualization and conservation tracking)
        float grad[9];
        InterpolateVelocityGradient(part.x, part.y, part.z, grad);
        part.wx = grad[7] - grad[5]; // dw/dy - dv/dz
        part.wy = grad[2] - grad[6]; // du/dz - dw/dx
        part.wz = grad[3] - grad[1]; // dv/dx - du/dy

        // Advect particle (dt is pre-computed above)
        part.x += part.vx * dt;
        part.y += part.vy * dt;
        part.z += part.vz * dt;

        // Clamp to domain (margin, maxX, maxY, maxZ are pre-computed above)
        if (part.x < margin) { part.x = margin; part.vx = std::max(0.0f, part.vx); }
        if (part.x > maxX) { part.x = maxX; part.vx = std::min(0.0f, part.vx); }
        if (part.y < margin) { part.y = margin; part.vy = std::max(0.0f, part.vy); }
        if (part.y > maxY) { part.y = maxY; part.vy = std::min(0.0f, part.vy); }
        if (part.z < margin) { part.z = margin; part.vz = std::max(0.0f, part.vz); }
        if (part.z > maxZ) { part.z = maxZ; part.vz = std::min(0.0f, part.vz); }

        // Handle solid collisions
        float cgx, cgy, cgz;
        WorldToGrid(part.x, part.y, part.z, cgx, cgy, cgz);
        int ci = static_cast<int>(cgx);
        int cj = static_cast<int>(cgy);
        int ck = static_cast<int>(cgz);

        if (InBounds(ci, cj, ck) && m_solidCells[GridIndex(ci, cj, ck)]) {
            // Push out of solid (simple approach)
            part.x -= part.vx * dt;
            part.y -= part.vy * dt;
            part.z -= part.vz * dt;
            part.vx *= -0.5f;
            part.vy *= -0.5f;
            part.vz *= -0.5f;
        }
    }
}

// =============================================================================
// GPU Simulation Steps
// =============================================================================

FluidSimParams COFLIPSystem_BuildParams(const COFLIPConfig& config, uint32_t particleCount) {
    FluidSimParams params{};
    params.gridSizeX = config.gridSizeX;
    params.gridSizeY = config.gridSizeY;
    params.gridSizeZ = config.gridSizeZ;
    params.particleCount = particleCount;
    params.cellSize = config.cellSize;
    params.invCellSize = 1.0f / config.cellSize;
    params.dt = config.dt;
    params.flipRatio = config.flipRatio;
    params.gravityX = config.gravityX;
    params.gravityY = config.gravityY;
    params.gravityZ = config.gravityZ;
    params.restDensity = config.restDensity;
    params.pressureIterations = config.pressureIterations;
    params.sorOmega = 1.7f;  // Typical SOR relaxation factor
    params.surfaceTension = config.surfaceTension;
    params.viscosity = config.viscosity;
    return params;
}

void COFLIPSystem::ParticleToGrid_GPU() {
    if (!m_gpuCompute || !m_gpuCompute->IsInitialized()) {
        ParticleToGrid_CPU();
        return;
    }
    FluidSimParams params = COFLIPSystem_BuildParams(m_config, m_activeParticles);
    m_gpuCompute->DispatchP2G(params);
    m_gpuCompute->DispatchNormalize(params);
}

void COFLIPSystem::ApplyExternalForces_GPU(float dt) {
    if (!m_gpuCompute || !m_gpuCompute->IsInitialized()) {
        ApplyExternalForces_CPU(dt);
        return;
    }
    FluidSimParams params = COFLIPSystem_BuildParams(m_config, m_activeParticles);
    params.dt = dt;
    m_gpuCompute->DispatchForces(params);
}

void COFLIPSystem::PressureSolve_GPU() {
    if (!m_gpuCompute || !m_gpuCompute->IsInitialized()) {
        ComputeDivergence_CPU();
        PressureSolve_CPU();
        ApplyPressureGradient_CPU();
        return;
    }
    FluidSimParams params = COFLIPSystem_BuildParams(m_config, m_activeParticles);
    m_gpuCompute->DispatchDivergence(params);
    m_gpuCompute->DispatchPressure(params, m_config.pressureIterations);
    m_gpuCompute->DispatchGradient(params);
}

void COFLIPSystem::GridToParticle_GPU() {
    if (!m_gpuCompute || !m_gpuCompute->IsInitialized()) {
        GridToParticle_CPU();
        return;
    }
    FluidSimParams params = COFLIPSystem_BuildParams(m_config, m_activeParticles);
    m_gpuCompute->DispatchG2P(params);
}

void COFLIPSystem::SyncParticlesToGPU() {
    if (m_gpuCompute && m_gpuCompute->IsInitialized()) {
        m_gpuCompute->UploadParticles(m_particles, m_activeParticles);
    }
}

void COFLIPSystem::SyncParticlesFromGPU() {
    if (m_gpuCompute && m_gpuCompute->IsInitialized()) {
        m_gpuCompute->DownloadParticles(m_particles, m_activeParticles);
    }
}

// =============================================================================
// Energy/Circulation Computation (Conservation Tracking)
// =============================================================================

float COFLIPSystem::ComputeKineticEnergy() const {
    float energy = 0;
    for (uint32_t p = 0; p < m_activeParticles; ++p) {
        const COFLIPParticle& part = m_particles[p];
        if (part.flags & 1) {
            float v2 = part.vx * part.vx + part.vy * part.vy + part.vz * part.vz;
            energy += 0.5f * part.mass * v2;
        }
    }
    return energy;
}

float COFLIPSystem::ComputePotentialEnergy() const {
    float energy = 0;
    for (uint32_t p = 0; p < m_activeParticles; ++p) {
        const COFLIPParticle& part = m_particles[p];
        if (part.flags & 1) {
            energy += part.mass * (-m_config.gravityY) * part.y; // Assuming gravity is negative Y
        }
    }
    return energy;
}

float COFLIPSystem::ComputeCirculation() const {
    // Total vorticity magnitude
    float circ = 0;
    for (uint32_t p = 0; p < m_activeParticles; ++p) {
        const COFLIPParticle& part = m_particles[p];
        if (part.flags & 1) {
            circ += std::sqrt(part.wx * part.wx + part.wy * part.wy + part.wz * part.wz) * part.volume;
        }
    }
    return circ;
}

void COFLIPSystem::UpdateStats() {
    m_stats.activeParticles = m_activeParticles;
    m_stats.totalEnergy = ComputeKineticEnergy() + ComputePotentialEnergy();
    m_stats.totalCirculation = ComputeCirculation();

    // Count fluid cells and find max velocity
    m_stats.fluidCells = 0;
    m_stats.maxVelocity = 0;

    for (uint32_t p = 0; p < m_activeParticles; ++p) {
        const COFLIPParticle& part = m_particles[p];
        if (part.flags & 1) {
            float v = std::sqrt(part.vx * part.vx + part.vy * part.vy + part.vz * part.vz);
            m_stats.maxVelocity = std::max(m_stats.maxVelocity, v);
        }
    }

    for (const auto& cell : m_grid) {
        if (cell.type == 1) m_stats.fluidCells++;
    }
}

} // namespace WulfNet
