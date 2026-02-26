// =============================================================================
// WulfNet Engine - CO-FLIP (Coadjoint Orbit FLIP) Implementation
// Based on "Fluid Implicit Particles on Coadjoint Orbits" (SIGGRAPH Asia 2024)
// =============================================================================

#include "COFLIPSystem.h"
#include "WulfNet/Compute/Fluids/VulkanFluidCompute.h"
#include "WulfNet/Compute/Vulkan/VulkanContext.h"

// Jolt includes - Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <Jolt/Compute/ComputeSystem.h>

#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

#ifdef WULFNET_HAS_OPENMP
#include <omp.h>
#endif

namespace WulfNet {

// Forward declaration
FluidSimParams COFLIPSystem_BuildParams(const COFLIPConfig& config, uint32_t particleCount);

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
    m_solidCells.resize(m_gridTotalCells, 0);

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
                    m_solidCells[GridIndex(i, j, k)] = 1;
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

bool COFLIPSystem::InitializeFromJolt(const COFLIPConfig& config, ::JPH::ComputeSystem* joltCompute) {
    if (m_initialized) {
        return false;
    }

    m_config = config;
    m_vulkanContext = nullptr;  // Not using WulfNet VulkanContext
    m_gpuEnabled = (joltCompute != nullptr) && config.useGPU;

    // Allocate grid
    m_gridTotalCells = config.gridSizeX * config.gridSizeY * config.gridSizeZ;
    m_grid.resize(m_gridTotalCells);
    m_solidCells.resize(m_gridTotalCells, 0);

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
                    m_solidCells[GridIndex(i, j, k)] = 1;
                    m_grid[GridIndex(i, j, k)].type = 2; // Solid
                }
            }
        }
    }

    // Initialize GPU via Jolt compute system
    if (m_gpuEnabled && joltCompute) {
        m_gpuCompute = std::make_unique<VulkanFluidCompute>();
        if (m_gpuCompute->InitializeFromJolt(joltCompute, config)) {
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
    m_prevSwapU.clear();
    m_prevSwapV.clear();
    m_prevSwapW.clear();
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
    if (!m_initialized) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Process emitters (must happen before the active-particle check so
    // emitters can inject particles into an otherwise empty system)
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

    // Skip simulation if there are no particles to simulate
    if (m_activeParticles == 0) return;

    if (m_gpuEnabled) {
        // GPU path - Use batched dispatch for maximum performance
        // This records all simulation stages into a single command buffer
        // and only waits once at the end, eliminating per-dispatch sync overhead
        auto gpuStart = std::chrono::high_resolution_clock::now();

        if (m_gpuCompute && m_gpuCompute->IsInitialized()) {
            FluidSimParams params = COFLIPSystem_BuildParams(m_config, m_activeParticles);
            params.dt = dt;
            m_gpuCompute->DispatchFullStepBatched(params);
        } else {
            // Fallback to CPU if GPU not available
            ParticleToGrid_CPU();
            {
                if (m_prevSwapU.size() != m_gridTotalCells) {
                    m_prevSwapU.resize(m_gridTotalCells);
                    m_prevSwapV.resize(m_gridTotalCells);
                    m_prevSwapW.resize(m_gridTotalCells);
                }
                for (uint32_t idx = 0; idx < m_gridTotalCells; ++idx) {
                    m_prevSwapU[idx] = m_grid[idx].u;
                    m_prevSwapV[idx] = m_grid[idx].v;
                    m_prevSwapW[idx] = m_grid[idx].w;
                }
                m_prevU.swap(m_prevSwapU);
                m_prevV.swap(m_prevSwapV);
                m_prevW.swap(m_prevSwapW);
            }
            ApplyExternalForces_CPU(dt);
            ComputeDivergence_CPU();
            PressureSolve_CPU();
            ApplyPressureGradient_CPU();
            GridToParticle_CPU();
        }

        auto gpuEnd = std::chrono::high_resolution_clock::now();
        m_stats.totalTimeMs = std::chrono::duration<float, std::milli>(gpuEnd - gpuStart).count();
    } else {
        // CPU path
        auto p2gStart = std::chrono::high_resolution_clock::now();
        ParticleToGrid_CPU();
        auto p2gEnd = std::chrono::high_resolution_clock::now();
        m_stats.p2gTimeMs = std::chrono::duration<float, std::milli>(p2gEnd - p2gStart).count();

        // Store previous velocities for FLIP update — extract into
        // contiguous arrays and swap with O(1) pointer swap.
        {
            // First copy current grid velocities into a temp buffer,
            // then swap temp with prev.  This avoids overwriting prev
            // before we can read it in G2P.
            if (m_prevSwapU.size() != m_gridTotalCells) {
                m_prevSwapU.resize(m_gridTotalCells);
                m_prevSwapV.resize(m_gridTotalCells);
                m_prevSwapW.resize(m_gridTotalCells);
            }
            for (uint32_t idx = 0; idx < m_gridTotalCells; ++idx) {
                m_prevSwapU[idx] = m_grid[idx].u;
                m_prevSwapV[idx] = m_grid[idx].v;
                m_prevSwapW[idx] = m_grid[idx].w;
            }
            m_prevU.swap(m_prevSwapU);
            m_prevV.swap(m_prevSwapV);
            m_prevW.swap(m_prevSwapW);
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

    // Use a fixed-seed fast RNG instead of std::random_device (which is
    // extremely slow on some platforms and was being recreated every call).
    static thread_local std::mt19937 gen(42u);
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

    // Reuse thread-local RNG — avoids expensive std::random_device creation.
    static thread_local std::mt19937 gen(42u);
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
                m_solidCells[idx] = 1;
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
                    m_solidCells[idx] = 1;
                    m_grid[idx].type = 2;
                }
            }
        }
    }
}

// =============================================================================
// B-Spline Basis Functions (for high-order interpolation)
// =============================================================================

// Quadratic B-spline (faster than cubic, 3x3x3=27 vs 4x4x4=64 samples)
// Centered at 0, support [-1.5, 1.5]
inline float QuadraticBSpline(float x) {
    float ax = std::abs(x);
    if (ax < 0.5f) {
        return 0.75f - ax * ax;
    } else if (ax < 1.5f) {
        float t = 1.5f - ax;
        return 0.5f * t * t;
    }
    return 0.0f;
}

inline float COFLIPSystem::BSpline(float x) const {
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

inline float COFLIPSystem::BSplineDerivative(float x) const {
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
    // Cubic B-spline interpolation with factored 1D weights:
    // Precompute 4 weights per dimension (12 BSpline calls total)
    // instead of evaluating BSpline 3× per grid point (192 calls total).

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

// Optimized version using quadratic B-spline (27 vs 64 samples)
void COFLIPSystem::InterpolateDivergenceFreeQuadratic(float x, float y, float z, float& vx, float& vy, float& vz) const {
    // Convert to grid coordinates
    float gx, gy, gz;
    WorldToGrid(x, y, z, gx, gy, gz);

    vx = 0; vy = 0; vz = 0;
    float totalWeightU = 0, totalWeightV = 0, totalWeightW = 0;

    // Interpolate u (at face centers offset by 0.5 in x)
    float ux = gx - 0.5f, uy = gy, uz = gz;
    int i0 = static_cast<int>(std::floor(ux + 0.5f)) - 1;
    int j0 = static_cast<int>(std::floor(uy + 0.5f)) - 1;
    int k0 = static_cast<int>(std::floor(uz + 0.5f)) - 1;

    for (int dk = 0; dk < 3; ++dk) {
        for (int dj = 0; dj < 3; ++dj) {
            for (int di = 0; di < 3; ++di) {
                int i = i0 + di, j = j0 + dj, k = k0 + dk;
                if (InBounds(i, j, k)) {
                    float w = QuadraticBSpline(ux - i) * QuadraticBSpline(uy - j) * QuadraticBSpline(uz - k);
                    vx += w * m_grid[GridIndex(i, j, k)].u;
                    totalWeightU += w;
                }
            }
        }
    }

    // Interpolate v (at face centers offset by 0.5 in y)
    float vxg = gx, vyg = gy - 0.5f, vzg = gz;
    i0 = static_cast<int>(std::floor(vxg + 0.5f)) - 1;
    j0 = static_cast<int>(std::floor(vyg + 0.5f)) - 1;
    k0 = static_cast<int>(std::floor(vzg + 0.5f)) - 1;

    for (int dk = 0; dk < 3; ++dk) {
        for (int dj = 0; dj < 3; ++dj) {
            for (int di = 0; di < 3; ++di) {
                int i = i0 + di, j = j0 + dj, k = k0 + dk;
                if (InBounds(i, j, k)) {
                    float w = QuadraticBSpline(vxg - i) * QuadraticBSpline(vyg - j) * QuadraticBSpline(vzg - k);
                    vy += w * m_grid[GridIndex(i, j, k)].v;
                    totalWeightV += w;
                }
            }
        }
    }

    // Interpolate w (at face centers offset by 0.5 in z)
    float wxg = gx, wyg = gy, wzg = gz - 0.5f;
    i0 = static_cast<int>(std::floor(wxg + 0.5f)) - 1;
    j0 = static_cast<int>(std::floor(wyg + 0.5f)) - 1;
    k0 = static_cast<int>(std::floor(wzg + 0.5f)) - 1;

    for (int dk = 0; dk < 3; ++dk) {
        for (int dj = 0; dj < 3; ++dj) {
            for (int di = 0; di < 3; ++di) {
                int i = i0 + di, j = j0 + dj, k = k0 + dk;
                if (InBounds(i, j, k)) {
                    float w = QuadraticBSpline(wxg - i) * QuadraticBSpline(wyg - j) * QuadraticBSpline(wzg - k);
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
    // This replaces the old 6 × InterpolateDivergenceFree finite-difference
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

// =============================================================================
// CPU Simulation Steps
// =============================================================================

void COFLIPSystem::ParticleToGrid_CPU() {
    // Reset grid using indexed loop (pointer arithmetic was slow)
    for (uint32_t idx = 0; idx < m_gridTotalCells; ++idx) {
        COFLIPCell& cell = m_grid[idx];
        cell.u = cell.v = cell.w = 0;
        cell.weightU = cell.weightV = cell.weightW = 0;
        cell.pressure = 0;
        cell.divergence = 0;
        if (!m_solidCells[idx]) {
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

        // Transfer to u-faces (staggered) — cubic B-spline (4×4×4 = 64 samples)
        // Cubic is needed for P2G to maintain pressure accuracy near solid
        // boundaries (wider 4-cell support vs quadratic's 3-cell support).
        // Factored: precompute 1D weights (12 calls) instead of 3D (192 calls).
        float ux = gx - 0.5f, uy = gy, uz = gz;
        int i0u = static_cast<int>(std::floor(ux)) - 1;
        int j0u = static_cast<int>(std::floor(uy)) - 1;
        int k0u = static_cast<int>(std::floor(uz)) - 1;

        float wxU[4], wyU[4], wzU[4];
        for (int d = 0; d < 4; ++d) {
            wxU[d] = BSpline(ux - (i0u + d));
            wyU[d] = BSpline(uy - (j0u + d));
            wzU[d] = BSpline(uz - (k0u + d));
        }

        for (int dk = 0; dk < 4; ++dk) {
            int k = k0u + dk;
            if (k < 0 || k >= static_cast<int>(m_config.gridSizeZ)) continue;
            float wk = wzU[dk];
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0u + dj;
                if (j < 0 || j >= static_cast<int>(m_config.gridSizeY)) continue;
                float wjk = wyU[dj] * wk;
                for (int di = 0; di < 4; ++di) {
                    int i = i0u + di;
                    if (i < 0 || i >= static_cast<int>(m_config.gridSizeX)) continue;
                    float w = wxU[di] * wjk;
                    int idx = GridIndex(i, j, k);
                    m_grid[idx].u += w * part.mass * part.vx;
                    m_grid[idx].weightU += w * part.mass;
                }
            }
        }

        // Transfer to v-faces — cubic B-spline, factored weights
        float vxg = gx, vyg = gy - 0.5f, vzg = gz;
        int i0v = static_cast<int>(std::floor(vxg)) - 1;
        int j0v = static_cast<int>(std::floor(vyg)) - 1;
        int k0v = static_cast<int>(std::floor(vzg)) - 1;

        float wxV[4], wyV[4], wzV[4];
        for (int d = 0; d < 4; ++d) {
            wxV[d] = BSpline(vxg - (i0v + d));
            wyV[d] = BSpline(vyg - (j0v + d));
            wzV[d] = BSpline(vzg - (k0v + d));
        }

        for (int dk = 0; dk < 4; ++dk) {
            int k = k0v + dk;
            if (k < 0 || k >= static_cast<int>(m_config.gridSizeZ)) continue;
            float wk = wzV[dk];
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0v + dj;
                if (j < 0 || j >= static_cast<int>(m_config.gridSizeY)) continue;
                float wjk = wyV[dj] * wk;
                for (int di = 0; di < 4; ++di) {
                    int i = i0v + di;
                    if (i < 0 || i >= static_cast<int>(m_config.gridSizeX)) continue;
                    float w = wxV[di] * wjk;
                    int idx = GridIndex(i, j, k);
                    m_grid[idx].v += w * part.mass * part.vy;
                    m_grid[idx].weightV += w * part.mass;
                }
            }
        }

        // Transfer to w-faces — cubic B-spline, factored weights
        float wxg2 = gx, wyg2 = gy, wzg2 = gz - 0.5f;
        int i0w = static_cast<int>(std::floor(wxg2)) - 1;
        int j0w = static_cast<int>(std::floor(wyg2)) - 1;
        int k0w = static_cast<int>(std::floor(wzg2)) - 1;

        float wxW[4], wyW[4], wzW[4];
        for (int d = 0; d < 4; ++d) {
            wxW[d] = BSpline(wxg2 - (i0w + d));
            wyW[d] = BSpline(wyg2 - (j0w + d));
            wzW[d] = BSpline(wzg2 - (k0w + d));
        }

        for (int dk = 0; dk < 4; ++dk) {
            int k = k0w + dk;
            if (k < 0 || k >= static_cast<int>(m_config.gridSizeZ)) continue;
            float wk = wzW[dk];
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0w + dj;
                if (j < 0 || j >= static_cast<int>(m_config.gridSizeY)) continue;
                float wjk = wyW[dj] * wk;
                for (int di = 0; di < 4; ++di) {
                    int i = i0w + di;
                    if (i < 0 || i >= static_cast<int>(m_config.gridSizeX)) continue;
                    float w = wxW[di] * wjk;
                    int idx = GridIndex(i, j, k);
                    m_grid[idx].w += w * part.mass * part.vz;
                    m_grid[idx].weightW += w * part.mass;
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
    // Red-Black Gauss-Seidel with SOR — converges ~2.5× faster than Jacobi.
    // This eliminates the need for the temporary pressure buffer (in-place update)
    // and allows us to halve the iteration count for equivalent accuracy.

    float dx2 = m_config.cellSize * m_config.cellSize;
    float scale = m_config.dt * m_config.restDensity;
    float omega = 1.7f;  // SOR relaxation factor (1.0 = plain GS, 1.7 = optimal for Poisson)

    const int32_t NX = static_cast<int32_t>(m_config.gridSizeX);
    const int32_t NY = static_cast<int32_t>(m_config.gridSizeY);
    const int32_t NZ = static_cast<int32_t>(m_config.gridSizeZ);

    for (uint32_t iter = 0; iter < m_config.pressureIterations; ++iter) {
        // Two sub-sweeps per iteration: Red cells then Black cells
        // Red: (i+j+k) % 2 == 0,  Black: (i+j+k) % 2 == 1
        for (int color = 0; color < 2; ++color) {
#ifdef WULFNET_HAS_OPENMP
            #pragma omp parallel for collapse(2) schedule(static)
#endif
            for (int32_t k = 1; k < NZ - 1; ++k) {
                for (int32_t j = 1; j < NY - 1; ++j) {
                    // Choose starting i so (i+j+k)%2 == color
                    int32_t iStart = 1 + ((1 + j + k + color) & 1);
                    for (int32_t i = iStart; i < NX - 1; i += 2) {
                        int idx = GridIndex(i, j, k);
                        if (m_grid[idx].type != 1) continue;

                        float pSum = 0.0f;
                        int neighbors = 0;

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
                            float pNew = (pSum - dx2 * m_grid[idx].divergence * scale) / neighbors;
                            // SOR: p = (1-ω)*p_old + ω*p_new
                            m_grid[idx].pressure = (1.0f - omega) * m_grid[idx].pressure + omega * pNew;
                        }
                    }
                }
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

        // Interpolate new grid velocity (PIC) — factored cubic B-spline
        // (12 BSpline calls instead of 192, via precomputed 1D weights)
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

        // Vorticity tracking removed — it was purely diagnostic and cost
        // ~384 BSpline evaluations per particle per frame.  The wx/wy/wz
        // fields remain zero (their initial value) which is fine for games.

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

        // Handle solid collisions — improved push-out with surface normal
        float cgx, cgy, cgz;
        WorldToGrid(part.x, part.y, part.z, cgx, cgy, cgz);
        int ci = static_cast<int>(cgx);
        int cj = static_cast<int>(cgy);
        int ck = static_cast<int>(cgz);

        if (InBounds(ci, cj, ck) && m_solidCells[GridIndex(ci, cj, ck)]) {
            // Compute solid surface normal by sampling neighbouring cells:
            // gradient of solid occupancy → points away from solid interior
            float nx = 0.0f, ny = 0.0f, nz = 0.0f;
            auto solidVal = [&](int i, int j, int k) -> float {
                if (!InBounds(i, j, k)) return 0.0f;
                return m_solidCells[GridIndex(i, j, k)] ? 1.0f : 0.0f;
            };
            nx = solidVal(ci - 1, cj, ck) - solidVal(ci + 1, cj, ck);
            ny = solidVal(ci, cj - 1, ck) - solidVal(ci, cj + 1, ck);
            nz = solidVal(ci, cj, ck - 1) - solidVal(ci, cj, ck + 1);
            float nLen = std::sqrt(nx * nx + ny * ny + nz * nz);

            if (nLen > 1e-6f) {
                // Normalise the surface normal
                float invLen = 1.0f / nLen;
                nx *= invLen;
                ny *= invLen;
                nz *= invLen;

                // Push particle out along normal (one cell + small margin)
                float pushDist = m_config.cellSize * 1.1f;
                part.x += nx * pushDist;
                part.y += ny * pushDist;
                part.z += nz * pushDist;

                // Project velocity onto surface: remove the normal component
                // and apply a small restitution bounce
                float vDotN = part.vx * nx + part.vy * ny + part.vz * nz;
                if (vDotN < 0.0f) {
                    // Particle was moving INTO the solid
                    float restitution = 0.1f; // Small bounce
                    part.vx -= (1.0f + restitution) * vDotN * nx;
                    part.vy -= (1.0f + restitution) * vDotN * ny;
                    part.vz -= (1.0f + restitution) * vDotN * nz;
                }
            } else {
                // Degenerate case (fully surrounded): revert and damp
                part.x -= part.vx * dt;
                part.y -= part.vy * dt;
                part.z -= part.vz * dt;
                part.vx *= 0.1f;
                part.vy *= 0.1f;
                part.vz *= 0.1f;
            }

            // Verify we escaped — if still in solid, force to last known good position
            WorldToGrid(part.x, part.y, part.z, cgx, cgy, cgz);
            ci = static_cast<int>(cgx);
            cj = static_cast<int>(cgy);
            ck = static_cast<int>(cgz);
            if (InBounds(ci, cj, ck) && m_solidCells[GridIndex(ci, cj, ck)]) {
                part.x -= part.vx * dt;
                part.y -= part.vy * dt;
                part.z -= part.vz * dt;
                part.vx = 0.0f;
                part.vy = 0.0f;
                part.vz = 0.0f;
            }
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
            energy += part.mass * (-m_config.gravityY) * part.y;
        }
    }
    return energy;
}

float COFLIPSystem::ComputeCirculation() const {
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

    // Fused single-pass over particles: compute KE, PE, and max velocity.
    // Circulation tracking removed (vorticity no longer computed for perf).
    float sumKE = 0.0f, sumPE = 0.0f;
    float maxV2 = 0.0f;
    const float negGravY = -m_config.gravityY;

    for (uint32_t p = 0; p < m_activeParticles; ++p) {
        const COFLIPParticle& part = m_particles[p];
        if (!(part.flags & 1)) continue;

        float v2 = part.vx * part.vx + part.vy * part.vy + part.vz * part.vz;
        sumKE += 0.5f * part.mass * v2;
        sumPE += part.mass * negGravY * part.y;
        if (v2 > maxV2) maxV2 = v2;
    }

    m_stats.totalEnergy = sumKE + sumPE;
    m_stats.totalCirculation = 0.0f;  // Vorticity tracking disabled for performance
    m_stats.maxVelocity = std::sqrt(maxV2);

    // Count fluid cells
    uint32_t fluidCells = 0;
    for (uint32_t idx = 0; idx < m_gridTotalCells; ++idx) {
        if (m_grid[idx].type == 1) ++fluidCells;
    }
    m_stats.fluidCells = fluidCells;
}

} // namespace WulfNet
