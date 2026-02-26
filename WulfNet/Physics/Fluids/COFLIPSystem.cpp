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
#include <cfloat>
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
    m_cellCount.clear();
    m_cellStart.clear();
    m_sortedParticles.clear();

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
#ifdef WULFNET_HAS_OPENMP
                #pragma omp parallel for schedule(static)
#endif
                for (int32_t idx = 0; idx < static_cast<int32_t>(m_gridTotalCells); ++idx) {
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
#ifdef WULFNET_HAS_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int32_t idx = 0; idx < static_cast<int32_t>(m_gridTotalCells); ++idx) {
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
// B-Spline, Grid Helpers, and Interpolation → COFLIPSystemInterp.cpp
// CPU Solvers (P2G, Forces, Divergence, Pressure, G2P) → COFLIPSystemCPU.cpp
// =============================================================================


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
#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:energy)
#endif
    for (int32_t p = 0; p < static_cast<int32_t>(m_activeParticles); ++p) {
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
#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:energy)
#endif
    for (int32_t p = 0; p < static_cast<int32_t>(m_activeParticles); ++p) {
        const COFLIPParticle& part = m_particles[p];
        if (part.flags & 1) {
            energy += part.mass * (-m_config.gravityY) * part.y;
        }
    }
    return energy;
}

float COFLIPSystem::ComputeCirculation() const {
    float circ = 0;
#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:circ)
#endif
    for (int32_t p = 0; p < static_cast<int32_t>(m_activeParticles); ++p) {
        const COFLIPParticle& part = m_particles[p];
        if (part.flags & 1) {
            circ += std::sqrt(part.wx * part.wx + part.wy * part.wy + part.wz * part.wz) * part.volume;
        }
    }
    return circ;
}

void COFLIPSystem::UpdateStats() {
    m_stats.activeParticles = m_activeParticles;

    // Fused single-pass over particles: compute KE, PE, max velocity, and Y extents.
    // Parallelized with OpenMP reductions for near-linear speedup.
    float sumKE = 0.0f, sumPE = 0.0f;
    float maxV2 = 0.0f;
    float pMinY = FLT_MAX, pMaxY = -FLT_MAX;
    const float negGravY = -m_config.gravityY;

#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:sumKE,sumPE) reduction(max:maxV2,pMaxY) reduction(min:pMinY)
#endif
    for (int32_t p = 0; p < static_cast<int32_t>(m_activeParticles); ++p) {
        const COFLIPParticle& part = m_particles[p];
        if (!(part.flags & 1)) continue;

        float v2 = part.vx * part.vx + part.vy * part.vy + part.vz * part.vz;
        sumKE += 0.5f * part.mass * v2;
        sumPE += part.mass * negGravY * part.y;
        if (v2 > maxV2) maxV2 = v2;
        if (part.y < pMinY) pMinY = part.y;
        if (part.y > pMaxY) pMaxY = part.y;
    }

    m_stats.totalEnergy = sumKE + sumPE;
    m_stats.totalCirculation = 0.0f;  // Vorticity tracking disabled for performance
    m_stats.maxVelocity = std::sqrt(maxV2);
    m_stats.minParticleY = (pMinY <= pMaxY) ? pMinY : 0.0f;
    m_stats.maxParticleY = (pMinY <= pMaxY) ? pMaxY : 0.0f;

    // Count fluid cells — parallel reduction
    uint32_t fluidCells = 0;
#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:fluidCells)
#endif
    for (int32_t idx = 0; idx < static_cast<int32_t>(m_gridTotalCells); ++idx) {
        if (m_grid[idx].type == 1) ++fluidCells;
    }
    m_stats.fluidCells = fluidCells;
}

} // namespace WulfNet
