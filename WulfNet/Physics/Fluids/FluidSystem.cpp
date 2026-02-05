// =============================================================================
// WulfNet Engine - MPM Fluid System Implementation
// =============================================================================

#include "FluidSystem.h"
#include "../../Compute/Vulkan/VulkanContext.h"
#include "../../Compute/Memory/ComputeBuffer.h"
#include "../../Compute/Shaders/ComputePipeline.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>

namespace WulfNet {

// =============================================================================
// Constructor / Destructor
// =============================================================================

FluidSystem::FluidSystem() = default;

FluidSystem::~FluidSystem() {
    Shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool FluidSystem::Initialize(const FluidSystemConfig& config) {
    if (m_initialized) {
        Shutdown();
    }

    m_config = config;

    // Allocate particles
    m_particles.resize(config.maxParticles);
    m_activeParticles = 0;

    // Initialize particles to inactive
    for (auto& p : m_particles) {
        p.flags = 0;  // No Active flag
    }

    // Create grid
    m_grid = std::make_unique<FluidGrid>();
    if (!m_grid->Initialize(config.gridResolutionX,
                            config.gridResolutionY,
                            config.gridResolutionZ,
                            config.cellSize)) {
        return false;
    }

    m_grid->SetBounds(config.boundsMinX, config.boundsMinY, config.boundsMinZ,
                      config.boundsMaxX, config.boundsMaxY, config.boundsMaxZ);

    // Add default water material
    m_materials.push_back(FluidMaterial::Water());

    m_initialized = true;
    return true;
}

void FluidSystem::Shutdown() {
    // Cleanup GPU resources (commented out - not implemented yet)
    // m_p2gPipeline.reset();
    // m_gridForcePipeline.reset();
    // m_pressurePipeline.reset();
    // m_g2pPipeline.reset();
    // m_particleBuffer.reset();
    // m_gridBuffer.reset();
    m_vulkanContext = nullptr;
    m_gpuEnabled = false;

    // Clear data
    m_particles.clear();
    m_grid.reset();
    m_materials.clear();
    m_emitters.clear();
    m_colliders.clear();
    m_buoyancyObjects.clear();

    m_activeParticles = 0;
    m_initialized = false;
}

bool FluidSystem::InitializeGPU(VulkanContext* context) {
    if (!context || !context->IsValid()) {
        return false;
    }

    m_vulkanContext = context;

    // TODO: Create GPU buffers when GPUBuffer API is finalized
    // For now, GPU mode uses CPU fallback
    // size_t particleBufferSize = m_config.maxParticles * sizeof(FluidParticle);
    // size_t gridBufferSize = m_grid->GetDataSize();

    // TODO: Create compute pipelines for P2G, forces, pressure, G2P
    // For now we'll use CPU fallback

    m_gpuEnabled = true;
    return true;
}

// =============================================================================
// Materials
// =============================================================================

uint32_t FluidSystem::AddMaterial(const FluidMaterial& material) {
    uint32_t id = static_cast<uint32_t>(m_materials.size());
    m_materials.push_back(material);
    return id;
}

FluidMaterial* FluidSystem::GetMaterial(uint32_t id) {
    if (id < m_materials.size()) {
        return &m_materials[id];
    }
    return nullptr;
}

// =============================================================================
// Emitters
// =============================================================================

uint32_t FluidSystem::AddEmitter(const FluidEmitter& emitter) {
    uint32_t id = static_cast<uint32_t>(m_emitters.size());
    m_emitters.push_back(emitter);
    return id;
}

FluidEmitter* FluidSystem::GetEmitter(uint32_t id) {
    if (id < m_emitters.size()) {
        return &m_emitters[id];
    }
    return nullptr;
}

void FluidSystem::RemoveEmitter(uint32_t id) {
    if (id < m_emitters.size()) {
        m_emitters[id].enabled = false;
    }
}

// =============================================================================
// Colliders
// =============================================================================

uint32_t FluidSystem::AddCollider(const FluidCollider& collider) {
    uint32_t id = static_cast<uint32_t>(m_colliders.size());
    m_colliders.push_back(collider);
    return id;
}

FluidCollider* FluidSystem::GetCollider(uint32_t id) {
    if (id < m_colliders.size()) {
        return &m_colliders[id];
    }
    return nullptr;
}

void FluidSystem::RemoveCollider(uint32_t id) {
    if (id < m_colliders.size()) {
        m_colliders[id].enabled = false;
    }
}

// =============================================================================
// Buoyancy Objects
// =============================================================================

uint32_t FluidSystem::AddBuoyancyObject(const BuoyancyObject& obj) {
    uint32_t id = static_cast<uint32_t>(m_buoyancyObjects.size());
    m_buoyancyObjects.push_back(obj);
    return id;
}

BuoyancyObject* FluidSystem::GetBuoyancyObject(uint32_t id) {
    if (id < m_buoyancyObjects.size()) {
        return &m_buoyancyObjects[id];
    }
    return nullptr;
}

void FluidSystem::RemoveBuoyancyObject(uint32_t id) {
    if (id < m_buoyancyObjects.size()) {
        m_buoyancyObjects.erase(m_buoyancyObjects.begin() + id);
    }
}

// =============================================================================
// Particle Management
// =============================================================================

void FluidSystem::AddParticle(float x, float y, float z, uint32_t materialId) {
    if (m_activeParticles >= m_config.maxParticles) {
        return;
    }

    FluidParticle& p = m_particles[m_activeParticles++];
    p.x = x; p.y = y; p.z = z;
    p.vx = 0; p.vy = 0; p.vz = 0;
    p.mass = 1.0f;
    p.density = m_materials[materialId].density;
    p.materialId = materialId;
    p.flags = static_cast<uint32_t>(ParticleFlags::Active);
    p.temperature = 293.15f;  // 20°C
    p.pressure = 0.0f;

    // Clear affine momentum (simplified - only diagonal)
    p.C00 = p.C11 = 0.0f;
}

void FluidSystem::AddParticleBox(float minX, float minY, float minZ,
                                  float maxX, float maxY, float maxZ,
                                  uint32_t materialId) {
    float spacing = m_config.cellSize / 2.0f;  // 2 particles per cell side

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> jitter(-spacing * 0.25f, spacing * 0.25f);

    for (float z = minZ + spacing * 0.5f; z < maxZ; z += spacing) {
        for (float y = minY + spacing * 0.5f; y < maxY; y += spacing) {
            for (float x = minX + spacing * 0.5f; x < maxX; x += spacing) {
                if (m_activeParticles >= m_config.maxParticles) return;

                // Add slight jitter to prevent grid artifacts
                AddParticle(x + jitter(gen),
                           y + jitter(gen),
                           z + jitter(gen),
                           materialId);
            }
        }
    }
}

void FluidSystem::AddParticleSphere(float cx, float cy, float cz, float radius,
                                     uint32_t materialId) {
    float spacing = m_config.cellSize / 2.0f;
    float r2 = radius * radius;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> jitter(-spacing * 0.25f, spacing * 0.25f);

    for (float z = cx - radius; z <= cx + radius; z += spacing) {
        for (float y = cy - radius; y <= cy + radius; y += spacing) {
            for (float x = cz - radius; x <= cz + radius; x += spacing) {
                float dx = x - cx, dy = y - cy, dz = z - cz;
                if (dx*dx + dy*dy + dz*dz <= r2) {
                    if (m_activeParticles >= m_config.maxParticles) return;

                    AddParticle(x + jitter(gen),
                               y + jitter(gen),
                               z + jitter(gen),
                               materialId);
                }
            }
        }
    }
}

void FluidSystem::ClearParticles() {
    m_activeParticles = 0;
    for (auto& p : m_particles) {
        p.flags = 0;
    }
}

// =============================================================================
// Simulation
// =============================================================================

void FluidSystem::Step(float deltaTime) {
    if (!m_initialized || m_activeParticles == 0) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Clamp timestep
    deltaTime = std::min(deltaTime, m_config.maxTimestep);

    // Substeps for stability
    float subDt = deltaTime / m_config.substeps;

    for (uint32_t sub = 0; sub < m_config.substeps; ++sub) {
        if (m_gpuEnabled) {
            StepGPU(subDt);
        } else {
            // CPU reference implementation
            EmitParticles(subDt);

            auto p2gStart = std::chrono::high_resolution_clock::now();
            ParticleToGrid();
            auto p2gEnd = std::chrono::high_resolution_clock::now();
            m_stats.p2gTimeMs = std::chrono::duration<float, std::milli>(p2gEnd - p2gStart).count();

            auto gridStart = std::chrono::high_resolution_clock::now();
            GridForces(subDt);
            PressureSolve();
            auto gridEnd = std::chrono::high_resolution_clock::now();
            m_stats.gridSolveTimeMs = std::chrono::duration<float, std::milli>(gridEnd - gridStart).count();

            auto g2pStart = std::chrono::high_resolution_clock::now();
            GridToParticle(subDt);
            auto g2pEnd = std::chrono::high_resolution_clock::now();
            m_stats.g2pTimeMs = std::chrono::duration<float, std::milli>(g2pEnd - g2pStart).count();

            auto collStart = std::chrono::high_resolution_clock::now();
            ParticleCollisions();
            auto collEnd = std::chrono::high_resolution_clock::now();
            m_stats.collisionTimeMs = std::chrono::duration<float, std::milli>(collEnd - collStart).count();
        }
    }

    // Post-step updates
    if (m_config.enableSleeping) {
        UpdateSleeping();
    }
    ComputeBuoyancy();
    UpdateStats();

    auto endTime = std::chrono::high_resolution_clock::now();
    m_stats.totalTimeMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();
}

void FluidSystem::Reset() {
    ClearParticles();
    m_grid->Reset();
    m_stats = FluidStats();

    // Reset emitters
    for (auto& e : m_emitters) {
        e.accumulatedTime = 0.0f;
        e.particlesEmitted = 0;
    }
}

// =============================================================================
// CPU Reference Implementation
// =============================================================================

void FluidSystem::EmitParticles(float deltaTime) {
    std::random_device rd;
    std::mt19937 gen(rd());

    for (auto& emitter : m_emitters) {
        if (!emitter.enabled) continue;

        emitter.accumulatedTime += deltaTime;
        float particleDt = 1.0f / emitter.emissionRate;

        while (emitter.accumulatedTime >= particleDt) {
            emitter.accumulatedTime -= particleDt;

            if (m_activeParticles >= m_config.maxParticles) return;

            float x = emitter.posX, y = emitter.posY, z = emitter.posZ;

            // Add variance based on emitter type
            if (emitter.type == EmitterType::Sphere) {
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                float rx, ry, rz;
                do {
                    rx = dist(gen); ry = dist(gen); rz = dist(gen);
                } while (rx*rx + ry*ry + rz*rz > 1.0f);
                x += rx * emitter.radius;
                y += ry * emitter.radius;
                z += rz * emitter.radius;
            } else if (emitter.type == EmitterType::Box) {
                std::uniform_real_distribution<float> distX(-emitter.sizeX/2, emitter.sizeX/2);
                std::uniform_real_distribution<float> distY(-emitter.sizeY/2, emitter.sizeY/2);
                std::uniform_real_distribution<float> distZ(-emitter.sizeZ/2, emitter.sizeZ/2);
                x += distX(gen);
                y += distY(gen);
                z += distZ(gen);
            }

            AddParticle(x, y, z, emitter.materialId);

            // Set initial velocity
            std::uniform_real_distribution<float> speedVar(1.0f - emitter.speedVariance,
                                                           1.0f + emitter.speedVariance);
            float speed = emitter.initialSpeed * speedVar(gen);

            FluidParticle& p = m_particles[m_activeParticles - 1];
            p.vx = emitter.dirX * speed;
            p.vy = emitter.dirY * speed;
            p.vz = emitter.dirZ * speed;

            emitter.particlesEmitted++;
        }
    }
}

void FluidSystem::ParticleToGrid() {
    // Reset grid
    m_grid->Reset();

    float invDx = m_grid->GetInvCellSize();

    // APIC P2G transfer
    for (uint32_t i = 0; i < m_activeParticles; ++i) {
        FluidParticle& p = m_particles[i];
        if (!HasFlag(p.flags, ParticleFlags::Active)) continue;

        // Get grid position
        float gx, gy, gz;
        m_grid->WorldToGrid(p.x, p.y, p.z, gx, gy, gz);

        int baseI = static_cast<int>(gx - 0.5f);
        int baseJ = static_cast<int>(gy - 0.5f);
        int baseK = static_cast<int>(gz - 0.5f);

        // Quadratic B-spline weights
        auto weight = [](float x) -> float {
            x = std::abs(x);
            if (x < 0.5f) return 0.75f - x * x;
            if (x < 1.5f) return 0.5f * (1.5f - x) * (1.5f - x);
            return 0.0f;
        };

        // Transfer to 3x3x3 neighborhood
        for (int dk = 0; dk < 3; ++dk) {
            for (int dj = 0; dj < 3; ++dj) {
                for (int di = 0; di < 3; ++di) {
                    int ci = baseI + di;
                    int cj = baseJ + dj;
                    int ck = baseK + dk;

                    if (!m_grid->IsInBounds(ci, cj, ck)) continue;

                    float wx = weight(gx - ci - 0.5f);
                    float wy = weight(gy - cj - 0.5f);
                    float wz = weight(gz - ck - 0.5f);
                    float w = wx * wy * wz;

                    if (w <= 0.0f) continue;

                    MACCell& cell = m_grid->GetCell(ci, cj, ck);

                    // Simplified PIC transfer (no full APIC for now)
                    float dx = (ci + 0.5f) - gx;
                    float dy = (cj + 0.5f) - gy;
                    float dz = (ck + 0.5f) - gz;
                    (void)dx; (void)dy; (void)dz;  // Will use for full APIC later

                    // Simple velocity transfer
                    float affineU = p.vx;
                    float affineV = p.vy;
                    float affineW = p.vz;

                    float wm = w * p.mass;

                    cell.u += wm * affineU;
                    cell.v += wm * affineV;
                    cell.w += wm * affineW;
                    cell.uWeight += wm;
                    cell.vWeight += wm;
                    cell.wWeight += wm;
                    cell.particleCount++;
                }
            }
        }
    }

    // Normalize velocities by weight
    for (uint32_t k = 0; k < m_config.gridResolutionZ; ++k) {
        for (uint32_t j = 0; j < m_config.gridResolutionY; ++j) {
            for (uint32_t i = 0; i < m_config.gridResolutionX; ++i) {
                MACCell& cell = m_grid->GetCell(i, j, k);
                if (cell.uWeight > 0.0f) cell.u /= cell.uWeight;
                if (cell.vWeight > 0.0f) cell.v /= cell.vWeight;
                if (cell.wWeight > 0.0f) cell.w /= cell.wWeight;
            }
        }
    }

    m_grid->MarkCellStates();
}

void FluidSystem::GridForces(float deltaTime) {
    // Apply gravity to grid velocities
    for (uint32_t k = 0; k < m_config.gridResolutionZ; ++k) {
        for (uint32_t j = 0; j < m_config.gridResolutionY; ++j) {
            for (uint32_t i = 0; i < m_config.gridResolutionX; ++i) {
                MACCell& cell = m_grid->GetCell(i, j, k);
                if (cell.state == CellState::Fluid) {
                    cell.v += m_config.gravity * deltaTime;
                }
            }
        }
    }

    // Enforce solid boundary conditions
    for (uint32_t k = 0; k < m_config.gridResolutionZ; ++k) {
        for (uint32_t j = 0; j < m_config.gridResolutionY; ++j) {
            // Left/right walls
            m_grid->GetCell(0, j, k).u = 0.0f;
            m_grid->GetCell(m_config.gridResolutionX - 1, j, k).u = 0.0f;
        }
    }

    for (uint32_t k = 0; k < m_config.gridResolutionZ; ++k) {
        for (uint32_t i = 0; i < m_config.gridResolutionX; ++i) {
            // Bottom/top walls
            m_grid->GetCell(i, 0, k).v = 0.0f;
            m_grid->GetCell(i, m_config.gridResolutionY - 1, k).v = 0.0f;
        }
    }

    for (uint32_t j = 0; j < m_config.gridResolutionY; ++j) {
        for (uint32_t i = 0; i < m_config.gridResolutionX; ++i) {
            // Front/back walls
            m_grid->GetCell(i, j, 0).w = 0.0f;
            m_grid->GetCell(i, j, m_config.gridResolutionZ - 1).w = 0.0f;
        }
    }
}

void FluidSystem::PressureSolve() {
    // Compute divergence
    m_grid->ComputeDivergence();

    // Jacobi iteration for pressure solve
    float scale = m_config.cellSize * m_config.cellSize;

    for (uint32_t iter = 0; iter < m_config.pressureIterations; ++iter) {
        for (uint32_t k = 1; k < m_config.gridResolutionZ - 1; ++k) {
            for (uint32_t j = 1; j < m_config.gridResolutionY - 1; ++j) {
                for (uint32_t i = 1; i < m_config.gridResolutionX - 1; ++i) {
                    MACCell& cell = m_grid->GetCell(i, j, k);
                    if (cell.state != CellState::Fluid) continue;

                    // Count non-solid neighbors
                    int numNeighbors = 0;
                    float pSum = 0.0f;

                    auto addNeighbor = [&](int ni, int nj, int nk) {
                        if (!m_grid->IsInBounds(ni, nj, nk)) return;
                        const MACCell& n = m_grid->GetCell(ni, nj, nk);
                        if (n.state != CellState::Solid) {
                            numNeighbors++;
                            pSum += n.pressure;
                        }
                    };

                    addNeighbor(i - 1, j, k);
                    addNeighbor(i + 1, j, k);
                    addNeighbor(i, j - 1, k);
                    addNeighbor(i, j + 1, k);
                    addNeighbor(i, j, k - 1);
                    addNeighbor(i, j, k + 1);

                    if (numNeighbors > 0) {
                        cell.pressure = (pSum - scale * cell.divergence) / numNeighbors;
                    }
                }
            }
        }
    }

    // Apply pressure gradient
    m_grid->ApplyPressureGradient();

    // Extrapolate velocities
    m_grid->ExtrapolateVelocity(3);
}

void FluidSystem::GridToParticle(float deltaTime) {
    float flipRatio = m_config.flipRatio;

    for (uint32_t i = 0; i < m_activeParticles; ++i) {
        FluidParticle& p = m_particles[i];
        if (!HasFlag(p.flags, ParticleFlags::Active)) continue;

        float gx, gy, gz;
        m_grid->WorldToGrid(p.x, p.y, p.z, gx, gy, gz);

        // Interpolate new velocity
        float newVx, newVy, newVz;
        m_grid->InterpolateVelocity(gx, gy, gz, newVx, newVy, newVz);

        // FLIP: v_new = v_old + (v_grid_new - v_grid_old)
        // PIC:  v_new = v_grid_new
        // Blend: v_new = flip_ratio * FLIP + (1 - flip_ratio) * PIC

        // For simplicity, we're doing PIC with some FLIP blend
        // Full FLIP would require storing old grid velocities
        p.vx = flipRatio * p.vx + (1.0f - flipRatio) * newVx + (newVx - p.vx) * flipRatio;
        p.vy = flipRatio * p.vy + (1.0f - flipRatio) * newVy + (newVy - p.vy) * flipRatio;
        p.vz = flipRatio * p.vz + (1.0f - flipRatio) * newVz + (newVz - p.vz) * flipRatio;

        // Actually just use the interpolated velocity for stability
        p.vx = newVx;
        p.vy = newVy;
        p.vz = newVz;

        // Update APIC affine momentum (simplified - just diagonal)
        float dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz;
        m_grid->VelocityGradient(gx, gy, gz,
                                  dudx, dudy, dudz,
                                  dvdx, dvdy, dvdz,
                                  dwdx, dwdy, dwdz);

        // Store simplified gradient (diagonal only)
        float scale = 4.0f * m_grid->GetInvCellSize() * m_grid->GetInvCellSize();
        p.C00 = dudx * scale;
        p.C11 = dvdy * scale;
        (void)dudy; (void)dudz; (void)dvdx; (void)dvdz; (void)dwdx; (void)dwdy; (void)dwdz; // For full APIC later

        // Advect particle
        p.x += p.vx * deltaTime;
        p.y += p.vy * deltaTime;
        p.z += p.vz * deltaTime;
    }
}

void FluidSystem::ParticleCollisions() {
    float eps = m_config.cellSize * 0.1f;  // Small push distance

    for (uint32_t i = 0; i < m_activeParticles; ++i) {
        FluidParticle& p = m_particles[i];
        if (!HasFlag(p.flags, ParticleFlags::Active)) continue;

        // World boundary collisions
        if (p.x < m_config.boundsMinX + eps) {
            p.x = m_config.boundsMinX + eps;
            p.vx = std::max(0.0f, p.vx);
        }
        if (p.x > m_config.boundsMaxX - eps) {
            p.x = m_config.boundsMaxX - eps;
            p.vx = std::min(0.0f, p.vx);
        }
        if (p.y < m_config.boundsMinY + eps) {
            p.y = m_config.boundsMinY + eps;
            p.vy = std::max(0.0f, p.vy);
        }
        if (p.y > m_config.boundsMaxY - eps) {
            p.y = m_config.boundsMaxY - eps;
            p.vy = std::min(0.0f, p.vy);
        }
        if (p.z < m_config.boundsMinZ + eps) {
            p.z = m_config.boundsMinZ + eps;
            p.vz = std::max(0.0f, p.vz);
        }
        if (p.z > m_config.boundsMaxZ - eps) {
            p.z = m_config.boundsMaxZ - eps;
            p.vz = std::min(0.0f, p.vz);
        }

        // TODO: Check against custom colliders
        for (const auto& collider : m_colliders) {
            if (!collider.enabled) continue;

            // Simplified collision handling for basic shapes
            // Full implementation would use signed distance functions
        }
    }
}

void FluidSystem::UpdateSleeping() {
    uint32_t sleeping = 0;
    float thresh2 = m_config.sleepThreshold * m_config.sleepThreshold;

    for (uint32_t i = 0; i < m_activeParticles; ++i) {
        FluidParticle& p = m_particles[i];
        if (!HasFlag(p.flags, ParticleFlags::Active)) continue;

        float v2 = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;

        if (v2 < thresh2) {
            p.flags |= static_cast<uint32_t>(ParticleFlags::Sleeping);
            sleeping++;
        } else {
            p.flags &= ~static_cast<uint32_t>(ParticleFlags::Sleeping);
        }
    }

    m_stats.sleepingParticles = sleeping;
}

void FluidSystem::ComputeBuoyancy() {
    // Compute buoyancy forces for registered objects
    for (auto& obj : m_buoyancyObjects) {
        obj.forceX = 0.0f;
        obj.forceY = 0.0f;
        obj.forceZ = 0.0f;
        obj.torqueX = 0.0f;
        obj.torqueY = 0.0f;
        obj.torqueZ = 0.0f;

        // TODO: Query Jolt for object position/shape
        // Sample fluid density around object
        // Compute buoyancy force: F = rho_fluid * g * V_submerged
        // Compute drag force: F = 0.5 * rho * v^2 * Cd * A
    }
}

void FluidSystem::UpdateStats() {
    m_stats.activeParticles = m_activeParticles;

    float sumV = 0.0f, maxV = 0.0f;
    float sumKE = 0.0f, sumPE = 0.0f;
    uint32_t surface = 0;

    for (uint32_t i = 0; i < m_activeParticles; ++i) {
        const FluidParticle& p = m_particles[i];
        if (!HasFlag(p.flags, ParticleFlags::Active)) continue;

        float v = std::sqrt(p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
        sumV += v;
        maxV = std::max(maxV, v);

        sumKE += 0.5f * p.mass * v * v;
        sumPE += p.mass * (-m_config.gravity) * p.y;

        if (HasFlag(p.flags, ParticleFlags::Surface)) {
            surface++;
        }
    }

    if (m_activeParticles > 0) {
        m_stats.averageVelocity = sumV / m_activeParticles;
    }
    m_stats.maxVelocity = maxV;
    m_stats.totalKineticEnergy = sumKE;
    m_stats.totalPotentialEnergy = sumPE;
    m_stats.surfaceParticles = surface;

    // Grid stats
    uint32_t usedCells = 0;
    float sumDensity = 0.0f;
    for (size_t i = 0; i < m_grid->GetCellCount(); ++i) {
        const MACCell& cell = m_grid->GetData()[i];
        if (cell.state == CellState::Fluid) {
            usedCells++;
            sumDensity += cell.density;
        }
    }
    m_stats.gridCellsUsed = usedCells;
    if (usedCells > 0) {
        m_stats.averageDensity = sumDensity / usedCells;
    }
}

// =============================================================================
// GPU Implementation
// =============================================================================

void FluidSystem::StepGPU(float deltaTime) {
    // TODO: Implement GPU compute path
    // For now, fall back to CPU

    SyncParticlesToGPU();

    // Dispatch compute shaders...
    // m_p2gPipeline->Dispatch(...)
    // m_gridForcePipeline->Dispatch(...)
    // etc.

    SyncParticlesFromGPU();
}

void FluidSystem::SyncParticlesToGPU() {
    // TODO: Implement when GPU buffers are ready
    // if (!m_particleBuffer) return;
    // void* mapped = m_particleBuffer->Map();
    // if (mapped) {
    //     memcpy(mapped, m_particles.data(), m_activeParticles * sizeof(FluidParticle));
    //     m_particleBuffer->Unmap();
    // }
}

void FluidSystem::SyncParticlesFromGPU() {
    // TODO: Implement when GPU buffers are ready
    // if (!m_particleBuffer) return;
    // const void* mapped = m_particleBuffer->Map();
    // if (mapped) {
    //     memcpy(m_particles.data(), mapped, m_activeParticles * sizeof(FluidParticle));
    //     m_particleBuffer->Unmap();
    // }
}

} // namespace WulfNet
