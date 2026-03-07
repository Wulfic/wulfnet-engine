#include "WaterSystemV3.h"
#include "../Core/Logging/Logger.h"
#include "../Core/Profiling/Profiler.h"
#include <WulfNet/Jolt/Physics/Body/BodyInterface.h>
#include <WulfNet/Jolt/Physics/Collision/TransformedShape.h>
#include <WulfNet/Jolt/Core/JobSystemThreadPool.h>
#include <algorithm>
#include <execution>
#include <numeric>
#include <cmath>
#include <sstream>

using namespace WulfNet::Physics;

// Diagnostic category for all WaterSystemV3 log output
static constexpr const char* LOG_CAT = "WaterV3";

double WaterStateSOA::CalculateTotalVolume(float cellArea) const {
    double totalVol = 0.0;
    for (float d : waterDepth) totalVol += static_cast<double>(d) * cellArea;
    return totalVol;
}

WaterSystemV3::WaterSystemV3(const WaterSystemV3Config& config, JPH::PhysicsSystem* physicsSystem)
    : m_config(config), m_joltSystem(physicsSystem) {
    uint32_t totalCells = m_config.width * m_config.height;
    m_state.Resize(totalCells);

    std::ostringstream ss;
    ss << "Initialized: grid=" << m_config.width << "x" << m_config.height
       << " cellSize=" << m_config.gridSize << "m"
       << " totalCells=" << totalCells
       << " gravity=" << m_config.gravity
       << " density=" << m_config.density << "kg/m^3"
       << " dtMax=" << m_config.dtMax << "s"
       << " memoryMB=" << (totalCells * (sizeof(float) * 2 + sizeof(WaterStateSOA::Float4))) / (1024.0 * 1024.0);
    WULFNET_INFO(LOG_CAT, ss.str());
}

void WaterSystemV3::AddWater(uint32_t x, uint32_t y, float vol) {
    if (x >= m_config.width || y >= m_config.height) {
        WULFNET_WARNING(LOG_CAT, "AddWater out-of-bounds: (" + std::to_string(x) + "," + std::to_string(y) + ")");
        return;
    }
    float depthAdded = vol / (m_config.gridSize * m_config.gridSize);
    m_state.waterDepth[y * m_config.width + x] += depthAdded;
    WULFNET_TRACE(LOG_CAT, "AddWater(" + std::to_string(x) + "," + std::to_string(y) + ") vol=" + std::to_string(vol) + " depth+=" + std::to_string(depthAdded));
}
void WaterSystemV3::RemoveWater(uint32_t x, uint32_t y, float vol) {
    if (x >= m_config.width || y >= m_config.height) {
        WULFNET_WARNING(LOG_CAT, "RemoveWater out-of-bounds: (" + std::to_string(x) + "," + std::to_string(y) + ")");
        return;
    }
    float& d = m_state.waterDepth[y * m_config.width + x];
    d = std::max(0.0f, d - vol / (m_config.gridSize * m_config.gridSize));
}

// -------------------------------------------------------------
// CPU Simulation — Parallel "Virtual Pipe" Explicit SWE method
// All three passes (flux, damping, depth) are parallelized across
// rows using std::execution::par (MSVC maps to PPL thread pool).
// Each row is independent: cells only read neighbor data and write
// to their own flux/depth, so no synchronization is needed.
// -------------------------------------------------------------
void WaterSystemV3::StepSimulationCPU(float dt) {
    WULFNET_ZONE_NAMED("WaterV3::StepCPU");
    WULFNET_SCOPED_TIMER("WaterV3::StepCPU");

    // 1. CFL Condition (Stability)
    int steps = static_cast<int>(std::ceil(dt / m_config.dtMax));
    float subDt = dt / steps;

    WULFNET_TRACE(LOG_CAT, "StepCPU: dt=" + std::to_string(dt) + " subSteps=" + std::to_string(steps) + " subDt=" + std::to_string(subDt));

    const uint32_t W = m_config.width;
    const uint32_t H = m_config.height;
    const float dx = m_config.gridSize;
    const float pipeCrossArea = 1.0f;

    // Pre-build row index vector for parallel dispatch
    if (m_rowIndices.size() != H) {
        m_rowIndices.resize(H);
        std::iota(m_rowIndices.begin(), m_rowIndices.end(), 0u);
    }

    // Pre-build flat index vector for damping (parallel over all cells)
    const uint32_t totalCells = W * H;
    if (m_cellIndices.size() != totalCells) {
        m_cellIndices.resize(totalCells);
        std::iota(m_cellIndices.begin(), m_cellIndices.end(), 0u);
    }

    for (int s = 0; s < steps; ++s) {
        const float C = subDt * m_config.gravity * pipeCrossArea / dx;

        // ===== Step A: Calculate Flux (parallel over rows) =====
        // Each cell reads neighbor terrain+depth (read-only this pass)
        // and writes only to its own flux[i]. Safe for concurrent execution.
        std::for_each(std::execution::par, m_rowIndices.begin(), m_rowIndices.end(),
            [&](uint32_t y) {
                for (uint32_t x = 0; x < W; ++x) {
                    uint32_t i = y * W + x;
                    float h_self = m_state.terrainHeight[i];
                    float d_self = m_state.waterDepth[i];
                    float elev_self = h_self + d_self;

                    uint32_t pL = (x > 0) ? i - 1 : i;
                    uint32_t pR = (x < W - 1) ? i + 1 : i;
                    uint32_t pT = (y > 0) ? i - W : i;
                    uint32_t pB = (y < H - 1) ? i + W : i;

                    m_state.flux[i].L = std::max(0.0f, m_state.flux[i].L + C * (elev_self - (m_state.terrainHeight[pL] + m_state.waterDepth[pL])));
                    m_state.flux[i].R = std::max(0.0f, m_state.flux[i].R + C * (elev_self - (m_state.terrainHeight[pR] + m_state.waterDepth[pR])));
                    m_state.flux[i].T = std::max(0.0f, m_state.flux[i].T + C * (elev_self - (m_state.terrainHeight[pT] + m_state.waterDepth[pT])));
                    m_state.flux[i].B = std::max(0.0f, m_state.flux[i].B + C * (elev_self - (m_state.terrainHeight[pB] + m_state.waterDepth[pB])));

                    // Boundaries
                    if (x == 0)     m_state.flux[i].L = 0;
                    if (x == W - 1) m_state.flux[i].R = 0;
                    if (y == 0)     m_state.flux[i].T = 0;
                    if (y == H - 1) m_state.flux[i].B = 0;

                    // Conservation scaling factor K
                    float totalFluxOut = m_state.flux[i].L + m_state.flux[i].R + m_state.flux[i].T + m_state.flux[i].B;
                    if (totalFluxOut > 0.0f) {
                        float K = std::min(1.0f, (d_self * dx * dx) / (subDt * totalFluxOut));
                        m_state.flux[i].L *= K;
                        m_state.flux[i].R *= K;
                        m_state.flux[i].T *= K;
                        m_state.flux[i].B *= K;
                    }
                }
            }
        );

        // ===== Step A.5: Flux damping (parallel over all cells) =====
        if (m_config.fluxDamping > 0.0f) {
            float dampFactor = std::exp(-m_config.fluxDamping * subDt);
            std::for_each(std::execution::par, m_cellIndices.begin(), m_cellIndices.end(),
                [&](uint32_t i) {
                    m_state.flux[i].L *= dampFactor;
                    m_state.flux[i].R *= dampFactor;
                    m_state.flux[i].T *= dampFactor;
                    m_state.flux[i].B *= dampFactor;
                }
            );
        }

        // ===== Step B: Update Water Depths (parallel over rows) =====
        // Each cell reads neighbor flux (written in Step A, read-only now)
        // and writes only to its own waterDepth[i]. Safe for concurrency.
        std::for_each(std::execution::par, m_rowIndices.begin(), m_rowIndices.end(),
            [&](uint32_t y) {
                for (uint32_t x = 0; x < W; ++x) {
                    uint32_t i = y * W + x;
                    float d_self = m_state.waterDepth[i];

                    uint32_t pL = (x > 0) ? i - 1 : i;
                    uint32_t pR = (x < W - 1) ? i + 1 : i;
                    uint32_t pT = (y > 0) ? i - W : i;
                    uint32_t pB = (y < H - 1) ? i + W : i;

                    float fluxOut = m_state.flux[i].L + m_state.flux[i].R + m_state.flux[i].T + m_state.flux[i].B;
                    float fluxIn = 0.0f;
                    if (x > 0)     fluxIn += m_state.flux[pL].R;
                    if (x < W - 1) fluxIn += m_state.flux[pR].L;
                    if (y > 0)     fluxIn += m_state.flux[pT].B;
                    if (y < H - 1) fluxIn += m_state.flux[pB].T;

                    float newDepth = d_self + subDt * (fluxIn - fluxOut) / (dx * dx);
                    m_state.waterDepth[i] = (newDepth < 1e-6f) ? 0.0f : newDepth;
                }
            }
        );
    }
}

// -------------------------------------------------------------
// Jolt Integration & GPU Hookups
// -------------------------------------------------------------

void WaterSystemV3::BuildSparseActiveTilesCPU() {
    WULFNET_ZONE_NAMED("WaterV3::BuildSparse");
    m_activeTiles.clear();
    const uint32_t groupSize = 8;
    // Use ceiling division to cover edge tiles when grid isn't divisible by 8
    uint32_t groupsX = (m_config.width + groupSize - 1) / groupSize;
    uint32_t groupsY = (m_config.height + groupSize - 1) / groupSize;

    m_activeTiles.reserve(groupsX * groupsY); // Avoid repeated allocations

    for (uint32_t gy = 0; gy < groupsY; ++gy) {
        for (uint32_t gx = 0; gx < groupsX; ++gx) {
            bool isActive = false;

            // Check 8x8 block for depth or kinetic energy
            for (uint32_t ly = 0; ly < groupSize && !isActive; ++ly) {
                for (uint32_t lx = 0; lx < groupSize && !isActive; ++lx) {
                    uint32_t x = gx * groupSize + lx;
                    uint32_t y = gy * groupSize + ly;
                    // Guard against out-of-bounds at grid edges
                    if (x >= m_config.width || y >= m_config.height) continue;
                    uint32_t idx = y * m_config.width + x;

                    if (m_state.waterDepth[idx] > 0.001f ||
                        m_state.flux[idx].L > 0.001f || m_state.flux[idx].R > 0.001f ||
                        m_state.flux[idx].T > 0.001f || m_state.flux[idx].B > 0.001f)
                    {
                        isActive = true;
                    }
                }
            }

            if (isActive) {
                m_activeTiles.push_back({gx, gy});
            }
        }
    }
    WULFNET_DEBUG(LOG_CAT, "SparseTiles: " + std::to_string(m_activeTiles.size()) + " active of " + std::to_string(groupsX * groupsY) + " total");
}

void WaterSystemV3::InitializeGPUBuffers() {
    WULFNET_INFO(LOG_CAT, "InitializeGPUBuffers: stub (awaiting WulfNet::Compute backend)");
    // Stub: This will be implemented leveraging WulfNet's CommandLists and Device
    // Allocates SSBOs/UAVs for DX12 and Vulkan compute pipelines.
}

void WaterSystemV3::RequestAsyncReadback() {
    WULFNET_TRACE(LOG_CAT, "RequestAsyncReadback: stub (awaiting WulfNet::Compute backend)");
    // Stub: Issues a GPU -> CPU async copy of the water depth buffer.
    // Will be implemented when WulfNet::Compute backend is ready.
}

void WaterSystemV3::DispatchCompute(float deltaTime) {
    WULFNET_TRACE(LOG_CAT, "DispatchCompute: stub dt=" + std::to_string(deltaTime));
    // Stub: Binds WaterSWEV3 shader. Maps parameters and dispatches WorkGroups (W/8, H/8).
}

float WaterSystemV3::SampleWaterSurfaceHeight(float worldX, float worldZ) const {
    // Bi-linear interpolation across m_state CPU-side arrays
    // Transform from world space to grid space using configured origin
    float fx = (worldX - m_config.originX) / m_config.gridSize;
    float fz = (worldZ - m_config.originZ) / m_config.gridSize;

    uint32_t x0 = std::min(m_config.width - 1, std::max(0u, (uint32_t)fx));
    uint32_t z0 = std::min(m_config.height - 1, std::max(0u, (uint32_t)fz));
    uint32_t x1 = std::min(m_config.width - 1, x0 + 1);
    uint32_t z1 = std::min(m_config.height - 1, z0 + 1);

    float tx = fx - (float)x0;
    float tz = fz - (float)z0;
    tx = std::max(0.0f, std::min(1.0f, tx));
    tz = std::max(0.0f, std::min(1.0f, tz));

    auto surfaceAt = [&](uint32_t cx, uint32_t cz) -> float {
        uint32_t idx = cz * m_config.width + cx;
        if (m_state.waterDepth[idx] <= 1e-4f) return -1e9f; // Sentinel for dry
        return m_state.terrainHeight[idx] + m_state.waterDepth[idx];
    };

    float s00 = surfaceAt(x0, z0);
    float s10 = surfaceAt(x1, z0);
    float s01 = surfaceAt(x0, z1);
    float s11 = surfaceAt(x1, z1);

    // Only interpolate across wet cells; ignore dry sentinels
    float sum = 0.0f;
    float weight = 0.0f;
    auto accumulate = [&](float s, float w) {
        if (s > -1e8f) { sum += s * w; weight += w; }
    };
    accumulate(s00, (1 - tx) * (1 - tz));
    accumulate(s10, tx * (1 - tz));
    accumulate(s01, (1 - tx) * tz);
    accumulate(s11, tx * tz);

    if (weight < 1e-6f) return -1e9f; // Entirely dry region
    return sum / weight;
}

void WaterSystemV3::ApplyBuoyancyForces(JPH::JobSystem* jobSystem) {
    WULFNET_ZONE_NAMED("WaterV3::Buoyancy");
    if (!m_joltSystem) {
        WULFNET_WARNING(LOG_CAT, "ApplyBuoyancyForces: no Jolt PhysicsSystem bound");
        return;
    }

    JPH::BodyInterface& bodyIF = m_joltSystem->GetBodyInterface();
    // Assuming m_jobContext.interactingBodies is populated with overlapping bodies

    // For a multi-threaded parallel-for approach over the physics objects:
// Real Multi-threaded Dispatch mapping directly onto Jolt's Thread Pool
    // This scales perfectly with your hardware's CPU cores and leverages async job batches.
    const int batchSize = 64;
    int numBodies = static_cast<int>(m_jobContext.interactingBodies.size());
    if (numBodies == 0) return;

    WULFNET_DEBUG(LOG_CAT, "Buoyancy: processing " + std::to_string(numBodies) + " bodies in batches of " + std::to_string(batchSize));

    // Collect all job handles to ensure completion before returning
    std::vector<JPH::JobHandle> jobHandles;
    jobHandles.reserve((numBodies + batchSize - 1) / batchSize);

    for (int startIndex = 0; startIndex < numBodies; startIndex += batchSize) {
        int endIndex = std::min(startIndex + batchSize, numBodies);

        // Create the Jolt Job — with 0 dependencies, it auto-queues for execution
        jobHandles.emplace_back(jobSystem->CreateJob(
            "WaterBuoyancyBatch", JPH::Color::sCyan,
            [this, &bodyIF, startIndex, endIndex]() {
                for (int i = startIndex; i < endIndex; ++i) {
                    JPH::BodyID id = m_jobContext.interactingBodies[i];
                    if (!bodyIF.IsActive(id)) continue;

                    // Get bounds via TransformedShape (BodyInterface has no GetWorldSpaceBounds)
                    JPH::TransformedShape ts = bodyIF.GetTransformedShape(id);
                    JPH::AABox bounds = ts.GetWorldSpaceBounds();

                    // Multi-sample heightfield for better stability on large objects
                    float minX = bounds.mMin.GetX();
                    float maxX = bounds.mMax.GetX();
                    float minZ = bounds.mMin.GetZ();
                    float maxZ = bounds.mMax.GetZ();
                    float centerX = (minX + maxX) * 0.5f;
                    float centerZ = (minZ + maxZ) * 0.5f;

                    // Sample 5 points and average only wet cells to avoid
                    // sentinel corruption at water/land boundaries
                    float samples[5] = {
                        SampleWaterSurfaceHeight(minX, minZ),
                        SampleWaterSurfaceHeight(maxX, minZ),
                        SampleWaterSurfaceHeight(minX, maxZ),
                        SampleWaterSurfaceHeight(maxX, maxZ),
                        SampleWaterSurfaceHeight(centerX, centerZ)
                    };

                    float avgWaterHeight = 0.0f;
                    int wetCount = 0;
                    for (float s : samples) {
                        if (s > -1e8f) { avgWaterHeight += s; ++wetCount; }
                    }
                    if (wetCount == 0) continue; // Entirely dry — no buoyancy
                    avgWaterHeight /= static_cast<float>(wetCount);

                    float objectBottom = bounds.mMin.GetY();

                    if (objectBottom < avgWaterHeight) {
                        float objectTop = bounds.mMax.GetY();
                        float submergedDepth = std::max(0.0f, avgWaterHeight - objectBottom);
                        float objectHeight = std::max(0.001f, objectTop - objectBottom);

                        float objVol = (maxX - minX) * objectHeight * (maxZ - minZ);
                        float fraction = std::min(1.0f, submergedDepth / objectHeight);
                        float submergedVol = objVol * fraction;

                        float upForce = m_config.density * submergedVol * m_config.gravity;
                        JPH::Vec3 force(0, upForce, 0);

                        // Hydrodynamic Drag Calculation
                        JPH::Vec3 velocity = bodyIF.GetLinearVelocity(id);
                        float vSq = velocity.LengthSq();

                        if (vSq > 0.001f) {
                            JPH::Vec3 dragDir = -velocity.Normalized();
                            float dragMagnitude = 0.5f * m_config.density * vSq * m_config.dragCoefficient * fraction;
                            force += dragDir * dragMagnitude;
                        }

                        bodyIF.AddForce(id, force);
                    }
                }
            }));
    }
    // JobHandles auto-destruct here, blocking until all jobs complete
}
