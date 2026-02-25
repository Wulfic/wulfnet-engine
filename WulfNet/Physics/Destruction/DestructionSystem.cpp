// =============================================================================
// WulfNet Engine - Destruction Physics System Implementation
// =============================================================================

#include "DestructionSystem.h"
#include <cstring>
#include <chrono>

namespace WulfNet {

// =============================================================================
// Constructor / Destructor
// =============================================================================

DestructionSystem::DestructionSystem() = default;

DestructionSystem::~DestructionSystem() {
    Shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool DestructionSystem::Initialize(const DestructionConfig& config) {
    if (m_initialized) return false;

    m_config = config;
    m_rngState = 12345; // Deterministic seed
    m_stats = DestructionStats{};
    m_initialized = true;
    return true;
}

void DestructionSystem::Shutdown() {
    m_destructibles.clear();
    m_allFragments.clear();
    m_fractureCallback = nullptr;
    m_stats = DestructionStats{};
    m_initialized = false;
}

// =============================================================================
// Deterministic RNG
// =============================================================================

float DestructionSystem::RandomFloat() {
    // xorshift32
    m_rngState ^= m_rngState << 13;
    m_rngState ^= m_rngState >> 17;
    m_rngState ^= m_rngState << 5;
    return static_cast<float>(m_rngState & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

float DestructionSystem::RandomRange(float minVal, float maxVal) {
    return minVal + RandomFloat() * (maxVal - minVal);
}

// =============================================================================
// Destructible Body Registration
// =============================================================================

uint32_t DestructionSystem::AddDestructible(JPH::BodyID bodyId,
                                            float threshold,
                                            uint32_t cellCount) {
    if (!m_initialized) return UINT32_MAX;

    DestructibleBody body;
    body.intactBodyId = bodyId;
    body.fractureThreshold = threshold;
    body.stressThreshold = threshold * 1000.0f; // Default stress = impulse * 1000

    if (cellCount == 0) cellCount = m_config.defaultCellCount;
    cellCount = std::min(cellCount, m_config.maxCellCount);

    // Generate a default box fracture pattern (1m cube) if no pattern set later
    body.pattern = GenerateBoxPattern(0.5f, 0.5f, 0.5f, cellCount);
    body.mass = body.pattern.totalVolume * body.pattern.density;

    uint32_t handle = static_cast<uint32_t>(m_destructibles.size());
    m_destructibles.push_back(std::move(body));
    return handle;
}

DestructibleBody* DestructionSystem::GetDestructible(uint32_t handle) {
    if (handle < m_destructibles.size()) return &m_destructibles[handle];
    return nullptr;
}

const DestructibleBody* DestructionSystem::GetDestructible(uint32_t handle) const {
    if (handle < m_destructibles.size()) return &m_destructibles[handle];
    return nullptr;
}

void DestructionSystem::RemoveDestructible(uint32_t handle) {
    if (handle < m_destructibles.size()) {
        m_destructibles[handle].enabled = false;
    }
}

// =============================================================================
// Voronoi Pattern Generation
// =============================================================================

void DestructionSystem::GenerateVoronoiSites(float minX, float minY, float minZ,
                                             float maxX, float maxY, float maxZ,
                                             uint32_t count,
                                             std::vector<VoronoiCell>& cells) {
    cells.resize(count);

    for (uint32_t i = 0; i < count; ++i) {
        cells[i].centerX = RandomRange(minX, maxX);
        cells[i].centerY = RandomRange(minY, maxY);
        cells[i].centerZ = RandomRange(minZ, maxZ);
        cells[i].detached = false;
    }
}

void DestructionSystem::ComputeVoronoiVolumes(FracturePattern& pattern) {
    if (pattern.cells.empty()) return;

    float totalX = pattern.boundMaxX - pattern.boundMinX;
    float totalY = pattern.boundMaxY - pattern.boundMinY;
    float totalZ = pattern.boundMaxZ - pattern.boundMinZ;

    // Approximate Voronoi cells using nearest-site assignment on a grid
    // For each sample point, find the nearest site and accumulate volume
    const int sampleRes = 16; // 16^3 = 4096 sample points for volume estimation
    float stepX = totalX / static_cast<float>(sampleRes);
    float stepY = totalY / static_cast<float>(sampleRes);
    float stepZ = totalZ / static_cast<float>(sampleRes);
    float sampleVol = stepX * stepY * stepZ;

    // Track bounding boxes and volume for each cell
    uint32_t cellCount = static_cast<uint32_t>(pattern.cells.size());
    std::vector<float> volumes(cellCount, 0.0f);
    std::vector<float> cellMinX(cellCount, 1e30f);
    std::vector<float> cellMinY(cellCount, 1e30f);
    std::vector<float> cellMinZ(cellCount, 1e30f);
    std::vector<float> cellMaxX(cellCount, -1e30f);
    std::vector<float> cellMaxY(cellCount, -1e30f);
    std::vector<float> cellMaxZ(cellCount, -1e30f);

    for (int sk = 0; sk < sampleRes; ++sk) {
        for (int sj = 0; sj < sampleRes; ++sj) {
            for (int si = 0; si < sampleRes; ++si) {
                float px = pattern.boundMinX + (static_cast<float>(si) + 0.5f) * stepX;
                float py = pattern.boundMinY + (static_cast<float>(sj) + 0.5f) * stepY;
                float pz = pattern.boundMinZ + (static_cast<float>(sk) + 0.5f) * stepZ;

                // Find nearest Voronoi site
                float bestDist = 1e30f;
                uint32_t bestIdx = 0;

                for (uint32_t c = 0; c < cellCount; ++c) {
                    float dx = px - pattern.cells[c].centerX;
                    float dy = py - pattern.cells[c].centerY;
                    float dz = pz - pattern.cells[c].centerZ;
                    float dist = dx * dx + dy * dy + dz * dz;
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdx = c;
                    }
                }

                volumes[bestIdx] += sampleVol;

                // Update bounding box
                cellMinX[bestIdx] = std::min(cellMinX[bestIdx], px);
                cellMinY[bestIdx] = std::min(cellMinY[bestIdx], py);
                cellMinZ[bestIdx] = std::min(cellMinZ[bestIdx], pz);
                cellMaxX[bestIdx] = std::max(cellMaxX[bestIdx], px);
                cellMaxY[bestIdx] = std::max(cellMaxY[bestIdx], py);
                cellMaxZ[bestIdx] = std::max(cellMaxZ[bestIdx], pz);
            }
        }
    }

    pattern.totalVolume = 0.0f;
    for (uint32_t c = 0; c < cellCount; ++c) {
        pattern.cells[c].volume = volumes[c];
        pattern.cells[c].mass = volumes[c] * pattern.density;
        pattern.cells[c].minX = cellMinX[c];
        pattern.cells[c].minY = cellMinY[c];
        pattern.cells[c].minZ = cellMinZ[c];
        pattern.cells[c].maxX = cellMaxX[c];
        pattern.cells[c].maxY = cellMaxY[c];
        pattern.cells[c].maxZ = cellMaxZ[c];
        pattern.totalVolume += volumes[c];
    }
}

FracturePattern DestructionSystem::GenerateBoxPattern(
    float halfExtX, float halfExtY, float halfExtZ,
    uint32_t cellCount, float density) {

    FracturePattern pattern;
    pattern.boundMinX = -halfExtX;
    pattern.boundMinY = -halfExtY;
    pattern.boundMinZ = -halfExtZ;
    pattern.boundMaxX = halfExtX;
    pattern.boundMaxY = halfExtY;
    pattern.boundMaxZ = halfExtZ;
    pattern.density = density;

    if (cellCount == 0) cellCount = 8;

    // Create a temporary system for RNG
    DestructionSystem tempSys;
    DestructionConfig tempConfig;
    tempSys.Initialize(tempConfig);

    tempSys.GenerateVoronoiSites(-halfExtX, -halfExtY, -halfExtZ,
                                  halfExtX, halfExtY, halfExtZ,
                                  cellCount, pattern.cells);
    tempSys.ComputeVoronoiVolumes(pattern);
    tempSys.Shutdown();

    return pattern;
}

FracturePattern DestructionSystem::GenerateSpherePattern(
    float radius, uint32_t cellCount, float density) {

    FracturePattern pattern;
    pattern.boundMinX = -radius;
    pattern.boundMinY = -radius;
    pattern.boundMinZ = -radius;
    pattern.boundMaxX = radius;
    pattern.boundMaxY = radius;
    pattern.boundMaxZ = radius;
    pattern.density = density;

    if (cellCount == 0) cellCount = 8;

    // Create a temporary system for RNG
    DestructionSystem tempSys;
    DestructionConfig tempConfig;
    tempSys.Initialize(tempConfig);

    // Generate sites within the sphere
    std::vector<VoronoiCell> candidates;
    // Over-generate to reject points outside sphere
    uint32_t attempts = cellCount * 10;
    for (uint32_t i = 0; i < attempts && candidates.size() < cellCount; ++i) {
        float x = tempSys.RandomRange(-radius, radius);
        float y = tempSys.RandomRange(-radius, radius);
        float z = tempSys.RandomRange(-radius, radius);
        if (x * x + y * y + z * z <= radius * radius) {
            VoronoiCell cell;
            cell.centerX = x;
            cell.centerY = y;
            cell.centerZ = z;
            candidates.push_back(cell);
        }
    }

    // If we couldn't get enough sites, add remaining uniformly
    while (candidates.size() < cellCount) {
        VoronoiCell cell;
        cell.centerX = tempSys.RandomRange(-radius * 0.5f, radius * 0.5f);
        cell.centerY = tempSys.RandomRange(-radius * 0.5f, radius * 0.5f);
        cell.centerZ = tempSys.RandomRange(-radius * 0.5f, radius * 0.5f);
        candidates.push_back(cell);
    }

    pattern.cells = std::move(candidates);
    tempSys.ComputeVoronoiVolumes(pattern);
    tempSys.Shutdown();

    return pattern;
}

// =============================================================================
// Impact Evaluation
// =============================================================================

bool DestructionSystem::EvaluateImpact(uint32_t handle,
                                       float impactX, float impactY, float impactZ,
                                       float impulse) {
    if (!m_initialized || handle >= m_destructibles.size())
        return false;

    auto& body = m_destructibles[handle];
    if (!body.enabled || body.fractured) return false;

    // Apply global impulse scale
    float scaledImpulse = impulse * m_config.globalImpulseScale;

    if (scaledImpulse >= body.fractureThreshold) {
        Fracture(handle, impactX, impactY, impactZ);
        return true;
    }

    return false;
}

// =============================================================================
// Fracture Execution
// =============================================================================

uint32_t DestructionSystem::Fracture(uint32_t handle,
                                     float impactX, float impactY, float impactZ) {
    if (!m_initialized || handle >= m_destructibles.size())
        return 0;

    auto& body = m_destructibles[handle];
    if (body.fractured) return 0;
    if (body.fractureLevel >= body.maxFractureLevel) return 0;

    auto start = std::chrono::high_resolution_clock::now();

    body.fractured = true;

    uint32_t fragmentCount = 0;
    uint32_t cellCount = body.pattern.GetCellCount();

    // Check performance limits
    uint32_t maxNew = m_config.maxFragmentsPerFrame;
    if (static_cast<uint32_t>(m_allFragments.size()) + cellCount > m_config.maxTotalFragments) {
        return 0; // Too many fragments
    }

    for (uint32_t i = 0; i < cellCount && fragmentCount < maxNew; ++i) {
        auto& cell = body.pattern.cells[i];
        if (cell.detached) continue;

        // Check minimum mass
        if (cell.mass < m_config.minFragmentMass) continue;

        cell.detached = true;

        // In a real implementation, we'd create Jolt bodies here.
        // For the CPU reference, we track fragment IDs.
        // We use a synthetic BodyID based on a counter to represent fragments
        // without requiring an actual Jolt PhysicsSystem.
        uint32_t fragId = static_cast<uint32_t>(m_allFragments.size()) + 1;
        JPH::BodyID fragmentBodyId(fragId);

        body.fragmentBodyIds.push_back(fragmentBodyId);
        m_allFragments.push_back(fragmentBodyId);
        fragmentCount++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    m_stats.fractureTimeMs += std::chrono::duration<float, std::milli>(end - start).count();
    m_stats.fracturesThisFrame++;
    m_stats.totalFragmentsGenerated += fragmentCount;
    m_stats.fracturedBodies++;

    // Emit fracture event
    if (m_fractureCallback) {
        FractureEvent evt;
        evt.destructibleIndex = handle;
        evt.originalBodyId = body.intactBodyId;
        evt.impactX = impactX;
        evt.impactY = impactY;
        evt.impactZ = impactZ;
        evt.impulse = body.fractureThreshold;
        evt.fragmentCount = fragmentCount;
        m_fractureCallback(evt);
    }

    return fragmentCount;
}

// =============================================================================
// Step
// =============================================================================

void DestructionSystem::Step(float deltaTime, JPH::PhysicsSystem* /*joltPhysics*/) {
    if (!m_initialized) return;

    auto start = std::chrono::high_resolution_clock::now();

    // Reset per-frame stats
    m_stats.fracturesThisFrame = 0;
    m_stats.evaluationTimeMs = 0.0f;
    m_stats.fractureTimeMs = 0.0f;

    // Update counts
    m_stats.totalDestructibles = 0;
    m_stats.fracturedBodies = 0;
    m_stats.activeFragments = static_cast<uint32_t>(m_allFragments.size());

    for (const auto& body : m_destructibles) {
        if (!body.enabled) continue;
        m_stats.totalDestructibles++;
        if (body.fractured) m_stats.fracturedBodies++;
    }

    // In a full implementation, we'd sync body positions from Jolt,
    // check contacts/impulses, and auto-fracture. For the CPU reference,
    // fracture is driven by explicit EvaluateImpact calls.

    (void)deltaTime; // Used in a full implementation for fragment lifetime

    auto end = std::chrono::high_resolution_clock::now();
    m_stats.evaluationTimeMs =
        std::chrono::duration<float, std::milli>(end - start).count();
}

} // namespace WulfNet
