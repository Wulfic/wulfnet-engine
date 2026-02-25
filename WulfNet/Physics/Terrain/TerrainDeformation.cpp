// =============================================================================
// WulfNet Engine - Terrain Deformation System Implementation
// =============================================================================

#include "TerrainDeformation.h"
#include <chrono>
#include <cstring>

namespace WulfNet {

// =============================================================================
// Constructor / Destructor
// =============================================================================

TerrainDeformation::TerrainDeformation()
    : m_defaultMaterial(TerrainMaterial::SoftSoil()) {
}

TerrainDeformation::~TerrainDeformation() {
    Shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool TerrainDeformation::Initialize(const TerrainDeformConfig& config) {
    if (m_initialized) {
        Shutdown();
    }

    m_config = config;

    uint32_t totalCells = config.gridSizeX * config.gridSizeZ;
    if (totalCells == 0) return false;

    // Allocate height arrays
    m_heights.resize(totalCells, 0.0f);
    m_originalHeights.resize(totalCells, 0.0f);
    m_deltas.resize(totalCells, 0.0f);

    // Allocate material array with default material
    m_materials.resize(totalCells, m_defaultMaterial);

    m_dirty = false;
    m_dirtyMinX = UINT32_MAX;
    m_dirtyMinZ = UINT32_MAX;
    m_dirtyMaxX = 0;
    m_dirtyMaxZ = 0;

    m_history.clear();
    m_frameCounter = 0;
    m_stats = TerrainDeformStats{};

    m_initialized = true;
    return true;
}

void TerrainDeformation::Shutdown() {
    m_heights.clear();
    m_originalHeights.clear();
    m_deltas.clear();
    m_materials.clear();
    m_history.clear();
    m_stats = TerrainDeformStats{};
    m_initialized = false;
}

// =============================================================================
// Height Field Management
// =============================================================================

void TerrainDeformation::SetHeightField(const float* heights,
                                         uint32_t sizeX, uint32_t sizeZ) {
    if (!m_initialized) return;

    uint32_t copyX = std::min(sizeX, m_config.gridSizeX);
    uint32_t copyZ = std::min(sizeZ, m_config.gridSizeZ);

    for (uint32_t z = 0; z < copyZ; ++z) {
        for (uint32_t x = 0; x < copyX; ++x) {
            uint32_t srcIdx = z * sizeX + x;
            uint32_t dstIdx = CellIndex(x, z);
            float h = heights[srcIdx];
            h = std::max(m_config.minHeight, std::min(m_config.maxHeight, h));

            m_heights[dstIdx] = h;
            m_originalHeights[dstIdx] = h;
            m_deltas[dstIdx] = 0.0f;
        }
    }
}

void TerrainDeformation::SetHeightAt(uint32_t x, uint32_t z, float height) {
    if (!IsInBounds(x, z)) return;
    uint32_t idx = CellIndex(x, z);
    m_heights[idx] = std::max(m_config.minHeight, std::min(m_config.maxHeight, height));
    m_originalHeights[idx] = m_heights[idx];
    m_deltas[idx] = 0.0f;
}

float TerrainDeformation::GetHeightAt(uint32_t x, uint32_t z) const {
    if (!IsInBounds(x, z)) return 0.0f;
    return m_heights[CellIndex(x, z)];
}

float TerrainDeformation::GetOriginalHeightAt(uint32_t x, uint32_t z) const {
    if (!IsInBounds(x, z)) return 0.0f;
    return m_originalHeights[CellIndex(x, z)];
}

float TerrainDeformation::GetDeltaAt(uint32_t x, uint32_t z) const {
    if (!IsInBounds(x, z)) return 0.0f;
    return m_deltas[CellIndex(x, z)];
}

float TerrainDeformation::SampleHeight(float worldX, float worldZ) const {
    if (!m_initialized) return 0.0f;

    float gx, gz;
    WorldToGrid(worldX, worldZ, gx, gz);

    // Bilinear interpolation
    int ix = static_cast<int>(gx);
    int iz = static_cast<int>(gz);
    float fx = gx - ix;
    float fz = gz - iz;

    // Clamp to bounds
    ix = std::max(0, std::min(static_cast<int>(m_config.gridSizeX) - 2, ix));
    iz = std::max(0, std::min(static_cast<int>(m_config.gridSizeZ) - 2, iz));

    float h00 = m_heights[CellIndex(ix, iz)];
    float h10 = m_heights[CellIndex(ix + 1, iz)];
    float h01 = m_heights[CellIndex(ix, iz + 1)];
    float h11 = m_heights[CellIndex(ix + 1, iz + 1)];

    float h0 = h00 * (1.0f - fx) + h10 * fx;
    float h1 = h01 * (1.0f - fx) + h11 * fx;
    return h0 * (1.0f - fz) + h1 * fz;
}

float TerrainDeformation::SampleOriginalHeight(float worldX, float worldZ) const {
    if (!m_initialized) return 0.0f;

    float gx, gz;
    WorldToGrid(worldX, worldZ, gx, gz);

    int ix = static_cast<int>(gx);
    int iz = static_cast<int>(gz);
    float fx = gx - ix;
    float fz = gz - iz;

    ix = std::max(0, std::min(static_cast<int>(m_config.gridSizeX) - 2, ix));
    iz = std::max(0, std::min(static_cast<int>(m_config.gridSizeZ) - 2, iz));

    float h00 = m_originalHeights[CellIndex(ix, iz)];
    float h10 = m_originalHeights[CellIndex(ix + 1, iz)];
    float h01 = m_originalHeights[CellIndex(ix, iz + 1)];
    float h11 = m_originalHeights[CellIndex(ix + 1, iz + 1)];

    float h0 = h00 * (1.0f - fx) + h10 * fx;
    float h1 = h01 * (1.0f - fx) + h11 * fx;
    return h0 * (1.0f - fz) + h1 * fz;
}

// =============================================================================
// Material Management
// =============================================================================

void TerrainDeformation::SetMaterial(uint32_t x, uint32_t z,
                                      const TerrainMaterial& material) {
    if (!IsInBounds(x, z)) return;
    m_materials[CellIndex(x, z)] = material;
}

void TerrainDeformation::SetMaterialRegion(uint32_t startX, uint32_t startZ,
                                            uint32_t endX, uint32_t endZ,
                                            const TerrainMaterial& material) {
    for (uint32_t z = startZ; z <= endZ && z < m_config.gridSizeZ; ++z) {
        for (uint32_t x = startX; x <= endX && x < m_config.gridSizeX; ++x) {
            m_materials[CellIndex(x, z)] = material;
        }
    }
}

const TerrainMaterial& TerrainDeformation::GetMaterial(uint32_t x, uint32_t z) const {
    if (!IsInBounds(x, z)) return m_defaultMaterial;
    return m_materials[CellIndex(x, z)];
}

// =============================================================================
// Deformation Operations
// =============================================================================

void TerrainDeformation::ApplyStamp(const DeformationStamp& stamp) {
    if (!m_initialized) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Convert world position to grid
    float gx, gz;
    WorldToGrid(stamp.worldX, stamp.worldZ, gx, gz);

    // Compute affected cell range
    float stampRadiusCells = 0.0f;
    if (stamp.shape == StampShape::Circle) {
        stampRadiusCells = stamp.radius / m_config.cellSize;
    } else {
        // Bounding sphere of rectangle
        float halfDiag = std::sqrt(stamp.width * stamp.width + stamp.length * stamp.length) * 0.5f;
        stampRadiusCells = halfDiag / m_config.cellSize;
    }

    int minCellX = std::max(0, static_cast<int>(gx - stampRadiusCells - 1));
    int minCellZ = std::max(0, static_cast<int>(gz - stampRadiusCells - 1));
    int maxCellX = std::min(static_cast<int>(m_config.gridSizeX) - 1,
                            static_cast<int>(gx + stampRadiusCells + 1));
    int maxCellZ = std::min(static_cast<int>(m_config.gridSizeZ) - 1,
                            static_cast<int>(gz + stampRadiusCells + 1));

    // Record history
    RecordHistory(minCellX, minCellZ, maxCellX - minCellX + 1, maxCellZ - minCellZ + 1);

    float totalDisplacedVolume = 0.0f;
    float cosR = std::cos(stamp.rotation);
    float sinR = std::sin(stamp.rotation);

    for (int cz = minCellZ; cz <= maxCellZ; ++cz) {
        for (int cx = minCellX; cx <= maxCellX; ++cx) {
            float wx, wz;
            GridToWorld(cx, cz, wx, wz);

            // Distance from stamp center
            float dx = wx - stamp.worldX;
            float dz = wz - stamp.worldZ;

            // Rotate into stamp's local space
            float localX = dx * cosR + dz * sinR;
            float localZ = -dx * sinR + dz * cosR;

            float normalizedDist = 0.0f;

            if (stamp.shape == StampShape::Circle) {
                float dist = std::sqrt(localX * localX + localZ * localZ);
                normalizedDist = dist / stamp.radius;
            } else if (stamp.shape == StampShape::Rectangle) {
                float nx = std::abs(localX) / (stamp.width * 0.5f);
                float nz = std::abs(localZ) / (stamp.length * 0.5f);
                normalizedDist = std::max(nx, nz);
            } else if (stamp.shape == StampShape::Custom && stamp.customWidth > 0) {
                // Sample from custom heightmap
                float u = (localX / stamp.width + 0.5f) * stamp.customWidth;
                float v = (localZ / stamp.length + 0.5f) * stamp.customHeight;
                int ui = std::max(0, std::min(static_cast<int>(stamp.customWidth) - 1, static_cast<int>(u)));
                int vi = std::max(0, std::min(static_cast<int>(stamp.customHeight) - 1, static_cast<int>(v)));
                float customH = stamp.customHeights[vi * stamp.customWidth + ui];
                ApplyDeformationAtCell(cx, cz, -stamp.depth * customH);
                totalDisplacedVolume += stamp.depth * customH * m_config.cellSize * m_config.cellSize;
                ExpandDirtyRegion(cx, cz);
                continue;
            }

            if (normalizedDist >= 1.0f) continue;

            // Falloff
            float falloff = 1.0f - std::pow(normalizedDist, stamp.falloffExponent);
            float delta = -stamp.depth * falloff;

            ApplyDeformationAtCell(cx, cz, delta);
            totalDisplacedVolume += std::abs(delta) * m_config.cellSize * m_config.cellSize;
            ExpandDirtyRegion(cx, cz);
        }
    }

    // Volume conservation: raise rim
    if (m_config.conserveVolume && stamp.rimHeight == 0.0f) {
        float rimRadius = stamp.radius > 0.0f ? stamp.radius : stamp.width * 0.5f;
        DisplaceRim(stamp.worldX, stamp.worldZ,
                    rimRadius, rimRadius * 1.5f,
                    totalDisplacedVolume * m_config.volumeConservationRatio);
    }

    m_stats.totalDeformations++;
    m_stats.totalVolumeDisplaced += totalDisplacedVolume;

    auto endTime = std::chrono::high_resolution_clock::now();
    m_stats.lastDeformTimeMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    m_frameCounter++;
}

void TerrainDeformation::ApplyExplosion(float worldX, float worldZ,
                                         float radius, float force) {
    DeformationStamp stamp;
    stamp.shape = StampShape::Circle;
    stamp.worldX = worldX;
    stamp.worldZ = worldZ;
    stamp.radius = radius;
    stamp.depth = force;
    stamp.falloffExponent = 1.5f;

    ApplyStamp(stamp);
}

void TerrainDeformation::ApplyTireTrack(float startX, float startZ,
                                         float endX, float endZ,
                                         float width, float depth,
                                         float tirePatternPeriod) {
    if (!m_initialized) return;

    float dx = endX - startX;
    float dz = endZ - startZ;
    float trackLength = std::sqrt(dx * dx + dz * dz);
    if (trackLength < 1e-6f) return;

    float dirX = dx / trackLength;
    float dirZ = dz / trackLength;
    float perpX = -dirZ;
    float perpZ = dirX;

    float stepSize = m_config.cellSize * 0.5f;
    int steps = static_cast<int>(trackLength / stepSize) + 1;

    for (int s = 0; s <= steps; ++s) {
        float t = static_cast<float>(s) / steps;
        float wx = startX + dx * t;
        float wz = startZ + dz * t;

        // Apply tire pattern modulation
        float patternMod = 1.0f;
        if (tirePatternPeriod > 0.0f) {
            float dist = t * trackLength;
            patternMod = 0.7f + 0.3f * std::sin(dist / tirePatternPeriod * 6.2831853f);
        }

        // Sample left and right tire edges
        float halfWidth = width * 0.5f;
        int numWidthSamples = std::max(1, static_cast<int>(width / (m_config.cellSize * 0.5f)));

        for (int w = 0; w <= numWidthSamples; ++w) {
            float wt = static_cast<float>(w) / numWidthSamples;
            float offset = -halfWidth + wt * width;

            float sx = wx + perpX * offset;
            float sz = wz + perpZ * offset;

            float gx, gz;
            WorldToGrid(sx, sz, gx, gz);
            int cx = static_cast<int>(gx + 0.5f);
            int cz = static_cast<int>(gz + 0.5f);

            if (IsInBounds(cx, cz)) {
                // Tire profile: deeper at center, raised at edges
                float widthFactor = 1.0f - 2.0f * std::abs(wt - 0.5f);
                float delta = -depth * widthFactor * patternMod;
                ApplyDeformationAtCell(cx, cz, delta);
                ExpandDirtyRegion(cx, cz);
            }
        }
    }

    m_stats.totalDeformations++;
    m_frameCounter++;
}

void TerrainDeformation::ApplyFootprint(float worldX, float worldZ,
                                         float rotation,
                                         float footLength, float footWidth,
                                         float depth) {
    DeformationStamp stamp;
    stamp.shape = StampShape::Rectangle;
    stamp.worldX = worldX;
    stamp.worldZ = worldZ;
    stamp.width = footWidth;
    stamp.length = footLength;
    stamp.depth = depth;
    stamp.rotation = rotation;
    stamp.falloffExponent = 3.0f;  // Sharp edges for footprints

    ApplyStamp(stamp);
}

void TerrainDeformation::ApplyMPMDeformation(const float* particlePositions,
                                              const float* particleForces,
                                              uint32_t particleCount,
                                              float dt) {
    if (!m_initialized || particleCount == 0) return;

    for (uint32_t i = 0; i < particleCount; ++i) {
        float px = particlePositions[i * 3 + 0];
        float pz = particlePositions[i * 3 + 2];
        float fy = particleForces[i * 3 + 1];  // Vertical force component

        // Only deform if force pushes downward
        if (fy >= 0.0f) continue;

        float gx, gz;
        WorldToGrid(px, pz, gx, gz);

        int cx = static_cast<int>(gx + 0.5f);
        int cz = static_cast<int>(gz + 0.5f);

        if (!IsInBounds(cx, cz)) continue;

        // Force → displacement (simplified: d = F * dt² / mass_terrain)
        // We use the terrain material hardness as a resistance factor
        const TerrainMaterial& mat = m_materials[CellIndex(cx, cz)];
        float resistance = mat.hardness + 0.01f;  // Avoid zero
        float delta = fy * dt * dt * mat.displacementScale / resistance;

        ApplyDeformationAtCell(cx, cz, delta);
        ExpandDirtyRegion(cx, cz);
    }

    m_stats.totalDeformations++;
    m_frameCounter++;
}

// =============================================================================
// Undo / Reset
// =============================================================================

void TerrainDeformation::Undo() {
    if (m_history.empty()) return;

    const DeformationEvent& event = m_history.back();

    for (uint32_t dz = 0; dz < event.cellCountZ; ++dz) {
        for (uint32_t dx = 0; dx < event.cellCountX; ++dx) {
            uint32_t x = event.cellStartX + dx;
            uint32_t z = event.cellStartZ + dz;
            if (!IsInBounds(x, z)) continue;

            uint32_t srcIdx = dz * event.cellCountX + dx;
            uint32_t dstIdx = CellIndex(x, z);

            m_heights[dstIdx] = event.previousHeights[srcIdx];
            m_deltas[dstIdx] = m_heights[dstIdx] - m_originalHeights[dstIdx];
            ExpandDirtyRegion(x, z);
        }
    }

    m_history.pop_back();
}

void TerrainDeformation::Reset() {
    if (!m_initialized) return;

    uint32_t totalCells = m_config.gridSizeX * m_config.gridSizeZ;
    std::memcpy(m_heights.data(), m_originalHeights.data(), totalCells * sizeof(float));
    std::memset(m_deltas.data(), 0, totalCells * sizeof(float));

    m_dirty = true;
    m_dirtyMinX = 0;
    m_dirtyMinZ = 0;
    m_dirtyMaxX = m_config.gridSizeX - 1;
    m_dirtyMaxZ = m_config.gridSizeZ - 1;

    m_history.clear();
    m_stats = TerrainDeformStats{};
}

// =============================================================================
// Dirty Region Tracking
// =============================================================================

void TerrainDeformation::GetDirtyBounds(uint32_t& minX, uint32_t& minZ,
                                         uint32_t& maxX, uint32_t& maxZ) const {
    minX = m_dirtyMinX;
    minZ = m_dirtyMinZ;
    maxX = m_dirtyMaxX;
    maxZ = m_dirtyMaxZ;
}

void TerrainDeformation::ClearDirty() {
    m_dirty = false;
    m_dirtyMinX = UINT32_MAX;
    m_dirtyMinZ = UINT32_MAX;
    m_dirtyMaxX = 0;
    m_dirtyMaxZ = 0;
}

// =============================================================================
// Normal Computation
// =============================================================================

void TerrainDeformation::ComputeNormal(uint32_t x, uint32_t z,
                                        float& nx, float& ny, float& nz) const {
    if (!IsInBounds(x, z)) {
        nx = 0.0f; ny = 1.0f; nz = 0.0f;
        return;
    }

    // Central differences
    float hL = (x > 0) ? GetHeightAt(x - 1, z) : GetHeightAt(x, z);
    float hR = (x < m_config.gridSizeX - 1) ? GetHeightAt(x + 1, z) : GetHeightAt(x, z);
    float hD = (z > 0) ? GetHeightAt(x, z - 1) : GetHeightAt(x, z);
    float hU = (z < m_config.gridSizeZ - 1) ? GetHeightAt(x, z + 1) : GetHeightAt(x, z);

    float dx = (hR - hL) / (2.0f * m_config.cellSize);
    float dz = (hU - hD) / (2.0f * m_config.cellSize);

    nx = -dx;
    ny = 1.0f;
    nz = -dz;

    // Normalize
    float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-8f) {
        nx /= len; ny /= len; nz /= len;
    } else {
        nx = 0.0f; ny = 1.0f; nz = 0.0f;
    }
}

// =============================================================================
// Internal Helpers
// =============================================================================

uint32_t TerrainDeformation::CellIndex(uint32_t x, uint32_t z) const {
    return z * m_config.gridSizeX + x;
}

void TerrainDeformation::WorldToGrid(float wx, float wz, float& gx, float& gz) const {
    gx = (wx - m_config.originX) / m_config.cellSize;
    gz = (wz - m_config.originZ) / m_config.cellSize;
}

void TerrainDeformation::GridToWorld(uint32_t gx, uint32_t gz, float& wx, float& wz) const {
    wx = m_config.originX + gx * m_config.cellSize;
    wz = m_config.originZ + gz * m_config.cellSize;
}

bool TerrainDeformation::IsInBounds(uint32_t x, uint32_t z) const {
    return x < m_config.gridSizeX && z < m_config.gridSizeZ;
}

void TerrainDeformation::RecordHistory(uint32_t startX, uint32_t startZ,
                                        uint32_t countX, uint32_t countZ) {
    if (m_history.size() >= m_config.maxHistorySize) {
        m_history.erase(m_history.begin());
    }

    DeformationEvent event;
    event.worldX = m_config.originX + startX * m_config.cellSize;
    event.worldZ = m_config.originZ + startZ * m_config.cellSize;
    event.radius = std::max(countX, countZ) * m_config.cellSize * 0.5f;
    event.timestamp = m_frameCounter;
    event.cellStartX = startX;
    event.cellStartZ = startZ;
    event.cellCountX = countX;
    event.cellCountZ = countZ;

    event.previousHeights.resize(countX * countZ);
    for (uint32_t dz = 0; dz < countZ; ++dz) {
        for (uint32_t dx = 0; dx < countX; ++dx) {
            uint32_t x = startX + dx;
            uint32_t z = startZ + dz;
            if (IsInBounds(x, z)) {
                event.previousHeights[dz * countX + dx] = m_heights[CellIndex(x, z)];
            }
        }
    }

    m_history.push_back(std::move(event));
}

void TerrainDeformation::ApplyDeformationAtCell(uint32_t x, uint32_t z,
                                                 float delta,
                                                 bool respectMaterial) {
    if (!IsInBounds(x, z)) return;

    uint32_t idx = CellIndex(x, z);

    if (respectMaterial) {
        const TerrainMaterial& mat = m_materials[idx];
        delta *= mat.displacementScale * (1.0f - mat.hardness);
    }

    float newDelta = m_deltas[idx] + delta;

    // Clamp to limits
    newDelta = std::max(-m_config.maxDeformDepth, std::min(m_config.maxDeformRaise, newDelta));

    m_deltas[idx] = newDelta;
    m_heights[idx] = m_originalHeights[idx] + newDelta;

    // Clamp height to valid range
    m_heights[idx] = std::max(m_config.minHeight, std::min(m_config.maxHeight, m_heights[idx]));

    // Track stats
    m_stats.cellsModified++;
    m_stats.maxDepthReached = std::max(m_stats.maxDepthReached, std::abs(newDelta));
}

void TerrainDeformation::DisplaceRim(float centerX, float centerZ,
                                      float innerRadius, float outerRadius,
                                      float volume) {
    if (volume <= 0.0f) return;

    // Distribute displaced volume in an annular ring
    float gx, gz;
    WorldToGrid(centerX, centerZ, gx, gz);

    float innerCells = innerRadius / m_config.cellSize;
    float outerCells = outerRadius / m_config.cellSize;

    int minCellX = std::max(0, static_cast<int>(gx - outerCells - 1));
    int minCellZ = std::max(0, static_cast<int>(gz - outerCells - 1));
    int maxCellX = std::min(static_cast<int>(m_config.gridSizeX) - 1,
                            static_cast<int>(gx + outerCells + 1));
    int maxCellZ = std::min(static_cast<int>(m_config.gridSizeZ) - 1,
                            static_cast<int>(gz + outerCells + 1));

    // Count rim cells to distribute volume evenly
    int rimCellCount = 0;
    for (int cz = minCellZ; cz <= maxCellZ; ++cz) {
        for (int cx = minCellX; cx <= maxCellX; ++cx) {
            float dx = cx - gx;
            float dz = cz - gz;
            float dist = std::sqrt(dx * dx + dz * dz);
            if (dist >= innerCells && dist <= outerCells) {
                rimCellCount++;
            }
        }
    }

    if (rimCellCount == 0) return;

    float cellArea = m_config.cellSize * m_config.cellSize;
    float raisePerCell = volume / (rimCellCount * cellArea);

    for (int cz = minCellZ; cz <= maxCellZ; ++cz) {
        for (int cx = minCellX; cx <= maxCellX; ++cx) {
            float dx = cx - gx;
            float dz = cz - gz;
            float dist = std::sqrt(dx * dx + dz * dz);
            if (dist >= innerCells && dist <= outerCells) {
                // Gaussian-like falloff within rim
                float t = (dist - innerCells) / (outerCells - innerCells);
                float rimFalloff = std::exp(-2.0f * t * t);
                ApplyDeformationAtCell(cx, cz, raisePerCell * rimFalloff, false);
                ExpandDirtyRegion(cx, cz);
            }
        }
    }
}

void TerrainDeformation::ExpandDirtyRegion(uint32_t x, uint32_t z) {
    m_dirty = true;
    m_dirtyMinX = std::min(m_dirtyMinX, x);
    m_dirtyMinZ = std::min(m_dirtyMinZ, z);
    m_dirtyMaxX = std::max(m_dirtyMaxX, x);
    m_dirtyMaxZ = std::max(m_dirtyMaxZ, z);
}

} // namespace WulfNet
