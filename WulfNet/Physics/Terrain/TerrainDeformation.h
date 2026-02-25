// =============================================================================
// WulfNet Engine - Terrain Deformation System
// =============================================================================
// Runtime terrain modification via height field manipulation.
// Integrates with Jolt's HeightFieldShape for physics-aware deformation.
//
// Features:
//   - Stamp-based deformation (footprints, tire tracks, craters)
//   - MPM-driven plastic deformation (soil displacement)
//   - Material-aware deformation response (hard rock vs soft mud)
//   - Undo/redo with deformation history
//   - Height field delta tracking for network sync
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <string>

namespace WulfNet {

// Forward declarations
class PhysicsWorld;

// =============================================================================
// Terrain Material
// =============================================================================

enum class TerrainMaterialType : uint32_t {
    Rock = 0,           // Very hard, minimal deformation
    HardSoil = 1,       // Packed earth
    SoftSoil = 2,       // Loose dirt
    Sand = 3,           // Granular, easy to displace
    Mud = 4,            // Wet, plastic deformation
    Snow = 5,           // Very soft, large displacement
    Grass = 6,          // Soil with vegetation resistance
    Gravel = 7,         // Loose stones
    Custom = 255
};

struct TerrainMaterial {
    TerrainMaterialType type = TerrainMaterialType::SoftSoil;

    float hardness = 0.5f;              // 0 = infinitely soft, 1 = infinitely hard
    float restitution = 0.0f;           // How much the terrain springs back (0 = permanent)
    float displacementScale = 1.0f;     // Multiplier for deformation depth
    float spreadFactor = 1.0f;          // How much displaced material spreads outward

    // Visual properties (for splatmap generation)
    float wetness = 0.0f;              // 0-1, affects color darkening
    float roughness = 0.8f;            // Surface roughness

    static TerrainMaterial Rock() {
        TerrainMaterial m;
        m.type = TerrainMaterialType::Rock;
        m.hardness = 0.95f;
        m.restitution = 0.0f;
        m.displacementScale = 0.05f;
        m.spreadFactor = 0.2f;
        m.roughness = 0.9f;
        return m;
    }

    static TerrainMaterial HardSoil() {
        TerrainMaterial m;
        m.type = TerrainMaterialType::HardSoil;
        m.hardness = 0.7f;
        m.restitution = 0.05f;
        m.displacementScale = 0.3f;
        m.spreadFactor = 0.5f;
        return m;
    }

    static TerrainMaterial SoftSoil() {
        TerrainMaterial m;
        m.type = TerrainMaterialType::SoftSoil;
        m.hardness = 0.4f;
        m.restitution = 0.1f;
        m.displacementScale = 1.0f;
        m.spreadFactor = 1.0f;
        return m;
    }

    static TerrainMaterial Sand() {
        TerrainMaterial m;
        m.type = TerrainMaterialType::Sand;
        m.hardness = 0.2f;
        m.restitution = 0.0f;
        m.displacementScale = 1.5f;
        m.spreadFactor = 1.8f;
        return m;
    }

    static TerrainMaterial Mud() {
        TerrainMaterial m;
        m.type = TerrainMaterialType::Mud;
        m.hardness = 0.1f;
        m.restitution = 0.15f;
        m.displacementScale = 2.0f;
        m.spreadFactor = 1.5f;
        m.wetness = 0.8f;
        return m;
    }

    static TerrainMaterial Snow() {
        TerrainMaterial m;
        m.type = TerrainMaterialType::Snow;
        m.hardness = 0.05f;
        m.restitution = 0.0f;
        m.displacementScale = 3.0f;
        m.spreadFactor = 0.8f;
        m.wetness = 0.3f;
        return m;
    }
};

// =============================================================================
// Deformation Stamp (for footprints, tire tracks, etc.)
// =============================================================================

enum class StampShape : uint32_t {
    Circle = 0,
    Rectangle = 1,
    Custom = 2      // Uses custom heightmap
};

struct DeformationStamp {
    StampShape shape = StampShape::Circle;

    // Position in world space
    float worldX = 0.0f;
    float worldZ = 0.0f;

    // Size
    float radius = 0.5f;           // For circle
    float width = 1.0f;            // For rectangle
    float length = 1.0f;           // For rectangle

    // Rotation (Y-axis angle in radians)
    float rotation = 0.0f;

    // Deformation
    float depth = 0.1f;            // Meters to depress
    float rimHeight = 0.0f;        // Height of displaced rim (auto-computed if 0)

    // Falloff
    float falloffExponent = 2.0f;  // 1 = linear, 2 = quadratic, higher = sharper edges

    // Custom stamp heightmap
    std::vector<float> customHeights;
    uint32_t customWidth = 0;
    uint32_t customHeight = 0;
};

// =============================================================================
// Deformation Event (for history tracking)
// =============================================================================

struct DeformationEvent {
    float worldX, worldZ;
    float radius;
    float deltaMin, deltaMax;
    uint32_t timestamp;             // Frame number
    uint32_t cellStartX, cellStartZ;
    uint32_t cellCountX, cellCountZ;
    std::vector<float> previousHeights;  // For undo
};

// =============================================================================
// Terrain Deformation Configuration
// =============================================================================

struct TerrainDeformConfig {
    // Height field dimensions
    uint32_t gridSizeX = 256;
    uint32_t gridSizeZ = 256;
    float cellSize = 0.5f;         // Meters per grid cell

    // World position of terrain origin
    float originX = 0.0f;
    float originY = 0.0f;
    float originZ = 0.0f;

    // Height range
    float minHeight = -10.0f;
    float maxHeight = 100.0f;

    // Deformation limits
    float maxDeformDepth = 5.0f;    // Max depression below original
    float maxDeformRaise = 2.0f;    // Max raise above original (rim displacement)

    // History
    uint32_t maxHistorySize = 1000; // Max deformation events to track

    // Volume conservation
    bool conserveVolume = true;     // Displaced material raises rim
    float volumeConservationRatio = 0.7f;  // How much volume is pushed up vs lost
};

// =============================================================================
// Terrain Statistics
// =============================================================================

struct TerrainDeformStats {
    uint32_t totalDeformations = 0;
    uint32_t cellsModified = 0;
    float totalVolumeDisplaced = 0.0f;  // Cubic meters
    float maxDepthReached = 0.0f;
    float lastDeformTimeMs = 0.0f;
};

// =============================================================================
// Terrain Deformation System
// =============================================================================

class TerrainDeformation {
public:
    TerrainDeformation();
    ~TerrainDeformation();

    // Initialization
    bool Initialize(const TerrainDeformConfig& config);
    void Shutdown();
    bool IsInitialized() const { return m_initialized; }

    // Height field management
    void SetHeightField(const float* heights, uint32_t sizeX, uint32_t sizeZ);
    void SetHeightAt(uint32_t x, uint32_t z, float height);
    float GetHeightAt(uint32_t x, uint32_t z) const;
    float GetOriginalHeightAt(uint32_t x, uint32_t z) const;
    float GetDeltaAt(uint32_t x, uint32_t z) const;

    // World-space height sampling (with bilinear interpolation)
    float SampleHeight(float worldX, float worldZ) const;
    float SampleOriginalHeight(float worldX, float worldZ) const;

    // Material per-cell
    void SetMaterial(uint32_t x, uint32_t z, const TerrainMaterial& material);
    void SetMaterialRegion(uint32_t startX, uint32_t startZ,
                           uint32_t endX, uint32_t endZ,
                           const TerrainMaterial& material);
    const TerrainMaterial& GetMaterial(uint32_t x, uint32_t z) const;

    // Deformation operations
    void ApplyStamp(const DeformationStamp& stamp);
    void ApplyExplosion(float worldX, float worldZ, float radius, float force);
    void ApplyTireTrack(float startX, float startZ, float endX, float endZ,
                        float width, float depth, float tirePatternPeriod = 0.0f);
    void ApplyFootprint(float worldX, float worldZ, float rotation,
                        float footLength, float footWidth, float depth);

    // MPM integration — apply deformation from MPM particle forces
    void ApplyMPMDeformation(const float* particlePositions,
                             const float* particleForces,
                             uint32_t particleCount,
                             float dt);

    // Undo/redo
    bool CanUndo() const { return !m_history.empty(); }
    void Undo();
    void Reset();

    // Access
    const float* GetHeights() const { return m_heights.data(); }
    const float* GetOriginalHeights() const { return m_originalHeights.data(); }
    const float* GetDeltas() const { return m_deltas.data(); }
    uint32_t GetGridSizeX() const { return m_config.gridSizeX; }
    uint32_t GetGridSizeZ() const { return m_config.gridSizeZ; }
    float GetCellSize() const { return m_config.cellSize; }
    const TerrainDeformConfig& GetConfig() const { return m_config; }
    const TerrainDeformStats& GetStats() const { return m_stats; }

    // Dirty region tracking (for Jolt HeightField updates)
    bool HasDirtyRegion() const { return m_dirty; }
    void GetDirtyBounds(uint32_t& minX, uint32_t& minZ,
                        uint32_t& maxX, uint32_t& maxZ) const;
    void ClearDirty();

    // Normal computation
    void ComputeNormal(uint32_t x, uint32_t z,
                       float& nx, float& ny, float& nz) const;

private:
    // Helpers
    uint32_t CellIndex(uint32_t x, uint32_t z) const;
    void WorldToGrid(float wx, float wz, float& gx, float& gz) const;
    void GridToWorld(uint32_t gx, uint32_t gz, float& wx, float& wz) const;
    bool IsInBounds(uint32_t x, uint32_t z) const;

    void RecordHistory(uint32_t startX, uint32_t startZ,
                       uint32_t countX, uint32_t countZ);
    void ApplyDeformationAtCell(uint32_t x, uint32_t z,
                                float delta, bool respectMaterial = true);
    void DisplaceRim(float centerX, float centerZ,
                     float innerRadius, float outerRadius,
                     float volume);
    void ExpandDirtyRegion(uint32_t x, uint32_t z);

    // Configuration
    TerrainDeformConfig m_config;
    bool m_initialized = false;

    // Height data
    std::vector<float> m_heights;          // Current heights
    std::vector<float> m_originalHeights;  // Initial heights (for reset)
    std::vector<float> m_deltas;           // height - original

    // Material per cell
    std::vector<TerrainMaterial> m_materials;
    TerrainMaterial m_defaultMaterial;

    // Dirty tracking
    bool m_dirty = false;
    uint32_t m_dirtyMinX = UINT32_MAX, m_dirtyMinZ = UINT32_MAX;
    uint32_t m_dirtyMaxX = 0, m_dirtyMaxZ = 0;

    // History
    std::vector<DeformationEvent> m_history;
    uint32_t m_frameCounter = 0;

    // Statistics
    TerrainDeformStats m_stats;
};

} // namespace WulfNet
