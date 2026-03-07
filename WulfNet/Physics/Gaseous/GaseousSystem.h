// =============================================================================
// WulfNet Engine - Gaseous Simulation System
// =============================================================================
// Eulerian grid-based simulation for smoke, fire, and explosions.
// Uses a MAC (Marker-And-Cell) staggered velocity grid with semi-Lagrangian
// advection, buoyancy forces, vorticity confinement, and Jacobi/Red-Black
// pressure projection.
//
// Supports:
//   - Smoke density (passive scalar transport)
//   - Temperature field (drives buoyancy)
//   - Fuel/reaction field (for fire simulation)
//   - Vorticity confinement (preserves small-scale turbulence)
//   - Emitters (point, sphere, box)
//   - Collider obstacles (from Jolt bodies)
//   - CPU reference path with GPU compute hooks
//
// References:
//   - Stam 1999 "Stable Fluids"
//   - Fedkiw et al. 2001 "Visual Simulation of Smoke"
//   - Nguyen et al. 2002 "Physically Based Modeling and Animation of Fire"
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

namespace WulfNet {

// Forward declarations
class VulkanContext;

// =============================================================================
// Gas Cell — stores quantities at grid nodes
// =============================================================================

struct alignas(16) GasCell {
    // Staggered face velocities (MAC grid)
    float u = 0.0f, v = 0.0f, w = 0.0f;

    // Cell-centered scalar fields
    float density = 0.0f;        // Smoke/soot density [0..∞)
    float temperature = 0.0f;    // Temperature (K above ambient)
    float fuel = 0.0f;           // Fuel concentration [0..1]
    float reaction = 0.0f;       // Reaction progress [0..1]

    // Pressure solve
    float pressure = 0.0f;
    float divergence = 0.0f;

    // Vorticity
    float vorticityX = 0.0f;
    float vorticityY = 0.0f;
    float vorticityZ = 0.0f;

    // Cell type
    enum class State : uint8_t {
        Air = 0,
        Gas = 1,
        Solid = 2
    };
    State state = State::Air;
    uint8_t _pad[3] = {};

    void Reset() {
        u = v = w = 0.0f;
        density = temperature = fuel = reaction = 0.0f;
        pressure = divergence = 0.0f;
        vorticityX = vorticityY = vorticityZ = 0.0f;
        state = State::Air;
    }
};

static_assert(sizeof(GasCell) == 64, "GasCell must be 64 bytes for GPU alignment");

// =============================================================================
// Gaseous System Configuration
// =============================================================================

struct GaseousSystemConfig {
    // Grid resolution
    uint32_t resolutionX = 64;
    uint32_t resolutionY = 64;
    uint32_t resolutionZ = 64;
    float cellSize = 0.1f;              // Meters per cell

    // Domain origin (world space)
    float originX = 0.0f;
    float originY = 0.0f;
    float originZ = 0.0f;

    // Physics
    float ambientTemperature = 300.0f;  // Kelvin
    float buoyancyAlpha = 0.1f;         // Smoke weight (density → downward)
    float buoyancyBeta = 0.5f;          // Thermal lift (temp → upward)
    float gravityY = -9.81f;

    // Dissipation (per second)
    float densityDissipation = 0.98f;   // Density decay rate [0..1]
    float temperatureDissipation = 0.95f;
    float velocityDissipation = 0.99f;
    float fuelDissipation = 0.99f;

    // Vorticity confinement
    float vorticityStrength = 0.5f;     // Confinement force scale

    // Combustion (fire)
    float ignitionTemperature = 500.0f; // K — fuel ignites above this
    float burnRate = 2.0f;              // Fuel consumed per second
    float burnTemperature = 1500.0f;    // K — heat released by combustion
    float sootGeneration = 0.5f;        // Smoke generated per fuel burned

    // Pressure solve
    uint32_t pressureIterations = 40;   // Jacobi iterations

    // Performance
    bool useGPU = false;
    uint32_t substeps = 1;
    float maxTimestep = 1.0f / 30.0f;
};

// =============================================================================
// Gas Emitter
// =============================================================================

enum class GasEmitterType : uint32_t {
    Point = 0,
    Sphere = 1,
    Box = 2
};

struct GasEmitter {
    GasEmitterType type = GasEmitterType::Point;

    // Position (world space)
    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;

    // Size
    float radius = 0.5f;                    // For sphere
    float sizeX = 1.0f, sizeY = 1.0f, sizeZ = 1.0f;  // For box

    // Emission rates
    float densityRate = 10.0f;              // Density injected per second
    float temperatureRate = 500.0f;         // Temperature injection (K)
    float fuelRate = 0.0f;                  // Fuel injection rate
    float velocityX = 0.0f, velocityY = 1.0f, velocityZ = 0.0f; // Direction

    bool enabled = true;
};

// =============================================================================
// Gas Obstacle (solid cells from Jolt bodies)
// =============================================================================

struct GasObstacle {
    enum class Shape : uint32_t {
        Sphere = 0,
        Box = 1
    };
    Shape shape = Shape::Box;

    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
    float halfExtentX = 0.5f, halfExtentY = 0.5f, halfExtentZ = 0.5f;
    float radius = 0.5f;

    bool enabled = true;
};

// =============================================================================
// Gaseous Statistics
// =============================================================================

struct GaseousStats {
    uint32_t activeCells = 0;       // Cells with density > threshold
    uint32_t solidCells = 0;
    float totalDensity = 0.0f;
    float maxDensity = 0.0f;
    float maxTemperature = 0.0f;
    float maxVelocity = 0.0f;
    float totalFuel = 0.0f;

    // Timing
    float advectTimeMs = 0.0f;
    float forcesTimeMs = 0.0f;
    float pressureTimeMs = 0.0f;
    float vorticityTimeMs = 0.0f;
    float totalTimeMs = 0.0f;
};

// =============================================================================
// Gaseous Simulation System
// =============================================================================

class GaseousSystem {
public:
    GaseousSystem();
    ~GaseousSystem();

    // Non-copyable
    GaseousSystem(const GaseousSystem&) = delete;
    GaseousSystem& operator=(const GaseousSystem&) = delete;

    // =========================================================================
    // Initialization
    // =========================================================================

    bool Initialize(const GaseousSystemConfig& config);
    void Shutdown();
    bool IsInitialized() const { return m_initialized; }

    // GPU setup (optional)
    bool InitializeGPU(VulkanContext* context);
    bool IsGPUEnabled() const { return m_gpuEnabled; }

    // =========================================================================
    // Emitters
    // =========================================================================

    uint32_t AddEmitter(const GasEmitter& emitter);
    GasEmitter* GetEmitter(uint32_t id);
    void RemoveEmitter(uint32_t id);
    uint32_t GetEmitterCount() const { return static_cast<uint32_t>(m_emitters.size()); }

    // =========================================================================
    // Obstacles
    // =========================================================================

    uint32_t AddObstacle(const GasObstacle& obstacle);
    GasObstacle* GetObstacle(uint32_t id);
    void RemoveObstacle(uint32_t id);
    uint32_t GetObstacleCount() const { return static_cast<uint32_t>(m_obstacles.size()); }

    // =========================================================================
    // Simulation
    // =========================================================================

    /// Step the simulation forward by deltaTime seconds
    void Step(float deltaTime);

    /// Reset all fields to zero
    void Reset();

    // =========================================================================
    // Field Access
    // =========================================================================

    /// Get a cell by grid index
    const GasCell& GetCell(uint32_t i, uint32_t j, uint32_t k) const;
    GasCell& GetCellMut(uint32_t i, uint32_t j, uint32_t k);

    /// Get density at a world position (trilinear interpolation)
    float SampleDensity(float wx, float wy, float wz) const;

    /// Get temperature at a world position
    float SampleTemperature(float wx, float wy, float wz) const;

    /// Get velocity at a world position (trilinear on staggered grid)
    void SampleVelocity(float wx, float wy, float wz,
                        float& vx, float& vy, float& vz) const;

    /// Set density directly at a grid cell
    void SetDensity(uint32_t i, uint32_t j, uint32_t k, float density);

    /// Set temperature directly at a grid cell
    void SetTemperature(uint32_t i, uint32_t j, uint32_t k, float temp);

    /// Set fuel directly at a grid cell
    void SetFuel(uint32_t i, uint32_t j, uint32_t k, float fuel);

    // =========================================================================
    // Grid Info
    // =========================================================================

    uint32_t GetResolutionX() const { return m_resX; }
    uint32_t GetResolutionY() const { return m_resY; }
    uint32_t GetResolutionZ() const { return m_resZ; }
    float GetCellSize() const { return m_cellSize; }
    uint32_t GetCellCount() const { return m_resX * m_resY * m_resZ; }

    // =========================================================================
    // Coordinate Conversion
    // =========================================================================

    void WorldToGrid(float wx, float wy, float wz,
                     float& gx, float& gy, float& gz) const;
    void GridToWorld(float gx, float gy, float gz,
                     float& wx, float& wy, float& wz) const;

    // =========================================================================
    // Statistics
    // =========================================================================

    const GaseousStats& GetStats() const { return m_stats; }
    const GaseousSystemConfig& GetConfig() const { return m_config; }

private:
    // Internal grid index
    uint32_t CellIndex(uint32_t i, uint32_t j, uint32_t k) const;
    bool InBounds(int i, int j, int k) const;

    // Simulation substeps (CPU reference)
    void ApplyEmitters(float dt);
    void MarkObstacles();
    void AdvectFields(float dt);
    void ApplyBuoyancy(float dt);
    void ComputeVorticity();
    void ApplyVorticityConfinement(float dt);
    void ApplyCombustion(float dt);
    void ComputeDivergence();
    void PressureSolve();
    void ApplyPressureGradient();
    void ApplyDissipation(float dt);
    void UpdateStats();

    // Trilinear interpolation on cell-centered field
    float InterpolateCellField(const std::vector<float>& field,
                               float gx, float gy, float gz) const;

    // Data
    GaseousSystemConfig m_config;
    bool m_initialized = false;
    bool m_gpuEnabled = false;

    uint32_t m_resX = 0, m_resY = 0, m_resZ = 0;
    float m_cellSize = 0.1f;
    float m_invCellSize = 10.0f;

    // Primary grid
    std::vector<GasCell> m_cells;

    // Scratch buffers for advection (ping-pong)
    std::vector<float> m_densityTemp;
    std::vector<float> m_temperatureTemp;
    std::vector<float> m_fuelTemp;
    std::vector<float> m_reactionTemp;
    std::vector<float> m_uTemp, m_vTemp, m_wTemp;
    std::vector<float> m_pressureTemp; // Double-buffer for parallel Jacobi

    // Emitters and obstacles
    std::vector<GasEmitter> m_emitters;
    std::vector<GasObstacle> m_obstacles;

    // Statistics
    GaseousStats m_stats;

    // GPU resources
    VulkanContext* m_vulkanContext = nullptr;
};

} // namespace WulfNet
