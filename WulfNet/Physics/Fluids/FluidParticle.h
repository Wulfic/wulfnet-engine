// =============================================================================
// WulfNet Engine - Fluid Particle Definition
// =============================================================================
// GPU-optimized particle structure for MPM/FLIP fluid simulation.
// Aligned for efficient GPU memory access and cache utilization.
// =============================================================================

#pragma once

#include <cstdint>
#include <cmath>

namespace WulfNet {

// =============================================================================
// Fluid Material Types
// =============================================================================

enum class FluidMaterialType : uint32_t {
    Water = 0,          // Low viscosity, surface tension
    Oil = 1,            // Medium viscosity
    Honey = 2,          // High viscosity
    Mud = 3,            // Non-Newtonian, Drucker-Prager
    Lava = 4,           // High viscosity + temperature
    Blood = 5,          // Non-Newtonian shear-thinning
    Custom = 255
};

// =============================================================================
// Fluid Material Properties
// =============================================================================

struct FluidMaterial {
    FluidMaterialType type = FluidMaterialType::Water;

    float density = 1000.0f;            // kg/m³ (water = 1000)
    float viscosity = 0.001f;           // Pa·s (water ≈ 0.001, honey ≈ 10)
    float surfaceTension = 0.072f;      // N/m (water ≈ 0.072)
    float stiffness = 50000.0f;         // Pressure stiffness
    float restDensity = 1000.0f;        // Rest density for pressure

    // Temperature properties (for lava, etc.)
    float thermalConductivity = 0.6f;   // W/(m·K)
    float specificHeat = 4186.0f;       // J/(kg·K)

    // Rendering
    float refractionIndex = 1.33f;      // Water ≈ 1.33
    float transparency = 0.9f;
    uint32_t color = 0x4080C0FF;        // RGBA

    // Presets
    static FluidMaterial Water() {
        FluidMaterial m;
        m.type = FluidMaterialType::Water;
        m.density = 1000.0f;
        m.viscosity = 0.001f;
        m.surfaceTension = 0.072f;
        m.color = 0x4080C0FF;
        return m;
    }

    static FluidMaterial Oil() {
        FluidMaterial m;
        m.type = FluidMaterialType::Oil;
        m.density = 900.0f;
        m.viscosity = 0.1f;
        m.surfaceTension = 0.032f;
        m.color = 0x40300080;
        return m;
    }

    static FluidMaterial Honey() {
        FluidMaterial m;
        m.type = FluidMaterialType::Honey;
        m.density = 1400.0f;
        m.viscosity = 10.0f;
        m.surfaceTension = 0.05f;
        m.color = 0xE0A020FF;
        return m;
    }

    static FluidMaterial Mud() {
        FluidMaterial m;
        m.type = FluidMaterialType::Mud;
        m.density = 1800.0f;
        m.viscosity = 1.0f;
        m.surfaceTension = 0.0f;
        m.color = 0x604020FF;
        return m;
    }

    static FluidMaterial Lava() {
        FluidMaterial m;
        m.type = FluidMaterialType::Lava;
        m.density = 2800.0f;
        m.viscosity = 100.0f;
        m.surfaceTension = 0.4f;
        m.thermalConductivity = 2.0f;
        m.color = 0xFF4000FF;
        return m;
    }
};

// =============================================================================
// Fluid Particle (GPU-aligned, 64 bytes)
// =============================================================================

struct alignas(16) FluidParticle {
    // Position (16 bytes)
    float x, y, z;
    float mass;

    // Velocity (16 bytes)
    float vx, vy, vz;
    float density;

    // APIC affine momentum - simplified (16 bytes)
    // Store only diagonal + one off-diagonal for basic transfer
    float C00, C11;  // Diagonal
    uint32_t materialId;
    uint32_t flags;

    // Properties (16 bytes)
    float temperature;
    float pressure;
    float _padding[2];
};

static_assert(sizeof(FluidParticle) == 64, "FluidParticle must be 64 bytes for GPU alignment");

// =============================================================================
// Particle Flags
// =============================================================================

enum class ParticleFlags : uint32_t {
    None = 0,
    Active = 1 << 0,
    Surface = 1 << 1,       // On fluid surface
    Boundary = 1 << 2,      // Near boundary
    Sleeping = 1 << 3,      // Optimized out (not moving)
    Spray = 1 << 4,         // Detached spray particle
    Foam = 1 << 5,          // Foam particle
    Bubble = 1 << 6,        // Trapped air bubble
    Ghost = 1 << 7,         // Ghost particle for boundary
};

inline ParticleFlags operator|(ParticleFlags a, ParticleFlags b) {
    return static_cast<ParticleFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline ParticleFlags operator&(ParticleFlags a, ParticleFlags b) {
    return static_cast<ParticleFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline bool HasFlag(uint32_t value, ParticleFlags flag) {
    return (value & static_cast<uint32_t>(flag)) != 0;
}

// =============================================================================
// Grid Cell (for MPM/FLIP transfer)
// =============================================================================

struct alignas(16) GridCell {
    // Velocity
    float vx, vy, vz;
    float mass;

    // Momentum
    float px, py, pz;
    float density;

    // Pressure solve
    float pressure;
    float divergence;
    uint32_t particleCount;
    uint32_t flags;
};

static_assert(sizeof(GridCell) == 48, "GridCell should be 48 bytes");

} // namespace WulfNet
