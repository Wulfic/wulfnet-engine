# WulfNet Engine - Technical Architecture Plan

## Executive Summary

WulfNet Engine is a **fully-featured, AAA-grade physics and game engine** built on top of [Jolt Physics](https://github.com/jrouwe/JoltPhysics), extending it with advanced physics simulations, GPU acceleration, and a complete rendering/audio pipeline. By leveraging Jolt's battle-tested rigid body physics (used in Horizon Forbidden West and Death Stranding 2), we focus development efforts on extending capabilities rather than reinventing solved problems.

**Core Philosophy: Extend, Don't Replace**

| Jolt Physics Provides | WulfNet Engine Adds |
|-----------------------|---------------------|
| Rigid body dynamics, collision detection | GPU-accelerated broadphase & solver |
| Soft bodies (cloth, volumetric) | Fluid dynamics (SPH, FLIP, APIC) |
| Vehicles (wheeled, tracked, motorcycles) | Gaseous simulation (smoke, fire, explosions) |
| Ragdolls with motor-driven animation | MPM deformables (mud, sand, snow) |
| Constraints & joints | Destruction physics (Voronoi fracture) |
| Character controllers | Advanced terrain deformation |
| Hair simulation (GPU) | Physically-based rendering pipeline |
| Buoyancy calculations | Global illumination (ray-traced/DDGI) |
| Deterministic simulation | Acoustic simulation & spatial audio |

The engine delivers **consistent 60 FPS** across all physics simulations with emphasis on **massive parallelization** (scaling to 64+ cores / 128+ threads) and GPU acceleration.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Jolt Physics Integration](#2-jolt-physics-integration)
3. [Extended Physics Systems](#3-extended-physics-systems)
4. [Rendering & Lighting System](#4-rendering--lighting-system)
5. [Audio & Acoustics System](#5-audio--acoustics-system)
6. [Optimization Strategies](#6-optimization-strategies)
7. [Implementation Phases](#7-implementation-phases)
8. [Directory Structure](#8-directory-structure)

---

## 1. Architecture Overview

### 1.1 Design Philosophy

| Principle | Description |
|-----------|-------------|
| **Extend Jolt, Don't Replace** | Use Jolt Physics for rigid/soft body simulation, add new physics types |
| **Data-Oriented Design (DOD)** | Maximize cache efficiency with Structure of Arrays (SoA) |
| **GPU-First Compute** | Offload heavy computations to GPU compute shaders |
| **Modular Integration** | Each WulfNet system integrates cleanly with Jolt's interfaces |
| **Comprehensive Logging** | Extensive logging at all levels for debugging |
| **Test-Driven Development** | JoltViewer extended as primary test environment |

### 1.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Game Logic  │  Scripting (Lua/C#)  │  Editor Tools  │  Asset Pipeline      │
├─────────────────────────────────────────────────────────────────────────────┤
│                           WULFNET ENGINE LAYER                               │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│   Extended   │   Renderer   │    Audio     │   Scene      │   Resource      │
│   Physics    │   (Vulkan)   │   System     │   Graph      │   Manager       │
├──────────────┴──────────────┴──────────────┴──────────────┴─────────────────┤
│                         JOLT PHYSICS FOUNDATION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Rigid Bodies  │  Soft Bodies  │  Vehicles  │  Characters  │  Constraints   │
│  Collision     │  Cloth/Hair   │  Ragdolls  │  Buoyancy    │  Broadphase    │
├─────────────────────────────────────────────────────────────────────────────┤
│                              CORE LAYER                                      │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│    Memory    │     Job      │    Math      │   Platform   │    Profiling    │
│   (Jolt)     │   (Jolt)     │   (Jolt)     │ Abstraction  │    (Tracy)      │
├──────────────┴──────────────┴──────────────┴──────────────┴─────────────────┤
│                              PLATFORM LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Windows  │  Linux  │  macOS  │  Vulkan/DX12  │  CUDA/Compute  │  Audio APIs │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 What Jolt Physics Provides (Already Complete)

Jolt Physics is a production-proven engine with the following capabilities we inherit:

**Core Systems:**
- Multi-threaded job system with work stealing
- Custom memory allocators with tracking
- SIMD-optimized math library (SSE4.2, AVX, AVX2, AVX-512, NEON)
- Platform abstraction (Windows, Linux, macOS, iOS, Android)

**Collision Detection:**
- Broadphase: Quad tree with efficient updates
- Narrowphase: GJK/EPA with feature caching
- Shapes: Sphere, Box, Capsule, Cylinder, Convex Hull, Mesh, HeightField, Compound
- Continuous collision detection (CCD)

**Rigid Body Dynamics:**
- Sequential impulse solver with islands
- All standard constraints (Fixed, Point, Distance, Hinge, Slider, Cone, etc.)
- Motor-driven constraints
- Contact manifold caching

**Soft Body Physics:**
- Position-based dynamics (XPBD)
- Edge, dihedral, volume constraints
- Collision with rigid bodies
- GPU hair simulation

**Specialized Systems:**
- Wheeled, tracked, motorcycle vehicles
- Character controllers (rigid body and virtual)
- Ragdoll animation blending
- Water buoyancy

**Quality Features:**
- Deterministic simulation
- Double precision support
- Extensive unit test suite
- Performance benchmarks

---

## 2. Jolt Physics Integration

### 2.1 Repository Structure

The repository is structured to keep Jolt Physics as an intact foundation:

```
wulfnet-engine/
├── Jolt/                    # Jolt Physics core library (DO NOT MODIFY)
├── JoltViewer/              # Jolt's sample viewer (extend for WulfNet)
├── Samples/                 # Jolt's sample tests (reference & extend)
├── TestFramework/           # Jolt's test framework with renderer
├── UnitTests/               # Jolt's unit tests
├── PerformanceTest/         # Jolt's performance benchmarks
├── HelloWorld/              # Simple Jolt example
├── Build/                   # Platform-specific build scripts
├── Assets/                  # Shared assets (fonts, shaders, models)
│
├── WulfNet/                 # NEW: WulfNet Engine extensions
│   ├── Core/                # Extended core utilities
│   ├── Physics/             # Extended physics (fluids, MPM, etc.)
│   ├── Rendering/           # Vulkan rendering pipeline
│   ├── Audio/               # Acoustic simulation
│   └── Integration/         # Jolt integration layer
│
├── WulfNetViewer/           # NEW: Extended JoltViewer for WulfNet
├── WulfNetTests/            # NEW: WulfNet-specific tests
└── WulfNetExamples/         # NEW: WulfNet example applications
```

### 2.2 Integration Strategy

**Principle: Composition over Modification**

We integrate with Jolt by:
1. **Wrapping** - Create WulfNet wrappers that compose Jolt types
2. **Extending** - Inherit from Jolt base classes where appropriate
3. **Intercepting** - Use Jolt's callback/listener systems
4. **Augmenting** - Add GPU acceleration to Jolt's CPU algorithms

```cpp
// Example: WulfNet physics world wrapping Jolt
namespace WulfNet {

class PhysicsWorld {
public:
    // Initialize with Jolt's physics system
    void Initialize(const PhysicsWorldSettings& settings);
    
    // Step simulation - internally calls Jolt + WulfNet extended physics
    void Step(float deltaTime);
    
    // Access Jolt directly when needed
    JPH::PhysicsSystem& GetJoltPhysics() { return *m_joltPhysics; }
    
    // Extended physics systems
    FluidSystem& GetFluidSystem() { return m_fluidSystem; }
    MPMSystem& GetMPMSystem() { return m_mpmSystem; }
    DestructionSystem& GetDestructionSystem() { return m_destructionSystem; }
    
private:
    // Jolt foundation
    std::unique_ptr<JPH::PhysicsSystem> m_joltPhysics;
    std::unique_ptr<JPH::JobSystemThreadPool> m_jobSystem;
    
    // WulfNet extensions (GPU-accelerated)
    FluidSystem m_fluidSystem;
    MPMSystem m_mpmSystem;
    GaseousSystem m_gaseousSystem;
    DestructionSystem m_destructionSystem;
};

} // namespace WulfNet
```

### 2.3 Coupling Between Jolt and WulfNet Physics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED PHYSICS FRAME                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ PHASE 1: Pre-Simulation (CPU + GPU Async)                               ││
│  │  - GPU: Begin fluid/MPM particle updates (async compute)                ││
│  │  - CPU: Collect coupling forces from previous frame                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ PHASE 2: Jolt Physics Step (CPU, multi-threaded)                        ││
│  │  - Broadphase collision detection                                       ││
│  │  - Narrowphase contact generation                                       ││
│  │  - Constraint solving (rigid + soft bodies)                             ││
│  │  - Position/velocity integration                                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ PHASE 3: WulfNet Extended Physics (GPU primary, CPU secondary)          ││
│  │  - Sync GPU fluid/MPM results                                           ││
│  │  - Fluid ↔ Rigid body coupling (buoyancy, drag)                         ││
│  │  - MPM ↔ Rigid body coupling (terrain deformation)                      ││
│  │  - Destruction trigger evaluation                                       ││
│  │  - Gaseous simulation (smoke, fire)                                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ PHASE 4: Post-Simulation                                                ││
│  │  - Event dispatch (collisions, triggers, destruction)                   ││
│  │  - Render data preparation                                              ││
│  │  - Begin next frame's GPU work (async)                                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Extended Physics Systems

These are the physics types WulfNet adds beyond Jolt's capabilities:

### 3.1 Fluid Dynamics (SPH / FLIP / APIC)

| Feature | Algorithm | GPU Accelerated | Target |
|---------|-----------|-----------------|--------|
| Water/Liquids | FLIP/APIC | ✓ Full | 1M particles |
| Viscous Fluids | SPH + Plasticity | ✓ Full | 500K particles |
| Surface Tension | Cohesion forces | ✓ Full | - |
| Rigid Coupling | Two-way forces | ✓ Partial | - |

```cpp
struct FluidSystem {
    // GPU buffers
    ComputeBuffer<FluidParticle> particles;
    ComputeBuffer<uint32_t> spatialHash;
    ComputeBuffer<GridCell> macGrid;
    
    // Simulation parameters
    FluidMaterial material;      // Viscosity, surface tension
    float timestep;
    uint32_t substeps;
    
    // Coupling with Jolt
    void ApplyBuoyancyToRigidBodies(JPH::PhysicsSystem& jolt);
    void CollideWithRigidBodies(const JPH::BroadPhaseQuery& broadphase);
};
```

### 3.2 Material Point Method (MPM) for Deformables

| Material | Constitutive Model | Target |
|----------|-------------------|--------|
| Mud/Wet Soil | Drucker-Prager + Saturation | 300K particles |
| Sand/Dirt | Drucker-Prager | 500K particles |
| Snow | Disney Snow Model | 500K particles |
| Rubber/Flesh | Neo-Hookean | 50K particles |

### 3.3 Gaseous Simulation

```cpp
struct GaseousSystem {
    // Eulerian grid
    ComputeBuffer<VelocityField> velocityGrid;   // MAC grid
    ComputeBuffer<float> densityGrid;
    ComputeBuffer<float> temperatureGrid;
    
    Vec3i gridResolution;        // e.g., 256³
    float cellSize;
    
    // Simulation
    void Advect(float dt);
    void ApplyForces(float dt);  // Buoyancy, vorticity confinement
    void Project();              // Pressure solve
};
```

### 3.4 Destruction Physics

Extends Jolt's rigid body system with:
- Pre-fractured Voronoi patterns
- Stress-based fracture triggering
- Fragment generation as new Jolt rigid bodies
- Secondary fracture for fragments

```cpp
class DestructibleBody {
    JPH::BodyID m_intactBody;
    std::vector<FracturePattern> m_patterns;
    float m_fractureThreshold;
    
    // On destruction, creates new bodies in Jolt
    void Fracture(JPH::PhysicsSystem& jolt, const Vec3& impactPoint, float impulse);
};
```

### 3.5 Advanced Terrain Deformation

Integrates with Jolt's HeightField shape:
- Runtime heightfield modification
- MPM-driven plastic deformation
- Tire tracks, footprints, craters
- Material-based deformation response

---

## 4. Rendering & Lighting System

### 4.1 Renderer Architecture

Built on Vulkan 1.3 with optional DX12 backend:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RENDER GRAPH                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Shadow Pass  │  GBuffer Pass  │  Lighting Pass  │  Volumetric  │  Post     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Cascade    │  │   Albedo    │  │   Direct    │  │   Fluid     │         │
│  │  Shadows    │->│   Normal    │->│   Lighting  │->│   Volume    │         │
│  │             │  │   Material  │  │   + GI      │  │   Render    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Integration with Jolt's Debug Renderer

Jolt provides a debug renderer in TestFramework. WulfNet extends this:

```cpp
// Extend Jolt's DebugRenderer for production rendering
class WulfNetRenderer : public JPH::DebugRenderer {
public:
    // Override Jolt's debug rendering with PBR pipeline
    void DrawTriangle(Vec3 v1, Vec3 v2, Vec3 v3, ColorArg color) override;
    void DrawLine(Vec3 from, Vec3 to, ColorArg color) override;
    
    // Extended WulfNet rendering
    void RenderFluidSurface(const FluidSystem& fluid);
    void RenderVolumetric(const GaseousSystem& gas);
    void RenderMPMParticles(const MPMSystem& mpm);
};
```

---

## 5. Audio & Acoustics System

### 5.1 Acoustic Simulation

- Ray-traced reverb using Jolt's collision queries
- Occlusion/obstruction using Jolt's ray casting
- Material-based absorption (query shape materials)
- HRTF/Ambisonics spatial rendering

```cpp
class AcousticSystem {
    // Uses Jolt's broadphase for occlusion queries
    void ComputeOcclusion(
        JPH::PhysicsSystem& jolt,
        const Vec3& listener,
        const Vec3& source,
        float& occlusion,
        float& obstruction
    );
    
    // Ray-traced impulse response
    ImpulseResponse TraceAcousticRays(
        JPH::PhysicsSystem& jolt,
        const Vec3& source,
        uint32_t numRays
    );
};
```

---

## 6. Optimization Strategies

### 6.1 GPU Acceleration Layer

WulfNet adds GPU compute acceleration for:
- Fluid simulation (entirely GPU)
- MPM simulation (entirely GPU)
- Broadphase acceleration (parallel to CPU)
- Particle rendering (GPU instancing)

### 6.2 Leveraging Jolt's Optimizations

Jolt already provides:
- SIMD-optimized math (Vec4, Mat44, Quat)
- Cache-friendly data layouts
- Multithreaded job system
- Island-based parallelism

WulfNet extends with:
- Async compute overlap
- GPU-CPU data streaming
- Predictive physics (for networking)

---

## 7. Implementation Phases

### Phase 1: Foundation & Integration ✅ COMPLETE (Via Jolt)

Jolt Physics provides:
- [x] Multi-threaded job system
- [x] Memory allocators
- [x] SIMD math library
- [x] Platform abstraction
- [x] Core physics types
- [x] Rigid body dynamics
- [x] Soft body physics
- [x] Vehicle physics
- [x] Character controllers
- [x] Ragdolls
- [x] Collision detection

### Phase 2: WulfNet Core Setup (Weeks 1-4) 🚧 CURRENT

```
Week 1-2:   Set up WulfNet/ directory structure
            Create CMake integration with Jolt
            Set up build configurations
            
Week 3-4:   Create WulfNet::PhysicsWorld wrapper
            Integrate Tracy profiler
            Set up logging infrastructure
```

**Deliverables:**
- [ ] WulfNet directory structure created
- [ ] CMake properly builds Jolt + WulfNet
- [ ] Basic Jolt wrapper (PhysicsWorld)
- [ ] Tracy profiler integration
- [ ] Logging system

### Phase 3: GPU Compute Foundation (Weeks 5-8)

```
Week 5-6:   Vulkan compute context setup
            Compute shader compilation pipeline
            
Week 7-8:   GPU memory management
            CPU-GPU synchronization utilities
            Basic compute shader tests
```

**Deliverables:**
- [ ] Vulkan compute context
- [ ] Shader compilation (HLSL → SPIR-V)
- [ ] GPU buffer management
- [ ] Async compute helpers

### Phase 4: Fluid Physics (Weeks 9-16)

```
Week 9-12:  SPH implementation (GPU)
            Neighbor search (spatial hashing)
            Basic fluid rendering
            
Week 13-16: FLIP/APIC solver
            Fluid ↔ Rigid body coupling
            Surface extraction (marching cubes)
```

**Deliverables:**
- [ ] GPU SPH solver
- [ ] FLIP/APIC solver  
- [ ] Two-way rigid body coupling
- [ ] Fluid surface mesh generation

### Phase 5: MPM Deformables (Weeks 17-24)

```
Week 17-20: MPM framework (P2G, G2P)
            Drucker-Prager material (sand/mud)
            
Week 21-24: Terrain deformation integration
            MPM ↔ Jolt rigid body coupling
            Snow/ice materials
```

**Deliverables:**
- [ ] GPU MPM solver
- [ ] Sand, mud, snow materials
- [ ] Terrain deformation system
- [ ] Rigid body interaction

### Phase 6: Extended Physics (Weeks 25-32)

```
Week 25-28: Gaseous simulation (Eulerian grid)
            Smoke/fire rendering
            
Week 29-32: Destruction system
            Voronoi pre-fracture
            Fragment physics via Jolt
```

**Deliverables:**
- [ ] Smoke/fire simulation
- [ ] Volumetric rendering
- [ ] Destruction physics
- [ ] Pre-fractured assets

### Phase 7: Rendering Pipeline (Weeks 33-40)

```
Week 33-36: Vulkan renderer foundation
            GBuffer, PBR materials
            
Week 37-40: Shadow mapping
            Global illumination (SSGI/DDGI)
            Volumetric effects
```

**Deliverables:**
- [ ] Vulkan rendering pipeline
- [ ] PBR material system
- [ ] Shadow mapping
- [ ] Global illumination
- [ ] Fluid/gas volumetric rendering

### Phase 8: Audio & Polish (Weeks 41-48)

```
Week 41-44: Audio system foundation
            Acoustic ray tracing
            Spatial audio
            
Week 45-48: Integration testing
            Performance optimization
            Documentation
```

**Deliverables:**
- [ ] Audio engine
- [ ] Acoustic simulation
- [ ] HRTF/Ambisonics
- [ ] Performance benchmarks
- [ ] API documentation

---

## 8. Directory Structure

```
wulfnet-engine/
├── CMakeLists.txt              # Root CMake (builds Jolt + WulfNet)
├── README.md                   # Project overview
├── ENGINE_PLAN.md              # This document
│
├── Jolt/                       # Jolt Physics library (UPSTREAM - minimal changes)
│   ├── AABBTree/
│   ├── Core/
│   ├── Geometry/
│   ├── Math/
│   ├── Physics/
│   ├── Skeleton/
│   └── ...
│
├── JoltViewer/                 # Jolt's viewer (use for reference/testing)
├── Samples/                    # Jolt's samples (use for reference)
├── TestFramework/              # Jolt's test framework (rendering utilities)
├── UnitTests/                  # Jolt's unit tests
├── PerformanceTest/            # Jolt's benchmarks
├── HelloWorld/                 # Simple Jolt example
│
├── WulfNet/                    # WulfNet Engine extensions
│   ├── Core/                   # Extended utilities
│   │   ├── Logging/            # Logging infrastructure
│   │   ├── Profiling/          # Tracy integration
│   │   └── Platform/           # Additional platform utilities
│   │
│   ├── Physics/                # Extended physics systems
│   │   ├── Fluids/             # SPH, FLIP, APIC
│   │   ├── MPM/                # Material Point Method
│   │   ├── Gaseous/            # Smoke, fire, explosions
│   │   ├── Destruction/        # Fracture physics
│   │   ├── Terrain/            # Deformable terrain
│   │   └── Integration/        # Jolt integration layer
│   │
│   ├── Compute/                # GPU compute infrastructure
│   │   ├── Vulkan/             # Vulkan compute backend
│   │   ├── Shaders/            # Compute shaders (HLSL)
│   │   └── Memory/             # GPU memory management
│   │
│   ├── Rendering/              # Rendering pipeline
│   │   ├── Backend/            # Vulkan abstraction
│   │   ├── Pipeline/           # Render passes
│   │   ├── Materials/          # PBR materials
│   │   └── Effects/            # Volumetrics, post-process
│   │
│   └── Audio/                  # Audio & acoustics
│       ├── Core/               # Mixer, sources
│       ├── Acoustics/          # Ray-traced reverb
│       └── Spatial/            # HRTF, Ambisonics
│
├── WulfNetViewer/              # Extended viewer application
├── WulfNetTests/               # WulfNet-specific tests
├── WulfNetExamples/            # Example applications
│
├── Build/                      # Platform-specific build scripts
├── Assets/                     # Shared assets
│   ├── Shaders/                # Graphics & compute shaders
│   ├── Fonts/
│   └── Models/
│
└── docs/                       # Documentation
    ├── Architecture.md
    ├── APIReference.md
    └── Images/
```

---

## 9. Performance Targets

### 9.1 Combined System Targets (60 FPS)

| System | Source | Target Count | GPU Accelerated |
|--------|--------|--------------|-----------------|
| Rigid Bodies | Jolt | 25,000 active | ✗ CPU (Jolt) |
| Soft Bodies | Jolt | 100,000 particles | ✗ CPU (Jolt) |
| Hair Strands | Jolt | 100,000 strands | ✓ GPU (Jolt) |
| Vehicles | Jolt | 100 wheeled | ✗ CPU (Jolt) |
| Ragdolls | Jolt | 500 active | ✗ CPU (Jolt) |
| **Fluid Particles** | WulfNet | 1,000,000 | ✓ GPU |
| **MPM Particles** | WulfNet | 500,000 | ✓ GPU |
| **Smoke/Fire Grid** | WulfNet | 256³ | ✓ GPU |
| **Destruction Fragments** | WulfNet+Jolt | 10,000 | Partial |

### 9.2 Frame Time Budget (16.67ms)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    FRAME TIME BREAKDOWN (16.67ms Budget)                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Jolt Physics (CPU)  ████████████░░░░░░░░░░░░░░░░░░░░░░░░  5.0ms           │
│    ├─ Broadphase     ██░░░░░░░░░                           0.8ms           │
│    ├─ Narrowphase    ███░░░░░░░░                           1.2ms           │
│    ├─ Solver         ████░░░░░░░                           1.8ms           │
│    └─ Integration    ███░░░░░░░░                           1.2ms           │
│                                                                             │
│  WulfNet GPU Physics ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░  4.0ms           │
│    ├─ Fluids         ██████░░░░                            2.5ms           │
│    ├─ MPM            ██░░░░░░░░░                           1.0ms           │
│    └─ Gaseous        █░░░░░░░░░░                           0.5ms           │
│                                                                             │
│  Rendering           ████████████████░░░░░░░░░░░░░░░░░░░░  6.5ms           │
│                                                                             │
│  Audio + Overhead    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.0ms           │
│                                                                             │
│  TOTAL               ████████████████████████████████████ 16.5ms ✓         │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Getting Started

### 10.1 Building the Project

```bash
# Clone the repository
git clone https://github.com/your-org/wulfnet-engine.git
cd wulfnet-engine

# Windows (Visual Studio 2022)
cd Build
cmake_vs2022_cl.bat
# Open VS2022/WulfNetEngine.sln

# Linux
cd Build
./cmake_linux_clang_gcc.sh Release clang++
cd Linux_Release
make -j$(nproc)
```

### 10.2 Running Samples

```bash
# Run JoltViewer (existing Jolt samples)
./bin/JoltViewer

# Run WulfNet extended viewer (when implemented)
./bin/WulfNetViewer
```

---

## 11. Contributing

Key principles:
1. **Don't modify Jolt/** - Keep upstream changes minimal for easy updates
2. **GPU-first for new physics** - Use compute shaders for heavy workloads
3. **Comprehensive testing** - Unit tests for all new systems
4. **Document as you go** - Update docs with each feature

---

## 12. Jolt Physics Upstream Sync

To update Jolt Physics to the latest version:

```bash
# Add Jolt as upstream remote (one-time)
git remote add jolt-upstream https://github.com/jrouwe/JoltPhysics.git

# Fetch latest changes
git fetch jolt-upstream

# Merge updates (resolve conflicts carefully)
git merge jolt-upstream/master --allow-unrelated-histories

# Or cherry-pick specific commits
git cherry-pick <commit-hash>
```

**Important:** Review all merge conflicts carefully. WulfNet modifications to Jolt files should be minimal and well-documented.

---

*Document Version: 4.0*  
*Created: February 2026*  
*Last Updated: February 2026*  
*WulfNet Engine Team*
