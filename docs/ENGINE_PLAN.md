# WulfNet Engine - Technical Architecture Plan

## Executive Summary

WulfNet Engine is a **fully-featured, AAA-grade physics and game engine** designed for modern multi-core hardware with emphasis on **massive parallelization** (scaling to 64+ cores / 128+ threads), GPU acceleration, SIMD optimization, and data-oriented design. The engine delivers **consistent 60 FPS** across all physics simulations and supports the complete spectrum of physics types:

- **Rigid Body Dynamics** - Stacking, constraints, joints, motors
- **Soft Body Physics** - Cloth, fabrics, flesh, rubber
- **Fluid Dynamics** - Water, oil, lava, viscous fluids
- **Gaseous Simulation** - Smoke, fire, explosions, clouds
- **Deformable Terrain** - Mud, sand, snow, soft ground
- **Granular Materials** - Dirt, gravel, debris
- **Destruction Physics** - Fracture, fragmentation, demolition
- **Vehicle Physics** - Wheeled, tracked, hovering, aircraft
- **Character Physics** - Ragdolls, inverse kinematics, procedural animation
- **Rope & Cable Physics** - Chains, wires, cables, hair
- **Buoyancy & Aerodynamics** - Floating objects, wind effects
- **Physically-Based Lighting** - Ray-traced GI, caustics, volumetrics
- **Acoustic Simulation** - Reverb, occlusion, spatial audio

---

## Table of Contents

1. [Core Architecture](#1-core-architecture)
2. [Physics System Design](#2-physics-system-design)
3. [Rendering & Lighting System](#3-rendering--lighting-system)
4. [Audio & Acoustics System](#4-audio--acoustics-system)
5. [Optimization Strategies](#5-optimization-strategies)
6. [Technology Stack](#6-technology-stack)
7. [Implementation Phases](#7-implementation-phases)
8. [Directory Structure](#8-directory-structure)

---

## 1. Core Architecture

### 1.1 Design Philosophy

| Principle | Description |
|-----------|-------------|
| **Data-Oriented Design (DOD)** | Maximize cache efficiency with Structure of Arrays (SoA) over Array of Structures (AoS) |
| **Entity Component System (ECS)** | Decouple data from behavior for parallelization and cache coherence |
| **Zero-Cost Abstractions** | Compile-time polymorphism over runtime virtual dispatch |
| **GPU-First Compute** | Offload physics calculations to GPU compute shaders |

### 1.2 Core Systems Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Game Logic  │  Scripting (Lua/C#)  │  Editor Tools  │  Asset Pipeline      │
├─────────────────────────────────────────────────────────────────────────────┤
│                              ENGINE LAYER                                    │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│   Physics    │   Renderer   │    Audio     │   Scene      │   Resource      │
│   System     │   System     │   System     │   Graph      │   Manager       │
├──────────────┴──────────────┴──────────────┴──────────────┴─────────────────┤
│                              CORE LAYER                                      │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│    Memory    │     Job      │    Math      │   Platform   │    Profiling    │
│   Allocators │   System     │   Library    │ Abstraction  │    & Debug      │
├──────────────┴──────────────┴──────────────┴──────────────┴─────────────────┤
│                              PLATFORM LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Windows  │  Linux  │  Vulkan/DX12  │  CUDA/OpenCL  │  Audio APIs           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Memory Management

```cpp
// Custom allocator hierarchy
├── SystemAllocator         // OS-level allocations
├── LinearAllocator         // Frame-based temporary allocations
├── PoolAllocator           // Fixed-size object pools (particles, entities)
├── StackAllocator          // LIFO allocations for hierarchical data
└── BuddyAllocator          // GPU memory management
```

**Key Features:**
- Pre-allocated memory pools to avoid runtime allocations
- NUMA-aware allocation for multi-socket systems
- GPU memory pooling with async transfer management
- Memory defragmentation for long-running simulations

### 1.4 Massively Parallel Job System (Multi-threading)

The job system is designed to **scale linearly to 128+ hardware threads** with minimal contention.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MASSIVELY PARALLEL JOB SCHEDULER                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                         THREAD POOL (N = Hardware Threads)                   ││
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐         ┌───────┐ ┌───────┐        ││
│  │  │ W[0]  │ │ W[1]  │ │ W[2]  │ │ W[3]  │   ...   │W[126] │ │W[127] │        ││
│  │  │ Core0 │ │ Core0 │ │ Core1 │ │ Core1 │         │Core63 │ │Core63 │        ││
│  │  │ SMT0  │ │ SMT1  │ │ SMT0  │ │ SMT1  │         │ SMT0  │ │ SMT1  │        ││
│  │  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘         └───┬───┘ └───┬───┘        ││
│  │      │         │         │         │                 │         │            ││
│  │      ▼         ▼         ▼         ▼                 ▼         ▼            ││
│  │  ┌───────────────────────────────────────────────────────────────────┐      ││
│  │  │              Per-Thread Lock-Free Work Queues                     │      ││
│  │  │   Queue[0]  Queue[1]  Queue[2]  ...  Queue[126]  Queue[127]       │      ││
│  │  └───────────────────────────────────────────────────────────────────┘      ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   CRITICAL   │  │     HIGH     │  │    NORMAL    │  │  BACKGROUND  │         │
│  │   Priority   │  │   Priority   │  │   Priority   │  │   Priority   │         │
│  │  (Physics)   │  │  (Rendering) │  │   (Audio)    │  │   (I/O)      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                         JOB DEPENDENCY GRAPH (DAG)                          ││
│  │   Broadphase ──► Narrowphase ──► Solver ──► Integration ──► Render Sync     ││
│  │        │              │            │             │                           ││
│  │        └──────────────┴────────────┴─────────────┘                           ││
│  │                    (Parallel Islands / Batches)                              ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### 1.4.1 Core Job System Features

| Feature | Implementation | Benefit |
|---------|----------------|--------|
| **Lock-Free Work Stealing** | Chase-Lev deque per thread | Zero contention on push, minimal on steal |
| **Fiber-Based Coroutines** | Lightweight 64KB stack fibers | Suspend/resume without OS overhead |
| **NUMA Awareness** | Thread-to-core affinity, local memory | 2-3x memory bandwidth on multi-socket |
| **Batch Processing** | SIMD-width job batches (8/16/32) | Amortize scheduling overhead |
| **Continuation Stealing** | Steal pending continuations | Keep caches hot |
| **Adaptive Spinning** | Spin → Yield → Sleep progression | Balance latency vs power |

#### 1.4.2 Parallel Job Patterns

```cpp
// Parallel-For with automatic batch sizing
void parallelFor(uint32_t count, uint32_t minBatchSize, JobFunc func) {
    uint32_t numWorkers = jobSystem.getWorkerCount();  // e.g., 128
    uint32_t batchSize = max(minBatchSize, count / (numWorkers * 4));
    uint32_t numBatches = (count + batchSize - 1) / batchSize;
    
    JobHandle handle = jobSystem.dispatch(numBatches, [=](uint32_t batchIdx) {
        uint32_t start = batchIdx * batchSize;
        uint32_t end = min(start + batchSize, count);
        for (uint32_t i = start; i < end; i++) {
            func(i);
        }
    });
    jobSystem.wait(handle);
}

// Parallel reduction (sum, min, max)
template<typename T>
T parallelReduce(T* data, uint32_t count, T identity, ReduceFunc<T> reduce) {
    // Thread-local accumulators to avoid false sharing
    alignas(64) T localResults[MAX_WORKERS];
    // ... parallel accumulation then tree reduction
}
```

#### 1.4.3 Thread Affinity & NUMA Optimization

```cpp
// Detect and configure for NUMA topology
void initializeJobSystem() {
    NumaTopology numa = queryNumaTopology();
    
    // Example: 2-socket system with 64 cores each (128 threads total)
    // Socket 0: Cores 0-31  (Threads 0-63)
    // Socket 1: Cores 32-63 (Threads 64-127)
    
    for (uint32_t node = 0; node < numa.nodeCount; node++) {
        for (uint32_t core : numa.nodes[node].cores) {
            // Pin worker threads to cores
            Worker* worker = createWorker(core);
            worker->setAffinity(core);
            worker->setPreferredMemoryNode(node);
            
            // Allocate per-thread data on local NUMA node
            worker->localArena = numaAlloc(node, ARENA_SIZE);
        }
    }
}
```

#### 1.4.4 Physics-Specific Parallelization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PARALLEL PHYSICS FRAME (Target: <16.67ms)                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  T=0ms    ┌─────────────────────────────────────────────────────────────────┐   │
│           │ BROADPHASE (All 128 threads)                                     │   │
│           │ - Parallel BVH update (parallel prefix + sort)                   │   │
│           │ - Parallel spatial hashing                                       │   │
│           └─────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│  T=1ms    ┌─────────────────────────▼─────────────────────────────────────────┐ │
│           │ NARROWPHASE (All 128 threads)                                      │ │
│           │ - Parallel collision detection per pair                           │ │
│           │ - Contact manifold generation                                      │ │
│           └───────────────────────────────────────────────────────────────────┘ │
│                                     │                                            │
│  T=3ms    ┌─────────────────────────▼─────────────────────────────────────────┐ │
│           │ ISLAND BUILDING (Single-threaded, fast O(n) union-find)           │ │
│           └───────────────────────────────────────────────────────────────────┘ │
│                                     │                                            │
│  T=3.5ms  ┌─────────────────────────▼─────────────────────────────────────────┐ │
│           │ CONSTRAINT SOLVER (Parallel by Island)                            │ │
│           │                                                                    │ │
│           │   Island 0    Island 1    Island 2    ...    Island N             │ │
│           │   (Thread     (Thread     (Thread           (Thread               │ │
│           │    Group 0)    Group 1)    Group 2)          Group N)             │ │
│           │                                                                    │ │
│           │ - Large islands: Graph-colored parallel Jacobi                    │ │
│           │ - Small islands: Batched sequential TGS                           │ │
│           └───────────────────────────────────────────────────────────────────┘ │
│                                     │                                            │
│  T=8ms    ┌─────────────────────────▼─────────────────────────────────────────┐ │
│           │ INTEGRATION & SUBSTEPS (All 128 threads)                           │ │
│           │ - Position/velocity integration                                    │ │
│           │ - Soft body substeps (parallel constraints)                        │ │
│           │ - Fluid substeps (GPU async compute)                               │ │
│           └───────────────────────────────────────────────────────────────────┘ │
│                                     │                                            │
│  T=12ms   ┌─────────────────────────▼─────────────────────────────────────────┐ │
│           │ SYNC & CALLBACKS (Minimal, event dispatch)                        │ │
│           └───────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  T<16.67ms ✓ Frame Complete                                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Physics System Design

### 2.0 Complete Physics Feature Matrix

| Physics Type | Algorithm | GPU Accelerated | 60 FPS Target | Coupling Support |
|--------------|-----------|-----------------|---------------|------------------|
| **Rigid Bodies** | TGS/PGS + Islands | ✓ (Broadphase) | 25,000 bodies | All |
| **Soft Bodies (Cloth)** | XPBD | ✓ Full | 100K particles | Rigid, Fluid |
| **Soft Bodies (Volumetric)** | FEM/XPBD | ✓ Full | 50K tetrahedra | Rigid |
| **Fluids (Liquid)** | FLIP/APIC + SPH | ✓ Full | 1M particles | Rigid, Soft |
| **Fluids (Viscous)** | SPH + Plasticity | ✓ Full | 500K particles | Rigid |
| **Gaseous (Smoke/Fire)** | Eulerian Grid | ✓ Full | 256³ grid | Rigid, Thermal |
| **Granular (Sand/Dirt)** | MPM Drucker-Prager | ✓ Full | 500K particles | Rigid, Fluid |
| **Snow** | MPM Snow Model | ✓ Full | 500K particles | Rigid |
| **Mud/Wet Soil** | MPM + Saturation | ✓ Full | 300K particles | Fluid |
| **Destruction** | Voronoi Fracture | ✓ Partial | 10K fragments | Rigid |
| **Vehicles (Wheeled)** | Raycast + Pacejka | ✗ CPU | 100 vehicles | Terrain |
| **Vehicles (Tracked)** | Multi-body + Terrain | ✗ CPU | 50 vehicles | Terrain, Mud |
| **Vehicles (Aircraft)** | 6-DOF Aerodynamics | ✗ CPU | 200 aircraft | Fluid (wind) |
| **Ragdolls** | Articulated XPBD | ✓ Partial | 500 characters | Rigid |
| **Ropes/Cables** | XPBD Position-Based | ✓ Full | 50K segments | Rigid, Soft |
| **Hair** | DER (Discrete Elastic Rods) | ✓ Full | 100K strands | Rigid, Fluid |
| **Buoyancy** | Volume Displacement | ✗ CPU | 1000 objects | Fluid |
| **Aerodynamics** | Thin Plate + Lift/Drag | ✗ CPU | 5000 objects | Fluid (wind) |
| **Magnetics** | Dipole Interactions | ✓ Partial | 10K objects | Rigid |
| **Ballistics** | Trajectory + Penetration | ✗ CPU | 10K projectiles | Rigid, Deform |

### 2.1 Physics Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED PHYSICS PIPELINE (60 FPS)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐ │
│  │BROADPHASE │──►│NARROWPHASE│──►│  ISLAND   │──►│  SOLVER   │──►│INTEGRATION│ │
│  │(Parallel) │   │(Parallel) │   │ BUILDER   │   │(Parallel) │   │(Parallel) │ │
│  └───────────┘   └───────────┘   └───────────┘   └───────────┘   └───────────┘ │
│        │               │               │               │               │        │
│        ▼               ▼               ▼               ▼               ▼        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                         PHYSICS SUBSYSTEMS (Parallel)                       ││
│  ├─────────────────────────────────────────────────────────────────────────────┤│
│  │                                                                              ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        ││
│  │  │ RIGID BODIES │ │ SOFT BODIES  │ │   FLUIDS     │ │  DEFORMABLES │        ││
│  │  │  (TGS/PGS)   │ │  (XPBD/FEM)  │ │(FLIP/SPH/Eul)│ │  (MPM/FEM)   │        ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        ││
│  │                                                                              ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        ││
│  │  │  VEHICLES    │ │  RAGDOLLS    │ │ROPES/CABLES  │ │ DESTRUCTION  │        ││
│  │  │(Raycast/MBD) │ │(Articulated) │ │  (XPBD/DER)  │ │  (Voronoi)   │        ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        ││
│  │                                                                              ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        ││
│  │  │  GASEOUS     │ │   BUOYANCY   │ │ AERODYNAMICS │ │  BALLISTICS  │        ││
│  │  │(Smoke/Fire)  │ │ (Archimedes) │ │(Lift/Drag)   │ │ (Projectile) │        ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        ││
│  │                                                                              ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                      │                                           │
│                                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                         COUPLING MANAGER                                     ││
│  │  - Rigid ↔ Fluid (Two-way forces, buoyancy, drag)                           ││
│  │  - Rigid ↔ Soft (Contact, attachment, tearing)                              ││
│  │  - Fluid ↔ Soft (Wet cloth, absorption, dripping)                           ││
│  │  - Fluid ↔ Deformable (Mud saturation, erosion)                             ││
│  │  - Thermal ↔ All (Heat transfer, phase changes)                             ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Broadphase Collision Detection

| Algorithm | Use Case | Complexity |
|-----------|----------|------------|
| **Sweep and Prune (SAP)** | General purpose, coherent motion | O(n log n) |
| **Bounding Volume Hierarchy (BVH)** | Static/semi-static geometry | O(log n) queries |
| **Spatial Hashing** | Uniform particle systems (fluids) | O(1) average |
| **Multi-Level Grid** | Mixed scale objects | Adaptive |

**GPU Acceleration:**
- Parallel radix sort for SAP
- GPU BVH construction with LBVH algorithm
- Compute shader spatial hashing

### 2.3 Rigid Body Dynamics

```cpp
struct RigidBody {
    // Transform (SoA layout in actual implementation)
    Vec3 position;
    Quat orientation;
    Vec3 linearVelocity;
    Vec3 angularVelocity;
    
    // Mass properties
    float inverseMass;
    Mat3 inverseInertiaTensor;
    
    // Material
    float friction;
    float restitution;
};
```

**Constraint Solver:**
- **Temporal Gauss-Seidel (TGS)** - Better convergence for stacking
- **Projected Gauss-Seidel (PGS)** - Faster per-iteration
- **Jacobi solver** for GPU parallelization
- Warm-starting with cached impulses

### 2.4 Fluid Physics System

#### 2.4.1 Hybrid FLIP/APIC Method (Primary)

```
┌─────────────────────────────────────────────────────────────┐
│                    FLIP/APIC Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Particles → Grid (P2G Transfer)                          │
│    - Transfer mass and momentum to MAC grid                 │
│    - APIC: Also transfer affine velocity field              │
│                                                              │
│ 2. Grid Operations                                           │
│    - Apply body forces (gravity, buoyancy)                  │
│    - Solve pressure (Conjugate Gradient / Multigrid)        │
│    - Enforce incompressibility (∇·v = 0)                    │
│                                                              │
│ 3. Grid → Particles (G2P Transfer)                          │
│    - FLIP: Δv = v_grid_new - v_grid_old                     │
│    - PIC/FLIP blend for stability                           │
│                                                              │
│ 4. Particle Advection                                        │
│    - RK2/RK4 integration                                    │
│    - Boundary handling                                       │
└─────────────────────────────────────────────────────────────┘
```

#### 2.4.2 SPH (Smoothed Particle Hydrodynamics) for Real-time

```cpp
// SPH Kernels
float W_poly6(float r, float h);      // Density estimation
Vec3 grad_W_spiky(Vec3 r, float h);   // Pressure forces
float laplacian_W_viscosity(float r, float h); // Viscosity
```

**Features:**
- Neighbor search with spatial hashing (O(1) lookups)
- Surface tension via cohesion forces
- Viscosity (XSPH method)
- Boundary handling with ghost particles
- Adaptive time-stepping (CFL condition)

#### 2.4.3 GPU Implementation

```
Compute Shader Dispatch:
├── cs_fluid_hash_particles.hlsl      (Spatial hashing)
├── cs_fluid_sort_particles.hlsl      (Radix sort by cell)
├── cs_fluid_density_pressure.hlsl    (Compute ρ, p)
├── cs_fluid_forces.hlsl              (Compute accelerations)
├── cs_fluid_integrate.hlsl           (Position/velocity update)
└── cs_fluid_surface.hlsl             (Marching cubes mesh)
```

### 2.5 Deformable Body Physics (Mud, Soft Ground)

#### 2.5.1 Material Point Method (MPM)

**Why MPM for Deformables:**
- Handles topology changes naturally (fracture, separation)
- Perfect for granular materials (mud, sand, snow)
- Unified framework for solids and fluids
- GPU-friendly grid-based computations

```
┌─────────────────────────────────────────────────────────────┐
│                      MPM Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│ Material Models:                                             │
│ ├── Neo-Hookean (elastic rubber)                            │
│ ├── Fixed Corotated (stiff materials)                       │
│ ├── Drucker-Prager (sand, mud, soil)                        │
│ ├── Von Mises (metals, plastic deformation)                 │
│ └── Snow (Disney's MPM snow model)                          │
├─────────────────────────────────────────────────────────────┤
│ Grid Operations:                                             │
│ 1. P2G: Transfer mass, momentum, affine momentum            │
│ 2. Grid update: Forces, boundary conditions                 │
│ 3. G2P: Update particle velocities, deformation gradient    │
│ 4. Particle advection and plasticity                        │
└─────────────────────────────────────────────────────────────┘
```

#### 2.5.2 Terrain Deformation System

```cpp
struct TerrainCell {
    float height;           // Base height
    float deformation;      // Accumulated deformation
    float moisture;         // Affects plasticity
    float compaction;       // How compressed the soil is
    MaterialType material;  // Mud, sand, rock, etc.
};
```

**Deformation Pipeline:**
1. Detect contact with deformable terrain
2. Calculate pressure distribution (Hertzian contact)
3. Apply plastic deformation based on material model
4. Propagate displacement to neighbors (viscoplastic flow)
5. Update terrain mesh/heightmap

### 2.6 Soft Body Physics (Cloth, Fabrics)

#### 2.6.1 Extended Position-Based Dynamics (XPBD)

```cpp
// Constraint types
struct DistanceConstraint {
    uint32_t particleA, particleB;
    float restLength;
    float compliance;  // 1/stiffness (XPBD parameter)
};

struct BendingConstraint {
    uint32_t particles[4];  // Quad for dihedral angle
    float restAngle;
    float compliance;
};

// XPBD Solver loop
for (int iter = 0; iter < numIterations; iter++) {
    for (auto& constraint : constraints) {
        // Compute Δx with compliance-aware correction
        float alpha = compliance / (dt * dt);
        Vec3 correction = computeCorrection(constraint, alpha);
        applyCorrection(particles, correction);
    }
}
```

#### 2.6.2 Cloth Simulation Features

| Feature | Implementation |
|---------|----------------|
| Stretching | Distance constraints (structural springs) |
| Shearing | Diagonal constraints |
| Bending | Dihedral angle constraints / isometric bending |
| Self-collision | Spatial hashing + continuous collision detection |
| Wind | Aerodynamic drag model |
| Friction | Velocity-based constraint modification |
| Tearing | Constraint removal when stress exceeds threshold |

#### 2.6.3 GPU Cloth Solver

```
Parallel Constraint Solving (Graph Coloring):
├── Color 0: Process all constraints in parallel
├── Color 1: Process all constraints in parallel
├── Color 2: Process all constraints in parallel
└── ... (typically 4-8 colors for cloth)
```

### 2.7 Vehicle Physics System

#### 2.7.1 Wheeled Vehicles

```cpp
struct WheelConfig {
    float radius;
    float width;
    float suspensionLength;
    float suspensionStiffness;   // Spring rate (N/m)
    float suspensionDamping;     // Damping coefficient
    float maxSteerAngle;         // For steerable wheels
    TireModel tireModel;         // Pacejka, Brush, or custom
};

struct VehicleState {
    RigidBody* chassis;
    Wheel wheels[MAX_WHEELS];
    float engineTorque;
    float brakeTorque;
    float steerAngle;
    DrivetrainType drivetrain;   // FWD, RWD, AWD
};
```

**Tire Model (Pacejka Magic Formula):**
```
F = D * sin(C * atan(B * slip - E * (B * slip - atan(B * slip))))

Where:
- B = Stiffness factor
- C = Shape factor
- D = Peak force
- E = Curvature factor
- slip = Slip ratio (longitudinal) or slip angle (lateral)
```

#### 2.7.2 Tracked Vehicles (Tanks, Bulldozers)

```cpp
struct TrackConfig {
    float trackWidth;
    float trackLength;
    uint32_t numLinks;           // Track link count
    float linkMass;
    float sprocketRadius;
    float tensionForce;
    std::vector<Vec3> wheelPositions;  // Road wheels, idler, sprocket
};

// Simulation modes:
// 1. Simplified: Single-body with track force model
// 2. Multi-body: Individual link simulation (high fidelity)
// 3. Hybrid: Simplified until track visible, then multi-body
```

#### 2.7.3 Aircraft / 6-DOF Aerodynamics

```cpp
struct AircraftConfig {
    float wingArea;
    float wingSpan;
    float aspectRatio;
    
    // Aerodynamic coefficients (lookup tables vs AoA)
    AeroCoeffTable liftCoeff;    // CL vs alpha
    AeroCoeffTable dragCoeff;    // CD vs alpha  
    AeroCoeffTable momentCoeff;  // CM vs alpha
    
    // Control surfaces
    ControlSurface ailerons;
    ControlSurface elevator;
    ControlSurface rudder;
    ControlSurface flaps;
};

// Force calculation
Vec3 computeAeroForces(AircraftState& state, Vec3 airVelocity) {
    float dynamicPressure = 0.5f * airDensity * dot(airVelocity, airVelocity);
    float alpha = computeAngleOfAttack(state, airVelocity);
    
    float CL = liftCoeff.sample(alpha);
    float CD = dragCoeff.sample(alpha);
    
    Vec3 lift = liftDirection * CL * dynamicPressure * wingArea;
    Vec3 drag = -normalize(airVelocity) * CD * dynamicPressure * wingArea;
    
    return lift + drag;
}
```

### 2.8 Ragdoll & Character Physics

```cpp
struct RagdollConfig {
    // Bone hierarchy
    struct Bone {
        uint32_t parentIndex;
        Capsule collider;
        float mass;
        JointLimits limits;      // Cone limits, twist limits
    };
    std::vector<Bone> bones;
    
    // Motor targets for active ragdoll
    bool useMotors;
    float motorStrength;
    Animation* targetPose;       // Blend between ragdoll and animation
};

// Active ragdoll blending
for (auto& bone : ragdoll.bones) {
    Quat animRot = animation.getBoneRotation(bone.id);
    Quat physRot = bone.body->getOrientation();
    
    // Apply motor torque toward animation pose
    Quat error = animRot * inverse(physRot);
    Vec3 torque = quaternionToAngularVelocity(error) * motorStrength;
    bone.body->applyTorque(torque);
}
```

### 2.9 Rope, Cable & Hair Physics

#### 2.9.1 XPBD Ropes/Cables

```cpp
struct RopeConfig {
    float length;
    uint32_t numSegments;
    float radius;
    float stretchCompliance;     // Low = stiff cable, High = elastic rope
    float bendCompliance;        // Low = rigid rod, High = flexible rope
    float torsionCompliance;     // Twist resistance
};
```

#### 2.9.2 Hair (Discrete Elastic Rods - DER)

```cpp
struct HairStrand {
    std::vector<Vec3> positions;      // Particle positions
    std::vector<Vec3> velocities;
    std::vector<float> restLengths;   // Edge lengths
    std::vector<Quat> materialFrames; // For twist
    std::vector<Vec2> restCurvatures; // Intrinsic curl
};

// GPU hair simulation: One thread-group per strand
// 128 strands per dispatch, 64 particles per strand = 8K particles/dispatch
```

### 2.10 Destruction Physics

```cpp
struct DestructibleConfig {
    Mesh* intactMesh;
    std::vector<Mesh*> fracturePatterns;  // Pre-computed Voronoi cells
    float fractureThreshold;               // Stress to trigger fracture
    float debrisLifetime;
    bool enableSecondaryFracture;          // Fragments can break further
};

// Fracture process:
// 1. Detect stress concentration (contact impulse > threshold)
// 2. Select fracture pattern based on impact location
// 3. Replace intact mesh with pre-fractured pieces
// 4. Apply explosion impulse to fragments
// 5. Each fragment becomes independent rigid body
```

### 2.11 Gaseous Physics (Smoke, Fire, Explosions)

```cpp
struct GasSimulationConfig {
    Vec3i gridResolution;        // e.g., 256³
    float cellSize;
    float temperatureDecay;
    float vorticityStrength;     // Turbulence enhancement
    float buoyancyFactor;
};

// Eulerian solver on GPU:
// 1. Advection (Semi-Lagrangian or MacCormack)
// 2. Apply forces (buoyancy, vorticity confinement)
// 3. Pressure projection (Jacobi iterations on GPU)
// 4. Temperature/density decay
```

### 2.12 Buoyancy & Aerodynamics

```cpp
// Buoyancy using mesh voxelization
Vec3 computeBuoyancy(RigidBody& body, FluidSurface& fluid) {
    float submergedVolume = 0;
    Vec3 submergedCentroid = Vec3(0);
    
    for (auto& voxel : body.voxels) {
        Vec3 worldPos = body.transform * voxel.localPos;
        float depth = fluid.getDepth(worldPos);
        
        if (depth > 0) {
            submergedVolume += voxel.volume;
            submergedCentroid += worldPos * voxel.volume;
        }
    }
    
    submergedCentroid /= submergedVolume;
    Vec3 buoyancyForce = Vec3(0, fluidDensity * gravity * submergedVolume, 0);
    
    body.applyForceAtPoint(buoyancyForce, submergedCentroid);
    return buoyancyForce;
}

// Aerodynamic drag
Vec3 computeDrag(RigidBody& body, Vec3 windVelocity) {
    Vec3 relativeVel = body.linearVelocity - windVelocity;
    float speed = length(relativeVel);
    float dragMagnitude = 0.5f * airDensity * speed * speed * body.dragCoeff * body.crossSection;
    return -normalize(relativeVel) * dragMagnitude;
}
```

### 2.13 Ballistics & Projectile Physics

```cpp
struct Projectile {
    Vec3 position;
    Vec3 velocity;
    float mass;
    float caliber;
    float dragCoeff;
    
    // Penetration model
    float armorPiercing;         // Penetration capability (mm RHA)
    ProjectileType type;         // AP, HEAT, HE, etc.
};

// High-speed projectile integration (substeps for accuracy)
void updateProjectile(Projectile& proj, float dt) {
    const int substeps = 10;  // 10 substeps for 1000+ m/s projectiles
    float subDt = dt / substeps;
    
    for (int i = 0; i < substeps; i++) {
        // Gravity
        proj.velocity.y -= 9.81f * subDt;
        
        // Drag (quadratic with velocity)
        float speed = length(proj.velocity);
        Vec3 drag = -normalize(proj.velocity) * (0.5f * airDensity * speed * speed * proj.dragCoeff);
        proj.velocity += drag / proj.mass * subDt;
        
        // Coriolis effect for long-range
        if (speed > 300.f) {
            proj.velocity += computeCoriolisAcceleration(proj) * subDt;
        }
        
        // Position update with collision check
        Vec3 newPos = proj.position + proj.velocity * subDt;
        if (raycast(proj.position, newPos, hit)) {
            handlePenetration(proj, hit);
            break;
        }
        proj.position = newPos;
    }
}
```

---

## 3. Rendering & Lighting System

### 3.1 Rendering Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RENDER GRAPH                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Pass: Shadow Maps  │  Pass: GBuffer  │  Pass: Lighting  │  Pass: Post      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Cascade   │    │   Albedo    │    │   Direct    │    │     TAA     │   │
│  │   Shadow    │ -> │   Normal    │ -> │   Diffuse   │ -> │    Bloom    │   │
│  │   Maps      │    │   Depth     │    │   Specular  │    │   Tonemap   │   │
│  │             │    │   Material  │    │   Indirect  │    │             │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Physically-Based Lighting

#### 3.2.1 BRDF Model

```cpp
// Cook-Torrance Microfacet BRDF
Vec3 BRDF(Vec3 L, Vec3 V, Vec3 N, Material mat) {
    Vec3 H = normalize(L + V);
    
    // Distribution (GGX/Trowbridge-Reitz)
    float D = D_GGX(N, H, mat.roughness);
    
    // Geometry (Smith's method with GGX)
    float G = G_SmithGGX(N, V, L, mat.roughness);
    
    // Fresnel (Schlick approximation)
    Vec3 F = F_Schlick(dot(H, V), mat.F0);
    
    // Specular
    Vec3 specular = (D * G * F) / (4.0 * dot(N, V) * dot(N, L));
    
    // Diffuse (Lambert with energy conservation)
    Vec3 diffuse = mat.albedo / PI * (1.0 - F);
    
    return diffuse + specular;
}
```

#### 3.2.2 Global Illumination

| Technique | Use Case | Performance |
|-----------|----------|-------------|
| **Ray-Traced GI** | High-end, hardware RT | ~2-4ms @ 1080p |
| **DDGI (Dynamic Diffuse GI)** | Probe-based, real-time | ~1-2ms |
| **Screen-Space GI (SSGI)** | Approximation, low cost | ~0.5-1ms |
| **Voxel Cone Tracing** | Medium quality, consistent | ~2-3ms |
| **Light Propagation Volumes** | Large outdoor scenes | ~1-2ms |

#### 3.2.3 Real-Time Ray Tracing Pipeline

```
┌────────────────────────────────────────────────────────┐
│              RT Pipeline (DXR / Vulkan RT)             │
├────────────────────────────────────────────────────────┤
│ 1. Build/Update TLAS (Top-Level Acceleration Structure)│
│ 2. Trace primary rays (optional for hybrid)            │
│ 3. Trace shadow rays (soft shadows)                    │
│ 4. Trace reflection rays (glossy reflections)          │
│ 5. Trace GI rays (indirect lighting)                   │
│ 6. Denoise (SVGF / NRD / custom temporal)             │
└────────────────────────────────────────────────────────┘
```

### 3.3 Volumetric Rendering

**For fluids, smoke, fog:**

```cpp
// Ray marching through volume
Vec3 raymarchVolume(Ray ray, Volume volume) {
    Vec3 accumulated = Vec3(0);
    float transmittance = 1.0;
    
    for (float t = tMin; t < tMax; t += stepSize) {
        Vec3 pos = ray.origin + ray.dir * t;
        float density = sampleDensity(volume, pos);
        
        if (density > 0) {
            // In-scattering (light contribution)
            Vec3 lighting = computeVolumetricLighting(pos);
            
            // Beer-Lambert absorption
            float absorption = exp(-density * stepSize * extinctionCoeff);
            
            accumulated += transmittance * (1.0 - absorption) * lighting;
            transmittance *= absorption;
            
            if (transmittance < 0.01) break; // Early exit
        }
    }
    return accumulated;
}
```

---

## 4. Audio & Acoustics System

### 4.1 Acoustic Simulation Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ACOUSTIC PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Source    │    │    Ray      │    │    IR       │    │   Spatial   │   │
│  │  Analysis   │ -> │   Tracing   │ -> │  Generation │ -> │   Audio     │   │
│  │             │    │  (Acoustic) │    │             │    │  (HRTF/     │   │
│  │             │    │             │    │             │    │   Ambisonics)│   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Acoustic Ray Tracing

```cpp
struct AcousticRay {
    Vec3 origin;
    Vec3 direction;
    float energy;           // Diminishes with distance and absorption
    float frequency;        // For frequency-dependent effects
    int bounceCount;
    float travelTime;       // For delay calculation
};

struct AcousticMaterial {
    float absorption[NUM_FREQ_BANDS];  // Per-frequency absorption
    float scattering;                   // Diffuse reflection
    float transmission;                 // Through material
};
```

**Ray Tracing Features:**
- Specular and diffuse reflections
- Frequency-dependent absorption
- Diffraction around edges (UTD - Uniform Theory of Diffraction)
- Transmission through materials
- Air absorption (high frequencies)

### 4.3 Impulse Response Generation

```
                    Direct Sound
                         │
                         ▼
    ┌────────────────────┬────────────────────┐
    │ Early Reflections  │  Late Reverberation │
    │ (0-80ms)           │  (80ms+)            │
    │                    │                     │
    │ ░░░░░░░░           │ ░░░░░░░░░░░░░░░░░░░░│
    │   ░░░░░░░          │   ░░░░░░░░░░░░░░░░░ │
    │     ░░░░░░         │     ░░░░░░░░░░░░░░  │
    │       ░░░░░        │       ░░░░░░░░░░░░  │
    │         ░░░░       │         ░░░░░░░░░░  │
    │           ░░░      │           ░░░░░░░░  │
    └────────────────────┴─────────────────────┘
              TIME (ms) ──────────────────────►
```

**Implementation:**
1. **Direct path** - Line-of-sight attenuation
2. **Early reflections** - First 2-3 bounces (image source method)
3. **Late reverb** - Statistical model or ray-traced

### 4.4 Spatial Audio Rendering

| Method | Description | CPU Cost |
|--------|-------------|----------|
| **HRTF** | Head-Related Transfer Function, binaural | Low |
| **Ambisonics** | Spherical harmonics, speaker-agnostic | Medium |
| **Wave Field Synthesis** | Physical reconstruction | High |
| **Object-Based Audio** | Per-object positioning | Low |

```cpp
// HRTF Convolution
void spatializeAudio(AudioBuffer& input, Vec3 sourcePos, Vec3 listenerPos) {
    Vec3 direction = normalize(sourcePos - listenerPos);
    float azimuth = atan2(direction.x, direction.z);
    float elevation = asin(direction.y);
    
    // Lookup or interpolate HRTF filters
    HRTFFilters filters = hrtfDatabase.lookup(azimuth, elevation);
    
    // Convolve with left/right ear filters
    convolve(input, filters.left, output.left);
    convolve(input, filters.right, output.right);
    
    // Apply distance attenuation and air absorption
    applyDistanceModel(output, length(sourcePos - listenerPos));
}
```

### 4.5 Real-Time Acoustic Features

- **Occlusion**: Low-pass filter when sound source is blocked
- **Obstruction**: Partial blocking with diffraction
- **Portals**: Sound propagation through doorways
- **Doppler Effect**: Frequency shift for moving sources
- **Environmental Zones**: Pre-baked reverb for different areas

---

## 5. Optimization Strategies

### 5.1 SIMD Vectorization

```cpp
// Example: Process 8 particles at once with AVX2
void updateParticles_AVX2(ParticleSystem& ps, float dt) {
    __m256 dt_vec = _mm256_set1_ps(dt);
    __m256 gravity = _mm256_set1_ps(-9.81f);
    
    for (size_t i = 0; i < ps.count; i += 8) {
        // Load 8 velocities Y component
        __m256 vy = _mm256_load_ps(&ps.velocityY[i]);
        
        // Apply gravity: vy += gravity * dt
        vy = _mm256_fmadd_ps(gravity, dt_vec, vy);
        
        // Store back
        _mm256_store_ps(&ps.velocityY[i], vy);
        
        // Similar for position update...
    }
}
```

### 5.2 GPU Compute Optimization

| Technique | Benefit |
|-----------|---------|
| Coalesced memory access | 10-100x bandwidth improvement |
| Shared memory tiling | Reduce global memory traffic |
| Warp-level primitives | Fast reductions and scans |
| Persistent threads | Reduce kernel launch overhead |
| Async compute | Overlap physics with rendering |

### 5.3 Cache Optimization

```cpp
// Hot/Cold data splitting
struct ParticleHot {    // Frequently accessed (fits in cache)
    float posX[MAX_PARTICLES];
    float posY[MAX_PARTICLES];
    float posZ[MAX_PARTICLES];
    float velX[MAX_PARTICLES];
    float velY[MAX_PARTICLES];
    float velZ[MAX_PARTICLES];
};

struct ParticleCold {   // Rarely accessed
    float mass[MAX_PARTICLES];
    uint32_t material[MAX_PARTICLES];
    float lifetime[MAX_PARTICLES];
};
```

### 5.4 Level of Detail (LOD) - Maintaining 60 FPS

```
Physics LOD System (All at 60 Hz update rate, reduced fidelity):
├── LOD 0 (0-20m):    Full simulation
│                     - All solver iterations
│                     - Full collision detection
│                     - High particle density
│
├── LOD 1 (20-50m):   Reduced fidelity
│                     - 50% solver iterations
│                     - Simplified collision (spheres/capsules)
│                     - 50% particle density
│
├── LOD 2 (50-100m):  Low fidelity
│                     - 25% solver iterations
│                     - AABB-only collision
│                     - 25% particle density
│                     - Fluids: Particle sprites instead of mesh
│
└── LOD 3 (100m+):    Minimal simulation
                      - Sleep detection aggressive
                      - Animation-driven (no physics solve)
                      - Particle impostors
                      - Cloth: Pre-baked animation
```

**Automatic LOD Transitions:**
- Smooth interpolation during transitions
- Hysteresis to prevent LOD thrashing
- Priority system based on player focus

### 5.5 Temporal Coherence

- **Warm-starting**: Use previous frame's solution as initial guess
- **Incremental BVH updates**: Only rebuild changed portions
- **Persistent spatial hashing**: Reuse hash table structure
- **Motion prediction**: Extrapolate for collision detection

---

## 6. Technology Stack

### 6.1 Languages & APIs

| Component | Technology |
|-----------|------------|
| Core Engine | C++20 (modules, concepts, coroutines) |
| GPU Compute | CUDA / Vulkan Compute / DirectX 12 |
| Rendering | Vulkan 1.3 / DirectX 12 Ultimate |
| Audio | Custom + Platform APIs (WASAPI, ALSA) |
| Scripting | Lua 5.4 / C# (optional) |
| Math | Custom SIMD library + GLM reference |
| Build | CMake + Ninja |

### 6.2 Third-Party Libraries (Optional Integration)

| Library | Purpose | License |
|---------|---------|---------|
| Jolt Physics | Reference/fallback rigid body | MIT |
| OpenVDB | Sparse volume data | MPL 2.0 |
| Tracy | Profiler | BSD-3 |
| FMOD/Wwise | Audio middleware (optional) | Commercial |
| ImGui | Debug UI | MIT |
| Assimp | Asset import | BSD |

### 6.3 Supported Platforms

- **Windows 10/11** (Primary - DX12 + Vulkan)
- **Linux** (Vulkan)
- **Steam Deck** (Vulkan, optimized for APU)
- **Future**: Consoles (PS5, Xbox Series X)

---

## 7. Implementation Phases

### Phase 1: Foundation (Months 1-3)

```
Week 1-4:   Project setup, build system, core types
Week 5-8:   Memory allocators, job system, math library
Week 9-12:  Platform abstraction, window/input, logging
```

**Deliverables:**
- [ ] Cross-platform build system (CMake)
- [ ] Custom allocators (linear, pool, stack)
- [ ] Lock-free job system with work stealing
- [ ] SIMD math library (Vec3, Vec4, Mat4, Quat)
- [ ] Platform abstraction layer
- [ ] Profiling infrastructure (Tracy integration)

### Phase 2: Core Physics (Months 4-7)

```
Week 13-16: Rigid body dynamics, collision detection
Week 17-20: Constraint solver (PGS/TGS)
Week 21-24: Broadphase optimization, GPU acceleration
Week 25-28: Basic soft body (XPBD cloth)
```

**Deliverables:**
- [ ] Rigid body simulation with stacking
- [ ] Collision shapes (sphere, box, capsule, convex hull, mesh)
- [ ] Joint constraints (hinge, ball, fixed, distance)
- [ ] GPU-accelerated broadphase
- [ ] Basic cloth simulation

### Phase 3: Advanced Physics (Months 8-11)

```
Week 29-32: Fluid simulation (SPH/FLIP)
Week 33-36: MPM for deformables (mud, sand)
Week 37-40: Terrain deformation system
Week 41-44: GPU optimization, coupling between systems
```

**Deliverables:**
- [ ] Real-time SPH fluid simulation
- [ ] FLIP/APIC for high-quality fluids
- [ ] MPM solver for granular materials
- [ ] Deformable terrain with tire tracks, footprints
- [ ] Two-way coupling (rigid-fluid, rigid-soft)

### Phase 4: Rendering & Lighting (Months 12-15)

```
Week 45-48: Render graph, GBuffer, PBR materials
Week 49-52: Shadow mapping, direct lighting
Week 53-56: Global illumination (DDGI or SSGI)
Week 57-60: Volumetric rendering for fluids
```

**Deliverables:**
- [ ] Vulkan/DX12 renderer with render graph
- [ ] PBR material system
- [ ] Cascaded shadow maps
- [ ] Screen-space reflections
- [ ] Real-time global illumination
- [ ] Volumetric fluid rendering

### Phase 5: Audio & Acoustics (Months 16-18)

```
Week 61-64: Audio system foundation, mixing
Week 65-68: Acoustic ray tracing, reverb
Week 69-72: Spatial audio (HRTF/Ambisonics)
```

**Deliverables:**
- [ ] Low-latency audio engine
- [ ] Acoustic simulation with ray tracing
- [ ] Dynamic reverb and occlusion
- [ ] Binaural audio / Ambisonics support
- [ ] Doppler and distance attenuation

### Phase 6: Polish & Tools (Months 19-21)

```
Week 73-76: Editor tools, debugging visualizations
Week 77-80: Performance optimization, profiling
Week 81-84: Documentation, examples, testing
```

**Deliverables:**
- [ ] Debug visualization for all physics systems
- [ ] Scene editor with physics preview
- [ ] Comprehensive documentation
- [ ] Example scenes and benchmarks
- [ ] Unit and integration tests

---

## 8. Directory Structure

```
wulfnet-engine/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── ENGINE_PLAN.md          # This document
│   ├── API_REFERENCE.md
│   └── ARCHITECTURE.md
│
├── src/
│   ├── core/
│   │   ├── memory/             # Allocators
│   │   ├── jobs/               # Threading & job system
│   │   ├── math/               # SIMD math library
│   │   ├── containers/         # Custom containers
│   │   └── platform/           # OS abstraction
│   │
│   ├── physics/
│   │   ├── collision/          # Broadphase & narrowphase
│   │   ├── dynamics/           # Rigid body solver
│   │   ├── constraints/        # Joints & contacts
│   │   ├── softbody/           # Cloth, XPBD
│   │   ├── fluids/             # SPH, FLIP, APIC
│   │   ├── deformable/         # MPM, FEM
│   │   └── coupling/           # Inter-system interaction
│   │
│   ├── rendering/
│   │   ├── backend/            # Vulkan/DX12 abstraction
│   │   ├── graph/              # Render graph
│   │   ├── lighting/           # PBR, GI, shadows
│   │   ├── volumetric/         # Fluid & fog rendering
│   │   └── postprocess/        # TAA, bloom, tonemap
│   │
│   ├── audio/
│   │   ├── core/               # Mixer, sources
│   │   ├── acoustics/          # Ray tracing, reverb
│   │   └── spatial/            # HRTF, ambisonics
│   │
│   ├── scene/
│   │   ├── ecs/                # Entity Component System
│   │   ├── graph/              # Scene hierarchy
│   │   └── serialization/      # Save/load
│   │
│   └── editor/                 # Debug tools & UI
│
├── shaders/
│   ├── compute/
│   │   ├── physics/            # GPU physics shaders
│   │   └── acoustics/          # GPU audio processing
│   └── graphics/
│       ├── gbuffer/
│       ├── lighting/
│       └── postprocess/
│
├── assets/
│   ├── materials/
│   ├── meshes/
│   └── audio/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── examples/
│   ├── fluid_demo/
│   ├── cloth_demo/
│   ├── terrain_deformation/
│   └── acoustic_demo/
│
└── tools/
    ├── asset_pipeline/
    └── profiler/
```

---

## 9. Performance Targets (60 FPS Baseline - All Systems)

### 9.1 Core Physics (All @ 60 Hz / 16.67ms Budget)

| System | Target Count | GPU Accelerated | Thread Scaling | Notes |
|--------|--------------|-----------------|----------------|-------|
| **Rigid Bodies** | 25,000 active | ✓ Broadphase | Linear to 64 cores | Island-parallel solver |
| **Cloth Particles** | 100,000 | ✓ Full | N/A (GPU) | 10 XPBD iterations |
| **Volumetric Soft Bodies** | 50,000 tetrahedra | ✓ Full | N/A (GPU) | FEM solver |
| **Fluids (SPH)** | 1,000,000 particles | ✓ Full | N/A (GPU) | Spatial hash neighbor search |
| **Fluids (FLIP)** | 2,000,000 particles | ✓ Full | N/A (GPU) | Sparse grid, multigrid pressure |
| **Smoke/Fire** | 256³ grid | ✓ Full | N/A (GPU) | 3 pressure iterations |
| **MPM (Mud/Sand/Snow)** | 500,000 particles | ✓ Full | N/A (GPU) | APIC transfer |
| **Destruction Fragments** | 10,000 pieces | ✓ Partial | Linear to 32 cores | Pre-fractured Voronoi |

### 9.2 Specialized Physics (All @ 60 Hz)

| System | Target Count | Parallelization | Notes |
|--------|--------------|-----------------|-------|
| **Vehicles (Wheeled)** | 100 simultaneous | 1 thread per vehicle | Raycast suspension |
| **Vehicles (Tracked)** | 50 simultaneous | 2 threads per vehicle | Multi-body tracks |
| **Aircraft** | 200 simultaneous | 1 thread per aircraft | 6-DOF aerodynamics |
| **Ragdolls** | 500 simultaneous | Batched, 128 threads | Active motor blending |
| **Ropes/Cables** | 50,000 segments | ✓ GPU | XPBD parallel colors |
| **Hair Strands** | 100,000 strands | ✓ GPU | DER, 64 particles/strand |
| **Buoyant Objects** | 1,000 objects | Batched, 128 threads | Voxelized intersection |
| **Aerodynamic Objects** | 5,000 objects | Batched, 128 threads | Thin plate model |
| **Projectiles** | 10,000 active | Batched, 128 threads | 10 substeps @ 1000m/s |

### 9.3 Rendering & Audio (All @ 60 Hz)

| System | Target | Notes |
|--------|--------|-------|
| **Draw Calls** | 10,000+ | GPU-driven rendering, indirect draws |
| **Triangles** | 50M+ visible | GPU culling, meshlet rendering |
| **Lights** | 10,000 punctual | Clustered/tiled deferred |
| **Shadows** | 16 shadow-casting lights | Cached shadow maps |
| **Ray-Traced GI** | 1 ray/pixel | Denoised, 4ms budget |
| **Volumetric Fog** | 160x90x64 froxels | 3D ray marching |
| **Acoustic Rays** | 2,048 rays/source | Background thread, async |
| **Audio Sources** | 256 simultaneous | HRTF spatialization |

### 9.4 Thread Utilization Targets

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THREAD UTILIZATION BY CORE COUNT (@ 60 FPS)                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Cores:    4        8       16       32       64      128                       │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  Physics: ████░░░  ███████  ███████████████  █████████████████████████████████  │
│           ~70%     ~90%     ~95%             ~98%     ~99%     ~99%              │
│                                                                                  │
│  Scaling: Linear scaling up to core count with diminishing returns beyond 64    │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                  │
│  Per-System Thread Distribution (128-thread example):                           │
│  ├── Rigid Body Broadphase:     32 threads (parallel sort/hash)                 │
│  ├── Rigid Body Narrowphase:    64 threads (pair processing)                    │
│  ├── Rigid Body Solver:         Variable (island-based, 1-64 per island)        │
│  ├── Vehicle Physics:           4 threads (100 vehicles / ~25 per thread)       │
│  ├── Ragdoll Physics:           8 threads (500 ragdolls / ~63 per thread)       │
│  ├── Ballistics:                4 threads (10K projectiles batched)             │
│  ├── Buoyancy/Aero:             4 threads (6K objects batched)                  │
│  ├── Audio Acoustics:           2 threads (background, async)                   │
│  ├── Scene Graph Update:        8 threads (transform hierarchy)                 │
│  └── Reserved/Overhead:         2 threads (main thread, OS)                     │
│                                                                                  │
│  Note: GPU handles fluids, soft bodies, hair, cloth, smoke independently        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 9.5 Memory Budget

| System | Memory (Typical Scene) | Notes |
|--------|------------------------|-------|
| Rigid Bodies (25K) | ~100 MB | SoA layout |
| Fluid Particles (1M) | ~200 MB | Position, velocity, density |
| MPM Particles (500K) | ~150 MB | + deformation gradient |
| Cloth (100K particles) | ~20 MB | Constraints pre-allocated |
| GPU Buffers | ~1-2 GB | Simulation + rendering |
| **Total Physics** | **~500 MB - 2 GB** | Scalable with scene |

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU driver bugs | Abstract compute backend, fallback to CPU |
| Memory fragmentation | Custom allocators, pre-allocation |
| Physics instability | Robust solver with position clamping |
| Performance regression | Automated benchmarking in CI |
| Scope creep | Strict phase milestones, MVP first |

---

## 11. Next Steps

1. **Initialize repository** with CMake project structure
2. **Implement core systems** (memory, jobs, math)
3. **Create basic rigid body** prototype to validate architecture
4. **Set up GPU compute** pipeline (Vulkan compute or CUDA)
5. **Iterate** based on profiling and benchmarking

---

---

## 12. Appendix: Multi-Core Scaling Benchmarks

### Expected Performance Scaling

| Core Count | Rigid Bodies/Frame | Cloth Particles | Fluid Particles | Overall FPS |
|------------|-------------------|-----------------|-----------------|-------------|
| 4 cores    | 5,000             | GPU-bound       | GPU-bound       | 60 FPS |
| 8 cores    | 10,000            | GPU-bound       | GPU-bound       | 60 FPS |
| 16 cores   | 18,000            | GPU-bound       | GPU-bound       | 60 FPS |
| 32 cores   | 23,000            | GPU-bound       | GPU-bound       | 60 FPS |
| 64 cores   | 25,000            | GPU-bound       | GPU-bound       | 60 FPS |
| 128 threads| 25,000            | GPU-bound       | GPU-bound       | 60 FPS |

*Note: Rigid body count plateaus due to island-based parallelism limits. GPU-accelerated systems (cloth, fluid, soft body) are not CPU-bound.*

### Frame Time Breakdown (64-core system, complex scene)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    FRAME TIME BREAKDOWN (16.67ms Budget)                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CPU Physics         ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  5.0ms       │
│    ├─ Broadphase     ██░░░░░░░░░░                              0.8ms       │
│    ├─ Narrowphase    ████░░░░░░░░                              1.5ms       │
│    ├─ Solver         █████░░░░░░░                              2.0ms       │
│    └─ Integration    █░░░░░░░░░░░                              0.7ms       │
│                                                                             │
│  GPU Physics (Async) ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  4.0ms       │
│    ├─ Fluids         ██████░░░░░                               2.5ms       │
│    ├─ Soft Bodies    ███░░░░░░░░                               1.0ms       │
│    └─ MPM            █░░░░░░░░░░░                              0.5ms       │
│                                                                             │
│  Rendering           ████████████████████░░░░░░░░░░░░░░░░░░░░  8.0ms       │
│    ├─ Shadow Maps    ███░░░░░░░░░                              1.5ms       │
│    ├─ GBuffer        ██░░░░░░░░░░                              1.0ms       │
│    ├─ Lighting + GI  ██████░░░░░░                              2.5ms       │
│    └─ Post Process   ██████░░░░░░                              3.0ms       │
│                                                                             │
│  Audio (Async)       ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.0ms       │
│                                                                             │
│  Overhead            █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.5ms       │
│                                                                             │
│  TOTAL               ██████████████████████████████████████░░ 14.5ms ✓     │
│                      (2.17ms headroom for spikes)                           │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 2.0*  
*Created: February 2026*  
*Last Updated: February 2026*  
*WulfNet Engine Team*
