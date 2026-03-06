# WulfNet Engine — Master Refactor & Integration Plan

> **Goal:** Transform the current collection of independently-developed modules into **one coherent engine** with a clean, pluggable public API that consumers can initialize with a few lines of code and extend via well-defined interfaces. **Maximize performance** by fully utilizing GPU compute (async), CPU multithreading (OpenMP / thread pool / Jolt jobs), SIMD, and data-oriented memory layouts.

**Created:** March 6, 2026  
**Status:** PLANNING  
**Pre-conditions:** 574 tests passing, all 8 phases of ENGINE_PLAN.md complete

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Target Architecture](#2-target-architecture)
3. [Refactor Passes](#3-refactor-passes)
   - [Pass 1 — Unified Math & Common Types](#pass-1--unified-math--common-types)
   - [Pass 2 — Namespace & Directory Normalization](#pass-2--namespace--directory-normalization)
   - [Pass 3 — Engine Core (Init, Loop, Shutdown)](#pass-3--engine-core-init-loop-shutdown)
   - [Pass 4 — Public API Layer](#pass-4--public-api-layer)
   - [Pass 5 — Wire Up PhysicsWorld ↔ Subsystems](#pass-5--wire-up-physicsworld--subsystems)
   - [Pass 6 — Rendering Abstraction](#pass-6--rendering-abstraction)
   - [Pass 7 — CMake & Build System Overhaul](#pass-7--cmake--build-system-overhaul)
   - [Pass 8 — Tests, Examples, & Documentation](#pass-8--tests-examples--documentation)
   - [Pass 9 — CPU Parallelism & Threading](#pass-9--cpu-parallelism--threading)
   - [Pass 10 — GPU Async Compute & SIMD](#pass-10--gpu-async-compute--simd)
4. [Performance Audit — Current State](#4-performance-audit--current-state)
5. [Migration Rules](#5-migration-rules)
6. [Verification Checklist](#6-verification-checklist)

---

## 1. Current State Assessment

### What We Have (Good)
- **6 major subsystems** fully implemented: Core, Physics (5 sub-modules), Compute, Procedural, Rendering, Audio
- **574 tests** across 22 test files — all passing
- **4 working examples** (HelloWulfNet, ComputeExample, IFSExample, SoftRasterExample)
- **Jolt Physics** cleanly preserved as upstream foundation
- Comprehensive per-module API docs in `docs/APIReference.md`

### What Needs Fixing (Problems)

| # | Problem | Impact |
|---|---------|--------|
| P1 | **No unified engine init/loop/shutdown** — each module manages its own lifecycle | Consumers must manually init 8+ systems in the correct order |
| P2 | **4 separate math/vector types** — `Mat3`(MPM), `GPUMat4x4`(IFS), `SoftVec3`(Rendering), `Vec3`(IFS) | Code duplication, can't pass data between modules |
| P3 | **PhysicsWorld doesn't wire up its subsystems** — `GetMPMSystem()`, `GetGaseousSystem()`, etc. are commented out stubs | The "integration layer" doesn't actually integrate |
| P4 | **Namespace inconsistency** — `WaterSystemV3` uses `WulfNet::Physics::` while everything else uses flat `WulfNet::` | API confusion, breaks auto-complete expectations |
| P5 | **Rendering directory lies** — Shadows, GI, Volumetrics, RenderPipeline all stuffed under `SoftwareRasterizer/` | Misleading structure; shadows ≠ software rasterizer |
| P6 | **No public API header** — `WulfNet.h` is an umbrella include, not an API contract | No versioning, no stability guarantees, no clear entry point |
| P7 | **Root CMakeLists.txt is dead** — real builds go through `build/CMakeLists.txt` via Jolt's scripts | Confusing for new developers; `cmake ..` from root does nothing useful |
| P8 | **Examples call APIs that don't exist** — `Logger::Initialize()`, `Logger::SetMinLevel()` static methods vs singleton pattern | Won't compile; incorrect examples erode trust |
| P9 | **No cross-module data flow** — Rendering can't read physics state, audio can't query scene geometry without manual plumbing | Modules are silos, not an engine |
| P10 | **No configuration/settings system** — each module has its own `*Config` struct with no centralized defaults | Consumers must configure 10+ structs manually |
| P11 | **Gaseous system is 100% serial** — 10+ solver passes over 3D grids with zero threading | 64³ grid = 262K cells × 10 passes per step, all single-threaded |
| P12 | **All rendering pixel passes are single-threaded** — lighting, SSAO, shadows, volumetrics | Full-screen per-pixel work (the heaviest cost) runs on 1 core |
| P13 | **GPU compute is fully synchronous** — every dispatch blocks CPU via `vkQueueWaitIdle()` | Zero CPU/GPU overlap; CPU idles while GPU works |
| P14 | **Per-frame thread creation** — `SoftwareRasterizer` spawns/joins N threads every frame | OS thread creation overhead each frame instead of persistent pool |
| P15 | **OpenMP linked but completely unused** — `omp.h` included, zero `#pragma omp` directives | CMake pays the link cost but no parallelism benefit |
| P16 | **SIMD used only for GBuffer clear** — no SIMD in rasterization, lighting, SSAO, physics | Leaves 4-8× throughput on the table for vectorizable inner loops |
| P17 | **AoS memory layout in GaseousSystem** — `GasCell` packs 10+ fields in one struct | Cache-hostile: each solver pass touches only 1-2 fields but loads entire 64-byte struct |
| P18 | **No async GPU readback** — `RequestAsyncReadback()` is a stub; all downloads block | CPU stalls on every GPU→CPU transfer |
| P19 | **Per-dispatch command buffer alloc/free** — new `VkCommandBuffer` each dispatch | Vulkan driver overhead; should use command buffer pool/ring |
| P20 | **Jolt's thread pool unused for WulfNet work** — only used for rigid body physics | Gaseous, MPM, destruction, rendering could all dispatch jobs on Jolt's existing pool |

---

## 2. Target Architecture

### 2.1 The Engine Object

After refactoring, consumers interact with the engine like this:

```cpp
#include <WulfNet/WulfNet.h>

int main() {
    // Single entry point
    WulfNet::EngineConfig config;
    config.appName = "My Game";
    config.enablePhysics = true;
    config.enableAudio = true;
    config.enableRendering = true;  // software raster by default
    config.enableCompute = true;    // Vulkan compute
    
    WulfNet::Engine engine;
    engine.Initialize(config);
    
    // Access subsystems through the engine
    auto& physics = engine.GetPhysics();       // PhysicsWorld (wraps Jolt + all extensions)
    auto& renderer = engine.GetRenderer();     // RenderPipeline
    auto& audio = engine.GetAudio();           // AudioMixer + Acoustics + Spatial
    auto& compute = engine.GetCompute();       // VulkanContext + pipelines
    
    // Game loop
    while (running) {
        engine.BeginFrame();
        
        // User logic here — full access to subsystems
        physics.GetFluidSystem().AddEmitter(...);
        renderer.Submit(meshes);
        audio.PlaySound(source);
        
        engine.EndFrame();   // Steps physics, renders, mixes audio, profiles
    }
    
    engine.Shutdown();  // Clean teardown in correct order
}
```

### 2.2 Target Directory Structure

```
WulfNet/
├── WulfNet.h                          # Public umbrella header
├── Engine.h / Engine.cpp              # NEW: Engine class (init/loop/shutdown)
├── EngineConfig.h                     # NEW: Centralized configuration
├── Version.h                          # NEW: Version macros
│
├── Core/
│   ├── Math/
│   │   ├── MathTypes.h               # NEW: Unified Vec2/Vec3/Vec4/Mat3/Mat4/Quat
│   │   ├── MathUtils.h               # NEW: lerp, clamp, smoothstep, etc.
│   │   └── PerlinNoise.h             # Existing (updated includes)
│   ├── Logging/
│   │   └── Logger.h / .cpp           # Existing (fix static API)
│   ├── Profiling/
│   │   └── Profiler.h / .cpp         # Existing
│   ├── Threading/                     # NEW directory
│   │   ├── ThreadPool.h / .cpp        # NEW: Persistent work-stealing thread pool
│   │   ├── JoltJobAdapter.h           # NEW: Dispatch WulfNet work as Jolt jobs
│   │   └── ParallelFor.h             # NEW: Convenience parallel-for utilities
│   ├── Memory/                        # NEW directory
│   │   └── FrameAllocator.h          # NEW: Linear allocator, reset per frame
│   └── System/
│       └── SystemMonitor.h / .cpp    # Existing
│
├── Physics/
│   ├── PhysicsWorld.h / .cpp          # Existing (UPGRADED: wires all subsystems)
│   ├── Fluids/
│   │   ├── FluidSystem.h / .cpp       # Renamed from WaterSystemV3 (namespace fix)
│   │   └── MarchingCubes.h / .cpp     # If surface extraction exists in WaterSystemV3
│   ├── MPM/
│   │   ├── ConstitutiveModel.h / .cpp # Existing (swap internal Mat3 → Core/Math)
│   │   └── MPMRigidCoupling.h / .cpp  # Existing
│   ├── Gaseous/
│   │   ├── GaseousSystem.h / .cpp     # Existing
│   │   └── GaseousSystemSolve.cpp     # Existing (split impl)
│   ├── Destruction/
│   │   └── DestructionSystem.h / .cpp # Existing
│   └── Terrain/
│       └── TerrainDeformation.h / .cpp # Existing
│
├── Compute/
│   ├── Compute.h                      # Existing umbrella
│   ├── Vulkan/
│   │   └── VulkanContext.h / .cpp     # Existing
│   ├── Memory/
│   │   └── ComputeBuffer.h / .cpp    # Existing
│   ├── Pipelines/                     # Renamed from Shaders/ (clarity)
│   │   └── ComputePipeline.h / .cpp  # Existing
│   ├── Fluids/
│   │   ├── SWEComputeGPU.h / .cpp    # Existing
│   │   └── JoltComputeAdapter.h      # Existing
│   └── Reduction/
│       └── ParallelReduction.h / .cpp # Existing
│
├── Rendering/
│   ├── RenderPipeline.h / .cpp        # MOVED from SoftwareRasterizer/
│   ├── SoftwareRasterizer/
│   │   ├── SoftwareRasterizer.h/.cpp  # Core rasterizer only
│   │   ├── GBuffer.h / .cpp          # Stays here (rasterizer-specific)
│   │   ├── DeferredShading.h / .cpp   # Stays here (rasterizer-specific)
│   │   └── OcclusionCuller.h / .cpp   # Stays here
│   ├── Lighting/                      # NEW directory
│   │   ├── ShadowMap.h / .cpp         # MOVED from SoftwareRasterizer/
│   │   └── GlobalIllumination.h / .cpp # MOVED from SoftwareRasterizer/
│   ├── Effects/                       # NEW directory
│   │   └── VolumetricRenderer.h / .cpp # MOVED from SoftwareRasterizer/
│   └── Types/
│       └── RenderTypes.h              # Renamed from SoftRasterTypes.h → shared types
│
├── Audio/
│   ├── Core/
│   │   ├── AudioTypes.h / .cpp        # Existing
│   │   └── AudioMixer.h / .cpp        # Existing
│   ├── Acoustics/
│   │   └── AcousticSystem.h / .cpp    # Existing
│   └── Spatial/
│       └── SpatialAudio.h / .cpp      # Existing
│
└── Procedural/
    └── IFS/
        ├── AffineTransform.h / .cpp   # Existing (swap Vec3/Mat → Core/Math)
        ├── TransformPresets.h / .cpp   # Existing
        ├── TransformBlender.h / .cpp   # Existing
        └── IFSSystem.h / .cpp         # Existing
```

### 2.3 Dependency Flow (Enforced)

```
         ┌──────────┐
         │  Engine   │  ← Owns everything, orchestrates lifecycle
         └────┬─────┘
              │
    ┌─────────┼─────────┬──────────┬──────────┐
    ▼         ▼         ▼          ▼          ▼
 Physics   Rendering  Audio    Compute   Procedural
    │         │                    ▲          │
    │         │                    │          │
    └─────────┴────────────────────┘          │
              │                               │
              ▼                               │
          Core/Math  ◄────────────────────────┘
          Core/Logging
          Core/Profiling
```

**Rules:**
- `Core/` depends on nothing (except std and platform APIs)
- `Compute/` depends on `Core/` only
- `Physics/`, `Rendering/`, `Audio/`, `Procedural/` depend on `Core/` and optionally `Compute/`
- `Engine` depends on everything and is the sole orchestrator
- **No lateral dependencies** (e.g., Rendering must NOT include Physics headers directly — data flows through Engine)

---

## 3. Refactor Passes

Each pass is designed to be completable independently, with tests verified after each pass.

---

### Pass 1 — Unified Math & Common Types
**Priority:** CRITICAL | **Risk:** Medium | **Estimated effort:** 1 session

**Why:** Four separate vector/matrix types make cross-module data flow impossible and duplicate code.

**Steps:**

- [ ] **1.1** Create `WulfNet/Core/Math/MathTypes.h`
  - Define `WulfNet::Vec2`, `Vec3`, `Vec4`, `Mat3`, `Mat4`, `Quat`
  - Make them GPU-compatible (alignas(16) for Vec4/Mat4)
  - Provide conversions to/from Jolt types (`JPH::Vec3`, `JPH::Mat44`, etc.)
  - Provide conversions to/from GPU types (for compute shaders)
  
- [ ] **1.2** Create `WulfNet/Core/Math/MathUtils.h`
  - `Lerp()`, `Clamp()`, `Smoothstep()`, `Remap()`
  - `DegreesToRadians()`, `RadiansToDegrees()`
  - Common constants: `Pi`, `TwoPi`, `Epsilon`

- [ ] **1.3** Migrate `ConstitutiveModel.h` — replace internal `Mat3` with `WulfNet::Mat3`
- [ ] **1.4** Migrate `AffineTransform.h` — replace internal `Vec3`, `GPUMat4x4` with `WulfNet::Vec3`, `WulfNet::Mat4`
- [ ] **1.5** Migrate `SoftRasterTypes.h` — replace `SoftVec2/3/4` with `WulfNet::Vec2/3/4` (keep `SoftVertex`, `SoftMesh` etc. as rendering-specific)
- [ ] **1.6** Update all `#include` paths across the codebase
- [ ] **1.7** Run all 574 tests — verify 100% pass

**Compatibility note:** Keep `using SoftVec3 = Vec3;` type aliases temporarily so downstream code compiles during transition.

---

### Pass 2 — Namespace & Directory Normalization
**Priority:** HIGH | **Risk:** Low | **Estimated effort:** 1 session

**Why:** `WaterSystemV3` sits in `WulfNet::Physics::` while all others use `WulfNet::`. Directory structure misrepresents module boundaries.

**Steps:**

- [ ] **2.1** Rename `WaterSystemV3` → `FluidSystem`
  - Move namespace from `WulfNet::Physics::` to `WulfNet::`
  - Move files from `Physics/WaterSystemV3.*` to `Physics/Fluids/FluidSystem.*`
  - Update all includes and references

- [ ] **2.2** Restructure `Rendering/` directory
  - Move `RenderPipeline.h/.cpp` to `Rendering/` (out of `SoftwareRasterizer/`)
  - Move `ShadowMap.h/.cpp` to `Rendering/Lighting/`
  - Move `GlobalIllumination.h/.cpp` to `Rendering/Lighting/`
  - Move `VolumetricRenderer.h/.cpp` to `Rendering/Effects/`
  - Move `SoftRasterTypes.h` to `Rendering/Types/RenderTypes.h`

- [ ] **2.3** Rename `Compute/Shaders/` → `Compute/Pipelines/` (the folder contains pipeline code, not shader source)

- [ ] **2.4** Update `WulfNet/CMakeLists.txt` with all new file paths
- [ ] **2.5** Update `WulfNet.h` umbrella includes
- [ ] **2.6** Run all tests — verify 100% pass

---

### Pass 3 — Engine Core (Init, Loop, Shutdown)
**Priority:** CRITICAL | **Risk:** Medium | **Estimated effort:** 1-2 sessions

**Why:** There's no single entry point. The engine is a bag of modules, not a coherent runtime.

**Steps:**

- [ ] **3.1** Create `WulfNet/Version.h`
  ```cpp
  #define WULFNET_VERSION_MAJOR 1
  #define WULFNET_VERSION_MINOR 0
  #define WULFNET_VERSION_PATCH 0
  #define WULFNET_VERSION_STRING "1.0.0"
  ```

- [ ] **3.2** Create `WulfNet/EngineConfig.h`
  - Single struct aggregating all module configs
  - Sensible defaults for everything (zero-config should work)
  - Feature flags: `enablePhysics`, `enableAudio`, `enableRendering`, `enableCompute`
  - Preset methods: `EngineConfig::Minimal()`, `EngineConfig::Full()`, `EngineConfig::HeadlessPhysics()`

- [ ] **3.3** Create `WulfNet/Engine.h` / `Engine.cpp`
  - `Engine::Initialize(const EngineConfig&)` — inits modules in dependency order:
    1. Logger
    2. Profiler
    3. SystemMonitor
    4. VulkanContext (if `enableCompute`)
    5. PhysicsWorld (if `enablePhysics`)
    6. RenderPipeline (if `enableRendering`)
    7. AudioMixer (if `enableAudio`)
  - `Engine::Shutdown()` — tears down in reverse order
  - `Engine::BeginFrame()` — starts profiling, polls input
  - `Engine::EndFrame()` — steps physics, renders, mixes audio, presents
  - `Engine::GetPhysics()`, `GetRenderer()`, `GetAudio()`, `GetCompute()` — subsystem access
  - `Engine::IsRunning()`, `GetFrameNumber()`, `GetDeltaTime()`

- [ ] **3.4** Implement ordered initialization with error handling
  - Each `Initialize()` returns a result/error
  - If a non-critical module fails (e.g., audio), engine continues with a warning
  - If a critical module fails (e.g., physics), engine fails with clear error

- [ ] **3.5** Implement frame loop in `BeginFrame()`/`EndFrame()`
  - Fixed timestep physics with interpolation
  - Frame timing / delta time calculation
  - Profiling zone auto-management
  
- [ ] **3.6** Run all tests — add Engine lifecycle tests

---

### Pass 4 — Public API Layer
**Priority:** HIGH | **Risk:** Low | **Estimated effort:** 1 session

**Why:** Consumers need a stable, documented API surface. Internal implementation details should be hidden.

**Steps:**

- [ ] **4.1** Fix Logger API inconsistency
  - Add static convenience methods: `Logger::Initialize()`, `Logger::SetMinLevel()`, `Logger::Info()`, etc.
  - These delegate to `Logger::Get()` internally
  - Both patterns (static and singleton) now work

- [ ] **4.2** Create `WulfNet/API.h` — export/visibility macros
  ```cpp
  #ifdef WULFNET_SHARED
    #ifdef WULFNET_EXPORT
      #define WULFNET_API __declspec(dllexport)  // or __attribute__((visibility("default")))
    #else
      #define WULFNET_API __declspec(dllimport)
    #endif
  #else
    #define WULFNET_API  // static lib — no decoration
  #endif
  ```

- [ ] **4.3** Mark all public classes with `WULFNET_API`

- [ ] **4.4** Create forward-declaration header `WulfNet/ForwardDecl.h`
  - Forward-declare all major types so consumers don't need to include everything

- [ ] **4.5** Review and stabilize all `*Config` structs
  - Ensure every config struct has a default constructor that produces valid defaults
  - Add `Validate()` methods to catch invalid configurations early

- [ ] **4.6** Run all tests

---

### Pass 5 — Wire Up PhysicsWorld ↔ Subsystems
**Priority:** CRITICAL | **Risk:** High | **Estimated effort:** 2 sessions

**Why:** PhysicsWorld has stubs for `GetMPMSystem()`, `GetGaseousSystem()`, `GetDestructionSystem()` but they're commented out. The subsystems exist independently but aren't connected.

**Steps:**

- [ ] **5.1** Uncomment and implement `PhysicsWorld` subsystem ownership
  ```cpp
  class PhysicsWorld {
      // Existing Jolt wrapper...
      
      // Extended subsystems (owned, lifecycle-managed)
      FluidSystem m_fluidSystem;
      GaseousSystem m_gaseousSystem;
      DestructionSystem m_destructionSystem;
      TerrainDeformation m_terrainDeformation;
      MPMRigidCoupling m_mpmCoupling;
      
  public:
      FluidSystem& GetFluidSystem();
      GaseousSystem& GetGaseousSystem();
      DestructionSystem& GetDestructionSystem();
      TerrainDeformation& GetTerrainDeformation();
      MPMRigidCoupling& GetMPMCoupling();
  };
  ```

- [ ] **5.2** Implement unified `PhysicsWorld::Step(float dt)`
  - Phase 1: Pre-sim — kick off GPU compute for fluids/MPM (async)
  - Phase 2: Jolt step — rigid bodies, soft bodies, constraints
  - Phase 3: Extended physics — sync GPU, apply coupling forces
  - Phase 4: Post-sim — dispatch events, prepare render data
  - This mirrors the frame phases from ENGINE_PLAN.md §2.3

- [ ] **5.3** Implement coupling interfaces
  - `FluidSystem` ↔ Jolt rigid bodies (buoyancy, drag forces)
  - `MPMRigidCoupling` ↔ Jolt bodies (penalty forces, terrain deformation)
  - `DestructionSystem` ↔ Jolt bodies (stress evaluation, fragment creation)
  - `GaseousSystem` ↔ scene (obstacle interaction)

- [ ] **5.4** Add `PhysicsWorld::CreateDestructibleBody()`, `CreateFluidEmitter()`, etc.
  - High-level convenience methods for common operations
  - Internally coordinate between Jolt and WulfNet subsystems

- [ ] **5.5** Integration tests — multi-system scenarios
  - Fluid + rigid body buoyancy test
  - Destruction + fragment physics test
  - Terrain deformation under rigid body impact test
  
- [ ] **5.6** Run all tests — verify nothing regressed

---

### Pass 6 — Rendering Abstraction
**Priority:** MEDIUM | **Risk:** Medium | **Estimated effort:** 1-2 sessions

**Why:** The software rasterizer is the only rendering backend. We need an abstraction layer so a Vulkan backend can be added later without breaking consumer code.

**Steps:**

- [ ] **6.1** Define `IRenderer` interface
  ```cpp
  class IRenderer {
  public:
      virtual ~IRenderer() = default;
      virtual void Initialize(const RenderConfig&) = 0;
      virtual void BeginFrame() = 0;
      virtual void Submit(const RenderableList&) = 0;
      virtual void EndFrame() = 0;
      virtual void Shutdown() = 0;
  };
  ```

- [ ] **6.2** Make `RenderPipeline` implement `IRenderer`
  - Current software rasterizer pipeline becomes the default backend
  - No behavior change, just interface conformance

- [ ] **6.3** Create `RenderableList` / `RenderCommand` abstraction
  - Decouple "what to render" from "how to render"
  - Physics subsystems produce renderables (fluid surface mesh, gas volume bounds, etc.)
  - Renderer consumes them without knowing the source

- [ ] **6.4** Connect physics → rendering data flow through Engine
  - `Engine::EndFrame()` collects renderables from physics and passes to renderer
  - No direct Physics→Rendering dependency

- [ ] **6.5** Run all tests

---

### Pass 7 — CMake & Build System Overhaul
**Priority:** HIGH | **Risk:** Low | **Estimated effort:** 1 session

**Why:** Root CMakeLists.txt is dead, builds only work through Jolt's scripts. Developers should be able to build WulfNet standalone.

**Steps:**

- [ ] **7.1** Make root `CMakeLists.txt` functional
  - `cmake -B out -S .` should work
  - Detect/find Jolt (either as subdirectory or installed package)
  - Build WulfNet, WulfNetTests, WulfNetExamples
  
- [ ] **7.2** Preserve Jolt build path
  - `build/CMakeLists.txt` still works for Jolt-first builds
  - Both entry points produce the same result

- [ ] **7.3** Add CMake install targets
  ```cmake
  install(TARGETS WulfNet EXPORT WulfNetTargets ...)
  install(DIRECTORY WulfNet/ DESTINATION include/WulfNet FILES_MATCHING PATTERN "*.h")
  ```

- [ ] **7.4** Add `find_package(WulfNet)` support
  - Generate `WulfNetConfig.cmake`
  - Consumers can: `find_package(WulfNet REQUIRED)` / `target_link_libraries(MyGame WulfNet::WulfNet)`

- [ ] **7.5** Add CMake presets (`CMakePresets.json`)
  - `windows-msvc-release`, `windows-clang-release`, `linux-gcc-release`, etc.
  - Replace the multitude of `.bat`/`.sh` scripts with one `cmake --preset` workflow

- [ ] **7.6** Verify build on at least Windows MSVC (primary platform)

---

### Pass 8 — Tests, Examples, & Documentation
**Priority:** HIGH | **Risk:** Low | **Estimated effort:** 1-2 sessions

**Why:** Examples are broken, docs reference old APIs, new Engine class needs examples.

**Steps:**

- [ ] **8.1** Fix all existing examples to compile and run
  - `HelloWulfNet` — update to use `Engine` class
  - `ComputeExample` — complete the TODO (load shader, run pipeline)
  - `IFSExample` — fix Logger API calls
  - `SoftRasterExample` — fix Logger API calls

- [ ] **8.2** Create new flagship example: `EngineDemo`
  - Demonstrates full engine: physics + fluid + rendering + audio
  - Uses the `Engine` class from Pass 3
  - Outputs a `.ppm` image or runs in a window

- [ ] **8.3** Update `docs/APIReference.md`
  - Add `Engine`, `EngineConfig` docs
  - Update all renamed/moved types
  - Add "Quick Start" section at top

- [ ] **8.4** Update `ENGINE_PLAN.md` to reflect completed refactor

- [ ] **8.5** Update `README.md`
  - Add quick-start code snippet
  - Update build instructions for new CMake workflow
  - Add architecture diagram

- [ ] **8.6** Add integration tests for Engine lifecycle
  - Test `Engine::Initialize()` / `Shutdown()` cycle
  - Test partial init (physics-only, audio-only)
  - Test frame loop (BeginFrame/EndFrame)

- [ ] **8.7** Final test run — all tests pass, all examples compile and run

---

### Pass 9 — CPU Parallelism & Threading
**Priority:** CRITICAL | **Risk:** Medium | **Estimated effort:** 2-3 sessions

**Why:** The audit found that **80% of compute-heavy code is single-threaded**. OpenMP is linked but unused. The software rasterizer creates/destroys threads every frame. Gaseous simulation, MPM, destruction, all rendering pixel passes, and shadow maps are fully serial. This is the single biggest performance win available.

**Architecture Decision: Hybrid Threading Model**
- **OpenMP** for data-parallel loops (grid cells, pixels, particles) — already linked, zero infrastructure cost
- **Persistent thread pool** for task-parallel work (render passes, shadow cascades) — replace per-frame `std::thread` creation
- **Jolt's `JobSystemThreadPool`** for physics-adjacent work (coupling, destruction evaluation) — reuse existing infrastructure

**Steps:**

#### 9.1 — Persistent Thread Pool
- [ ] **9.1.1** Create `WulfNet/Core/Threading/ThreadPool.h / .cpp`
  - Fixed thread count = `std::thread::hardware_concurrency() - 1` (leave 1 for main)
  - Work-stealing deque per thread (Chase-Lev)
  - `Submit(task)` → `std::future<T>`
  - `ParallelFor(begin, end, body)` — divides range across threads
  - `ParallelForTiled(width, height, tileSize, body)` — 2D tile dispatch for rendering
  - Integrates with profiler (`WULFNET_ZONE` per task)
  
- [ ] **9.1.2** Replace `SoftwareRasterizer` per-frame thread spawn
  - Remove `std::vector<std::thread> m_threads` and per-frame `emplace_back`/`join`
  - Use `ThreadPool::ParallelFor` over object indices
  - Eliminates OS thread creation overhead every frame

#### 9.2 — Gaseous System Parallelization (BIGGEST SINGLE WIN)
- [ ] **9.2.1** Add `#pragma omp parallel for collapse(2) schedule(static)` to ALL solver passes:
  - `ApplyBuoyancy()` — each cell independent
  - `ApplyCombustion()` — each cell independent
  - `ComputeVorticity()` — read-only neighbor access, no write conflicts
  - `ApplyVorticityConfinement()` — each cell independent
  - `ComputeDivergence()` — each cell independent
  - `ApplyPressureGradient()` — each cell independent
  - `AdvectFields()` — each cell independent (reads from old buffer)
  - `ApplyDissipation()` — each cell independent
  - `UpdateStats()` — use `reduction(+:sum)` / `reduction(max:maxVal)`
  
- [ ] **9.2.2** Pressure solve — switch to explicit parallel Jacobi
  - Double-buffer the pressure array (read old, write new)
  - `#pragma omp parallel for collapse(2)` on inner loop
  - Swap buffers after each iteration
  
- [ ] **9.2.3** Convert `GasCell` AoS → SoA layout for cache efficiency
  - Separate arrays: `float* density`, `float* temperature`, `float* fuel`, `float* u/v/w`, etc.
  - Each solver pass streams through only the fields it needs
  - Expected speedup: 2-4× from cache alone (64-byte cell → 4-byte field)

#### 9.3 — Rendering Pixel Passes Parallelization
- [ ] **9.3.1** `DeferredShading::Apply()` — row-parallel
  - `#pragma omp parallel for schedule(dynamic, 4)` on outer `y` loop
  - Each row writes to independent pixel range — no synchronization needed
  
- [ ] **9.3.2** `RenderPipeline::PassLighting()` — row-parallel
  - Same pattern: `#pragma omp parallel for` on outer `y` loop
  - All reads (GBuffer, shadow maps) are read-only; writes are to independent output pixels
  
- [ ] **9.3.3** `GlobalIllumination::Compute()` — row-parallel (**most expensive pass**)
  - SSAO hemisphere sampling per pixel is completely independent
  - `#pragma omp parallel for schedule(dynamic, 2)` (dynamic because ray count varies)
  - Expected speedup: near-linear with core count (16 samples × W×H pixels)
  
- [ ] **9.3.4** `GlobalIllumination::BlurAOBuffer()` — row-parallel
  - Box blur rows are independent
  - `#pragma omp parallel for schedule(static)`
  
- [ ] **9.3.5** `VolumetricRenderer::Render()` — tile-parallel
  - Each pixel ray-marches independently
  - Use `ThreadPool::ParallelForTiled(width, height, 16, ...)` for cache-friendly tiles
  - Front-to-back compositing is per-pixel — no conflicts

#### 9.4 — Shadow Map Parallelization
- [ ] **9.4.1** Parallel cascade rendering
  - Each cascade has its own depth buffer — fully independent
  - Dispatch cascade rendering as parallel tasks: `ThreadPool::Submit` per cascade
  - Wait on all futures before lighting pass
  
- [ ] **9.4.2** Parallel point light face rendering
  - 6 cube faces per point light are independent
  - Same pattern: parallel task per face
  
- [ ] **9.4.3** Per-cascade triangle rasterization — row-parallel within each cascade
  - Use `#pragma omp parallel for` on triangle loop (atomic depth test needed → use `std::atomic<float>` compare-exchange or tile-based no-conflict approach)

#### 9.5 — Physics Subsystem Parallelization
- [ ] **9.5.1** `MPMRigidCoupling::ComputeCoupling()`
  - `#pragma omp parallel for schedule(dynamic, 64)` on particle loop
  - Each particle's force accumulation is independent
  - Body force accumulation needs `#pragma omp atomic` or per-thread accumulation + reduce
  
- [ ] **9.5.2** `ConstitutiveModel` stress evaluation
  - Per-particle SVD + return mapping is independent and expensive
  - `#pragma omp parallel for` on particle loop
  
- [ ] **9.5.3** `DestructionSystem::ComputeVoronoiVolumes()`
  - Sample grid loop is embarrassingly parallel
  - `#pragma omp parallel for collapse(3)` on 16³ grid
  - Cell accumulation needs `#pragma omp atomic`
  
- [ ] **9.5.4** `TerrainDeformation::ApplyMPMDeformation()`
  - Per-particle terrain query is independent
  - Heightfield writes need atomic or tile decomposition
  - Lower priority (usually small particle count)

#### 9.6 — Leverage Jolt's Job System for WulfNet Work
- [ ] **9.6.1** Create `WulfNet/Core/Threading/JoltJobAdapter.h`
  - Utility to dispatch WulfNet work items as Jolt jobs
  - `SubmitAsJoltJob(JPH::JobSystem&, name, func)` → returns job handle
  - Allows WulfNet extended physics to run on the same thread pool as Jolt
  
- [ ] **9.6.2** Dispatch destruction stress evaluation as Jolt jobs during `PhysicsWorld::Step()`
- [ ] **9.6.3** Dispatch coupling force computation as Jolt jobs during `PhysicsWorld::Step()`

#### 9.7 — Verification
- [ ] Run all 574+ tests with OpenMP enabled — verify thread-safe (no races)
- [ ] Add performance benchmarks: before/after for each parallelized system
- [ ] Test on 4-core, 8-core, 16-core configurations (thread count scaling)

---

### Pass 10 — GPU Async Compute & SIMD
**Priority:** CRITICAL | **Risk:** High | **Estimated effort:** 2-3 sessions

**Why:** GPU compute is fully synchronous — the CPU sits idle during every `vkQueueWaitIdle()`. No async readback exists. SIMD is only used for GBuffer clear. Fixing these unlocks massive throughput: CPU/GPU overlap doubles effective bandwidth, and SIMD gives 4-8× on vectorizable inner loops.

**Steps:**

#### 10.1 — Async GPU Compute Framework
- [ ] **10.1.1** Add frame-pipelined command buffer management to `VulkanContext`
  ```cpp
  // Double-buffered: submit frame N, CPU works on N+1 while GPU executes N
  struct FrameResources {
      VkCommandBuffer cmdBuffer;
      VkFence fence;           // Signal when GPU done
      VkSemaphore semaphore;   // Signal between queues
      bool inFlight = false;
  };
  static constexpr int FRAMES_IN_FLIGHT = 2;
  FrameResources m_frames[FRAMES_IN_FLIGHT];
  ```
  
- [ ] **10.1.2** Replace `SubmitAndWait()` with `Submit()` + deferred `WaitForFrame()`
  - `Submit(frameIndex)` — submits without blocking
  - `WaitForFrame(frameIndex)` — blocks only when results needed (before readback)
  - `PollFrame(frameIndex)` — non-blocking check if done
  
- [ ] **10.1.3** Add `ComputePipeline::DispatchAsync()` (no immediate wait)
  - Records into current frame's command buffer
  - Multiple dispatches chain with pipeline barriers (not queue waits)
  - Final `Submit()` sends entire batch

- [ ] **10.1.4** Command buffer pool/ring
  - Pre-allocate N command buffers from the pool
  - Cycle through them per frame instead of alloc/free each dispatch
  - Eliminates per-dispatch Vulkan driver overhead

#### 10.2 — Async GPU Readback
- [ ] **10.2.1** Implement `ComputeBuffer::DownloadAsync()`
  - Uses a staging buffer + fence
  - `DownloadAsync()` kicks off GPU→staging copy with a fence
  - `IsDownloadReady()` polls the fence
  - `GetDownloadedData()` maps staging buffer (only after fence signals)
  
- [ ] **10.2.2** Implement `WaterSystemV3::RequestAsyncReadback()` (currently a stub)
  - Kicks off async download at end of GPU step
  - CPU reads results next frame (1-frame latency, but zero stall)
  
- [ ] **10.2.3** Apply same pattern to `IFSSystem` particle download
  - Current: 6× `DispatchAndWait()` per iteration
  - Target: 6 dispatches in 1 command buffer → 1 submit → async readback

#### 10.3 — CPU/GPU Overlap in Engine Frame
- [ ] **10.3.1** Implement pipelined frame in `Engine::BeginFrame()` / `EndFrame()`
  ```
  Frame N:
    BeginFrame():
      - Wait for GPU frame N-2 (2 frames in flight)
      - Read back GPU results from frame N-1 (async, should be done)
      - Begin recording GPU commands for frame N
    
    [User code / physics / audio — CPU work while GPU runs N-1]
    
    EndFrame():
      - Submit GPU frame N (non-blocking)
      - Kick off async readback for frame N results
      - Present / swap
  ```

- [ ] **10.3.2** Overlap physics CPU work with previous frame's GPU compute
  - While Jolt steps rigid bodies (CPU), GPU runs fluid/MPM compute from previous kick
  - Coupling uses GPU results from frame N-1 (1-frame latency, acceptable for visual physics)

#### 10.4 — Multi-Queue GPU Dispatch (Advanced)
- [ ] **10.4.1** Detect async compute queue (separate from graphics/compute)
  - Most discrete GPUs have 2+ compute queues
  - `VulkanContext::FindAsyncComputeQueue()` at init
  
- [ ] **10.4.2** Dispatch independent workloads on separate queues
  - Fluid simulation → async compute queue
  - IFS/procedural → main compute queue
  - Use semaphores for cross-queue synchronization
  
- [ ] **10.4.3** Separate transfer queue for buffer uploads/downloads
  - DMA engine on discrete GPUs runs in parallel with compute
  - Upload particle data while compute shader runs

#### 10.5 — SIMD Optimization
- [ ] **10.5.1** Unified math types with SIMD storage (from Pass 1)
  - `Vec4` backed by `__m128` (SSE) or `float32x4_t` (NEON)
  - `Mat4` backed by `__m128[4]`
  - Benefit: all code using unified math gets SIMD automatically
  
- [ ] **10.5.2** SIMD rasterization inner loop
  - Barycentric coordinate computation: 4 pixels at once with SSE
  - Depth interpolation: 4 pixels at once
  - Attribute interpolation: 4 pixels at once
  - Expected: 2-4× speedup on scanline interior
  
- [ ] **10.5.3** SIMD deferred lighting
  - Process 4 pixels at once in the lighting pass
  - N·L dot product, specular power, color accumulation — all SIMD-friendly
  - Shadow map sampling remains scalar (irregular access pattern)
  
- [ ] **10.5.4** SIMD SSAO
  - Hemisphere sample direction computation: batch 4 samples
  - Depth comparison: 4 samples at once
  - Most impactful on GI which is already the bottleneck
  
- [ ] **10.5.5** SIMD gaseous solver (supplement to OpenMP)
  - Process 4 adjacent grid cells in SSE registers
  - SoA layout (from 9.2.3) enables direct `_mm_load_ps` on field arrays
  - Combined with OpenMP row-parallel: each thread uses SIMD within its rows

- [ ] **10.5.6** Use Jolt's SIMD math where interfacing with Jolt
  - `JPH::Vec3` / `JPH::Vec4` in `PhysicsWorld`, `MPMRigidCoupling`, `DestructionSystem`
  - Avoid scalar→SIMD→scalar conversions at Jolt boundaries

#### 10.6 — Data-Oriented Memory Optimization
- [ ] **10.6.1** `GaseousSystem` AoS → SoA (detailed in 9.2.3)
  - Separate arrays per field: `density[]`, `temperature[]`, `u[]`, `v[]`, `w[]`
  - Align arrays to 64 bytes (cache line) for SIMD streaming
  
- [ ] **10.6.2** Pre-allocate frame-transient memory
  - Create `FrameAllocator` — linear allocator reset each frame
  - Replace per-frame `std::vector` resizes in AudioMixer, active tile lists, temp buffers
  - Zero allocation cost after first frame
  
- [ ] **10.6.3** Command buffer ring for Vulkan
  - Pre-allocate N command buffers (N = `FRAMES_IN_FLIGHT` × dispatches-per-frame)
  - Round-robin assignment, reset after fence confirms completion
  
- [ ] **10.6.4** Particle SoA for MPM (future)
  - Current `MPMParticle` struct is AoS (position, velocity, deformation gradient)
  - SoA enables SIMD P2G scatter and G2P gather
  - Lower priority — MPM is GPU-targeted long-term

#### 10.7 — Compiler & Build Optimizations
- [ ] **10.7.1** Explicit optimization flags in WulfNet CMakeLists.txt
  ```cmake
  # Release optimizations
  if(MSVC)
    target_compile_options(WulfNet PRIVATE
      $<$<CONFIG:Release>:/O2 /Ob3 /GL /Oi /Ot /fp:fast /arch:AVX2>
    )
    target_link_options(WulfNet PRIVATE $<$<CONFIG:Release>:/LTCG>)
  else()
    target_compile_options(WulfNet PRIVATE
      $<$<CONFIG:Release>:-O3 -march=native -ffast-math -flto>
    )
  endif()
  ```
  
- [ ] **10.7.2** Profile-guided optimization (PGO) setup
  - Add CMake preset for instrumented build
  - Run benchmarks to generate profile data
  - Rebuild with `/LTCG:PGU` (MSVC) or `-fprofile-use` (GCC/Clang)
  
- [ ] **10.7.3** Enable IPO/LTO for WulfNet (currently force-disabled due to Jolt conflict)
  - Apply LTO only to WulfNet target, not Jolt
  - `set_target_properties(WulfNet PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)`

#### 10.8 — Verification
- [ ] Benchmark: GPU frame time before/after async compute
- [ ] Benchmark: CPU utilization (should see >80% across all cores during render pass)
- [ ] Benchmark: Gaseous 64³ step time before/after (target: 5×+ speedup)
- [ ] Benchmark: Full-screen lighting pass before/after SIMD + OMP
- [ ] Verify zero visual regression: render comparison before/after
- [ ] Run all tests with AVX2 + OpenMP + async compute enabled
- [ ] Memory profiler: verify zero per-frame allocations in hot paths

---

## 4. Performance Audit — Current State

This section documents the parallelism audit as of March 2026, before any optimization work.

### 4.1 Parallelism Scorecard

| Module | Current Status | Parallelism Used | Biggest Win |
|--------|---------------|-----------------|-------------|
| WaterSystemV3 CPU | ✅ Parallel | `std::execution::par` on rows | Async GPU readback |
| WaterSystemV3 GPU | ⚠️ GPU-parallel, CPU-sync | Vulkan compute (sync dispatch) | Double-buffer + fence polling |
| **Gaseous System** | ❌ **Fully serial** | None | **OpenMP on all 10 passes (largest win)** |
| **MPM Coupling** | ❌ **Fully serial** | None | OpenMP on particle loop |
| MPM ConstitutiveModel | ❌ Fully serial | None | OpenMP on per-particle SVD |
| Destruction | ❌ Fully serial | None | OpenMP on Voronoi sampling |
| Terrain | ❌ Fully serial | None | Low priority |
| **Software Rasterizer** (objects) | ✅ Multi-threaded | `std::thread` per frame | **Persistent thread pool** |
| **Deferred Lighting** | ❌ **Fully serial** | None | **Row-parallel OpenMP** |
| **SSAO / GI** | ❌ **Fully serial** | None | **Row-parallel OpenMP (most expensive pass)** |
| **Shadow Maps** | ❌ **Fully serial** | None | **Parallel cascades + faces** |
| **Volumetric Rendering** | ❌ **Fully serial** | None | **Tile-parallel ray march** |
| Audio Mixer | ❌ Fully serial | None | Low priority (small source count) |
| **GPU Compute Model** | ❌ **Synchronous** | `vkQueueWaitIdle()` blocks CPU | **Async dispatch + double-buffer** |
| **SIMD** | ⚠️ GBuffer clear only | AVX2/SSE2 for memset | **Raster, lighting, SSAO inner loops** |

### 4.2 GPU Compute Model — Current

```
CURRENT (synchronous — CPU idles):
  CPU: [Record] → [Submit] → [████ IDLE ████] → [Record] → [Submit] → [████ IDLE ████]
  GPU:                        [  Execute   ]                          [  Execute   ]

TARGET (async pipelined — zero idle):
  CPU: [Record N] → [Submit N] → [Record N+1] → [Submit N+1] → [Read N] → ...
  GPU:                [Execute N] ─────────────→ [Execute N+1] ────────→ ...
```

### 4.3 Threading Model — Current vs Target

```
CURRENT:
  Main Thread:  [Init] → [Physics] → [Render (serial pixels)] → [Audio] → [Present]
  Jolt Threads: [        Rigid body solver         ] → [idle ──────────────────────]
  GPU:          [idle] → [idle] → [Compute dispatch] → [idle ─────────────────────]

TARGET:
  Main Thread:  [Init] → [BeginFrame] → [User Logic] → [EndFrame] → [Present]
  Thread Pool:  [Physics coupling] [Shadow cascades] [GI rows] [Lighting rows] [Volumetric tiles]
  Jolt Threads: [Rigid body solver] [Destruction eval] [MPM coupling jobs]
  GPU:          [Fluid compute ███████] [MPM compute ██] [IFS ██] [Readback ─→]
                 ↕ overlap with CPU ↕
```

### 4.4 Memory Layout Issues

| Structure | Current | Target | Impact |
|-----------|---------|--------|--------|
| `GasCell` | AoS — 64 bytes, 10 fields | SoA — separate `float[]` per field | 2-4× cache hit rate per pass |
| `MPMParticle` | AoS — pos, vel, F, volume | SoA arrays (long-term, GPU target) | Better P2G/G2P streaming |
| `SoftwareRasterizer::m_threads` | Created/destroyed per frame | Persistent `ThreadPool` | Eliminates OS overhead |
| AudioMixer temp buffer | Per-frame resize potential | Pre-allocated at init | Zero allocation in hot path |
| Vulkan `VkCommandBuffer` | Alloc/free per dispatch | Ring buffer (pre-allocated) | Eliminates driver overhead |
| `m_activeTiles` | Clear + push_back per step | Pre-allocated, swap buffer | Eliminates reallocation |

---

## 5. Migration Rules

These rules apply across ALL passes to keep the refactor safe:

### 5.1 Never Break Tests
- Run the full test suite after every logical change
- If tests break, fix before moving forward
- Track test count: it should only go UP (574 → 600+)

### 5.2 One Concern Per Commit
- Don't mix "rename file" with "change behavior"
- Renames and moves in one commit, logic changes in another

### 5.3 Backward Compatibility During Transition
- When renaming types, add `using OldName = NewName;` aliases temporarily
- Mark aliases with `[[deprecated("Use NewName instead")]]`
- Remove aliases in a final cleanup commit after all tests pass

### 5.4 Preserve Jolt Upstream
- **Never modify files in `Jolt/`**
- All integration goes through WulfNet wrappers
- Coupling with Jolt only through `Physics/PhysicsWorld` and `Compute/Fluids/JoltComputeAdapter`

### 5.5 Include-What-You-Use
- Every `.cpp` includes exactly the headers it needs
- No reliance on transitive includes
- Umbrella headers (`WulfNet.h`, `Compute.h`) are for consumers, not internal code

---

## 6. Verification Checklist

After ALL passes are complete, verify:

### Architecture & API
- [ ] `WulfNet::Engine` can be constructed, initialized, run for 100 frames, and shut down cleanly
- [ ] All 574+ tests pass
- [ ] All examples compile and produce expected output
- [ ] `cmake -B out -S .` works from the repo root on Windows
- [ ] `cmake --build out --config Release` produces `WulfNet.lib` and all executables
- [ ] No `WulfNet::Physics::` nested namespace exists (flat `WulfNet::` everywhere)
- [ ] Only ONE vector type (`WulfNet::Vec3`) exists across the codebase
- [ ] Only ONE matrix type (`WulfNet::Mat3`, `WulfNet::Mat4`) exists across the codebase
- [ ] `PhysicsWorld::Step()` drives all physics subsystems (fluid, MPM, gas, destruction, terrain)
- [ ] `docs/APIReference.md` is accurate and up-to-date
- [ ] `README.md` has a working quick-start snippet
- [ ] No `// TODO`, `// FIXME`, or `// HACK` comments remain without a tracking issue

### Performance & Parallelism
- [ ] Gaseous system uses OpenMP on all solver passes (10+)
- [ ] All rendering pixel passes (lighting, SSAO, volumetric) are row/tile-parallel
- [ ] Shadow cascades render in parallel
- [ ] `SoftwareRasterizer` uses persistent `ThreadPool` (no per-frame thread creation)
- [ ] GPU compute dispatches are async (CPU doesn't block on `vkQueueWaitIdle` in main loop)
- [ ] GPU readback is async (1-frame latency, zero CPU stall)
- [ ] At least 2 frames in flight for GPU compute
- [ ] Command buffers use a ring/pool (no per-dispatch alloc/free)
- [ ] `GasCell` AoS converted to SoA layout
- [ ] Unified `Vec4`/`Mat4` types backed by SIMD intrinsics
- [ ] SIMD used in at least: rasterization inner loop, deferred lighting, SSAO sampling
- [ ] Explicit `/O2 /arch:AVX2` (MSVC) or `-O3 -march=native` in WulfNet CMake Release config
- [ ] Zero per-frame heap allocations in hot paths (verified with profiler)
- [ ] Benchmark suite shows ≥3× improvement on gaseous 64³ step time
- [ ] Benchmark suite shows ≥2× improvement on full-screen SSAO pass
- [ ] CPU utilization >80% across all cores during heavy frames (verified with profiler)

---

## Appendix A — File Move/Rename Map

| Current Path | New Path | Action |
|-------------|----------|--------|
| `WulfNet/WulfNet.h` | `WulfNet/WulfNet.h` | UPDATE includes |
| — | `WulfNet/Engine.h` | **CREATE** |
| — | `WulfNet/Engine.cpp` | **CREATE** |
| — | `WulfNet/EngineConfig.h` | **CREATE** |
| — | `WulfNet/Version.h` | **CREATE** |
| — | `WulfNet/API.h` | **CREATE** |
| — | `WulfNet/ForwardDecl.h` | **CREATE** |
| — | `WulfNet/Core/Math/MathTypes.h` | **CREATE** |
| — | `WulfNet/Core/Math/MathUtils.h` | **CREATE** |
| — | `WulfNet/Core/Threading/ThreadPool.h` | **CREATE** |
| — | `WulfNet/Core/Threading/ThreadPool.cpp` | **CREATE** |
| — | `WulfNet/Core/Threading/JoltJobAdapter.h` | **CREATE** |
| — | `WulfNet/Core/Memory/FrameAllocator.h` | **CREATE** |
| `WulfNet/Physics/WaterSystemV3.h` | `WulfNet/Physics/Fluids/FluidSystem.h` | RENAME + MOVE |
| `WulfNet/Physics/WaterSystemV3.cpp` | `WulfNet/Physics/Fluids/FluidSystem.cpp` | RENAME + MOVE |
| `WulfNet/Physics/Integration/PhysicsWorld.h` | `WulfNet/Physics/PhysicsWorld.h` | MOVE up one level |
| `WulfNet/Physics/Integration/PhysicsWorld.cpp` | `WulfNet/Physics/PhysicsWorld.cpp` | MOVE up one level |
| `WulfNet/Rendering/SoftwareRasterizer/RenderPipeline.h` | `WulfNet/Rendering/RenderPipeline.h` | MOVE up one level |
| `WulfNet/Rendering/SoftwareRasterizer/RenderPipeline.cpp` | `WulfNet/Rendering/RenderPipeline.cpp` | MOVE up one level |
| `WulfNet/Rendering/SoftwareRasterizer/ShadowMap.h` | `WulfNet/Rendering/Lighting/ShadowMap.h` | MOVE |
| `WulfNet/Rendering/SoftwareRasterizer/ShadowMap.cpp` | `WulfNet/Rendering/Lighting/ShadowMap.cpp` | MOVE |
| `WulfNet/Rendering/SoftwareRasterizer/GlobalIllumination.h` | `WulfNet/Rendering/Lighting/GlobalIllumination.h` | MOVE |
| `WulfNet/Rendering/SoftwareRasterizer/GlobalIllumination.cpp` | `WulfNet/Rendering/Lighting/GlobalIllumination.cpp` | MOVE |
| `WulfNet/Rendering/SoftwareRasterizer/VolumetricRenderer.h` | `WulfNet/Rendering/Effects/VolumetricRenderer.h` | MOVE |
| `WulfNet/Rendering/SoftwareRasterizer/VolumetricRenderer.cpp` | `WulfNet/Rendering/Effects/VolumetricRenderer.cpp` | MOVE |
| `WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h` | `WulfNet/Rendering/Types/RenderTypes.h` | RENAME + MOVE |
| `WulfNet/Compute/Shaders/ComputePipeline.h` | `WulfNet/Compute/Pipelines/ComputePipeline.h` | MOVE |
| `WulfNet/Compute/Shaders/ComputePipeline.cpp` | `WulfNet/Compute/Pipelines/ComputePipeline.cpp` | MOVE |

---

## Appendix B — Execution Order

| Pass | Depends On | Can Parallelize With |
|------|-----------|---------------------|
| Pass 1 (Math) | — | — |
| Pass 2 (Namespaces) | Pass 1 | — |
| Pass 3 (Engine Core) | Pass 1, 2 | — |
| Pass 4 (Public API) | Pass 3 | — |
| Pass 5 (Physics Wiring) | Pass 2, 3 | Pass 6, 7 |
| Pass 6 (Rendering) | Pass 2, 3 | Pass 5, 7 |
| Pass 7 (CMake) | Pass 2 | Pass 5, 6 |
| **Pass 9 (CPU Parallel)** | **Pass 2, 3** | **Pass 5, 6, 7** |
| **Pass 10 (GPU Async + SIMD)** | **Pass 1, 3, 9** | **Pass 5** |
| Pass 8 (Docs/Tests) | ALL |  — |

**Critical path:** Pass 1 → Pass 2 → Pass 3 → Pass 4 → Pass 5/6/7/9 (parallel) → Pass 10 → Pass 8

**Performance-focused critical path:** Pass 1 (SIMD types) → Pass 9 (OpenMP + ThreadPool) → Pass 10 (GPU async + SIMD inner loops)

---

*This is a living document. Update status checkboxes as passes are completed.*
