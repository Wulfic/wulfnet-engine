# WulfNet Integration Plan — Full Merge & Simplification

**Date:** March 7, 2026  
**Goal:** Merge all Jolt-branded top-level folders (`Samples/`, `UnitTests/`, `TestFramework/`, `JoltViewer/`, `PerformanceTest/`) directly into the WulfNet folder structure. No nesting — content merges flat into the appropriate WulfNet directories. After completion, zero Jolt-named folders remain at the top level. `Jolt/` stays only as the upstream physics library source.

---

## Current State (9 top-level source folders)

| Folder | Role | Build System |
|--------|------|-------------|
| `Jolt/` | Physics library (upstream dependency) | `Jolt/Jolt.cmake` |
| `JoltViewer/` | Jolt recording playback viewer (154 LoC) | `JoltViewer.cmake`, links `TestFramework` |
| `Samples/` | Interactive visual test/demo app (~250 files) | `Samples.cmake`, links `TestFramework` + `WulfNet` |
| `TestFramework/` | Rendering/UI framework (DX12/Vulkan/Metal) | `TestFramework.cmake`, links `Jolt` |
| `UnitTests/` | Jolt-only headless unit tests (doctest, ~80 files) | `UnitTests.cmake`, links `Jolt` |
| `PerformanceTest/` | Jolt perf benchmarks (10 files) | `PerformanceTest.cmake`, links `Jolt` |
| `WulfNet/` | Engine library (static, ~80 files) | `WulfNet/CMakeLists.txt`, links `Jolt` |
| `WulfNetExamples/` | 5 standalone CLI demos | `WulfNetExamples/CMakeLists.txt` |
| `WulfNetTests/` | 537 tests + 39 benchmarks, custom harness | `WulfNetTests/CMakeLists.txt` |

**Problems:**
1. 5 of 9 source folders are Jolt-branded — project identity is WulfNet Engine
2. Two build entry points (`CMakeLists.txt` vs `build/CMakeLists.txt`), neither builds everything
3. Two test frameworks (custom TestHarness vs doctest), two test executables
4. 7 dead backward-compat redirect headers, deprecated type aliases
5. JoltViewer is a ~150 line recording playback tool with no WulfNet integration

---

## Target State (4 top-level source folders)

```
wulfnet-engine/
├── CMakeLists.txt          ← Single build entry point
├── Jolt/                   ← Upstream physics lib (unchanged, dependency only)
├── WulfNet/                ← Engine library + application framework
├── WulfNetTests/           ← ALL tests + benchmarks (WulfNet + Jolt)
├── WulfNetExamples/        ← ALL examples + visual demos
├── Assets/                 ← Unchanged
├── build/                  ← Build scripts (simplified)
├── cmake/                  ← CMake helpers
└── _ref/                   ← Reference / archived
```

**6 Jolt-branded folders eliminated.** Only `Jolt/` remains (the actual library).

---

## Integration Plan

### Phase 1: Cleanup — Remove Deprecated & Dead Code
**Effort:** Small | **Risk:** Low

#### 1a. Delete backward-compat redirect headers

| File to Delete | Redirects To |
|---|---|
| `WulfNet/Rendering/SoftwareRasterizer/VolumetricRenderer.h` | `WulfNet/Rendering/Effects/VolumetricRenderer.h` |
| `WulfNet/Rendering/SoftwareRasterizer/ShadowMap.h` | `WulfNet/Rendering/Lighting/ShadowMap.h` |
| `WulfNet/Rendering/SoftwareRasterizer/GlobalIllumination.h` | `WulfNet/Rendering/Lighting/GlobalIllumination.h` |
| `WulfNet/Rendering/SoftwareRasterizer/RenderPipeline.h` | `WulfNet/Rendering/RenderPipeline.h` |
| `WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h` | `WulfNet/Rendering/Types/RenderTypes.h` |
| `WulfNet/Physics/WaterSystemV3.h` | `WulfNet/Physics/Fluids/FluidSystem.h` |
| `WulfNet/Compute/Shaders/ComputePipeline.h` | `WulfNet/Compute/Pipelines/ComputePipeline.h` |

#### 1b. Update all `#include` paths referencing deleted redirects
~15 files across WulfNetTests, WulfNetExamples, WulfNet internals, and Samples need updates to canonical include paths.

#### 1c. Replace deprecated type aliases
- `GPUMat4x4` → `Mat4` everywhere (~25 usages across tests + Samples)
- `SoftVec2/3/4` → `Vec2/3/4` in `WulfNet/Rendering/Types/RenderTypes.h` (remove alias definitions and all usages)

#### 1d. Delete dead files
- `WulfNetTests/WulfNetTests.cpp.bak`

---

### Phase 2: Merge TestFramework into WulfNet
**Effort:** Medium | **Risk:** Medium

TestFramework is a rendering/UI/windowing library (DX12, Vulkan, Metal) with platform-specific Input and Window backends. It's a natural fit as the **application framework** layer of the WulfNet engine.

#### 2a. Move contents into `WulfNet/Framework/`
Rename to "Framework" (not "TestFramework" — it's not test-specific):

```
WulfNet/Framework/                  ← Was TestFramework/
├── Application/                    (Application.cpp/.h, DebugUI, EntryPoint)
├── External/                       (Perlin.cpp, stb_truetype.h)
├── Image/                          (BlitSurface, LoadBMP, LoadTGA, Surface, ZoomImage)
├── Input/                          (Keyboard, Mouse + Win/Linux/MacOS backends)
├── Renderer/                       (DebugRendererImp, Font, Renderer + DX12/VK/MTL backends)
├── UI/                             (UIManager, UIButton, UICheckBox, UISlider, etc.)
├── Utils/                          (CustomMemoryHook, AssetStream, ReadData, Log)
├── Window/                         (ApplicationWindow + Win/Linux/MacOS backends)
├── Framework.cmake                 ← Was TestFramework.cmake
└── Framework.h                     ← Was TestFramework.h
```

#### 2b. Update CMake
- `Framework.cmake`: Update `TEST_FRAMEWORK_ROOT` → `FRAMEWORK_ROOT`, point to `WulfNet/Framework/`
- Target name: Rename from `TestFramework` to `WulfNetFramework` (or keep `TestFramework` as the CMake target for link compat and rename later)
- Include directories stay the same relative to the Framework root — **no source-level `#include` changes** in consuming files (`<Application/Application.h>`, `<Renderer/...>`, `<UI/...>` all still resolve)

#### 2c. Update `Framework.h` (was `TestFramework.h`)
- Only the filename and any self-referential comments change
- Precompiled header target updates in CMake

---

### Phase 3: Merge UnitTests + PerformanceTest into WulfNetTests
**Effort:** Medium | **Risk:** Medium

#### 3a. Move Jolt unit test files directly into `WulfNetTests/`

The Jolt tests are organized by domain (`Core/`, `Math/`, `Physics/`, `Geometry/`, `Compute/`). The existing WulfNet tests are flat files at the root (`CoreTests.cpp`, `PhysicsWorldTests.cpp`, etc.). These don't conflict — different names, different structure. Merge directly:

```
WulfNetTests/
├── CMakeLists.txt
│
│  ── Existing WulfNet test infrastructure ──
├── TestHarness.h
├── BenchmarkHarness.h
├── WulfNetTests.cpp                (WulfNet test entry point)
├── WulfNetExtendedTests.cpp        (WulfNet extended test entry point)
├── CoreTests.cpp                   (WulfNet core tests)
├── PhysicsWorldTests.cpp           (WulfNet physics)
├── VulkanComputeTests.cpp
├── IFSTransformTests.cpp
├── SoftwareRendererTests.cpp
├── ... (20+ WulfNet test files)
├── PerformanceBenchmarks.cpp       (WulfNet benchmarks)
│
│  ── Merged from UnitTests/ ──
├── doctest.h
├── UnitTestFramework.cpp / .h      (doctest main/config)
├── PhysicsTestContext.cpp / .h      (Jolt test helper)
├── UnitTestLayers.h                ← Renamed from Layers.h (avoids collision with Samples Layers.h)
├── LoggingBodyActivationListener.h
├── LoggingContactListener.h
├── LoggingCharacterContactListener.h
├── Core/                           (14 Jolt core tests)
│   ├── ArrayTest.cpp
│   ├── BinaryHeapTest.cpp
│   ├── JobSystemTest.cpp
│   └── ...
├── Geometry/                       (7 Jolt geometry tests)
├── Math/                           (14 Jolt math tests)
├── Physics/                        (33 Jolt physics tests)
├── Compute/                        (Jolt ComputeTests.cpp)
├── ObjectStream/                   (ObjectStreamTest.cpp)
│
│  ── Merged from PerformanceTest/ ──
└── Benchmarks/
    ├── PerformanceTest.cpp         (Jolt perf test entry point)
    ├── PerformanceTestScene.h
    ├── PerfTestLayers.h            ← Renamed from Layers.h
    ├── PyramidScene.h
    ├── RagdollScene.h
    ├── ConvexVsMeshScene.h
    ├── CharacterVirtualScene.h
    ├── LargeMeshScene.h
    └── MaxBodiesScene.h
```

**Naming conflict resolution:** Three different `Layers.h` files exist:
- `UnitTests/Layers.h` (11 layers, complex collision matrix) → rename to `UnitTestLayers.h`
- `PerformanceTest/Layers.h` (2 layers, simple) → rename to `PerfTestLayers.h`
- `Samples/Layers.h` (8 layers, medium) → stays as `SamplesLayers.h` under WulfNetExamples (Phase 4)

Each renamed file needs a one-line `#include` update in its consumers.

#### 3b. Keep two test frameworks side-by-side (for now)
- **WulfNet tests** continue using `TestHarness.h` (custom macros, `main()` in `WulfNetTests.cpp`)
- **Jolt tests** continue using `doctest.h` (`main()` in `UnitTestFramework.cpp`)
- No forced migration. Both produce separate executables. Can unify later.

#### 3c. Update WulfNetTests/CMakeLists.txt
Build **four** executables from this directory:

| Target | Framework | Sources | Links |
|--------|-----------|---------|-------|
| `WulfNetTests` | TestHarness | 7 WulfNet test files | `WulfNet` + `Jolt` |
| `WulfNetExtendedTests` | TestHarness | 20 WulfNet test files | `WulfNet` + `Jolt` |
| `UnitTests` | doctest | Core/ + Geometry/ + Math/ + Physics/ + Compute/ + ObjectStream/ | `WulfNet` + `Jolt` |
| `PerformanceTest` | standalone | Benchmarks/*.cpp | `WulfNet` + `Jolt` |

Keep Jolt-facing executable names (`UnitTests`, `PerformanceTest`) for CTest compatibility. The directory is WulfNet-branded; binary names are cosmetic.

#### 3d. Delete old folders
- `UnitTests/`
- `PerformanceTest/`

---

### Phase 4: Merge Samples + JoltViewer into WulfNetExamples
**Effort:** Medium | **Risk:** Medium

#### 4a. Move Samples content directly into WulfNetExamples

Merge alongside the existing WulfNet CLI examples — no nesting:

```
WulfNetExamples/
├── CMakeLists.txt
│
│  ── Existing WulfNet CLI examples ──
├── ComputeExample/
├── EngineDemo/
├── HelloWulfNet/
├── IFSExample/
├── SoftRasterExample/
│
│  ── Merged from Samples/ (visual demo app) ──
├── SamplesApp.cpp / .h
├── Samples.cmake
├── Samples.h
├── SamplesLayers.h                 ← Renamed from Layers.h (avoids ambiguity)
├── Tests/
│   ├── Test.cpp / .h
│   ├── BroadPhase/
│   ├── Character/
│   ├── Constraints/
│   ├── ConvexCollision/
│   ├── General/
│   ├── Hair/
│   ├── Rig/
│   ├── ScaledShapes/
│   ├── Shapes/
│   ├── SoftBody/
│   ├── Tools/
│   ├── Vehicle/
│   ├── Water/
│   └── WulfNet/                    (existing WulfNet visual demos — 6 tests)
└── Utils/
    ├── ContactListenerImpl.cpp/.h
    ├── DebugRendererSP.h
    ├── RagdollLoader.cpp/.h
    ├── ShapeCreator.cpp/.h
    └── SoftBodyCreator.cpp/.h
```

#### 4b. Rename `Samples/Layers.h` → `SamplesLayers.h`
Different collision layer config from the test/perf versions. One `#include` update in `SamplesApp.h`.

#### 4c. Update Samples.cmake
- `SAMPLES_ROOT` → `${PHYSICS_REPO_ROOT}/WulfNetExamples`
- All `${SAMPLES_ROOT}/Tests/...` paths unchanged (Tests/ directory keeps structure)
- Link `WulfNetFramework` (was `TestFramework`) + `WulfNet`

#### 4d. Archive JoltViewer
Move `JoltViewer/` → `_ref/JoltViewer/`. Remove from the build.
154 lines of recording playback code — no WulfNet integration, low value to keep building.

#### 4e. Update WulfNetExamples/CMakeLists.txt
- Incorporate Samples target definition (from Samples.cmake)
- Keep existing 5 WulfNet CLI example targets
- All targets link `WulfNet`; Samples additionally links `WulfNetFramework`

#### 4f. Delete old folders
- `Samples/`
- `JoltViewer/`

---

### Phase 5: Build System Consolidation
**Effort:** Medium | **Risk:** High — test after every change

#### 5a. Root `CMakeLists.txt` = single canonical entry point

Builds everything:
1. `Jolt` — physics library (`include(Jolt/Jolt.cmake)`)
2. `WulfNetFramework` — app framework lib (`include(WulfNet/Framework/Framework.cmake)`)
3. `WulfNet` — engine library (`add_subdirectory(WulfNet)`)
4. Tests (if `WULFNET_BUILD_TESTS`):
   - `WulfNetTests` + `WulfNetExtendedTests` (WulfNet tests)
   - `UnitTests` (Jolt doctest tests)
   - `PerformanceTest` (Jolt benchmarks)
5. Examples (if `WULFNET_BUILD_EXAMPLES`):
   - `Samples` (visual demo app)
   - `HelloWulfNet`, `ComputeExample`, `IFSExample`, `SoftRasterExample`, `EngineDemo`

#### 5b. Simplify `build/CMakeLists.txt`
Convert to a thin delegator:
```cmake
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
set(PHYSICS_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../)
add_subdirectory(${PHYSICS_REPO_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/wulfnet)
```
Or update all `build/*.bat` / `build/*.sh` to use `-S ..` and remove `build/CMakeLists.txt` entirely.

#### 5c. Unified CMake options
```cmake
option(WULFNET_BUILD_TESTS    "Build all test suites"        ON)
option(WULFNET_BUILD_EXAMPLES "Build all examples and demos" ON)
option(WULFNET_ENABLE_TRACY   "Enable Tracy profiler"        OFF)
```
Remove: `TARGET_UNIT_TESTS`, `TARGET_HELLO_WORLD`, `TARGET_PERFORMANCE_TEST`, `TARGET_SAMPLES`, `TARGET_VIEWER`, `TARGET_WULFNET`, `TARGET_WULFNET_TESTS`, `TARGET_WULFNET_EXAMPLES`

#### 5d. Update build scripts
All `build/*.bat` and `build/*.sh` update their `-S` flag. Script filenames stay unchanged (they describe compiler/platform).

#### 5e. Clean up WulfNet/CMakeLists.txt
Remove references to deleted redirect headers from source/header file lists.

---

### Phase 6: Final Verification
**Effort:** Small | **Risk:** N/A

1. **Clean build** — `cmake -B out -S . -DCMAKE_BUILD_TYPE=Release && cmake --build out --config Release`
2. **WulfNet tests** — `WulfNetTests.exe` + `WulfNetExtendedTests.exe`: all 537 pass
3. **Jolt tests** — `UnitTests.exe`: all doctest tests pass
4. **Samples** — Launch, navigate WulfNet tests, verify all visual demos render
5. **CLI examples** — Run `HelloWulfNet`, `ComputeExample`, `IFSExample`, `SoftRasterExample`, `EngineDemo`
6. **Benchmarks** — `PerformanceTest.exe` runs successfully
7. **Update docs** — `README.md`, `ENGINE_PLAN.md` build instructions

---

## Execution Order & Dependencies

```
Phase 1 (Cleanup — no structural changes)
    ├── 1a: Delete 7 redirect headers
    ├── 1b: Update ~15 #include paths
    ├── 1c: Replace GPUMat4x4 + SoftVec aliases
    └── 1d: Delete .bak file
         │
Phase 2 (TestFramework → WulfNet/Framework/)
    ├── 2a: Move + rename directory
    ├── 2b: Update Framework.cmake paths
    └── 2c: Update Framework.h
         │
         ├───────────────────────────────────────────┐
         │                                           │
Phase 3 (UnitTests + PerfTest → WulfNetTests/)     Phase 4 (Samples + JoltViewer → WulfNetExamples/)
    ├── 3a: Move files flat + rename Layers.h        ├── 4a: Move Samples content flat
    ├── 3b: Keep dual test frameworks                ├── 4b: Rename Samples/Layers.h
    ├── 3c: Update WulfNetTests/CMakeLists.txt       ├── 4c: Archive JoltViewer to _ref/
    └── 3d: Delete UnitTests/ + PerformanceTest/     ├── 4d: Update WulfNetExamples/CMakeLists.txt
         │                                           └── 4e: Delete Samples/ + JoltViewer/
         └──────────────┬────────────────────────────┘
                        │
                  Phase 5 (Build System Consolidation)
                    ├── 5a: Root CMakeLists.txt = single entry point
                    ├── 5b: Simplify build/CMakeLists.txt
                    ├── 5c: Unify CMake options
                    ├── 5d: Update build scripts
                    └── 5e: Clean up source lists
                        │
                  Phase 6 (Verification)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| `Layers.h` naming collision | Rename to `UnitTestLayers.h`, `PerfTestLayers.h`, `SamplesLayers.h` — 1 include update each |
| Breaking 537 WulfNet tests | TestHarness.h stays unchanged; files move with their infrastructure |
| Breaking Jolt doctest tests | doctest.h + UnitTestFramework move together; no framework change |
| Samples visual regression | Entire test tree moves intact; only CMake root path changes |
| `#include` breakage | CMake `target_include_directories` resolves paths; source-level includes use relative paths from the target root |
| Build script chaos | Update scripts one at a time; verify build after each phase |

---

## Estimated Effort

| Phase | Files Touched | Estimated Time |
|-------|--------------|---------------|
| Phase 1: Cleanup | ~20 files (edits) | 1-2 hours |
| Phase 2: Framework merge | ~100 files (moves) + 3 CMake edits | 1-2 hours |
| Phase 3: Tests merge | ~90 files (moves) + CMake rewrite | 3-4 hours |
| Phase 4: Examples merge | ~250 files (moves) + CMake rewrite | 2-3 hours |
| Phase 5: Build consolidation | ~15 CMake/script files | 2-3 hours |
| Phase 6: Verification | N/A | 1-2 hours |
| **Total** | **~475 files** | **10-16 hours** |

---

## Post-Integration File Structure (Final)

```
wulfnet-engine/
├── CMakeLists.txt                      ← Single build entry point
├── ENGINE_PLAN.md
├── README.md
│
├── Jolt/                               ← Upstream physics library (UNCHANGED)
│
├── WulfNet/                            ← Engine library
│   ├── CMakeLists.txt
│   ├── WulfNet.h                       (umbrella header)
│   ├── Engine.cpp / .h                 (engine lifecycle)
│   ├── EngineConfig.h / API.h / Version.h / ForwardDecl.h
│   ├── Audio/                          (AudioMixer, AcousticSystem, SpatialAudio)
│   ├── Compute/                        (Vulkan context, pipelines, fluids, reduction)
│   ├── Core/                           (Logger, Math, Memory, Profiling, System, Threading)
│   ├── Physics/                        (PhysicsWorld, Fluids, Gaseous, Destruction, MPM, Terrain)
│   ├── Procedural/                     (IFS transforms, presets, blender, system)
│   ├── Rendering/                      (RenderPipeline, SoftwareRasterizer, Lighting, Effects, Types)
│   │   └── (redirect headers REMOVED)
│   └── Framework/                      ← MERGED from TestFramework/
│       ├── Framework.cmake
│       ├── Framework.h
│       ├── Application/
│       ├── External/
│       ├── Image/
│       ├── Input/                      (+ Win/Linux/MacOS backends)
│       ├── Renderer/                   (+ DX12/VK/MTL backends)
│       ├── UI/
│       ├── Utils/
│       └── Window/                     (+ Win/Linux/MacOS backends)
│
├── WulfNetTests/                       ← ALL tests + benchmarks
│   ├── CMakeLists.txt                  (4 targets: WulfNetTests, WulfNetExtendedTests, UnitTests, PerformanceTest)
│   │
│   │  ── WulfNet tests (TestHarness) ──
│   ├── TestHarness.h
│   ├── BenchmarkHarness.h
│   ├── WulfNetTests.cpp / WulfNetExtendedTests.cpp
│   ├── CoreTests.cpp
│   ├── PhysicsWorldTests.cpp
│   ├── VulkanComputeTests.cpp
│   ├── IFSTransformTests.cpp
│   ├── SoftwareRendererTests.cpp
│   ├── PipelineIntegrationTests.cpp
│   ├── SystemMonitorTests.cpp
│   ├── AdvancedPhysicsTests.cpp
│   ├── IntegrationTests.cpp
│   ├── ConstitutiveModelTests.cpp
│   ├── TerrainDeformationTests.cpp
│   ├── MPMRigidCouplingTests.cpp
│   ├── GaseousSystemTests.cpp
│   ├── DestructionSystemTests.cpp
│   ├── ShadowMapTests.cpp
│   ├── GlobalIlluminationTests.cpp
│   ├── VolumetricRendererTests.cpp
│   ├── RenderPipelineTests.cpp
│   ├── AudioEngineTests.cpp
│   ├── AcousticSystemTests.cpp
│   ├── SpatialAudioTests.cpp
│   ├── WaterSystemV3Tests.cpp
│   ├── EngineLifecycleTests.cpp
│   ├── ThreadingTests.cpp
│   ├── FrameAllocatorTests.cpp
│   ├── PerformanceBenchmarks.cpp
│   │
│   │  ── Jolt tests (doctest) — merged from UnitTests/ ──
│   ├── doctest.h
│   ├── UnitTestFramework.cpp / .h
│   ├── PhysicsTestContext.cpp / .h
│   ├── UnitTestLayers.h               ← Renamed from Layers.h
│   ├── LoggingBodyActivationListener.h
│   ├── LoggingContactListener.h
│   ├── LoggingCharacterContactListener.h
│   ├── Core/                           (ArrayTest, BinaryHeapTest, JobSystemTest, ... 14 files)
│   ├── Geometry/                       (ClosestPointTests, ConvexHullBuilderTest, ... 7 files)
│   ├── Math/                           (BVec16Tests, DMat44Tests, Vec3Tests, ... 14 files)
│   ├── Physics/                        (BroadPhaseTests, CastShapeTests, SoftBodyTests, ... 33 files)
│   ├── Compute/                        (ComputeTests.cpp)
│   ├── ObjectStream/                   (ObjectStreamTest.cpp)
│   │
│   │  ── Benchmarks — merged from PerformanceTest/ ──
│   └── Benchmarks/
│       ├── PerformanceTest.cpp
│       ├── PerformanceTestScene.h
│       ├── PerfTestLayers.h            ← Renamed from Layers.h
│       ├── PyramidScene.h
│       ├── RagdollScene.h
│       ├── ConvexVsMeshScene.h
│       ├── CharacterVirtualScene.h
│       ├── LargeMeshScene.h
│       └── MaxBodiesScene.h
│
├── WulfNetExamples/                    ← ALL examples + visual demos
│   ├── CMakeLists.txt                  (6 targets: Samples + 5 CLI examples)
│   │
│   │  ── WulfNet CLI examples (existing) ──
│   ├── ComputeExample/
│   ├── EngineDemo/
│   ├── HelloWulfNet/
│   ├── IFSExample/
│   ├── SoftRasterExample/
│   │
│   │  ── Visual demo app — merged from Samples/ ──
│   ├── SamplesApp.cpp / .h
│   ├── Samples.cmake
│   ├── Samples.h
│   ├── SamplesLayers.h                 ← Renamed from Layers.h
│   ├── Tests/
│   │   ├── Test.cpp / .h
│   │   ├── BroadPhase/
│   │   ├── Character/
│   │   ├── Constraints/
│   │   ├── ConvexCollision/
│   │   ├── General/
│   │   ├── Hair/
│   │   ├── Rig/
│   │   ├── ScaledShapes/
│   │   ├── Shapes/
│   │   ├── SoftBody/
│   │   ├── Tools/
│   │   ├── Vehicle/
│   │   ├── Water/
│   │   └── WulfNet/                    (DamBreak, WaterBox, WulfNetPhysics, Advanced, WaterV3)
│   └── Utils/
│       ├── ContactListenerImpl.cpp/.h
│       ├── DebugRendererSP.h
│       ├── RagdollLoader.cpp/.h
│       ├── ShapeCreator.cpp/.h
│       └── SoftBodyCreator.cpp/.h
│
├── Assets/                             ← Unchanged
├── build/                              ← Simplified (delegates to root CMakeLists.txt)
├── cmake/                              ← CMake helpers
└── _ref/                               ← Archived
    ├── BG-C-Software-Renderer/
    ├── Iterated-Function-Systems/
    └── JoltViewer/                     ← ARCHIVED from JoltViewer/

DELETED (fully absorbed, no Jolt-named folders remain):
  TestFramework/     → WulfNet/Framework/
  UnitTests/         → WulfNetTests/ (flat merge)
  PerformanceTest/   → WulfNetTests/Benchmarks/
  Samples/           → WulfNetExamples/ (flat merge)
  JoltViewer/        → _ref/JoltViewer/
```
