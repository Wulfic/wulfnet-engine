# WulfNet Engine

A **fully-featured, AAA-grade physics and game engine** built on top of [Jolt Physics](https://github.com/jrouwe/JoltPhysics). WulfNet extends Jolt's battle-tested rigid body physics with advanced simulations including fluids, deformables, destruction, and a complete rendering/audio pipeline.

**Version 1.0.0** — 537 tests, 100% pass rate

## Quick Start

```cpp
#include <WulfNet/WulfNet.h>
#include <WulfNet/Engine.h>

int main() {
    using namespace WulfNet;

    Logger::Initialize();

    // Configure the engine (HeadlessPhysics = physics + compute, no rendering/audio)
    EngineConfig config = EngineConfig::HeadlessPhysics();
    config.appName = "MyApp";

    Engine engine;
    if (engine.Initialize(config) != EngineInitResult::Success)
        return 1;

    // Game loop
    while (engine.IsRunning()) {
        engine.BeginFrame();

        PhysicsWorld& physics = engine.GetPhysics();
        // ... your game logic ...

        engine.EndFrame();  // Steps physics at fixed timestep
    }

    engine.Shutdown();
    return 0;
}
```

See [WulfNetExamples/EngineDemo](WulfNetExamples/EngineDemo/main.cpp) for a complete example with physics scene setup, frame simulation, and PPM image output.

| Jolt Physics Provides | WulfNet Engine Adds |
|-----------------------|---------------------|
| Rigid body dynamics | GPU-accelerated physics |
| Soft bodies (cloth, volumetric) | Fluid dynamics (SPH, FLIP, APIC) |
| Vehicles (wheeled, tracked) | Gaseous simulation (smoke, fire) |
| Ragdolls & characters | MPM deformables (mud, sand, snow) |
| Constraints & joints | Destruction physics |
| Hair simulation (GPU) | PBR rendering pipeline |
| Buoyancy | Acoustic simulation |

## ✨ Features

### From Jolt Physics (Included)
- **Rigid Body Simulation** - High-performance multi-threaded solver
- **Collision Detection** - Sphere, Box, Capsule, Convex Hull, Mesh, HeightField
- **Constraints** - Fixed, Hinge, Slider, Cone, Distance, 6-DOF, and more
- **Soft Bodies** - XPBD-based cloth, volumetric deformables
- **Vehicles** - Wheeled, tracked, motorcycles
- **Characters** - Rigid body and virtual character controllers
- **Hair Simulation** - GPU-accelerated strand simulation

### WulfNet Extensions (In Development)
- **Fluid Dynamics** - SPH/FLIP/APIC solvers (GPU)
- **Material Point Method** - Mud, sand, snow simulation (GPU)
- **Gaseous Physics** - Smoke, fire, explosions
- **Destruction** - Voronoi fracture with Jolt integration
- **Terrain Deformation** - Real-time heightfield modification
- **Vulkan Renderer** - PBR materials, GI, volumetrics
- **Acoustic Simulation** - Ray-traced reverb, HRTF spatial audio

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                              │
│    Game Logic  │  EngineDemo  │  HelloWulfNet  │  Custom App         │
├──────────────────────────────────────────────────────────────────────┤
│              WulfNet::Engine  (single entry point)                    │
│  Initialize() → BeginFrame() / EndFrame() → Shutdown()               │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  PhysicsWorld│  RenderPipeline│ AudioMixer  │  VulkanContext        │
│  (Jolt + ext)│  (IRenderer) │  (multi-src) │  (GPU compute)        │
├──────────────┤──────────────┤──────────────┤────────────────────────┤
│ FluidSystem  │ ShadowMap    │ AcousticSys  │ ComputePipeline       │
│ GaseousSystem│ GlobalIllum  │ SpatialAudio │ ComputeBuffer<T>      │
│ Destruction  │ Volumetric   │              │ ParallelReduction     │
│ TerrainDeform│ Deferred     │              │                        │
│ MPMCoupling  │ OcclusionCull│              │                        │
├──────────────┴──────────────┴──────────────┴────────────────────────┤
│                     JOLT PHYSICS FOUNDATION                          │
│   Rigid Bodies │ Soft Bodies │ Vehicles │ Constraints │ Collision   │
├──────────────────────────────────────────────────────────────────────┤
│                        CORE LAYER                                     │
│   Logger │ Profiler (Tracy) │ SystemMonitor │ MathTypes │ Platform  │
└──────────────────────────────────────────────────────────────────────┘
```

## Getting Started

### Prerequisites
- **CMake** 3.20 or higher
- **C++17** compatible compiler
  - Visual Studio 2022+ (Windows)
  - GCC 11+ / Clang 14+ (Linux)
  - Xcode 14+ (macOS)
- **Vulkan SDK** (optional, for GPU compute)

### Building

#### Windows (Visual Studio 2022)
```bash
cd build
cmake_vs2022_cl.bat
cmake --build VS2022_CL --config Release

# Or use CMake presets:
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
```

#### Linux
```bash
cd build
./cmake_linux_clang_gcc.sh Release clang++
cmake --build Linux_Release -j$(nproc)

# Or use CMake presets:
cmake --preset linux-clang-release
cmake --build --preset linux-clang-release
```

### Running Tests

```bash
cd build

# Core tests (84 tests)
./VS2022_CL/WulfNetTests/Release/WulfNetTests.exe

# Extended tests (453 tests, 19 suites)
./VS2022_CL/WulfNetTests/Release/WulfNetExtendedTests.exe

# Run a specific suite
./VS2022_CL/WulfNetTests/Release/WulfNetExtendedTests.exe --suite=engine
./VS2022_CL/WulfNetTests/Release/WulfNetExtendedTests.exe --suite=audio
./VS2022_CL/WulfNetTests/Release/WulfNetExtendedTests.exe --suite=benchmark
```

### Running Examples

```bash
cd build

# Flagship demo (physics + rendering + audio)
./VS2022_CL/WulfNetExamples/Release/EngineDemo.exe

# Basic physics example
./VS2022_CL/WulfNetExamples/Release/HelloWulfNet.exe

# GPU compute demonstration
./VS2022_CL/WulfNetExamples/Release/ComputeExample.exe
```

## Project Structure

```
wulfnet-engine/
├── Jolt/                  # Jolt Physics core (upstream, never modify)
├── WulfNet/               # WulfNet Engine library
│   ├── Engine.h/cpp       # Single entry point
│   ├── EngineConfig.h     # Configuration struct + presets
│   ├── Version.h          # Version macros (1.0.0)
│   ├── API.h              # Export/import macros
│   ├── ForwardDecl.h      # 130+ forward declarations
│   ├── Core/              # Logger, Profiler, SystemMonitor, Math
│   ├── Physics/           # PhysicsWorld, Fluids, MPM, Gas, Destruction, Terrain
│   ├── Compute/           # VulkanContext, ComputePipeline, ComputeBuffer
│   ├── Rendering/         # RenderPipeline, SoftwareRasterizer, Lighting, Effects
│   ├── Procedural/        # IFS fractal system
│   └── Audio/             # AudioMixer, AcousticSystem, SpatialAudio
├── WulfNetTests/          # 537 tests (84 core + 453 extended)
├── WulfNetExamples/       # 5 examples (HelloWulfNet, EngineDemo, etc.)
├── build/                 # Build scripts and CMake presets
├── docs/                  # API reference, architecture docs
└── Assets/                # Shaders, fonts, models
```

## 📊 Performance Targets

| System | Target | Source |
|--------|--------|--------|
| Rigid Bodies | 25,000 active @ 60 FPS | Jolt |
| Soft Body Particles | 100,000 @ 60 FPS | Jolt |
| Hair Strands | 100,000 @ 60 FPS | Jolt (GPU) |
| Fluid Particles | 1,000,000 @ 60 FPS | WulfNet (GPU) |
| MPM Particles | 500,000 @ 60 FPS | WulfNet (GPU) |

## Documentation

- [**docs/APIReference.md**](docs/APIReference.md) - Complete API reference (v1.0.0)
- [**ENGINE_PLAN.md**](ENGINE_PLAN.md) - Technical architecture and roadmap
- [**REFACTOR_MASTER_PLAN.md**](REFACTOR_MASTER_PLAN.md) - 10-pass refactor plan
- [**docs/Architecture.md**](docs/Architecture.md) - Jolt Physics architecture

## 🤝 Contributing

Contributions welcome! Please follow these principles:

1. **Don't modify Jolt/** - Keep upstream changes minimal for easy updates
2. **GPU-first for new physics** - Use compute shaders for heavy workloads
3. **Comprehensive testing** - Unit tests for all new systems
4. **Document as you go** - Update docs with each feature

## 📜 License

WulfNet Engine extensions are licensed under [MIT License](LICENSE).

Jolt Physics is licensed under the [MIT License](https://github.com/jrouwe/JoltPhysics/blob/master/LICENSE).

## 🙏 Acknowledgments

- [**Jolt Physics**](https://github.com/jrouwe/JoltPhysics) by Jorrit Rouwe - The foundation of this engine
- The Jolt Physics community and contributors

---

*WulfNet Engine v1.0.0 — Built for performance, designed for extensibility*
