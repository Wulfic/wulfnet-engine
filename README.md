# WulfNet Engine

A **fully-featured, AAA-grade physics and game engine** built on top of [Jolt Physics](https://github.com/jrouwe/JoltPhysics). WulfNet extends Jolt's battle-tested rigid body physics with advanced simulations including fluids, deformables, destruction, and a complete rendering/audio pipeline.

## 🎯 Project Vision

WulfNet Engine leverages Jolt Physics (used in Horizon Forbidden West and Death Stranding 2) as its foundation, focusing development on **extending capabilities** rather than reinventing solved problems.

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

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     WULFNET ENGINE LAYER                         │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│   Extended   │   Renderer   │    Audio     │    Scene Graph    │
│   Physics    │   (Vulkan)   │   System     │                   │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│                    JOLT PHYSICS FOUNDATION                       │
├─────────────────────────────────────────────────────────────────┤
│  Rigid Bodies  │  Soft Bodies  │  Vehicles  │  Constraints      │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Getting Started

### Prerequisites
- **CMake** 3.25 or higher
- **C++20** compatible compiler
  - Visual Studio 2022 (Windows)
  - GCC 11+ / Clang 14+ (Linux)
  - Xcode 14+ (macOS)
- **Vulkan SDK** (optional, for rendering/GPU compute)

### Building

#### Windows (Visual Studio 2022)
```bash
cd Build
cmake_vs2022_cl.bat
# Open Build/VS2022_CL/WulfNetEngine.sln
```

#### Linux
```bash
cd Build
./cmake_linux_clang_gcc.sh Release clang++
cd Linux_Release
make -j$(nproc)
```

#### macOS
```bash
cd Build
./cmake_xcode_macos.sh
# Open Build/XCode_macOS/WulfNetEngine.xcodeproj
```

### Running Samples

```bash
# Run Jolt's sample viewer (physics demos)
./bin/JoltViewer

# Run performance benchmarks
./bin/PerformanceTest
```

## 📁 Project Structure

```
wulfnet-engine/
├── Jolt/                 # Jolt Physics core (upstream)
├── JoltViewer/           # Interactive physics demos
├── Samples/              # Physics test scenes
├── TestFramework/        # Test utilities & debug renderer
├── UnitTests/            # Jolt unit tests
├── PerformanceTest/      # Benchmarks
│
├── WulfNet/              # WulfNet extensions (coming soon)
│   ├── Physics/          # Fluids, MPM, destruction
│   ├── Compute/          # GPU compute layer
│   ├── Rendering/        # Vulkan renderer
│   └── Audio/            # Acoustic simulation
│
├── Build/                # Platform-specific build scripts
├── Assets/               # Shared assets
└── docs/                 # Documentation
```

## 📊 Performance Targets

| System | Target | Source |
|--------|--------|--------|
| Rigid Bodies | 25,000 active @ 60 FPS | Jolt |
| Soft Body Particles | 100,000 @ 60 FPS | Jolt |
| Hair Strands | 100,000 @ 60 FPS | Jolt (GPU) |
| Fluid Particles | 1,000,000 @ 60 FPS | WulfNet (GPU) |
| MPM Particles | 500,000 @ 60 FPS | WulfNet (GPU) |

## 📖 Documentation

- [**ENGINE_PLAN.md**](ENGINE_PLAN.md) - Full technical architecture and roadmap
- [**docs/Architecture.md**](docs/Architecture.md) - Jolt Physics architecture
- [**docs/Samples.md**](docs/Samples.md) - Sample documentation

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

*WulfNet Engine - Built for performance, designed for extensibility*
