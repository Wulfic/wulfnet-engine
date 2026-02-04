# WulfNet Engine

<p align="center">
  <strong>A High-Performance, AAA-Grade Game Engine</strong><br>
  Built for massive parallelization and modern hardware
</p>

---

## Overview

WulfNet Engine is a fully-featured, high-performance game engine designed for modern AAA game development. Built from the ground up with **C++20**, it leverages cutting-edge techniques including:

- **Massive Parallelization**: Scales to 64+ cores and 128+ threads
- **SIMD Optimization**: SSE4.2, AVX2, and AVX512 support
- **Lock-Free Job System**: Work-stealing scheduler with fiber-based coroutines
- **Custom Memory Allocators**: Linear, pool, and stack allocators with cache-line alignment
- **Cross-Platform**: Windows and Linux support

## Requirements

### Build Requirements

| Requirement | Version |
|-------------|---------|
| CMake | 3.25+ |
| C++ Compiler | C++20 support (MSVC 2022+, GCC 12+, Clang 15+) |
| Vulkan SDK | 1.3+ (optional, for rendering) |

### Supported Platforms

| Platform | Architecture | Status |
|----------|--------------|--------|
| Windows 10/11 | x64 | ✅ Primary |
| Linux | x64 | ✅ Supported |

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/your-org/wulfnet-engine.git
cd wulfnet-engine
```

### Configure and Build (Windows)

```powershell
# Configure with CMake
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release
```

### Configure and Build (Linux)

```bash
# Configure with CMake
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `WULFNET_BUILD_TESTS` | ON | Build unit tests |
| `WULFNET_BUILD_EXAMPLES` | ON | Build example applications |
| `WULFNET_BUILD_DOCS` | OFF | Build documentation |
| `WULFNET_ENABLE_PROFILING` | OFF | Enable Tracy profiler integration |
| `WULFNET_SIMD_LEVEL` | AUTO | SIMD level: AUTO, SSE42, AVX2, AVX512 |

## Architecture

```
wulfnet-engine/
├── src/
│   ├── core/           # Core utilities, types, memory, jobs
│   │   ├── memory/     # Custom allocators
│   │   ├── math/       # SIMD math library
│   │   ├── jobs/       # Job system and fibers
│   │   └── platform/   # Platform abstraction
│   ├── physics/        # Physics engine (coming soon)
│   ├── rendering/      # Vulkan/DX12 renderer (coming soon)
│   └── audio/          # Audio system (coming soon)
├── examples/           # Example applications
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Core Systems

### Memory System

WulfNet uses custom allocators for deterministic performance:

```cpp
#include "core/memory/Memory.h"

using namespace WulfNet;

// Linear allocator for frame-temporary data
LinearAllocator frameAlloc(1024 * 1024);  // 1MB
auto* temp = frameAlloc.alloc<MyStruct>(16);

// Pool allocator for fixed-size objects
PoolAllocator<Entity> entityPool(1024);
Entity* entity = entityPool.alloc();

// Stack allocator for LIFO allocations
StackAllocator stackAlloc(64 * 1024);
auto marker = stackAlloc.getMarker();
// ... allocations ...
stackAlloc.freeToMarker(marker);
```

### Math Library (SIMD)

Hardware-accelerated math with automatic SIMD dispatch:

```cpp
#include "core/math/Math.h"

using namespace WulfNet;

Vec3 position(1.0f, 2.0f, 3.0f);
Vec3 velocity(0.1f, 0.0f, 0.0f);
position += velocity * deltaTime;

Quat rotation = Quat::fromAxisAngle(Vec3::up(), radians(45.0f));
Vec3 forward = rotation.rotate(Vec3::forward());

Mat4 transform = Mat4::translation(position) * Mat4::fromQuat(rotation);
```

### Job System

Lock-free, work-stealing job scheduler:

```cpp
#include "core/jobs/JobSystem.h"

using namespace WulfNet;

JobSystem& jobs = JobSystem::get();

// Simple parallel for
std::vector<Entity> entities(10000);
jobs.parallelFor(0, entities.size(), [&](u32 start, u32 end) {
    for (u32 i = start; i < end; i++) {
        updateEntity(entities[i]);
    }
}, 256);  // Batch size

// Job with dependencies
Job physicsJob = jobs.createJob([](void*) {
    simulatePhysics();
});

Job renderJob = jobs.createJob([](void*) {
    renderScene();
}, &physicsJob);  // Depends on physics

jobs.submit(physicsJob);
jobs.submit(renderJob);
jobs.wait(renderJob);
```

### Logging

Thread-safe logging with multiple severity levels:

```cpp
#include "core/Log.h"

using namespace WulfNet;

WULFNET_LOG_INFO("Engine initialized");
WULFNET_LOG_WARN("Low memory: {} MB remaining", availableMemory);
WULFNET_LOG_ERROR("Failed to load resource: {}", path);
```

### Platform Abstraction

Unified API for platform-specific operations:

```cpp
#include "core/platform/Platform.h"

using namespace WulfNet::Platform;

SystemInfo info = getSystemInfo();
LOG_INFO("CPU: {} cores, {} threads", info.numPhysicalCores, info.numLogicalCores);
LOG_INFO("RAM: {} GB", info.totalSystemMemory / (1024 * 1024 * 1024));

// High-precision timing
f64 start = getTimeSeconds();
// ... work ...
f64 elapsed = getTimeSeconds() - start;

// Thread utilities
setThreadName("Worker_0");
setThreadAffinity(0);  // Pin to core 0
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Frame Budget | 16.67ms (60 FPS) |
| Physics Step | < 4ms for 10K rigid bodies |
| Job Latency | < 1μs dispatch overhead |
| Memory Overhead | < 5% fragmentation |

## Development Roadmap

### Phase 1: Foundation ✅ (Current)
- [x] Build system and project structure
- [x] Core types and utilities
- [x] Memory allocators
- [x] SIMD math library
- [x] Job system
- [x] Platform abstraction

### Phase 2: Physics (Upcoming)
- [ ] Collision detection (broadphase/narrowphase)
- [ ] Rigid body dynamics
- [ ] Constraint solver
- [ ] Spatial partitioning

### Phase 3: Rendering
- [ ] Vulkan backend
- [ ] DX12 backend
- [ ] PBR materials
- [ ] Shadow mapping
- [ ] Post-processing

### Phase 4: Game Systems
- [ ] Entity-Component System
- [ ] Scene management
- [ ] Resource pipeline
- [ ] Audio system

## Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) before submitting pull requests.

### Code Style

- C++20 with modern idioms
- RAII for resource management
- No exceptions in hot paths
- Prefer `constexpr` where possible
- 4-space indentation

## License

WulfNet Engine is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built with ❤️ for high-performance game development</em>
</p>
