# WulfNet Engine Extensions

This directory contains WulfNet's extensions to Jolt Physics.

## Directory Structure (Planned)

```
WulfNet/
├── Core/                   # Extended utilities
│   ├── Logging/            # Logging infrastructure
│   ├── Profiling/          # Tracy integration
│   └── Platform/           # Additional platform utilities
│
├── Physics/                # Extended physics systems
│   ├── Fluids/             # SPH, FLIP, APIC
│   ├── MPM/                # Material Point Method
│   ├── Gaseous/            # Smoke, fire, explosions
│   ├── Destruction/        # Fracture physics
│   ├── Terrain/            # Deformable terrain
│   └── Integration/        # Jolt integration layer
│
├── Compute/                # GPU compute infrastructure
│   ├── Vulkan/             # Vulkan compute backend
│   ├── Shaders/            # Compute shaders (HLSL)
│   └── Memory/             # GPU memory management
│
├── Rendering/              # Rendering pipeline
│   ├── Backend/            # Vulkan abstraction
│   ├── Pipeline/           # Render passes
│   ├── Materials/          # PBR materials
│   └── Effects/            # Volumetrics, post-process
│
└── Audio/                  # Audio & acoustics
    ├── Core/               # Mixer, sources
    ├── Acoustics/          # Ray-traced reverb
    └── Spatial/            # HRTF, Ambisonics
```

## Status

🚧 **In Development** - Phase 2 (Core Setup) is in progress.

See [ENGINE_PLAN.md](../ENGINE_PLAN.md) for the full roadmap.
