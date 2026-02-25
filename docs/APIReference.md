# WulfNet Engine — API Reference

*Version 0.8.0 — February 2026*

This document provides a concise reference for all public APIs in the WulfNet Engine library.
For full Doxygen-generated documentation, build the `docs` target (`doxygen Doxyfile` from the repo root).

---

## Table of Contents

1. [Core Systems](#1-core-systems)
2. [Physics Systems](#2-physics-systems)
3. [GPU Compute Systems](#3-gpu-compute-systems)
4. [Procedural Systems](#4-procedural-systems)
5. [Rendering Systems](#5-rendering-systems)
6. [Audio Systems](#6-audio-systems)

---

## 1. Core Systems

### 1.1 Logger (`WulfNet::Logger`)

**Header:** `WulfNet/Core/Logging/Logger.h`

Thread-safe, multi-sink logging with color console output, file output, and callback hooks.

| Method | Description |
|--------|-------------|
| `static Logger& Get()` | Singleton accessor |
| `void SetMinLevel(LogLevel level)` | Minimum severity to emit |
| `void Log(LogLevel, const char* fmt, ...)` | Variadic log |
| `void AddFileSink(const char* path)` | Log to file |
| `void AddCallbackSink(LogCallback cb)` | Custom callback sink |
| `void Flush()` | Flush all sinks |

**Log Levels:** `Trace`, `Debug`, `Info`, `Warning`, `Error`, `Fatal`

**Macros:** `WULFNET_LOG_TRACE(fmt, ...)`, `WULFNET_LOG_DEBUG(...)`, `WULFNET_LOG_INFO(...)`, `WULFNET_LOG_WARN(...)`, `WULFNET_LOG_ERROR(...)`, `WULFNET_LOG_FATAL(...)`

---

### 1.2 Profiler (`WulfNet::Profiler`, `WulfNet::ScopedTimer`)

**Header:** `WulfNet/Core/Profiling/Profiler.h`

Lightweight profiling with Tracy integration and standalone fallback.

| Type | Method / Usage |
|------|---------------|
| `ScopedTimer` | RAII timer — measures scope lifetime |
| `ManualTimer` | `Start()`, `Stop()`, `ElapsedMs()` |
| `Profiler::Get()` | Singleton with frame tracking |
| `WULFNET_PROFILE_SCOPE(name)` | Macro — auto ScopedTimer |
| `WULFNET_PROFILE_FRAME()` | Macro — end-of-frame marker |

---

### 1.3 SystemMonitor (`WulfNet::SystemMonitor`)

**Header:** `WulfNet/Core/System/SystemMonitor.h`

Real-time CPU, RAM, GPU, VRAM monitoring with NVML integration.

| Method | Description |
|--------|-------------|
| `bool Initialize()` | Detect hardware |
| `void Update()` | Refresh all metrics |
| `float GetCPUUsage()` | 0–100% |
| `uint64_t GetRAMUsedBytes()` / `Total` | System memory |
| `float GetGPUUsage()` | 0–100% (NVML) |
| `uint64_t GetVRAMUsedBytes()` / `Total` | Video memory (NVML) |

---

## 2. Physics Systems

### 2.1 PhysicsWorld (`WulfNet::PhysicsWorld`)

**Header:** `WulfNet/Physics/Integration/PhysicsWorld.h`

Wraps Jolt Physics with body creation, constraint management, queries, and callbacks.

| Method | Description |
|--------|-------------|
| `bool Initialize(const PhysicsWorldSettings&)` | Create Jolt system |
| `void Shutdown()` | Destroy |
| `void Step(float dt)` | Advance simulation |
| `BodyHandle CreateBody(const BodyDef&)` | Add rigid body |
| `void DestroyBody(BodyHandle)` | Remove body |
| `void AddConstraint(...)` | Create joint |
| `RayCastResult RayCast(...)` | Ray query |
| `void SetContactCallback(...)` | Collision events |

---

### 2.2 Fluid Systems

#### FluidGrid (`WulfNet::FluidGrid`)

**Header:** `WulfNet/Physics/Fluids/FluidGrid.h`

MAC-staggered velocity grid for FLIP/APIC simulation.

| Method | Description |
|--------|-------------|
| `bool Initialize(resX, resY, resZ, cellSize)` | Allocate grid |
| `void Reset()` | Clear all cells |
| `MACCell& GetCell(i, j, k)` | Direct cell access |
| `void InterpolateVelocity(gx, gy, gz, &vx, &vy, &vz)` | Trilinear velocity sample |
| `void WorldToGrid(...)` / `GridToWorld(...)` | Coordinate conversion |
| `float ComputeDivergence()` | For pressure solve |
| `void ApplyPressureGradient()` | Enforce incompressibility |
| `void ExtrapolateVelocity(layers)` | Fill empty cells |

#### FluidSystem (`WulfNet::FluidSystem`)

**Header:** `WulfNet/Physics/Fluids/FluidSystem.h`

MPM+APIC particle-based fluid solver with 6 material presets.

**Presets:** Water, Oil, Honey, Mud, Lava, Blood

#### COFLIPSystem (`WulfNet::COFLIPSystem`)

**Header:** `WulfNet/Physics/Fluids/COFLIPSystem.h`

SIGGRAPH Asia 2024 CO-FLIP method — combined PIC/FLIP with full CPU+GPU paths.

| Method | Description |
|--------|-------------|
| `bool Initialize(const COFLIPConfig&)` | Setup solver |
| `void Step(float dt)` | Advance (P2G → Pressure → G2P) |
| `void AddParticles(...)` | Emit fluid particles |
| `bool InitializeGPU(VulkanContext*)` | Enable GPU path |

#### FluidSurface (`WulfNet::FluidSurface`)

**Header:** `WulfNet/Physics/Fluids/FluidSurface.h`

Marching cubes mesh extraction from particle/grid density fields.

| Method | Description |
|--------|-------------|
| `bool Initialize(const SurfaceConfig&)` | Allocate voxel grid |
| `void SplatParticles(particles, count)` | Deposit density |
| `void SmoothDensity(passes)` | Gaussian blur |
| `void ExtractSurface(isovalue)` | March cubes → triangles |
| `const SurfaceVertex* GetVertices()` | Output mesh data |

---

### 2.3 MPM (Material Point Method)

#### ConstitutiveModel (`WulfNet::ConstitutiveModel`)

**Header:** `WulfNet/Physics/MPM/ConstitutiveModel.h`

Abstract base for MPM stress computation + deformation projection.

**Concrete models:**

| Model | Material Types |
|-------|---------------|
| `NeoHookeanModel` | Rubber, Flesh |
| `DruckerPragerModel` | Sand, WetMud, DrySoil |
| `SnowModel` | Snow, Ice |
| `ViscousFluidModel` | Viscous liquids |

**Factory:** `const ConstitutiveModel* GetConstitutiveModel(MPMMaterialType type)`

**Presets:** `MPMMaterialParams::Rubber()`, `::Flesh()`, `::Sand()`, `::WetMud()`, `::DrySoil()`, `::Snow()`, `::Ice()`, `::ViscousFluid(viscosity, density)`

**Utility:** `SVDResult ComputeSVD3x3(const Mat3& A)` — Jacobi iteration 3×3 SVD

#### MPMRigidCoupling (`WulfNet::MPMRigidCoupling`)

**Header:** `WulfNet/Physics/MPM/MPMRigidCoupling.h`

Penalty-based bidirectional MPM ↔ rigid body forces via SDF queries.

| Method | Description |
|--------|-------------|
| `void AddCollider(const MPMCollider&)` | Register rigid body collider |
| `void ComputeCouplingForces(particles, count)` | Evaluate forces |
| `void ApplyForcesToBodies(PhysicsWorld&)` | Push forces to Jolt |

---

### 2.4 Terrain Deformation (`WulfNet::TerrainDeformation`)

**Header:** `WulfNet/Physics/Terrain/TerrainDeformation.h`

Runtime heightfield modification with volume conservation.

| Method | Description |
|--------|-------------|
| `bool Initialize(const TerrainConfig&)` | Allocate height grid |
| `void ApplyDeformation(x, z, shape, depth)` | Deform terrain |
| `void ApplyTireTrack(...)` | Tire track impression |
| `void ApplyCrater(x, z, radius, depth)` | Explosion crater |
| `void ApplyFootprint(x, z, sizeX, sizeZ, depth)` | Character footprint |
| `bool Undo()` / `void Reset()` | History management |

---

### 2.5 Gaseous Simulation (`WulfNet::GaseousSystem`)

**Header:** `WulfNet/Physics/Gaseous/GaseousSystem.h`

Eulerian smoke/fire simulator with semi-Lagrangian advection, Jacobi pressure solve, buoyancy, vorticity confinement, and combustion.

| Method | Description |
|--------|-------------|
| `bool Initialize(const GaseousSystemConfig&)` | Create grid |
| `void Step(float dt)` | Full simulation step |
| `uint32_t AddEmitter(const GasEmitter&)` | Point/Sphere/Box emitters |
| `uint32_t AddObstacle(const GasObstacle&)` | Solid boundaries |
| `float SampleDensity(wx, wy, wz)` | Trilinear density query |
| `float SampleTemperature(wx, wy, wz)` | Trilinear temperature query |
| `void SampleVelocity(wx, wy, wz, &vx, &vy, &vz)` | Velocity field query |
| `void SetDensity(i, j, k, value)` | Direct cell write |

---

### 2.6 Destruction System (`WulfNet::DestructionSystem`)

**Header:** `WulfNet/Physics/Destruction/DestructionSystem.h`

Voronoi pre-fracture with impulse/stress threshold evaluation.

| Method | Description |
|--------|-------------|
| `bool Initialize(const DestructionConfig&)` | Setup system |
| `uint32_t AddDestructible(bodyId, threshold, cellCount)` | Register breakable body |
| `bool EvaluateImpact(handle, x, y, z, impulse)` | Check fracture threshold |
| `uint32_t Fracture(handle, x, y, z)` | Execute fracture → fragments |
| `void Step(float dt, JPH::PhysicsSystem*)` | Update fragments |
| `void SetFractureCallback(FractureCallback)` | Event hook |
| `static FracturePattern GenerateBoxPattern(...)` | Pre-compute Voronoi cells |
| `static FracturePattern GenerateSpherePattern(...)` | Sphere Voronoi pattern |

---

## 3. GPU Compute Systems

### 3.1 VulkanContext (`WulfNet::VulkanContext`)

**Header:** `WulfNet/Compute/Vulkan/VulkanContext.h`

Headless Vulkan 1.3 compute context with validation layers and device selection.

| Method | Description |
|--------|-------------|
| `bool Initialize(const VulkanConfig&)` | Create instance + device |
| `void Shutdown()` | Destroy Vulkan resources |
| `VkDevice GetDevice()` | Raw Vulkan device |
| `VkQueue GetComputeQueue()` | Compute queue handle |
| `uint32_t GetComputeQueueFamily()` | Queue family index |

### 3.2 ComputeBuffer<T> (`WulfNet::ComputeBuffer<T>`)

**Header:** `WulfNet/Compute/Memory/ComputeBuffer.h`

Templated GPU buffer with staging, upload, download, map/unmap, and resize.

| Method | Description |
|--------|-------------|
| `bool Initialize(VulkanContext&, size_t count, Usage)` | Allocate |
| `void Upload(const T* data, size_t count)` | CPU → GPU |
| `void Download(T* data, size_t count)` | GPU → CPU |
| `T* Map()` / `void Unmap()` | Mapped access |
| `void Resize(size_t newCount)` | Grow/shrink |

### 3.3 ComputePipeline (`WulfNet::ComputePipeline`)

**Header:** `WulfNet/Compute/Shaders/ComputePipeline.h`

SPIR-V shader loading, descriptor sets, push constants, specialization constants.

### 3.4 ParallelReduction (`WulfNet::ParallelReduction`)

**Header:** `WulfNet/Compute/Reduction/ParallelReduction.h`

GPU min/max/sum, bounding box, and centroid computation.

---

## 4. Procedural Systems

### 4.1 IFS (Iterated Function Systems)

**Headers:** `WulfNet/Procedural/IFS/AffineTransform.h`, `TransformPresets.h`, `TransformBlender.h`, `IFSSystem.h`

GPU-accelerated fractal generation via the chaos game method.

| Class | Purpose |
|-------|---------|
| `AffineTransform` | GPU-compatible 3×3+translation matrix builders |
| `TransformPresets` | Named fractal presets (Sierpinski, Fern, Dragon, etc.) |
| `TransformBlender` | Smooth interpolation between preset sets |
| `IFSSystem` | Full pipeline: chaos game → voxelize → LOD → render |

---

## 5. Rendering Systems

### 5.1 Software Rasterizer

**Header:** `WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h`

Self-contained math types, camera, lights, materials for CPU rendering.

**Key types:** `SoftVec3`, `SoftVec4`, `SoftMat4`, `SoftColorRGBA8`, `SoftCamera`, `SoftMesh`, `SoftTexture`, `SoftTransform`, `SoftDirectionalLight`, `SoftPointLight`, `SoftMaterial`

### 5.2 GBuffer (`WulfNet::GBuffer`)

**Header:** `WulfNet/Rendering/SoftwareRasterizer/GBuffer.h`

| Method | Description |
|--------|-------------|
| `bool Initialize(width, height)` | Allocate buffers |
| `void Clear(skyTop, skyBottom)` | SIMD sky gradient clear |
| `void SetColor/Normal/Depth(x, y, value)` | Write pixel |
| `bool DepthTest(x, y, depth)` | Z-test and update |

### 5.3 SoftwareRasterizer (`WulfNet::SoftwareRasterizer`)

**Header:** `WulfNet/Rendering/SoftwareRasterizer/SoftwareRasterizer.h`

Scanline rasterizer with backface culling, perspective-correct interpolation, multi-threaded.

| Method | Description |
|--------|-------------|
| `bool Initialize(const SoftRasterizerConfig&)` | Setup |
| `void RenderMesh(const SoftMesh&, const SoftTransform&, const SoftCamera&, GBuffer&)` | Rasterize mesh |

### 5.4 DeferredShading (`WulfNet::DeferredShading`)

**Header:** `WulfNet/Rendering/SoftwareRasterizer/DeferredShading.h`

Screen-space lighting: directional + point lights, hemisphere ambient, distance fog, Fresnel.

### 5.5 OcclusionCuller (`WulfNet::OcclusionCuller`)

**Header:** `WulfNet/Rendering/SoftwareRasterizer/OcclusionCuller.h`

Low-resolution CPU occlusion culling with AABB visibility testing.

### 5.6 ShadowSystem (`WulfNet::ShadowSystem`)

**Header:** `WulfNet/Rendering/SoftwareRasterizer/ShadowMap.h`

| Class | Description |
|-------|-------------|
| `ShadowCascade` | Single cascade depth buffer for CSM |
| `PointLightShadow` | 6-face cube shadow map |
| `ShadowSystem` | Unified cascade + point-light shadow manager |

| Method | Description |
|--------|-------------|
| `bool Initialize(const ShadowSystemConfig&)` | Allocate cascades |
| `void ComputeCascadeSplits(const SoftCamera&)` | Log/linear splits |
| `void RenderDirectionalShadows(...)` | Rasterize into cascades |
| `float SampleDirectionalShadow(worldPos)` | PCF shadow query |
| `float SamplePointLightShadow(lightIdx, worldPos)` | Cube shadow query |

### 5.7 GlobalIllumination (`WulfNet::GlobalIllumination`)

**Header:** `WulfNet/Rendering/SoftwareRasterizer/GlobalIllumination.h`

| Feature | Description |
|---------|-------------|
| SSAO | Fibonacci hemisphere, per-pixel hash, configurable |
| Indirect Bounce | One-bounce diffuse from GBuffer |
| Light Probes | L1 spherical harmonics, quadratic falloff |

| Method | Description |
|--------|-------------|
| `bool Initialize(width, height, config)` | Setup |
| `void Compute(const GBuffer&, const SoftCamera&)` | Full GI pass |
| `float SampleAO(x, y)` | Per-pixel occlusion |
| `SoftVec3 EvaluateProbes(worldPos, normal)` | Probe irradiance |

### 5.8 VolumetricRenderer (`WulfNet::VolumetricRenderer`)

**Header:** `WulfNet/Rendering/SoftwareRasterizer/VolumetricRenderer.h`

Ray-marching through AABB volumes with Beer-Lambert absorption and Henyey-Greenstein scattering.

| Method | Description |
|--------|-------------|
| `bool Initialize(width, height, config)` | Setup |
| `void AddVolume(const VolumeSampler&)` | Register volume |
| `void Render(GBuffer&, const SoftCamera&)` | Full screen ray march |
| `VolumetricSample MarchRay(origin, dir, maxDist, sampler)` | Single ray |
| `static float PhaseHG(cosTheta, g)` | Phase function |
| `SoftVec3 EvaluateEmission(temperature)` | Blackbody-like ramp |

### 5.9 RenderPipeline (`WulfNet::RenderPipeline`)

**Header:** `WulfNet/Rendering/SoftwareRasterizer/RenderPipeline.h`

Unified orchestrator: Shadow → GBuffer → GI → Lighting → Volumetric.

| Method | Description |
|--------|-------------|
| `bool Initialize(const RenderPipelineConfig&)` | Wire all stages |
| `void Shutdown()` | Release resources |
| `void RenderFrame(transforms, count, camera)` | Full frame |
| `const uint32_t* GetColorBuffer()` | Final output |
| `const RenderStats& GetStats()` | Per-pass timing |

---

## 6. Audio Systems

### 6.1 AudioBuffer (`WulfNet::AudioBuffer`)

**Header:** `WulfNet/Audio/Core/AudioTypes.h`

Interleaved float audio buffer with generation, mixing, and analysis.

| Method | Description |
|--------|-------------|
| `bool Initialize(const AudioFormat&, frameCount)` | Allocate |
| `static GenerateSine(freq, dur, sampleRate, amp)` | Sine wave |
| `static GenerateNoise(dur, sampleRate, amp)` | White noise |
| `static GenerateSilence(dur, sampleRate)` | Zero buffer |
| `bool LoadFromFloat(data, frameCount, fmt)` | Import raw data |
| `float ComputeRMS()` / `ComputePeak()` | Analysis |
| `void MixIn(other, gain, offset)` | Additive mix |
| `void ApplyGain(gain)` | Scale all samples |
| `void Normalize(targetPeak)` | Peak normalize |
| `void Clear()` / `Resize(frames)` | Buffer management |

### 6.2 AudioSource (`WulfNet::AudioSource`)

**Header:** `WulfNet/Audio/Core/AudioTypes.h`

Playback controller with gain, pan, loop, and fade.

| Method | Description |
|--------|-------------|
| `void SetBuffer(const AudioBuffer*)` | Attach audio data |
| `void Play()` / `Stop()` / `Pause()` / `Resume()` | Transport |
| `int ReadFrames(float* outStereo, frameCount)` | Render to stereo |
| `void SetGain(float)` / `SetPan(float)` | Per-source control |
| `void SetPosition(x, y, z)` | 3D position |
| `void SetVelocity(x, y, z)` | For Doppler |

### 6.3 AudioMixer (`WulfNet::AudioMixer`)

**Header:** `WulfNet/Audio/Core/AudioMixer.h`

Multi-source mixer with tanh soft-clip limiter and statistics.

| Method | Description |
|--------|-------------|
| `bool Initialize(const AudioMixerConfig&)` | Setup |
| `int AddSource(AudioSource*)` | Register source |
| `int MixFrame(float* stereo, frameCount)` | Mix all active sources |
| `int MixFrame(AudioBuffer&, frameCount)` | Mix into AudioBuffer |
| `void SetMasterGain(float)` | Master volume |
| `const AudioMixerStats& GetStats()` | Peak, RMS, clip count |

### 6.4 AcousticSystem (`WulfNet::AcousticSystem`)

**Header:** `WulfNet/Audio/Acoustics/AcousticSystem.h`

Ray-traced acoustic simulation with pluggable physics backend.

**Materials:** 6-band absorption (125Hz–4kHz), scattering, transmission.
**Presets:** `AcousticMaterial::Concrete()`, `Wood()`, `Glass()`, `Carpet()`, `Curtain()`, `Metal()`

| Method | Description |
|--------|-------------|
| `bool Initialize(const AcousticConfig&)` | Setup |
| `void SetRayCastFunction(AcousticRayCastFn)` | Plugin physics ray-cast |
| `int AddMaterial(const AcousticMaterial&)` | Register material |
| `float ComputeOcclusion(src, lst)` | 0.0 (clear) to 1.0 (blocked) |
| `float ComputeObstruction(src, lst, rays)` | Multi-ray partial block |
| `ImpulseResponse TraceImpulseResponse(src, lst)` | Full ray-traced reverb |
| `RoomEstimate EstimateRoom(pos)` | Volume, SA, RT60 estimation |
| `static float ComputeRT60_Sabine(V, S, α)` | Sabine formula |
| `static float ComputeRT60_Eyring(V, S, α)` | Eyring formula |
| `static float DistanceAttenuation(d, ref, max, rolloff)` | Inverse power law |

### 6.5 SpatialAudio (`WulfNet::SpatialAudio`)

**Header:** `WulfNet/Audio/Spatial/SpatialAudio.h`

Binaural HRTF, first-order Ambisonics, and Doppler effect processing.

| Method | Description |
|--------|-------------|
| `bool Initialize(sampleRate)` | Setup |
| `HRTFResult ComputeHRTF(srcX, srcY, srcZ)` | ITD + ILD computation |
| `AudioBuffer ApplyHRTF(monoInput, hrtfResult)` | Mono → stereo binaural |
| `static EncodeAmbisonics(az, el, gain)` | FOA B-format encode |
| `static DecodeAmbisonics(bformat, speakers)` | Decode to speaker gains |
| `static CreateStereoLayout()` / `CreateQuadLayout()` | Speaker presets |
| `float ComputeDopplerShift(src_pos, src_vel, lst_pos, lst_vel)` | Pitch shift factor |
| `float ComputeDistanceGain(distance)` | Attenuation curve eval |
| `static WorldToListenerLocal(src, lst_pos, lst_fwd, lst_up, &local)` | Transform |
| `static CartesianToSpherical(x, y, z, &az, &el, &dist)` | Coordinate conversion |

**Attenuation models:** `None`, `Linear`, `Inverse`, `Exponential`

---

## Coordinate Conventions

| Convention | Value |
|------------|-------|
| Up axis | +Y |
| Forward | -Z (listener local space) |
| Right | +X (cross product of forward × up) |
| Azimuth | 0 = front, positive = right |
| Elevation | 0 = horizontal, positive = up |
| Audio format | Interleaved float, stereo (L, R, L, R, ...) |

---

## Build & Test

```bash
# Configure (VS2022)
cd build && cmake_vs2022_cl.bat

# Build library
cmake --build build/VS2022_CL --target WulfNet --config Release

# Build & run tests
cmake --build build/VS2022_CL --target WulfNetExtendedTests --config Release
./build/VS2022_CL/WulfNetTests/Release/WulfNetExtendedTests.exe

# Run specific suite
WulfNetExtendedTests --suite=audio
WulfNetExtendedTests --suite=benchmark

# Generate Doxygen docs
doxygen Doxyfile
```

---

*Document generated: February 2026*
*WulfNet Engine v0.8.0 — 535+ tests, 100% pass rate*
