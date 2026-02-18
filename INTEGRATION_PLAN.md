# Integration Plan: Software Rasterizer + IFS Fractal System into WulfNet Engine

## Context

WulfNet Engine needs two new subsystems adapted from open-source reference projects:

1. **BG-C-Software-Renderer** -- A CPU software rasterizer with SIMD (SSE/AVX2), multi-threaded scanline rendering, deferred shading, and PBR-lite lighting. Useful for CPU occlusion culling, headless/fallback rendering, and as PBR reference math for the future GPU renderer (Phase 7).

2. **Iterated Function Systems (IFS)** -- A GPU-accelerated fractal particle system using the chaos game algorithm and affine transformations. Provides procedural geometry generation, voxelization with ambient occlusion, and GPU parallel reduction utilities.

Both Unity-based projects will be ported to native C++ using WulfNet's existing Vulkan compute infrastructure (`ComputeBuffer<T>`, `ComputePipeline`, `VulkanContext`) and rendered through Jolt's TestFramework renderer.

Reference source is cloned at `_ref/BG-C-Software-Renderer/` and `_ref/Iterated-Function-Systems/`.

---

## New File Structure

```
WulfNet/
  Rendering/SoftwareRasterizer/
    SoftRasterTypes.h          # SoftVertex, SoftMesh, SoftTransform, SoftCamera, lights
    GBuffer.h / .cpp           # Framebuffer management + SIMD clear + sky gradient
    SoftwareRasterizer.h / .cpp  # Core scanline rasterizer (from MainEngine.cpp)
    DeferredShading.h / .cpp   # Deferred lighting pass (sun, point lights, fog, reflections)
    OcclusionCuller.h / .cpp   # Low-res CPU occlusion culling utility

  Procedural/IFS/
    AffineTransform.h / .cpp   # Affine matrix builder (scale, shear, rotate, translate)
    TransformPresets.h / .cpp  # Fractal presets (Sierpinski, Vicsek, procedural)
    TransformBlender.h / .cpp  # Interpolation between preset sets
    IFSSystem.h / .cpp         # Main GPU IFS system (chaos game + iterated expansion)
    IFSVoxelizer.h / .cpp      # Voxelization + 3x3x3 neighbor ambient occlusion

  Compute/Reduction/
    ParallelReduction.h / .cpp # Reusable GPU parallel min/max/sum reduction

Assets/Shaders/Compute/
    reduce_global.comp         # Multi-pass parallel reduction (specialization constants)
    reduce_final.comp          # Final reduction pass
    reduce_to_transform.comp   # Bounds -> scale+translate framing transform
    ifs_init_particles.comp    # Initialize particle grid
    ifs_chaos_game.comp        # Chaos game random transform iteration
    ifs_iterated_expand.comp   # Deterministic tree expansion
    ifs_clear_voxels.comp      # Clear voxel grid
    ifs_voxelize.comp          # Voxelize particle positions into 3D grid
    ifs_clear_occlusion.comp   # Clear AO grid
    ifs_calc_occlusion.comp    # 3x3x3 neighbor AO computation
    ifs_lod_first.comp         # First LOD iteration for bounds prediction
    ifs_lod_iterate.comp       # Subsequent LOD expansion iterations

WulfNetExamples/
    IFSExample.cpp             # Standalone IFS demo
    SoftRasterExample.cpp      # Standalone software rasterizer demo
```

### Files to Modify

| File | Change |
|------|--------|
| `WulfNet/Compute/Shaders/ComputePipeline.h/.cpp` | Add specialization constant support (`VkSpecializationInfo`) |
| `WulfNet/CMakeLists.txt` | Add all new source files |
| `WulfNet/WulfNet.h` | Add new module includes |
| `WulfNet/Compute/Compute.h` | Add ParallelReduction include |
| `WulfNetExamples/CMakeLists.txt` | Add example executables |

---

## Implementation Steps

### Step 1: Extend ComputePipeline for Specialization Constants

**Why first**: The IFS parallel reduction shaders use specialization constants to switch between min/max/sum operations without recompiling shaders. This is needed before any IFS compute work.

**What to do**:
- Add `std::vector<std::pair<uint32_t, uint32_t>> specializationConstants` to pipeline creation
- Wire through `VkSpecializationInfo` in `VkComputePipelineCreateInfo`
- Small change (~50 lines) to existing [ComputePipeline.h](WulfNet/Compute/Shaders/ComputePipeline.h)

---

### Step 2: GPU Parallel Reduction (reusable utility)

**Files**: `WulfNet/Compute/Reduction/ParallelReduction.h/.cpp` + 3 shaders

**Port from**: `_ref/Iterated-Function-Systems/Assets/Compute Shaders/ParallelReduce.compute`

**Key classes**:
```cpp
class ParallelReduction {
    void Reduce(input, output, count, ReductionOp::Min|Max|Sum);
    void ComputeBounds(positions, boundsOutput, count);         // min + max
    void ComputeBoundsAndCentroid(positions, output, count);    // min + max + sum
    void BoundsToTransform(boundsData, transformOutput, ...);   // auto-framing
};
```

**HLSL -> GLSL porting rules**:
- `groupshared float3` -> `shared vec3`
- `GroupMemoryBarrierWithGroupSync()` -> `barrier(); memoryBarrierShared()`
- `#pragma multi_compile` -> `layout(constant_id = N)` specialization constants
- `[numthreads(128,1,1)]` -> `layout(local_size_x = 128) in;`
- `SV_DispatchThreadID` -> `gl_GlobalInvocationID`
- Sequential addressing (Reduction #3) as default -- best general-purpose strategy

**Testing**: Create random vec3 buffer, reduce to min/max, verify against CPU reference.

---

### Step 3: Affine Transforms + Fractal Presets (CPU-only)

**Files**: `AffineTransform.h/.cpp`, `TransformPresets.h/.cpp`, `TransformBlender.h/.cpp`

**Port from**: `AffineTransformations.cs`, `AttractorPresets.cs`, `TransformSet.cs`, `SetBlender.cs`, `ProceduralWizard.cs`

**Key design decisions**:
- Use `JPH::Mat44` for CPU-side matrix math (SIMD-optimized)
- Define `GPUMat4x4 { float m[16]; }` for GPU upload (row-major, matching original HLSL layout)
- In GLSL shaders, do explicit `transformPoint()` instead of using `mat4` to avoid row/column-major confusion
- Port all 7 presets: Sierpinski Triangle 2D/3D, Vicsek 2D/3D, Sierpinski Carpet 2D/3D, Procedural
- `TransformBlender`: linear interpolation with lerp/slerp + exponential decay smoothing

---

### Step 4: IFS Compute Shaders (HLSL -> GLSL port)

**Files**: 9 new `.comp` shaders in `Assets/Shaders/Compute/`

**Port from**: `_ref/Iterated-Function-Systems/Assets/Compute Shaders/UpdateParticles.compute`

**Kernels to port**:
1. `ifs_init_particles.comp` -- Initialize particles in cube grid (`local_size_x = 64`)
2. `ifs_chaos_game.comp` -- Random transform via Hugo Elias hash + affine matrix multiply
3. `ifs_iterated_expand.comp` -- Deterministic tree expansion with generation offsets
4. `ifs_clear_voxels.comp` -- Zero out voxel grid buffer
5. `ifs_voxelize.comp` -- Map particle positions into 3D voxel grid + apply final transform
6. `ifs_clear_occlusion.comp` -- Zero out AO grid
7. `ifs_calc_occlusion.comp` -- 3x3x3 neighbor counting for ambient occlusion
8. `ifs_lod_first.comp` -- Seed LOD by applying each transform to origin
9. `ifs_lod_iterate.comp` -- Expand LOD by applying all transforms to each input point

**Matrix convention**: Store as flat `float[16]` row-major in SSBO, multiply explicitly in shader to avoid mat4 column-major confusion.

---

### Step 5: IFSSystem Core Implementation

**Files**: `IFSSystem.h/.cpp`, `IFSVoxelizer.h/.cpp`

**Pattern to follow**: [VulkanFluidCompute.h](WulfNet/Compute/Fluids/VulkanFluidCompute.h) -- multi-pipeline GPU system with push constants, buffer layout, batched command buffers.

**Core IFSSystem class**:
```cpp
class IFSSystem {
    bool Initialize(const IFSConfig& config, VulkanContext* vulkan);
    void Update(float dt);           // iterate + voxelize + predict
    void SetPreset(IFSPreset preset);
    void Render(Renderer* renderer); // draw via Jolt TestFramework
    VkBuffer GetParticleBuffer();    // direct GPU buffer access
};
```

**GPU buffer layout**:
- `ComputeBuffer<float>` for particle positions (3 floats per particle, packed)
- `ComputeBuffer<GPUMat4x4>` for affine transforms (up to 32)
- `ComputeBuffer<int32_t>` for voxel grid (dimension^3)
- `ComputeBuffer<float>` for occlusion grid (dimension^3)
- LOD iteration buffers (geometric growth: transformCount^1 through transformCount^N)

**Initial rendering**: CPU readback of particle positions -> upload to Jolt `RenderPrimitive` as point cloud. This works across all backends (VK/DX12/MTL). Direct GPU buffer sharing can be optimized later.

---

### Step 6: Software Rasterizer Types + GBuffer

**Files**: `SoftRasterTypes.h`, `GBuffer.h/.cpp`

**Port from**: Struct definitions and `Clear()` function in `MainEngine.cpp` (lines 26-741)

**Key types**: `SoftVertex` (pos+normal+uv), `SoftMesh` (verts, indices, face normals, material), `SoftTransform` (JPH::Vec3 pos, JPH::Quat rot, scale, mesh index, tint), `SoftCamera`, `SoftPointLight`

**GBuffer**: 3 buffers (color RGBA8, normal RGBA8, depth float32). SIMD-accelerated clear with sky gradient (AVX2 with SSE fallback). Runtime CPU feature detection for AVX2 via `__cpuidex`.

**Design**: All original global state (`Width`, `Height`, `cameraForward`, `RenderTexture`, `Depth`, etc.) becomes class members. No global state. Enables multiple rasterizer instances.

---

### Step 7: Software Rasterizer Core

**Files**: `SoftwareRasterizer.h/.cpp`

**Port from**: `RenderObjectsPooled()` in `MainEngine.cpp` (lines 1174-end)

**Core rasterization pipeline** (per triangle):
1. Backface culling (face normal vs view direction)
2. World-to-screen projection (perspective divide)
3. Near-plane clipping (push vertices to near plane)
4. Screen-space bounding box + clip to viewport
5. Scanline rasterization with edge functions
6. Perspective-correct barycentric interpolation (UV, normals, depth)
7. Per-pixel depth test and write
8. Texture sampling + vertex color tinting
9. Normal buffer write (for deferred pass)

**Threading**: `std::thread` pool with atomic counter work-stealing (same pattern as original). Object-level parallelism (each thread processes a chunk of objects). Optionally use OpenMP if available.

**SIMD**: Keep SSE/AVX2 intrinsics for inner pixel loops (edge evaluation, normal interpolation, UV computation, depth comparison). Guard AVX2 with runtime detection.

---

### Step 8: Deferred Shading + Occlusion Culler

**Files**: `DeferredShading.h/.cpp`, `OcclusionCuller.h/.cpp`

**Port from**: `Differed()` function in `MainEngine.cpp` (lines 912-1153)

**Deferred pass features**:
- Directional light (Lambert diffuse)
- Hemisphere ambient (sky/ground blend by normal.y)
- Point lights with distance attenuation and early-out checks
- Distance fog
- Fresnel reflections + specular highlights + metalness

**OcclusionCuller**: Wraps a low-resolution SoftwareRasterizer (e.g. 256x144):
```cpp
class OcclusionCuller {
    void RenderOccluders(const SoftTransform* occluders, int count, const SoftCamera& camera);
    bool IsVisible(const JPH::AABox& worldBounds) const;  // test against z-buffer
    void TestVisibility(const JPH::AABox* bounds, bool* results, int count) const;
};
```

---

### Step 9: CMake + Build Integration

- Add all new `.cpp`/`.h` to `WulfNet/CMakeLists.txt`
- Add `glslc` compilation rules for all new `.comp` shaders
- Create `IFSExample` and `SoftRasterExample` executables in `WulfNetExamples/`
- Ensure SPIR-V outputs go to `Assets/Shaders/Compute/`

---

### Step 10: Example Applications + Verification

**IFSExample**: Standalone app that:
1. Initializes VulkanContext
2. Creates IFSSystem with Sierpinski Triangle 3D preset
3. Runs iteration loop, downloads particles, renders via Jolt TestFramework
4. Demonstrates preset switching and blending between fractals

**SoftRasterExample**: Standalone app that:
1. Creates SoftwareRasterizer at 1280x720
2. Adds test meshes (cube, sphere)
3. Renders + deferred shading pass
4. Displays result via Jolt TestFramework as a textured quad

---

## Verification Plan

1. **ParallelReduction**: Create 100K random vec3 buffer, GPU reduce to min/max/sum, compare against CPU reference -- must match within float epsilon
2. **AffineTransform**: Generate matrices for each preset, multiply test point through all transforms, verify convergence to known attractor shape
3. **IFS Compute**: Run chaos game for 1000 iterations on Sierpinski Triangle, download positions, verify all points lie within expected bounds
4. **Software Rasterizer**: Render a single textured triangle at known screen coordinates, compare pixel output against hand-calculated expected values. Test depth buffer correctness with overlapping triangles
5. **OcclusionCuller**: Render a wall as occluder, test that objects behind wall return `IsVisible() == false`, objects in front return `true`
6. **Build**: Full CMake regeneration + build with VS2022, zero warnings at `/W4`
