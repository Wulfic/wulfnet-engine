# =============================================================================
# WulfNet Engine Extensions - CMake Configuration
# =============================================================================
# This CMakeLists.txt builds the WulfNet Engine library and links with Jolt.
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# =============================================================================
# WulfNet Library
# =============================================================================

set(WULFNET_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Collect source files
set(WULFNET_SOURCES
    # Engine Core
    Engine.cpp

    # Core
    Core/Logging/Logger.cpp
    Core/Profiling/Profiler.cpp
    Core/System/SystemMonitor.cpp
    Core/System/PerfLogger.cpp
    Core/Threading/ThreadPool.cpp

    # Physics Integration
    Physics/Integration/PhysicsWorld.cpp

    # Physics - Fluids (SWE 2.5D) [renamed from WaterSystemV3]
    Physics/Fluids/FluidSystem.cpp

    # Physics - MPM Constitutive Models
    Physics/MPM/ConstitutiveModel.cpp
    Physics/MPM/MPMRigidCoupling.cpp

    # Physics - Terrain Deformation
    Physics/Terrain/TerrainDeformation.cpp

    # Physics - Gaseous Simulation
    Physics/Gaseous/GaseousSystem.cpp
    Physics/Gaseous/GaseousSystemSolve.cpp

    # Physics - Destruction System
    Physics/Destruction/DestructionSystem.cpp

    # GPU Compute
    Compute/Vulkan/VulkanContext.cpp
    Compute/Vulkan/VulkanLoader.cpp
    Compute/Vulkan/VulkanFramePipeline.cpp
    Compute/Memory/ComputeBuffer.cpp
    Compute/Pipelines/ComputePipeline.cpp
    Compute/Fluids/SWEComputeGPU.cpp
    Compute/Reduction/ParallelReduction.cpp

    # Procedural - IFS
    Procedural/IFS/AffineTransform.cpp
    Procedural/IFS/TransformPresets.cpp
    Procedural/IFS/TransformBlender.cpp
    Procedural/IFS/IFSSystem.cpp

    # Rendering - Software Rasterizer
    Rendering/SoftwareRasterizer/GBuffer.cpp
    Rendering/SoftwareRasterizer/SoftwareRasterizer.cpp
    Rendering/SoftwareRasterizer/DeferredShading.cpp
    Rendering/SoftwareRasterizer/OcclusionCuller.cpp

    # Rendering - Lighting
    Rendering/Lighting/ShadowMap.cpp
    Rendering/Lighting/GlobalIllumination.cpp

    # Rendering - Effects
    Rendering/Effects/VolumetricRenderer.cpp

    # Rendering - Pipeline
    Rendering/RenderPipeline.cpp

    # Audio - Core
    Audio/Core/AudioTypes.cpp
    Audio/Core/AudioMixer.cpp

    # Audio - Acoustics
    Audio/Acoustics/AcousticSystem.cpp

    # Audio - Spatial
    Audio/Spatial/SpatialAudio.cpp
)

set(WULFNET_HEADERS
    # Main header
    WulfNet.h

    # Engine Core & Public API
    API.h
    Version.h
    ForwardDecl.h
    EngineConfig.h
    Engine.h

    # Core
    Core/Logging/Logger.h
    Core/Profiling/Profiler.h
    Core/System/SystemMonitor.h
    Core/System/PerfLogger.h
    Core/Threading/ThreadPool.h
    Core/Memory/FrameAllocator.h
    Core/Math/MathTypes.h
    Core/Math/MathUtils.h
    Core/Math/PerlinNoise.h

    # Physics Integration
    Physics/Integration/PhysicsWorld.h

    # Physics - Fluids (SWE 2.5D) [renamed from WaterSystemV3]
    Physics/Fluids/FluidSystem.h
    Physics/WaterSystemV3.h       # Backward-compat redirect

    # Physics - MPM Constitutive Models
    Physics/MPM/ConstitutiveModel.h
    Physics/MPM/MPMRigidCoupling.h

    # Physics - Terrain Deformation
    Physics/Terrain/TerrainDeformation.h

    # Physics - Gaseous Simulation
    Physics/Gaseous/GaseousSystem.h

    # Physics - Destruction System
    Physics/Destruction/DestructionSystem.h

    # GPU Compute
    Compute/Compute.h
    Compute/Vulkan/VulkanContext.h
    Compute/Memory/ComputeBuffer.h
    Compute/Pipelines/ComputePipeline.h
    Compute/Shaders/ComputePipeline.h    # Backward-compat redirect
    Compute/Fluids/SWEComputeGPU.h
    Compute/Reduction/ParallelReduction.h

    # Procedural - IFS
    Procedural/IFS/AffineTransform.h
    Procedural/IFS/TransformPresets.h
    Procedural/IFS/TransformBlender.h
    Procedural/IFS/IFSSystem.h

    # Rendering - Types
    Rendering/Types/RenderTypes.h

    # Rendering - Software Rasterizer (core)
    Rendering/SoftwareRasterizer/SoftRasterTypes.h   # Backward-compat redirect
    Rendering/SoftwareRasterizer/GBuffer.h
    Rendering/SoftwareRasterizer/SoftwareRasterizer.h
    Rendering/SoftwareRasterizer/DeferredShading.h
    Rendering/SoftwareRasterizer/OcclusionCuller.h
    Rendering/SoftwareRasterizer/ShadowMap.h          # Backward-compat redirect
    Rendering/SoftwareRasterizer/GlobalIllumination.h # Backward-compat redirect
    Rendering/SoftwareRasterizer/VolumetricRenderer.h # Backward-compat redirect
    Rendering/SoftwareRasterizer/RenderPipeline.h     # Backward-compat redirect

    # Rendering - Lighting (new location)
    Rendering/Lighting/ShadowMap.h
    Rendering/Lighting/GlobalIllumination.h

    # Rendering - Effects (new location)
    Rendering/Effects/VolumetricRenderer.h

    # Rendering - Pipeline & Abstraction
    Rendering/IRenderer.h
    Rendering/RenderCommand.h
    Rendering/RenderPipeline.h

    # Audio - Core
    Audio/Core/AudioTypes.h
    Audio/Core/AudioMixer.h

    # Audio - Acoustics
    Audio/Acoustics/AcousticSystem.h

    # Audio - Spatial
    Audio/Spatial/SpatialAudio.h
)

# Create the WulfNet library
# Disable INTERPROCEDURAL_OPTIMIZATION before creating the target to avoid /GL being added
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
add_library(WulfNet STATIC ${WULFNET_SOURCES} ${WULFNET_HEADERS})

# Explicitly disable IPO for this target
set_target_properties(WulfNet PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

# Set C++ standard
target_compile_features(WulfNet PUBLIC cxx_std_17)

# Include directories
target_include_directories(WulfNet
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link with Jolt
target_link_libraries(WulfNet PUBLIC Jolt)

# Find and link OpenMP for CPU parallelization
# MSVC: OpenMP support requires /openmp flag. Using /openmp:llvm for full
# OpenMP 3.1+ support (min/max reductions, etc). Without this, find_package
# silently fails, making ALL #pragma omp directives no-ops (single-threaded).
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")
endif()
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(WulfNet PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(WulfNet PRIVATE WULFNET_HAS_OPENMP)
    message(STATUS "  OpenMP found: CPU parallelization enabled")
else()
    # Manual fallback for MSVC — /openmp is already set above, just define the macro
    if(MSVC)
        target_compile_definitions(WulfNet PRIVATE WULFNET_HAS_OPENMP)
        message(STATUS "  OpenMP: MSVC /openmp flag set manually — CPU parallelization enabled")
    else()
        message(STATUS "  OpenMP not found: CPU parallelization will be disabled")
    endif()
endif()

# Find and link Vulkan for GPU compute
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    target_include_directories(WulfNet PRIVATE ${Vulkan_INCLUDE_DIRS})
    # Don't link Vulkan library - we use dynamic loading
    target_compile_definitions(WulfNet PRIVATE WULFNET_HAS_VULKAN)
    message(STATUS "  Vulkan SDK found: GPU compute enabled")
else()
    message(STATUS "  Vulkan SDK not found: GPU compute will be disabled at runtime")
endif()

# MSVC-specific settings - disable some warnings that are too strict for our code
if(MSVC)
    target_compile_options(WulfNet PRIVATE
        # Release-specific optimizations
        $<$<CONFIG:Release>:/Oi>   # Enable intrinsic functions
        $<$<CONFIG:Release>:/Ot>   # Favor fast code
        $<$<CONFIG:Release>:/GS->  # Disable security checks in hot paths
        # Warning suppressions
        /wd4820  # padding added
        /wd4514  # unreferenced inline function removed
        /wd4625  # copy constructor was implicitly deleted
        /wd4626  # assignment operator was implicitly deleted
        /wd5026  # move constructor was implicitly deleted
        /wd5027  # move assignment operator was implicitly deleted
        /wd4464  # relative include path contains '..'
        /wd4100  # unreferenced parameter
        /wd4365  # signed/unsigned mismatch
        /wd4061  # enumerator not explicitly handled
        /wd5219  # implicit conversion
        /wd4668  # macro not defined
        /wd4530  # C++ exception handler used
        /wd5039  # pointer to throwing function passed to extern C
        /wd4710  # function not inlined
        /wd4711  # function selected for automatic inline expansion
        /wd4743  # vtable size different
        /EHsc    # Enable C++ exceptions
        /GL-     # Disable whole program optimization (LTCG conflicts with Jolt)
        /WX-     # Don't treat warnings as errors
    )
else()
    # GCC/Clang Release optimizations
    target_compile_options(WulfNet PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-fno-math-errno>  # Safe FP optimization (no errno on math funcs)
    )
endif()

# SIMD architecture flags (opt-in to avoid breaking builds on older hardware)
option(WULFNET_ENABLE_AVX2 "Enable AVX2 instruction set for SIMD optimizations" OFF)
if(WULFNET_ENABLE_AVX2)
    if(MSVC)
        target_compile_options(WulfNet PRIVATE /arch:AVX2)
    else()
        target_compile_options(WulfNet PRIVATE -mavx2 -mfma)
    endif()
    target_compile_definitions(WulfNet PUBLIC WULFNET_HAS_AVX2)
    message(STATUS "  AVX2 SIMD enabled")
elseif(MSVC)
    # MSVC defaults to SSE2 on x64 — no flag needed
    target_compile_definitions(WulfNet PUBLIC WULFNET_HAS_SSE2)
else()
    # Check for SSE2 on GCC/Clang (enabled by default on x86_64)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-msse2" HAS_SSE2)
    if(HAS_SSE2)
        target_compile_definitions(WulfNet PUBLIC WULFNET_HAS_SSE2)
    endif()
endif()

# Compiler definitions
target_compile_definitions(WulfNet
    PUBLIC
        # Enable Tracy profiler (can be controlled via option)
        $<$<BOOL:${WULFNET_ENABLE_TRACY}>:WULFNET_ENABLE_TRACY>
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(WulfNet PUBLIC WULFNET_PLATFORM_WINDOWS)
elseif(APPLE)
    target_compile_definitions(WulfNet PUBLIC WULFNET_PLATFORM_MACOS)
elseif(UNIX)
    target_compile_definitions(WulfNet PUBLIC WULFNET_PLATFORM_LINUX)
endif()

# Set folder for IDE
set_target_properties(WulfNet PROPERTIES FOLDER "WulfNet")

# Source groups for IDE
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${WULFNET_SOURCES} ${WULFNET_HEADERS})

# =============================================================================
# Shader Compilation (GLSL -> SPIR-V)
# =============================================================================

find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")

if(GLSLC)
    set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Assets/Shaders/Compute")
    set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Assets/Shaders/Compute")

    set(COMP_SHADERS
        reduce_global
        reduce_final
        reduce_to_transform
        ifs_init_particles
        ifs_chaos_game
        ifs_iterated_expand
        ifs_clear_voxels
        ifs_voxelize
        ifs_clear_occlusion
        ifs_calc_occlusion
        ifs_lod_first
        ifs_lod_iterate
    )

    set(SPV_OUTPUTS "")
    foreach(SHADER ${COMP_SHADERS})
        set(INPUT_FILE "${SHADER_SOURCE_DIR}/${SHADER}.comp")
        set(OUTPUT_FILE "${SHADER_OUTPUT_DIR}/${SHADER}.spv")

        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${GLSLC} -o ${OUTPUT_FILE} ${INPUT_FILE}
            DEPENDS ${INPUT_FILE}
            COMMENT "Compiling ${SHADER}.comp -> ${SHADER}.spv"
        )

        list(APPEND SPV_OUTPUTS ${OUTPUT_FILE})
    endforeach()

    add_custom_target(CompileShaders ALL DEPENDS ${SPV_OUTPUTS})
    add_dependencies(WulfNet CompileShaders)
    message(STATUS "  glslc found: Shader compilation enabled")
else()
    message(STATUS "  glslc not found: Shaders must be pre-compiled")
endif()

message(STATUS "WulfNet Engine library configured")
