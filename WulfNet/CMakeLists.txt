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
    # Core
    Core/Logging/Logger.cpp
    Core/Profiling/Profiler.cpp
    Core/System/SystemMonitor.cpp

    # Physics Integration
    Physics/Integration/PhysicsWorld.cpp

    # Physics - Fluids (MPM)
    Physics/Fluids/FluidSystem.cpp
    Physics/Fluids/FluidGrid.cpp

    # Physics - Fluids (CO-FLIP)
    Physics/Fluids/COFLIPSystem.cpp
    Physics/Fluids/FluidSurface.cpp

    # GPU Compute
    Compute/Vulkan/VulkanContext.cpp
    Compute/Memory/ComputeBuffer.cpp
    Compute/Shaders/ComputePipeline.cpp
    Compute/Fluids/VulkanFluidCompute.cpp
)

set(WULFNET_HEADERS
    # Main header
    WulfNet.h

    # Core
    Core/Logging/Logger.h
    Core/Profiling/Profiler.h
    Core/System/SystemMonitor.h

    # Physics Integration
    Physics/Integration/PhysicsWorld.h

    # Physics - Fluids (MPM)
    Physics/Fluids/FluidParticle.h
    Physics/Fluids/FluidGrid.h
    Physics/Fluids/FluidSystem.h

    # Physics - Fluids (CO-FLIP)
    Physics/Fluids/COFLIPSystem.h
    Physics/Fluids/FluidSurface.h

    # GPU Compute
    Compute/Compute.h
    Compute/Vulkan/VulkanContext.h
    Compute/Memory/ComputeBuffer.h
    Compute/Shaders/ComputePipeline.h
    Compute/Fluids/VulkanFluidCompute.h
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
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(WulfNet PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(WulfNet PRIVATE WULFNET_HAS_OPENMP)
    message(STATUS "  OpenMP found: CPU parallelization enabled")
else()
    message(STATUS "  OpenMP not found: CPU parallelization will be disabled")
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

message(STATUS "WulfNet Engine library configured")
