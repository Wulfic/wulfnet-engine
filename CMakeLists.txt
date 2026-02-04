# =============================================================================
# WulfNet Engine - Root CMake Configuration
# =============================================================================
# A fully-featured, AAA-grade physics and game engine designed for modern
# multi-core hardware with massive parallelization support.
# =============================================================================

cmake_minimum_required(VERSION 3.25)

# -----------------------------------------------------------------------------
# Project Definition
# -----------------------------------------------------------------------------
project(WulfNetEngine
    VERSION 0.1.0
    DESCRIPTION "AAA-grade physics and game engine with massive parallelization"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# C++ Standard Configuration
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
include(FetchContent)

# GLFW - Windowing & Input
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
    GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# -----------------------------------------------------------------------------
# Build Type Configuration
# -----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel"
    )
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# -----------------------------------------------------------------------------
# Platform Detection
# -----------------------------------------------------------------------------
if(WIN32)
    set(WULFNET_PLATFORM_WINDOWS TRUE)
    set(WULFNET_PLATFORM_NAME "Windows")
    add_compile_definitions(WULFNET_PLATFORM_WINDOWS=1)
elseif(UNIX AND NOT APPLE)
    set(WULFNET_PLATFORM_LINUX TRUE)
    set(WULFNET_PLATFORM_NAME "Linux")
    add_compile_definitions(WULFNET_PLATFORM_LINUX=1)
elseif(APPLE)
    set(WULFNET_PLATFORM_MACOS TRUE)
    set(WULFNET_PLATFORM_NAME "macOS")
    add_compile_definitions(WULFNET_PLATFORM_MACOS=1)
endif()

message(STATUS "Target Platform: ${WULFNET_PLATFORM_NAME}")

# -----------------------------------------------------------------------------
# Compiler Detection & Configuration
# -----------------------------------------------------------------------------
if(MSVC)
    set(WULFNET_COMPILER_MSVC TRUE)
    
    # MSVC-specific flags (applied to all configurations)
    add_compile_options(
        /W4                     # High warning level
        /MP                     # Multi-processor compilation
        /permissive-            # Strict conformance mode
        /Zc:__cplusplus         # Correct __cplusplus macro
        /Zc:preprocessor        # Standard-conforming preprocessor
        /utf-8                  # UTF-8 source and execution charset
    )
    
    # Enable AVX2 for SIMD (if supported)
    add_compile_options(/arch:AVX2)
    
    # Per-configuration flags using generator expressions
    add_compile_options(
        "$<$<CONFIG:Debug>:/Od;/Zi;/RTC1>"
        "$<$<CONFIG:Release>:/O2;/Oi;/GL>"
        "$<$<CONFIG:RelWithDebInfo>:/O2;/Zi>"
    )
    
    add_link_options(
        "$<$<CONFIG:Release>:/LTCG>"
    )
    
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(WULFNET_COMPILER_GCC_LIKE TRUE)
    
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Werror                 # Warnings as errors
        -march=native           # Optimize for current CPU (enables AVX2 if supported)
        -ffast-math             # Aggressive floating-point optimizations
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g)
    else()
        add_compile_options(-O3 -flto)
        add_link_options(-flto)
    endif()
endif()

# -----------------------------------------------------------------------------
# SIMD Detection
# -----------------------------------------------------------------------------
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX512)
    set(WULFNET_SIMD_LEVEL "AVX512")
    add_compile_definitions(WULFNET_SIMD_AVX512=1)
elseif(COMPILER_SUPPORTS_AVX2)
    set(WULFNET_SIMD_LEVEL "AVX2")
    add_compile_definitions(WULFNET_SIMD_AVX2=1)
else()
    set(WULFNET_SIMD_LEVEL "SSE4.2")
    add_compile_definitions(WULFNET_SIMD_SSE42=1)
endif()

message(STATUS "SIMD Level: ${WULFNET_SIMD_LEVEL}")

# -----------------------------------------------------------------------------
# Output Directories
# -----------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(WULFNET_BUILD_TESTS "Build unit tests" ON)
option(WULFNET_BUILD_EXAMPLES "Build example applications" ON)
option(WULFNET_ENABLE_PROFILING "Enable Tracy profiler integration" ON)
option(WULFNET_ENABLE_GPU_PHYSICS "Enable GPU-accelerated physics (requires Vulkan)" ON)
option(WULFNET_USE_DOUBLE_PRECISION "Use double precision for physics" OFF)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
# Find Vulkan SDK
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan SDK found: ${Vulkan_INCLUDE_DIRS}")
    set(WULFNET_HAS_VULKAN TRUE)
    add_compile_definitions(WULFNET_HAS_VULKAN=1)
else()
    message(WARNING "Vulkan SDK not found. GPU features will be disabled.")
    set(WULFNET_HAS_VULKAN FALSE)
endif()

# Thread support
find_package(Threads REQUIRED)

# -----------------------------------------------------------------------------
# Engine Library Subdirectories
# -----------------------------------------------------------------------------
add_subdirectory(src/core)
add_subdirectory(src/physics)      # Phase 2 - Collision & Dynamics
add_subdirectory(src/rendering)    # Phase 4
# add_subdirectory(src/audio)        # Phase 5
# add_subdirectory(src/scene)        # Phase 3

# -----------------------------------------------------------------------------
# Main Engine Library (aggregates all subsystems)
# -----------------------------------------------------------------------------
add_library(WulfNetEngine INTERFACE)

target_link_libraries(WulfNetEngine INTERFACE
    WulfNet::Core
    WulfNetPhysics
    WulfNet::Rendering
    # WulfNet::Audio
    # WulfNet::Scene
)

add_library(WulfNet::Engine ALIAS WulfNetEngine)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(WULFNET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------
if(WULFNET_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS WulfNetEngine
    EXPORT WulfNetEngineTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== WulfNet Engine Configuration ===")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Platform:       ${WULFNET_PLATFORM_NAME}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD Level:     ${WULFNET_SIMD_LEVEL}")
message(STATUS "  Vulkan:         ${WULFNET_HAS_VULKAN}")
message(STATUS "  Tests:          ${WULFNET_BUILD_TESTS}")
message(STATUS "  Examples:       ${WULFNET_BUILD_EXAMPLES}")
message(STATUS "  Profiling:      ${WULFNET_ENABLE_PROFILING}")
message(STATUS "=====================================")
message(STATUS "")
