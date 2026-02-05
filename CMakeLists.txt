# =============================================================================
# WulfNet Engine - Root CMake Configuration
# =============================================================================
# A fully-featured, AAA-grade physics and game engine built on Jolt Physics.
#
# This CMakeLists.txt serves as the root build configuration that:
# 1. Builds Jolt Physics as the foundation
# 2. Builds WulfNet Engine extensions
# 3. Builds examples and tests
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(WulfNetEngine
    VERSION 0.1.0
    DESCRIPTION "AAA-grade physics and game engine built on Jolt Physics"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(WULFNET_BUILD_JOLT_SAMPLES "Build Jolt Physics samples and viewers" ON)
option(WULFNET_BUILD_JOLT_TESTS "Build Jolt Physics unit tests" ON)
option(WULFNET_BUILD_WULFNET "Build WulfNet Engine extensions" ON)
option(WULFNET_BUILD_EXAMPLES "Build WulfNet examples" ON)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Set the root directory for finding Jolt includes
set(PHYSICS_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# Output Directories
# -----------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Per-configuration output directories
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
endforeach()

# -----------------------------------------------------------------------------
# Build Jolt Physics via its Build directory
# -----------------------------------------------------------------------------
# The Build/CMakeLists.txt contains all Jolt build configuration
# We include it as a subdirectory to build Jolt
add_subdirectory(Build)

# -----------------------------------------------------------------------------
# WulfNet Engine Extensions (Future)
# -----------------------------------------------------------------------------
if(WULFNET_BUILD_WULFNET AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/WulfNet/CMakeLists.txt")
    add_subdirectory(WulfNet)
endif()

# -----------------------------------------------------------------------------
# WulfNet Examples (Future)
# -----------------------------------------------------------------------------
if(WULFNET_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/WulfNetExamples/CMakeLists.txt")
    add_subdirectory(WulfNetExamples)
endif()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== WulfNet Engine Configuration ===")
message(STATUS "  WulfNet Version:     ${PROJECT_VERSION}")
message(STATUS "  Jolt Samples:        ${WULFNET_BUILD_JOLT_SAMPLES}")
message(STATUS "  Jolt Tests:          ${WULFNET_BUILD_JOLT_TESTS}")
message(STATUS "  WulfNet Extensions:  ${WULFNET_BUILD_WULFNET}")
message(STATUS "  WulfNet Examples:    ${WULFNET_BUILD_EXAMPLES}")
message(STATUS "=====================================")
message(STATUS "")
