# =============================================================================
# WulfNet Engine - Root CMake Configuration
# =============================================================================
# Standalone build entry point for WulfNet Engine and all targets.
#
# BUILD INSTRUCTIONS:
# ------------------
# Option A (recommended — uses root CMakeLists.txt):
#   cmake -B out -S . -DCMAKE_BUILD_TYPE=Release
#   cmake --build out --config Release
#
# Option B (Jolt-first, unchanged from upstream):
#   cd build
#   cmake_vs2022_cl.bat
#   cmake --build VS2022_CL --config Release
#
# Both paths produce the same result.
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(WulfNetEngine
    VERSION 1.0.0
    DESCRIPTION "AAA-grade physics and game engine built on Jolt Physics"
    LANGUAGES CXX
)

message(STATUS "")
message(STATUS "=== WulfNet Engine v${PROJECT_VERSION} ===")
message(STATUS "")

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(WULFNET_BUILD_TESTS     "Build WulfNet test suites"    ON)
option(WULFNET_BUILD_EXAMPLES  "Build WulfNet examples"       ON)
option(WULFNET_ENABLE_TRACY    "Enable Tracy profiler"        OFF)

# Jolt Physics options (match defaults from build/CMakeLists.txt)
option(USE_ASSERTS             "Enable Jolt asserts"          OFF)
option(DOUBLE_PRECISION        "Use double precision math"    OFF)
option(GENERATE_DEBUG_SYMBOLS  "Generate debug symbols"       ON)
option(OVERRIDE_CXX_FLAGS      "Override CMAKE_CXX_FLAGS"    ON)
option(CROSS_PLATFORM_DETERMINISTIC "Cross platform deterministic" OFF)
option(INTERPROCEDURAL_OPTIMIZATION "Enable IPO/LTO"          ON)
option(FLOATING_POINT_EXCEPTIONS_ENABLED "Enable FP exceptions" ON)
option(CPP_EXCEPTIONS_ENABLED  "Enable C++ exceptions"        OFF)
option(CPP_RTTI_ENABLED        "Enable C++ RTTI"              OFF)
option(OBJECT_LAYER_BITS       "ObjectLayer bits (16 or 32)"  16)
option(USE_SSE4_1              "Enable SSE4.1"                ON)
option(USE_SSE4_2              "Enable SSE4.2"                ON)
option(USE_AVX                 "Enable AVX"                   ON)
option(USE_AVX2                "Enable AVX2"                  ON)
option(USE_AVX512              "Enable AVX512"                OFF)
option(USE_LZCNT               "Enable LZCNT"                 ON)
option(USE_TZCNT               "Enable TZCNT"                 ON)
option(USE_F16C                "Enable F16C"                  ON)
option(USE_FMADD               "Enable FMADD"                 ON)
option(ENABLE_ALL_WARNINGS     "Enable all warnings"          ON)

set(PHYSICS_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/)

# -----------------------------------------------------------------------------
# C++ Standard
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# OVERRIDE_CXX_FLAGS — replicate logic from build/CMakeLists.txt
# -----------------------------------------------------------------------------
if(OVERRIDE_CXX_FLAGS)
    if(MSVC)
        set(CMAKE_CXX_FLAGS_DEBUG "/GS /W4 /Zi /Od /Ob0 /RTC1")
        set(CMAKE_CXX_FLAGS_RELEASE "/GS- /W4 /O2 /Ob2 /Oi /Ot /fp:fast")
    endif()
endif()

# Detect platform
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows" OR "${CMAKE_SYSTEM_NAME}" STREQUAL "WindowsStore")
    set(WULFNET_PLATFORM_WINDOWS TRUE)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    set(WULFNET_PLATFORM_LINUX TRUE)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(WULFNET_PLATFORM_MACOS TRUE)
endif()

# --- Interprocedural Optimization ---
function(SET_INTERPROCEDURAL_OPTIMIZATION)
    if(INTERPROCEDURAL_OPTIMIZATION AND NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "WindowsStore")
        include(CheckIPOSupported OPTIONAL RESULT_VARIABLE CHECK_IPO_MODULE)
        if(CHECK_IPO_MODULE)
            check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_OUTPUT)
            if(IPO_SUPPORTED)
                set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON PARENT_SCOPE)
            endif()
        endif()
    endif()
endfunction()
SET_INTERPROCEDURAL_OPTIMIZATION()

# =============================================================================
# Jolt Physics (built from source, lives in Jolt/ subdirectory)
# =============================================================================
message(STATUS "Building Jolt Physics from source...")
include(${PHYSICS_REPO_ROOT}/Jolt/Jolt.cmake)

if(XCODE)
    set_property(TARGET Jolt PROPERTY XCODE_ATTRIBUTE_OTHER_CPLUSPLUSFLAGS[arch=x86_64] "$(inherited) -msse4.2 -mpopcnt")
endif()

# =============================================================================
# WulfNet Engine Library
# =============================================================================
message(STATUS "Configuring WulfNet Engine...")
add_subdirectory(WulfNet)

# =============================================================================
# WulfNet Tests
# =============================================================================
if(WULFNET_BUILD_TESTS)
    message(STATUS "Configuring WulfNet Tests...")
    add_subdirectory(WulfNetTests)
endif()

# =============================================================================
# WulfNet Examples
# =============================================================================
if(WULFNET_BUILD_EXAMPLES)
    message(STATUS "Configuring WulfNet Examples...")
    add_subdirectory(WulfNetExamples)
endif()

# =============================================================================
# Install Targets
# =============================================================================
include(GNUInstallDirs)

install(TARGETS WulfNet Jolt
    EXPORT WulfNetTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install WulfNet headers (preserving directory structure)
install(DIRECTORY WulfNet/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/WulfNet
    FILES_MATCHING PATTERN "*.h"
)

# Install Jolt headers
install(DIRECTORY Jolt/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Jolt
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inl"
)

# Generate and install CMake package config
install(EXPORT WulfNetTargets
    FILE WulfNetTargets.cmake
    NAMESPACE WulfNet::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/WulfNet
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/WulfNetConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/WulfNetConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/WulfNet
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/WulfNetConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/WulfNetConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/WulfNetConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/WulfNet
)

message(STATUS "")
message(STATUS "=== WulfNet Engine configured ===")
message(STATUS "")
