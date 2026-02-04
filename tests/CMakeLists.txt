# =============================================================================
# WulfNet Engine - Tests
# =============================================================================

cmake_minimum_required(VERSION 3.25)

project(WulfNet_Tests LANGUAGES CXX)

# Find or fetch a testing framework
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
)
FetchContent_MakeAvailable(Catch2)

# Test executable
add_executable(WulfNet_Tests
    test_types.cpp
    test_allocators.cpp
    test_math.cpp
    test_jobs.cpp
    test_physics.cpp
    test_physics_sleep.cpp
    test_collision_convexhull.cpp
    test_collision_gjk.cpp
    test_collision_epa.cpp
)

target_link_libraries(WulfNet_Tests PRIVATE
    WulfNetCore
    WulfNetPhysics
    Catch2::Catch2WithMain
)

target_compile_features(WulfNet_Tests PRIVATE cxx_std_20)

# Work around Catch2 compatibility issue with /Zc:preprocessor
if(MSVC)
    target_compile_options(WulfNet_Tests PRIVATE /Zc:preprocessor-)
    # Force include our compatibility header before everything else
    target_compile_options(WulfNet_Tests PRIVATE /FI"${CMAKE_CURRENT_SOURCE_DIR}/tests_pch.h")
endif()

# Enable testing
include(CTest)
include(Catch)
catch_discover_tests(WulfNet_Tests)
