# =============================================================================
# WulfNet Engine - Tests CMake Configuration
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# WulfNet Unit Tests
set(WULFNET_TEST_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/WulfNetTests.cpp
)

add_executable(WulfNetTests ${WULFNET_TEST_SRC_FILES})
target_link_libraries(WulfNetTests LINK_PUBLIC WulfNet Jolt)

# Set C++ standard
target_compile_features(WulfNetTests PUBLIC cxx_std_17)

# Include directories
target_include_directories(WulfNetTests PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# MSVC-specific settings
# We need to remove /WX from this target because of vtable size mismatches
# between Jolt (no exceptions) and standard library headers
if(MSVC)
    # Get current compile options and remove /WX
    get_target_property(WULFNET_TESTS_CXX_FLAGS WulfNetTests COMPILE_OPTIONS)
    if(WULFNET_TESTS_CXX_FLAGS)
        list(REMOVE_ITEM WULFNET_TESTS_CXX_FLAGS "/WX")
        set_target_properties(WulfNetTests PROPERTIES COMPILE_OPTIONS "${WULFNET_TESTS_CXX_FLAGS}")
    endif()

    # Override compile flags to explicitly disable /WX and add warning suppressions
    target_compile_options(WulfNetTests PRIVATE
        /WX-  # Explicitly disable warnings as errors
        /wd4820 /wd4514 /wd4625 /wd4626 /wd5026 /wd5027
        /wd4464 /wd4100 /wd4365 /wd4061 /wd5219 /wd4668 /wd4530
        /wd4710 /wd4711 /wd4743  # vtable size differences, inline expansion warnings
        /GL-  # Disable whole program optimization to avoid LTCG issues
    )
    target_link_options(WulfNetTests PUBLIC
        "/SUBSYSTEM:CONSOLE"
        "/LTCG:OFF"  # Disable link-time code generation
    )
endif()

set_target_properties(WulfNetTests PROPERTIES FOLDER "WulfNet/Tests")

# Register as CTest
enable_testing()
add_test(NAME WulfNetTests COMMAND WulfNetTests)

message(STATUS "WulfNet Tests configured")
