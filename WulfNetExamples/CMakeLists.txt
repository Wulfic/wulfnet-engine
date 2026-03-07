# =============================================================================
# WulfNet Engine - Examples CMake Configuration
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# =============================================================================
# HelloWulfNet example
# =============================================================================

set(HELLO_WULFNET_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/HelloWulfNet/main.cpp
)

add_executable(HelloWulfNet ${HELLO_WULFNET_SRC_FILES})
target_link_libraries(HelloWulfNet LINK_PUBLIC WulfNet Jolt)

# Set C++ standard
target_compile_features(HelloWulfNet PUBLIC cxx_std_17)

# Include directories
target_include_directories(HelloWulfNet PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# MSVC-specific settings
# We need to disable /WX because of vtable size mismatches
# between Jolt (no exceptions) and standard library headers
if(MSVC)
    target_compile_options(HelloWulfNet PRIVATE
        /WX-  # Explicitly disable warnings as errors
        /wd4820 /wd4514 /wd4625 /wd4626 /wd5026 /wd5027
        /wd4464 /wd4100 /wd4365 /wd4061 /wd5219 /wd4668 /wd4530
        /wd4710 /wd4711 /wd4743  # vtable size differences, inline expansion warnings
        /GL-  # Disable whole program optimization to avoid LTCG issues
    )
    target_link_options(HelloWulfNet PUBLIC
        "/SUBSYSTEM:CONSOLE"
        "/LTCG:OFF"  # Disable link-time code generation
    )
endif()

set_target_properties(HelloWulfNet PROPERTIES FOLDER "WulfNet/Examples")

# =============================================================================
# ComputeExample - GPU Compute demonstration
# =============================================================================

set(COMPUTE_EXAMPLE_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/ComputeExample/main.cpp
)

add_executable(ComputeExample ${COMPUTE_EXAMPLE_SRC_FILES})
target_link_libraries(ComputeExample LINK_PUBLIC WulfNet Jolt)

# Set C++ standard
target_compile_features(ComputeExample PUBLIC cxx_std_17)

# Include directories
target_include_directories(ComputeExample PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# MSVC-specific settings
if(MSVC)
    target_compile_options(ComputeExample PRIVATE
        /WX-
        /wd4820 /wd4514 /wd4625 /wd4626 /wd5026 /wd5027
        /wd4464 /wd4100 /wd4365 /wd4061 /wd5219 /wd4668 /wd4530
        /wd4710 /wd4711 /wd4743 /wd4191  # Vulkan function pointer warnings
        /GL-
    )
    target_link_options(ComputeExample PUBLIC
        "/SUBSYSTEM:CONSOLE"
        "/LTCG:OFF"
    )
endif()

set_target_properties(ComputeExample PROPERTIES FOLDER "WulfNet/Examples")

# =============================================================================
# IFSExample - GPU IFS Fractal demonstration
# =============================================================================

set(IFS_EXAMPLE_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/IFSExample/main.cpp
)

add_executable(IFSExample ${IFS_EXAMPLE_SRC_FILES})
target_link_libraries(IFSExample LINK_PUBLIC WulfNet Jolt)

target_compile_features(IFSExample PUBLIC cxx_std_17)

target_include_directories(IFSExample PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if(MSVC)
    target_compile_options(IFSExample PRIVATE
        /WX-
        /wd4820 /wd4514 /wd4625 /wd4626 /wd5026 /wd5027
        /wd4464 /wd4100 /wd4365 /wd4061 /wd5219 /wd4668 /wd4530
        /wd4710 /wd4711 /wd4743 /wd4191
        /GL-
    )
    target_link_options(IFSExample PUBLIC
        "/SUBSYSTEM:CONSOLE"
        "/LTCG:OFF"
    )
endif()

set_target_properties(IFSExample PROPERTIES FOLDER "WulfNet/Examples")

# =============================================================================
# SoftRasterExample - CPU Software Rasterizer demonstration
# =============================================================================

set(SOFTRASTER_EXAMPLE_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/SoftRasterExample/main.cpp
)

add_executable(SoftRasterExample ${SOFTRASTER_EXAMPLE_SRC_FILES})
target_link_libraries(SoftRasterExample LINK_PUBLIC WulfNet Jolt)

target_compile_features(SoftRasterExample PUBLIC cxx_std_17)

target_include_directories(SoftRasterExample PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if(MSVC)
    target_compile_options(SoftRasterExample PRIVATE
        /WX-
        /wd4820 /wd4514 /wd4625 /wd4626 /wd5026 /wd5027
        /wd4464 /wd4100 /wd4365 /wd4061 /wd5219 /wd4668 /wd4530
        /wd4710 /wd4711 /wd4743
        /GL-
    )
    target_link_options(SoftRasterExample PUBLIC
        "/SUBSYSTEM:CONSOLE"
        "/LTCG:OFF"
    )
endif()

set_target_properties(SoftRasterExample PROPERTIES FOLDER "WulfNet/Examples")

# =============================================================================
# EngineDemo - Flagship engine demonstration
# =============================================================================

set(ENGINE_DEMO_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/EngineDemo/main.cpp
)

add_executable(EngineDemo ${ENGINE_DEMO_SRC_FILES})
target_link_libraries(EngineDemo LINK_PUBLIC WulfNet Jolt)

target_compile_features(EngineDemo PUBLIC cxx_std_17)

target_include_directories(EngineDemo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if(MSVC)
    target_compile_options(EngineDemo PRIVATE
        /WX-
        /wd4820 /wd4514 /wd4625 /wd4626 /wd5026 /wd5027
        /wd4464 /wd4100 /wd4365 /wd4061 /wd5219 /wd4668 /wd4530
        /wd4710 /wd4711 /wd4743
        /GL-
    )
    target_link_options(EngineDemo PUBLIC
        "/SUBSYSTEM:CONSOLE"
        "/LTCG:OFF"
    )
endif()

set_target_properties(EngineDemo PROPERTIES FOLDER "WulfNet/Examples")

# =============================================================================
# Samples - Interactive physics test/demo application (DX12/Vulkan/Metal)
# Requires WulfNetFramework (graphics backend)
# =============================================================================
if (TEST_FRAMEWORK_AVAILABLE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/Samples.cmake)
endif()

message(STATUS "WulfNet Examples configured")
