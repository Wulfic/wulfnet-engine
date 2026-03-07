# =============================================================================
# WulfNet Engine - Tests CMake Configuration
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# WulfNet Unit Tests
set(WULFNET_TEST_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/WulfNetTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CoreTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PhysicsWorldTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VulkanComputeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IFSTransformTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SoftwareRendererTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PipelineIntegrationTests.cpp
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

# =============================================================================
# WulfNet Extended Tests (SystemMonitor, Advanced Physics, Integration, Water V3)
# =============================================================================
set(WULFNET_EXTENDED_TEST_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/WulfNetExtendedTests.cpp      ${CMAKE_CURRENT_SOURCE_DIR}/WaterSystemV3Tests.cpp    ${CMAKE_CURRENT_SOURCE_DIR}/SystemMonitorTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AdvancedPhysicsTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ConstitutiveModelTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TerrainDeformationTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MPMRigidCouplingTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GaseousSystemTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DestructionSystemTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShadowMapTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GlobalIlluminationTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VolumetricRendererTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderPipelineTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AudioEngineTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AcousticSystemTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SpatialAudioTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PerformanceBenchmarks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/EngineLifecycleTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ThreadingTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FrameAllocatorTests.cpp
)

add_executable(WulfNetExtendedTests ${WULFNET_EXTENDED_TEST_SRC_FILES})
target_link_libraries(WulfNetExtendedTests LINK_PUBLIC WulfNet Jolt)

# Set C++ standard
target_compile_features(WulfNetExtendedTests PUBLIC cxx_std_17)

# Include directories
target_include_directories(WulfNetExtendedTests PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# MSVC-specific settings (same as WulfNetTests)
if(MSVC)
    get_target_property(WULFNET_EXT_TESTS_CXX_FLAGS WulfNetExtendedTests COMPILE_OPTIONS)
    if(WULFNET_EXT_TESTS_CXX_FLAGS)
        list(REMOVE_ITEM WULFNET_EXT_TESTS_CXX_FLAGS "/WX")
        set_target_properties(WulfNetExtendedTests PROPERTIES COMPILE_OPTIONS "${WULFNET_EXT_TESTS_CXX_FLAGS}")
    endif()

    target_compile_options(WulfNetExtendedTests PRIVATE
        /WX-
        /wd4820 /wd4514 /wd4625 /wd4626 /wd5026 /wd5027
        /wd4464 /wd4100 /wd4365 /wd4061 /wd5219 /wd4668 /wd4530
        /wd4710 /wd4711 /wd4743
        /GL-
    )
    target_link_options(WulfNetExtendedTests PUBLIC
        "/SUBSYSTEM:CONSOLE"
        "/LTCG:OFF"
    )
endif()

set_target_properties(WulfNetExtendedTests PROPERTIES FOLDER "WulfNet/Tests")

# Register as CTest
enable_testing()
add_test(NAME WulfNetTests COMMAND WulfNetTests)
add_test(NAME WulfNetExtendedTests COMMAND WulfNetExtendedTests)

# Add per-suite CTest entries for fine-grained test selection
add_test(NAME WulfNetExtended_Monitor COMMAND WulfNetExtendedTests --suite=monitor)
add_test(NAME WulfNetExtended_Physics COMMAND WulfNetExtendedTests --suite=physics)
add_test(NAME WulfNetExtended_Integration COMMAND WulfNetExtendedTests --suite=integration)
add_test(NAME WulfNetExtended_MPM COMMAND WulfNetExtendedTests --suite=mpm)
add_test(NAME WulfNetExtended_Terrain COMMAND WulfNetExtendedTests --suite=terrain)
add_test(NAME WulfNetExtended_Coupling COMMAND WulfNetExtendedTests --suite=coupling)
add_test(NAME WulfNetExtended_Gaseous COMMAND WulfNetExtendedTests --suite=gaseous)
add_test(NAME WulfNetExtended_Destruction COMMAND WulfNetExtendedTests --suite=destruction)
add_test(NAME WulfNetExtended_Shadow COMMAND WulfNetExtendedTests --suite=shadow)
add_test(NAME WulfNetExtended_GI COMMAND WulfNetExtendedTests --suite=gi)
add_test(NAME WulfNetExtended_Volumetric COMMAND WulfNetExtendedTests --suite=volumetric)
add_test(NAME WulfNetExtended_RenderPipeline COMMAND WulfNetExtendedTests --suite=pipeline)
add_test(NAME WulfNetExtended_AudioEngine COMMAND WulfNetExtendedTests --suite=audio)
add_test(NAME WulfNetExtended_AcousticSystem COMMAND WulfNetExtendedTests --suite=acoustic)
add_test(NAME WulfNetExtended_SpatialAudio COMMAND WulfNetExtendedTests --suite=spatial)
add_test(NAME WulfNetExtended_PerfBenchmarks COMMAND WulfNetExtendedTests --suite=benchmark)
add_test(NAME WulfNetExtended_Engine COMMAND WulfNetExtendedTests --suite=engine)
add_test(NAME WulfNetExtended_Threading COMMAND WulfNetExtendedTests --suite=threading)

message(STATUS "WulfNet Tests configured")
message(STATUS "WulfNet Extended Tests configured")
