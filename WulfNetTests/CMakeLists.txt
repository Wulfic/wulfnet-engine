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

# =============================================================================
# Jolt Unit Tests (doctest framework — merged from UnitTests/)
# =============================================================================
set(JOLT_UNIT_TEST_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestFramework.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestFramework.h
    ${CMAKE_CURRENT_SOURCE_DIR}/PhysicsTestContext.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PhysicsTestContext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestLayers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LoggingBodyActivationListener.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LoggingContactListener.h
    ${CMAKE_CURRENT_SOURCE_DIR}/doctest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Compute/ComputeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/ArrayTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/BinaryHeapTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/FPFlushDenormalsTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/HashCombineTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/InsertionSortTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/JobSystemTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/LinearCurveTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/PreciseMathTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/ScopeExitTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/STLLocalAllocatorTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/StringToolsTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/QuickSortTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/UnorderedSetTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/UnorderedMapTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry/ClosestPointTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry/ConvexHullBuilderTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry/EllipseTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry/EPATests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry/GJKTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry/PlaneTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry/RayAABoxTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/BVec16Tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/DMat44Tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/DVec3Tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/EigenValueSymmetricTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/HalfFloatTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/Mat44Tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/MathTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/MatrixTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/QuatTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/UVec4Tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/TrigonometryTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/Vec3Tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/Vec4Tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/VectorTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/ActiveEdgesTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/BroadPhaseTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/CastShapeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/CharacterVirtualTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/CollideShapeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/CollidePointTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/CollisionGroupTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/ContactListenerTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/ConvexVsTrianglesTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/DistanceConstraintTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/EstimateCollisionResponseTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/HeightFieldShapeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/HingeConstraintTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/MotionQualityLinearCastTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/MutableCompoundShapeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/ObjectLayerPairFilterTableTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/ObjectLayerPairFilterMaskTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/OffsetCenterOfMassShapeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/PathConstraintTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/PhysicsDeterminismTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/PhysicsStepListenerTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/PhysicsTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/RayShapeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/SensorTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/ShapeFilterTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/ShapeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/SixDOFConstraintTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/SliderConstraintTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/SoftBodyTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/SubShapeIDTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/TaperedCylinderShapeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/TransformedShapeTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics/WheeledVehicleTests.cpp
)

if (ENABLE_OBJECT_STREAM)
    list(APPEND JOLT_UNIT_TEST_SRC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/ObjectStream/ObjectStreamTest.cpp
    )
endif()

add_executable(UnitTests ${JOLT_UNIT_TEST_SRC_FILES})
target_include_directories(UnitTests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(UnitTests LINK_PUBLIC Jolt)
target_compile_features(UnitTests PUBLIC cxx_std_17)

# Precompiled header (Jolt.h)
if (CMAKE_GENERATOR STREQUAL "Ninja Multi-Config" AND MSVC)
    target_precompile_headers(UnitTests PRIVATE "${PHYSICS_REPO_ROOT}/WulfNet/Jolt/Jolt.h")
else()
    target_precompile_headers(UnitTests PRIVATE "$<$<NOT:$<CONFIG:ReleaseCoverage>>:${PHYSICS_REPO_ROOT}/WulfNet/Jolt/Jolt.h>")
endif()

if(MSVC)
    target_compile_options(UnitTests PRIVATE
        /WX-
        /wd4820 /wd4514 /wd4625 /wd4626 /wd5026 /wd5027
        /wd4464 /wd4100 /wd4365 /wd4061 /wd5219 /wd4668 /wd4530
        /wd4710 /wd4711 /wd4743
        /GL-
    )
    target_link_options(UnitTests PUBLIC "/SUBSYSTEM:CONSOLE" "/LTCG:OFF")
endif()

set_target_properties(UnitTests PROPERTIES FOLDER "WulfNet/Tests")
add_test(NAME UnitTests COMMAND UnitTests)

message(STATUS "Jolt Unit Tests configured")

# =============================================================================
# Jolt Performance Test (merged from PerformanceTest/)
# =============================================================================
set(JOLT_PERF_TEST_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks/PerformanceTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks/PerformanceTestScene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks/PerfTestLayers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks/PyramidScene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks/RagdollScene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks/ConvexVsMeshScene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks/CharacterVirtualScene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks/LargeMeshScene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks/MaxBodiesScene.h
)

add_executable(PerformanceTest ${JOLT_PERF_TEST_SRC_FILES})
target_include_directories(PerformanceTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks)
target_link_libraries(PerformanceTest LINK_PUBLIC Jolt)
target_compile_features(PerformanceTest PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(PerformanceTest PRIVATE
        /WX-
        /wd4820 /wd4514 /wd4625 /wd4626 /wd5026 /wd5027
        /wd4464 /wd4100 /wd4365 /wd4061 /wd5219 /wd4668 /wd4530
        /wd4710 /wd4711 /wd4743
        /GL-
    )
    target_link_options(PerformanceTest PUBLIC "/SUBSYSTEM:CONSOLE" "/LTCG:OFF")
endif()

set_target_properties(PerformanceTest PROPERTIES
    FOLDER "WulfNet/Tests"
    VS_DEBUGGER_WORKING_DIRECTORY "${PHYSICS_REPO_ROOT}"
)
add_test(NAME PerformanceTest COMMAND PerformanceTest)

message(STATUS "Jolt Performance Test configured")
