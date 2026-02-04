# =============================================================================
# WulfNet Engine - Examples
# =============================================================================

cmake_minimum_required(VERSION 3.25)

# Basic example application
add_executable(Example_Basic
    basic/main.cpp
)

target_link_libraries(Example_Basic PRIVATE
    WulfNet::Core
)

target_compile_features(Example_Basic PRIVATE cxx_std_20)

# Job system benchmark
add_executable(Example_JobBenchmark
    job_benchmark/main.cpp
)

target_link_libraries(Example_JobBenchmark PRIVATE
    WulfNet::Core
)

target_compile_features(Example_JobBenchmark PRIVATE cxx_std_20)

# Test World (Physics Visualizer)
add_executable(Example_TestWorld
    test_world/TestWorld.cpp
)

target_link_libraries(Example_TestWorld PRIVATE
    WulfNet::Core
    WulfNetPhysics
    WulfNet::Rendering
)

if(WIN32)
    target_link_libraries(Example_TestWorld PRIVATE opengl32)
elseif(UNIX)
    target_link_libraries(Example_TestWorld PRIVATE GL)
endif()

target_compile_features(Example_TestWorld PRIVATE cxx_std_20)
