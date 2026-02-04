# =============================================================================
# WulfNet Engine - Core Module
# =============================================================================
# Foundation systems: memory, jobs, math, containers, platform abstraction
# =============================================================================

# -----------------------------------------------------------------------------
# Core Library Sources
# -----------------------------------------------------------------------------

# Memory subsystem
set(MEMORY_SOURCES
    memory/Allocator.cpp
    memory/LinearAllocator.cpp
    memory/PoolAllocator.cpp
    memory/StackAllocator.cpp
)

set(MEMORY_HEADERS
    memory/Allocator.h
    memory/LinearAllocator.h
    memory/PoolAllocator.h
    memory/StackAllocator.h
    memory/Memory.h
)

# Job system
set(JOBS_SOURCES
    jobs/JobSystem.cpp
    jobs/WorkStealingQueue.cpp
    jobs/Fiber.cpp
)

set(JOBS_HEADERS
    jobs/JobSystem.h
    jobs/WorkStealingQueue.h
    jobs/Fiber.h
    jobs/Job.h
)

# Math library
set(MATH_SOURCES
    math/Math.cpp
)

set(MATH_HEADERS
    math/Math.h
    math/Vec3.h
    math/Vec4.h
    math/Mat4.h
    math/Quat.h
    math/SIMD.h
    math/MathUtils.h
)

# Containers (Future - Phase 2+)
# set(CONTAINERS_HEADERS
#     containers/Array.h
#     containers/HashMap.h
#     containers/RingBuffer.h
# )
set(CONTAINERS_HEADERS)

# Platform abstraction
set(PLATFORM_SOURCES
    platform/Platform.cpp
    platform/Window.cpp
)

set(PLATFORM_HEADERS
    platform/Platform.h
    platform/Window.h
    # Window.h, Input.h, Timer.h - to be added in Phase 6
)

# Core types and utilities
set(CORE_HEADERS
    Types.h
    Assert.h
    Log.h
    Profiler.h
)

set(CORE_SOURCES
    Log.cpp
    Profiler.cpp
)

# Platform-specific sources
# Note: Window system sources will be added in Phase 6 (Test World)
if(WULFNET_PLATFORM_WINDOWS)
    list(APPEND PLATFORM_SOURCES
        platform/Platform.cpp
    )
elseif(WULFNET_PLATFORM_LINUX)
    list(APPEND PLATFORM_SOURCES
        platform/Platform.cpp
    )
endif()

# -----------------------------------------------------------------------------
# Core Library Target
# -----------------------------------------------------------------------------
add_library(WulfNetCore STATIC
    ${MEMORY_SOURCES}
    ${JOBS_SOURCES}
    ${MATH_SOURCES}
    ${PLATFORM_SOURCES}
    ${CORE_SOURCES}
    
    # Headers (for IDE visibility)
    ${MEMORY_HEADERS}
    ${JOBS_HEADERS}
    ${MATH_HEADERS}
    ${CONTAINERS_HEADERS}
    ${PLATFORM_HEADERS}
    ${CORE_HEADERS}
)

add_library(WulfNet::Core ALIAS WulfNetCore)

# -----------------------------------------------------------------------------
# Include Directories
# -----------------------------------------------------------------------------
target_include_directories(WulfNetCore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
target_link_libraries(WulfNetCore
    PUBLIC
        Threads::Threads
        glfw
)

if(WULFNET_PLATFORM_WINDOWS)
    target_link_libraries(WulfNetCore
        PUBLIC
            synchronization     # WaitOnAddress for job system
            kernel32
    )
endif()

# -----------------------------------------------------------------------------
# Precompiled Header (optional, speeds up compilation)
# -----------------------------------------------------------------------------
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
    target_precompile_headers(WulfNetCore
        PRIVATE
            <cstdint>
            <cstddef>
            <cstring>
            <memory>
            <atomic>
            <thread>
    )
endif()

# -----------------------------------------------------------------------------
# Compiler Features
# -----------------------------------------------------------------------------
target_compile_features(WulfNetCore PUBLIC cxx_std_20)
