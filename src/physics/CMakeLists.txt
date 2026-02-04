# =============================================================================
# WulfNet Engine - Physics Module
# =============================================================================
# Physics simulation including collision detection, rigid body dynamics,
# soft body physics, fluids, and deformables.
# =============================================================================

set(PHYSICS_SOURCES
    # Collision Detection
    collision/AABB.cpp
    collision/BoundingSphere.cpp
    collision/CollisionShape.cpp
    collision/Broadphase.cpp
    collision/Narrowphase.cpp
    collision/GJK.cpp
    
    # Dynamics (Phase 2)
    dynamics/RigidBody.cpp
    dynamics/RigidBodyWorld.cpp
    
    # Constraints (Phase 2)
    constraints/ContactSolver.cpp
    constraints/Joint.cpp

    # Soft Body (Phase 2)
    softbody/Cloth.cpp
)

set(PHYSICS_HEADERS
    # Collision Detection
    collision/AABB.h
    collision/BoundingSphere.h
    collision/CollisionShape.h
    collision/Broadphase.h
    collision/CollisionTypes.h
    collision/Narrowphase.h
    collision/GJK.h
    
    # Dynamics (Phase 2)
    dynamics/RigidBody.h
    dynamics/RigidBodyWorld.h
    
    # Constraints (Phase 2)
    constraints/ContactSolver.h
    constraints/Joint.h

    # Soft Body (Phase 2)
    softbody/Cloth.h
)

add_library(WulfNetPhysics STATIC
    ${PHYSICS_SOURCES}
    ${PHYSICS_HEADERS}
)

target_include_directories(WulfNetPhysics
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(WulfNetPhysics
    PUBLIC
        WulfNetCore
)

# Enable SIMD optimizations
if(MSVC)
    target_compile_options(WulfNetPhysics PRIVATE /arch:AVX2)
else()
    target_compile_options(WulfNetPhysics PRIVATE -mavx2 -mfma)
endif()
