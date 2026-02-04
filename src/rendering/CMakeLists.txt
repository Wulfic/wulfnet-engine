set(RENDERING_SOURCES
    GLLoader.h
    GLLoader.cpp
    DebugRenderer.h
    DebugRenderer.cpp
)

add_library(WulfNetRendering STATIC ${RENDERING_SOURCES})
add_library(WulfNet::Rendering ALIAS WulfNetRendering)

target_include_directories(WulfNetRendering PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_link_libraries(WulfNetRendering PUBLIC 
    WulfNet::Core
    glfw
)

if(WIN32)
    target_link_libraries(WulfNetRendering PRIVATE opengl32)
elseif(UNIX)
    target_link_libraries(WulfNetRendering PRIVATE GL)
endif()

target_compile_features(WulfNetRendering PUBLIC cxx_std_20)
