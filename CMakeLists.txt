# =============================================================================
# WulfNet Engine - Root CMake Configuration
# =============================================================================
# A fully-featured, AAA-grade physics and game engine built on Jolt Physics.
#
# BUILD INSTRUCTIONS:
# ------------------
# For Jolt Physics (the foundation):
#   cd Build
#   cmake_vs2022_cl.bat      (Windows)
#   ./cmake_linux_clang_gcc.sh Release clang++  (Linux)
#
# This root CMakeLists.txt is for future WulfNet-specific builds.
# Currently, use the Build/ directory scripts to build Jolt Physics.
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(WulfNetEngine
    VERSION 0.1.0
    DESCRIPTION "AAA-grade physics and game engine built on Jolt Physics"
    LANGUAGES CXX
)

message(STATUS "")
message(STATUS "=== WulfNet Engine ===")
message(STATUS "")
message(STATUS "To build Jolt Physics (the foundation):")
message(STATUS "  cd Build")
message(STATUS "  cmake_vs2022_cl.bat  (Windows)")
message(STATUS "  ./cmake_linux_clang_gcc.sh Release clang++  (Linux)")
message(STATUS "")
message(STATUS "WulfNet extensions will be added in future phases.")
message(STATUS "See ENGINE_PLAN.md for the development roadmap.")
message(STATUS "")

# -----------------------------------------------------------------------------
# Options (for future WulfNet builds)
# -----------------------------------------------------------------------------
option(WULFNET_BUILD_EXTENSIONS "Build WulfNet Engine extensions" ON)
option(WULFNET_BUILD_EXAMPLES "Build WulfNet examples" ON)

# -----------------------------------------------------------------------------
# WulfNet Engine Extensions (Future - Phase 2+)
# -----------------------------------------------------------------------------
if(WULFNET_BUILD_EXTENSIONS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/WulfNet/CMakeLists.txt")
    # WulfNet will be built here once development begins
    # add_subdirectory(WulfNet)
    message(STATUS "WulfNet extensions directory exists but is not yet implemented.")
endif()

