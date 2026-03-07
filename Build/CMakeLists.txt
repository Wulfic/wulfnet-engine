# =============================================================================
# WulfNet Engine - Build Directory Delegator
# =============================================================================
# This file exists for backward compatibility with build scripts in this
# directory. The canonical build entry point is the root CMakeLists.txt.
#
# Preferred usage:  cmake -S .. -B <build-dir>
# Legacy usage:     cmake -S .  -B <build-dir>   (from this directory)
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Delegate to the root CMakeLists.txt
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/wulfnet)
