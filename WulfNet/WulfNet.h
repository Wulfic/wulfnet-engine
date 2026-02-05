// =============================================================================
// WulfNet Engine - WulfNet.h
// =============================================================================
// Main include header for WulfNet Engine.
// Include this header to get access to all WulfNet functionality.
// =============================================================================

#pragma once

// Version information
#define WULFNET_VERSION_MAJOR 0
#define WULFNET_VERSION_MINOR 1
#define WULFNET_VERSION_PATCH 0
#define WULFNET_VERSION_STRING "0.1.0"

// =============================================================================
// Core Systems
// =============================================================================

#include "Core/Logging/Logger.h"
#include "Core/Profiling/Profiler.h"

// =============================================================================
// Physics Systems
// =============================================================================

#include "Physics/Integration/PhysicsWorld.h"

// =============================================================================
// GPU Compute Systems
// =============================================================================

#include "Compute/Compute.h"

// =============================================================================
// Namespace Alias
// =============================================================================

// For convenience, you can use WN:: instead of WulfNet::
namespace WN = WulfNet;
