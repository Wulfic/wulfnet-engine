// =============================================================================
// WulfNet Engine - WulfNet.h
// =============================================================================
// Main include header for WulfNet Engine.
// Include this header to get access to all WulfNet functionality.
//
// Modules:
//   Core        - Logging, profiling, system monitoring
//   Physics     - Jolt integration, fluids, MPM, gaseous, destruction, terrain
//   Compute     - Vulkan GPU compute, shader pipeline, parallel reduction
//   Procedural  - IFS fractal generation
//   Rendering   - Software rasterizer, shadows, GI, volumetrics, pipeline
//   Audio       - Mixer, acoustic simulation, spatial audio (HRTF/Ambisonics)
// =============================================================================

#pragma once

// Version information
#define WULFNET_VERSION_MAJOR 0
#define WULFNET_VERSION_MINOR 8
#define WULFNET_VERSION_PATCH 0
#define WULFNET_VERSION_STRING "0.8.0"

// =============================================================================
// Core Systems
// =============================================================================

#include "Core/Logging/Logger.h"
#include "Core/Profiling/Profiler.h"
#include "Core/System/SystemMonitor.h"
#include "Core/Math/PerlinNoise.h"

// =============================================================================
// Physics Systems
// =============================================================================

#include "Physics/Integration/PhysicsWorld.h"

// Physics - Water (SWE 2.5D)
#include "Physics/WaterSystemV3.h"

// Physics - Material Point Method
#include "Physics/MPM/ConstitutiveModel.h"
#include "Physics/MPM/MPMRigidCoupling.h"

// Physics - Terrain Deformation
#include "Physics/Terrain/TerrainDeformation.h"

// Physics - Gaseous Simulation
#include "Physics/Gaseous/GaseousSystem.h"

// Physics - Destruction
#include "Physics/Destruction/DestructionSystem.h"

// =============================================================================
// GPU Compute Systems
// =============================================================================

#include "Compute/Compute.h"

// =============================================================================
// Procedural Systems
// =============================================================================

#include "Procedural/IFS/AffineTransform.h"
#include "Procedural/IFS/TransformPresets.h"
#include "Procedural/IFS/TransformBlender.h"
#include "Procedural/IFS/IFSSystem.h"

// =============================================================================
// Rendering Systems
// =============================================================================

#include "Rendering/SoftwareRasterizer/SoftRasterTypes.h"
#include "Rendering/SoftwareRasterizer/GBuffer.h"
#include "Rendering/SoftwareRasterizer/SoftwareRasterizer.h"
#include "Rendering/SoftwareRasterizer/DeferredShading.h"
#include "Rendering/SoftwareRasterizer/OcclusionCuller.h"
#include "Rendering/SoftwareRasterizer/ShadowMap.h"
#include "Rendering/SoftwareRasterizer/GlobalIllumination.h"
#include "Rendering/SoftwareRasterizer/VolumetricRenderer.h"
#include "Rendering/SoftwareRasterizer/RenderPipeline.h"

// =============================================================================
// Audio Systems
// =============================================================================

#include "Audio/Core/AudioTypes.h"
#include "Audio/Core/AudioMixer.h"
#include "Audio/Acoustics/AcousticSystem.h"
#include "Audio/Spatial/SpatialAudio.h"

// =============================================================================
// Namespace Alias
// =============================================================================

// For convenience, you can use WN:: instead of WulfNet::
namespace WN = WulfNet;
