// =============================================================================
// WulfNet Engine - IFS (Iterated Function System)
// =============================================================================
// GPU-accelerated fractal particle system using the chaos game algorithm
// and deterministic iterated expansion with affine transformations.
// Ported from IteratedFunctionSystem.cs / ChaosGame.cs (Unity IFS reference).
// =============================================================================

#pragma once

#include "WulfNet/Compute/Pipelines/ComputePipeline.h"
#include "WulfNet/Compute/Memory/ComputeBuffer.h"
#include "WulfNet/Compute/Reduction/ParallelReduction.h"
#include "WulfNet/Procedural/IFS/AffineTransform.h"
#include "WulfNet/Procedural/IFS/TransformPresets.h"
#include "WulfNet/Procedural/IFS/TransformBlender.h"
#include <string>
#include <vector>

namespace WulfNet {

// =============================================================================
// IFS Configuration
// =============================================================================

struct IFSConfig {
    // Particle initialization
    uint32_t cubeResolution = 32;       // Cube grid resolution (particles = res^3)
    float cubeSize = 0.01f;             // Spacing between grid points

    // Chaos game
    uint32_t chaosIterationsPerFrame = 8;

    // Voxelization
    uint32_t voxelGridSize = 64;        // Voxel grid dimension
    uint32_t voxelGridBounds = 2;       // World-space bounds of voxel grid

    // LOD prediction
    uint32_t lodIterations = 4;         // Iterations for bounds prediction

    // Framing
    float targetBoundsSize = 1.0f;
    float scalePadding = 0.9f;

    // Initial preset
    IFSPreset initialPreset = IFSPreset::SierpinskiTriangle3D;

    // Shader path
    std::string shaderPath = "Assets/Shaders/Compute";
};

// =============================================================================
// IFS Push Constants (must match shader layouts)
// =============================================================================

struct IFSInitParams {
    uint32_t cubeResolution;
    float cubeSize;
};

struct IFSChaosParams {
    uint32_t seed;
    uint32_t transformCount;
    int32_t batchIndex;
    int32_t particleCount;
};

struct IFSIteratedParams {
    uint32_t transformCount;
    uint32_t generationOffset;
    uint32_t generationLimit;
};

struct IFSVoxelizeParams {
    uint32_t gridSize;
    uint32_t gridBounds;
    uint32_t transformCount;
    int32_t particleCount;
};

struct IFSClearParams {
    uint32_t memoryOffset;
};

struct IFSOcclusionParams {
    uint32_t gridSize;
    uint32_t memoryOffset;
};

struct IFSLODParams {
    int32_t transformCount;
};

// =============================================================================
// IFS System
// =============================================================================

class IFSSystem {
public:
    IFSSystem();
    ~IFSSystem();

    // Non-copyable
    IFSSystem(const IFSSystem&) = delete;
    IFSSystem& operator=(const IFSSystem&) = delete;

    /// Initialize the IFS system with GPU resources
    bool Initialize(const IFSConfig& config);

    /// Release all GPU resources
    void Shutdown();

    /// Check if system is initialized
    bool IsInitialized() const { return m_initialized; }

    // ==========================================================================
    // Simulation
    // ==========================================================================

    /// Run one frame of chaos game iteration + voxelization + bounds prediction
    void Update(float dt);

    /// Set a new preset (rebuilds transform set)
    void SetPreset(IFSPreset preset);

    /// Set a new target preset for blending
    void BlendToPreset(IFSPreset preset);

    /// Get current transform count
    uint32_t GetTransformCount() const { return m_transformCount; }

    /// Get particle count
    uint32_t GetParticleCount() const { return m_particleCount; }

    // ==========================================================================
    // Buffer Access (for rendering)
    // ==========================================================================

    /// Download particle positions to CPU buffer for rendering
    bool DownloadParticles(std::vector<float>& positions);

    /// Get voxel grid data for rendering
    bool DownloadVoxelGrid(std::vector<int32_t>& voxels);

    /// Get occlusion grid data
    bool DownloadOcclusionGrid(std::vector<float>& occlusion);

    /// Get GPU particle buffer handle
    VkBuffer GetParticleBuffer() const;

    /// Get voxel grid size
    uint32_t GetVoxelGridSize() const { return m_config.voxelGridSize; }

private:
    bool LoadShaders();
    bool CreateBuffers();
    void UploadTransforms();

    // Dispatch helpers
    void DispatchInitParticles();
    void DispatchChaosGame();
    void DispatchVoxelize();
    void DispatchOcclusion();
    void DispatchLODPrediction();

    bool m_initialized = false;
    IFSConfig m_config;

    // Particle state
    uint32_t m_particleCount = 0;
    uint32_t m_transformCount = 0;
    uint32_t m_frameCount = 0;

    // Transform management
    std::vector<Mat4> m_transforms;
    TransformBlender m_blender;
    bool m_isBlending = false;

    // Parallel reduction
    ParallelReduction m_reduction;

    // Compute pipelines
    ComputePipeline m_initParticlesPipeline;
    ComputePipeline m_chaosGamePipeline;
    ComputePipeline m_iteratedExpandPipeline;
    ComputePipeline m_clearVoxelsPipeline;
    ComputePipeline m_voxelizePipeline;
    ComputePipeline m_clearOcclusionPipeline;
    ComputePipeline m_calcOcclusionPipeline;
    ComputePipeline m_lodFirstPipeline;
    ComputePipeline m_lodIteratePipeline;

    // GPU buffers
    ComputeBuffer<float> m_particleBuffer;          // vec4 per particle (xyz + pad)
    ComputeBuffer<float> m_transformBuffer;          // mat4 per transform (16 floats)
    ComputeBuffer<float> m_finalTransformBuffer;     // Single mat4
    ComputeBuffer<int32_t> m_voxelGrid;              // gridSize^3
    ComputeBuffer<float> m_occlusionGrid;            // gridSize^3
    ComputeBuffer<float> m_boundsBuffer;             // min/max/sum (3 vec4s)

    // LOD prediction buffers
    ComputeBuffer<float> m_lodInput;
    ComputeBuffer<float> m_lodOutput;
};

} // namespace WulfNet
