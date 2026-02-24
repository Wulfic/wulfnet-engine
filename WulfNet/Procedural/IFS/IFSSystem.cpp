// =============================================================================
// WulfNet Engine - IFS System Implementation
// =============================================================================

#include "WulfNet/Procedural/IFS/IFSSystem.h"
#include "WulfNet/Core/Logging/Logger.h"
#include "WulfNet/Core/Profiling/Profiler.h"
#include <cstring>

namespace WulfNet {

IFSSystem::IFSSystem() = default;

IFSSystem::~IFSSystem() {
    Shutdown();
}

bool IFSSystem::Initialize(const IFSConfig& config) {
    WULFNET_ZONE();

    if (m_initialized) {
        WULFNET_WARNING("IFS", "Already initialized");
        return true;
    }

    m_config = config;
    m_particleCount = config.cubeResolution * config.cubeResolution * config.cubeResolution;

    // Initialize parallel reduction
    if (!m_reduction.Initialize()) {
        WULFNET_ERROR("IFS", "Failed to initialize parallel reduction");
        return false;
    }

    // Load all compute shaders
    if (!LoadShaders()) {
        WULFNET_ERROR("IFS", "Failed to load IFS compute shaders");
        return false;
    }

    // Create GPU buffers
    if (!CreateBuffers()) {
        WULFNET_ERROR("IFS", "Failed to create IFS GPU buffers");
        return false;
    }

    // Set initial preset
    SetPreset(config.initialPreset);

    // Initialize particles in grid
    DispatchInitParticles();

    m_initialized = true;
    WULFNET_INFO("IFS", "IFS system initialized with " + std::to_string(m_particleCount) +
                 " particles, preset: " + std::to_string(static_cast<int>(config.initialPreset)));
    return true;
}

void IFSSystem::Shutdown() {
    m_initialized = false;
    m_particleBuffer = {};
    m_transformBuffer = {};
    m_finalTransformBuffer = {};
    m_voxelGrid = {};
    m_occlusionGrid = {};
    m_boundsBuffer = {};
    m_lodInput = {};
    m_lodOutput = {};
}

bool IFSSystem::LoadShaders() {
    WULFNET_ZONE();

    const std::string& path = m_config.shaderPath;

    auto loadPipeline = [&](ComputePipeline& pipeline, const std::string& name,
                            const std::vector<ShaderBinding>& bindings,
                            uint32_t pushSize, uint32_t localSize) -> bool {
        auto spirv = ShaderUtils::LoadSPIRV(path + "/" + name + ".spv");
        if (spirv.empty()) {
            WULFNET_ERROR("IFS", "Failed to load shader: " + name);
            return false;
        }

        ComputePipelineDesc desc;
        desc.spirvCode = std::move(spirv);
        desc.bindings = bindings;
        if (pushSize > 0) desc.pushConstants = {0, pushSize};
        desc.localSizeX = localSize;
        desc.name = name;

        return pipeline.Create(desc);
    };

    // Init particles
    if (!loadPipeline(m_initParticlesPipeline, "ifs_init_particles",
        {{0, ShaderBindingType::StorageBuffer, "ParticleBuffer"}},
        sizeof(IFSInitParams), 64)) return false;

    // Chaos game
    if (!loadPipeline(m_chaosGamePipeline, "ifs_chaos_game",
        {{0, ShaderBindingType::StorageBuffer, "ParticleBuffer"},
         {1, ShaderBindingType::StorageBuffer, "TransformBuffer"}},
        sizeof(IFSChaosParams), 64)) return false;

    // Iterated expand
    if (!loadPipeline(m_iteratedExpandPipeline, "ifs_iterated_expand",
        {{0, ShaderBindingType::StorageBuffer, "ParticleBuffer"},
         {1, ShaderBindingType::StorageBuffer, "TransformBuffer"}},
        sizeof(IFSIteratedParams), 128)) return false;

    // Clear voxels
    if (!loadPipeline(m_clearVoxelsPipeline, "ifs_clear_voxels",
        {{0, ShaderBindingType::StorageBuffer, "VoxelGrid"}},
        sizeof(IFSClearParams), 64)) return false;

    // Voxelize
    if (!loadPipeline(m_voxelizePipeline, "ifs_voxelize",
        {{0, ShaderBindingType::StorageBuffer, "ParticleBuffer"},
         {1, ShaderBindingType::StorageBuffer, "TransformBuffer"},
         {2, ShaderBindingType::StorageBuffer, "FinalTransformBuffer"},
         {3, ShaderBindingType::StorageBuffer, "VoxelGrid"}},
        sizeof(IFSVoxelizeParams), 64)) return false;

    // Clear occlusion
    if (!loadPipeline(m_clearOcclusionPipeline, "ifs_clear_occlusion",
        {{0, ShaderBindingType::StorageBuffer, "OcclusionGrid"}},
        sizeof(IFSClearParams), 64)) return false;

    // Calculate occlusion
    if (!loadPipeline(m_calcOcclusionPipeline, "ifs_calc_occlusion",
        {{0, ShaderBindingType::StorageBuffer, "VoxelGrid"},
         {1, ShaderBindingType::StorageBuffer, "OcclusionGrid"}},
        sizeof(IFSOcclusionParams), 64)) return false;

    // LOD first
    if (!loadPipeline(m_lodFirstPipeline, "ifs_lod_first",
        {{0, ShaderBindingType::StorageBuffer, "OutputBuffer"},
         {1, ShaderBindingType::StorageBuffer, "TransformBuffer"}},
        sizeof(IFSLODParams), 1)) return false;

    // LOD iterate
    if (!loadPipeline(m_lodIteratePipeline, "ifs_lod_iterate",
        {{0, ShaderBindingType::StorageBuffer, "InputBuffer"},
         {1, ShaderBindingType::StorageBuffer, "OutputBuffer"},
         {2, ShaderBindingType::StorageBuffer, "TransformBuffer"}},
        sizeof(IFSLODParams), 64)) return false;

    WULFNET_DEBUG("IFS", "All IFS compute shaders loaded successfully");
    return true;
}

bool IFSSystem::CreateBuffers() {
    WULFNET_ZONE();

    uint32_t gridVolume = m_config.voxelGridSize * m_config.voxelGridSize * m_config.voxelGridSize;

    // Particle buffer: 4 floats per particle (vec4)
    if (!m_particleBuffer.Allocate(m_particleCount * 4, GPUBufferUsage::ComputeStorage))
        return false;

    // Transform buffer: 16 floats per mat4, max 32 transforms
    if (!m_transformBuffer.Allocate(32 * 16, GPUBufferUsage::ComputeStorage))
        return false;

    // Final transform: single mat4 (16 floats)
    if (!m_finalTransformBuffer.Allocate(16, GPUBufferUsage::ComputeStorage))
        return false;

    // Voxel grid
    if (!m_voxelGrid.Allocate(gridVolume, GPUBufferUsage::ComputeStorage))
        return false;

    // Occlusion grid
    if (!m_occlusionGrid.Allocate(gridVolume, GPUBufferUsage::ComputeStorage))
        return false;

    // Bounds buffer: 3 vec4s (min, max, sum)
    if (!m_boundsBuffer.Allocate(12, GPUBufferUsage::ComputeStorage))
        return false;

    // LOD buffers: size grows geometrically with transform count
    // Allocate for worst case (32 transforms, 4 iterations = 32^4 = ~1M)
    uint32_t maxLodSize = 32 * 32 * 32 * 32 * 4;  // * 4 for vec4
    if (!m_lodInput.Allocate(maxLodSize, GPUBufferUsage::ComputeStorage))
        return false;
    if (!m_lodOutput.Allocate(maxLodSize, GPUBufferUsage::ComputeStorage))
        return false;

    WULFNET_DEBUG("IFS", "IFS GPU buffers created");
    return true;
}

void IFSSystem::UploadTransforms() {
    WULFNET_ZONE();

    // Upload transform matrices to GPU
    std::vector<float> flatData(m_transforms.size() * 16);
    for (size_t i = 0; i < m_transforms.size(); ++i) {
        std::memcpy(&flatData[i * 16], m_transforms[i].m, 16 * sizeof(float));
    }
    m_transformBuffer.Upload(flatData.data(), flatData.size());
}

void IFSSystem::SetPreset(IFSPreset preset) {
    auto instructions = TransformPresets::GetPreset(preset);
    m_transforms = TransformPresets::BuildMatrices(instructions);
    m_transformCount = static_cast<uint32_t>(m_transforms.size());
    m_isBlending = false;

    if (m_initialized) {
        UploadTransforms();
    }
}

void IFSSystem::BlendToPreset(IFSPreset preset) {
    auto targetInstructions = TransformPresets::GetPreset(preset);

    if (!m_isBlending) {
        // Initialize blender with current and target
        auto currentInstructions = TransformPresets::GetPreset(m_config.initialPreset);
        m_blender.SetSets(currentInstructions, targetInstructions);
        m_isBlending = true;
    } else {
        m_blender.SwitchTarget(targetInstructions);
    }
}

void IFSSystem::Update(float dt) {
    WULFNET_ZONE();

    if (!m_initialized) return;

    // Update blend if active
    if (m_isBlending) {
        m_blender.Update(dt, 3.0f);
        m_transforms = m_blender.GetBlendedMatrices();
        m_transformCount = static_cast<uint32_t>(m_transforms.size());
        UploadTransforms();
    }

    // Run chaos game iterations
    DispatchChaosGame();

    // Compute bounds and framing transform
    DispatchLODPrediction();

    // Voxelize
    DispatchVoxelize();

    // Compute ambient occlusion
    DispatchOcclusion();

    m_frameCount++;
}

void IFSSystem::DispatchInitParticles() {
    WULFNET_ZONE();

    IFSInitParams params;
    params.cubeResolution = m_config.cubeResolution;
    params.cubeSize = m_config.cubeSize;

    m_initParticlesPipeline.BindBuffer(0, m_particleBuffer);
    m_initParticlesPipeline.SetPushConstants(params);
    m_initParticlesPipeline.DispatchAndWait(
        m_initParticlesPipeline.CalculateGroupCount(m_particleCount));
}

void IFSSystem::DispatchChaosGame() {
    WULFNET_ZONE();

    m_chaosGamePipeline.BindBuffer(0, m_particleBuffer);
    m_chaosGamePipeline.BindBuffer(1, m_transformBuffer);

    for (uint32_t i = 0; i < m_config.chaosIterationsPerFrame; ++i) {
        IFSChaosParams params;
        params.seed = m_frameCount * 1000 + i * 137;
        params.transformCount = m_transformCount;
        params.batchIndex = static_cast<int32_t>(i);
        params.particleCount = static_cast<int32_t>(m_particleCount);

        m_chaosGamePipeline.SetPushConstants(params);
        m_chaosGamePipeline.DispatchAndWait(
            m_chaosGamePipeline.CalculateGroupCount(m_particleCount));
    }
}

void IFSSystem::DispatchLODPrediction() {
    WULFNET_ZONE();

    // Use parallel reduction to compute bounds from particles
    m_reduction.ComputeBoundsAndCentroid(m_particleBuffer, m_boundsBuffer, m_particleCount);

    // Convert bounds to framing transform
    BoundsToTransformParams btParams;
    btParams.targetBoundsSize = m_config.targetBoundsSize;
    btParams.scalePadding = m_config.scalePadding;
    btParams.particleCount = static_cast<float>(m_particleCount);

    m_reduction.BoundsToTransform(m_boundsBuffer, m_finalTransformBuffer, btParams);
}

void IFSSystem::DispatchVoxelize() {
    WULFNET_ZONE();

    uint32_t gridVolume = m_config.voxelGridSize * m_config.voxelGridSize * m_config.voxelGridSize;

    // Clear voxel grid
    IFSClearParams clearParams;
    clearParams.memoryOffset = 0;
    m_clearVoxelsPipeline.BindBuffer(0, m_voxelGrid);
    m_clearVoxelsPipeline.SetPushConstants(clearParams);
    m_clearVoxelsPipeline.DispatchAndWait(
        m_clearVoxelsPipeline.CalculateGroupCount(gridVolume));

    // Voxelize particle positions
    IFSVoxelizeParams voxParams;
    voxParams.gridSize = m_config.voxelGridSize;
    voxParams.gridBounds = m_config.voxelGridBounds;
    voxParams.transformCount = m_transformCount;
    voxParams.particleCount = static_cast<int32_t>(m_particleCount);

    m_voxelizePipeline.BindBuffer(0, m_particleBuffer);
    m_voxelizePipeline.BindBuffer(1, m_transformBuffer);
    m_voxelizePipeline.BindBuffer(2, m_finalTransformBuffer);
    m_voxelizePipeline.BindBuffer(3, m_voxelGrid);
    m_voxelizePipeline.SetPushConstants(voxParams);
    m_voxelizePipeline.DispatchAndWait(
        m_voxelizePipeline.CalculateGroupCount(m_particleCount));
}

void IFSSystem::DispatchOcclusion() {
    WULFNET_ZONE();

    uint32_t gridVolume = m_config.voxelGridSize * m_config.voxelGridSize * m_config.voxelGridSize;

    // Clear occlusion grid
    IFSClearParams clearParams;
    clearParams.memoryOffset = 0;
    m_clearOcclusionPipeline.BindBuffer(0, m_occlusionGrid);
    m_clearOcclusionPipeline.SetPushConstants(clearParams);
    m_clearOcclusionPipeline.DispatchAndWait(
        m_clearOcclusionPipeline.CalculateGroupCount(gridVolume));

    // Calculate ambient occlusion
    IFSOcclusionParams occParams;
    occParams.gridSize = m_config.voxelGridSize;
    occParams.memoryOffset = 0;

    m_calcOcclusionPipeline.BindBuffer(0, m_voxelGrid);
    m_calcOcclusionPipeline.BindBuffer(1, m_occlusionGrid);
    m_calcOcclusionPipeline.SetPushConstants(occParams);
    m_calcOcclusionPipeline.DispatchAndWait(
        m_calcOcclusionPipeline.CalculateGroupCount(gridVolume));
}

bool IFSSystem::DownloadParticles(std::vector<float>& positions) {
    WULFNET_ZONE();
    positions.resize(m_particleCount * 4);
    return m_particleBuffer.Download(positions.data(), positions.size());
}

bool IFSSystem::DownloadVoxelGrid(std::vector<int32_t>& voxels) {
    uint32_t vol = m_config.voxelGridSize * m_config.voxelGridSize * m_config.voxelGridSize;
    voxels.resize(vol);
    return m_voxelGrid.Download(voxels.data(), vol);
}

bool IFSSystem::DownloadOcclusionGrid(std::vector<float>& occlusion) {
    uint32_t vol = m_config.voxelGridSize * m_config.voxelGridSize * m_config.voxelGridSize;
    occlusion.resize(vol);
    return m_occlusionGrid.Download(occlusion.data(), vol);
}

VkBuffer IFSSystem::GetParticleBuffer() const {
    return m_particleBuffer.GetVkBuffer();
}

} // namespace WulfNet
