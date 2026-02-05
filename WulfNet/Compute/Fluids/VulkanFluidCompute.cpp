// =============================================================================
// WulfNet Engine - Vulkan Fluid Compute Integration
// =============================================================================
// GPU compute shader integration for CO-FLIP fluid simulation.
// =============================================================================

#include "VulkanFluidCompute.h"
#include <fstream>
#include <filesystem>
#include <iostream>  // For fallback logging

namespace WulfNet {

// Simple logging fallback (Logger macros require category)
#define FLUID_LOG_INFO(msg)  std::cout << "[FluidCompute] " << msg << std::endl
#define FLUID_LOG_WARN(msg)  std::cout << "[FluidCompute WARN] " << msg << std::endl
#define FLUID_LOG_ERROR(msg) std::cerr << "[FluidCompute ERROR] " << msg << std::endl

// =============================================================================
// Constructor / Destructor
// =============================================================================

VulkanFluidCompute::VulkanFluidCompute() = default;

VulkanFluidCompute::~VulkanFluidCompute() {
    Shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool VulkanFluidCompute::Initialize(VulkanContext* vulkan, const COFLIPConfig& config,
                                     const std::string& shaderPath) {
    if (m_initialized) {
        FLUID_LOG_WARN("VulkanFluidCompute already initialized");
        return true;
    }

    if (!vulkan || !vulkan->IsValid()) {
        FLUID_LOG_ERROR("VulkanFluidCompute: Invalid Vulkan context");
        return false;
    }

    m_vulkan = vulkan;
    m_gridSizeX = config.gridSizeX;
    m_gridSizeY = config.gridSizeY;
    m_gridSizeZ = config.gridSizeZ;
    m_gridTotalCells = m_gridSizeX * m_gridSizeY * m_gridSizeZ;

    // Estimate max particles (8 per cell is typical)
    m_maxParticles = m_gridTotalCells * config.particlesPerCell;

    FLUID_LOG_INFO("VulkanFluidCompute: Initializing GPU compute");
    FLUID_LOG_INFO("  Grid: {}x{}x{} = {} cells", m_gridSizeX, m_gridSizeY, m_gridSizeZ, m_gridTotalCells);
    FLUID_LOG_INFO("  Max particles: {}", m_maxParticles);

    // Create GPU buffers
    if (!CreateBuffers(config)) {
        FLUID_LOG_ERROR("VulkanFluidCompute: Failed to create GPU buffers");
        return false;
    }

    // Load and create compute pipelines
    if (!LoadShaders(shaderPath)) {
        FLUID_LOG_ERROR("VulkanFluidCompute: Failed to load shaders");
        return false;
    }

    m_initialized = true;
    FLUID_LOG_INFO("VulkanFluidCompute: GPU compute ready");
    return true;
}

void VulkanFluidCompute::Shutdown() {
    if (!m_initialized) return;

    FLUID_LOG_INFO("VulkanFluidCompute: Shutting down");

    // Wait for any pending GPU work
    if (m_vulkan && m_vulkan->IsValid()) {
        m_vulkan->WaitIdle();
    }

    // Destroy pipelines
    m_p2gPipeline.reset();
    m_normalizePipeline.reset();
    m_forcesPipeline.reset();
    m_divergencePipeline.reset();
    m_pressurePipeline.reset();
    m_gradientPipeline.reset();
    m_g2pPipeline.reset();

    // Release buffers
    m_particleBuffer.reset();
    m_gridBuffer.reset();
    m_prevVelocityBuffer.reset();

    m_vulkan = nullptr;
    m_initialized = false;
}

// =============================================================================
// Buffer Creation
// =============================================================================

bool VulkanFluidCompute::CreateBuffers(const COFLIPConfig& config) {
    try {
        // Particle buffer (device local, transfer enabled)
        m_particleBuffer = std::make_unique<ComputeBuffer<COFLIPParticle>>(
            m_maxParticles,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        // Grid buffer
        m_gridBuffer = std::make_unique<ComputeBuffer<COFLIPCell>>(
            m_gridTotalCells,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        // Previous velocity buffer (3 floats per cell face = 3 * gridSize)
        // Actually we need previous U, V, W stored separately for FLIP
        size_t totalFaces = static_cast<size_t>(m_gridSizeX + 1) * m_gridSizeY * m_gridSizeZ +
                            m_gridSizeX * static_cast<size_t>(m_gridSizeY + 1) * m_gridSizeZ +
                            m_gridSizeX * m_gridSizeY * static_cast<size_t>(m_gridSizeZ + 1);
        m_prevVelocityBuffer = std::make_unique<ComputeBuffer<float>>(
            totalFaces,
            GPUBufferUsage::ComputeStorage,
            GPUMemoryLocation::DeviceLocal
        );

        FLUID_LOG_INFO("  Particle buffer: {} MB", m_particleBuffer->GetSizeBytes() / (1024 * 1024));
        FLUID_LOG_INFO("  Grid buffer: {} MB", m_gridBuffer->GetSizeBytes() / (1024 * 1024));
        FLUID_LOG_INFO("  Prev velocity buffer: {} MB", m_prevVelocityBuffer->GetSizeBytes() / (1024 * 1024));

        return m_particleBuffer->IsValid() && m_gridBuffer->IsValid() && m_prevVelocityBuffer->IsValid();
    }
    catch (const std::exception& e) {
        FLUID_LOG_ERROR("  Buffer creation failed: {}", e.what());
        return false;
    }
}

// =============================================================================
// Shader Loading
// =============================================================================

bool VulkanFluidCompute::LoadShaders(const std::string& shaderPath) {
    // Define shader bindings (matches our GLSL compute shaders)
    // Binding 0: Particles (storage buffer)
    // Binding 1: Grid (storage buffer)
    // Binding 2: Previous velocities (storage buffer)
    std::vector<ShaderBinding> fluidBindings = {
        {0, ShaderBindingType::StorageBuffer, "particles"},
        {1, ShaderBindingType::StorageBuffer, "grid"},
        {2, ShaderBindingType::StorageBuffer, "prevVelocity"}
    };

    // Push constant range for simulation parameters
    PushConstantRange pushRange{0, sizeof(FluidSimParams)};

    auto loadPipeline = [&](const std::string& name, uint32_t localX = 256, uint32_t localY = 1, uint32_t localZ = 1)
        -> std::unique_ptr<ComputePipeline>
    {
        std::string path = shaderPath + "/" + name + ".spv";

        // Check if file exists
        if (!std::filesystem::exists(path)) {
            FLUID_LOG_ERROR("  Shader not found: {}", path);
            return nullptr;
        }

        auto pipeline = std::make_unique<ComputePipeline>();
        if (!pipeline->CreateFromFile(path, fluidBindings, pushRange)) {
            FLUID_LOG_ERROR("  Failed to create pipeline: {}", name);
            return nullptr;
        }

        FLUID_LOG_INFO("  Loaded shader: {}", name);
        return pipeline;
    };

    // Load all CO-FLIP shaders
    m_p2gPipeline = loadPipeline("coflip_p2g", 256, 1, 1);
    m_normalizePipeline = loadPipeline("coflip_normalize", 8, 8, 8);
    m_forcesPipeline = loadPipeline("coflip_forces", 8, 8, 8);
    m_divergencePipeline = loadPipeline("coflip_divergence", 8, 8, 8);
    m_pressurePipeline = loadPipeline("coflip_pressure", 8, 8, 8);
    m_gradientPipeline = loadPipeline("coflip_gradient", 8, 8, 8);
    m_g2pPipeline = loadPipeline("coflip_g2p", 256, 1, 1);

    // Check all pipelines loaded successfully
    bool allLoaded = m_p2gPipeline && m_normalizePipeline && m_forcesPipeline &&
                     m_divergencePipeline && m_pressurePipeline && m_gradientPipeline &&
                     m_g2pPipeline;

    if (!allLoaded) {
        FLUID_LOG_WARN("  Some shaders failed to load - GPU compute will be disabled");
        return false;
    }

    return true;
}

// =============================================================================
// Buffer Upload/Download
// =============================================================================

bool VulkanFluidCompute::UploadParticles(const std::vector<COFLIPParticle>& particles, uint32_t count) {
    if (!m_initialized || !m_particleBuffer) return false;
    if (count == 0) return true;
    if (count > m_maxParticles) {
        FLUID_LOG_WARN("Particle count {} exceeds max {}, clamping", count, m_maxParticles);
        count = m_maxParticles;
    }
    return m_particleBuffer->Upload(particles.data(), count, 0);
}

bool VulkanFluidCompute::DownloadParticles(std::vector<COFLIPParticle>& particles, uint32_t count) {
    if (!m_initialized || !m_particleBuffer) return false;
    if (count == 0) return true;
    if (particles.size() < count) {
        particles.resize(count);
    }
    return m_particleBuffer->Download(particles.data(), count, 0);
}

bool VulkanFluidCompute::UploadGrid(const std::vector<COFLIPCell>& grid) {
    if (!m_initialized || !m_gridBuffer) return false;
    if (grid.size() != m_gridTotalCells) {
        FLUID_LOG_ERROR("Grid size mismatch: {} vs {}", grid.size(), m_gridTotalCells);
        return false;
    }
    return m_gridBuffer->Upload(grid.data(), m_gridTotalCells, 0);
}

bool VulkanFluidCompute::DownloadGrid(std::vector<COFLIPCell>& grid) {
    if (!m_initialized || !m_gridBuffer) return false;
    if (grid.size() != m_gridTotalCells) {
        grid.resize(m_gridTotalCells);
    }
    return m_gridBuffer->Download(grid.data(), m_gridTotalCells, 0);
}

// =============================================================================
// Compute Dispatches
// =============================================================================

void VulkanFluidCompute::InsertMemoryBarrier() {
    // Memory barrier is handled internally by Vulkan command buffer submission
    // When using DispatchAndWait, barriers are implicit
    // For async dispatch, we'd need to insert VK_ACCESS_SHADER_WRITE_BIT -> VK_ACCESS_SHADER_READ_BIT
}

void VulkanFluidCompute::DispatchP2G(const FluidSimParams& params) {
    if (!m_p2gPipeline || !m_p2gPipeline->IsValid()) return;

    // Bind buffers
    m_p2gPipeline->BindBuffer(0, *m_particleBuffer);
    m_p2gPipeline->BindBuffer(1, *m_gridBuffer);
    m_p2gPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_p2gPipeline->UpdateBindings();

    // Set push constants
    m_p2gPipeline->SetPushConstants(params);

    // Dispatch one thread per particle
    uint32_t groupCount = m_p2gPipeline->CalculateGroupCount(params.particleCount);
    m_p2gPipeline->DispatchAndWait(groupCount, 1, 1);
}

void VulkanFluidCompute::DispatchNormalize(const FluidSimParams& params) {
    if (!m_normalizePipeline || !m_normalizePipeline->IsValid()) return;

    m_normalizePipeline->BindBuffer(0, *m_particleBuffer);
    m_normalizePipeline->BindBuffer(1, *m_gridBuffer);
    m_normalizePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_normalizePipeline->UpdateBindings();

    m_normalizePipeline->SetPushConstants(params);

    // Dispatch over grid cells (8x8x8 workgroups)
    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    m_normalizePipeline->DispatchAndWait(gx, gy, gz);
}

void VulkanFluidCompute::DispatchForces(const FluidSimParams& params) {
    if (!m_forcesPipeline || !m_forcesPipeline->IsValid()) return;

    m_forcesPipeline->BindBuffer(0, *m_particleBuffer);
    m_forcesPipeline->BindBuffer(1, *m_gridBuffer);
    m_forcesPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_forcesPipeline->UpdateBindings();

    m_forcesPipeline->SetPushConstants(params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    m_forcesPipeline->DispatchAndWait(gx, gy, gz);
}

void VulkanFluidCompute::DispatchDivergence(const FluidSimParams& params) {
    if (!m_divergencePipeline || !m_divergencePipeline->IsValid()) return;

    m_divergencePipeline->BindBuffer(0, *m_particleBuffer);
    m_divergencePipeline->BindBuffer(1, *m_gridBuffer);
    m_divergencePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_divergencePipeline->UpdateBindings();

    m_divergencePipeline->SetPushConstants(params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    m_divergencePipeline->DispatchAndWait(gx, gy, gz);
}

void VulkanFluidCompute::DispatchPressure(const FluidSimParams& params, uint32_t iterations) {
    if (!m_pressurePipeline || !m_pressurePipeline->IsValid()) return;

    m_pressurePipeline->BindBuffer(0, *m_particleBuffer);
    m_pressurePipeline->BindBuffer(1, *m_gridBuffer);
    m_pressurePipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_pressurePipeline->UpdateBindings();

    m_pressurePipeline->SetPushConstants(params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;

    // Run multiple Jacobi iterations
    for (uint32_t i = 0; i < iterations; ++i) {
        m_pressurePipeline->DispatchAndWait(gx, gy, gz);
    }
}

void VulkanFluidCompute::DispatchGradient(const FluidSimParams& params) {
    if (!m_gradientPipeline || !m_gradientPipeline->IsValid()) return;

    m_gradientPipeline->BindBuffer(0, *m_particleBuffer);
    m_gradientPipeline->BindBuffer(1, *m_gridBuffer);
    m_gradientPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_gradientPipeline->UpdateBindings();

    m_gradientPipeline->SetPushConstants(params);

    uint32_t gx = (m_gridSizeX + 7) / 8;
    uint32_t gy = (m_gridSizeY + 7) / 8;
    uint32_t gz = (m_gridSizeZ + 7) / 8;
    m_gradientPipeline->DispatchAndWait(gx, gy, gz);
}

void VulkanFluidCompute::DispatchG2P(const FluidSimParams& params) {
    if (!m_g2pPipeline || !m_g2pPipeline->IsValid()) return;

    m_g2pPipeline->BindBuffer(0, *m_particleBuffer);
    m_g2pPipeline->BindBuffer(1, *m_gridBuffer);
    m_g2pPipeline->BindBuffer(2, *m_prevVelocityBuffer);
    m_g2pPipeline->UpdateBindings();

    m_g2pPipeline->SetPushConstants(params);

    uint32_t groupCount = m_g2pPipeline->CalculateGroupCount(params.particleCount);
    m_g2pPipeline->DispatchAndWait(groupCount, 1, 1);
}

void VulkanFluidCompute::DispatchFullStep(const FluidSimParams& params) {
    // Full CO-FLIP simulation step on GPU
    // 1. Particle to Grid (P2G)
    DispatchP2G(params);

    // 2. Normalize grid velocities
    DispatchNormalize(params);

    // 3. Apply external forces
    DispatchForces(params);

    // 4. Compute divergence
    DispatchDivergence(params);

    // 5. Pressure solve (multiple iterations)
    DispatchPressure(params, params.pressureIterations);

    // 6. Apply pressure gradient
    DispatchGradient(params);

    // 7. Grid to Particle (G2P) with FLIP/PIC blend
    DispatchG2P(params);
}

// =============================================================================
// Buffer Access
// =============================================================================

VkBuffer VulkanFluidCompute::GetParticleBuffer() const {
    return m_particleBuffer ? m_particleBuffer->GetVkBuffer() : nullptr;
}

size_t VulkanFluidCompute::GetParticleBufferSize() const {
    return m_particleBuffer ? m_particleBuffer->GetSizeBytes() : 0;
}

VkBuffer VulkanFluidCompute::GetGridBuffer() const {
    return m_gridBuffer ? m_gridBuffer->GetVkBuffer() : nullptr;
}

} // namespace WulfNet
