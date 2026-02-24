// =============================================================================
// WulfNet Engine - GPU Parallel Reduction Implementation
// =============================================================================

#include "WulfNet/Compute/Reduction/ParallelReduction.h"
#include "WulfNet/Core/Logging/Logger.h"
#include "WulfNet/Core/Profiling/Profiler.h"

namespace WulfNet {

bool ParallelReduction::Initialize() {
    WULFNET_ZONE();

    if (m_initialized) return true;

    // Load SPIR-V for reduction shaders
    auto globalSpirv = ShaderUtils::LoadSPIRV("Assets/Shaders/Compute/reduce_global.spv");
    auto finalSpirv = ShaderUtils::LoadSPIRV("Assets/Shaders/Compute/reduce_final.spv");
    auto boundsSpirv = ShaderUtils::LoadSPIRV("Assets/Shaders/Compute/reduce_to_transform.spv");

    if (globalSpirv.empty() || finalSpirv.empty() || boundsSpirv.empty()) {
        WULFNET_ERROR("Reduction", "Failed to load reduction shader SPIR-V files");
        return false;
    }

    // Create 3 global pipelines (one per operation: min/max/add)
    const char* opNames[] = {"Min", "Max", "Add"};
    for (uint32_t op = 0; op < 3; ++op) {
        ComputePipelineDesc desc;
        desc.spirvCode = globalSpirv;
        desc.bindings = {
            {0, ShaderBindingType::StorageBuffer, "InputBuffer"},
            {1, ShaderBindingType::StorageBuffer, "OutputBuffer"}
        };
        desc.specializationConstants = {{0, op}};
        desc.localSizeX = GROUP_SIZE;
        desc.name = std::string("reduce_global_") + opNames[op];

        if (!m_globalPipelines[op].Create(desc)) {
            WULFNET_ERROR("Reduction", "Failed to create global reduction pipeline for " +
                          std::string(opNames[op]));
            return false;
        }
    }

    // Create 3 final pipelines
    for (uint32_t op = 0; op < 3; ++op) {
        ComputePipelineDesc desc;
        desc.spirvCode = finalSpirv;
        desc.bindings = {
            {0, ShaderBindingType::StorageBuffer, "InputBuffer"},
            {1, ShaderBindingType::StorageBuffer, "OutputBuffer"}
        };
        desc.specializationConstants = {{0, op}};
        desc.pushConstants = {0, sizeof(uint32_t)};
        desc.localSizeX = GROUP_SIZE;
        desc.name = std::string("reduce_final_") + opNames[op];

        if (!m_finalPipelines[op].Create(desc)) {
            WULFNET_ERROR("Reduction", "Failed to create final reduction pipeline for " +
                          std::string(opNames[op]));
            return false;
        }
    }

    // Create bounds-to-transform pipeline
    {
        ComputePipelineDesc desc;
        desc.spirvCode = boundsSpirv;
        desc.bindings = {
            {0, ShaderBindingType::StorageBuffer, "BoundsBuffer"},
            {1, ShaderBindingType::StorageBuffer, "TransformBuffer"}
        };
        desc.pushConstants = {0, sizeof(BoundsToTransformParams)};
        desc.localSizeX = 1;
        desc.name = "reduce_to_transform";

        if (!m_boundsToTransformPipeline.Create(desc)) {
            WULFNET_ERROR("Reduction", "Failed to create bounds-to-transform pipeline");
            return false;
        }
    }

    m_initialized = true;
    WULFNET_INFO("Reduction", "Parallel reduction system initialized");
    return true;
}

void ParallelReduction::Reduce(ComputeBuffer<float>& input, ComputeBuffer<float>& output,
                                uint32_t count, ReductionOp op) {
    WULFNET_ZONE();

    if (!m_initialized) return;

    uint32_t opIdx = static_cast<uint32_t>(op);

    // Calculate workgroup count for the global pass
    uint32_t groupCount = (count + GROUP_SIZE - 1) / GROUP_SIZE;

    if (groupCount <= 1) {
        // Small enough for a single final pass
        m_finalPipelines[opIdx].BindBuffer(0, input);
        m_finalPipelines[opIdx].BindBuffer(1, output);
        m_finalPipelines[opIdx].SetPushConstants(count);
        m_finalPipelines[opIdx].DispatchAndWait(1);
        return;
    }

    // Ensure temp buffer is large enough (4 floats per vec4, one per workgroup)
    uint32_t tempSize = groupCount * 4;
    if (m_tempBuffer.GetCount() < tempSize) {
        m_tempBuffer.Allocate(tempSize, GPUBufferUsage::ComputeStorage);
    }

    // Global pass: reduce each workgroup to a single value
    m_globalPipelines[opIdx].BindBuffer(0, input);
    m_globalPipelines[opIdx].BindBuffer(1, m_tempBuffer);
    m_globalPipelines[opIdx].DispatchAndWait(groupCount);

    // Final pass: reduce the workgroup results to a single value
    m_finalPipelines[opIdx].BindBuffer(0, m_tempBuffer);
    m_finalPipelines[opIdx].BindBuffer(1, output);
    m_finalPipelines[opIdx].SetPushConstants(groupCount);
    m_finalPipelines[opIdx].DispatchAndWait(1);
}

void ParallelReduction::ComputeBounds(ComputeBuffer<float>& positions,
                                       ComputeBuffer<float>& boundsOutput,
                                       uint32_t count) {
    WULFNET_ZONE();
    // Run min and max reductions into the same output buffer
    // output[0] = min, output[1] = max
    Reduce(positions, boundsOutput, count, ReductionOp::Min);
    Reduce(positions, boundsOutput, count, ReductionOp::Max);
}

void ParallelReduction::ComputeBoundsAndCentroid(ComputeBuffer<float>& positions,
                                                  ComputeBuffer<float>& boundsOutput,
                                                  uint32_t count) {
    WULFNET_ZONE();
    Reduce(positions, boundsOutput, count, ReductionOp::Min);
    Reduce(positions, boundsOutput, count, ReductionOp::Max);
    Reduce(positions, boundsOutput, count, ReductionOp::Add);
}

void ParallelReduction::BoundsToTransform(ComputeBuffer<float>& boundsData,
                                           ComputeBuffer<float>& transformOutput,
                                           const BoundsToTransformParams& params) {
    WULFNET_ZONE();
    m_boundsToTransformPipeline.BindBuffer(0, boundsData);
    m_boundsToTransformPipeline.BindBuffer(1, transformOutput);
    m_boundsToTransformPipeline.SetPushConstants(params);
    m_boundsToTransformPipeline.DispatchAndWait(1);
}

} // namespace WulfNet
