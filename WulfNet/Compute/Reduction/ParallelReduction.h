// =============================================================================
// WulfNet Engine - GPU Parallel Reduction
// =============================================================================
// Reusable GPU parallel min/max/sum reduction for vec3 data.
// Uses specialization constants to select operation without shader recompilation.
// =============================================================================

#pragma once

#include "WulfNet/Compute/Pipelines/ComputePipeline.h"
#include "WulfNet/Compute/Memory/ComputeBuffer.h"

namespace WulfNet {

enum class ReductionOp : uint32_t {
    Min = 0,
    Max = 1,
    Add = 2
};

struct BoundsToTransformParams {
    float targetBoundsSize = 1.0f;
    float scalePadding = 0.9f;
    float particleCount = 1.0f;
};

class ParallelReduction {
public:
    ParallelReduction() = default;
    ~ParallelReduction() = default;

    bool Initialize();

    /// Single-operation reduction: reduces input vec4 buffer to a single vec3 result
    /// stored at outputBuffer[opSlot] where opSlot = 0(min), 1(max), 2(sum)
    void Reduce(ComputeBuffer<float>& input, ComputeBuffer<float>& output,
                uint32_t count, ReductionOp op);

    /// Compute bounding box (min + max) of positions
    /// output[0] = min, output[1] = max
    void ComputeBounds(ComputeBuffer<float>& positions,
                       ComputeBuffer<float>& boundsOutput,
                       uint32_t count);

    /// Compute bounding box + centroid (min + max + sum)
    /// output[0] = min, output[1] = max, output[2] = sum
    void ComputeBoundsAndCentroid(ComputeBuffer<float>& positions,
                                  ComputeBuffer<float>& boundsOutput,
                                  uint32_t count);

    /// Convert bounds data to a framing transformation matrix
    void BoundsToTransform(ComputeBuffer<float>& boundsData,
                           ComputeBuffer<float>& transformOutput,
                           const BoundsToTransformParams& params);

private:
    bool CreatePipeline(ReductionOp op, const std::vector<uint32_t>& spirv,
                        const std::string& name, ComputePipeline& pipeline,
                        bool hasPushConstants = false, uint32_t pushSize = 0);

    static constexpr uint32_t GROUP_SIZE = 128;

    // 3 pipelines per operation (min/max/add) for global pass
    ComputePipeline m_globalPipelines[3];
    // 3 pipelines for final pass
    ComputePipeline m_finalPipelines[3];
    // Bounds-to-transform pipeline
    ComputePipeline m_boundsToTransformPipeline;

    // Intermediate buffer for multi-pass reduction
    ComputeBuffer<float> m_tempBuffer;

    bool m_initialized = false;
};

} // namespace WulfNet
