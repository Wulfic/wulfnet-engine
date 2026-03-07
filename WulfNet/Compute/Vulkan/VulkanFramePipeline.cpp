// =============================================================================
// WulfNet Engine - Vulkan Frame-Pipelined Async Compute Implementation
// =============================================================================
// Frame-pipelined GPU resource management (double-buffered async dispatch)
// and command buffer pool for the VulkanContext class.
// Extracted from VulkanContext.cpp for maintainability.
// =============================================================================

#include "WulfNet/Compute/Vulkan/VulkanContext.h"
#include "WulfNet/Compute/Vulkan/VulkanLoader.h"
#include "WulfNet/Core/Logging/Logger.h"

#include <vector>

namespace WulfNet {

// =============================================================================
// Frame-Pipelined Async Compute (10.1)
// =============================================================================

bool VulkanContext::InitializeFrameResources() {
    if (m_framesInitialized) return true;
    if (!m_device || !m_computeCommandPool) return false;

    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        auto& frame = m_frames[i];

        // Allocate a command buffer from the compute pool
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_computeCommandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (g_vkFuncs.vkAllocateCommandBuffers(m_device, &allocInfo, &frame.cmdBuffer) != VK_SUCCESS) {
            WULFNET_ERROR("Compute", "Failed to allocate frame command buffer %d", i);
            DestroyFrameResources();
            return false;
        }

        // Create signaled fence (so first WaitForFrame doesn't block forever)
        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        if (g_vkFuncs.vkCreateFence(m_device, &fenceInfo, nullptr, &frame.fence) != VK_SUCCESS) {
            WULFNET_ERROR("Compute", "Failed to create frame fence %d", i);
            DestroyFrameResources();
            return false;
        }

        // Create semaphore for cross-queue synchronization
        VkSemaphoreCreateInfo semInfo = {};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        if (g_vkFuncs.vkCreateSemaphore(m_device, &semInfo, nullptr, &frame.semaphore) != VK_SUCCESS) {
            WULFNET_ERROR("Compute", "Failed to create frame semaphore %d", i);
            DestroyFrameResources();
            return false;
        }

        frame.inFlight = false;
        frame.recording = false;
    }

    m_framesInitialized = true;
    WULFNET_INFO("Compute", "Frame-pipelined resources initialized (%d frames in flight)", FRAMES_IN_FLIGHT);
    return true;
}

void VulkanContext::DestroyFrameResources() {
    if (!m_device) return;

    // Free command buffers from pool
    for (auto& cb : m_cmdBufferPool) {
        if (cb) {
            g_vkFuncs.vkFreeCommandBuffers(m_device, m_computeCommandPool, 1, &cb);
        }
    }
    m_cmdBufferPool.clear();

    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        auto& frame = m_frames[i];

        if (frame.semaphore && g_vkFuncs.vkDestroySemaphore) {
            g_vkFuncs.vkDestroySemaphore(m_device, frame.semaphore, nullptr);
            frame.semaphore = nullptr;
        }
        if (frame.fence && g_vkFuncs.vkDestroyFence) {
            g_vkFuncs.vkDestroyFence(m_device, frame.fence, nullptr);
            frame.fence = nullptr;
        }
        if (frame.cmdBuffer && m_computeCommandPool) {
            g_vkFuncs.vkFreeCommandBuffers(m_device, m_computeCommandPool, 1, &frame.cmdBuffer);
            frame.cmdBuffer = nullptr;
        }
        frame.inFlight = false;
        frame.recording = false;
    }
    m_framesInitialized = false;
}

bool VulkanContext::BeginFrame(int frameIndex) {
    if (frameIndex < 0 || frameIndex >= FRAMES_IN_FLIGHT) return false;
    if (!m_initialized) return false;

    // Lazy-init frame resources on first use
    if (!m_framesInitialized && !InitializeFrameResources()) {
        return false;
    }

    auto& frame = m_frames[frameIndex];

    // Wait for this frame's previous GPU work to complete
    if (frame.inFlight) {
        g_vkFuncs.vkWaitForFences(m_device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
        g_vkFuncs.vkResetFences(m_device, 1, &frame.fence);
        frame.inFlight = false;
    }

    // Reset and begin recording the command buffer
    g_vkFuncs.vkResetCommandBuffer(frame.cmdBuffer, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (g_vkFuncs.vkBeginCommandBuffer(frame.cmdBuffer, &beginInfo) != VK_SUCCESS) {
        return false;
    }

    frame.recording = true;
    return true;
}

bool VulkanContext::SubmitFrame(int frameIndex) {
    if (frameIndex < 0 || frameIndex >= FRAMES_IN_FLIGHT) return false;
    auto& frame = m_frames[frameIndex];
    if (!frame.recording) return false;

    // End recording
    g_vkFuncs.vkEndCommandBuffer(frame.cmdBuffer);
    frame.recording = false;

    // Submit without blocking — fence will signal when GPU is done
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &frame.cmdBuffer;

    if (g_vkFuncs.vkQueueSubmit(m_computeQueue, 1, &submitInfo, frame.fence) != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to submit frame %d command buffer", frameIndex);
        return false;
    }

    frame.inFlight = true;
    return true;
}

bool VulkanContext::WaitForFrame(int frameIndex) {
    if (frameIndex < 0 || frameIndex >= FRAMES_IN_FLIGHT) return false;
    if (!m_framesInitialized) return false;

    auto& frame = m_frames[frameIndex];
    if (!frame.inFlight) return true; // Already complete

    VkResult result = g_vkFuncs.vkWaitForFences(m_device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
    if (result == VK_SUCCESS) {
        g_vkFuncs.vkResetFences(m_device, 1, &frame.fence);
        frame.inFlight = false;
        return true;
    }
    return false;
}

bool VulkanContext::PollFrame(int frameIndex) const {
    if (frameIndex < 0 || frameIndex >= FRAMES_IN_FLIGHT) return false;
    if (!m_framesInitialized) return false;

    const auto& frame = m_frames[frameIndex];
    if (!frame.inFlight) return true; // Already complete

    VkResult result = g_vkFuncs.vkGetFenceStatus(m_device, frame.fence);
    return (result == VK_SUCCESS);
}

VkCommandBuffer VulkanContext::GetFrameCommandBuffer(int frameIndex) const {
    if (frameIndex < 0 || frameIndex >= FRAMES_IN_FLIGHT) return nullptr;
    return m_frames[frameIndex].cmdBuffer;
}

// =============================================================================
// Command Buffer Pool (10.1.4)
// =============================================================================

bool VulkanContext::GrowCommandBufferPool(int count) {
    if (!m_device || !m_computeCommandPool) return false;

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_computeCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(count);

    std::vector<VkCommandBuffer> newBuffers(count);
    if (g_vkFuncs.vkAllocateCommandBuffers(m_device, &allocInfo, newBuffers.data()) != VK_SUCCESS) {
        WULFNET_ERROR("Compute", "Failed to grow command buffer pool by %d", count);
        return false;
    }

    m_cmdBufferPool.insert(m_cmdBufferPool.end(), newBuffers.begin(), newBuffers.end());
    return true;
}

VkCommandBuffer VulkanContext::AcquireCommandBuffer() {
    if (m_cmdBufferPool.empty()) {
        if (!GrowCommandBufferPool(INITIAL_CMD_POOL_SIZE)) {
            return nullptr;
        }
    }

    VkCommandBuffer cb = m_cmdBufferPool.back();
    m_cmdBufferPool.pop_back();

    // Reset command buffer before reuse
    g_vkFuncs.vkResetCommandBuffer(cb, 0);
    return cb;
}

void VulkanContext::ReturnCommandBuffer(VkCommandBuffer cmdBuffer) {
    if (cmdBuffer) {
        m_cmdBufferPool.push_back(cmdBuffer);
    }
}

} // namespace WulfNet
