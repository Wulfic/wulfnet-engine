// =============================================================================
// WulfNet Engine - Frame Allocator Tests
// =============================================================================

#include "TestHarness.h"
#include "WulfNet/Core/Memory/FrameAllocator.h"
#include <vector>
#include <cstring>

// =============================================================================
// Tests
// =============================================================================

static void Test_FrameAllocator_BasicAlloc() {
    auto& alloc = WulfNet::FrameAllocator::Get();
    alloc.Initialize(1024 * 1024);
    alloc.BeginFrame();

    float* buf = alloc.Alloc<float>(256);
    EXPECT_NE(buf, nullptr);

    // Write and read back
    for (int i = 0; i < 256; ++i) buf[i] = static_cast<float>(i);
    EXPECT_TRUE(buf[0] == 0.0f);
    EXPECT_TRUE(buf[255] == 255.0f);
}

static void Test_FrameAllocator_Alignment() {
    auto& alloc = WulfNet::FrameAllocator::Get();
    alloc.Initialize(1024 * 1024);
    alloc.BeginFrame();

    // Allocate 1 byte to misalign, then allocate with 16-byte alignment
    uint8_t* single = alloc.Alloc<uint8_t>(1);
    EXPECT_NE(single, nullptr);

    float* aligned = alloc.Alloc<float>(4);
    EXPECT_NE(aligned, nullptr);
    EXPECT_TRUE((reinterpret_cast<uintptr_t>(aligned) % alignof(float)) == 0);
}

static void Test_FrameAllocator_BeginFrameResets() {
    auto& alloc = WulfNet::FrameAllocator::Get();
    alloc.Initialize(1024);
    alloc.BeginFrame();

    alloc.Alloc<float>(64);
    size_t usageAfterAlloc = alloc.GetCurrentUsage();
    EXPECT_GT(usageAfterAlloc, static_cast<size_t>(0));

    alloc.BeginFrame();
    EXPECT_EQ(alloc.GetCurrentUsage(), static_cast<size_t>(0));
}

static void Test_FrameAllocator_MultipleAllocs() {
    auto& alloc = WulfNet::FrameAllocator::Get();
    alloc.Initialize(1024 * 1024);
    alloc.BeginFrame();

    // Multiple allocations in the same frame
    int* a = alloc.Alloc<int>(100);
    float* b = alloc.Alloc<float>(200);
    double* c = alloc.Alloc<double>(50);

    EXPECT_NE(a, nullptr);
    EXPECT_NE(b, nullptr);
    EXPECT_NE(c, nullptr);

    // They should not overlap
    uintptr_t aEnd = reinterpret_cast<uintptr_t>(a) + 100 * sizeof(int);
    uintptr_t bStart = reinterpret_cast<uintptr_t>(b);
    EXPECT_GE(bStart, aEnd);
}

static void Test_FrameAllocator_AllocZeroed() {
    auto& alloc = WulfNet::FrameAllocator::Get();
    alloc.Initialize(1024 * 1024);
    alloc.BeginFrame();

    int* buf = alloc.AllocZeroed<int>(128);
    EXPECT_NE(buf, nullptr);

    bool allZero = true;
    for (int i = 0; i < 128; ++i) {
        if (buf[i] != 0) { allZero = false; break; }
    }
    EXPECT_TRUE(allZero);
}

static void Test_FrameAllocator_PeakUsage() {
    auto& alloc = WulfNet::FrameAllocator::Get();
    alloc.Initialize(1024 * 1024);
    alloc.BeginFrame();

    alloc.Alloc<float>(1000);  // 4000 bytes
    size_t peakAfterFirst = alloc.GetPeakUsage();

    alloc.BeginFrame();  // Reset
    alloc.Alloc<float>(500);  // 2000 bytes

    // Peak should still be from the first frame
    EXPECT_GE(alloc.GetPeakUsage(), peakAfterFirst);
}

static void Test_FrameAllocator_OutOfSpace() {
    auto& alloc = WulfNet::FrameAllocator::Get();
    // Shutdown first to force a fresh allocation at the small size
    alloc.Shutdown();
    alloc.Initialize(256);  // Very small — 256 bytes
    alloc.BeginFrame();

    // This should succeed (128 bytes)
    float* ok = alloc.Alloc<float>(32);
    EXPECT_NE(ok, nullptr);

    // This should fail (more than remaining space)
    float* fail = alloc.Alloc<float>(1000);
    EXPECT_EQ(fail, nullptr);

    // Restore to normal size
    alloc.Initialize(1024 * 1024);
}

// =============================================================================
// Registration
// =============================================================================

void RegisterFrameAllocatorTests() {
    RUN_TEST("FrameAllocator_BasicAlloc", Test_FrameAllocator_BasicAlloc);
    RUN_TEST("FrameAllocator_Alignment", Test_FrameAllocator_Alignment);
    RUN_TEST("FrameAllocator_BeginFrameResets", Test_FrameAllocator_BeginFrameResets);
    RUN_TEST("FrameAllocator_MultipleAllocs", Test_FrameAllocator_MultipleAllocs);
    RUN_TEST("FrameAllocator_AllocZeroed", Test_FrameAllocator_AllocZeroed);
    RUN_TEST("FrameAllocator_PeakUsage", Test_FrameAllocator_PeakUsage);
    RUN_TEST("FrameAllocator_OutOfSpace", Test_FrameAllocator_OutOfSpace);
}
