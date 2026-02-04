// =============================================================================
// WulfNet Engine - SIMD Intrinsics Wrapper
// =============================================================================
// Platform-agnostic SIMD intrinsics for AVX2/SSE4.2
// =============================================================================

#pragma once

#include "../Types.h"

// Include appropriate SIMD headers
#if defined(WULFNET_SIMD_AVX512) || defined(WULFNET_SIMD_AVX2)
    #include <immintrin.h>
#elif defined(WULFNET_SIMD_SSE42)
    #include <nmmintrin.h>
#endif

namespace WulfNet {
namespace SIMD {

// =============================================================================
// SIMD Types
// =============================================================================

// 128-bit vector (4 floats)
using Vec4f = __m128;
using Vec4i = __m128i;
using Vec2d = __m128d;

// 256-bit vector (8 floats) - AVX2
#if defined(WULFNET_SIMD_AVX2) || defined(WULFNET_SIMD_AVX512)
using Vec8f = __m256;
using Vec8i = __m256i;
using Vec4d = __m256d;
#endif

// =============================================================================
// Load/Store Operations (128-bit)
// =============================================================================

// Load 4 floats (aligned)
WULFNET_FORCEINLINE Vec4f load4f(const float* ptr) {
    return _mm_load_ps(ptr);
}

// Load 4 floats (unaligned)
WULFNET_FORCEINLINE Vec4f loadu4f(const float* ptr) {
    return _mm_loadu_ps(ptr);
}

// Load single float broadcast to all lanes
WULFNET_FORCEINLINE Vec4f broadcast4f(float value) {
    return _mm_set1_ps(value);
}

// Load 4 different floats
WULFNET_FORCEINLINE Vec4f set4f(float x, float y, float z, float w) {
    return _mm_set_ps(w, z, y, x);  // Note: reversed order for SSE
}

// Store 4 floats (aligned)
WULFNET_FORCEINLINE void store4f(float* ptr, Vec4f v) {
    _mm_store_ps(ptr, v);
}

// Store 4 floats (unaligned)
WULFNET_FORCEINLINE void storeu4f(float* ptr, Vec4f v) {
    _mm_storeu_ps(ptr, v);
}

// Zero vector
WULFNET_FORCEINLINE Vec4f zero4f() {
    return _mm_setzero_ps();
}

// =============================================================================
// Arithmetic Operations (128-bit)
// =============================================================================

WULFNET_FORCEINLINE Vec4f add4f(Vec4f a, Vec4f b) { return _mm_add_ps(a, b); }
WULFNET_FORCEINLINE Vec4f sub4f(Vec4f a, Vec4f b) { return _mm_sub_ps(a, b); }
WULFNET_FORCEINLINE Vec4f mul4f(Vec4f a, Vec4f b) { return _mm_mul_ps(a, b); }
WULFNET_FORCEINLINE Vec4f div4f(Vec4f a, Vec4f b) { return _mm_div_ps(a, b); }

// Fused multiply-add: a * b + c
WULFNET_FORCEINLINE Vec4f fmadd4f(Vec4f a, Vec4f b, Vec4f c) {
    #if defined(__FMA__)
        return _mm_fmadd_ps(a, b, c);
    #else
        return _mm_add_ps(_mm_mul_ps(a, b), c);
    #endif
}

// Fused multiply-subtract: a * b - c
WULFNET_FORCEINLINE Vec4f fmsub4f(Vec4f a, Vec4f b, Vec4f c) {
    #if defined(__FMA__)
        return _mm_fmsub_ps(a, b, c);
    #else
        return _mm_sub_ps(_mm_mul_ps(a, b), c);
    #endif
}

// Negation
WULFNET_FORCEINLINE Vec4f neg4f(Vec4f v) {
    return _mm_xor_ps(v, _mm_set1_ps(-0.0f));
}

// Absolute value
WULFNET_FORCEINLINE Vec4f abs4f(Vec4f v) {
    const Vec4f mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    return _mm_and_ps(v, mask);
}

// Min/Max
WULFNET_FORCEINLINE Vec4f min4f(Vec4f a, Vec4f b) { return _mm_min_ps(a, b); }
WULFNET_FORCEINLINE Vec4f max4f(Vec4f a, Vec4f b) { return _mm_max_ps(a, b); }

// Reciprocal (approximate, ~12-bit precision)
WULFNET_FORCEINLINE Vec4f rcp4f(Vec4f v) { return _mm_rcp_ps(v); }

// Reciprocal square root (approximate)
WULFNET_FORCEINLINE Vec4f rsqrt4f(Vec4f v) { return _mm_rsqrt_ps(v); }

// Square root
WULFNET_FORCEINLINE Vec4f sqrt4f(Vec4f v) { return _mm_sqrt_ps(v); }

// =============================================================================
// Shuffle and Swizzle Operations
// =============================================================================

// Shuffle using compile-time mask
template<int X, int Y, int Z, int W>
WULFNET_FORCEINLINE Vec4f shuffle4f(Vec4f v) {
    return _mm_shuffle_ps(v, v, _MM_SHUFFLE(W, Z, Y, X));
}

// Shuffle from two vectors
template<int X, int Y, int Z, int W>
WULFNET_FORCEINLINE Vec4f shuffle4f(Vec4f a, Vec4f b) {
    return _mm_shuffle_ps(a, b, _MM_SHUFFLE(W, Z, Y, X));
}

// Common swizzles
WULFNET_FORCEINLINE Vec4f xxxx(Vec4f v) { return shuffle4f<0, 0, 0, 0>(v); }
WULFNET_FORCEINLINE Vec4f yyyy(Vec4f v) { return shuffle4f<1, 1, 1, 1>(v); }
WULFNET_FORCEINLINE Vec4f zzzz(Vec4f v) { return shuffle4f<2, 2, 2, 2>(v); }
WULFNET_FORCEINLINE Vec4f wwww(Vec4f v) { return shuffle4f<3, 3, 3, 3>(v); }

// Extract single element
WULFNET_FORCEINLINE float extractX(Vec4f v) { return _mm_cvtss_f32(v); }
WULFNET_FORCEINLINE float extractY(Vec4f v) { return _mm_cvtss_f32(shuffle4f<1, 1, 1, 1>(v)); }
WULFNET_FORCEINLINE float extractZ(Vec4f v) { return _mm_cvtss_f32(shuffle4f<2, 2, 2, 2>(v)); }
WULFNET_FORCEINLINE float extractW(Vec4f v) { return _mm_cvtss_f32(shuffle4f<3, 3, 3, 3>(v)); }

// =============================================================================
// Dot Product and Cross Product
// =============================================================================

// Dot product of 3D vectors (x, y, z components)
WULFNET_FORCEINLINE float dot3(Vec4f a, Vec4f b) {
    #if defined(__SSE4_1__)
        return _mm_cvtss_f32(_mm_dp_ps(a, b, 0x71));  // mask: xyz -> x
    #else
        Vec4f m = mul4f(a, b);
        Vec4f s1 = add4f(m, shuffle4f<1, 0, 0, 0>(m));  // x+y
        Vec4f s2 = add4f(s1, shuffle4f<2, 0, 0, 0>(m)); // x+y+z
        return extractX(s2);
    #endif
}

// Dot product of 4D vectors
WULFNET_FORCEINLINE float dot4(Vec4f a, Vec4f b) {
    #if defined(__SSE4_1__)
        return _mm_cvtss_f32(_mm_dp_ps(a, b, 0xF1));  // all components -> x
    #else
        Vec4f m = mul4f(a, b);
        Vec4f s1 = add4f(m, shuffle4f<2, 3, 0, 1>(m));
        Vec4f s2 = add4f(s1, shuffle4f<1, 0, 3, 2>(s1));
        return extractX(s2);
    #endif
}

// Cross product of 3D vectors
WULFNET_FORCEINLINE Vec4f cross3(Vec4f a, Vec4f b) {
    // (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
    Vec4f a_yzx = shuffle4f<1, 2, 0, 3>(a);
    Vec4f b_yzx = shuffle4f<1, 2, 0, 3>(b);
    Vec4f c = fmsub4f(a, b_yzx, mul4f(a_yzx, b));
    return shuffle4f<1, 2, 0, 3>(c);
}

// =============================================================================
// Comparison Operations
// =============================================================================

WULFNET_FORCEINLINE Vec4f cmpEq4f(Vec4f a, Vec4f b) { return _mm_cmpeq_ps(a, b); }
WULFNET_FORCEINLINE Vec4f cmpNe4f(Vec4f a, Vec4f b) { return _mm_cmpneq_ps(a, b); }
WULFNET_FORCEINLINE Vec4f cmpLt4f(Vec4f a, Vec4f b) { return _mm_cmplt_ps(a, b); }
WULFNET_FORCEINLINE Vec4f cmpLe4f(Vec4f a, Vec4f b) { return _mm_cmple_ps(a, b); }
WULFNET_FORCEINLINE Vec4f cmpGt4f(Vec4f a, Vec4f b) { return _mm_cmpgt_ps(a, b); }
WULFNET_FORCEINLINE Vec4f cmpGe4f(Vec4f a, Vec4f b) { return _mm_cmpge_ps(a, b); }

// Select: (mask & a) | (~mask & b) - conditional select
WULFNET_FORCEINLINE Vec4f select4f(Vec4f mask, Vec4f a, Vec4f b) {
    #if defined(__SSE4_1__)
        return _mm_blendv_ps(b, a, mask);
    #else
        return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
    #endif
}

// =============================================================================
// AVX2 Operations (256-bit)
// =============================================================================

#if defined(WULFNET_SIMD_AVX2) || defined(WULFNET_SIMD_AVX512)

// Load/Store (8 floats)
WULFNET_FORCEINLINE Vec8f load8f(const float* ptr) { return _mm256_load_ps(ptr); }
WULFNET_FORCEINLINE Vec8f loadu8f(const float* ptr) { return _mm256_loadu_ps(ptr); }
WULFNET_FORCEINLINE Vec8f broadcast8f(float value) { return _mm256_set1_ps(value); }
WULFNET_FORCEINLINE void store8f(float* ptr, Vec8f v) { _mm256_store_ps(ptr, v); }
WULFNET_FORCEINLINE void storeu8f(float* ptr, Vec8f v) { _mm256_storeu_ps(ptr, v); }
WULFNET_FORCEINLINE Vec8f zero8f() { return _mm256_setzero_ps(); }

// Arithmetic
WULFNET_FORCEINLINE Vec8f add8f(Vec8f a, Vec8f b) { return _mm256_add_ps(a, b); }
WULFNET_FORCEINLINE Vec8f sub8f(Vec8f a, Vec8f b) { return _mm256_sub_ps(a, b); }
WULFNET_FORCEINLINE Vec8f mul8f(Vec8f a, Vec8f b) { return _mm256_mul_ps(a, b); }
WULFNET_FORCEINLINE Vec8f div8f(Vec8f a, Vec8f b) { return _mm256_div_ps(a, b); }

WULFNET_FORCEINLINE Vec8f fmadd8f(Vec8f a, Vec8f b, Vec8f c) {
    return _mm256_fmadd_ps(a, b, c);
}

WULFNET_FORCEINLINE Vec8f fmsub8f(Vec8f a, Vec8f b, Vec8f c) {
    return _mm256_fmsub_ps(a, b, c);
}

WULFNET_FORCEINLINE Vec8f min8f(Vec8f a, Vec8f b) { return _mm256_min_ps(a, b); }
WULFNET_FORCEINLINE Vec8f max8f(Vec8f a, Vec8f b) { return _mm256_max_ps(a, b); }
WULFNET_FORCEINLINE Vec8f sqrt8f(Vec8f v) { return _mm256_sqrt_ps(v); }
WULFNET_FORCEINLINE Vec8f rsqrt8f(Vec8f v) { return _mm256_rsqrt_ps(v); }

// Comparison
WULFNET_FORCEINLINE Vec8f cmpLt8f(Vec8f a, Vec8f b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
WULFNET_FORCEINLINE Vec8f cmpGt8f(Vec8f a, Vec8f b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }

WULFNET_FORCEINLINE Vec8f select8f(Vec8f mask, Vec8f a, Vec8f b) {
    return _mm256_blendv_ps(b, a, mask);
}

#endif // AVX2

} // namespace SIMD
} // namespace WulfNet
