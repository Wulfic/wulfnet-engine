// =============================================================================
// WulfNet Engine - Core Types
// =============================================================================
// Fundamental type definitions, platform macros, and compiler intrinsics
// =============================================================================

#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <limits>

namespace WulfNet {

// =============================================================================
// Fixed-Width Integer Types
// =============================================================================

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

using usize = std::size_t;
using isize = std::ptrdiff_t;

using byte = u8;

// =============================================================================
// Compiler Detection
// =============================================================================

#if defined(_MSC_VER)
    #define WULFNET_COMPILER_MSVC 1
    #define WULFNET_COMPILER_VERSION _MSC_VER
#elif defined(__clang__)
    #define WULFNET_COMPILER_CLANG 1
    #define WULFNET_COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100)
#elif defined(__GNUC__)
    #define WULFNET_COMPILER_GCC 1
    #define WULFNET_COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100)
#else
    #error "Unknown compiler"
#endif

// =============================================================================
// Platform Macros
// =============================================================================

#if defined(WULFNET_PLATFORM_WINDOWS)
    #define WULFNET_DEBUGBREAK() __debugbreak()
    #define WULFNET_EXPORT __declspec(dllexport)
    #define WULFNET_IMPORT __declspec(dllimport)
#elif defined(WULFNET_PLATFORM_LINUX) || defined(WULFNET_PLATFORM_MACOS)
    #define WULFNET_DEBUGBREAK() __builtin_trap()
    #define WULFNET_EXPORT __attribute__((visibility("default")))
    #define WULFNET_IMPORT
#else
    #define WULFNET_DEBUGBREAK() ((void)0)
    #define WULFNET_EXPORT
    #define WULFNET_IMPORT
#endif

// =============================================================================
// Compiler Hints & Intrinsics
// =============================================================================

#if WULFNET_COMPILER_MSVC
    #define WULFNET_FORCEINLINE __forceinline
    #define WULFNET_NOINLINE    __declspec(noinline)
    #define WULFNET_RESTRICT    __restrict
    #define WULFNET_LIKELY(x)   (x)
    #define WULFNET_UNLIKELY(x) (x)
    #define WULFNET_ASSUME(x)   __assume(x)
    #define WULFNET_UNREACHABLE() __assume(0)
#else
    #define WULFNET_FORCEINLINE inline __attribute__((always_inline))
    #define WULFNET_NOINLINE    __attribute__((noinline))
    #define WULFNET_RESTRICT    __restrict__
    #define WULFNET_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define WULFNET_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define WULFNET_ASSUME(x)   do { if (!(x)) __builtin_unreachable(); } while(0)
    #define WULFNET_UNREACHABLE() __builtin_unreachable()
#endif

// =============================================================================
// Alignment Macros
// =============================================================================

#define WULFNET_ALIGNAS(x) alignas(x)
#define WULFNET_CACHE_LINE 64

// Cache line aligned for avoiding false sharing
#define WULFNET_CACHE_ALIGNED WULFNET_ALIGNAS(WULFNET_CACHE_LINE)

// SIMD alignment (32 bytes for AVX2, 64 for AVX512)
#if defined(WULFNET_SIMD_AVX512)
    #define WULFNET_SIMD_ALIGNMENT 64
#else
    #define WULFNET_SIMD_ALIGNMENT 32
#endif

#define WULFNET_SIMD_ALIGNED WULFNET_ALIGNAS(WULFNET_SIMD_ALIGNMENT)

// =============================================================================
// Memory Size Literals
// =============================================================================

constexpr usize operator""_KB(unsigned long long x) { return x * 1024; }
constexpr usize operator""_MB(unsigned long long x) { return x * 1024 * 1024; }
constexpr usize operator""_GB(unsigned long long x) { return x * 1024 * 1024 * 1024; }

// =============================================================================
// Numeric Constants
// =============================================================================

template<typename T>
struct NumericLimits {
    static constexpr T min() { return std::numeric_limits<T>::min(); }
    static constexpr T max() { return std::numeric_limits<T>::max(); }
    static constexpr T epsilon() { return std::numeric_limits<T>::epsilon(); }
};

// =============================================================================
// Type Traits Helpers
// =============================================================================

template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template<typename T>
concept Integral = std::is_integral_v<T>;

template<typename T>
concept Trivial = std::is_trivial_v<T>;

template<typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

// =============================================================================
// Handle Types (Type-safe opaque handles)
// =============================================================================

template<typename Tag, typename T = u32>
struct Handle {
    T value{0};
    
    constexpr Handle() = default;
    constexpr explicit Handle(T v) : value(v) {}
    
    constexpr bool isValid() const { return value != 0; }
    constexpr explicit operator bool() const { return isValid(); }
    
    constexpr bool operator==(const Handle& other) const { return value == other.value; }
    constexpr bool operator!=(const Handle& other) const { return value != other.value; }
    constexpr bool operator<(const Handle& other) const { return value < other.value; }
    
    static constexpr Handle invalid() { return Handle{0}; }
};

// Common handle types
struct EntityTag {};
struct RigidBodyTag {};
struct DistanceJointTag {};
struct BallJointTag {};
struct FixedJointTag {};
struct HingeJointTag {};
struct ColliderTag {};
struct ConstraintTag {};
struct MeshTag {};
struct MaterialTag {};
struct TextureTag {};

using EntityHandle    = Handle<EntityTag, u32>;
using RigidBodyHandle = Handle<RigidBodyTag, u32>;
using DistanceJointHandle = Handle<DistanceJointTag, u32>;
using BallJointHandle = Handle<BallJointTag, u32>;
using FixedJointHandle = Handle<FixedJointTag, u32>;
using HingeJointHandle = Handle<HingeJointTag, u32>;
using ColliderHandle  = Handle<ColliderTag, u32>;
using ConstraintHandle = Handle<ConstraintTag, u32>;
using MeshHandle      = Handle<MeshTag, u32>;
using MaterialHandle  = Handle<MaterialTag, u32>;
using TextureHandle   = Handle<TextureTag, u32>;

// =============================================================================
// Non-Copyable / Non-Moveable Base Classes
// =============================================================================

class NonCopyable {
protected:
    NonCopyable() = default;
    ~NonCopyable() = default;
    
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
};

class NonMoveable : public NonCopyable {
protected:
    NonMoveable() = default;
    ~NonMoveable() = default;
    
    NonMoveable(NonMoveable&&) = delete;
    NonMoveable& operator=(NonMoveable&&) = delete;
};

// =============================================================================
// Utility Functions
// =============================================================================

template<typename T>
constexpr T alignUp(T value, T alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

template<typename T>
constexpr T alignDown(T value, T alignment) {
    return value & ~(alignment - 1);
}

template<typename T>
constexpr bool isAligned(T value, T alignment) {
    return (value & (alignment - 1)) == 0;
}

template<typename T>
constexpr bool isPowerOfTwo(T value) {
    return value != 0 && (value & (value - 1)) == 0;
}

template<typename T>
constexpr T nextPowerOfTwo(T value) {
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    if constexpr (sizeof(T) > 1) value |= value >> 8;
    if constexpr (sizeof(T) > 2) value |= value >> 16;
    if constexpr (sizeof(T) > 4) value |= value >> 32;
    value++;
    return value;
}

// =============================================================================
// Span (Non-owning view of contiguous memory)
// =============================================================================

template<typename T>
class Span {
public:
    constexpr Span() : m_data(nullptr), m_size(0) {}
    constexpr Span(T* data, usize size) : m_data(data), m_size(size) {}
    
    template<usize N>
    constexpr Span(T (&arr)[N]) : m_data(arr), m_size(N) {}
    
    constexpr T* data() const { return m_data; }
    constexpr usize size() const { return m_size; }
    constexpr bool empty() const { return m_size == 0; }
    
    constexpr T& operator[](usize index) { return m_data[index]; }
    constexpr const T& operator[](usize index) const { return m_data[index]; }
    
    constexpr T* begin() { return m_data; }
    constexpr T* end() { return m_data + m_size; }
    constexpr const T* begin() const { return m_data; }
    constexpr const T* end() const { return m_data + m_size; }
    
    constexpr Span subspan(usize offset, usize count = ~usize(0)) const {
        if (offset >= m_size) return Span{};
        count = (count > m_size - offset) ? (m_size - offset) : count;
        return Span(m_data + offset, count);
    }
    
private:
    T* m_data;
    usize m_size;
};

} // namespace WulfNet
