// =============================================================================
// WulfNet Engine - Assert Macros
// =============================================================================
// Debug assertions with customizable behavior
// =============================================================================

#pragma once

#include "Types.h"
#include <cstdio>
#include <cstdlib>

namespace WulfNet {

// =============================================================================
// Assert Handler
// =============================================================================

using AssertHandler = void(*)(const char* expression, const char* message, 
                               const char* file, int line);

inline AssertHandler g_assertHandler = nullptr;

inline void setAssertHandler(AssertHandler handler) {
    g_assertHandler = handler;
}

inline void defaultAssertHandler(const char* expression, const char* message,
                                  const char* file, int line) {
    std::fprintf(stderr, 
        "\n======================================\n"
        "ASSERTION FAILED!\n"
        "--------------------------------------\n"
        "Expression: %s\n"
        "Message:    %s\n"
        "File:       %s\n"
        "Line:       %d\n"
        "======================================\n",
        expression, message ? message : "(none)", file, line
    );
    std::fflush(stderr);
}

[[noreturn]] inline void assertFailed(const char* expression, const char* message,
                                       const char* file, int line) {
    if (g_assertHandler) {
        g_assertHandler(expression, message, file, line);
    } else {
        defaultAssertHandler(expression, message, file, line);
    }
    
    WULFNET_DEBUGBREAK();
    std::abort();
}

} // namespace WulfNet

// =============================================================================
// Assert Macros
// =============================================================================

#if defined(NDEBUG) || defined(WULFNET_DISABLE_ASSERTS)
    #define WULFNET_ASSERT(expr) ((void)0)
    #define WULFNET_ASSERT_MSG(expr, msg) ((void)0)
    #define WULFNET_VERIFY(expr) ((void)(expr))
#else
    #define WULFNET_ASSERT(expr) \
        do { \
            if (WULFNET_UNLIKELY(!(expr))) { \
                ::WulfNet::assertFailed(#expr, nullptr, __FILE__, __LINE__); \
            } \
        } while (0)
    
    #define WULFNET_ASSERT_MSG(expr, msg) \
        do { \
            if (WULFNET_UNLIKELY(!(expr))) { \
                ::WulfNet::assertFailed(#expr, msg, __FILE__, __LINE__); \
            } \
        } while (0)
    
    // Verify always evaluates expression but only asserts in debug
    #define WULFNET_VERIFY(expr) WULFNET_ASSERT(expr)
#endif

// Static assert with better messages
#define WULFNET_STATIC_ASSERT(expr, msg) static_assert(expr, msg)

// Compile-time checks
#define WULFNET_ENSURE_SIZE(type, size) \
    WULFNET_STATIC_ASSERT(sizeof(type) == size, #type " must be " #size " bytes")

#define WULFNET_ENSURE_ALIGNMENT(type, alignment) \
    WULFNET_STATIC_ASSERT(alignof(type) == alignment, #type " must be aligned to " #alignment " bytes")
