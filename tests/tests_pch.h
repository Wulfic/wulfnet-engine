// =============================================================================
// WulfNet Engine - Forced Include Header for Tests
// =============================================================================
// This header is force-included before all other headers in test files
// to ensure proper compatibility with third-party libraries.
// =============================================================================

#pragma once

// Ensure assert is available for Catch2
#include <cstdlib>
#include <cassert>

// Fallback in case cassert doesn't define assert properly
#ifndef assert
    #ifdef NDEBUG
        #define assert(expr) ((void)0)
    #else
        #include <cstdio>
        #define assert(expr) \
            do { \
                if (!(expr)) { \
                    std::fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", \
                        #expr, __FILE__, __LINE__); \
                    std::abort(); \
                } \
            } while (0)
    #endif
#endif
