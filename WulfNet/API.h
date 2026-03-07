// =============================================================================
// WulfNet Engine - API Export/Visibility Macros
// =============================================================================
// Controls symbol visibility for shared library builds.
// For static library builds (default), these are no-ops.
//
// Define WULFNET_SHARED when building as a shared library.
// Define WULFNET_EXPORT when building the library (not consuming it).
// =============================================================================

#pragma once

#ifdef WULFNET_SHARED
    #ifdef _WIN32
        #ifdef WULFNET_EXPORT
            #define WULFNET_API __declspec(dllexport)
        #else
            #define WULFNET_API __declspec(dllimport)
        #endif
    #elif defined(__GNUC__) || defined(__clang__)
        #ifdef WULFNET_EXPORT
            #define WULFNET_API __attribute__((visibility("default")))
        #else
            #define WULFNET_API
        #endif
    #else
        #define WULFNET_API
    #endif
#else
    // Static library — no decoration needed
    #define WULFNET_API
#endif

// For classes that should never be exported (internal implementation detail)
#define WULFNET_INTERNAL

// For deprecated APIs that should still work but encourage migration
#define WULFNET_DEPRECATED(msg) [[deprecated(msg)]]
