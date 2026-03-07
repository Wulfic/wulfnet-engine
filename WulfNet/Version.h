// =============================================================================
// WulfNet Engine - Version Information
// =============================================================================
// Semantic versioning: MAJOR.MINOR.PATCH
//   MAJOR — breaking API changes
//   MINOR — new features, backward-compatible
//   PATCH — bug fixes, backward-compatible
// =============================================================================

#pragma once

#define WULFNET_VERSION_MAJOR 1
#define WULFNET_VERSION_MINOR 0
#define WULFNET_VERSION_PATCH 0
#define WULFNET_VERSION_STRING "1.0.0"

// Compile-time version check: WULFNET_VERSION_AT_LEAST(1, 0, 0)
#define WULFNET_VERSION_AT_LEAST(major, minor, patch) \
    ((WULFNET_VERSION_MAJOR > (major)) || \
     (WULFNET_VERSION_MAJOR == (major) && WULFNET_VERSION_MINOR > (minor)) || \
     (WULFNET_VERSION_MAJOR == (major) && WULFNET_VERSION_MINOR == (minor) && WULFNET_VERSION_PATCH >= (patch)))
