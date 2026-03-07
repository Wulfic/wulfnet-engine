// =============================================================================
// BACKWARD COMPATIBILITY REDIRECT
// WaterSystemV3.h has been renamed to Physics/Fluids/FluidSystem.h
// and moved from WulfNet::Physics:: namespace to flat WulfNet:: namespace.
// This header provides aliases so existing code continues to compile.
// =============================================================================
#pragma once

#include "Fluids/FluidSystem.h"

// Provide backward-compatible aliases in the old WulfNet::Physics:: namespace
namespace WulfNet {
namespace Physics {
    using FluidSystemConfig   = WulfNet::FluidSystemConfig;
    using WaterSystemV3Config = WulfNet::FluidSystemConfig;
    using FluidSystem         = WulfNet::FluidSystem;
    using WaterSystemV3       = WulfNet::FluidSystem;
    using WaterStateSOA       = WulfNet::WaterStateSOA;
    using MacroTileID         = WulfNet::MacroTileID;
    using DispatchIndirectArgs = WulfNet::DispatchIndirectArgs;
} // namespace Physics
} // namespace WulfNet
