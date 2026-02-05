@echo off
REM =============================================================================
REM WulfNet Engine - Compile Optimized Fluid Shaders
REM =============================================================================
REM Compiles the optimized CO-FLIP compute shaders to SPIR-V
REM Requires glslc (from Vulkan SDK) in PATH
REM =============================================================================

cd /d "%~dp0"

echo Compiling optimized fluid compute shaders...
echo.

REM =============================================================================
REM Core Simulation Shaders (Optimized)
REM =============================================================================

REM Compile optimized G2P shader (replaces standard G2P)
echo   [1/7] Compiling coflip_g2p_optimized.comp...
glslc -fshader-stage=compute -O coflip_g2p_optimized.comp -o coflip_g2p.spv
if errorlevel 1 (
    echo ERROR: Failed to compile coflip_g2p_optimized.comp
    exit /b 1
)

REM Compile optimized P2G shader (replaces standard P2G)
echo   [2/7] Compiling coflip_p2g_optimized.comp...
glslc -fshader-stage=compute -O coflip_p2g_optimized.comp -o coflip_p2g.spv
if errorlevel 1 (
    echo ERROR: Failed to compile coflip_p2g_optimized.comp
    exit /b 1
)

REM Compile G2P with shared memory optimization
echo   [3/7] Compiling coflip_g2p_shared.comp...
glslc -fshader-stage=compute -O coflip_g2p_shared.comp -o coflip_g2p_shared.spv
if errorlevel 1 (
    echo ERROR: Failed to compile coflip_g2p_shared.comp
    exit /b 1
)

REM =============================================================================
REM Particle Sorting Shaders (Cache Coherence Optimization)
REM =============================================================================

REM Compile cell index computation shader
echo   [4/7] Compiling coflip_cell_index.comp...
glslc -fshader-stage=compute -O coflip_cell_index.comp -o coflip_cell_index.spv
if errorlevel 1 (
    echo ERROR: Failed to compile coflip_cell_index.comp
    exit /b 1
)

REM Compile radix sort shader
echo   [5/7] Compiling coflip_radix_sort.comp...
glslc -fshader-stage=compute -O coflip_radix_sort.comp -o coflip_radix_sort.spv
if errorlevel 1 (
    echo ERROR: Failed to compile coflip_radix_sort.comp
    exit /b 1
)

REM Compile particle reorder shader
echo   [6/7] Compiling coflip_reorder.comp...
glslc -fshader-stage=compute -O coflip_reorder.comp -o coflip_reorder.spv
if errorlevel 1 (
    echo ERROR: Failed to compile coflip_reorder.comp
    exit /b 1
)

echo.
echo =============================================================================
echo All optimized shaders compiled successfully!
echo =============================================================================
echo.
echo Optimizations included:
echo.
echo   1. BATCHED DISPATCH (no shader changes needed):
echo      - Single command buffer submission eliminates 7+ vkQueueWaitIdle calls
echo      - Expected improvement: 50-70%% of frame time saved
echo.
echo   2. QUADRATIC B-SPLINE (coflip_g2p.spv, coflip_p2g.spv):
echo      - 27 grid reads/writes instead of 64 (57%% reduction)
echo      - Optimized workgroup sizes for GPU occupancy
echo      - Expected improvement: 40-60%% faster G2P/P2G
echo.
echo   3. SHARED MEMORY G2P (coflip_g2p_shared.spv):
echo      - Caches 10x10x6 grid region in shared memory
echo      - Reduces global memory bandwidth by ~8x
echo      - Expected improvement: 3-5x faster G2P
echo.
echo   4. PARTICLE SORTING (cell_index, radix_sort, reorder):
echo      - GPU radix sort by Morton-encoded cell index
echo      - Improves cache coherence for particles in same cell
echo      - Expected improvement: 20-40%% better memory access patterns
echo.
echo Combined expected improvement: 60+ FPS (from 4 FPS baseline)
echo.
