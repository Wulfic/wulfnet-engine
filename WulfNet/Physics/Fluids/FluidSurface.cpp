// =============================================================================
// WulfNet Engine - Fluid Surface Implementation (Marching Cubes)
// GPU-accelerated isosurface generation for smooth water rendering
// =============================================================================

#include "FluidSurface.h"
#include "COFLIPSystem.h"
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <chrono>

#ifdef WULFNET_HAS_OPENMP
#include <omp.h>
#endif

namespace WulfNet {

// =============================================================================
// Marching Cubes tables + ExtractSurface + ProcessCell + InterpolateVertex
// + ComputeNormal + GPU stubs --> FluidSurfaceMarch.cpp
// =============================================================================
// =============================================================================
// Constructor / Destructor
// =============================================================================

FluidSurface::FluidSurface() = default;

FluidSurface::~FluidSurface() {
    Shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool FluidSurface::Initialize(const FluidSurfaceConfig& config, VulkanContext* vulkan) {
    if (m_initialized) return false;

    m_config = config;
    m_vulkanContext = vulkan;
    m_gpuEnabled = (vulkan != nullptr) && config.useGPU;

    // Allocate density grid
    m_gridTotalCells = config.gridSizeX * config.gridSizeY * config.gridSizeZ;
    m_density.resize(m_gridTotalCells, 0.0f);

    // Reserve mesh storage
    m_vertices.reserve(m_gridTotalCells);
    m_triangles.reserve(m_gridTotalCells);
    m_indices.reserve(m_gridTotalCells * 3);

    m_initialized = true;
    return true;
}

void FluidSurface::Shutdown() {
    m_density.clear();
    m_vertices.clear();
    m_triangles.clear();
    m_indices.clear();

    m_vulkanContext = nullptr;
    m_gpuEnabled = false;
    m_initialized = false;
}

// =============================================================================
// Grid Helpers
// =============================================================================

int FluidSurface::GridIndex(int i, int j, int k) const {
    return i + j * m_config.gridSizeX + k * m_config.gridSizeX * m_config.gridSizeY;
}

bool FluidSurface::InBounds(int i, int j, int k) const {
    return i >= 0 && i < static_cast<int>(m_config.gridSizeX) &&
           j >= 0 && j < static_cast<int>(m_config.gridSizeY) &&
           k >= 0 && k < static_cast<int>(m_config.gridSizeZ);
}

void FluidSurface::WorldToGrid(float wx, float wy, float wz, float& gx, float& gy, float& gz) const {
    gx = wx / m_config.cellSize;
    gy = wy / m_config.cellSize;
    gz = wz / m_config.cellSize;
}

void FluidSurface::GridToWorld(int i, int j, int k, float& wx, float& wy, float& wz) const {
    wx = i * m_config.cellSize;
    wy = j * m_config.cellSize;
    wz = k * m_config.cellSize;
}

float FluidSurface::GetDensity(int i, int j, int k) const {
    if (!InBounds(i, j, k)) return 0.0f;
    return m_density[GridIndex(i, j, k)];
}

void FluidSurface::SetDensity(int i, int j, int k, float value) {
    if (InBounds(i, j, k)) {
        m_density[GridIndex(i, j, k)] = value;
    }
}

// =============================================================================
// Surface Generation
// =============================================================================

void FluidSurface::GenerateSurface(const COFLIPSystem& fluid) {
    if (!m_initialized) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Clear density
    auto splatStart = std::chrono::high_resolution_clock::now();
    ClearDensity();

    // =======================================================================
    // Parallel particle splatting with thread-local density accumulation.
    // Each thread splats into its own density array to avoid contention,
    // then we merge them in a parallel reduction.
    // =======================================================================
    const auto& particles = fluid.GetParticles();
    uint32_t activeCount = fluid.GetActiveParticleCount();

#ifdef WULFNET_HAS_OPENMP
    const int nThreads = omp_get_max_threads();
#else
    const int nThreads = 1;
#endif

    // Allocate/resize thread-local density buffers (persistent)
    if (m_splatThreadCount != nThreads) {
        m_splatThreadDensity.resize(nThreads);
        for (auto& td : m_splatThreadDensity) {
            td.resize(m_gridTotalCells);
        }
        m_splatThreadCount = nThreads;
    }

    // Clear thread-local buffers in parallel
#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& td = m_splatThreadDensity[tid];
        #pragma omp for schedule(static)
        for (int32_t idx = 0; idx < static_cast<int32_t>(m_gridTotalCells); ++idx) {
            td[idx] = 0.0f;
        }
    }
#else
    std::fill(m_splatThreadDensity[0].begin(), m_splatThreadDensity[0].end(), 0.0f);
#endif

    // Parallel particle splatting — each thread writes to its own density grid
#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int32_t p = 0; p < static_cast<int32_t>(activeCount); ++p) {
        const auto& part = particles[p];
        if (!(part.flags & 1)) continue;

#ifdef WULFNET_HAS_OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        auto& td = m_splatThreadDensity[tid];

        // Inline splatting (same logic as SplatParticle but writes to thread-local buffer)
        float gx, gy, gz;
        WorldToGrid(part.x, part.y, part.z, gx, gy, gz);

        float radius = m_config.splatRadius;
        float sigma = m_config.smoothingSigma;
        float invTwoSigma2 = -1.0f / (2.0f * sigma * sigma);
        float radiusSq = radius * radius;

        int iMin = std::max(0, static_cast<int>(gx - radius));
        int iMax = std::min(static_cast<int>(m_config.gridSizeX) - 1, static_cast<int>(gx + radius));
        int jMin = std::max(0, static_cast<int>(gy - radius));
        int jMax = std::min(static_cast<int>(m_config.gridSizeY) - 1, static_cast<int>(gy + radius));
        int kMin = std::max(0, static_cast<int>(gz - radius));
        int kMax = std::min(static_cast<int>(m_config.gridSizeZ) - 1, static_cast<int>(gz + radius));

        for (int k = kMin; k <= kMax; ++k) {
            float dz = k - gz;
            float dz2 = dz * dz;
            for (int j = jMin; j <= jMax; ++j) {
                float dy = j - gy;
                float dy2dz2 = dy * dy + dz2;
                if (dy2dz2 > radiusSq) continue;
                for (int i = iMin; i <= iMax; ++i) {
                    float dx = i - gx;
                    float dist2 = dx*dx + dy2dz2;
                    if (dist2 > radiusSq) continue;
                    float ex = dist2 * (-invTwoSigma2);
                    float w = 1.0f / (1.0f + ex + 0.5f * ex * ex);
                    td[GridIndex(i, j, k)] += w;
                }
            }
        }
    }

    // Merge thread-local density grids into m_density
#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t idx = 0; idx < static_cast<int32_t>(m_gridTotalCells); ++idx) {
        float sum = 0.0f;
        for (int t = 0; t < nThreads; ++t) {
            sum += m_splatThreadDensity[t][idx];
        }
        m_density[idx] = sum;
    }

    auto splatEnd = std::chrono::high_resolution_clock::now();
    m_stats.splatTimeMs = std::chrono::duration<float, std::milli>(splatEnd - splatStart).count();

    // Smooth density field
    auto smoothStart = std::chrono::high_resolution_clock::now();
    SmoothDensity();
    auto smoothEnd = std::chrono::high_resolution_clock::now();
    m_stats.smoothTimeMs = std::chrono::duration<float, std::milli>(smoothEnd - smoothStart).count();

    // Extract surface via marching cubes
    auto mcStart = std::chrono::high_resolution_clock::now();
    ExtractSurface();
    auto mcEnd = std::chrono::high_resolution_clock::now();
    m_stats.marchingCubesTimeMs = std::chrono::duration<float, std::milli>(mcEnd - mcStart).count();

    auto endTime = std::chrono::high_resolution_clock::now();
    m_stats.totalTimeMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    m_stats.vertexCount = static_cast<uint32_t>(m_vertices.size());
    m_stats.triangleCount = static_cast<uint32_t>(m_triangles.size());

    // Cache vertex Y-extent (eliminates per-frame scan in DrawSurface).
    // Parallel reduction over the vertex array.
    if (!m_vertices.empty()) {
        float vMinY = FLT_MAX, vMaxY = -FLT_MAX;
        const int32_t nVerts = static_cast<int32_t>(m_vertices.size());
#ifdef WULFNET_HAS_OPENMP
        #pragma omp parallel for schedule(static) reduction(min:vMinY) reduction(max:vMaxY)
#endif
        for (int32_t i = 0; i < nVerts; ++i) {
            float y = m_vertices[i].y;
            if (y < vMinY) vMinY = y;
            if (y > vMaxY) vMaxY = y;
        }
        m_stats.minVertexY = vMinY;
        m_stats.maxVertexY = vMaxY;
    } else {
        m_stats.minVertexY = 0.0f;
        m_stats.maxVertexY = 0.0f;
    }
}

void FluidSurface::ClearDensity() {
#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for schedule(static)
    for (int32_t idx = 0; idx < static_cast<int32_t>(m_gridTotalCells); ++idx) {
        m_density[idx] = 0.0f;
    }
#else
    std::fill(m_density.begin(), m_density.end(), 0.0f);
#endif
}

void FluidSurface::SplatParticle(float x, float y, float z, float weight) {
    float gx, gy, gz;
    WorldToGrid(x, y, z, gx, gy, gz);

    float radius = m_config.splatRadius;
    float sigma = m_config.smoothingSigma;
    float invTwoSigma2 = -1.0f / (2.0f * sigma * sigma);
    float radiusSq = radius * radius;  // Pre-compute for early rejection

    int iMin = std::max(0, static_cast<int>(gx - radius));
    int iMax = std::min(static_cast<int>(m_config.gridSizeX) - 1, static_cast<int>(gx + radius));
    int jMin = std::max(0, static_cast<int>(gy - radius));
    int jMax = std::min(static_cast<int>(m_config.gridSizeY) - 1, static_cast<int>(gy + radius));
    int kMin = std::max(0, static_cast<int>(gz - radius));
    int kMax = std::min(static_cast<int>(m_config.gridSizeZ) - 1, static_cast<int>(gz + radius));

    for (int k = kMin; k <= kMax; ++k) {
        float dz = k - gz;
        float dz2 = dz * dz;
        for (int j = jMin; j <= jMax; ++j) {
            float dy = j - gy;
            float dy2dz2 = dy * dy + dz2;
            // Early reject entire row if already outside radius
            if (dy2dz2 > radiusSq) continue;
            for (int i = iMin; i <= iMax; ++i) {
                float dx = i - gx;
                float dist2 = dx*dx + dy2dz2;

                if (dist2 > radiusSq) continue;

                // Fast Gaussian approximation: exp(-x) ≈ 1/(1+x+0.5x²) for x>=0.
                // This is ~3× faster than std::exp and sufficiently accurate for
                // splatting density (visual, not physical).
                float ex = dist2 * (-invTwoSigma2);  // positive value
                float w = weight / (1.0f + ex + 0.5f * ex * ex);
                m_density[GridIndex(i, j, k)] += w;
            }
        }
    }
}

void FluidSurface::SmoothDensity() {
    // 3x3x3 box blur — reads from m_density, writes to m_smoothTemp.
    // Parallelizable because each output cell is independent.
    if (m_smoothTemp.size() != m_gridTotalCells) {
        m_smoothTemp.resize(m_gridTotalCells, 0.0f);
    } else {
#ifdef WULFNET_HAS_OPENMP
        #pragma omp parallel for schedule(static)
        for (int32_t idx = 0; idx < static_cast<int32_t>(m_gridTotalCells); ++idx) {
            m_smoothTemp[idx] = 0.0f;
        }
#else
        std::fill(m_smoothTemp.begin(), m_smoothTemp.end(), 0.0f);
#endif
    }

    const int32_t NX = static_cast<int32_t>(m_config.gridSizeX);
    const int32_t NY = static_cast<int32_t>(m_config.gridSizeY);
    const int32_t NZ = static_cast<int32_t>(m_config.gridSizeZ);

#ifdef WULFNET_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int32_t k = 1; k < NZ - 1; ++k) {
        for (int32_t j = 1; j < NY - 1; ++j) {
            for (int32_t i = 1; i < NX - 1; ++i) {
                float sum = 0;
                for (int dk = -1; dk <= 1; ++dk) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        for (int di = -1; di <= 1; ++di) {
                            sum += m_density[GridIndex(i + di, j + dj, k + dk)];
                        }
                    }
                }
                m_smoothTemp[GridIndex(i, j, k)] = sum / 27.0f;
            }
        }
    }

    // Swap instead of copy — O(1) vs O(N)
    std::swap(m_density, m_smoothTemp);
}

} // namespace WulfNet
