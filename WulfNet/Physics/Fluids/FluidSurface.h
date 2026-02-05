// =============================================================================
// WulfNet Engine - Fluid Surface Extraction via Marching Cubes
// GPU-accelerated isosurface generation for smooth water rendering
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <memory>

namespace WulfNet {

// Forward declarations
class VulkanContext;
class COFLIPSystem;

// =============================================================================
// Surface Vertex - GPU-aligned for rendering
// =============================================================================
struct alignas(16) FluidSurfaceVertex {
    float x, y, z;      // Position (12 bytes)
    float nx, ny, nz;   // Normal (12 bytes)
    float u, v;         // UV (8 bytes) - for caustics/foam
    // Total: 32 bytes
};
static_assert(sizeof(FluidSurfaceVertex) == 32, "FluidSurfaceVertex must be 32 bytes");

// =============================================================================
// Surface Triangle
// =============================================================================
struct FluidSurfaceTriangle {
    uint32_t v0, v1, v2;
};

// =============================================================================
// Density Grid Cell - for splatting particles
// =============================================================================
struct alignas(4) DensityCell {
    float density;
};

// =============================================================================
// Fluid Surface Configuration
// =============================================================================
struct FluidSurfaceConfig {
    // Grid resolution for density field (can be different from simulation grid)
    uint32_t gridSizeX = 64;
    uint32_t gridSizeY = 64;
    uint32_t gridSizeZ = 64;

    // Cell size in world units
    float cellSize = 0.1f;

    // Particle splatting radius (in cells)
    float splatRadius = 2.0f;

    // Gaussian kernel sigma for smoothing
    float smoothingSigma = 1.0f;

    // Isosurface level (density threshold)
    float isoLevel = 0.5f;

    // Enable GPU acceleration
    bool useGPU = true;

    // Smooth normals via averaging
    bool smoothNormals = true;
};

// =============================================================================
// Fluid Surface Statistics
// =============================================================================
struct FluidSurfaceStats {
    uint32_t vertexCount = 0;
    uint32_t triangleCount = 0;
    float splatTimeMs = 0.0f;
    float smoothTimeMs = 0.0f;
    float marchingCubesTimeMs = 0.0f;
    float totalTimeMs = 0.0f;
};

// =============================================================================
// GPU Buffer Handle
// =============================================================================
struct GPUSurfaceBufferHandle {
    uint64_t handle = 0;
    size_t size = 0;
    bool valid() const { return handle != 0; }
};

// =============================================================================
// Fluid Surface - Marching Cubes mesh extraction
// =============================================================================
class FluidSurface {
public:
    FluidSurface();
    ~FluidSurface();

    // Initialization
    bool Initialize(const FluidSurfaceConfig& config, VulkanContext* vulkan = nullptr);
    void Shutdown();
    bool IsInitialized() const { return m_initialized; }

    // Generate surface from fluid system
    void GenerateSurface(const COFLIPSystem& fluid);

    // Manual density field operations
    void ClearDensity();
    void SplatParticle(float x, float y, float z, float weight = 1.0f);
    void SmoothDensity();
    void ExtractSurface();

    // Mesh access
    const std::vector<FluidSurfaceVertex>& GetVertices() const { return m_vertices; }
    const std::vector<FluidSurfaceTriangle>& GetTriangles() const { return m_triangles; }
    const std::vector<uint32_t>& GetIndices() const { return m_indices; }

    uint32_t GetVertexCount() const { return static_cast<uint32_t>(m_vertices.size()); }
    uint32_t GetTriangleCount() const { return static_cast<uint32_t>(m_triangles.size()); }
    uint32_t GetIndexCount() const { return static_cast<uint32_t>(m_indices.size()); }

    // GPU buffer access for rendering
    GPUSurfaceBufferHandle GetVertexBuffer() const { return m_vertexBufferHandle; }
    GPUSurfaceBufferHandle GetIndexBuffer() const { return m_indexBufferHandle; }

    const FluidSurfaceStats& GetStats() const { return m_stats; }
    const FluidSurfaceConfig& GetConfig() const { return m_config; }

    // Direct density grid access
    float GetDensity(int i, int j, int k) const;
    void SetDensity(int i, int j, int k, float value);

private:
    // Grid helpers
    int GridIndex(int i, int j, int k) const;
    bool InBounds(int i, int j, int k) const;
    void WorldToGrid(float wx, float wy, float wz, float& gx, float& gy, float& gz) const;
    void GridToWorld(int i, int j, int k, float& wx, float& wy, float& wz) const;

    // Marching cubes helpers
    void ProcessCell(int i, int j, int k);
    void InterpolateVertex(int i1, int j1, int k1, int i2, int j2, int k2,
                          float& vx, float& vy, float& vz) const;
    void ComputeNormal(float x, float y, float z, float& nx, float& ny, float& nz) const;

    // GPU operations
    void SplatDensity_GPU(const COFLIPSystem& fluid);
    void SmoothDensity_GPU();
    void MarchingCubes_GPU();

    // Update GPU buffers
    void UploadMeshToGPU();

private:
    // Configuration
    FluidSurfaceConfig m_config;
    bool m_initialized = false;
    bool m_gpuEnabled = false;

    // Density grid
    std::vector<float> m_density;
    uint32_t m_gridTotalCells = 0;

    // Output mesh
    std::vector<FluidSurfaceVertex> m_vertices;
    std::vector<FluidSurfaceTriangle> m_triangles;
    std::vector<uint32_t> m_indices;

    // Statistics
    FluidSurfaceStats m_stats;

    // GPU resources
    VulkanContext* m_vulkanContext = nullptr;
    GPUSurfaceBufferHandle m_densityBufferHandle;
    GPUSurfaceBufferHandle m_vertexBufferHandle;
    GPUSurfaceBufferHandle m_indexBufferHandle;

    // GPU pipeline handles
    uint64_t m_splatPipeline = 0;
    uint64_t m_smoothPipeline = 0;
    uint64_t m_marchingCubesPipeline = 0;
};

// =============================================================================
// Marching Cubes Tables (defined in cpp)
// =============================================================================
namespace MarchingCubesTables {
    extern const int EdgeTable[256];
    extern const int TriTable[256][16];
}

} // namespace WulfNet
