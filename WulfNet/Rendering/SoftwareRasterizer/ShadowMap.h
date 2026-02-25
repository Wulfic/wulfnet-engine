// =============================================================================
// WulfNet Engine - Shadow Mapping System
// =============================================================================
// Cascade Shadow Maps (CSM) for directional lights and cube-projected shadow
// maps for point lights. Uses the same SoftwareRasterizer pipeline as the
// OcclusionCuller: render scene depth from the light's perspective, then
// sample the shadow depth buffer during lighting to compute a shadow factor.
//
// Architecture:
//   - ShadowCascade: one depth-only rasterizer for a slice of the frustum
//   - ShadowMap: manages N cascades for a directional light
//   - PointLightShadow: 6-face cube shadow map for a single point light
//   - ShadowSystem: top-level manager that holds all shadow maps and provides
//     a unified SampleShadow(worldPos) -> float [0,1] query
// =============================================================================

#pragma once

#include "WulfNet/Rendering/SoftwareRasterizer/SoftRasterTypes.h"
#include "WulfNet/Rendering/SoftwareRasterizer/GBuffer.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace WulfNet {

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for a single shadow cascade
struct ShadowCascadeConfig {
    int   resolution = 512;     ///< Depth buffer width and height
    float nearDist   = 0.1f;    ///< Near distance of this cascade slice
    float farDist    = 50.0f;   ///< Far distance of this cascade slice
    float orthoSize  = 40.0f;   ///< Orthographic half-extent (world units)
};

/// Configuration for the entire shadow system
struct ShadowSystemConfig {
    int   numCascades        = 3;           ///< Number of CSM cascades
    int   cascadeResolution  = 512;         ///< Resolution per cascade
    float maxShadowDistance  = 150.0f;      ///< Max distance shadows are visible
    float cascadeSplitLambda = 0.75f;       ///< Log/linear split ratio [0..1]
    float shadowBias         = 0.005f;      ///< Depth bias to prevent acne
    float normalBias         = 0.02f;       ///< Normal offset bias
    int   pcfSamples         = 1;           ///< 1 = hard shadow, >1 = PCF softening
    int   pointLightResolution = 256;       ///< Resolution for point light shadow faces
};

// =============================================================================
// Shadow Cascade (one slice of the CSM)
// =============================================================================

class ShadowCascade {
public:
    ShadowCascade() = default;

    /// Initialize the depth buffer
    bool Initialize(int resolution);

    /// Build an orthographic light-view-projection for this cascade
    void ComputeLightMatrix(const SoftVec3& lightDir, const SoftVec3& focusCenter,
                            float orthoSize, float nearClip, float farClip);

    /// Clear the depth buffer for a new render
    void Clear();

    /// Write a depth value at the given position (light-space NDC [-1,1])
    /// Returns true if depth test passes (closer than existing)
    bool WriteDepth(float ndcX, float ndcY, float depth);

    /// Sample the shadow depth at a world position
    /// Returns: 0.0 = fully in shadow, 1.0 = fully lit
    float SampleShadow(const SoftVec3& worldPos, float bias) const;

    /// Project a world-space point into this cascade's light-space NDC
    SoftVec3 WorldToLightNDC(const SoftVec3& worldPos) const;

    /// Accessors
    int GetResolution() const { return m_resolution; }
    const float* GetDepthBuffer() const { return m_depthBuffer.data(); }
    float GetNearClip() const { return m_nearClip; }
    float GetFarClip() const { return m_farClip; }
    float GetOrthoSize() const { return m_orthoSize; }
    SoftVec3 GetLightForward() const { return m_lightForward; }
    SoftVec3 GetLightRight() const { return m_lightRight; }
    SoftVec3 GetLightUp() const { return m_lightUp; }
    SoftVec3 GetLightPosition() const { return m_lightPosition; }

private:
    int m_resolution = 0;
    float m_nearClip = 0.1f;
    float m_farClip = 100.0f;
    float m_orthoSize = 40.0f;

    // Light-space basis
    SoftVec3 m_lightForward  = {0.0f, 0.0f, 1.0f};
    SoftVec3 m_lightRight    = {1.0f, 0.0f, 0.0f};
    SoftVec3 m_lightUp       = {0.0f, 1.0f, 0.0f};
    SoftVec3 m_lightPosition = {0.0f, 0.0f, 0.0f};

    std::vector<float> m_depthBuffer;
};

// =============================================================================
// Point Light Shadow (6-face cube projection)
// =============================================================================

class PointLightShadow {
public:
    PointLightShadow() = default;

    /// Initialize all 6 face depth buffers
    bool Initialize(int resolution);

    /// Clear all 6 faces
    void Clear();

    /// Compute the 6 face cameras from the light position
    void SetLightPosition(const SoftVec3& position, float range);

    /// Write a depth value on the appropriate face
    bool WriteDepth(int face, float ndcX, float ndcY, float depth);

    /// Sample shadow at a world position
    /// Returns 0.0 = in shadow, 1.0 = lit
    float SampleShadow(const SoftVec3& worldPos, float bias) const;

    /// Determine which face (0-5: +X,-X,+Y,-Y,+Z,-Z) a direction maps to
    static int DirectionToFace(const SoftVec3& dir);

    /// Project world pos to a specific face's NDC
    SoftVec3 WorldToFaceNDC(const SoftVec3& worldPos, int face) const;

    /// Accessors
    int GetResolution() const { return m_resolution; }
    SoftVec3 GetPosition() const { return m_position; }
    float GetRange() const { return m_range; }
    const float* GetFaceDepthBuffer(int face) const;

private:
    int m_resolution = 0;
    SoftVec3 m_position = {};
    float m_range = 10.0f;

    // 6 faces: +X, -X, +Y, -Y, +Z, -Z
    struct CubeFace {
        SoftVec3 forward;
        SoftVec3 up;
        SoftVec3 right;
        std::vector<float> depthBuffer;
    };
    CubeFace m_faces[6];
};

// =============================================================================
// Shadow System (top-level manager)
// =============================================================================

class ShadowSystem {
public:
    ShadowSystem() = default;

    /// Initialize the shadow system
    bool Initialize(const ShadowSystemConfig& config = {});

    /// Compute cascade splits for the camera frustum
    void ComputeCascadeSplits(const SoftCamera& camera);

    /// Render shadow depth maps for a directional light
    /// Transforms must be the same scene transforms used for the main render.
    /// Internally projects each mesh vertex into light space and writes depth.
    void RenderDirectionalShadows(const SoftDirectionalLight& light,
                                   const SoftCamera& camera,
                                   const SoftTransform* transforms, int transformCount,
                                   const std::vector<SoftMesh>& meshes);

    /// Render shadow maps for a point light
    void RenderPointLightShadow(int lightIndex,
                                 const SoftPointLight& light,
                                 const SoftTransform* transforms, int transformCount,
                                 const std::vector<SoftMesh>& meshes);

    /// Query the directional light shadow factor at a world position
    /// Returns 0.0 = fully shadowed, 1.0 = fully lit
    float SampleDirectionalShadow(const SoftVec3& worldPos) const;

    /// Query a point light shadow factor at a world position
    float SamplePointLightShadow(int lightIndex, const SoftVec3& worldPos) const;

    /// Select which cascade a world-space point falls into
    int SelectCascade(const SoftVec3& worldPos, const SoftCamera& camera) const;

    /// Clear all shadow maps
    void ClearAll();

    /// Accessors
    const ShadowSystemConfig& GetConfig() const { return m_config; }
    int GetCascadeCount() const { return static_cast<int>(m_cascades.size()); }
    const ShadowCascade& GetCascade(int index) const { return m_cascades[index]; }
    int GetPointLightShadowCount() const { return static_cast<int>(m_pointShadows.size()); }
    const PointLightShadow& GetPointLightShadow(int index) const { return m_pointShadows[index]; }
    const std::vector<float>& GetCascadeSplits() const { return m_cascadeSplits; }

private:
    /// Rasterize a single triangle into a cascade's depth buffer
    void RasterizeShadowTriangle(ShadowCascade& cascade,
                                  const SoftVec3& v0, const SoftVec3& v1, const SoftVec3& v2);

    /// Rasterize a single triangle into a point light face
    void RasterizeShadowTriangleFace(PointLightShadow& shadow, int face,
                                      const SoftVec3& v0, const SoftVec3& v1, const SoftVec3& v2);

    /// Transform a point by a SoftTransform (position, rotation, scale)
    SoftVec3 TransformPoint(const SoftVec3& point, const SoftTransform& transform) const;

    ShadowSystemConfig m_config;
    std::vector<ShadowCascade> m_cascades;
    std::vector<float> m_cascadeSplits;  // cascade boundary distances
    std::vector<PointLightShadow> m_pointShadows;

    // Cached camera data for cascade selection
    SoftVec3 m_cachedCameraPos = {};
    SoftVec3 m_cachedCameraFwd = {};
};

} // namespace WulfNet
