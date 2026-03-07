// =============================================================================
// WulfNet Engine - ForwardDecl.h
// =============================================================================
// Forward declarations of all major public types in the WulfNet engine.
// Include this header when you need to reference WulfNet types by pointer or
// reference without pulling in the full definitions.
//
// NOTE: enum / enum class types with explicit underlying types are declared
//       with their underlying type so they can be used in signatures.
//       Aligned structs (alignas) cannot carry alignment in a forward
//       declaration, so they are declared as plain struct here — include the
//       real header before creating instances.
// =============================================================================

#pragma once

#include <cstdint>

namespace WulfNet {

// =========================================================================
// Core - Engine
// =========================================================================
class  Engine;
struct EngineConfig;
enum class EngineInitResult;                   // Engine.h

// =========================================================================
// Core - Logging
// =========================================================================
enum class LogLevel : uint8_t;                 // Core/Logging/Logger.h
struct LogEntry;
class  ILogSink;
class  ConsoleLogSink;                         // : ILogSink
class  FileLogSink;                            // : ILogSink
class  CallbackLogSink;                        // : ILogSink
class  Logger;

// =========================================================================
// Core - System / Profiling
// =========================================================================
struct SystemStats;
class  SystemMonitor;
class  ScopedTimer;
class  ManualTimer;

// =========================================================================
// Core - Math
// =========================================================================
struct Vec2;
struct Vec3;
struct Vec4;                                   // alignas(16) in definition
struct Mat3;
struct Mat4;
struct Quat;
class  PerlinNoise;

// =========================================================================
// Physics - Integration (PhysicsWorld)
// =========================================================================
struct PhysicsWorldSettings;
struct ContactEvent;
class  PhysicsWorld;
class  MPMSystem;                              // fwd-declared in PhysicsWorld.h

// =========================================================================
// Physics - Fluids (SWE 2.5D)
// =========================================================================
struct FluidSystemConfig;
struct MacroTileID;
struct WaterStateSOA;
class  FluidSystem;

// =========================================================================
// Physics - Gaseous
// =========================================================================
struct GasCell;                                // alignas(16) in definition
struct GaseousSystemConfig;
enum class GasEmitterType : uint32_t;
struct GasEmitter;
struct GasObstacle;
struct GaseousStats;
class  GaseousSystem;

// =========================================================================
// Physics - Destruction
// =========================================================================
struct VoronoiCell;
struct FracturePattern;
struct DestructibleBody;
struct FractureEvent;
struct DestructionConfig;
struct DestructionStats;
class  DestructionSystem;

// =========================================================================
// Physics - Terrain Deformation
// =========================================================================
enum class TerrainMaterialType : uint32_t;
enum class StampShape : uint32_t;
struct TerrainMaterial;
struct DeformationStamp;
struct DeformationEvent;
struct TerrainDeformConfig;
struct TerrainDeformStats;
class  TerrainDeformation;

// =========================================================================
// Physics - MPM (Material Point Method)
// =========================================================================
enum class MPMMaterialType : uint32_t;
struct SVDResult;
struct MPMMaterialParams;
struct MPMParticle;                            // alignas(16) in definition
class  ConstitutiveModel;
class  NeoHookeanModel;                        // : ConstitutiveModel
class  DruckerPragerModel;                     // : ConstitutiveModel
class  SnowModel;                              // : ConstitutiveModel
class  ViscousFluidModel;                      // : ConstitutiveModel
struct MPMCouplingConfig;
struct MPMCouplingStats;
struct CoupledRigidBody;
class  MPMRigidCoupling;

// =========================================================================
// Rendering - Pipeline
// =========================================================================
struct RenderPipelineConfig;
struct RenderStats;
class  RenderPipeline;

// =========================================================================
// Rendering - Types
// =========================================================================
struct SoftColorRGBA8;
struct SoftVertex;
struct SoftMaterial;
struct SoftMesh;
struct SoftTransform;
struct SoftCamera;
struct SoftPointLight;
struct SoftDirectionalLight;
struct SoftTexture;

// =========================================================================
// Rendering - Software Rasterizer
// =========================================================================
struct SoftRasterizerConfig;
class  SoftwareRasterizer;
struct AABox;
struct OcclusionCullerConfig;
class  OcclusionCuller;
class  GBuffer;
struct DeferredShadingConfig;
class  DeferredShading;

// =========================================================================
// Rendering - Lighting / Shadows
// =========================================================================
struct ShadowCascadeConfig;
struct ShadowSystemConfig;
class  ShadowCascade;
class  PointLightShadow;
class  ShadowSystem;

// =========================================================================
// Rendering - Global Illumination
// =========================================================================
struct SSAOConfig;
struct IndirectLightConfig;
struct LightProbe;
struct GlobalIlluminationConfig;
class  GlobalIllumination;

// =========================================================================
// Rendering - Volumetric Effects
// =========================================================================
struct VolumeRegion;
struct EmissionKeyframe;
struct VolumetricConfig;
struct VolumeSampler;
struct VolumetricSample;
class  VolumetricRenderer;

// =========================================================================
// Audio - Core
// =========================================================================
enum class AudioSampleFormat;
struct AudioFormat;
class  AudioBuffer;
struct AudioSourceConfig;
class  AudioSource;
struct AudioListener;
struct AudioMixerConfig;
struct AudioMixerStats;
class  AudioMixer;

// =========================================================================
// Audio - Acoustics
// =========================================================================
struct AcousticMaterial;
struct AcousticRayHit;
struct ReflectionTap;
struct ImpulseResponse;
struct RoomEstimate;
struct AcousticConfig;
class  AcousticSystem;

// =========================================================================
// Audio - Spatial
// =========================================================================
enum class AttenuationModel : uint8_t;
struct AttenuationCurve;
struct HRTFParams;
struct HRTFResult;
struct AmbisonicsBFormat;
struct AmbisonicsSpeaker;
struct DopplerConfig;
class  SpatialAudio;

// =========================================================================
// Compute - Vulkan
// =========================================================================
struct GPUDeviceInfo;
struct VulkanContextSettings;
class  VulkanContext;

// =========================================================================
// Compute - Memory / Buffers
// =========================================================================
enum class GPUBufferUsage : uint32_t;
enum class GPUMemoryLocation;
class  GPUBufferBase;
template<typename T> class ComputeBuffer;      // : GPUBufferBase (class template)
struct ParticleData;                           // alignas(16) in definition

// =========================================================================
// Compute - Pipelines
// =========================================================================
enum class ShaderBindingType;
struct ShaderBinding;
struct PushConstantRange;
struct SpecializationConstant;
struct ComputePipelineDesc;
class  ComputePipeline;

// =========================================================================
// Compute - Reduction
// =========================================================================
enum class ReductionOp : uint32_t;
struct BoundsToTransformParams;
class  ParallelReduction;

// =========================================================================
// Compute - Fluids (GPU)
// =========================================================================
class  SWEComputeGPU;
class  JoltComputeAdapter;

// =========================================================================
// Procedural - IFS (Iterated Function Systems)
// =========================================================================
enum class IFSPreset;
struct ProceduralConfig;
class  TransformBlender;
struct IFSConfig;
struct IFSInitParams;
struct IFSChaosParams;
struct IFSIteratedParams;
struct IFSVoxelizeParams;
struct IFSClearParams;
struct IFSOcclusionParams;
struct IFSLODParams;
class  IFSSystem;
struct TransformInstructions;

} // namespace WulfNet
