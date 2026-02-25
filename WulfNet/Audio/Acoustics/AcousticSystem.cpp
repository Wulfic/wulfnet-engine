// =============================================================================
// WulfNet Engine - Acoustic Simulation Implementation
// =============================================================================

#include "WulfNet/Audio/Acoustics/AcousticSystem.h"
#include <cstring>
#include <algorithm>
#include <numeric>

namespace WulfNet {

// =============================================================================
// Acoustic Material Presets
// =============================================================================

// Absorption values per band: [125, 250, 500, 1k, 2k, 4k] Hz
AcousticMaterial AcousticMaterial::Concrete() {
    AcousticMaterial m;
    m.absorption[0] = 0.01f; m.absorption[1] = 0.01f; m.absorption[2] = 0.02f;
    m.absorption[3] = 0.02f; m.absorption[4] = 0.02f; m.absorption[5] = 0.03f;
    m.scattering = 0.1f;
    m.transmission = 0.0f;
    return m;
}

AcousticMaterial AcousticMaterial::Wood() {
    AcousticMaterial m;
    m.absorption[0] = 0.15f; m.absorption[1] = 0.11f; m.absorption[2] = 0.10f;
    m.absorption[3] = 0.07f; m.absorption[4] = 0.06f; m.absorption[5] = 0.07f;
    m.scattering = 0.3f;
    m.transmission = 0.01f;
    return m;
}

AcousticMaterial AcousticMaterial::Glass() {
    AcousticMaterial m;
    m.absorption[0] = 0.35f; m.absorption[1] = 0.25f; m.absorption[2] = 0.18f;
    m.absorption[3] = 0.12f; m.absorption[4] = 0.07f; m.absorption[5] = 0.04f;
    m.scattering = 0.05f;
    m.transmission = 0.1f;
    return m;
}

AcousticMaterial AcousticMaterial::Carpet() {
    AcousticMaterial m;
    m.absorption[0] = 0.01f; m.absorption[1] = 0.05f; m.absorption[2] = 0.10f;
    m.absorption[3] = 0.20f; m.absorption[4] = 0.45f; m.absorption[5] = 0.65f;
    m.scattering = 0.7f;
    m.transmission = 0.0f;
    return m;
}

AcousticMaterial AcousticMaterial::Curtain() {
    AcousticMaterial m;
    m.absorption[0] = 0.07f; m.absorption[1] = 0.31f; m.absorption[2] = 0.49f;
    m.absorption[3] = 0.75f; m.absorption[4] = 0.70f; m.absorption[5] = 0.60f;
    m.scattering = 0.8f;
    m.transmission = 0.15f;
    return m;
}

AcousticMaterial AcousticMaterial::Metal() {
    AcousticMaterial m;
    m.absorption[0] = 0.04f; m.absorption[1] = 0.04f; m.absorption[2] = 0.03f;
    m.absorption[3] = 0.03f; m.absorption[4] = 0.03f; m.absorption[5] = 0.03f;
    m.scattering = 0.05f;
    m.transmission = 0.0f;
    return m;
}

const AcousticMaterial AcousticSystem::kDefaultMaterial = AcousticMaterial::Concrete();

// =============================================================================
// Helpers
// =============================================================================

static constexpr float kAcPi = 3.14159265358979323846f;
static constexpr float kGoldenRatio = 1.61803398874989484820f;

// =============================================================================
// AcousticSystem
// =============================================================================

bool AcousticSystem::Initialize(const AcousticConfig& config) {
    if (config.maxBounces < 0 || config.numRays <= 0) return false;
    if (config.maxDistance <= 0.0f || config.speedOfSound <= 0.0f) return false;

    m_config = config;
    m_materials.clear();
    m_initialized = true;
    return true;
}

void AcousticSystem::Shutdown() {
    m_materials.clear();
    m_rayCast = nullptr;
    m_initialized = false;
}

void AcousticSystem::SetRayCastFunction(AcousticRayCastFn fn) {
    m_rayCast = std::move(fn);
}

int AcousticSystem::AddMaterial(const AcousticMaterial& material) {
    m_materials.push_back(material);
    return static_cast<int>(m_materials.size()) - 1;
}

const AcousticMaterial& AcousticSystem::GetMaterial(int id) const {
    if (id < 0 || id >= static_cast<int>(m_materials.size())) return kDefaultMaterial;
    return m_materials[id];
}

// =============================================================================
// Occlusion / Obstruction
// =============================================================================

float AcousticSystem::ComputeOcclusion(float srcX, float srcY, float srcZ,
                                        float lstX, float lstY, float lstZ) const {
    if (!m_rayCast) return 1.0f; // No geometry = fully clear

    float dx = lstX - srcX;
    float dy = lstY - srcY;
    float dz = lstZ - srcZ;
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (dist < 1e-6f) return 1.0f;

    float invDist = 1.0f / dist;
    dx *= invDist; dy *= invDist; dz *= invDist;

    AcousticRayHit hit = m_rayCast(srcX, srcY, srcZ, dx, dy, dz, dist);

    if (hit.hit && hit.distance < dist - 0.01f) {
        // Ray blocked — compute transmission through material
        const AcousticMaterial& mat = GetMaterial(hit.materialId);
        return mat.transmission; // 0.0 = fully occluded, higher = partial
    }

    return 1.0f; // Clear line of sight
}

float AcousticSystem::ComputeObstruction(float srcX, float srcY, float srcZ,
                                          float lstX, float lstY, float lstZ,
                                          int numSampleRays) const {
    if (!m_rayCast) return 1.0f;
    if (numSampleRays <= 0) return 1.0f;

    float dx = lstX - srcX;
    float dy = lstY - srcY;
    float dz = lstZ - srcZ;
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < 1e-6f) return 1.0f;

    // Cast multiple rays around the direct path with small offsets
    float invDist = 1.0f / dist;
    float ndx = dx * invDist, ndy = dy * invDist, ndz = dz * invDist;

    int clearRays = 0;

    for (int i = 0; i < numSampleRays; ++i) {
        // Generate a small random cone offset around the direct path
        float rx, ry, rz;
        GenerateRayDirection(i, numSampleRays, rx, ry, rz);

        // Blend mostly toward the true direction
        float blend = 0.05f; // Small deviation
        float sx = ndx + rx * blend;
        float sy = ndy + ry * blend;
        float sz = ndz + rz * blend;

        // Normalize
        float len = std::sqrt(sx * sx + sy * sy + sz * sz);
        if (len < 1e-6f) continue;
        sx /= len; sy /= len; sz /= len;

        AcousticRayHit hit = m_rayCast(srcX, srcY, srcZ, sx, sy, sz, dist * 1.1f);

        if (!hit.hit || hit.distance >= dist - 0.01f) {
            clearRays++;
        }
    }

    return static_cast<float>(clearRays) / static_cast<float>(numSampleRays);
}

// =============================================================================
// Impulse Response
// =============================================================================

ImpulseResponse AcousticSystem::TraceImpulseResponse(float srcX, float srcY, float srcZ,
                                                      float lstX, float lstY, float lstZ) const {
    ImpulseResponse ir;

    if (!m_initialized) return ir;

    float dist = std::sqrt((lstX - srcX) * (lstX - srcX) +
                           (lstY - srcY) * (lstY - srcY) +
                           (lstZ - srcZ) * (lstZ - srcZ));

    // Direct path
    ir.directTime = dist / m_config.speedOfSound;
    ir.directEnergy = DistanceAttenuation(dist) * AirAbsorption(dist);

    // Check direct path occlusion
    float occ = ComputeOcclusion(srcX, srcY, srcZ, lstX, lstY, lstZ);
    ir.directOccluded = (occ < 0.5f);
    ir.directEnergy *= occ;

    if (!m_rayCast) {
        // Without geometry, only direct path is meaningful
        return ir;
    }

    // Cast rays from source and trace bouncing reflections
    for (int r = 0; r < m_config.numRays; ++r) {
        float dx, dy, dz;
        GenerateRayDirection(r, m_config.numRays, dx, dy, dz);

        float energy = 1.0f;
        float pathLength = 0.0f;
        float ox = srcX, oy = srcY, oz = srcZ;

        for (int bounce = 0; bounce < m_config.maxBounces && energy > m_config.energyThreshold; ++bounce) {
            float remaining = m_config.maxDistance - pathLength;
            if (remaining <= 0.0f) break;

            AcousticRayHit hit = m_rayCast(ox, oy, oz, dx, dy, dz, remaining);

            if (!hit.hit) break; // Ray escaped the scene

            pathLength += hit.distance;

            // Apply absorption from surface material
            const AcousticMaterial& mat = GetMaterial(hit.materialId);

            // Average absorption across all bands (simplified)
            float avgAbsorb = 0.0f;
            for (int b = 0; b < AcousticMaterial::kNumBands; ++b) {
                avgAbsorb += mat.absorption[b];
            }
            avgAbsorb /= static_cast<float>(AcousticMaterial::kNumBands);

            energy *= (1.0f - avgAbsorb); // Reflect remaining energy
            energy *= AirAbsorption(hit.distance);

            // Check if this reflected ray reaches the listener
            // (within a reception sphere around the listener)
            float hitX = ox + dx * hit.distance;
            float hitY = oy + dy * hit.distance;
            float hitZ = oz + dz * hit.distance;

            float toListX = lstX - hitX;
            float toListY = lstY - hitY;
            float toListZ = lstZ - hitZ;
            float toListDist = std::sqrt(toListX * toListX + toListY * toListY + toListZ * toListZ);

            // Reception check: does a straight line from hit point reach listener?
            if (toListDist < m_config.maxDistance * 0.5f) {
                float invLD = (toListDist > 1e-6f) ? (1.0f / toListDist) : 0.0f;
                float ldx = toListX * invLD;
                float ldy = toListY * invLD;
                float ldz = toListZ * invLD;

                AcousticRayHit listenerCheck = m_rayCast(hitX, hitY, hitZ, ldx, ldy, ldz, toListDist);
                bool reachesListener = (!listenerCheck.hit || listenerCheck.distance >= toListDist - 0.01f);

                if (reachesListener) {
                    float totalDist = pathLength + toListDist;
                    float arrivalTime = totalDist / m_config.speedOfSound;
                    float tapEnergy = energy * DistanceAttenuation(toListDist);

                    ReflectionTap tap;
                    tap.time = arrivalTime;
                    tap.energy = tapEnergy;
                    tap.direction[0] = -ldx;
                    tap.direction[1] = -ldy;
                    tap.direction[2] = -ldz;
                    tap.bounces = bounce + 1;
                    ir.taps.push_back(tap);
                }
            }

            // Reflect ray off surface
            Reflect(dx, dy, dz, hit.normalX, hit.normalY, hit.normalZ, dx, dy, dz);

            // Advance origin to hit point (with small offset along normal to prevent self-intersection)
            ox = ox + (dx * hit.distance) + hit.normalX * 0.001f;
            // Fix: recalculate properly
            ox = hitX + hit.normalX * 0.001f;
            oy = hitY + hit.normalY * 0.001f;
            oz = hitZ + hit.normalZ * 0.001f;
        }
    }

    // Classify early vs late reflections
    for (const auto& tap : ir.taps) {
        if (tap.time < m_config.earlyLateBoundary) {
            ir.earlyEnergy += tap.energy;
        } else {
            ir.lateEnergy += tap.energy;
        }
    }

    // Normalize energies by ray count for consistent results
    float invRays = 1.0f / static_cast<float>(m_config.numRays);
    ir.earlyEnergy *= invRays;
    ir.lateEnergy *= invRays;

    return ir;
}

// =============================================================================
// Room Estimation
// =============================================================================

RoomEstimate AcousticSystem::EstimateRoom(float posX, float posY, float posZ) const {
    RoomEstimate room;
    if (!m_rayCast || !m_initialized) return room;

    int numRays = m_config.roomProbeRays;
    if (numRays <= 0) return room;

    float totalDist = 0.0f;
    float totalAbsorption = 0.0f;
    int hitCount = 0;

    for (int r = 0; r < numRays; ++r) {
        float dx, dy, dz;
        GenerateRayDirection(r, numRays, dx, dy, dz);

        AcousticRayHit hit = m_rayCast(posX, posY, posZ, dx, dy, dz, m_config.maxDistance);

        if (hit.hit) {
            totalDist += hit.distance;
            hitCount++;

            // Accumulate absorption
            const AcousticMaterial& mat = GetMaterial(hit.materialId);
            float avgAbsorb = 0.0f;
            for (int b = 0; b < AcousticMaterial::kNumBands; ++b) {
                avgAbsorb += mat.absorption[b];
            }
            avgAbsorb /= static_cast<float>(AcousticMaterial::kNumBands);
            totalAbsorption += avgAbsorb;
        }
    }

    if (hitCount < 3) return room; // Not enough geometry to estimate

    room.meanFreePath = totalDist / static_cast<float>(hitCount);
    room.avgAbsorption = totalAbsorption / static_cast<float>(hitCount);

    // Estimate volume from mean free path: V ≈ (4/3) * π * (meanFreePath)³ * 0.25
    // More accurately: for a convex room, V ≈ S * L / 4 where L = mean free path
    // We estimate surface area from the hit distribution: S ≈ N * 4π * meanFreePath²
    // Then V = S * L / 4
    // Simplified approach using mean free path relationship: V = (S * L) / 4
    // and L = 4V/S → S = 4V/L → V = S*L/4 → V = L³ * 4π/3 (sphere approximation)
    float r = room.meanFreePath;
    room.volume = (4.0f / 3.0f) * kAcPi * r * r * r;
    room.surfaceArea = 4.0f * room.volume / room.meanFreePath; // S = 4V/L

    room.rt60 = ComputeRT60_Sabine(room.volume, room.surfaceArea, room.avgAbsorption);

    return room;
}

float AcousticSystem::ComputeRT60_Sabine(float volume, float surfaceArea, float avgAbsorption) {
    if (surfaceArea <= 0.0f || avgAbsorption <= 0.0f) return 0.0f;
    // Sabine equation: RT60 = 0.161 * V / (S * α)
    return 0.161f * volume / (surfaceArea * avgAbsorption);
}

float AcousticSystem::ComputeRT60_Eyring(float volume, float surfaceArea, float avgAbsorption) {
    if (surfaceArea <= 0.0f || avgAbsorption <= 0.0f || avgAbsorption >= 1.0f) return 0.0f;
    // Eyring equation: RT60 = 0.161 * V / (-S * ln(1 - α))
    return 0.161f * volume / (-surfaceArea * std::log(1.0f - avgAbsorption));
}

// =============================================================================
// Distance Attenuation
// =============================================================================

float AcousticSystem::DistanceAttenuation(float distance, float refDistance,
                                           float maxDistance, float rolloff) {
    if (distance <= refDistance) return 1.0f;
    if (distance >= maxDistance) return 0.0f;

    // Inverse-distance law with configurable rolloff exponent
    float atten = std::pow(refDistance / distance, rolloff);
    return std::max(0.0f, std::min(1.0f, atten));
}

float AcousticSystem::AirAbsorption(float distance) const {
    // Simple exponential air absorption (more realistic at high frequencies)
    return std::exp(-m_config.airAbsorption * distance);
}

// =============================================================================
// Impulse Response to AudioBuffer
// =============================================================================

AudioBuffer ImpulseResponse::ToAudioBuffer(int sampleRate, float durationSec) const {
    AudioBuffer buf;
    AudioFormat fmt;
    fmt.sampleRate = sampleRate;
    fmt.channels = 1;
    fmt.format = AudioSampleFormat::Float32;

    int frames = static_cast<int>(durationSec * sampleRate);
    if (frames <= 0) return buf;

    buf.Initialize(fmt, frames);
    float* data = buf.GetData();

    // Place direct sound
    int directSample = static_cast<int>(directTime * sampleRate);
    if (directSample >= 0 && directSample < frames) {
        data[directSample] += directEnergy;
    }

    // Place reflection taps
    for (const auto& tap : taps) {
        int sample = static_cast<int>(tap.time * sampleRate);
        if (sample >= 0 && sample < frames) {
            data[sample] += tap.energy;
        }
    }

    return buf;
}

// =============================================================================
// Private helpers
// =============================================================================

void AcousticSystem::GenerateRayDirection(int index, int total, float& dx, float& dy, float& dz) const {
    // Fibonacci sphere: uniform distribution on unit sphere (deterministic)
    float y = 1.0f - (2.0f * static_cast<float>(index) + 1.0f) / static_cast<float>(total);
    float radius = std::sqrt(1.0f - y * y);
    float theta = 2.0f * kAcPi * static_cast<float>(index) / kGoldenRatio;

    dx = radius * std::cos(theta);
    dy = y;
    dz = radius * std::sin(theta);
}

void AcousticSystem::Reflect(float dx, float dy, float dz,
                              float nx, float ny, float nz,
                              float& rx, float& ry, float& rz) {
    // r = d - 2(d·n)n
    float dot = dx * nx + dy * ny + dz * nz;
    rx = dx - 2.0f * dot * nx;
    ry = dy - 2.0f * dot * ny;
    rz = dz - 2.0f * dot * nz;
}

} // namespace WulfNet
