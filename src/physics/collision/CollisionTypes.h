// =============================================================================
// WulfNet Engine - Collision Types
// =============================================================================
// Common types for collision detection system.
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/math/Vec3.h"

namespace WulfNet {

// =============================================================================
// Shape Type Enumeration
// =============================================================================

enum class ShapeType : u8 {
    Sphere = 0,
    Box,
    Capsule,
    Plane,
    ConvexHull,
    Mesh,
    Compound,
    Count
};

inline const char* shapeTypeToString(ShapeType type) {
    switch (type) {
        case ShapeType::Sphere:     return "Sphere";
        case ShapeType::Box:        return "Box";
        case ShapeType::Capsule:    return "Capsule";
        case ShapeType::Plane:      return "Plane";
        case ShapeType::ConvexHull: return "ConvexHull";
        case ShapeType::Mesh:       return "Mesh";
        case ShapeType::Compound:   return "Compound";
        default:                  return "Unknown";
    }
}

// =============================================================================
// Collision Layer (Bitmask)
// =============================================================================

enum class CollisionLayer : u32 {
    None       = 0,
    Default    = 1 << 0,
    Static     = 1 << 1,
    Dynamic    = 1 << 2,
    Kinematic  = 1 << 3,
    Trigger    = 1 << 4,
    Player     = 1 << 5,
    Enemy      = 1 << 6,
    Projectile = 1 << 7,
    Terrain    = 1 << 8,
    Water      = 1 << 9,
    All        = 0xFFFFFFFF
};

inline CollisionLayer operator|(CollisionLayer a, CollisionLayer b) {
    return static_cast<CollisionLayer>(static_cast<u32>(a) | static_cast<u32>(b));
}

inline CollisionLayer operator&(CollisionLayer a, CollisionLayer b) {
    return static_cast<CollisionLayer>(static_cast<u32>(a) & static_cast<u32>(b));
}

inline CollisionLayer operator~(CollisionLayer a) {
    return static_cast<CollisionLayer>(~static_cast<u32>(a));
}

inline bool hasLayer(CollisionLayer mask, CollisionLayer layer) {
    return (static_cast<u32>(mask) & static_cast<u32>(layer)) != 0;
}

inline bool canLayersCollide(CollisionLayer a, CollisionLayer b) {
    // For now, any overlapping bits means layers can collide
    return (static_cast<u32>(a) & static_cast<u32>(b)) != 0;
}

// =============================================================================
// Raycast Hit Result
// =============================================================================

struct RaycastHit {
    Vec3 point;           // World-space hit point
    Vec3 normal;          // Surface normal at hit
    f32 distance = 0.0f;  // Distance from ray origin
    u32 bodyId = 0;       // ID of hit body
    bool hit = false;     // Whether ray hit anything
};

// =============================================================================
// Contact Point
// =============================================================================

struct ContactPoint {
    Vec3 positionWorldA;  // Contact point on body A in world space
    Vec3 positionWorldB;  // Contact point on body B in world space
    Vec3 normal;          // Contact normal (from A to B)
    f32 penetration;      // Penetration depth (negative = separated)
    f32 normalImpulse = 0.0f;   // Cached normal impulse for warm-starting
    f32 tangentImpulse = 0.0f;  // Cached tangent impulse magnitude
    Vec3 tangent = Vec3::zero();
};

// =============================================================================
// Contact Manifold
// =============================================================================

struct ContactManifold {
    static constexpr u32 MAX_CONTACTS = 4;
    
    ContactPoint contacts[MAX_CONTACTS];
    u32 contactCount = 0;
    u32 bodyIdA = 0;
    u32 bodyIdB = 0;
    
    void addContact(const ContactPoint& cp) {
        if (contactCount < MAX_CONTACTS) {
            contacts[contactCount++] = cp;
        }
    }
    
    void clear() {
        contactCount = 0;
    }
};

// =============================================================================
// Collision Pair (for broadphase)
// =============================================================================

struct CollisionPair {
    u32 bodyIdA = 0;
    u32 bodyIdB = 0;
    bool isNew = false;   // True if pair was just created this frame
    bool isPersistent = false; // True if pair existed last frame
    
    bool operator==(const CollisionPair& other) const {
        return (bodyIdA == other.bodyIdA && bodyIdB == other.bodyIdB) ||
               (bodyIdA == other.bodyIdB && bodyIdB == other.bodyIdA);
    }
};

} // namespace WulfNet
