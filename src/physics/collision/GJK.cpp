// =============================================================================
// WulfNet Engine - GJK/EPA Collision Detection Implementation
// =============================================================================

#include "GJK.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace WulfNet {

// Helper struct for EPA faces
struct EPAFace {
    int ids[3]; // Indices into polytope vertices
    Vec3 normal;
    f32 distance;
};

SupportPoint GJK::support(const RigidBody* bodyA, const CollisionShape* shapeA,
                          const RigidBody* bodyB, const CollisionShape* shapeB,
                          Vec3 dir) {
    if (dir.lengthSq() < Math::EPSILON) {
        dir = Vec3(1, 0, 0);
    }
    
    // Support point A in direction dir
    Vec3 dirA = bodyA->getTransform().inverseTransformDirection(dir);
    Vec3 pA = shapeA->support(dirA);
    Vec3 worldPA = bodyA->getTransform().transformPoint(pA);
    
    // Support point B in direction -dir
    Vec3 dirB = bodyB->getTransform().inverseTransformDirection(-dir);
    Vec3 pB = shapeB->support(dirB);
    Vec3 worldPB = bodyB->getTransform().transformPoint(pB);
    
    return SupportPoint(worldPA - worldPB, worldPA, worldPB);
}

bool GJK::intersect(const RigidBody* bodyA, const CollisionShape* shapeA,
                    const RigidBody* bodyB, const CollisionShape* shapeB,
                    Simplex* outSimplex) {
    // Initial direction: Center to Center
    Vec3 direction = bodyB->getPosition() - bodyA->getPosition();
    if (direction.lengthSq() < Math::EPSILON) {
        direction = Vec3(1, 0, 0); 
    }
    
    Simplex simplex;
    SupportPoint S = support(bodyA, shapeA, bodyB, shapeB, direction);
    simplex.push_front(S);
    
    direction = -S.p; // Next direction towards origin
    
    // Max iterations to prevent infinite loops (collision usually found in <10 iterations)
    for (int i = 0; i < 64; ++i) {
        SupportPoint A = support(bodyA, shapeA, bodyB, shapeB, direction);
        
        if (A.p.dot(direction) < 0) {
            return false; // Did not cross origin, no intersection
        }
        
        simplex.push_front(A);
        
        if (handleSimplex(simplex, direction)) {
            if (outSimplex) {
                *outSimplex = simplex;
            }
            return true;
        }
        
        // Normalize direction to preven precision issues
        if (direction.lengthSq() > Math::EPSILON) {
            direction = direction.normalized();
        }
    }
    
    return false;
}

bool GJK::handleSimplex(Simplex& simplex, Vec3& direction) {
    switch (simplex.size) {
        case 2: return lineCase(simplex, direction);
        case 3: return triangleCase(simplex, direction);
        case 4: return tetrahedronCase(simplex, direction);
        default: return false; // Should not happen
    }
}

bool GJK::lineCase(Simplex& simplex, Vec3& direction) {
    Vec3 A = simplex[0].p;
    Vec3 B = simplex[1].p;
    
    Vec3 AB = B - A;
    Vec3 AO = -A;
    
    if (AB.dot(AO) > 0) {
        // Origin is in AB region
        // Direction is perpendicular to AB pointing towards Origin
        // (AB x AO) x AB
        direction = AB.cross(AO).cross(AB);
        // Simplex stays {A, B}
    } else {
        // Origin is in A region
        simplex.size = 1; // Discard B
        simplex[0] = simplex[0]; 
        direction = AO;
    }
    
    return false;
}

bool GJK::triangleCase(Simplex& simplex, Vec3& direction) {
    Vec3 A = simplex[0].p;
    Vec3 B = simplex[1].p;
    Vec3 C = simplex[2].p;
    
    Vec3 AB = B - A;
    Vec3 AC = C - A;
    Vec3 AO = -A;
    
    Vec3 ABC = AB.cross(AC);
    
    // AB x ABC points AWAY from C in the plane of the triangle
    Vec3 ABPerp = AB.cross(ABC);
    // ABC x AC points AWAY from B in the plane of the triangle
    Vec3 ACPerp = ABC.cross(AC);
    
    if (ABPerp.dot(AO) > 0) {
        // Outside AB
        if (AB.dot(AO) > 0) {
            simplex.size = 2; // Keep A, B
            simplex[0] = simplex[0]; // A
            simplex[1] = simplex[1]; // B
            direction = AB.cross(AO).cross(AB);
            return false;
        } else {
             // Region A
             simplex.size = 1;
             simplex[0] = simplex[0]; // A
             direction = AO;
             return false;
        }
    }
    
    if (ACPerp.dot(AO) > 0) {
        // Outside AC
        if (AC.dot(AO) > 0) {
            simplex.size = 2;
            simplex[0] = simplex[0]; // A
            simplex[1] = simplex[2]; // C
            direction = AC.cross(AO).cross(AC);
            return false;
        } else {
            // Region A
            simplex.size = 1;
            simplex[0] = simplex[0]; // A
            direction = AO;
            return false;
        }
    }
    
    // Inside triangle, check if above or below
    if (ABC.dot(AO) > 0) {
        direction = ABC;
    } else {
        // Flip winding so that normal points towards origin
        SupportPoint temp = simplex[1];
        simplex[1] = simplex[2];
        simplex[2] = temp;
        // Now {A, C, B}
        direction = -ABC;
    }
    
    return false;
}

bool GJK::tetrahedronCase(Simplex& simplex, Vec3& direction) {
    Vec3 A = simplex[0].p;
    Vec3 B = simplex[1].p;
    Vec3 C = simplex[2].p;
    Vec3 D = simplex[3].p;
    
    Vec3 AB = B - A;
    Vec3 AC = C - A;
    Vec3 AD = D - A;
    Vec3 AO = -A;
    
    Vec3 ABC = AB.cross(AC);
    Vec3 ACD = AC.cross(AD);
    Vec3 ADB = AD.cross(AB);
    
    // Ensure normals point outward (away from 4th vertex)
    if (ABC.dot(AD) > 0) ABC = -ABC;
    if (ACD.dot(AB) > 0) ACD = -ACD;
    if (ADB.dot(AC) > 0) ADB = -ADB;
    
    if (ABC.dot(AO) > 0) {
        // Outside ABC
        simplex.size = 3;
        simplex[0] = simplex[0]; // A
        simplex[1] = simplex[1]; // B
        simplex[2] = simplex[2]; // C
        return triangleCase(simplex, direction);
    }
    
    if (ACD.dot(AO) > 0) {
        simplex.size = 3;
        simplex[0] = simplex[0]; // A
        simplex[1] = simplex[2]; // C
        simplex[2] = simplex[3]; // D
        return triangleCase(simplex, direction);
    }
    
    if (ADB.dot(AO) > 0) {
        simplex.size = 3;
        simplex[0] = simplex[0]; // A
        simplex[1] = simplex[3]; // D
        simplex[2] = simplex[1]; // B
        return triangleCase(simplex, direction);
    }
    
    return true; // Origin inside tetrahedron
}

// Helper to add faces to polytope
void addFace(std::vector<EPAFace>& faces, int a, int b, int c, const std::vector<SupportPoint>& vertices) {
    EPAFace face;
    face.ids[0] = a;
    face.ids[1] = b;
    face.ids[2] = c;
    
    Vec3 pA = vertices[a].p;
    Vec3 pB = vertices[b].p;
    Vec3 pC = vertices[c].p;
    
    face.normal = (pB - pA).cross(pC - pA).normalized();
    face.distance = face.normal.dot(pA);
    
    // Ensure normal points away from origin
    // Since origin is inside the polytope, the dot product of normal and point on face should be positive
    // if the normal points away from origin.
    if (face.distance < 0) {
        face.normal = -face.normal;
        face.distance = -face.distance;
        // Swap winding to maintain consistency
        int temp = face.ids[1];
        face.ids[1] = face.ids[2];
        face.ids[2] = temp;
    }
    
    faces.push_back(face);
}

struct Edge {
    int a, b;
    bool operator==(const Edge& other) const { return a == other.a && b == other.b; }
};

bool GJK::computePenetration(const RigidBody* bodyA, const CollisionShape* shapeA,
                           const RigidBody* bodyB, const CollisionShape* shapeB,
                           Vec3& outNormal, f32& outDepth, Vec3& outContactPointA, Vec3& outContactPointB)
{
    Simplex simplex;
    if (!intersect(bodyA, shapeA, bodyB, shapeB, &simplex)) {
        return false;
    }

    // Expand simplex to tetrahedron if necessary (handle degenerate cases)
    if (simplex.size == 1) {
         simplex.push_front(support(bodyA, shapeA, bodyB, shapeB, Vec3(1,0,0)));
    }
    if (simplex.size == 2) {
         Vec3 v = simplex[1].p - simplex[0].p;
         Vec3 n = v.cross(Vec3(1,0,0));
         if (n.lengthSq() < 0.001f) n = v.cross(Vec3(0,1,0));
         simplex.push_front(support(bodyA, shapeA, bodyB, shapeB, n));
    }
    if (simplex.size == 3) {
         Vec3 n = (simplex[1].p - simplex[0].p).cross(simplex[2].p - simplex[0].p);
         if (n.lengthSq() < 0.001f) n = Vec3(1,0,0); 
         simplex.push_front(support(bodyA, shapeA, bodyB, shapeB, n));
    }
    // Ensure size is 4 now
    if (simplex.size < 4) {
         // Should not happen unless support failing repeatedly
         return false; 
    }

    // Convert Simplex to Polytope
    std::vector<SupportPoint> vertices;
    std::vector<EPAFace> faces;
    
    for(int i=0; i<simplex.size; ++i) {
        vertices.push_back(simplex[i]);
    }

    addFace(faces, 0, 1, 2, vertices);
    addFace(faces, 0, 2, 3, vertices);
    addFace(faces, 0, 3, 1, vertices);
    addFace(faces, 1, 3, 2, vertices);
    
    const int MAX_ITERATIONS = 64;
    const f32 TOLERANCE = 0.0001f;
    
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        // Find closest face
        f32 minDist = std::numeric_limits<f32>::max();
        int closestFaceIdx = -1;
        
        for(size_t i=0; i<faces.size(); ++i) {
            if (faces[i].distance < minDist) {
                minDist = faces[i].distance;
                closestFaceIdx = (int)i;
            }
        }
        
        if (closestFaceIdx == -1) break;
        
        const EPAFace& closestFace = faces[closestFaceIdx];
        
        // Search direction is face normal
        SupportPoint p = support(bodyA, shapeA, bodyB, shapeB, closestFace.normal);
        
        f32 d = p.p.dot(closestFace.normal);
        
        if (d - closestFace.distance < TOLERANCE) {
            // Converged
            outNormal = -closestFace.normal; // Normal points OUT of MD, we want separation direction (into MD/Origin)
            outDepth = d;
            
            // Calculate contact points using Barycentric coordinates
            const EPAFace& f = closestFace;
            // The contact point on the MD boundary is not simply (p . normal) * normal
            // That is the projection of Origin onto the plane. And since Origin is inside MD,
            // and we found the closest face of MD to Origin, the closest point on the MD boundary IS the projection.
            // So yes, pointOnMD = normal * distance.
            
            Vec3 pointOnMD = closestFace.normal * closestFace.distance;
            
            // Barycentric coordinates for pointOnMD on triangle (vertices[f.ids[0/1/2]])
            Vec3 a = vertices[f.ids[0]].p;
            Vec3 b = vertices[f.ids[1]].p;
            Vec3 c = vertices[f.ids[2]].p;
            
            // Cramers rule or similar for Barycentric coords
            Vec3 v0 = b - a, v1 = c - a, v2 = pointOnMD - a;
            float d00 = v0.dot(v0);
            float d01 = v0.dot(v1);
            float d11 = v1.dot(v1);
            float d20 = v2.dot(v0);
            float d21 = v2.dot(v1);
            float denom = d00 * d11 - d01 * d01;
            
            if (std::abs(denom) < Math::EPSILON) {
                 // Degenerate triangle
                 outContactPointA = vertices[f.ids[0]].a;
                 outContactPointB = vertices[f.ids[0]].b;
                 return true;
            }
            
            float v = (d11 * d20 - d01 * d21) / denom;
            float w = (d00 * d21 - d01 * d20) / denom;
            float u = 1.0f - v - w;
            
            outContactPointA = vertices[f.ids[0]].a * u + vertices[f.ids[1]].a * v + vertices[f.ids[2]].a * w;
            outContactPointB = vertices[f.ids[0]].b * u + vertices[f.ids[1]].b * v + vertices[f.ids[2]].b * w;
            
            return true;
        }
        
        // Expand Polytope
        std::vector<EPAFace> newFaces;
        std::vector<Edge> horizon;
        
        for(size_t i=0; i<faces.size(); ++i) {
            // Check visibility: Is point p in front of face plane?
            if (faces[i].normal.dot(p.p - vertices[faces[i].ids[0]].p) > 0) {
                // Face is visible, remove it and add edges to horizon candidate list
                Edge edges[3] = {
                    {faces[i].ids[0], faces[i].ids[1]},
                    {faces[i].ids[1], faces[i].ids[2]},
                    {faces[i].ids[2], faces[i].ids[0]}
                };
                
                for(int j=0; j<3; ++j) {
                    // Check if inverse edge is already in horizon?
                    // Actually, simpler logic: Add all edges of removed faces to a list.
                    // Edges that appear exactly once are the horizon.
                    // Edges that appear twice (once in each direction, shared by 2 removed faces) are interior to the removed patch.
                    // We need to properly dedup.
                    // Or search existing horizon for opposite edge.
                    
                    bool foundReverse = false;
                    for(auto it = horizon.begin(); it != horizon.end(); ++it) {
                        if (it->b == edges[j].a && it->a == edges[j].b) {
                            // Found shared edge, remove both
                            horizon.erase(it);
                            foundReverse = true;
                            break;
                        }
                    }
                    if (!foundReverse) {
                        horizon.push_back(edges[j]);
                    }
                }
            } else {
                newFaces.push_back(faces[i]);
            }
        }
        
        vertices.push_back(p);
        int newIdx = (int)vertices.size() - 1;
        
        for (const auto& edge : horizon) {
             addFace(newFaces, edge.a, edge.b, newIdx, vertices);
        }
        
        faces = newFaces;
    }
    
    // Fallback if max iterations reached
    if (!faces.empty()) {
         outNormal = -faces[0].normal;
         outDepth = faces[0].distance;
         outContactPointA = vertices[faces[0].ids[0]].a;
         outContactPointB = vertices[faces[0].ids[0]].b;
         return true;
    }
    
    return true;
}

} // namespace WulfNet
