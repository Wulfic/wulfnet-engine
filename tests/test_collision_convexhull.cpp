#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "physics/collision/CollisionShape.h"
#include "core/math/Vec3.h"
#include <vector>

using namespace WulfNet;

TEST_CASE("ConvexHullShape properties and support function", "[physics][collision][convexhull]") {
    // Define vertices for a simple cube centered at origin with side length 2
    std::vector<Vec3> vertices = {
        Vec3(-1.0f, -1.0f, -1.0f), Vec3(1.0f, -1.0f, -1.0f),
        Vec3(1.0f,  1.0f, -1.0f), Vec3(-1.0f,  1.0f, -1.0f),
        Vec3(-1.0f, -1.0f,  1.0f), Vec3(1.0f, -1.0f,  1.0f),
        Vec3(1.0f,  1.0f,  1.0f), Vec3(-1.0f,  1.0f,  1.0f)
    };
    
    ConvexHullShape hull(vertices);
    
    SECTION("Type is correctly identified") {
        REQUIRE(hull.getType() == ShapeType::ConvexHull);
    }
    
    SECTION("Local AABB is computed correctly") {
        AABB aabb = hull.getLocalAABB();
        
        REQUIRE_THAT(aabb.min.x, Catch::Matchers::WithinRel(-1.0f));
        REQUIRE_THAT(aabb.min.y, Catch::Matchers::WithinRel(-1.0f));
        REQUIRE_THAT(aabb.min.z, Catch::Matchers::WithinRel(-1.0f));
        
        REQUIRE_THAT(aabb.max.x, Catch::Matchers::WithinRel(1.0f));
        REQUIRE_THAT(aabb.max.y, Catch::Matchers::WithinRel(1.0f));
        REQUIRE_THAT(aabb.max.z, Catch::Matchers::WithinRel(1.0f));
    }
    
    SECTION("Bounding Sphere is computed correctly") {
        BoundingSphere sphere = hull.getLocalBoundingSphere();
        // Distance from center (0,0,0) to corner (1,1,1) is sqrt(1+1+1) = sqrt(3) ~= 1.732
        REQUIRE_THAT(sphere.center.length(), Catch::Matchers::WithinAbs(0.0f, 0.001f));
        REQUIRE_THAT(sphere.radius, Catch::Matchers::WithinRel(1.73205f, 0.001f));
    }
    
    SECTION("Support function returns extremal points") {
        // Direction +X: Should return a vertex with x=1
        Vec3 supX = hull.support(Vec3(1, 0, 0));
        REQUIRE_THAT(supX.x, Catch::Matchers::WithinRel(1.0f));
        
        // Direction -X: Should return a vertex with x=-1
        Vec3 supNegX = hull.support(Vec3(-1, 0, 0));
        REQUIRE_THAT(supNegX.x, Catch::Matchers::WithinRel(-1.0f));
        
        // Direction +Y: y=1
        Vec3 supY = hull.support(Vec3(0, 1, 0));
        REQUIRE_THAT(supY.y, Catch::Matchers::WithinRel(1.0f));
        
        // Diagonal direction (1, 1, 1): Should return (1, 1, 1)
        Vec3 supDiag = hull.support(Vec3(1, 1, 1));
        REQUIRE_THAT(supDiag.x, Catch::Matchers::WithinRel(1.0f));
        REQUIRE_THAT(supDiag.y, Catch::Matchers::WithinRel(1.0f));
        REQUIRE_THAT(supDiag.z, Catch::Matchers::WithinRel(1.0f));
    }
}
