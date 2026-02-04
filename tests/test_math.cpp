// =============================================================================
// WulfNet Engine - Math Tests
// =============================================================================

#include <cassert>
#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "core/math/Math.h"

using namespace WulfNet;
using namespace WulfNet::Math;
using Catch::Matchers::WithinAbs;

constexpr f32 TEST_EPSILON = 0.0001f;

TEST_CASE("Vec3", "[math]") {
    SECTION("Default construction") {
        Vec3 v;
        REQUIRE_THAT(v.x, WithinAbs(0.0f, TEST_EPSILON));
        REQUIRE_THAT(v.y, WithinAbs(0.0f, TEST_EPSILON));
        REQUIRE_THAT(v.z, WithinAbs(0.0f, TEST_EPSILON));
    }
    
    SECTION("Value construction") {
        Vec3 v(1.0f, 2.0f, 3.0f);
        REQUIRE_THAT(v.x, WithinAbs(1.0f, TEST_EPSILON));
        REQUIRE_THAT(v.y, WithinAbs(2.0f, TEST_EPSILON));
        REQUIRE_THAT(v.z, WithinAbs(3.0f, TEST_EPSILON));
    }
    
    SECTION("Addition") {
        Vec3 a(1.0f, 2.0f, 3.0f);
        Vec3 b(4.0f, 5.0f, 6.0f);
        Vec3 c = a + b;
        REQUIRE_THAT(c.x, WithinAbs(5.0f, TEST_EPSILON));
        REQUIRE_THAT(c.y, WithinAbs(7.0f, TEST_EPSILON));
        REQUIRE_THAT(c.z, WithinAbs(9.0f, TEST_EPSILON));
    }
    
    SECTION("Subtraction") {
        Vec3 a(5.0f, 7.0f, 9.0f);
        Vec3 b(1.0f, 2.0f, 3.0f);
        Vec3 c = a - b;
        REQUIRE_THAT(c.x, WithinAbs(4.0f, TEST_EPSILON));
        REQUIRE_THAT(c.y, WithinAbs(5.0f, TEST_EPSILON));
        REQUIRE_THAT(c.z, WithinAbs(6.0f, TEST_EPSILON));
    }
    
    SECTION("Scalar multiplication") {
        Vec3 v(1.0f, 2.0f, 3.0f);
        Vec3 result = v * 2.0f;
        REQUIRE_THAT(result.x, WithinAbs(2.0f, TEST_EPSILON));
        REQUIRE_THAT(result.y, WithinAbs(4.0f, TEST_EPSILON));
        REQUIRE_THAT(result.z, WithinAbs(6.0f, TEST_EPSILON));
    }
    
    SECTION("Dot product") {
        Vec3 a(1.0f, 0.0f, 0.0f);
        Vec3 b(0.0f, 1.0f, 0.0f);
        f32 dotVal = a.dot(b);
        REQUIRE_THAT(dotVal, WithinAbs(0.0f, TEST_EPSILON));
        
        Vec3 c(1.0f, 2.0f, 3.0f);
        Vec3 d(4.0f, 5.0f, 6.0f);
        f32 dotVal2 = c.dot(d);
        REQUIRE_THAT(dotVal2, WithinAbs(32.0f, TEST_EPSILON));  // 1*4 + 2*5 + 3*6
    }
    
    SECTION("Cross product") {
        Vec3 x(1.0f, 0.0f, 0.0f);
        Vec3 y(0.0f, 1.0f, 0.0f);
        Vec3 z = x.cross(y);
        REQUIRE_THAT(z.x, WithinAbs(0.0f, TEST_EPSILON));
        REQUIRE_THAT(z.y, WithinAbs(0.0f, TEST_EPSILON));
        REQUIRE_THAT(z.z, WithinAbs(1.0f, TEST_EPSILON));
    }
    
    SECTION("Length") {
        Vec3 v(3.0f, 4.0f, 0.0f);
        f32 len = v.length();
        f32 lenSq = v.lengthSq();
        REQUIRE_THAT(len, WithinAbs(5.0f, TEST_EPSILON));
        REQUIRE_THAT(lenSq, WithinAbs(25.0f, TEST_EPSILON));
    }
    
    SECTION("Normalization") {
        Vec3 v(3.0f, 4.0f, 0.0f);
        Vec3 n = v.normalized();
        f32 nLen = n.length();
        REQUIRE_THAT(nLen, WithinAbs(1.0f, TEST_EPSILON));
        REQUIRE_THAT(n.x, WithinAbs(0.6f, TEST_EPSILON));
        REQUIRE_THAT(n.y, WithinAbs(0.8f, TEST_EPSILON));
    }
}

TEST_CASE("Vec4", "[math]") {
    SECTION("Construction and access") {
        Vec4 v(1.0f, 2.0f, 3.0f, 4.0f);
        REQUIRE_THAT(v.x, WithinAbs(1.0f, TEST_EPSILON));
        REQUIRE_THAT(v.y, WithinAbs(2.0f, TEST_EPSILON));
        REQUIRE_THAT(v.z, WithinAbs(3.0f, TEST_EPSILON));
        REQUIRE_THAT(v.w, WithinAbs(4.0f, TEST_EPSILON));
    }
    
    SECTION("Vec3 construction") {
        Vec3 v3(1.0f, 2.0f, 3.0f);
        Vec4 v4(v3, 1.0f);
        REQUIRE_THAT(v4.x, WithinAbs(1.0f, TEST_EPSILON));
        REQUIRE_THAT(v4.y, WithinAbs(2.0f, TEST_EPSILON));
        REQUIRE_THAT(v4.z, WithinAbs(3.0f, TEST_EPSILON));
        REQUIRE_THAT(v4.w, WithinAbs(1.0f, TEST_EPSILON));
    }
    
    SECTION("Dot product") {
        Vec4 a(1.0f, 2.0f, 3.0f, 4.0f);
        Vec4 b(5.0f, 6.0f, 7.0f, 8.0f);
        f32 d = a.dot(b);
        REQUIRE_THAT(d, WithinAbs(70.0f, TEST_EPSILON));
    }
}

TEST_CASE("Quat", "[math]") {
    SECTION("Identity quaternion") {
        Quat q = Quat::identity();
        REQUIRE_THAT(q.x, WithinAbs(0.0f, TEST_EPSILON));
        REQUIRE_THAT(q.y, WithinAbs(0.0f, TEST_EPSILON));
        REQUIRE_THAT(q.z, WithinAbs(0.0f, TEST_EPSILON));
        REQUIRE_THAT(q.w, WithinAbs(1.0f, TEST_EPSILON));
    }
}

TEST_CASE("Mat4", "[math]") {
    SECTION("Identity matrix") {
        Mat4 m = Mat4::identity();
        const f32* dptr = m.data;
        f32 d0 = dptr[0];
        REQUIRE_THAT(d0, WithinAbs(1.0f, TEST_EPSILON));
    }
}

TEST_CASE("MathAABB", "[math]") {
    SECTION("Construction") {
        MathAABB box(Vec3(-1, -1, -1), Vec3(1, 1, 1));
        bool c = box.contains(Vec3(0, 0, 0));
        REQUIRE(c);
    }
}
