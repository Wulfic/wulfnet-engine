// =============================================================================
// WulfNet Engine - Types Tests
// =============================================================================

#include <cassert>
#include <catch2/catch_all.hpp>
#include "core/Types.h"

using namespace WulfNet;

TEST_CASE("Fixed-width type sizes", "[types]") {
    SECTION("Signed integers") {
        REQUIRE(sizeof(i8) == 1);
        REQUIRE(sizeof(i16) == 2);
        REQUIRE(sizeof(i32) == 4);
        REQUIRE(sizeof(i64) == 8);
    }
    
    SECTION("Unsigned integers") {
        REQUIRE(sizeof(u8) == 1);
        REQUIRE(sizeof(u16) == 2);
        REQUIRE(sizeof(u32) == 4);
        REQUIRE(sizeof(u64) == 8);
    }
    
    SECTION("Floating point") {
        REQUIRE(sizeof(f32) == 4);
        REQUIRE(sizeof(f64) == 8);
    }
    
    SECTION("Size types") {
        REQUIRE(sizeof(usize) == sizeof(void*));
        REQUIRE(sizeof(isize) == sizeof(void*));
    }
}

TEST_CASE("Handle type-safety", "[types]") {
    struct EntityTag {};
    struct ComponentTag {};
    
    using EntityHandle = Handle<EntityTag, u32>;
    using ComponentHandle = Handle<ComponentTag, u32>;
    
    SECTION("Default construction creates invalid handle") {
        EntityHandle handle;
        REQUIRE_FALSE(handle.isValid());
        REQUIRE(handle.value == 0);
    }
    
    SECTION("Valid handle construction") {
        EntityHandle handle(42);
        REQUIRE(handle.isValid());
        REQUIRE(handle.value == 42);
    }
    
    SECTION("Handle equality") {
        EntityHandle a(10);
        EntityHandle b(10);
        EntityHandle c(20);  // Different value
        EntityHandle d(11);  // Different value
        
        REQUIRE(a == b);
        REQUIRE_FALSE(a == c);
        REQUIRE_FALSE(a == d);
    }
}

TEST_CASE("Span functionality", "[types]") {
    SECTION("From raw pointer and size") {
        int data[] = {1, 2, 3, 4, 5};
        Span<int> span(data, 5);
        
        REQUIRE(span.size() == 5);
        REQUIRE_FALSE(span.empty());
        REQUIRE(span[0] == 1);
        REQUIRE(span[4] == 5);
    }
    
    SECTION("Empty span") {
        Span<int> span;
        REQUIRE(span.size() == 0);
        REQUIRE(span.empty());
        REQUIRE(span.data() == nullptr);
    }
    
    SECTION("Iteration") {
        int data[] = {10, 20, 30};
        Span<int> span(data, 3);
        
        int sum = 0;
        for (int val : span) {
            sum += val;
        }
        REQUIRE(sum == 60);
    }
    
    SECTION("Subspan") {
        int data[] = {1, 2, 3, 4, 5};
        Span<int> span(data, 5);
        
        auto sub = span.subspan(1, 3);
        REQUIRE(sub.size() == 3);
        REQUIRE(sub[0] == 2);
        REQUIRE(sub[2] == 4);
    }
}

TEST_CASE("Utility functions", "[types]") {
    SECTION("isPowerOfTwo") {
        REQUIRE(isPowerOfTwo(1));
        REQUIRE(isPowerOfTwo(2));
        REQUIRE(isPowerOfTwo(4));
        REQUIRE(isPowerOfTwo(1024));
        
        REQUIRE_FALSE(isPowerOfTwo(0));
        REQUIRE_FALSE(isPowerOfTwo(3));
        REQUIRE_FALSE(isPowerOfTwo(100));
    }
    
    SECTION("alignUp") {
        REQUIRE(alignUp(0, 16) == 0);
        REQUIRE(alignUp(1, 16) == 16);
        REQUIRE(alignUp(16, 16) == 16);
        REQUIRE(alignUp(17, 16) == 32);
        REQUIRE(alignUp(100, 64) == 128);
    }
    
    SECTION("alignDown") {
        REQUIRE(alignDown(0, 16) == 0);
        REQUIRE(alignDown(15, 16) == 0);
        REQUIRE(alignDown(16, 16) == 16);
        REQUIRE(alignDown(31, 16) == 16);
        REQUIRE(alignDown(100, 64) == 64);
    }
    
    SECTION("kilobytes/megabytes/gigabytes") {
        // REQUIRE(kilobytes(1) == 1024);
        // REQUIRE(megabytes(1) == 1024 * 1024);
        // REQUIRE(gigabytes(1) == 1024 * 1024 * 1024);
    }
}
