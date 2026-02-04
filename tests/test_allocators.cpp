// =============================================================================
// WulfNet Engine - Allocator Tests
// =============================================================================

#include <cassert>
#include <catch2/catch_all.hpp>
#include "core/memory/Memory.h"
#include "core/memory/LinearAllocator.h"
#include "core/memory/StackAllocator.h"
#include "core/memory/PoolAllocator.h"
#include "core/Types.h"

using namespace WulfNet;

TEST_CASE("LinearAllocator", "[memory]") {
    SECTION("Basic allocation") {
        LinearAllocator allocator(1024);
        
        void* ptr1 = allocator.allocate(64, 16);
        REQUIRE(ptr1 != nullptr);
        REQUIRE(reinterpret_cast<uintptr_t>(ptr1) % 16 == 0);
        
        void* ptr2 = allocator.allocate(64, 16);
        REQUIRE(ptr2 != nullptr);
        REQUIRE(ptr2 != ptr1);
    }
    
    SECTION("Reset clears all allocations") {
        LinearAllocator allocator(1024);
        
        void* ptr1 = allocator.allocate(512);
        REQUIRE(ptr1 != nullptr);
        
        allocator.reset();
        
        void* ptr2 = allocator.allocate(512);
        REQUIRE(ptr2 == ptr1);  // Should reuse same memory
    }
    
    SECTION("Returns nullptr when out of memory") {
        LinearAllocator allocator(64);
        
        void* ptr1 = allocator.allocate(32);
        REQUIRE(ptr1 != nullptr);
        
        void* ptr2 = allocator.allocate(64);  // Won't fit
        REQUIRE(ptr2 == nullptr);
    }
    
    SECTION("Typed allocation") {
        LinearAllocator allocator(1024);
        
        struct alignas(32) TestStruct {
            float data[8];
        };
        
        // Use create or manual cast. Original test used alloc<T>.
        TestStruct* obj = allocator.create<TestStruct>(); 
        REQUIRE(obj != nullptr);
        REQUIRE(reinterpret_cast<uintptr_t>(obj) % 32 == 0);
    }
}

TEST_CASE("PoolAllocator", "[memory]") {
    struct TestObject {
        u32 id;
        f32 value;
        char name[24];
    };
    
    // PoolAllocator is not templated and takes strict block size/count
    
    SECTION("Basic allocation and deallocation") {
        PoolAllocator pool(sizeof(TestObject), 64, alignof(TestObject));
        
        TestObject* obj1 = static_cast<TestObject*>(pool.allocate(sizeof(TestObject), alignof(TestObject)));
        REQUIRE(obj1 != nullptr);
        obj1->id = 42;
        obj1->value = 3.14f;
        
        TestObject* obj2 = static_cast<TestObject*>(pool.allocate(sizeof(TestObject), alignof(TestObject)));
        REQUIRE(obj2 != nullptr);
        REQUIRE(obj2 != obj1);
        
        pool.deallocate(obj1);
        
        // Next allocation should reuse freed slot
        TestObject* obj3 = static_cast<TestObject*>(pool.allocate(sizeof(TestObject), alignof(TestObject)));
        REQUIRE(obj3 == obj1);
    }
    
    SECTION("Returns nullptr when pool exhausted") {
        PoolAllocator pool(sizeof(TestObject), 2, alignof(TestObject));
        
        TestObject* obj1 = static_cast<TestObject*>(pool.allocate(sizeof(TestObject), alignof(TestObject)));
        TestObject* obj2 = static_cast<TestObject*>(pool.allocate(sizeof(TestObject), alignof(TestObject)));
        REQUIRE(obj1 != nullptr);
        REQUIRE(obj2 != nullptr);
        
        TestObject* obj3 = static_cast<TestObject*>(pool.allocate(sizeof(TestObject), alignof(TestObject)));
        REQUIRE(obj3 == nullptr);
    }
    
    SECTION("Can allocate full pool after freeing all") {
        PoolAllocator pool(sizeof(TestObject), 4, alignof(TestObject));
        
        TestObject* objs[4];
        for (int i = 0; i < 4; i++) {
            objs[i] = static_cast<TestObject*>(pool.allocate(sizeof(TestObject), alignof(TestObject)));
            REQUIRE(objs[i] != nullptr);
        }
        
        for (int i = 0; i < 4; i++) {
            pool.deallocate(objs[i]);
        }
        
        for (int i = 0; i < 4; i++) {
            TestObject* obj = static_cast<TestObject*>(pool.allocate(sizeof(TestObject), alignof(TestObject)));
            REQUIRE(obj != nullptr);
        }
    }
}

TEST_CASE("StackAllocator", "[memory]") {
    SECTION("Basic stack allocation") {
        StackAllocator allocator(1024);
        
        void* ptr1 = allocator.allocate(64, 16);
        REQUIRE(ptr1 != nullptr);
        REQUIRE(reinterpret_cast<uintptr_t>(ptr1) % 16 == 0);
        
        void* ptr2 = allocator.allocate(64, 16);
        REQUIRE(ptr2 != nullptr);
        REQUIRE(ptr2 > ptr1);  // Stack grows upward
    }
    
    SECTION("Marker-based deallocation") {
        StackAllocator allocator(1024);
        
        void* ptr1 = allocator.allocate(64);
        auto marker = allocator.getMarker();
        
        void* ptr2 = allocator.allocate(64);
        void* ptr3 = allocator.allocate(64);
        REQUIRE(ptr2 != nullptr);
        REQUIRE(ptr3 != nullptr);
        
        allocator.freeToMarker(marker);
        
        // Next allocation should start where ptr2 was
        void* ptr4 = allocator.allocate(64);
        REQUIRE(ptr4 == ptr2);
    }
    
    SECTION("Reset clears entire stack") {
        StackAllocator allocator(1024);
        
        allocator.allocate(256);
        allocator.allocate(256);
        allocator.allocate(256);
        
        allocator.reset();
        
        void* ptr = allocator.allocate(512);
        REQUIRE(ptr != nullptr);
    }
}

TEST_CASE("Allocator alignment", "[memory]") {
    SECTION("Various alignments") {
        LinearAllocator allocator(4096);
        
        constexpr usize alignments[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
        
        for (usize align : alignments) {
            void* ptr = allocator.allocate(64, align);
            REQUIRE(ptr != nullptr);
            REQUIRE(reinterpret_cast<uintptr_t>(ptr) % align == 0);
        }
    }
}
