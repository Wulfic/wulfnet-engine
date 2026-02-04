// =============================================================================
// WulfNet Engine - Broadphase Collision Detection
// =============================================================================
// Spatial hashing broadphase for efficient AABB overlap detection.
// =============================================================================

#pragma once

#include "core/Types.h"
#include "core/Log.h"
#include "core/Assert.h"
#include "CollisionTypes.h"
#include "AABB.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>

namespace WulfNet {

// =============================================================================
// Broadphase Interface
// =============================================================================

struct BroadphaseProxy {
    u32 id = 0;
    AABB aabb;
    CollisionLayer layer = CollisionLayer::Default;
    void* userData = nullptr;
    
    BroadphaseProxy() = default;
    BroadphaseProxy(u32 id_, const AABB& aabb_, CollisionLayer layer_ = CollisionLayer::Default, void* data = nullptr)
        : id(id_), aabb(aabb_), layer(layer_), userData(data) {}
};

class IBroadphase : public NonCopyable {
public:
    virtual ~IBroadphase() = default;
    
    // Proxy Management
    virtual u32 createProxy(const AABB& aabb, CollisionLayer layer = CollisionLayer::Default, void* userData = nullptr) = 0;
    virtual void destroyProxy(u32 proxyId) = 0;
    virtual void updateProxy(u32 proxyId, const AABB& aabb) = 0;
    
    // Queries
    virtual void findOverlappingPairs(std::vector<CollisionPair>& outPairs) = 0;
    virtual void queryAABB(const AABB& aabb, std::vector<u32>& outProxyIds) = 0;
    virtual void queryRay(const Vec3& origin, const Vec3& direction, f32 maxDistance,
                         std::vector<u32>& outProxyIds) = 0;
    
    // Stats
    virtual size_t getProxyCount() const = 0;
    virtual void clear() = 0;
};

// =============================================================================
// Spatial Hash Broadphase
// =============================================================================

class SpatialHashBroadphase : public IBroadphase {
public:
    explicit SpatialHashBroadphase(f32 cellSize = 10.0f)
        : m_cellSize(cellSize)
        , m_invCellSize(1.0f / cellSize)
        , m_nextProxyId(1)
    {
        WULFNET_LOG_INFO("SpatialHashBroadphase", "Initialized with cellSize=%.2f", cellSize);
    }
    
    ~SpatialHashBroadphase() override {
        WULFNET_LOG_DEBUG("SpatialHashBroadphase", "Destroyed (%u proxies)", (u32)m_proxies.size());
    }
    
    // Configuration
    f32 getCellSize() const { return m_cellSize; }
    
    void setCellSize(f32 size) {
        m_cellSize = size;
        m_invCellSize = 1.0f / size;
        rebuildGrid();
    }
    
    // IBroadphase Implementation
    u32 createProxy(const AABB& aabb, CollisionLayer layer = CollisionLayer::Default, 
                    void* userData = nullptr) override {
        u32 id = m_nextProxyId++;
        m_proxies[id] = BroadphaseProxy(id, aabb, layer, userData);
        insertProxyIntoCells(id, aabb);
        
        WULFNET_LOG_TRACE("SpatialHashBroadphase", "Created proxy {} at ({:.2f},{:.2f},{:.2f})-({:.2f},{:.2f},{:.2f})",
            id, aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z);
        
        return id;
    }
    
    void destroyProxy(u32 proxyId) override {
        auto it = m_proxies.find(proxyId);
        if (it == m_proxies.end()) {
            WULFNET_LOG_WARN("SpatialHashBroadphase", "Attempted to destroy non-existent proxy {}", proxyId);
            return;
        }
        
        removeProxyFromCells(proxyId, it->second.aabb);
        m_proxies.erase(it);
        
        WULFNET_LOG_TRACE("SpatialHashBroadphase", "Destroyed proxy {}", proxyId);
    }
    
    void updateProxy(u32 proxyId, const AABB& newAABB) override {
        auto it = m_proxies.find(proxyId);
        if (it == m_proxies.end()) {
            WULFNET_LOG_WARN("SpatialHashBroadphase", "Attempted to update non-existent proxy {}", proxyId);
            return;
        }
        
        BroadphaseProxy& proxy = it->second;
        
        // Check if cells need updating (optimization: skip if still in same cells)
        if (!aabbsShareCells(proxy.aabb, newAABB)) {
            removeProxyFromCells(proxyId, proxy.aabb);
            insertProxyIntoCells(proxyId, newAABB);
        }
        
        proxy.aabb = newAABB;
    }
    
    void findOverlappingPairs(std::vector<CollisionPair>& outPairs) override {
        outPairs.clear();
        std::unordered_set<u64> testedPairs;
        
        for (const auto& [cellHash, proxyIds] : m_cells) {
            const size_t count = proxyIds.size();
            
            for (size_t i = 0; i < count; ++i) {
                for (size_t j = i + 1; j < count; ++j) {
                    u32 idA = proxyIds[i];
                    u32 idB = proxyIds[j];
                    
                    // Ensure consistent ordering
                    if (idA > idB) std::swap(idA, idB);
                    
                    // Create unique pair key
                    u64 pairKey = (static_cast<u64>(idA) << 32) | idB;
                    
                    // Skip if already tested
                    if (testedPairs.count(pairKey) > 0) continue;
                    testedPairs.insert(pairKey);
                    
                    // Get proxies
                    auto itA = m_proxies.find(idA);
                    auto itB = m_proxies.find(idB);
                    if (itA == m_proxies.end() || itB == m_proxies.end()) continue;
                    
                    const BroadphaseProxy& proxyA = itA->second;
                    const BroadphaseProxy& proxyB = itB->second;
                    
                    // Check layer collision
                    if (!canLayersCollide(proxyA.layer, proxyB.layer)) continue;
                    
                    // Fine-grained AABB test
                    if (proxyA.aabb.intersects(proxyB.aabb)) {
                        CollisionPair pair;
                        pair.bodyIdA = idA;
                        pair.bodyIdB = idB;
                        pair.isNew = true;  // Will be updated by collision manager
                        outPairs.push_back(pair);
                    }
                }
            }
        }
        
        WULFNET_LOG_TRACE("SpatialHashBroadphase", "Found {} overlapping pairs from {} cells",
            outPairs.size(), m_cells.size());
    }
    
    void queryAABB(const AABB& aabb, std::vector<u32>& outProxyIds) override {
        outProxyIds.clear();
        std::unordered_set<u32> foundIds;
        
        i32 minCellX, minCellY, minCellZ;
        i32 maxCellX, maxCellY, maxCellZ;
        getCellRange(aabb, minCellX, minCellY, minCellZ, maxCellX, maxCellY, maxCellZ);
        
        for (i32 x = minCellX; x <= maxCellX; ++x) {
            for (i32 y = minCellY; y <= maxCellY; ++y) {
                for (i32 z = minCellZ; z <= maxCellZ; ++z) {
                    u64 hash = hashCell(x, y, z);
                    auto it = m_cells.find(hash);
                    if (it != m_cells.end()) {
                        for (u32 id : it->second) {
                            if (foundIds.count(id) == 0) {
                                auto proxyIt = m_proxies.find(id);
                                if (proxyIt != m_proxies.end() && 
                                    proxyIt->second.aabb.intersects(aabb)) {
                                    foundIds.insert(id);
                                    outProxyIds.push_back(id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    void queryRay(const Vec3& origin, const Vec3& direction, f32 maxDistance,
                 std::vector<u32>& outProxyIds) override {
        outProxyIds.clear();
        
        // Create AABB for ray extent
        Vec3 endPoint = origin + direction * maxDistance;
        AABB rayAABB = AABB::fromPoints(origin, endPoint);
        
        // Precompute inverse direction for slab method
        Vec3 dirInv(
            std::abs(direction.x) > 1e-6f ? 1.0f / direction.x : 1e6f,
            std::abs(direction.y) > 1e-6f ? 1.0f / direction.y : 1e6f,
            std::abs(direction.z) > 1e-6f ? 1.0f / direction.z : 1e6f
        );
        
        // Query all potential candidates
        std::vector<u32> candidates;
        queryAABB(rayAABB, candidates);
        
        // Fine test with ray-AABB intersection
        for (u32 id : candidates) {
            auto it = m_proxies.find(id);
            if (it != m_proxies.end()) {
                f32 tMin, tMax;
                if (it->second.aabb.rayIntersects(origin, dirInv, tMin, tMax) && tMin <= maxDistance) {
                    outProxyIds.push_back(id);
                }
            }
        }
    }
    
    size_t getProxyCount() const override {
        return m_proxies.size();
    }
    
    void clear() override {
        m_proxies.clear();
        m_cells.clear();
        WULFNET_LOG_DEBUG("SpatialHashBroadphase", "Cleared all proxies");
    }
    
    // Get proxy by ID
    const BroadphaseProxy* getProxy(u32 id) const {
        auto it = m_proxies.find(id);
        return it != m_proxies.end() ? &it->second : nullptr;
    }
    
    BroadphaseProxy* getProxy(u32 id) {
        auto it = m_proxies.find(id);
        return it != m_proxies.end() ? &it->second : nullptr;
    }
    
private:
    // Hash function for 3D cell coordinates
    u64 hashCell(i32 x, i32 y, i32 z) const {
        // Large primes for hash mixing
        constexpr u64 p1 = 73856093ULL;
        constexpr u64 p2 = 19349663ULL;
        constexpr u64 p3 = 83492791ULL;
        
        u64 hx = static_cast<u64>(static_cast<u32>(x)) * p1;
        u64 hy = static_cast<u64>(static_cast<u32>(y)) * p2;
        u64 hz = static_cast<u64>(static_cast<u32>(z)) * p3;
        
        return hx ^ hy ^ hz;
    }
    
    i32 toCellCoord(f32 val) const {
        return static_cast<i32>(std::floor(val * m_invCellSize));
    }
    
    void getCellRange(const AABB& aabb, 
                     i32& minX, i32& minY, i32& minZ,
                     i32& maxX, i32& maxY, i32& maxZ) const {
        minX = toCellCoord(aabb.min.x);
        minY = toCellCoord(aabb.min.y);
        minZ = toCellCoord(aabb.min.z);
        maxX = toCellCoord(aabb.max.x);
        maxY = toCellCoord(aabb.max.y);
        maxZ = toCellCoord(aabb.max.z);
    }
    
    void insertProxyIntoCells(u32 proxyId, const AABB& aabb) {
        i32 minX, minY, minZ, maxX, maxY, maxZ;
        getCellRange(aabb, minX, minY, minZ, maxX, maxY, maxZ);
        
        for (i32 x = minX; x <= maxX; ++x) {
            for (i32 y = minY; y <= maxY; ++y) {
                for (i32 z = minZ; z <= maxZ; ++z) {
                    u64 hash = hashCell(x, y, z);
                    m_cells[hash].push_back(proxyId);
                }
            }
        }
    }
    
    void removeProxyFromCells(u32 proxyId, const AABB& aabb) {
        i32 minX, minY, minZ, maxX, maxY, maxZ;
        getCellRange(aabb, minX, minY, minZ, maxX, maxY, maxZ);
        
        for (i32 x = minX; x <= maxX; ++x) {
            for (i32 y = minY; y <= maxY; ++y) {
                for (i32 z = minZ; z <= maxZ; ++z) {
                    u64 hash = hashCell(x, y, z);
                    auto it = m_cells.find(hash);
                    if (it != m_cells.end()) {
                        auto& vec = it->second;
                        vec.erase(std::remove(vec.begin(), vec.end(), proxyId), vec.end());
                        if (vec.empty()) {
                            m_cells.erase(it);
                        }
                    }
                }
            }
        }
    }
    
    bool aabbsShareCells(const AABB& a, const AABB& b) const {
        i32 aMinX, aMinY, aMinZ, aMaxX, aMaxY, aMaxZ;
        i32 bMinX, bMinY, bMinZ, bMaxX, bMaxY, bMaxZ;
        getCellRange(a, aMinX, aMinY, aMinZ, aMaxX, aMaxY, aMaxZ);
        getCellRange(b, bMinX, bMinY, bMinZ, bMaxX, bMaxY, bMaxZ);
        
        return (aMinX == bMinX && aMinY == bMinY && aMinZ == bMinZ &&
                aMaxX == bMaxX && aMaxY == bMaxY && aMaxZ == bMaxZ);
    }
    
    void rebuildGrid() {
        std::unordered_map<u32, BroadphaseProxy> oldProxies = std::move(m_proxies);
        m_cells.clear();
        
        for (auto& [id, proxy] : oldProxies) {
            m_proxies[id] = proxy;
            insertProxyIntoCells(id, proxy.aabb);
        }
        
        WULFNET_LOG_DEBUG("SpatialHashBroadphase", "Rebuilt grid with {} proxies", m_proxies.size());
    }
    
    f32 m_cellSize;
    f32 m_invCellSize;
    u32 m_nextProxyId;
    
    std::unordered_map<u32, BroadphaseProxy> m_proxies;
    std::unordered_map<u64, std::vector<u32>> m_cells;  // Cell hash -> proxy IDs
};

// =============================================================================
// GPU-Ready Spatial Hash Broadphase (CPU fallback, sort-based)
// =============================================================================

class GpuSpatialHashBroadphase : public IBroadphase {
public:
    explicit GpuSpatialHashBroadphase(f32 cellSize = 10.0f)
        : m_cellSize(cellSize)
        , m_invCellSize(1.0f / cellSize)
        , m_nextProxyId(1)
    {
        WULFNET_LOG_INFO("GpuSpatialHashBroadphase", "Initialized with cellSize={:.2f}", cellSize);
    }

    ~GpuSpatialHashBroadphase() override {
        WULFNET_LOG_DEBUG("GpuSpatialHashBroadphase", "Destroyed ({} proxies)", m_proxies.size());
    }

    f32 getCellSize() const { return m_cellSize; }

    void setCellSize(f32 size) {
        m_cellSize = size;
        m_invCellSize = 1.0f / size;
    }

    u32 createProxy(const AABB& aabb, CollisionLayer layer = CollisionLayer::Default,
                    void* userData = nullptr) override {
        u32 id = m_nextProxyId++;
        m_proxies[id] = BroadphaseProxy(id, aabb, layer, userData);

        WULFNET_LOG_TRACE("GpuSpatialHashBroadphase", "Created proxy {}", id);
        return id;
    }

    void destroyProxy(u32 proxyId) override {
        auto it = m_proxies.find(proxyId);
        if (it == m_proxies.end()) {
            WULFNET_LOG_WARN("GpuSpatialHashBroadphase", "Attempted to destroy non-existent proxy {}", proxyId);
            return;
        }

        m_proxies.erase(it);
        WULFNET_LOG_TRACE("GpuSpatialHashBroadphase", "Destroyed proxy {}", proxyId);
    }

    void updateProxy(u32 proxyId, const AABB& aabb) override {
        auto it = m_proxies.find(proxyId);
        if (it == m_proxies.end()) {
            WULFNET_LOG_WARN("GpuSpatialHashBroadphase", "Attempted to update non-existent proxy {}", proxyId);
            return;
        }

        it->second.aabb = aabb;
    }

    void findOverlappingPairs(std::vector<CollisionPair>& outPairs) override {
        outPairs.clear();
        m_cellEntries.clear();

        if (m_proxies.empty()) {
            return;
        }

        // Build cell entries (GPU-friendly key/value stream)
        for (const auto& [id, proxy] : m_proxies) {
            i32 minX, minY, minZ, maxX, maxY, maxZ;
            getCellRange(proxy.aabb, minX, minY, minZ, maxX, maxY, maxZ);

            for (i32 x = minX; x <= maxX; ++x) {
                for (i32 y = minY; y <= maxY; ++y) {
                    for (i32 z = minZ; z <= maxZ; ++z) {
                        m_cellEntries.push_back(CellEntry{hashCell(x, y, z), id});
                    }
                }
            }
        }

        if (m_cellEntries.empty()) {
            return;
        }

        std::sort(m_cellEntries.begin(), m_cellEntries.end(),
            [](const CellEntry& a, const CellEntry& b) { return a.key < b.key; });

        std::unordered_set<u64> testedPairs;

        // Scan runs of identical keys
        size_t i = 0;
        while (i < m_cellEntries.size()) {
            size_t j = i + 1;
            while (j < m_cellEntries.size() && m_cellEntries[j].key == m_cellEntries[i].key) {
                ++j;
            }

            // Generate pairs within [i, j)
            for (size_t a = i; a < j; ++a) {
                for (size_t b = a + 1; b < j; ++b) {
                    u32 idA = m_cellEntries[a].proxyId;
                    u32 idB = m_cellEntries[b].proxyId;

                    if (idA > idB) std::swap(idA, idB);
                    u64 pairKey = (static_cast<u64>(idA) << 32) | idB;

                    if (testedPairs.count(pairKey) > 0) continue;
                    testedPairs.insert(pairKey);

                    auto itA = m_proxies.find(idA);
                    auto itB = m_proxies.find(idB);
                    if (itA == m_proxies.end() || itB == m_proxies.end()) continue;

                    const BroadphaseProxy& proxyA = itA->second;
                    const BroadphaseProxy& proxyB = itB->second;

                    if (!canLayersCollide(proxyA.layer, proxyB.layer)) continue;

                    if (proxyA.aabb.intersects(proxyB.aabb)) {
                        CollisionPair pair;
                        pair.bodyIdA = idA;
                        pair.bodyIdB = idB;
                        pair.isNew = true;
                        outPairs.push_back(pair);
                    }
                }
            }

            i = j;
        }

        WULFNET_LOG_TRACE("GpuSpatialHashBroadphase", "Found {} overlapping pairs from {} cell entries",
            outPairs.size(), m_cellEntries.size());
    }

    void queryAABB(const AABB& aabb, std::vector<u32>& outProxyIds) override {
        outProxyIds.clear();
        outProxyIds.reserve(m_proxies.size());

        for (const auto& [id, proxy] : m_proxies) {
            if (proxy.aabb.intersects(aabb)) {
                outProxyIds.push_back(id);
            }
        }
    }

    void queryRay(const Vec3& origin, const Vec3& direction, f32 maxDistance,
                  std::vector<u32>& outProxyIds) override {
        outProxyIds.clear();

        Vec3 dirInv(
            std::abs(direction.x) > 1e-6f ? 1.0f / direction.x : 1e6f,
            std::abs(direction.y) > 1e-6f ? 1.0f / direction.y : 1e6f,
            std::abs(direction.z) > 1e-6f ? 1.0f / direction.z : 1e6f
        );

        for (const auto& [id, proxy] : m_proxies) {
            f32 tMin, tMax;
            if (proxy.aabb.rayIntersects(origin, dirInv, tMin, tMax) && tMin <= maxDistance) {
                outProxyIds.push_back(id);
            }
        }
    }

    size_t getProxyCount() const override {
        return m_proxies.size();
    }

    void clear() override {
        m_proxies.clear();
        m_cellEntries.clear();
        WULFNET_LOG_DEBUG("GpuSpatialHashBroadphase", "Cleared all proxies");
    }

private:
    struct CellEntry {
        u64 key = 0;
        u32 proxyId = 0;
    };

    u64 hashCell(i32 x, i32 y, i32 z) const {
        constexpr u64 p1 = 73856093ULL;
        constexpr u64 p2 = 19349663ULL;
        constexpr u64 p3 = 83492791ULL;

        u64 hx = static_cast<u64>(static_cast<u32>(x)) * p1;
        u64 hy = static_cast<u64>(static_cast<u32>(y)) * p2;
        u64 hz = static_cast<u64>(static_cast<u32>(z)) * p3;

        return hx ^ hy ^ hz;
    }

    i32 toCellCoord(f32 val) const {
        return static_cast<i32>(std::floor(val * m_invCellSize));
    }

    void getCellRange(const AABB& aabb,
                      i32& minX, i32& minY, i32& minZ,
                      i32& maxX, i32& maxY, i32& maxZ) const {
        minX = toCellCoord(aabb.min.x);
        minY = toCellCoord(aabb.min.y);
        minZ = toCellCoord(aabb.min.z);
        maxX = toCellCoord(aabb.max.x);
        maxY = toCellCoord(aabb.max.y);
        maxZ = toCellCoord(aabb.max.z);
    }

    f32 m_cellSize;
    f32 m_invCellSize;
    u32 m_nextProxyId;

    std::unordered_map<u32, BroadphaseProxy> m_proxies;
    std::vector<CellEntry> m_cellEntries;
};

} // namespace WulfNet
