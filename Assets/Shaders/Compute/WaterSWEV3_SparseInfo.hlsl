// WulfNet Engine V3 Water System
// Sparse Tile Classification Pass (HLSL)
// Identifies 8x8 thread blocks that have active water and appends them to a dispatch list.

#define TILE_SIZE 8

cbuffer WaterConfig : register(b0)
{
    float gridSize;
    float dt;
    float gravity;
    float pipeCrossArea;
    uint width;
    uint height;
};

StructuredBuffer<float> waterDepth     : register(t0);
StructuredBuffer<float4> fluxMap       : register(t1);

// An Append Structured Buffer allows atomic appends from multiple thread groups
AppendStructuredBuffer<uint2> activeTileList : register(u0);

// Use WaveActive math inside the group to quickly vote on activity
groupshared uint shared_isActive;

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void ClassifyTiles(uint3 dispatchThreadID : SV_DispatchThreadID,
                   uint3 groupID : SV_GroupID,
                   uint groupIndex : SV_GroupIndex)
{
    // Reset shared state
    if (groupIndex == 0) {
        shared_isActive = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint x = dispatchThreadID.x;
    uint y = dispatchThreadID.y;

    bool localActive = false;

    if (x < width && y < height) {
        uint idx = y * width + x;
        float d = waterDepth[idx];
        float4 f = fluxMap[idx];

        // If there's water volume or moving flux, tile is active
        if (d > 0.001f || f.x > 0.001f || f.y > 0.001f || f.z > 0.001f || f.w > 0.001f) {
            localActive = true;
        }
    }

    // Atomic OR to check if *anyone* in the 8x8 block is active
    if (localActive) {
        InterlockedOr(shared_isActive, 1);
    }

    GroupMemoryBarrierWithGroupSync();

    // The thread leader appends to the indirect buffer
    if (groupIndex == 0 && shared_isActive > 0) {
        activeTileList.Append(uint2(groupID.x, groupID.y));
    }
}
