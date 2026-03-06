// WulfNet Engine V3 Water System
// Shallow Water Equations - DX12 Compute Shader Backend
// Optimizations: Wavefront/Warp level execution, Struct-Of-Arrays

#define THREAD_GROUP_SIZE_X 8
#define THREAD_GROUP_SIZE_Y 8

cbuffer WaterConfig : register(b0)
{
    float gridSize;
    float dt;
    float gravity;
    float pipeCrossArea;
    uint width;
    uint height;
    float padding1;
    float padding2;
};

// Input/Output Buffers (SSBO / UAV)
// SoA layout matches CPU for zero-copy readbacks
RWStructuredBuffer<float> terrainHeight : register(u0);
RWStructuredBuffer<float> waterDepth    : register(u1);
// OPTIMIZATION: Packed 128-bit aligned vector fetching
// Replaces 4 distinct 32-bit fetches. Bumps memory throughput on Nvidia GPUs significantly!
RWStructuredBuffer<float4> fluxMap      : register(u2);

// -------------------------------------------------------------
// Pass 1: Flux Calculation w/ LDS (Local Data Share)
// -------------------------------------------------------------
groupshared float shared_H[10][10]; // 8x8 + 1-cell padding border

[numthreads(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y, 1)]
void ComputeFlux(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    uint x = dispatchThreadID.x;
    uint y = dispatchThreadID.y;
    uint lx = groupThreadID.x;
    uint ly = groupThreadID.y;

    uint idx = y * width + x;

    // Load Center
    float H_self = 0.0f;
    if (x < width && y < height) {
        H_self = terrainHeight[idx] + waterDepth[idx];
        shared_H[ly + 1][lx + 1] = H_self;
    }

    // Load Halos
    if (lx == 0 && x > 0)
        shared_H[ly + 1][0] = terrainHeight[idx - 1] + waterDepth[idx - 1];
    if (lx == THREAD_GROUP_SIZE_X - 1 && x < width - 1)
        shared_H[ly + 1][lx + 2] = terrainHeight[idx + 1] + waterDepth[idx + 1];
    if (ly == 0 && y > 0)
        shared_H[0][lx + 1] = terrainHeight[idx - width] + waterDepth[idx - width];
    if (ly == THREAD_GROUP_SIZE_Y - 1 && y < height - 1)
        shared_H[ly + 2][lx + 1] = terrainHeight[idx + width] + waterDepth[idx + width];

    // Synchronize thread blocks before continuing!
    GroupMemoryBarrierWithGroupSync();

    if (x >= width || y >= height) return;

    // Neighbor depths via ultra-fast L1 cache (LDS)
    float H_L = (x > 0) ? shared_H[ly + 1][lx] : H_self;
    float H_R = (x < width - 1) ? shared_H[ly + 1][lx + 2] : H_self;
    float H_T = (y > 0) ? shared_H[ly][lx + 1] : H_self;
    float H_B = (y < height - 1) ? shared_H[ly + 2][lx + 1] : H_self;

    float C = dt * gravity * pipeCrossArea / gridSize;
    float4 currentFlux = fluxMap[idx];

    // Outward flux equations using shared memory height map
    float fL = max(0.0f, currentFlux.x + C * (H_self - H_L));
    float fR = max(0.0f, currentFlux.y + C * (H_self - H_R));
    float fT = max(0.0f, currentFlux.z + C * (H_self - H_T));
    float fB = max(0.0f, currentFlux.w + C * (H_self - H_B));

    // Boundary conditions
    if (x == 0) fL = 0.0f;
    if (x == width - 1) fR = 0.0f;
    if (y == 0) fT = 0.0f;
    if (y == height - 1) fB = 0.0f;

    // Scaling factor K (Volume conservation)
    float totalFlux = fL + fR + fT + fB;
    if (totalFlux > 0.0f) {
        float K = min(1.0f, (waterDepth[idx] * gridSize * gridSize) / (dt * totalFlux));
        fL *= K; fR *= K; fT *= K; fB *= K;
    }

    // Write coalesced float4 back to unified buffer
    fluxMap[idx] = float4(fL, fR, fT, fB);
}

// -------------------------------------------------------------
// Pass 2: Water Depth Update
// -------------------------------------------------------------
[numthreads(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y, 1)]
void UpdateDepth(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint x = dispatchThreadID.x;
    uint y = dispatchThreadID.y;

    if (x >= width || y >= height) return;

    uint idx = y * width + x;

    uint pL = (x > 0) ? idx - 1 : idx;
    uint pR = (x < width - 1) ? idx + 1 : idx;
    uint pT = (y > 0) ? idx - width : idx;
    uint pB = (y < height - 1) ? idx + width : idx;

    float currentDepth = waterDepth[idx];

    // Read unified flux map once
    float4 myFlux = fluxMap[idx];

    // Flux OUT of self
    float totalFluxOut = myFlux.x + myFlux.y + myFlux.z + myFlux.w;

    // Flux IN from neighbors
    float totalFluxIn = 0.0f;
    if (x > 0) totalFluxIn += fluxMap[pL].y; // Right pipe of Left neighbor
    if (x < width - 1) totalFluxIn += fluxMap[pR].x; // Left pipe of Right neighbor
    if (y > 0) totalFluxIn += fluxMap[pT].w; // Bottom pipe of Top neighbor
    if (y < height - 1) totalFluxIn += fluxMap[pB].z; // Top pipe of Bottom neighbor
    float newDepth = currentDepth + (deltaV / (gridSize * gridSize));

    waterDepth[idx] = max(0.0f, newDepth);
}
