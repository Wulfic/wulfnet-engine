#include <Framework.h>
#include <Tests/WulfNet/WulfNetWaterV3Test.h>
#include <WulfNet/Jolt/Physics/Body/BodyCreationSettings.h>
#include <WulfNet/Jolt/Physics/Collision/Shape/BoxShape.h>
#include <WulfNet/Jolt/Physics/Collision/Shape/SphereShape.h>
#include <WulfNet/Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <SamplesLayers.h>
#include <Renderer/DebugRendererImp.h>
#include <algorithm>
#include <vector>

JPH_IMPLEMENT_RTTI_VIRTUAL(WulfNetWaterV3Test)
{
    JPH_ADD_BASE_CLASS(WulfNetWaterV3Test, Test)
}

void WulfNetWaterV3Test::Initialize()
{
    // 1. Initialize Ground
    // Floor
    CreateFloor();

    // 2. Initialize WulfNet Water System V3 (SWE 2.5D)
    WulfNet::FluidSystemConfig config;
    config.width = 128;
    config.height = 128;
    config.gridSize = 0.5f; // 64x64 meter area
    config.gravity = 9.8f;
    config.originX = -(config.width * config.gridSize) / 2.0f;
    config.originZ = -(config.height * config.gridSize) / 2.0f;

    mWaterSystem = new WulfNet::FluidSystem(config, mPhysicsSystem);

    auto& state = mWaterSystem->GetCPUState();

    // Create a terrain basin
    for (uint32_t y = 0; y < config.height; ++y) {
        for (uint32_t x = 0; x < config.width; ++x) {
            float dx = x - (config.width / 2.0f);
            float dy = y - (config.height / 2.0f);
            float dist = std::sqrt(dx*dx + dy*dy);

            uint32_t idx = y * config.width + x;

            // Sloped bowl shape
            state.terrainHeight[idx] = std::pow(dist * 0.05f, 2.0f);

            // Fill center with water
            if (dist < 20.0f) {
                mWaterSystem->AddWater(x, y, 4.0f);
            }
        }
    }

    // 3. Drop floating objects
    JPH::RefConst<JPH::Shape> boxShape = new JPH::BoxShape(JPH::Vec3(0.5f, 0.5f, 0.5f));
    JPH::RefConst<JPH::Shape> sphereShape = new JPH::SphereShape(0.5f);
    JPH::RefConst<JPH::Shape> cylinderShape = new JPH::CylinderShape(1.0f, 0.3f);

    for (int i = 0; i < 50; ++i) {
        JPH::Vec3 pos(
            (float(rand()) / RAND_MAX) * 20.0f - 10.0f,
            10.0f + (float(rand()) / RAND_MAX) * 10.0f,
            (float(rand()) / RAND_MAX) * 20.0f - 10.0f
        );

        JPH::RefConst<JPH::Shape> shape;
        int shapeType = rand() % 3;
        if (shapeType == 0) shape = boxShape;
        else if (shapeType == 1) shape = sphereShape;
        else shape = cylinderShape;

        JPH::BodyCreationSettings bcs(shape, pos, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
        // Make them light (wood/plastic density vs water 1000)
        bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateMassAndInertia;
        bcs.mMassPropertiesOverride.mMass = 200.0f * shape->GetVolume(); // Density ~200kg/m^3 (floats well)

        JPH::Body* body = mBodyInterface->CreateBody(bcs);
        mBodyInterface->AddBody(body->GetID(), JPH::EActivation::Activate);
        mFloatingBodies.push_back(body->GetID());
    }
}

void WulfNetWaterV3Test::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
    if (!mWaterSystem) return;

    mTime += inParams.mDeltaTime;

    // Simulate water mathematically
    mWaterSystem->StepSimulationCPU(inParams.mDeltaTime);

    // Provide bodies to fluid system so it can run Jolt async buoyancy jobs
    auto& context = mWaterSystem->GetJobContextForTesting();
    context.interactingBodies = mFloatingBodies;

    mWaterSystem->ApplyBuoyancyForces(mJobSystem);

	// Build smooth water surface mesh using CreateTriangleBatch + DrawGeometry
	const auto& config = mWaterSystem->GetConfig();
	const auto& state = mWaterSystem->GetCPUState();
	const uint32_t W = config.width;
	const uint32_t H = config.height;
	const float gs = config.gridSize;

	JPH::Vec3 offset(config.originX, 0.0f, config.originZ);

	// --- Step 1: Compute per-vertex water surface heights and depth-based color ---
	// We store surface height for every grid vertex (terrain + water).
	// A vertex is "wet" if it or any neighbor has meaningful water depth.
	std::vector<float> surfaceH(W * H);
	std::vector<bool>  wet(W * H, false);

	for (uint32_t y = 0; y < H; ++y) {
		for (uint32_t x = 0; x < W; ++x) {
			uint32_t idx = y * W + x;
			surfaceH[idx] = state.terrainHeight[idx] + state.waterDepth[idx];
			if (state.waterDepth[idx] > 0.01f)
				wet[idx] = true;
		}
	}

	// Expand wet mask by 3 cells so edges taper smoothly
	std::vector<bool> wetExpanded = wet;
	for (int expand = 0; expand < 3; ++expand) {
		std::vector<bool> prev = wetExpanded;
		for (uint32_t y = 0; y < H; ++y) {
			for (uint32_t x = 0; x < W; ++x) {
				if (prev[y * W + x]) {
					if (x > 0)     wetExpanded[y * W + (x - 1)] = true;
					if (x < W - 1) wetExpanded[y * W + (x + 1)] = true;
					if (y > 0)     wetExpanded[(y - 1) * W + x] = true;
					if (y < H - 1) wetExpanded[(y + 1) * W + x] = true;
				}
			}
		}
	}

	// --- Step 1.5: Smooth surface heights for visual rendering (3-pass 3x3 box filter) ---
	// Operates on wetExpanded cells so boundary transitions are smooth
	for (int pass = 0; pass < 3; ++pass) {
		std::vector<float> tmp = surfaceH;
		for (uint32_t y = 1; y < H - 1; ++y) {
			for (uint32_t x = 1; x < W - 1; ++x) {
				uint32_t idx = y * W + x;
				if (!wetExpanded[idx]) continue;
				float sum = 0.0f, wt = 0.0f;
				for (int dy = -1; dy <= 1; ++dy) {
					for (int dx = -1; dx <= 1; ++dx) {
						uint32_t ni = (y + dy) * W + (x + dx);
						if (wetExpanded[ni]) {
							float w = (dx == 0 && dy == 0) ? 4.0f : 1.0f;
							sum += tmp[ni] * w;
							wt += w;
						}
					}
				}
				if (wt > 0.0f) surfaceH[idx] = sum / wt;
			}
		}
	}

	// --- Step 2: Compute smooth per-vertex normals (area-weighted face normals) ---
	std::vector<JPH::Vec3> normals(W * H, JPH::Vec3::sZero());

	for (uint32_t y = 0; y < H - 1; ++y) {
		for (uint32_t x = 0; x < W - 1; ++x) {
			uint32_t i00 = y * W + x;
			uint32_t i10 = i00 + 1;
			uint32_t i01 = i00 + W;
			uint32_t i11 = i00 + W + 1;

			// Skip quads in entirely dry regions
			if (!wetExpanded[i00] && !wetExpanded[i10] && !wetExpanded[i01] && !wetExpanded[i11])
				continue;

			JPH::Vec3 p00 = offset + JPH::Vec3(x * gs,       surfaceH[i00], y * gs);
			JPH::Vec3 p10 = offset + JPH::Vec3((x + 1) * gs, surfaceH[i10], y * gs);
			JPH::Vec3 p01 = offset + JPH::Vec3(x * gs,       surfaceH[i01], (y + 1) * gs);
			JPH::Vec3 p11 = offset + JPH::Vec3((x + 1) * gs, surfaceH[i11], (y + 1) * gs);

			// Upper-left triangle: p00, p10, p01
			JPH::Vec3 n1 = (p10 - p00).Cross(p01 - p00);
			normals[i00] += n1;
			normals[i10] += n1;
			normals[i01] += n1;

			// Lower-right triangle: p10, p11, p01
			JPH::Vec3 n2 = (p11 - p10).Cross(p01 - p10);
			normals[i10] += n2;
			normals[i11] += n2;
			normals[i01] += n2;
		}
	}

	// Normalize
	for (uint32_t i = 0; i < W * H; ++i) {
		float len = normals[i].Length();
		if (len > 1.0e-6f)
			normals[i] = normals[i] / len;
		else
			normals[i] = JPH::Vec3::sAxisY();
	}

	// --- Step 3: Build indexed vertex/triangle arrays ---
	std::vector<JPH::DebugRenderer::Vertex> vertices;
	std::vector<uint32> indices;
	vertices.reserve(W * H);

	// Map from grid index to vertex index (-1 = not emitted)
	std::vector<int> vtxMap(W * H, -1);

	for (uint32_t y = 0; y < H - 1; ++y) {
		for (uint32_t x = 0; x < W - 1; ++x) {
			uint32_t i00 = y * W + x;
			uint32_t i10 = i00 + 1;
			uint32_t i01 = i00 + W;
			uint32_t i11 = i00 + W + 1;

			// Need at least one wet vertex in expanded mask to draw this quad
			if (!wetExpanded[i00] && !wetExpanded[i10] && !wetExpanded[i01] && !wetExpanded[i11])
				continue;

			// Helper lambda: emit vertex if not yet emitted
			auto emitVertex = [&](uint32_t gx, uint32_t gy, uint32_t gi) -> uint32 {
				if (vtxMap[gi] >= 0)
					return (uint32)vtxMap[gi];

				JPH::DebugRenderer::Vertex v;
				JPH::Vec3 pos = offset + JPH::Vec3(gx * gs, surfaceH[gi], gy * gs);
				v.mPosition = { pos.GetX(), pos.GetY(), pos.GetZ() };

				JPH::Vec3 n = normals[gi];
				v.mNormal = { n.GetX(), n.GetY(), n.GetZ() };
				v.mUV = { float(gx) / float(W), float(gy) / float(H) };

				// Depth-based water color: deeper = darker blue-green, shallow = lighter
				float depth = state.waterDepth[gi];
				float t = std::min(depth / 4.0f, 1.0f);  // normalize to [0,1] over 4m depth
				// Shallow: light cyan (100, 200, 220, 160), Deep: dark ocean blue (10, 40, 100, 210)
				uint8_t cr = (uint8_t)(100 - t * 90);
				uint8_t cg = (uint8_t)(200 - t * 160);
				uint8_t cb = (uint8_t)(220 - t * 120);
				uint8_t ca = (uint8_t)(230 + t * 25);
				v.mColor = JPH::Color(cr, cg, cb, ca);

				uint32 vi = (uint32)vertices.size();
				vertices.push_back(v);
				vtxMap[gi] = (int)vi;
				return vi;
			};

			uint32 v00 = emitVertex(x, y, i00);
			uint32 v10 = emitVertex(x + 1, y, i10);
			uint32 v01 = emitVertex(x, y + 1, i01);
			uint32 v11 = emitVertex(x + 1, y + 1, i11);

			// Upper-left triangle
			indices.push_back(v00);
			indices.push_back(v10);
			indices.push_back(v01);

			// Lower-right triangle (completes the quad)
			indices.push_back(v10);
			indices.push_back(v11);
			indices.push_back(v01);
		}
	}

	// --- Step 3a: Bottom mesh — terrain-following underside fills the water volume ---
	for (uint32_t y = 0; y < H - 1; ++y) {
		for (uint32_t x = 0; x < W - 1; ++x) {
			uint32_t i00 = y * W + x, i10 = i00 + 1, i01 = i00 + W, i11 = i00 + W + 1;
			if (!wetExpanded[i00] && !wetExpanded[i10] && !wetExpanded[i01] && !wetExpanded[i11])
				continue;

			auto emitBtm = [&](uint32_t gx, uint32_t gy, uint32_t gi) -> uint32 {
				JPH::DebugRenderer::Vertex v;
				JPH::Vec3 pos = offset + JPH::Vec3(gx * gs, state.terrainHeight[gi], gy * gs);
				v.mPosition = { pos.GetX(), pos.GetY(), pos.GetZ() };
				v.mNormal   = { 0.0f, -1.0f, 0.0f };
				v.mUV       = { 0.0f, 0.0f };
				v.mColor    = JPH::Color(8, 20, 55, 255);
				uint32 vi   = (uint32)vertices.size();
				vertices.push_back(v);
				return vi;
			};

			uint32 b00 = emitBtm(x, y, i00), b10 = emitBtm(x+1, y, i10);
			uint32 b01 = emitBtm(x, y+1, i01), b11 = emitBtm(x+1, y+1, i11);
			// Reversed winding for downward-facing
			indices.push_back(b00); indices.push_back(b01); indices.push_back(b10);
			indices.push_back(b10); indices.push_back(b01); indices.push_back(b11);
		}
	}

	// --- Step 3.5: Edge skirts — solid walls from water surface down to terrain ---
	{
		uint32_t qW = W - 1, qH = H - 1;
		std::vector<bool> qR(qW * qH, false);
		for (uint32_t qy = 0; qy < qH; ++qy)
			for (uint32_t qx = 0; qx < qW; ++qx) {
				uint32_t i = qy * W + qx;
				if (wetExpanded[i] || wetExpanded[i+1] || wetExpanded[i+W] || wetExpanded[i+W+1])
					qR[qy * qW + qx] = true;
			}

		auto skirtVtx = [&](float wx, float wy, float wz) -> uint32 {
			JPH::DebugRenderer::Vertex v;
			v.mPosition = { wx, wy, wz };
			v.mNormal   = { 0.0f, -1.0f, 0.0f };
			v.mUV       = { 0.0f, 0.0f };
			v.mColor    = JPH::Color(15, 40, 100, 230);
			uint32 vi   = (uint32)vertices.size();
			vertices.push_back(v);
			return vi;
		};

		auto skirtQuad = [&](float ax, float atop, float abot, float az,
		                      float bx, float btop, float bbot, float bz) {
			uint32 t0 = skirtVtx(ax, atop, az), b0 = skirtVtx(ax, abot, az);
			uint32 t1 = skirtVtx(bx, btop, bz), b1 = skirtVtx(bx, bbot, bz);
			indices.push_back(t0); indices.push_back(t1); indices.push_back(b0);
			indices.push_back(t1); indices.push_back(b1); indices.push_back(b0);
		};

		float oX = offset.GetX(), oZ = offset.GetZ();
		for (uint32_t qy = 0; qy < qH; ++qy)
			for (uint32_t qx = 0; qx < qW; ++qx) {
				if (!qR[qy * qW + qx]) continue;
				uint32_t i00 = qy*W + qx, i10 = i00+1, i01 = i00+W, i11 = i01+1;
				float x0 = oX + qx * gs, x1 = oX + (qx+1) * gs;
				float z0 = oZ + qy * gs, z1 = oZ + (qy+1) * gs;
				if (qx == 0 || !qR[qy*qW + qx-1])
					skirtQuad(x0, surfaceH[i00], state.terrainHeight[i00], z0,
					          x0, surfaceH[i01], state.terrainHeight[i01], z1);
				if (qx+1 >= qW || !qR[qy*qW + qx+1])
					skirtQuad(x1, surfaceH[i10], state.terrainHeight[i10], z0,
					          x1, surfaceH[i11], state.terrainHeight[i11], z1);
				if (qy == 0 || !qR[(qy-1)*qW + qx])
					skirtQuad(x0, surfaceH[i00], state.terrainHeight[i00], z0,
					          x1, surfaceH[i10], state.terrainHeight[i10], z0);
				if (qy+1 >= qH || !qR[(qy+1)*qW + qx])
					skirtQuad(x0, surfaceH[i01], state.terrainHeight[i01], z1,
					          x1, surfaceH[i11], state.terrainHeight[i11], z1);
			}
	}

	// --- Step 4: Submit the batch to the debug renderer ---
	if (!vertices.empty() && !indices.empty()) {
		JPH::DebugRenderer::Batch batch = mDebugRenderer->CreateTriangleBatch(
			vertices.data(), (int)vertices.size(),
			indices.data(), (int)indices.size());

		JPH::AABox bounds;
		for (auto& v : vertices)
			bounds.Encapsulate(JPH::Vec3(v.mPosition.x, v.mPosition.y, v.mPosition.z));

		JPH::DebugRenderer::GeometryRef geom = new JPH::DebugRenderer::Geometry(batch, bounds);

		mDebugRenderer->DrawGeometry(
			JPH::RMat44::sIdentity(),
			JPH::Color(255, 255, 255, 255),
			geom,
			JPH::DebugRenderer::ECullMode::Off,
			JPH::DebugRenderer::ECastShadow::Off,
			JPH::DebugRenderer::EDrawMode::Solid);
	}
}
