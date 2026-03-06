// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT
// Modified: WulfNet V3 Water Physics Integration

#include <Samples.h>

#include <Tests/Water/WaterShapeTest.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/MutableCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>
#include <algorithm>
#include <vector>

JPH_IMPLEMENT_RTTI_VIRTUAL(WaterShapeTest)
{
	JPH_ADD_BASE_CLASS(WaterShapeTest, Test)
}

void WaterShapeTest::Initialize()
{
	CreateFloor();

	// --- Initialize WulfNet V3 Water System ---
	WulfNet::Physics::WaterSystemV3Config config;
	config.width    = 128;
	config.height   = 128;
	config.gridSize = 1.0f;
	config.gravity  = 9.81f;
	config.fluxDamping = 0.3f;
	config.originX  = -(config.width * config.gridSize) / 2.0f;
	config.originZ  = -(config.height * config.gridSize) / 2.0f;

	mWaterSystem = new WulfNet::Physics::WaterSystemV3(config, mPhysicsSystem);
	auto &state = mWaterSystem->GetCPUState();

	float poolCenterX = config.width / 2.0f;
	float poolCenterZ = config.height / 2.0f;
	float waterRadius = 50.0f;

	for (uint32_t y = 0; y < config.height; ++y)
		for (uint32_t x = 0; x < config.width; ++x) {
			uint32_t idx = y * config.width + x;
			float dx = x - poolCenterX;
			float dz = y - poolCenterZ;
			float dist = std::sqrt(dx * dx + dz * dz);

			// Gentle bowl terrain — rises slightly toward edges
			state.terrainHeight[idx] = 7.0f + dist * dist * 0.001f;

			// Fill water in circular region with smooth edge taper
			if (dist < waterRadius) {
				float edgeFade = (dist > waterRadius - 10.0f)
					? (waterRadius - dist) / 10.0f : 1.0f;
				mWaterSystem->AddWater(x, y, 3.0f * edgeFade);
			}
		}

	// --- Create test shapes (same variety as original, dropped from y=20) ---
	auto addBody = [&](const ShapeSettings *inShape, RVec3Arg inPos, float inMass) {
		BodyCreationSettings bcs(inShape, inPos, Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING);
		bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateMassAndInertia;
		bcs.mMassPropertiesOverride.mMass = inMass;
		Body *b = mBodyInterface->CreateBody(bcs);
		mBodyInterface->AddBody(b->GetID(), EActivation::Activate);
		mFloatingBodies.push_back(b->GetID());
	};

	auto addBodyDirect = [&](const Shape *inShape, RVec3Arg inPos, float inMass) {
		BodyCreationSettings bcs(inShape, inPos, Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING);
		bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateMassAndInertia;
		bcs.mMassPropertiesOverride.mMass = inMass;
		Body *b = mBodyInterface->CreateBody(bcs);
		mBodyInterface->AddBody(b->GetID(), EActivation::Activate);
		mFloatingBodies.push_back(b->GetID());
	};

	// Scaled box
	addBodyDirect(new ScaledShape(new BoxShape(Vec3(1.0f, 2.0f, 2.5f)), Vec3(0.5f, 0.6f, -0.7f)), RVec3(-10, 20, 0), 300.0f);

	// Box
	addBodyDirect(new BoxShape(Vec3(1.0f, 2.0f, 2.5f)), RVec3(-7, 20, 0), 400.0f);

	// Sphere
	addBodyDirect(new SphereShape(2.0f), RVec3(-3, 20, 0), 300.0f);

	// Static compound
	Ref<StaticCompoundShapeSettings> static_compound = new StaticCompoundShapeSettings;
	static_compound->AddShape(Vec3(2.0f, 0, 0), Quat::sIdentity(), new SphereShape(2.0f));
	static_compound->AddShape(Vec3(-1.0f, 0, 0), Quat::sIdentity(), new SphereShape(1.0f));
	addBody(static_compound, RVec3(3, 20, 0), 400.0f);

	// Tetrahedron
	Array<Vec3> tetrahedron;
	tetrahedron.push_back(Vec3(-2, 0, -2));
	tetrahedron.push_back(Vec3(0, 0, 2));
	tetrahedron.push_back(Vec3(2, 0, -2));
	tetrahedron.push_back(Vec3(0, -2, 0));
	Ref<ConvexHullShapeSettings> tetrahedron_shape = new ConvexHullShapeSettings(tetrahedron);
	addBody(tetrahedron_shape, RVec3(10, 20, 0), 250.0f);

	// Non-uniform scaled tetrahedron
	addBody(new ScaledShapeSettings(tetrahedron_shape, Vec3(1, -1.5f, 2.0f)), RVec3(15, 20, 0), 250.0f);

	// Convex hull box
	Array<Vec3> box;
	box.push_back(Vec3(1.5f, 1.0f, 0.5f));  box.push_back(Vec3(-1.5f, 1.0f, 0.5f));
	box.push_back(Vec3(1.5f, -1.0f, 0.5f)); box.push_back(Vec3(-1.5f, -1.0f, 0.5f));
	box.push_back(Vec3(1.5f, 1.0f, -0.5f)); box.push_back(Vec3(-1.5f, 1.0f, -0.5f));
	box.push_back(Vec3(1.5f, -1.0f, -0.5f));box.push_back(Vec3(-1.5f, -1.0f, -0.5f));
	addBody(new ConvexHullShapeSettings(box), RVec3(18, 20, 0), 300.0f);

	// Random convex shape
	default_random_engine random;
	uniform_real_distribution<float> hull_size(0.1f, 1.9f);
	Array<Vec3> points;
	for (int j = 0; j < 20; ++j)
		points.push_back(hull_size(random) * Vec3::sRandom(random));
	addBody(new ConvexHullShapeSettings(points), RVec3(21, 20, 0), 200.0f);

	// Mutable compound
	Ref<MutableCompoundShapeSettings> mutable_compound = new MutableCompoundShapeSettings;
	mutable_compound->AddShape(Vec3(1.0f, 0, 0), Quat::sIdentity(), new BoxShape(Vec3(0.5f, 0.75f, 1.0f)));
	mutable_compound->AddShape(Vec3(-1.0f, 0, 0), Quat::sIdentity(), new SphereShape(1.0f));
	addBody(mutable_compound, RVec3(25, 20, 0), 350.0f);

	// Box with offset center of mass
	addBody(new OffsetCenterOfMassShapeSettings(Vec3(-1.0f, 0.0f, 0.0f), new BoxShape(Vec3(2.0f, 0.25f, 0.25f))), RVec3(30, 20, 0), 200.0f);
}

void WaterShapeTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	if (!mWaterSystem) return;

	mTime += inParams.mDeltaTime;

	// Step V3 SWE simulation
	mWaterSystem->StepSimulationCPU(inParams.mDeltaTime);

	// Feed floating bodies into V3 buoyancy system
	auto &context = mWaterSystem->GetJobContextForTesting();
	context.interactingBodies = mFloatingBodies;
	mWaterSystem->ApplyBuoyancyForces(mJobSystem);

	// --- Render smooth V3 water surface ---
	const auto &config = mWaterSystem->GetConfig();
	const auto &state  = mWaterSystem->GetCPUState();
	const uint32_t W  = config.width;
	const uint32_t H  = config.height;
	const float    gs = config.gridSize;
	Vec3 offset(config.originX, 0.0f, config.originZ);

	// Surface heights + wet mask
	std::vector<float> surfaceH(W * H);
	std::vector<bool>  wet(W * H, false);
	for (uint32_t gy = 0; gy < H; ++gy)
		for (uint32_t gx = 0; gx < W; ++gx) {
			uint32_t i = gy * W + gx;
			surfaceH[i] = state.terrainHeight[i] + state.waterDepth[i];
			if (state.waterDepth[i] > 0.01f) wet[i] = true;
		}

	// Expand wet mask by 3 cells for smooth taper
	std::vector<bool> wetExp = wet;
	for (int expand = 0; expand < 3; ++expand) {
		std::vector<bool> prev = wetExp;
		for (uint32_t gy = 0; gy < H; ++gy)
			for (uint32_t gx = 0; gx < W; ++gx)
				if (prev[gy * W + gx]) {
					if (gx > 0)     wetExp[gy * W + (gx-1)] = true;
					if (gx < W-1)   wetExp[gy * W + (gx+1)] = true;
					if (gy > 0)     wetExp[(gy-1) * W + gx]  = true;
					if (gy < H-1)   wetExp[(gy+1) * W + gx]  = true;
				}
	}

	// Smooth surface heights (3-pass, operates on wetExp for smooth boundary)
	for (int pass = 0; pass < 3; ++pass) {
		std::vector<float> tmp = surfaceH;
		for (uint32_t gy = 1; gy < H - 1; ++gy)
			for (uint32_t gx = 1; gx < W - 1; ++gx) {
				uint32_t idx = gy * W + gx;
				if (!wetExp[idx]) continue;
				float sum = 0.0f, wt = 0.0f;
				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx) {
						uint32_t ni = (gy + dy) * W + (gx + dx);
						if (wetExp[ni]) {
							float w = (dx == 0 && dy == 0) ? 4.0f : 1.0f;
							sum += tmp[ni] * w;
							wt += w;
						}
					}
				if (wt > 0.0f) surfaceH[idx] = sum / wt;
			}
	}

	// Smooth per-vertex normals
	std::vector<Vec3> normals(W * H, Vec3::sZero());
	for (uint32_t gy = 0; gy < H - 1; ++gy)
		for (uint32_t gx = 0; gx < W - 1; ++gx) {
			uint32_t i00 = gy * W + gx, i10 = i00 + 1, i01 = i00 + W, i11 = i00 + W + 1;
			if (!wetExp[i00] && !wetExp[i10] && !wetExp[i01] && !wetExp[i11]) continue;
			Vec3 p00 = offset + Vec3(gx * gs,     surfaceH[i00], gy * gs);
			Vec3 p10 = offset + Vec3((gx+1)*gs,   surfaceH[i10], gy * gs);
			Vec3 p01 = offset + Vec3(gx * gs,     surfaceH[i01], (gy+1)*gs);
			Vec3 p11 = offset + Vec3((gx+1)*gs,   surfaceH[i11], (gy+1)*gs);
			Vec3 n1 = (p10 - p00).Cross(p01 - p00);
			normals[i00] += n1; normals[i10] += n1; normals[i01] += n1;
			Vec3 n2 = (p11 - p10).Cross(p01 - p10);
			normals[i10] += n2; normals[i11] += n2; normals[i01] += n2;
		}
	for (uint32_t i = 0; i < W * H; ++i) {
		float len = normals[i].Length();
		normals[i] = len > 1e-6f ? normals[i] / len : Vec3::sAxisY();
	}

	// Build indexed mesh
	std::vector<DebugRenderer::Vertex> vertices;
	std::vector<uint32> indices;
	vertices.reserve(W * H);
	std::vector<int> vtxMap(W * H, -1);

	for (uint32_t gy = 0; gy < H - 1; ++gy)
		for (uint32_t gx = 0; gx < W - 1; ++gx) {
			uint32_t i00 = gy * W + gx, i10 = i00 + 1, i01 = i00 + W, i11 = i00 + W + 1;
			if (!wetExp[i00] && !wetExp[i10] && !wetExp[i01] && !wetExp[i11]) continue;

			auto emitVtx = [&](uint32_t vx, uint32_t vy, uint32_t vi) -> uint32 {
				if (vtxMap[vi] >= 0) return (uint32)vtxMap[vi];
				DebugRenderer::Vertex v;
				Vec3 pos = offset + Vec3(vx * gs, surfaceH[vi], vy * gs);
				v.mPosition = { pos.GetX(), pos.GetY(), pos.GetZ() };
				v.mNormal   = { normals[vi].GetX(), normals[vi].GetY(), normals[vi].GetZ() };
				v.mUV       = { float(vx) / float(W), float(vy) / float(H) };
				float depth = state.waterDepth[vi];
				float t = std::min(depth / 4.0f, 1.0f);
				v.mColor = Color((uint8)(100 - t*90), (uint8)(200 - t*160), (uint8)(220 - t*120), (uint8)(230 + t*25));
				uint32 idx = (uint32)vertices.size();
				vertices.push_back(v);
				vtxMap[vi] = (int)idx;
				return idx;
			};

			uint32 v00 = emitVtx(gx, gy, i00), v10 = emitVtx(gx+1, gy, i10);
			uint32 v01 = emitVtx(gx, gy+1, i01), v11 = emitVtx(gx+1, gy+1, i11);
			indices.push_back(v00); indices.push_back(v10); indices.push_back(v01);
			indices.push_back(v10); indices.push_back(v11); indices.push_back(v01);
		}

	// Bottom mesh: terrain-following underside fills the water volume
	for (uint32_t gy = 0; gy < H - 1; ++gy)
		for (uint32_t gx = 0; gx < W - 1; ++gx) {
			uint32_t i00 = gy * W + gx, i10 = i00 + 1, i01 = i00 + W, i11 = i00 + W + 1;
			if (!wetExp[i00] && !wetExp[i10] && !wetExp[i01] && !wetExp[i11]) continue;

			auto emitBtm = [&](uint32_t vx, uint32_t vy, uint32_t vi) -> uint32 {
				DebugRenderer::Vertex v;
				Vec3 pos = offset + Vec3(vx * gs, state.terrainHeight[vi], vy * gs);
				v.mPosition = { pos.GetX(), pos.GetY(), pos.GetZ() };
				v.mNormal   = { 0.0f, -1.0f, 0.0f };
				v.mUV       = { 0.0f, 0.0f };
				v.mColor    = Color(8, 20, 55, 255);
				uint32 idx2 = (uint32)vertices.size();
				vertices.push_back(v);
				return idx2;
			};

			uint32 b00 = emitBtm(gx, gy, i00), b10 = emitBtm(gx+1, gy, i10);
			uint32 b01 = emitBtm(gx, gy+1, i01), b11 = emitBtm(gx+1, gy+1, i11);
			indices.push_back(b00); indices.push_back(b01); indices.push_back(b10);
			indices.push_back(b10); indices.push_back(b01); indices.push_back(b11);
		}

	// Edge skirts: solid walls from water surface down to terrain at boundary
	{
		uint32_t qW = W - 1, qH = H - 1;
		std::vector<bool> qR(qW * qH, false);
		for (uint32_t qy = 0; qy < qH; ++qy)
			for (uint32_t qx = 0; qx < qW; ++qx) {
				uint32_t i = qy * W + qx;
				if (wetExp[i] || wetExp[i+1] || wetExp[i+W] || wetExp[i+W+1])
					qR[qy * qW + qx] = true;
			}

		auto skirtVtx = [&](float wx, float wy, float wz) -> uint32 {
			DebugRenderer::Vertex v;
			v.mPosition = { wx, wy, wz };
			v.mNormal   = { 0.0f, -1.0f, 0.0f };
			v.mUV       = { 0.0f, 0.0f };
			v.mColor    = Color(15, 40, 100, 230);
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

	// Submit
	if (!vertices.empty() && !indices.empty()) {
		DebugRenderer::Batch batch = mDebugRenderer->CreateTriangleBatch(
			vertices.data(), (int)vertices.size(), indices.data(), (int)indices.size());
		AABox bounds;
		for (auto &v : vertices)
			bounds.Encapsulate(Vec3(v.mPosition.x, v.mPosition.y, v.mPosition.z));
		DebugRenderer::GeometryRef geom = new DebugRenderer::Geometry(batch, bounds);
		mDebugRenderer->DrawGeometry(RMat44::sIdentity(), Color::sWhite, geom,
			DebugRenderer::ECullMode::Off, DebugRenderer::ECastShadow::Off, DebugRenderer::EDrawMode::Solid);
	}
}
