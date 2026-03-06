// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2024 Jorrit Rouwe
// SPDX-License-Identifier: MIT
// Modified: WulfNet V3 Water Physics Integration

#include <Samples.h>

#include <Tests/Water/BoatTest.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Layers.h>
#include <Renderer/DebugRendererImp.h>
#include <Input/Keyboard.h>
#include <algorithm>
#include <vector>

JPH_IMPLEMENT_RTTI_VIRTUAL(BoatTest)
{
	JPH_ADD_BASE_CLASS(BoatTest, Test)
}

void BoatTest::Initialize()
{
	// --- Initialize WulfNet V3 Water System ---
	WulfNet::Physics::WaterSystemV3Config config;
	config.width    = 200;
	config.height   = 200;
	config.gridSize = 1.0f;
	config.gravity  = 9.81f;
	config.fluxDamping = 0.2f;  // Lower damping so boat wake persists
	config.originX  = -(config.width * config.gridSize) / 2.0f;
	config.originZ  = -(config.height * config.gridSize) / 2.0f;

	mWaterSystem = new WulfNet::Physics::WaterSystemV3(config, mPhysicsSystem);
	auto &state = mWaterSystem->GetCPUState();

	float oceanCenterX = config.width / 2.0f;
	float oceanCenterZ = config.height / 2.0f;
	float waterRadius = 90.0f;

	for (uint32_t y = 0; y < config.height; ++y)
		for (uint32_t x = 0; x < config.width; ++x) {
			uint32_t idx = y * config.width + x;
			float dx = x - oceanCenterX;
			float dz = y - oceanCenterZ;
			float dist = std::sqrt(dx * dx + dz * dz);

			// Very gentle bowl floor
			state.terrainHeight[idx] = 2.0f + dist * dist * 0.0003f;

			// Fill water in circular region with smooth edge taper
			if (dist < waterRadius) {
				float edgeFade = (dist > waterRadius - 15.0f)
					? (waterRadius - dist) / 15.0f : 1.0f;
				mWaterSystem->AddWater(x, y, 3.0f * edgeFade);
			}
		}

	// Add initial wave perturbation in center
	uint32_t cx = (uint32_t)oceanCenterX, cy = (uint32_t)oceanCenterZ;
	for (int dy = -8; dy <= 8; ++dy)
		for (int dx = -8; dx <= 8; ++dx) {
			float dist = std::sqrt((float)(dx*dx + dy*dy));
			if (dist < 8.0f)
				mWaterSystem->AddWater(cx + dx, cy + dy, 2.0f * (1.0f - dist / 8.0f));
		}

	// Create boat
	ConvexHullShapeSettings boat_hull;
	boat_hull.mPoints = {
		Vec3(-cHalfBoatTopWidth, cHalfBoatHeight, -cHalfBoatLength),
		Vec3(cHalfBoatTopWidth, cHalfBoatHeight, -cHalfBoatLength),
		Vec3(-cHalfBoatTopWidth, cHalfBoatHeight, cHalfBoatLength),
		Vec3(cHalfBoatTopWidth, cHalfBoatHeight, cHalfBoatLength),
		Vec3(-cHalfBoatBottomWidth, -cHalfBoatHeight, -cHalfBoatLength),
		Vec3(cHalfBoatBottomWidth, -cHalfBoatHeight, -cHalfBoatLength),
		Vec3(-cHalfBoatBottomWidth, -cHalfBoatHeight, cHalfBoatLength),
		Vec3(cHalfBoatBottomWidth, -cHalfBoatHeight, cHalfBoatLength),
		Vec3(0, cHalfBoatHeight, cHalfBoatLength + cBoatBowLength)
	};
	boat_hull.SetEmbedded();
	OffsetCenterOfMassShapeSettings com_offset(Vec3(0, -cHalfBoatHeight, 0), &boat_hull);
	com_offset.SetEmbedded();
	RVec3 position(0, 8, 0); // Drop boat above water surface
	BodyCreationSettings boat(&com_offset, position, Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING);
	boat.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
	boat.mMassPropertiesOverride.mMass = cBoatMass;
	mBoatBody = mBodyInterface->CreateBody(boat);
	mBodyInterface->AddBody(mBoatBody->GetID(), EActivation::Activate);
	mFloatingBodies.push_back(mBoatBody->GetID());

	// Create some barrels to float in the water
	default_random_engine random;
	BodyCreationSettings barrel(new CylinderShape(1.0f, 0.7f), RVec3::sZero(), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING);
	barrel.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
	barrel.mMassPropertiesOverride.mMass = cBarrelMass;
	for (int i = 0; i < 10; ++i)
	{
		barrel.mPosition = RVec3(-10.0f + i * 2.0f, 8, 10);
		barrel.mRotation = Quat::sRandom(random);
		Body *b = mBodyInterface->CreateBody(barrel);
		mBodyInterface->AddBody(b->GetID(), EActivation::Activate);
		mFloatingBodies.push_back(b->GetID());
	}

	UpdateCameraPivot();
}

void BoatTest::ProcessInput(const ProcessInputParams &inParams)
{
	// Determine acceleration and brake
	mForward = 0.0f;
	if (inParams.mKeyboard->IsKeyPressed(EKey::Up))
		mForward = 1.0f;
	else if (inParams.mKeyboard->IsKeyPressed(EKey::Down))
		mForward = -1.0f;

	// Steering
	mRight = 0.0f;
	if (inParams.mKeyboard->IsKeyPressed(EKey::Left))
		mRight = -1.0f;
	else if (inParams.mKeyboard->IsKeyPressed(EKey::Right))
		mRight = 1.0f;
}

void BoatTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	if (!mWaterSystem) return;

	mTime += inParams.mDeltaTime;

	// Step V3 SWE simulation
	mWaterSystem->StepSimulationCPU(inParams.mDeltaTime);

	// Feed all floating bodies into the V3 buoyancy system
	auto &context = mWaterSystem->GetJobContextForTesting();
	context.interactingBodies = mFloatingBodies;
	mWaterSystem->ApplyBuoyancyForces(mJobSystem);

	// On user input, assure that the boat is active
	if (mRight != 0.0f || mForward != 0.0f)
		mBodyInterface->ActivateBody(mBoatBody->GetID());

	// Apply propeller forces — only when propeller is below water surface
	// Use V3 water system grid to check water height at propeller position
	const auto &config = mWaterSystem->GetConfig();
	const auto &state  = mWaterSystem->GetCPUState();
	Vec3 gridOffset(config.originX, 0.0f, config.originZ);

	RVec3 propeller_position = mBoatBody->GetWorldTransform() * Vec3(0, -cHalfBoatHeight, -cHalfBoatLength);

	// Convert world position to grid coordinates to sample water height
	float gridX = ((float)propeller_position.GetX() - gridOffset.GetX()) / config.gridSize;
	float gridZ = ((float)propeller_position.GetZ() - gridOffset.GetZ()) / config.gridSize;
	uint32_t gx = (uint32_t)std::max(0.0f, std::min(gridX, (float)(config.width - 1)));
	uint32_t gz = (uint32_t)std::max(0.0f, std::min(gridZ, (float)(config.height - 1)));
	uint32_t gi = gz * config.width + gx;
	float waterSurface = state.terrainHeight[gi] + state.waterDepth[gi];

	if (waterSurface > (float)propeller_position.GetY())
	{
		Vec3 forward = mBoatBody->GetRotation().RotateAxisZ();
		Vec3 right = mBoatBody->GetRotation().RotateAxisX();
		mBoatBody->AddImpulse((forward * mForward * cForwardAcceleration + right * Sign(mForward) * mRight * cSteerAcceleration) * cBoatMass * inParams.mDeltaTime, propeller_position);

		// Also disturb water at propeller position if moving (creates wake)
		if (std::abs(mForward) > 0.1f && gx > 0 && gx < config.width - 1 && gz > 0 && gz < config.height - 1)
			mWaterSystem->AddWater(gx, gz, 0.02f * std::abs(mForward));
	}

	UpdateCameraPivot();

	// --- Render smooth V3 water surface ---
	const uint32_t W  = config.width;
	const uint32_t H  = config.height;
	const float    gs = config.gridSize;
	Vec3 offset = gridOffset;

	// Surface heights + wet mask
	std::vector<float> surfaceH(W * H);
	std::vector<bool>  wet(W * H, false);
	for (uint32_t y = 0; y < H; ++y)
		for (uint32_t x = 0; x < W; ++x) {
			uint32_t idx = y * W + x;
			surfaceH[idx] = state.terrainHeight[idx] + state.waterDepth[idx];
			if (state.waterDepth[idx] > 0.01f) wet[idx] = true;
		}

	// Expand wet mask by 3 cells for smooth taper
	std::vector<bool> wetExp = wet;
	for (int expand = 0; expand < 3; ++expand) {
		std::vector<bool> prev = wetExp;
		for (uint32_t y = 0; y < H; ++y)
			for (uint32_t x = 0; x < W; ++x)
				if (prev[y * W + x]) {
					if (x > 0)     wetExp[y * W + (x-1)] = true;
					if (x < W-1)   wetExp[y * W + (x+1)] = true;
					if (y > 0)     wetExp[(y-1) * W + x]  = true;
					if (y < H-1)   wetExp[(y+1) * W + x]  = true;
				}
	}

	// Smooth surface heights (3-pass, operates on wetExp for smooth boundary)
	for (int pass = 0; pass < 3; ++pass) {
		std::vector<float> tmp = surfaceH;
		for (uint32_t y = 1; y < H - 1; ++y)
			for (uint32_t x = 1; x < W - 1; ++x) {
				uint32_t idx = y * W + x;
				if (!wetExp[idx]) continue;
				float sum = 0.0f, wt = 0.0f;
				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx) {
						uint32_t ni = (y + dy) * W + (x + dx);
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
	for (uint32_t y = 0; y < H - 1; ++y)
		for (uint32_t x = 0; x < W - 1; ++x) {
			uint32_t i00 = y * W + x, i10 = i00 + 1, i01 = i00 + W, i11 = i00 + W + 1;
			if (!wetExp[i00] && !wetExp[i10] && !wetExp[i01] && !wetExp[i11]) continue;
			Vec3 p00 = offset + Vec3(x * gs,     surfaceH[i00], y * gs);
			Vec3 p10 = offset + Vec3((x+1)*gs,   surfaceH[i10], y * gs);
			Vec3 p01 = offset + Vec3(x * gs,     surfaceH[i01], (y+1)*gs);
			Vec3 p11 = offset + Vec3((x+1)*gs,   surfaceH[i11], (y+1)*gs);
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

	for (uint32_t y = 0; y < H - 1; ++y)
		for (uint32_t x = 0; x < W - 1; ++x) {
			uint32_t i00 = y * W + x, i10 = i00 + 1, i01 = i00 + W, i11 = i00 + W + 1;
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

			uint32 v00 = emitVtx(x, y, i00), v10 = emitVtx(x+1, y, i10);
			uint32 v01 = emitVtx(x, y+1, i01), v11 = emitVtx(x+1, y+1, i11);
			indices.push_back(v00); indices.push_back(v10); indices.push_back(v01);
			indices.push_back(v10); indices.push_back(v11); indices.push_back(v01);
		}

	// Bottom mesh: terrain-following underside fills the water volume
	for (uint32_t y = 0; y < H - 1; ++y)
		for (uint32_t x = 0; x < W - 1; ++x) {
			uint32_t i00 = y * W + x, i10 = i00 + 1, i01 = i00 + W, i11 = i00 + W + 1;
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

			uint32 b00 = emitBtm(x, y, i00), b10 = emitBtm(x+1, y, i10);
			uint32 b01 = emitBtm(x, y+1, i01), b11 = emitBtm(x+1, y+1, i11);
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

void BoatTest::SaveInputState(StateRecorder &inStream) const
{
	inStream.Write(mForward);
	inStream.Write(mRight);
}

void BoatTest::RestoreInputState(StateRecorder &inStream)
{
	inStream.Read(mForward);
	inStream.Read(mRight);
}

void BoatTest::GetInitialCamera(CameraState &ioState) const
{
	// Position camera behind boat
	RVec3 cam_tgt = RVec3(0, 0, 5);
	ioState.mPos = RVec3(0, 10, -15);
	ioState.mForward = Vec3(cam_tgt - ioState.mPos).Normalized();
}

void BoatTest::UpdateCameraPivot()
{
	// Pivot is center of boat and rotates with boat around Y axis only
	Vec3 fwd = mBoatBody->GetRotation().RotateAxisZ();
	fwd.SetY(0.0f);
	float len = fwd.Length();
	if (len != 0.0f)
		fwd /= len;
	else
		fwd = Vec3::sAxisZ();
	Vec3 up = Vec3::sAxisY();
	Vec3 right = up.Cross(fwd);
	mCameraPivot = RMat44(Vec4(right, 0), Vec4(up, 0), Vec4(fwd, 0), mBoatBody->GetPosition());
}
