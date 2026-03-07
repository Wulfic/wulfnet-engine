#include <Framework.h>
#include <Tests/WulfNet/WaterBoxTest.h>
#include <WulfNet/Jolt/Physics/Body/BodyCreationSettings.h>
#include <WulfNet/Jolt/Physics/Collision/Shape/BoxShape.h>
#include <WulfNet/Jolt/Physics/Collision/Shape/SphereShape.h>
#include <WulfNet/Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <SamplesLayers.h>
#include <Renderer/DebugRendererImp.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iomanip>

JPH_IMPLEMENT_RTTI_VIRTUAL(WaterBoxTest)
{
	JPH_ADD_BASE_CLASS(WaterBoxTest, Test)
}

// ============================================================================
// Initialize — create the glass box, water system, and floating objects
// ============================================================================
void WaterBoxTest::Initialize()
{
	// ---- Floor ----
	CreateFloor();

	// ---- Water System (high resolution, small area) ----
	WulfNet::FluidSystemConfig config;
	config.width    = cGridW;
	config.height   = cGridH;
	config.gridSize = cGridSize;
	config.gravity  = 9.81f;
	// Center the grid at world origin
	config.originX  = -(config.width  * config.gridSize) / 2.0f;
	config.originZ  = -(config.height * config.gridSize) / 2.0f;

	mWaterSystem = new WulfNet::FluidSystem(config, mPhysicsSystem);

	// ---- Terrain: flat floor inside the box, raised walls outside ----
	auto& state = mWaterSystem->GetCPUState();
	const float halfCells = cBoxHalfExtent / cGridSize;   // cells from center to wall
	const float center    = cGridW / 2.0f;

	for (uint32_t y = 0; y < cGridH; ++y) {
		for (uint32_t x = 0; x < cGridW; ++x) {
			uint32_t idx = y * cGridW + x;
			float dx = (float)x - center;
			float dy = (float)y - center;

			// Inside the box: flat floor at y=0
			// Outside the box: raised terrain to act as containment
			if (std::abs(dx) <= halfCells && std::abs(dy) <= halfCells)
				state.terrainHeight[idx] = 0.0f;
			else
				state.terrainHeight[idx] = cBoxWallHeight + 2.0f;  // well above wall top
		}
	}

	// ---- Glass box walls (4 static bodies) ----
	// These are thin, tall box shapes positioned at each edge.
	// They serve as physical containment and are rendered transparently.
	auto createWall = [&](JPH::Vec3 pos, JPH::Vec3 halfExtents) {
		JPH::RefConst<JPH::Shape> wallShape = new JPH::BoxShape(halfExtents);
		JPH::BodyCreationSettings bcs(wallShape, pos, JPH::Quat::sIdentity(),
			JPH::EMotionType::Static, Layers::NON_MOVING);
		JPH::Body* body = mBodyInterface->CreateBody(bcs);
		mBodyInterface->AddBody(body->GetID(), JPH::EActivation::DontActivate);
		mWallBodies.push_back(body->GetID());
	};

	float wallHalfH = cBoxWallHeight / 2.0f;
	float wallCenterY = wallHalfH;
	float ext = cBoxHalfExtent;
	float thick = cBoxWallThick;

	// +X wall
	createWall(JPH::Vec3( ext + thick, wallCenterY, 0.0f),
	           JPH::Vec3(thick, wallHalfH, ext + thick));
	// -X wall
	createWall(JPH::Vec3(-ext - thick, wallCenterY, 0.0f),
	           JPH::Vec3(thick, wallHalfH, ext + thick));
	// +Z wall
	createWall(JPH::Vec3(0.0f, wallCenterY,  ext + thick),
	           JPH::Vec3(ext + thick, wallHalfH, thick));
	// -Z wall
	createWall(JPH::Vec3(0.0f, wallCenterY, -ext - thick),
	           JPH::Vec3(ext + thick, wallHalfH, thick));

	// ---- Seed a small initial pool so there's something to see immediately ----
	float fillRadius = 12.0f / cGridSize;  // ~12m radius
	for (uint32_t gy = 0; gy < cGridH; ++gy) {
		for (uint32_t gx = 0; gx < cGridW; ++gx) {
			float dx = (float)gx - center;
			float dy = (float)gy - center;
			float dist = std::sqrt(dx * dx + dy * dy);
			if (dist < fillRadius && std::abs(dx) <= halfCells && std::abs(dy) <= halfCells)
				mWaterSystem->AddWater(gx, gy, 0.8f);  // gentle initial fill
		}
	}

	// ---- Floating objects ----
	JPH::RefConst<JPH::Shape> boxShape      = new JPH::BoxShape(JPH::Vec3(0.4f, 0.4f, 0.4f));
	JPH::RefConst<JPH::Shape> sphereShape   = new JPH::SphereShape(0.4f);
	JPH::RefConst<JPH::Shape> cylinderShape = new JPH::CylinderShape(0.6f, 0.25f);

	for (int i = 0; i < 30; ++i) {
		JPH::Vec3 pos(
			(float(rand()) / RAND_MAX) * 12.0f - 6.0f,
			6.0f + (float(rand()) / RAND_MAX) * 4.0f,
			(float(rand()) / RAND_MAX) * 12.0f - 6.0f
		);

		JPH::RefConst<JPH::Shape> shape;
		int shapeType = rand() % 3;
		if (shapeType == 0) shape = boxShape;
		else if (shapeType == 1) shape = sphereShape;
		else shape = cylinderShape;

		JPH::BodyCreationSettings bcs(shape, pos, JPH::Quat::sIdentity(),
			JPH::EMotionType::Dynamic, Layers::MOVING);
		bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateMassAndInertia;
		bcs.mMassPropertiesOverride.mMass = 200.0f * shape->GetVolume();  // ~200 kg/m³ (floats)

		JPH::Body* body = mBodyInterface->CreateBody(bcs);
		mBodyInterface->AddBody(body->GetID(), JPH::EActivation::Activate);
		mFloatingBodies.push_back(body->GetID());
	}

	// ---- System monitor ----
	WulfNet::SystemMonitor::Get().Initialize();
}

// ============================================================================
// Pre-physics update — emitter, simulation step, rendering
// ============================================================================
void WaterBoxTest::PrePhysicsUpdate(const PreUpdateParams &inParams)
{
	if (!mWaterSystem) return;

	mTime += inParams.mDeltaTime;

	// ---- FPS / performance stats (update every 0.5s) ----
	mFrameCount++;
	mStatsTimer += inParams.mDeltaTime;
	if (mStatsTimer >= 0.5f) {
		mCurrentFPS  = (float)mFrameCount / mStatsTimer;
		mFrameTimeMs = (mStatsTimer / (float)mFrameCount) * 1000.0f;
		mFrameCount  = 0;
		mStatsTimer  = 0.0f;
		WulfNet::SystemMonitor::Get().Update();
	}

	// ---- Water emitter: add water from center-top in a small radius ----
	if (mEmitterActive) {
		const auto& config = mWaterSystem->GetConfig();
		float centerX = config.width  / 2.0f;
		float centerY = config.height / 2.0f;
		float emitRadius = 3.0f / cGridSize;   // ~3m radius emission area

		// Accumulate volume based on rate and delta time
		mEmitterAccum += mEmitterRate * inParams.mDeltaTime;

		// Distribute accumulated volume across emitter cells
		if (mEmitterAccum > 0.01f) {
			int cellCount = 0;
			// Count cells in emitter area
			for (int dy = -(int)emitRadius; dy <= (int)emitRadius; ++dy) {
				for (int dx = -(int)emitRadius; dx <= (int)emitRadius; ++dx) {
					if (dx * dx + dy * dy <= (int)(emitRadius * emitRadius))
						cellCount++;
				}
			}

			if (cellCount > 0) {
				float perCell = mEmitterAccum / (float)cellCount;
				for (int dy = -(int)emitRadius; dy <= (int)emitRadius; ++dy) {
					for (int dx = -(int)emitRadius; dx <= (int)emitRadius; ++dx) {
						if (dx * dx + dy * dy > (int)(emitRadius * emitRadius))
							continue;
						int gx = (int)centerX + dx;
						int gy = (int)centerY + dy;
						if (gx >= 0 && gx < (int)config.width && gy >= 0 && gy < (int)config.height)
							mWaterSystem->AddWater((uint32_t)gx, (uint32_t)gy, perCell);
					}
				}
				mEmitterAccum = 0.0f;
			}
		}

		// Stop emitter once water level is high enough (~6m)
		auto& st = mWaterSystem->GetCPUState();
		uint32_t midIdx = (cGridH / 2) * cGridW + (cGridW / 2);
		if (st.waterDepth[midIdx] > 6.0f)
			mEmitterActive = false;
	}

	// ---- Step water simulation ----
	mWaterSystem->StepSimulationCPU(inParams.mDeltaTime);

	// ---- Buoyancy ----
	auto& context = mWaterSystem->GetJobContextForTesting();
	context.interactingBodies = mFloatingBodies;
	mWaterSystem->ApplyBuoyancyForces(mJobSystem);

	// ---- Render glass box ----
	RenderGlassBox();

	// ---- Render water surface (matching V3 reference exactly) ----
	const auto& config = mWaterSystem->GetConfig();
	const auto& state  = mWaterSystem->GetCPUState();
	const uint32_t W  = config.width;
	const uint32_t H  = config.height;
	const float gs     = config.gridSize;

	JPH::Vec3 offset(config.originX, 0.0f, config.originZ);

	// -- Compute per-vertex surface heights --
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

	// Expand wet mask by 3 cells for smooth edges
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

	// -- Smooth surface heights (3-pass box filter) --
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

	// -- Compute per-vertex normals (area-weighted) --
	std::vector<JPH::Vec3> normals(W * H, JPH::Vec3::sZero());

	for (uint32_t y = 0; y < H - 1; ++y) {
		for (uint32_t x = 0; x < W - 1; ++x) {
			uint32_t i00 = y * W + x;
			uint32_t i10 = i00 + 1;
			uint32_t i01 = i00 + W;
			uint32_t i11 = i00 + W + 1;

			if (!wetExpanded[i00] && !wetExpanded[i10] && !wetExpanded[i01] && !wetExpanded[i11])
				continue;

			JPH::Vec3 p00 = offset + JPH::Vec3(x * gs,       surfaceH[i00], y * gs);
			JPH::Vec3 p10 = offset + JPH::Vec3((x + 1) * gs, surfaceH[i10], y * gs);
			JPH::Vec3 p01 = offset + JPH::Vec3(x * gs,       surfaceH[i01], (y + 1) * gs);
			JPH::Vec3 p11 = offset + JPH::Vec3((x + 1) * gs, surfaceH[i11], (y + 1) * gs);

			JPH::Vec3 n1 = (p10 - p00).Cross(p01 - p00);
			normals[i00] += n1;  normals[i10] += n1;  normals[i01] += n1;

			JPH::Vec3 n2 = (p11 - p10).Cross(p01 - p10);
			normals[i10] += n2;  normals[i11] += n2;  normals[i01] += n2;
		}
	}

	for (uint32_t i = 0; i < W * H; ++i) {
		float len = normals[i].Length();
		if (len > 1.0e-6f)
			normals[i] = normals[i] / len;
		else
			normals[i] = JPH::Vec3::sAxisY();
	}

	// -- Build indexed mesh (top surface + bottom + edge skirts) --
	std::vector<JPH::DebugRenderer::Vertex> vertices;
	std::vector<uint32> indices;
	vertices.reserve(W * H);

	std::vector<int> vtxMap(W * H, -1);

	for (uint32_t y = 0; y < H - 1; ++y) {
		for (uint32_t x = 0; x < W - 1; ++x) {
			uint32_t i00 = y * W + x;
			uint32_t i10 = i00 + 1;
			uint32_t i01 = i00 + W;
			uint32_t i11 = i00 + W + 1;

			if (!wetExpanded[i00] && !wetExpanded[i10] && !wetExpanded[i01] && !wetExpanded[i11])
				continue;

			auto emitVertex = [&](uint32_t gx, uint32_t gy, uint32_t gi) -> uint32 {
				if (vtxMap[gi] >= 0)
					return (uint32)vtxMap[gi];

				JPH::DebugRenderer::Vertex v;
				JPH::Vec3 pos = offset + JPH::Vec3(gx * gs, surfaceH[gi], gy * gs);
				v.mPosition = { pos.GetX(), pos.GetY(), pos.GetZ() };

				JPH::Vec3 n = normals[gi];
				v.mNormal = { n.GetX(), n.GetY(), n.GetZ() };
				v.mUV = { float(gx) / float(W), float(gy) / float(H) };

				// Depth-based water color: deeper = darker blue-green
				float depth = state.waterDepth[gi];
				float t = std::min(depth / 4.0f, 1.0f);
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

			indices.push_back(v00);  indices.push_back(v10);  indices.push_back(v01);
			indices.push_back(v10);  indices.push_back(v11);  indices.push_back(v01);
		}
	}

	// -- Bottom mesh (terrain-following underside) --
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
			indices.push_back(b00); indices.push_back(b01); indices.push_back(b10);
			indices.push_back(b10); indices.push_back(b01); indices.push_back(b11);
		}
	}

	// -- Edge skirts (solid walls from surface down to terrain at water boundary) --
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

	// -- Submit water mesh to debug renderer --
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

// ============================================================================
// Render the 4 glass box walls as semi-transparent blue-tinted panels
// ============================================================================
void WaterBoxTest::RenderGlassBox()
{
	float ext   = cBoxHalfExtent;
	float h     = cBoxWallHeight;
	JPH::Color glass(180, 220, 240, 60);  // light blue, very transparent

	// Helper: draw a single rectangular wall panel as two triangles
	auto drawPanel = [&](JPH::Vec3 a, JPH::Vec3 b, JPH::Vec3 c, JPH::Vec3 d) {
		JPH::Vec3 normal = (b - a).Cross(d - a);
		float len = normal.Length();
		if (len > 1e-6f) normal = normal / len;

		JPH::DebugRenderer::Vertex va, vb, vc, vd;
		va.mPosition = { a.GetX(), a.GetY(), a.GetZ() };
		vb.mPosition = { b.GetX(), b.GetY(), b.GetZ() };
		vc.mPosition = { c.GetX(), c.GetY(), c.GetZ() };
		vd.mPosition = { d.GetX(), d.GetY(), d.GetZ() };

		va.mNormal = vb.mNormal = vc.mNormal = vd.mNormal =
			{ normal.GetX(), normal.GetY(), normal.GetZ() };
		va.mUV = { 0, 0 };  vb.mUV = { 1, 0 };
		vc.mUV = { 1, 1 };  vd.mUV = { 0, 1 };
		va.mColor = vb.mColor = vc.mColor = vd.mColor = glass;

		JPH::DebugRenderer::Vertex verts[] = { va, vb, vc, vd };
		uint32 idxs[] = { 0, 1, 3, 1, 2, 3 };

		JPH::DebugRenderer::Batch batch = mDebugRenderer->CreateTriangleBatch(verts, 4, idxs, 6);

		JPH::AABox bounds;
		bounds.Encapsulate(a); bounds.Encapsulate(b);
		bounds.Encapsulate(c); bounds.Encapsulate(d);

		JPH::DebugRenderer::GeometryRef geom = new JPH::DebugRenderer::Geometry(batch, bounds);
		mDebugRenderer->DrawGeometry(
			JPH::RMat44::sIdentity(),
			JPH::Color(255, 255, 255, 255),
			geom,
			JPH::DebugRenderer::ECullMode::Off,
			JPH::DebugRenderer::ECastShadow::Off,
			JPH::DebugRenderer::EDrawMode::Solid);
	};

	// 4 walls: each is a quad from floor (y=0) to wall top (y=h)
	// +X wall
	drawPanel(JPH::Vec3( ext, 0, -ext), JPH::Vec3( ext, 0,  ext),
	          JPH::Vec3( ext, h,  ext), JPH::Vec3( ext, h, -ext));
	// -X wall
	drawPanel(JPH::Vec3(-ext, 0,  ext), JPH::Vec3(-ext, 0, -ext),
	          JPH::Vec3(-ext, h, -ext), JPH::Vec3(-ext, h,  ext));
	// +Z wall
	drawPanel(JPH::Vec3(-ext, 0,  ext), JPH::Vec3( ext, 0,  ext),
	          JPH::Vec3( ext, h,  ext), JPH::Vec3(-ext, h,  ext));
	// -Z wall
	drawPanel(JPH::Vec3( ext, 0, -ext), JPH::Vec3(-ext, 0, -ext),
	          JPH::Vec3(-ext, h, -ext), JPH::Vec3( ext, h, -ext));
}

// ============================================================================
// Camera — angled overhead view looking into the glass box
// ============================================================================
void WaterBoxTest::GetInitialCamera(CameraState &ioState) const
{
	// Position above and to the side for a clear view into the box
	ioState.mPos = JPH::RVec3(18.0f, 14.0f, 18.0f);
	ioState.mForward = JPH::Vec3(-0.57f, -0.50f, -0.57f).Normalized();
}

// ============================================================================
// Status overlay — FPS, system stats, water info
// ============================================================================
String WaterBoxTest::GetStatusString() const
{
	const WulfNet::SystemStats &sys = WulfNet::SystemMonitor::Get().GetStats();

	std::ostringstream oss;
	oss << std::fixed;

	oss << "FPS: " << std::setprecision(1) << mCurrentFPS
		<< "  (" << std::setprecision(2) << mFrameTimeMs << " ms)\n";

	oss << std::setprecision(1);
	oss << "CPU: " << sys.cpuUsagePercent << "%  "
		<< "RAM: " << WulfNet::FormatBytes(sys.processMemoryBytes)
		<< " / " << WulfNet::FormatBytes(sys.ramTotalBytes)
		<< " (" << sys.ramUsagePercent << "%)\n";

	if (sys.gpuUsageAvailable) {
		oss << "GPU: " << sys.gpuUsagePercent << "%";
		if (!sys.gpuName.empty())
			oss << " (" << sys.gpuName << ")";
		oss << "\n";
	}

	if (sys.vramUsageAvailable) {
		oss << "VRAM: " << WulfNet::FormatBytes(sys.vramUsedBytes)
			<< " / " << WulfNet::FormatBytes(sys.vramTotalBytes)
			<< " (" << sys.vramUsagePercent << "%)\n";
	}

	// Water info
	oss << "\n";
	if (mWaterSystem) {
		const auto& st = mWaterSystem->GetCPUState();
		uint32_t midIdx = (cGridH / 2) * cGridW + (cGridW / 2);
		float centerDepth = st.waterDepth[midIdx];
		oss << "Water depth (center): " << std::setprecision(2) << centerDepth << " m\n";
		oss << "Emitter: " << (mEmitterActive ? "ACTIVE" : "STOPPED") << "\n";
	}
	oss << "Bodies: " << mFloatingBodies.size() << "\n";
	oss << "Time: " << std::setprecision(1) << mTime << " s\n";

	return String(oss.str().c_str());
}
