// =============================================================================
// WulfNet Engine - Water Physics Diagnostics
// =============================================================================
// Shared diagnostic logging utilities for all water physics tests.
// Writes detailed per-frame analysis to water_diagnostics.log and console.
// Detects NaN/Inf, energy conservation violations, volume loss, etc.
// =============================================================================

#pragma once

#include <WulfNet/Core/Logging/Logger.h>
#include <WulfNet/Physics/Fluids/COFLIPSystem.h>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <memory>
#include <chrono>
#include <cfloat>

// Category constants for log filtering
#define WATER_LOG_CAT "WaterPhysics"
#define COFLIP_LOG_CAT "CO-FLIP"
#define SWE_LOG_CAT "SWE-V5"

// Convenience macros for water diagnostics logging
#define WATER_LOG_INFO(msg)    WULFNET_INFO(WATER_LOG_CAT, msg)
#define WATER_LOG_WARN(msg)    WULFNET_WARNING(WATER_LOG_CAT, msg)
#define WATER_LOG_ERROR(msg)   WULFNET_ERROR(WATER_LOG_CAT, msg)
#define WATER_LOG_DEBUG(msg)   WULFNET_DEBUG(WATER_LOG_CAT, msg)

#define COFLIP_LOG_INFO(msg)   WULFNET_INFO(COFLIP_LOG_CAT, msg)
#define COFLIP_LOG_WARN(msg)   WULFNET_WARNING(COFLIP_LOG_CAT, msg)
#define COFLIP_LOG_ERROR(msg)  WULFNET_ERROR(COFLIP_LOG_CAT, msg)
#define COFLIP_LOG_DEBUG(msg)  WULFNET_DEBUG(COFLIP_LOG_CAT, msg)

#define SWE_LOG_INFO(msg)      WULFNET_INFO(SWE_LOG_CAT, msg)
#define SWE_LOG_WARN(msg)      WULFNET_WARNING(SWE_LOG_CAT, msg)
#define SWE_LOG_ERROR(msg)     WULFNET_ERROR(SWE_LOG_CAT, msg)
#define SWE_LOG_DEBUG(msg)     WULFNET_DEBUG(SWE_LOG_CAT, msg)

// =============================================================================
// Water Diagnostics Logger — initialises file sink and provides analysis
// =============================================================================
class WaterDiagnostics {
public:
	/// Call once at test startup to attach a file log sink and set verbosity.
	static void Initialize(const std::string& testName)
	{
		auto& logger = WulfNet::Logger::Get();
		logger.SetMinLevel(WulfNet::LogLevel::Debug);

		// Add a console sink if none present (typically already there)
		if (!s_consoleSink)
		{
			s_consoleSink = std::make_shared<WulfNet::ConsoleLogSink>(true);
			logger.AddSink(s_consoleSink);
		}

		// Create/replace the file sink for this test run
		if (s_fileSink)
			logger.RemoveSink(s_fileSink);

		s_fileSink = std::make_shared<WulfNet::FileLogSink>("water_diagnostics.log");
		logger.AddSink(s_fileSink);

		s_frameNumber = 0;
		s_prevEnergy = -1.0f;
		s_prevVolume = -1.0f;
		s_prevParticleCount = 0;
		s_nanDetected = false;

		std::ostringstream oss;
		oss << "====== WATER DIAGNOSTICS START: " << testName << " ======";
		WATER_LOG_INFO(oss.str());
	}

	/// Flush and detach sinks on test teardown.
	static void Shutdown()
	{
		WATER_LOG_INFO("====== WATER DIAGNOSTICS END ======");

		auto& logger = WulfNet::Logger::Get();
		logger.Flush();

		if (s_fileSink)
		{
			logger.RemoveSink(s_fileSink);
			s_fileSink.reset();
		}
	}

	// -----------------------------------------------------------------
	// CO-FLIP Diagnostics
	// -----------------------------------------------------------------

	/// Log CO-FLIP configuration at initialization.
	static void LogCOFLIPConfig(const WulfNet::COFLIPConfig& cfg)
	{
		std::ostringstream oss;
		oss << std::fixed << std::setprecision(4);
		oss << "[CONFIG] Grid: " << cfg.gridSizeX << "x" << cfg.gridSizeY << "x" << cfg.gridSizeZ
		    << " | CellSize: " << cfg.cellSize << "m"
		    << " | Domain: " << (cfg.gridSizeX * cfg.cellSize) << "x"
		                     << (cfg.gridSizeY * cfg.cellSize) << "x"
		                     << (cfg.gridSizeZ * cfg.cellSize) << "m"
		    << " | TotalCells: " << (cfg.gridSizeX * cfg.gridSizeY * cfg.gridSizeZ);
		COFLIP_LOG_INFO(oss.str());

		oss.str(""); oss.clear();
		oss << "[CONFIG] dt: " << cfg.dt << "s"
		    << " | FLIP ratio: " << cfg.flipRatio
		    << " | PressureIter: " << cfg.pressureIterations
		    << " | ParticlesPerCell: " << cfg.particlesPerCell
		    << " | GPU: " << (cfg.useGPU ? "YES" : "NO");
		COFLIP_LOG_INFO(oss.str());
	}

	/// Log per-frame CO-FLIP diagnostics. Call after Step().
	static void LogCOFLIPFrame(const WulfNet::COFLIPSystem& system, float fps)
	{
		s_frameNumber++;
		const auto& stats = system.GetStats();
		const auto& particles = system.GetParticles();
		uint32_t activeCount = system.GetActiveParticleCount();

		std::ostringstream oss;
		oss << std::fixed;

		// === Frame header ===
		oss << "[F" << s_frameNumber << "] "
		    << std::setprecision(1) << fps << " FPS";

		// === Particle count & change ===
		int32_t particleDelta = static_cast<int32_t>(activeCount) - static_cast<int32_t>(s_prevParticleCount);
		oss << " | Particles: " << activeCount;
		if (particleDelta != 0)
			oss << " (" << (particleDelta > 0 ? "+" : "") << particleDelta << ")";

		// === Timing breakdown ===
		oss << std::setprecision(2)
		    << " | Sim: " << stats.totalTimeMs << "ms"
		    << " [P2G:" << stats.p2gTimeMs
		    << " Press:" << stats.pressureTimeMs
		    << " G2P:" << stats.g2pTimeMs << "]";
		COFLIP_LOG_INFO(oss.str());

		// === Physics diagnostics ===
		oss.str(""); oss.clear();
		oss << std::fixed << std::setprecision(4);

		// Velocity analysis
		float maxVel = 0.0f, avgVel = 0.0f;
		float minVx = FLT_MAX, maxVx = -FLT_MAX;
		float minVy = FLT_MAX, maxVy = -FLT_MAX;
		float minVz = FLT_MAX, maxVz = -FLT_MAX;
		float minY = FLT_MAX, maxY = -FLT_MAX;
		uint32_t nanCount = 0, infCount = 0;
		uint32_t stationaryCount = 0;  // near-zero velocity

		for (uint32_t i = 0; i < activeCount; ++i)
		{
			const auto& p = particles[i];
			if (!(p.flags & 1)) continue;  // Skip inactive

			// NaN/Inf detection
			if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z) ||
			    std::isnan(p.vx) || std::isnan(p.vy) || std::isnan(p.vz))
			{
				nanCount++;
				continue;
			}
			if (std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z) ||
			    std::isinf(p.vx) || std::isinf(p.vy) || std::isinf(p.vz))
			{
				infCount++;
				continue;
			}

			float speed = std::sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
			if (speed > maxVel) maxVel = speed;
			avgVel += speed;

			if (speed < 0.001f) stationaryCount++;

			if (p.vx < minVx) minVx = p.vx;
			if (p.vx > maxVx) maxVx = p.vx;
			if (p.vy < minVy) minVy = p.vy;
			if (p.vy > maxVy) maxVy = p.vy;
			if (p.vz < minVz) minVz = p.vz;
			if (p.vz > maxVz) maxVz = p.vz;
			if (p.y < minY) minY = p.y;
			if (p.y > maxY) maxY = p.y;
		}
		if (activeCount > 0) avgVel /= activeCount;

		oss << "  [VELOCITY] Max: " << std::setprecision(3) << maxVel
		    << " Avg: " << avgVel
		    << " | Vx[" << minVx << "," << maxVx << "]"
		    << " Vy[" << minVy << "," << maxVy << "]"
		    << " Vz[" << minVz << "," << maxVz << "]";
		COFLIP_LOG_DEBUG(oss.str());

		// Particle vertical extent
		oss.str(""); oss.clear();
		oss << std::fixed << std::setprecision(3);
		oss << "  [BOUNDS] Y-range: [" << minY << ", " << maxY << "]"
		    << " | Stationary: " << stationaryCount << "/" << activeCount
		    << " (" << std::setprecision(1)
		    << (activeCount > 0 ? 100.0f * stationaryCount / activeCount : 0.0f) << "%)";
		COFLIP_LOG_DEBUG(oss.str());

		// Energy conservation check
		float energy = stats.totalEnergy;
		float circulation = stats.totalCirculation;
		oss.str(""); oss.clear();
		oss << std::fixed << std::setprecision(4);
		oss << "  [ENERGY] Total: " << energy
		    << " | Circulation: " << circulation;
		if (s_prevEnergy >= 0.0f && s_prevEnergy > 0.001f)
		{
			float energyChange = (energy - s_prevEnergy) / s_prevEnergy * 100.0f;
			oss << " | EnergyDelta: " << std::setprecision(2)
			    << (energyChange > 0.0f ? "+" : "") << energyChange << "%";

			// Warn if energy increases significantly (should only decrease with damping)
			if (energyChange > 5.0f)
			{
				COFLIP_LOG_WARN("  [ANOMALY] Energy INCREASED by " +
				                std::to_string(energyChange) + "% — possible instability!");
			}
		}
		COFLIP_LOG_DEBUG(oss.str());

		// NaN/Inf alerts
		if (nanCount > 0)
		{
			COFLIP_LOG_ERROR("  [CRITICAL] " + std::to_string(nanCount) +
			                 " particles have NaN values!");
			s_nanDetected = true;
		}
		if (infCount > 0)
		{
			COFLIP_LOG_ERROR("  [CRITICAL] " + std::to_string(infCount) +
			                 " particles have Inf values!");
		}

		// Max velocity warning (CFL condition)
		const auto& cfg = system.GetConfig();
		float cflLimit = cfg.cellSize / cfg.dt;  // Approximate CFL bound
		if (maxVel > cflLimit * 0.8f)
		{
			oss.str(""); oss.clear();
			oss << "  [WARNING] Max velocity " << std::setprecision(2) << maxVel
			    << " approaches CFL limit " << cflLimit
			    << " — risk of tunnelling / instability!";
			COFLIP_LOG_WARN(oss.str());
		}

		// Grid cell statistics
		oss.str(""); oss.clear();
		oss << "  [GRID] FluidCells: " << stats.fluidCells
		    << " / " << (cfg.gridSizeX * cfg.gridSizeY * cfg.gridSizeZ)
		    << " (" << std::setprecision(1)
		    << (100.0f * stats.fluidCells /
		        (cfg.gridSizeX * cfg.gridSizeY * cfg.gridSizeZ)) << "%)";
		COFLIP_LOG_DEBUG(oss.str());

		// Update state for next frame comparison
		s_prevEnergy = energy;
		s_prevParticleCount = activeCount;
	}

	// -----------------------------------------------------------------
	// SWE (V5 Sheet Water) Diagnostics — only available when WaterV5 header is included
	// -----------------------------------------------------------------

#ifdef WULFNET_WATER_V5_AVAILABLE

	/// Log SWE sheet water configuration.
	static void LogSWEConfig(const struct SheetWaterConfig& cfg)
	{
		std::ostringstream oss;
		oss << std::fixed << std::setprecision(4);
		oss << "[CONFIG] Grid: " << cfg.gridSizeX << "x" << cfg.gridSizeZ
		    << " (" << (cfg.gridSizeX * cfg.gridSizeZ) << " cells)"
		    << " | CellSize: " << cfg.cellSize << "m"
		    << " | Domain: " << (cfg.gridSizeX * cfg.cellSize) << "x"
		                     << (cfg.gridSizeZ * cfg.cellSize) << "m";
		SWE_LOG_INFO(oss.str());

		oss.str(""); oss.clear();
		oss << "[CONFIG] Gravity: " << cfg.gravity
		    << " | Damping: " << cfg.damping
		    << " | Viscosity: " << cfg.viscosity
		    << " | Substeps: " << cfg.substeps
		    << " | Noise: " << (cfg.noiseEnabled ? "ON" : "OFF");
		SWE_LOG_INFO(oss.str());
	}

	/// Log per-frame SWE diagnostics. Call after StepSWE().
	static void LogSWEFrame(const std::vector<struct WaterCell>& grid,
	                         uint32_t gridSizeX, uint32_t gridSizeZ,
	                         float cellSize, float simTimeMs, float fps,
	                         float totalWater)
	{
		s_frameNumber++;
		const uint32_t total = gridSizeX * gridSizeZ;

		std::ostringstream oss;
		oss << std::fixed;

		// === Frame header ===
		oss << "[F" << s_frameNumber << "] "
		    << std::setprecision(1) << fps << " FPS"
		    << " | Sim: " << std::setprecision(2) << simTimeMs << "ms";

		// === Water volume ===
		oss << " | WaterVol: " << std::setprecision(3) << totalWater;
		if (s_prevVolume >= 0.0f && s_prevVolume > 0.001f)
		{
			float volChange = (totalWater - s_prevVolume) / s_prevVolume * 100.0f;
			oss << " (" << (volChange > 0.0f ? "+" : "") << std::setprecision(2) << volChange << "%)";

			if (std::abs(volChange) > 1.0f)
			{
				SWE_LOG_WARN("  [ANOMALY] Water volume changed by " +
				             std::to_string(volChange) +
				             "% — conservation violation!");
			}
		}
		SWE_LOG_INFO(oss.str());

		// === Cell analysis ===
		uint32_t wetCells = 0;
		float maxDepth = 0.0f, maxVel2 = 0.0f;
		float totalKE = 0.0f;
		float minDepth = FLT_MAX;
		uint32_t nanCount = 0;
		float maxFlowVx = 0.0f, maxFlowVz = 0.0f;
		uint32_t negativeDepthCells = 0;

		for (uint32_t i = 0; i < total; ++i)
		{
			const auto& cell = grid[i];

			// NaN detection
			if (std::isnan(cell.waterHeight) || std::isnan(cell.vx) || std::isnan(cell.vz) ||
			    std::isnan(cell.terrainHeight))
			{
				nanCount++;
				continue;
			}

			if (cell.waterHeight < 0.0f)
				negativeDepthCells++;

			if (cell.waterHeight > 0.001f)
			{
				wetCells++;
				if (cell.waterHeight > maxDepth) maxDepth = cell.waterHeight;
				if (cell.waterHeight < minDepth) minDepth = cell.waterHeight;

				float vel2 = cell.vx * cell.vx + cell.vz * cell.vz;
				if (vel2 > maxVel2) maxVel2 = vel2;

				totalKE += 0.5f * cell.waterHeight * vel2;

				if (std::abs(cell.vx) > std::abs(maxFlowVx)) maxFlowVx = cell.vx;
				if (std::abs(cell.vz) > std::abs(maxFlowVz)) maxFlowVz = cell.vz;
			}
		}

		float maxVel = std::sqrt(maxVel2);

		oss.str(""); oss.clear();
		oss << std::fixed << std::setprecision(3);
		oss << "  [CELLS] Wet: " << wetCells << "/" << total
		    << " (" << std::setprecision(1)
		    << (100.0f * wetCells / total) << "%)"
		    << " | Depth: [" << std::setprecision(4)
		    << (wetCells > 0 ? minDepth : 0.0f) << ", "
		    << maxDepth << "]";
		SWE_LOG_DEBUG(oss.str());

		oss.str(""); oss.clear();
		oss << std::fixed << std::setprecision(3);
		oss << "  [FLOW] MaxVelocity: " << maxVel
		    << " | MaxVx: " << maxFlowVx << " MaxVz: " << maxFlowVz
		    << " | KineticEnergy: " << std::setprecision(4) << totalKE;
		SWE_LOG_DEBUG(oss.str());

		// CFL condition check for SWE: v * dt/dx should be < 1
		float cflMax = maxVel * (1.0f / 60.0f) / cellSize;
		if (cflMax > 0.8f)
		{
			oss.str(""); oss.clear();
			oss << "  [WARNING] CFL number = " << std::setprecision(2) << cflMax
			    << " (>0.8) — solver may be unstable!";
			SWE_LOG_WARN(oss.str());
		}

		// NaN alerts
		if (nanCount > 0)
		{
			SWE_LOG_ERROR("  [CRITICAL] " + std::to_string(nanCount) +
			              " cells have NaN values!");
		}
		if (negativeDepthCells > 0)
		{
			SWE_LOG_WARN("  [ANOMALY] " + std::to_string(negativeDepthCells) +
			             " cells have negative water depth!");
		}

		s_prevVolume = totalWater;
	}

#endif // WULFNET_WATER_V5_AVAILABLE

	/// Log a scenario event (dam break, drop, eruption, etc.)
	static void LogEvent(const std::string& category, const std::string& event)
	{
		WULFNET_INFO(category, "[EVENT] " + event);
	}

	/// Return the current frame number.
	static uint32_t GetFrameNumber() { return s_frameNumber; }

	/// Whether any NaN was ever detected during this test run.
	static bool WasNaNDetected() { return s_nanDetected; }

private:
	static inline std::shared_ptr<WulfNet::ConsoleLogSink> s_consoleSink;
	static inline std::shared_ptr<WulfNet::FileLogSink> s_fileSink;
	static inline uint32_t s_frameNumber = 0;
	static inline float s_prevEnergy = -1.0f;
	static inline float s_prevVolume = -1.0f;
	static inline uint32_t s_prevParticleCount = 0;
	static inline bool s_nanDetected = false;
};
