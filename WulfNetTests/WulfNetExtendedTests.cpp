// =============================================================================
// WulfNet Engine - Extended Test Suite Entry Point
// =============================================================================
// Runs all WulfNet extended test suites: SystemMonitor,
// advanced physics, integration/stress tests, water V3, and more.
//
// Usage:
//   WulfNetExtendedTests                 # Run all suites
//   WulfNetExtendedTests --suite=monitor # Run only SystemMonitor tests
//   WulfNetExtendedTests --suite=physics # Run only advanced physics tests
//   WulfNetExtendedTests --suite=integration # Run only integration tests
//   WulfNetExtendedTests --suite=mpm         # Run only constitutive model tests
//   WulfNetExtendedTests --suite=terrain      # Run only terrain deformation tests
//   WulfNetExtendedTests --suite=coupling     # Run only MPM-rigid coupling tests
// =============================================================================

#include "TestHarness.h"
#include <WulfNet/WulfNet.h>
#include <WulfNet/Core/Logging/Logger.h>
#include <string>
#include <cstring>

int main(int argc, char** argv) {
    // Suppress engine log output during tests
    WulfNet::Logger::Get().SetMinLevel(WulfNet::LogLevel::Error);

    // Parse optional --suite= argument
    std::string selectedSuite = "all";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--suite=") == 0) {
            selectedSuite = arg.substr(8);
        }
        if (arg == "--help" || arg == "-h") {
            std::cout << "WulfNet Extended Tests" << std::endl;
            std::cout << "  --suite=all          Run all test suites (default)" << std::endl;
            std::cout << "  --suite=monitor      System monitor tests" << std::endl;
            std::cout << "  --suite=physics      Advanced physics tests" << std::endl;
            std::cout << "  --suite=integration  Integration & stress tests" << std::endl;
            std::cout << "  --suite=mpm          Constitutive model tests" << std::endl;
            std::cout << "  --suite=terrain      Terrain deformation tests" << std::endl;
            std::cout << "  --suite=coupling     MPM-rigid body coupling tests" << std::endl;
            std::cout << "  --suite=gaseous      Gaseous simulation tests" << std::endl;
            std::cout << "  --suite=destruction  Destruction physics tests" << std::endl;
            std::cout << "  --suite=shadow       Shadow mapping tests" << std::endl;
            std::cout << "  --suite=gi           Global illumination tests" << std::endl;
            std::cout << "  --suite=volumetric   Volumetric renderer tests" << std::endl;
            std::cout << "  --suite=pipeline     Render pipeline tests" << std::endl;
            std::cout << "  --suite=audio        Audio engine tests" << std::endl;
            std::cout << "  --suite=acoustic     Acoustic system tests" << std::endl;
            std::cout << "  --suite=spatial      Spatial audio tests" << std::endl;
            std::cout << "  --suite=benchmark    Performance benchmarks" << std::endl;
            std::cout << "  --suite=water        Water system V3 tests" << std::endl;
            std::cout << "  --suite=engine       Engine lifecycle tests" << std::endl;
            std::cout << "  --suite=threading    Threading & parallelism tests" << std::endl;
            return 0;
        }
    }

    std::cout << "=== WulfNet Engine - Extended Test Suite ===" << std::endl;
    std::cout << "Suite: " << selectedSuite << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // System Monitor Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "monitor") {
        std::cout << "--- System Monitor Tests ---" << std::endl;
        RegisterSystemMonitorTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Advanced Physics Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "physics") {
        std::cout << "--- Advanced Physics Tests ---" << std::endl;
        RegisterAdvancedPhysicsTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Integration & Stress Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "integration") {
        std::cout << "--- Integration & Stress Tests ---" << std::endl;
        RegisterIntegrationTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Constitutive Model Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "mpm") {
        std::cout << "--- Constitutive Model Tests ---" << std::endl;
        RegisterConstitutiveModelTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Terrain Deformation Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "terrain") {
        std::cout << "--- Terrain Deformation Tests ---" << std::endl;
        RegisterTerrainDeformationTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // MPM Rigid Body Coupling Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "coupling") {
        std::cout << "--- MPM Rigid Body Coupling Tests ---" << std::endl;
        RegisterMPMRigidCouplingTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Gaseous Simulation Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "gaseous") {
        std::cout << "--- Gaseous Simulation Tests ---" << std::endl;
        RegisterGaseousSystemTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Destruction Physics Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "destruction") {
        std::cout << "--- Destruction Physics Tests ---" << std::endl;
        RegisterDestructionSystemTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Shadow Mapping Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "shadow") {
        std::cout << "--- Shadow Mapping Tests ---" << std::endl;
        RegisterShadowMapTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Global Illumination Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "gi") {
        std::cout << "--- Global Illumination Tests ---" << std::endl;
        RegisterGlobalIlluminationTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Volumetric Renderer Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "volumetric") {
        std::cout << "--- Volumetric Renderer Tests ---" << std::endl;
        RegisterVolumetricRendererTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Render Pipeline Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "pipeline") {
        std::cout << "--- Render Pipeline Tests ---" << std::endl;
        RegisterRenderPipelineTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Audio Engine Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "audio") {
        std::cout << "--- Audio Engine Tests ---" << std::endl;
        RegisterAudioEngineTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Acoustic System Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "acoustic") {
        std::cout << "--- Acoustic System Tests ---" << std::endl;
        RegisterAcousticSystemTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Spatial Audio Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "spatial") {
        std::cout << "--- Spatial Audio Tests ---" << std::endl;
        RegisterSpatialAudioTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Performance Benchmarks
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "benchmark") {
        std::cout << "--- Performance Benchmarks ---" << std::endl;
        RegisterPerformanceBenchmarks();
        std::cout << std::endl;
    }

    // =========================================================================
    // Engine Lifecycle Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "engine") {
        std::cout << "--- Engine Lifecycle Tests ---" << std::endl;
        RegisterEngineLifecycleTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Threading & Parallelism Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "threading") {
        std::cout << "--- Threading & Parallelism Tests ---" << std::endl;
        RegisterThreadingTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Frame Allocator Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "memory") {
        std::cout << "--- Frame Allocator Tests ---" << std::endl;
        RegisterFrameAllocatorTests();
        std::cout << std::endl;
    }

    // =========================================================================
    // Water System V3 Tests
    // =========================================================================
    if (selectedSuite == "all" || selectedSuite == "water") {
        // Enable full logging for water diagnostics
        auto prevLevel = WulfNet::Logger::Get().GetMinLevel();
        WulfNet::Logger::Get().SetMinLevel(WulfNet::LogLevel::Trace);

        std::cout << "--- Water System V3 Tests ---" << std::endl;
        RegisterWaterSystemV3Tests();
        std::cout << std::endl;

        WulfNet::Logger::Get().SetMinLevel(prevLevel);
    }

    return PrintTestReport();
}
