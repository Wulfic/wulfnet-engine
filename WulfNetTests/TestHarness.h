// =============================================================================
// WulfNet Engine - Test Harness
// =============================================================================
// Shared test framework for all WulfNet unit tests.
// Provides macros, test runner, and registration infrastructure.
// =============================================================================

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <functional>

// =============================================================================
// Test State (shared across all test files)
// =============================================================================

struct TestState {
    int testsRun = 0;
    int testsPassed = 0;
    int testsFailed = 0;
    int testsSkipped = 0;
    std::vector<std::string> failedTests;
    std::vector<std::string> skippedTests;
    const char* currentTest = nullptr;
    bool currentTestPassed = true;
    std::string failureReason;

    static TestState& Get() {
        static TestState instance;
        return instance;
    }
};

// =============================================================================
// Assertion Macros
// =============================================================================

#define EXPECT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            TestState::Get().currentTestPassed = false; \
            TestState::Get().failureReason = "Expected true: " #condition; \
            return; \
        } \
    } while(0)

#define EXPECT_FALSE(condition) EXPECT_TRUE(!(condition))

#define EXPECT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            TestState::Get().currentTestPassed = false; \
            TestState::Get().failureReason = "Expected equal: " #a " == " #b; \
            return; \
        } \
    } while(0)

#define EXPECT_NE(a, b) \
    do { \
        if ((a) == (b)) { \
            TestState::Get().currentTestPassed = false; \
            TestState::Get().failureReason = "Expected not equal: " #a " != " #b; \
            return; \
        } \
    } while(0)

#define EXPECT_GE(a, b) \
    do { \
        if ((a) < (b)) { \
            TestState::Get().currentTestPassed = false; \
            TestState::Get().failureReason = "Expected >= : " #a " >= " #b; \
            return; \
        } \
    } while(0)

#define EXPECT_LE(a, b) \
    do { \
        if ((a) > (b)) { \
            TestState::Get().currentTestPassed = false; \
            TestState::Get().failureReason = "Expected <= : " #a " <= " #b; \
            return; \
        } \
    } while(0)

#define EXPECT_GT(a, b) \
    do { \
        if ((a) <= (b)) { \
            TestState::Get().currentTestPassed = false; \
            TestState::Get().failureReason = "Expected > : " #a " > " #b; \
            return; \
        } \
    } while(0)

#define EXPECT_LT(a, b) \
    do { \
        if ((a) >= (b)) { \
            TestState::Get().currentTestPassed = false; \
            TestState::Get().failureReason = "Expected < : " #a " < " #b; \
            return; \
        } \
    } while(0)

#define EXPECT_NEAR(a, b, tolerance) \
    do { \
        if (std::abs(static_cast<double>(a) - static_cast<double>(b)) > static_cast<double>(tolerance)) { \
            TestState::Get().currentTestPassed = false; \
            TestState::Get().failureReason = "Expected near: " #a " ~= " #b " (tol=" #tolerance ")"; \
            return; \
        } \
    } while(0)

#define SKIP_TEST(reason) \
    do { \
        TestState::Get().currentTestPassed = true; \
        TestState::Get().failureReason = std::string("SKIPPED: ") + reason; \
        return; \
    } while(0)

// =============================================================================
// Test Runner
// =============================================================================

inline void RunTest(const char* name, void (*testFunc)()) {
    TestState& state = TestState::Get();
    state.testsRun++;
    state.currentTest = name;
    state.currentTestPassed = true;
    state.failureReason.clear();

    std::cout << "  Running: " << name << "... ";
    std::cout.flush();

    testFunc();

    if (state.currentTestPassed) {
        if (state.failureReason.find("SKIPPED") == 0) {
            state.testsSkipped++;
            state.skippedTests.push_back(std::string(name) + ": " + state.failureReason);
            std::cout << "SKIPPED" << std::endl;
        } else {
            state.testsPassed++;
            std::cout << "PASSED" << std::endl;
        }
    } else {
        state.testsFailed++;
        state.failedTests.push_back(std::string(name) + ": " + state.failureReason);
        std::cout << "FAILED: " << state.failureReason << std::endl;
    }
}

#define RUN_TEST(name, func) RunTest(name, func)

// =============================================================================
// Test Suite Registration (forward declarations)
// =============================================================================

void RegisterSystemMonitorTests();
void RegisterAdvancedPhysicsTests();
void RegisterIntegrationTests();
void RegisterConstitutiveModelTests();
void RegisterTerrainDeformationTests();
void RegisterMPMRigidCouplingTests();
void RegisterGaseousSystemTests();
void RegisterDestructionSystemTests();
void RegisterShadowMapTests();
void RegisterGlobalIlluminationTests();
void RegisterVolumetricRendererTests();
void RegisterRenderPipelineTests();
void RegisterAudioEngineTests();
void RegisterAcousticSystemTests();
void RegisterSpatialAudioTests();
void RegisterPerformanceBenchmarks();
void RegisterWaterSystemV3Tests();

// =============================================================================
// Report Printer
// =============================================================================

inline int PrintTestReport() {
    const TestState& state = TestState::Get();

    std::cout << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "  Test Results" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "  Total:   " << state.testsRun << std::endl;
    std::cout << "  Passed:  " << state.testsPassed << std::endl;
    std::cout << "  Failed:  " << state.testsFailed << std::endl;
    std::cout << "  Skipped: " << state.testsSkipped << std::endl;

    if (state.testsFailed > 0) {
        std::cout << std::endl;
        std::cout << "  FAILURES:" << std::endl;
        for (const auto& failure : state.failedTests) {
            std::cout << "    - " << failure << std::endl;
        }
    }

    if (state.testsSkipped > 0) {
        std::cout << std::endl;
        std::cout << "  SKIPPED:" << std::endl;
        for (const auto& skip : state.skippedTests) {
            std::cout << "    - " << skip << std::endl;
        }
    }

    std::cout << "=================================================" << std::endl;

    if (state.testsFailed > 0) {
        std::cout << "  RESULT: FAIL" << std::endl;
        return 1;
    }

    std::cout << "  RESULT: PASS" << std::endl;
    return 0;
}
