#pragma once

#include <Tests/Test.h>
#include <WulfNet/Physics/WaterSystemV3.h>

// Real-time flowing water V3 test using Jolt and SWE
class WulfNetWaterV3Test : public Test
{
public:
    JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV3Test)

    virtual void            Initialize() override;
    virtual void            PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
    WulfNet::Physics::WaterSystemV3* mWaterSystem = nullptr;
    float mTime = 0.0f;
    std::vector<JPH::BodyID> mFloatingBodies;
};
