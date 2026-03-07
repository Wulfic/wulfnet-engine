#pragma once

#include <Tests/Test.h>
#include <WulfNet/Physics/Fluids/FluidSystem.h>

// Real-time flowing water V3 test using Jolt and SWE
class WulfNetWaterV3Test : public Test
{
public:
    JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, WulfNetWaterV3Test)

    virtual void            Initialize() override;
    virtual void            PrePhysicsUpdate(const PreUpdateParams &inParams) override;

private:
    WulfNet::FluidSystem* mWaterSystem = nullptr;
    float mTime = 0.0f;
    std::vector<JPH::BodyID> mFloatingBodies;
};
