// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <WulfNet/Jolt/Jolt.h>

#include <WulfNet/Jolt/Physics/Constraints/Constraint.h>
#include <WulfNet/Jolt/Physics/StateRecorder.h>
#include <WulfNet/Jolt/ObjectStream/TypeDeclarations.h>
#include <WulfNet/Jolt/Core/StreamUtils.h>

JPH_NAMESPACE_BEGIN

JPH_IMPLEMENT_SERIALIZABLE_VIRTUAL(ConstraintSettings)
{
	JPH_ADD_BASE_CLASS(ConstraintSettings, SerializableObject)

	JPH_ADD_ATTRIBUTE(ConstraintSettings, mEnabled)
	JPH_ADD_ATTRIBUTE(ConstraintSettings, mDrawConstraintSize)
	JPH_ADD_ATTRIBUTE(ConstraintSettings, mConstraintPriority)
	JPH_ADD_ATTRIBUTE(ConstraintSettings, mNumVelocityStepsOverride)
	JPH_ADD_ATTRIBUTE(ConstraintSettings, mNumPositionStepsOverride)
	JPH_ADD_ATTRIBUTE(ConstraintSettings, mUserData)
}

void ConstraintSettings::SaveBinaryState(StreamOut &inStream) const
{
	inStream.Write(GetRTTI()->GetHash());
	inStream.Write(mEnabled);
	inStream.Write(mDrawConstraintSize);
	inStream.Write(mConstraintPriority);
	inStream.Write(mNumVelocityStepsOverride);
	inStream.Write(mNumPositionStepsOverride);
}

void ConstraintSettings::RestoreBinaryState(StreamIn &inStream)
{
	// Type hash read by sRestoreFromBinaryState
	inStream.Read(mEnabled);
	inStream.Read(mDrawConstraintSize);
	inStream.Read(mConstraintPriority);
	inStream.Read(mNumVelocityStepsOverride);
	inStream.Read(mNumPositionStepsOverride);
}

ConstraintSettings::ConstraintResult ConstraintSettings::sRestoreFromBinaryState(StreamIn &inStream)
{
	return StreamUtils::RestoreObject<ConstraintSettings>(inStream, &ConstraintSettings::RestoreBinaryState);
}

void Constraint::SaveState(StateRecorder &inStream) const
{
	inStream.Write(mEnabled);
}

void Constraint::RestoreState(StateRecorder &inStream)
{
	inStream.Read(mEnabled);
}

void Constraint::ToConstraintSettings(ConstraintSettings &outSettings) const
{
	outSettings.mEnabled = mEnabled;
	outSettings.mConstraintPriority = mConstraintPriority;
	outSettings.mNumVelocityStepsOverride = mNumVelocityStepsOverride;
	outSettings.mNumPositionStepsOverride = mNumPositionStepsOverride;
	outSettings.mUserData = mUserData;
#ifdef JPH_DEBUG_RENDERER
	outSettings.mDrawConstraintSize = mDrawConstraintSize;
#endif // JPH_DEBUG_RENDERER
}

JPH_NAMESPACE_END
