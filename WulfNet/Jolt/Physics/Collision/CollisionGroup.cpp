// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <WulfNet/Jolt/Jolt.h>

#include <WulfNet/Jolt/Physics/Collision/CollisionGroup.h>
#include <WulfNet/Jolt/ObjectStream/TypeDeclarations.h>
#include <WulfNet/Jolt/Core/StreamIn.h>
#include <WulfNet/Jolt/Core/StreamOut.h>

JPH_NAMESPACE_BEGIN

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(CollisionGroup)
{
	JPH_ADD_ATTRIBUTE(CollisionGroup, mGroupFilter)
	JPH_ADD_ATTRIBUTE(CollisionGroup, mGroupID)
	JPH_ADD_ATTRIBUTE(CollisionGroup, mSubGroupID)
}

const CollisionGroup CollisionGroup::sInvalid;

void CollisionGroup::SaveBinaryState(StreamOut &inStream) const
{
	inStream.Write(mGroupID);
	inStream.Write(mSubGroupID);
}

void CollisionGroup::RestoreBinaryState(StreamIn &inStream)
{
	inStream.Read(mGroupID);
	inStream.Read(mSubGroupID);
}

JPH_NAMESPACE_END
