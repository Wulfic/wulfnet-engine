// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <WulfNet/Jolt/Jolt.h>

#include <WulfNet/Jolt/Physics/Collision/PhysicsMaterial.h>
#include <WulfNet/Jolt/Physics/Collision/PhysicsMaterialSimple.h>
#include <WulfNet/Jolt/Core/StreamUtils.h>

JPH_NAMESPACE_BEGIN

RefConst<PhysicsMaterial> PhysicsMaterial::sDefault;

JPH_IMPLEMENT_SERIALIZABLE_VIRTUAL(PhysicsMaterial)
{
	JPH_ADD_BASE_CLASS(PhysicsMaterial, SerializableObject)
}

void PhysicsMaterial::SaveBinaryState(StreamOut &inStream) const
{
	inStream.Write(GetRTTI()->GetHash());
}

void PhysicsMaterial::RestoreBinaryState(StreamIn &inStream)
{
	// RTTI hash is read in sRestoreFromBinaryState
}

PhysicsMaterial::PhysicsMaterialResult PhysicsMaterial::sRestoreFromBinaryState(StreamIn &inStream)
{
	return StreamUtils::RestoreObject<PhysicsMaterial>(inStream, &PhysicsMaterial::RestoreBinaryState);
}

JPH_NAMESPACE_END
