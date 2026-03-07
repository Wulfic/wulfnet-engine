// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <WulfNet/Jolt/Jolt.h>

#include <WulfNet/Jolt/Core/TickCounter.h>
#include <WulfNet/Jolt/Core/IncludeWindows.h>

JPH_NAMESPACE_BEGIN

#if defined(JPH_PLATFORM_WINDOWS_UWP) || (defined(JPH_PLATFORM_WINDOWS) && defined(JPH_CPU_ARM))

uint64 GetProcessorTickCount()
{
	LARGE_INTEGER count;
	QueryPerformanceCounter(&count);
	return uint64(count.QuadPart);
}

#endif // JPH_PLATFORM_WINDOWS_UWP || (JPH_PLATFORM_WINDOWS && JPH_CPU_ARM)

JPH_NAMESPACE_END
