// =============================================================================
// WulfNet Engine - Profiler.cpp
// =============================================================================

#include "Profiler.h"
#include "../Logging/Logger.h"
#include <sstream>

namespace WulfNet {

void ScopedTimer::LogTime(long long microseconds) {
    std::ostringstream oss;
    oss << m_name << " took ";

    if (microseconds >= 1000000) {
        oss << (microseconds / 1000000.0) << " s";
    } else if (microseconds >= 1000) {
        oss << (microseconds / 1000.0) << " ms";
    } else {
        oss << microseconds << " μs";
    }

    WULFNET_TRACE("Profiler", oss.str());
}

} // namespace WulfNet
