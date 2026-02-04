// =============================================================================
// WulfNet Engine - Logging System Implementation
// =============================================================================

#include "Log.h"

#if WULFNET_PLATFORM_WINDOWS
#include <windows.h>
#endif

namespace WulfNet {

#if WULFNET_PLATFORM_WINDOWS
void Logger::enableWindowsAnsiColors() {
    // Enable ANSI escape sequences on Windows 10+
    HANDLE hConsole = GetStdHandle(STD_ERROR_HANDLE);
    if (hConsole != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(hConsole, &mode)) {
            mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hConsole, mode);
        }
    }
}
#endif

} // namespace WulfNet
