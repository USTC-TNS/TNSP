module;

#ifdef _WIN32
#include <windows.h>
#endif

export module TAT.windows_console_guard;

namespace TAT {
   /**
    * Singleton, control color ansi in windows
    */
   struct evil_t {
#ifdef _WIN32
      void set_handle(const auto& which_handle) {
         HANDLE handle = GetStdHandle(which_handle);
         DWORD mode = 0;
         GetConsoleMode(handle, &mode);
         SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
      }
      evil_t() {
         set_handle(STD_OUTPUT_HANDLE);
         set_handle(STD_ERROR_HANDLE);
      }
#endif
   };
   const evil_t evil;
} // namespace TAT
