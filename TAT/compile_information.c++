module;

#include <string>

export module TAT.compile_information;

namespace TAT {
   /**
    * Debug flag
    */
   export constexpr bool debug_mode =
#ifdef NDEBUG
         false
#else
         true
#endif
         ;

#ifndef TAT_VERSION
#define TAT_VERSION "0.3.0"
#endif
   /**
    * TAT version
    */
   export const char* version = TAT_VERSION;

   /**
    * TAT informations about compiler and license
    */
   export const char* information = "TAT " TAT_VERSION " ("
#ifdef TAT_BUILD_TYPE
                                    "" TAT_BUILD_TYPE ", "
#endif
                                    "" __DATE__ ", " __TIME__
#ifdef TAT_COMPILER_INFORMATION
                                    ", " TAT_COMPILER_INFORMATION
#endif
                                    ")\n"
                                    "Copyright (C) 2019-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>\n"
                                    "This is free software; see the source for copying conditions.  There is NO\n"
                                    "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.";
} // namespace TAT
