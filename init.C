#include "cling/Interpreter/Interpreter.h"
#pragma cling load("liblapack.so")
#pragma cling add_include_path("./include")
#include <TAT/TAT.hpp>
void init() {
   gCling->allowRedefinition();
}
