#define TAT_USE_CPU
#define TAT_TEST_MAIN
#define TAT_USE_GESVDX
// if GESVDX in use, GESDD will not use
#define TAT_USE_GESDD
// #define TAT_USE_GEQP3
// GEQP3 not understand, maybe useful if R will drop

#include "TAT.hpp"
