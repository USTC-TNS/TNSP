/* TAT
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef TAT_HPP_
#define TAT_HPP_

#ifndef TAT_VERSION
#define TAT_VERSION "unknown"
#endif // TAT_VERSION

#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>

#define PASS std::cerr << "calling a passing function at " << __FILE__ << ":" << __LINE__ << " in " << __PRETTY_FUNCTION__ << std::endl, exit(233)
#define ENABLE_IF(...) class = typename std::enable_if<__VA_ARGS__::value>::type

#if (!defined TAT_USE_CPU && !defined TAT_USE_CUDA && !defined TAT_USE_DCU && !defined TAT_USE_SW)
#warning use CPU by default
#define TAT_USE_CPU
#endif

#ifdef TAT_USE_CPU
extern "C"
{
#include <mkl.h>
} // extern "C"
#include <hptt.h>

// SVD
#if (defined TAT_USE_GESDD && defined TAT_USE_GESVD) || (defined TAT_USE_GESVD && defined TAT_USE_GESVDX) || (defined TAT_USE_GESVDX && defined TAT_USE_GESDD)
#error only one of GESDD, GESVD and GESVDX could be in use
#endif
#if (!defined TAT_USE_GESDD && !defined TAT_USE_GESVD && !defined TAT_USE_GESVDX)
#warning must use one of GESDD, GESVD and GESVDX, default use GESVD now
#define TAT_USE_GESVD
#endif

// QR
#if (defined TAT_USE_GEQRF && defined TAT_USE_GEQP3)
#error only one of GEQRF and GEQP3 could be in use
#endif
#if (!defined TAT_USE_GEQRF && !defined TAT_USE_GEQP3)
#warning must use one of GEQRF and GEQP3, default use GEQRF now
#define TAT_USE_GEQRF
#endif

#ifdef TAT_USE_GEQP3
#warning GEQP3 is current unusable
#endif

#endif // TAT_USE_CPU

#include "Tensor.hpp"
#include "Site.hpp"

#endif // TAT_HPP_