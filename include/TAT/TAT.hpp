/**
 * \file TAT.hpp
 *
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

#pragma once
#ifndef TAT_HPP
#define TAT_HPP

#ifndef __cplusplus
#error "only work for c++"
#else
#ifdef _MSVC_LANG
#if _MSVC_LANG < 201703L
#error require c++17 or later
#endif
#else
#if __cplusplus < 201703L
#error require c++17 or later
#endif
#endif
#endif

#ifndef NDEBUG
#include <iostream>
namespace TAT {
   struct Evil {
      ~Evil() {
         std::clog << "\n\nPremature optimization is the root of all evil!\n"
                      "                                       --- Donald Knuth\n\n\n";
      }
   };
   const Evil evil;
} // namespace TAT
#endif

// clang-format off
#include "tensor.hpp"
#include "scalar.hpp"
#include "io.hpp"
// clang-format on

#endif
