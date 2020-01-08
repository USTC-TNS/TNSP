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

#pragma once
#ifndef TAT_HPP
#define TAT_HPP
#include "TAT/init.hpp"
#endif
