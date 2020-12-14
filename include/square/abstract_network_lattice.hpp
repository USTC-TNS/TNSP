/**
 * \file abstract_network_lattice.hpp
 *
 * Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef SQUARE_ABSTRACT_NETWORK_LATTICE_HPP
#define SQUARE_ABSTRACT_NETWORK_LATTICE_HPP

#include <vector>

#include "abstract_lattice.hpp"

namespace square {
   template<typename T>
   struct AbstractNetworkLattice : AbstractLattice<T> {
      Size dimension_virtual;
      std::vector<std::vector<Tensor<T>>> lattice;

      AbstractNetworkLattice(int M, int N, Size D, Size d) : AbstractLattice<T>(M, N, d), dimension_virtual(D) {
         for (auto i = 0; i < M; i++) {
            auto& row = lattice.emplace_back();
            for (auto j = 0; j < N; j++) {
               auto name_list = std::vector<Name>{"P"};
               auto dimension_list = std::vector<Size>{this->dimension_physics};
               if (i != 0) {
                  name_list.push_back("U");
                  dimension_list.push_back(dimension_virtual);
               }
               if (j != 0) {
                  name_list.push_back("L");
                  dimension_list.push_back(dimension_virtual);
               }
               if (i != M - 1) {
                  name_list.push_back("D");
                  dimension_list.push_back(dimension_virtual);
               }
               if (j != N - 1) {
                  name_list.push_back("R");
                  dimension_list.push_back(dimension_virtual);
               }
               row.emplace_back(std::move(name_list), dimension_list).set(random::normal<T>(0, 1));
            }
         }
      }
   };
} // namespace square

#endif