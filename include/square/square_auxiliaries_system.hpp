/**
 * \file square_auxiliaries_system.hpp
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
#ifndef SQUARE_SQUARE_AUXILIARIES_SYSTEM_HPP
#define SQUARE_SQUARE_AUXILIARIES_SYSTEM_HPP

#include <lazy.hpp>

#include "basic.hpp"

namespace square {
   template<typename T>
   struct SquareAuxiliariesSystem {
      int M;
      int N;
      Size dimension_cut;
      lazy::Graph graph;
      std::vector<std::vector<std::shared_ptr<lazy::root<Tensor<T>>>>> lattice;
      std::map<std::tuple<std::string, int, int>, std::shared_ptr<lazy::root<Tensor<T>>>> auxiliaries;

      SquareAuxiliariesSystem(int M, int N, Size Dc) : M(M), N(N), dimension_cut(Dc) {
         lazy::use_graph(graph);
         for (auto i = 0; i < M; i++) {
            auto& row = lattice.emplace_back();
            for (auto j = 0; j < N; j++) {
               row.push_back(lazy::Root<Tensor<T>>());
            }
         }

         std::cout << "!!!!\n";

         lazy::use_graph(lazy::default_graph);
      }
   };
}; // namespace square
#endif