/**
 * \file abstract_lattice.hpp
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
#ifndef SQUARE_ABSTRACT_LATTICE_HPP
#define SQUARE_ABSTRACT_LATTICE_HPP

#include <map>
#include <memory>
#include <vector>

#include "basic.hpp"

namespace square {
   template<typename T>
   struct AbstractLattice {
      int M;
      int N;
      Size dimension_physics;
      std::map<std::vector<std::tuple<int, int>>, std::shared_ptr<const Tensor<T>>> hamiltonians;

      AbstractLattice(int M, int N, Size d) : M(M), N(N), dimension_physics(d) {}

      void set_all_site(std::shared_ptr<const Tensor<T>> hamiltonian) {
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               hamiltonians[{{i, j}}] = hamiltonian;
            }
         }
      };

      void set_all_vertical_bond(std::shared_ptr<const Tensor<T>> hamiltonian) {
         for (auto i = 0; i < M - 1; i++) {
            for (auto j = 0; j < N; j++) {
               hamiltonians[{{i, j}, {i + 1, j}}] = hamiltonian;
            }
         }
      }

      void set_all_horizontal_bond(std::shared_ptr<const Tensor<T>> hamiltonian) {
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N - 1; j++) {
               hamiltonians[{{i, j}, {i, j + 1}}] = hamiltonian;
            }
         }
      }
   };
} // namespace square

#endif
