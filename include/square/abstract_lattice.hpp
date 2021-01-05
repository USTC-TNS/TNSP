/**
 * \file abstract_lattice.hpp
 *
 * Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "basic.hpp"

namespace square {
   template<typename T>
   struct AbstractLattice {
      int M;
      int N;
      Size dimension_physics;
      std::map<std::vector<std::tuple<int, int>>, std::shared_ptr<const Tensor<T>>> hamiltonians;

      AbstractLattice() : M(0), N(0), dimension_physics(0), hamiltonians(){};
      AbstractLattice(const AbstractLattice<T>&) = default;
      AbstractLattice(AbstractLattice<T>&&) = default;
      AbstractLattice<T>& operator=(const AbstractLattice<T>&) = default;
      AbstractLattice<T>& operator=(AbstractLattice<T>&&) = default;

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

      // other hamiltonian setter...
   };

   template<typename T>
   std::ostream& operator<(std::ostream& out, const AbstractLattice<T>& lattice) {
      using TAT::operator<;
      out < lattice.M < lattice.N;
      out < lattice.dimension_physics;
      // pool
      std::vector<std::shared_ptr<const Tensor<T>>> pool;
      std::map<std::vector<std::tuple<int, int>>, int> pool_map;
      for (const auto& [positions, tensor] : lattice.hamiltonians) {
         if (auto found = std::find(pool.begin(), pool.end(), tensor); found != pool.end()) {
            pool_map[positions] = std::distance(pool.begin(), found);
         } else {
            pool_map[positions] = pool.size();
            pool.push_back(tensor);
         }
      }
      Size pool_size = pool.size();
      // data
      out < pool_size;
      for (const auto& tensor : pool) {
         out < *tensor;
      }
      Size map_size = pool_map.size();
      out < map_size;
      for (const auto& [positions, index] : pool_map) {
         out < positions < index;
      }
      return out;
   }

   template<typename T>
   std::istream& operator>(std::istream& in, AbstractLattice<T>& lattice) {
      using TAT::operator>;
      in > lattice.M > lattice.N;
      in > lattice.dimension_physics;
      Size pool_size;
      in > pool_size;
      std::vector<std::shared_ptr<const Tensor<T>>> pool;
      for (auto i = 0; i < pool_size; i++) {
         Tensor<T> tensor;
         in > tensor;
         pool.push_back(std::make_shared<const Tensor<T>>(std::move(tensor)));
      }
      Size map_size;
      in > map_size;
      lattice.hamiltonians.clear();
      for (auto i = 0; i < map_size; i++) {
         std::vector<std::tuple<int, int>> positions;
         int index;
         in > positions > index;
         lattice.hamiltonians[std::move(positions)] = pool[index];
      }
      return in;
   }
} // namespace square

#endif
