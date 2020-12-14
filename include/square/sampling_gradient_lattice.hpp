/**
 * \file sampling_gradient_lattice.hpp
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
#ifndef SQUARE_SAMPLING_GRADIENT_LATTICE_HPP
#define SQUARE_SAMPLING_GRADIENT_LATTICE_HPP

#include "abstract_network_lattice.hpp"
#include "square_auxiliaries_system.hpp"

namespace square {
   template<typename T>
   struct SpinConfiguration : SquareAuxiliariesSystem<T> {
      const SamplingGradientLattice<T>* owner;
      std::vector<std::vector<int>> configuration;

      using SquareAuxiliariesSystem<T>::M;
      using SquareAuxiliariesSystem<T>::N;
      using SquareAuxiliariesSystem<T>::dimension_cut;
      using SquareAuxiliariesSystem<T>::lattice;
      using SquareAuxiliariesSystem<T>::operator();

      SpinConfiguration(const SamplingGradientLattice<T>* owner) :
            SquareAuxiliariesSystem<T>(owner->M, owner->N, owner->dimension_cut), owner(owner) {
         for (auto i = 0; i < M; i++) {
            auto& row = configuration.emplace_back();
            for (auto j = 0; j < N; j++) {
               row.push_back(-1);
            }
         }
      }

      void set_spin(const std::tuple<int, int>& position, int spin) {
         auto [x, y] = position;
         if (configuration[x][y] != spin) {
            lattice[x][y]->set(owner->lattice[x][y].shrink({{"P", spin}}));
            configuration[x][y] = spin;
         }
      }

      auto operator()(const std::map<std::tuple<int, int>, int>& replacement) const {
         auto real_replacement = std::map<std::tuple<int, int>, Tensor<T>>();
         for (auto& [position, spin] : replacement) {
            auto [x, y] = position;
            if (configuration[x][y] != spin) {
               real_replacement[{x, y}] = owner->lattice[x][y].shrink({{"P", spin}});
            }
         }
         return operator()(real_replacement);
      }
   };

   template<typename T>
   struct SamplingGradientLattice : AbstractNetworkLattice<T> {
      Size dimension_cut;
      SpinConfiguration<T> spin;

      SamplingGradientLattice(int M, int N, Size D, Size Dc, Size d) : AbstractNetworkLattice<T>(M, N, D, d), dimension_cut(Dc), spin(this) {}

      SamplingGradientLattice(const SimpleUpdateLattice<T>&);

      void initialize_spin(std::function<int(int, int)> function) {
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               set_spin({i, j}, function(i, j));
            }
         }
      }
   };
} // namespace square

#endif