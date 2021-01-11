/**
 * \file conversion.hpp
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
#ifndef SQUARE_CONVERSION_HPP
#define SQUARE_CONVERSION_HPP

#include "exact_lattice.hpp"
#include "sampling_gradient_lattice.hpp"
#include "simple_update_lattice.hpp"

namespace square {
   template<typename T>
   SamplingGradientLattice<T>::SamplingGradientLattice(const SimpleUpdateLattice<T>& other, Size Dc) :
         AbstractNetworkLattice<T>(other), dimension_cut(Dc), spin(this) {
      for (auto i = 0; i < M; i++) {
         for (auto j = 0; j < N; j++) {
            auto to_multiple = lattice[i][j];
            to_multiple = other.try_multiple(to_multiple, i, j, 'D');
            to_multiple = other.try_multiple(to_multiple, i, j, 'R');
            lattice[i][j] = std::move(to_multiple);
            lattice[i][j] /= lattice[i][j].template norm<-1>();
         }
      }
   }

   template<typename T>
   SimpleUpdateLattice<T>::SimpleUpdateLattice(const SamplingGradientLattice<T>& other) : AbstractNetworkLattice<T>(other) {}

   template<typename T>
   ExactLattice<T>::ExactLattice(const SamplingGradientLattice<T>& other) : AbstractLattice<T>(other) {
      vector = Tensor<T>(1);
      for (auto i = 0; i < M; i++) {
         for (auto j = 0; j < N; j++) {
            auto to_contract = other.lattice[i][j];
            vector = vector.contract(
                  to_contract.edge_rename({{"D", "D" + std::to_string(j)}, {"P", "P-" + std::to_string(i) + "-" + std::to_string(j)}}),
                  {{"R", "L"}, {"D" + std::to_string(j), "U"}});
            vector /= vector.template norm<-1>();
         }
      }
   }

   template<typename T>
   ExactLattice<T>::ExactLattice(const SimpleUpdateLattice<T>& other) : AbstractLattice<T>(other) {
      vector = Tensor<T>(1);
      for (auto i = 0; i < M; i++) {
         for (auto j = 0; j < N; j++) {
            auto to_contract = other.lattice[i][j];
            to_contract = other.try_multiple(to_contract, i, j, 'D');
            to_contract = other.try_multiple(to_contract, i, j, 'R');
            vector = vector.contract(
                  to_contract.edge_rename({{"D", "D" + std::to_string(j)}, {"P", "P-" + std::to_string(i) + "-" + std::to_string(j)}}),
                  {{"R", "L"}, {"D" + std::to_string(j), "U"}});
            vector /= vector.template norm<-1>();
         }
      }
   }

} // namespace square

#endif
