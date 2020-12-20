/**
 * \file exact_lattice.hpp
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
#ifndef SQUARE_EXACT_LATTICE_HPP
#define SQUARE_EXACT_LATTICE_HPP

#include <TAT/TAT.hpp>
#include <map>
#include <type_traits>
#include <vector>

#include "abstract_lattice.hpp"
#include "basic.hpp"

namespace square {
   template<typename T>
   struct ExactLattice : AbstractLattice<T> {
      Tensor<T> vector;

      using AbstractLattice<T>::M;
      using AbstractLattice<T>::N;
      using AbstractLattice<T>::dimension_physics;
      using AbstractLattice<T>::hamiltonians;

      ExactLattice() : AbstractLattice<T>(), vector(){};
      ExactLattice(const ExactLattice<T>&) = default;
      ExactLattice(ExactLattice<T>&&) = default;
      ExactLattice<T>& operator=(const ExactLattice<T>&) = default;
      ExactLattice<T>& operator=(ExactLattice<T>&&) = default;

      ExactLattice(int M, int N, Size d) : AbstractLattice<T>(M, N, d) {
         auto name_list = std::vector<Name>();
         for (auto i = 0; i < M; i++) {
            for (auto j = 0; j < N; j++) {
               name_list.push_back("P-" + std::to_string(i) + "-" + std::to_string(j));
            }
         }
         vector = Tensor<T>(std::move(name_list), std::vector<Size>(M * N, dimension_physics)).set(random::normal<T>(0, 1));
         vector /= vector.template norm<-1>();
      }

      explicit ExactLattice(const SimpleUpdateLattice<T>& other);
      explicit ExactLattice(const SamplingGradientLattice<T>& other);

      real<T> update(int total_step, real<T> approximate_energy = -0.5) {
         std::cout << clear_line << "Exact update done, total_step=" << total_step << "\n" << std::flush;
         real<T> total_approximate_energy = std::abs(approximate_energy) * M * N;
         real<T> energy = 0;
         for (auto step = 0; step < total_step; step++) {
            auto temporary_vector = vector.same_shape().zero();
            for (const auto& [positions, value] : hamiltonians) {
               auto position_number = positions.size();
               auto map_in = std::map<Name, Name>();
               auto map_out = std::map<Name, Name>();
               for (auto i = 0; i < position_number; i++) {
                  Name physics_name = "P-" + std::to_string(std::get<0>(positions[i])) + "-" + std::to_string(std::get<1>(positions[i]));
                  map_in["I" + std::to_string(i)] = physics_name;
                  map_out["O" + std::to_string(i)] = physics_name;
               }
               temporary_vector += vector.contract_all_edge(value->edge_rename(map_in)).edge_rename(map_out);
            }
            vector *= total_approximate_energy;
            vector -= temporary_vector;
            // v <- a v - H v = (a - H) v => E = a - v'/v
            real<T> norm_max = vector.template norm<-1>();
            energy = total_approximate_energy - norm_max;
            vector /= norm_max;
            std::cout << clear_line << "Exact updating, total_step=" << total_step << ", step=" << (step + 1) << ", Energy=" << energy / (M * N)
                      << "\r" << std::flush;
         }
         std::cout << clear_line << "Exact update done, total_step=" << total_step << ", Energy=" << energy / (M * N) << "\n" << std::flush;
         return energy / (M * N);
      }

      real<T> denominator() const {
         return vector.contract_all_edge(vector).template to<real<T>>();
      }

      template<typename C>
      real<T> observe(const std::vector<std::tuple<int, int>>& positions, const Tensor<C>& observer, bool calculate_denominator) const {
         using common = std::common_type_t<C, T>;
         auto common_vector = vector.template to<common>();
         auto common_observer = observer.template to<common>();
         auto position_number = positions.size();
         auto map_in = std::map<Name, Name>();
         auto map_out = std::map<Name, Name>();
         for (auto i = 0; i < position_number; i++) {
            Name physics_name = "P-" + std::to_string(std::get<0>(positions[i])) + "-" + std::to_string(std::get<1>(positions[i]));
            map_in["I" + std::to_string(i)] = physics_name;
            map_out["O" + std::to_string(i)] = physics_name;
         }
         auto numerator = common_vector.contract_all_edge(common_observer.edge_rename(map_in))
                                .edge_rename(map_out)
                                .contract_all_edge(common_vector)
                                .template to<real<T>>();
         if (calculate_denominator) {
            return real<T>(numerator) / denominator();
         } else {
            return real<T>(numerator);
         }
      }

      real<T> observe_energy() {
         real<T> energy = 0;
         for (const auto& [positions, observer] : hamiltonians) {
            energy += observe(positions, *observer, false);
         }
         return energy / denominator() / (M * N);
      }
   };

   template<typename T>
   std::ostream& operator<(std::ostream& out, const ExactLattice<T>& lattice) {
      using TAT::operator<;
      out < static_cast<const AbstractLattice<T>&>(lattice);
      out < lattice.vector;
      return out;
   }

   template<typename T>
   std::istream& operator>(std::istream& in, ExactLattice<T>& lattice) {
      using TAT::operator>;
      in > static_cast<AbstractLattice<T>&>(lattice);
      in > lattice.vector;
      return in;
   }
} // namespace square

#endif
