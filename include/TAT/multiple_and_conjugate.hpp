/**
 * \file multiple_and_conjugate.hpp
 *
 * Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_MULTIPLE_AND_CONJUGATE_HPP
#define TAT_MULTIPLE_AND_CONJUGATE_HPP

#include "tensor.hpp"

namespace TAT {
   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry> Tensor<ScalarType, Symmetry>::multiple(const SingularType& S, const Name& name, char direction, bool division) const {
      bool different_direction;
      if (direction == 'u' || direction == 'U') {
         different_direction = false;
      } else if (direction == 'v' || direction == 'V') {
         different_direction = true;
      } else {
         return copy();
      }
#ifdef TAT_USE_SINGULAR_MATRIX
      if (division) {
         if (different_direction) {
            // v
            return contract(S.map([](const ScalarType& value) { return value != 0 ? 1. / value : 0; }), {{name, V_edge}})
                  .edge_rename({{U_edge, name}});
            // TODO also elementwise
         } else {
            // u
            return contract(S.map([](const ScalarType& value) { return value != 0 ? 1. / value : 0; }), {{name, U_edge}})
                  .edge_rename({{V_edge, name}});
         }
      } else {
         if (different_direction) {
            // v
            return contract(S, {{name, V_edge}}).edge_rename({{U_edge, name}});
         } else {
            // u
            return contract(S, {{name, U_edge}}).edge_rename({{V_edge, name}});
         }
      }
#else
      const auto found = name_to_index.find(name);
      if (found == name_to_index.end()) {
         TAT_warning_or_error_when_multiple_name_missing("Edge not Found in Multiple");
         return copy();
      }
      auto result = same_shape();
      auto index = found->second;
      for (const auto& [symmetries, block_source] : core->blocks) {
         auto& block_destination = result.core->blocks.at(symmetries);
         auto symmetry_of_s = symmetries[index];
         if (different_direction) {
            symmetry_of_s = -symmetry_of_s;
         }
         const auto& vector_in_S = S.value.at(symmetry_of_s);
         auto i = 0;
         Size m = 1;
         for (; i < index; i++) {
            m *= core->edges[i].map.at(symmetries[i]);
         }
         Size k = core->edges[i].map.at(symmetries[i]);
         Size n = 1;
         for (i++; i < names.size(); i++) {
            n *= core->edges[i].map.at(symmetries[i]);
         }
         if (vector_in_S.size() != k) {
            TAT_error("Vector Size incompatible in Multiple with a tensor");
         }
         const auto* data_source = block_source.data();
         auto* data_destination = block_destination.data();
         if (division) {
            for (Size a = 0; a < m; a++) {
               for (Size b = 0; b < k; b++) {
                  auto v = vector_in_S[b];
                  for (Size c = 0; c < n; c++) {
                     *(data_destination++) = *(data_source++) / v;
                  }
               }
            }
         } else {
            for (Size a = 0; a < m; a++) {
               for (Size b = 0; b < k; b++) {
                  auto v = vector_in_S[b];
                  for (Size c = 0; c < n; c++) {
                     *(data_destination++) = *(data_source++) * v;
                  }
               }
            }
         }
      }
      return result;
#endif
   }

   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry> Tensor<ScalarType, Symmetry>::conjugate() const {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry> && is_real_v<ScalarType>) {
         return copy();
      }
      auto result_edges = std::vector<Edge<Symmetry>>();
      result_edges.reserve(names.size());
      for (const auto& edge : core->edges) {
         auto& result_edge = result_edges.emplace_back();
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            result_edge.arrow = !edge.arrow;
         }
         for (const auto& [symmetry, dimension] : edge.map) {
            result_edge.map[-symmetry] = dimension;
         }
      }
      auto transpose_flag = std::vector<Rank>(names.size(), 0);
      auto valid_flag = std::vector<bool>(1, true);
      auto result = Tensor<ScalarType, Symmetry>(names, result_edges);
      for (const auto& [symmetries, block] : core->blocks) {
         auto result_symmetries = std::vector<Symmetry>();
         for (const auto& symmetry : symmetries) {
            result_symmetries.push_back(-symmetry);
         }
         // result.core->blocks.at(result_symmetries) <- block
         const Size total_size = block.size();
         ScalarType* destination = result.core->blocks.at(result_symmetries).data();
         const ScalarType* source = block.data();
         bool parity = false;
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            parity = Symmetry::get_split_merge_parity(symmetries, transpose_flag, valid_flag);
         }
         if constexpr (is_complex_v<ScalarType>) {
            if (parity) {
               for (Size i = 0; i < total_size; i++) {
                  destination[i] = -std::conj(source[i]);
               }
            } else {
               for (Size i = 0; i < total_size; i++) {
                  destination[i] = std::conj(source[i]);
               }
            }
         } else {
            if (parity) {
               for (Size i = 0; i < total_size; i++) {
                  destination[i] = -source[i];
               }
            } else {
               for (Size i = 0; i < total_size; i++) {
                  destination[i] = source[i];
               }
            }
         }
      }
      return result;
   }

} // namespace TAT
#endif