/**
 * \file tensor_miscellaneous.hpp
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
#ifndef TAT_TENSOR_MISCELLANEOUS_HPP
#define TAT_TENSOR_MISCELLANEOUS_HPP

#include "tensor.hpp"
#include "timer.hpp"

namespace TAT {
   template<Size n, typename ScalarType, typename ScalarTypeS>
   void multiple_kernel_small_n(Size m, Size k, ScalarType* data_destination, const ScalarType* data_source, const ScalarTypeS* S) {
      for (Size a = 0; a < m; a++) {
         for (Size b = 0; b < k; b++) {
            auto v = S[b];
            const auto* data_source_block = &data_source[(a * k + b) * n];
            auto* data_destination_block = &data_destination[(a * k + b) * n];
            for (Size c = 0; c < n; c++) {
               data_destination_block[c] = data_source_block[c] * v;
            }
         }
      }
   }

   template<typename ScalarType, typename ScalarTypeS>
   void multiple_kernel(Size m, Size k, Size n, ScalarType* data_destination, const ScalarType* data_source, const ScalarTypeS* S) {
#define TAT_CALL_MULTIPLE_SMALL_N(N)                                      \
   case N:                                                                \
      multiple_kernel_small_n<N>(m, k, data_destination, data_source, S); \
      break;
      switch (n) {
         case 0:
            break;
            TAT_CALL_MULTIPLE_SMALL_N(1);
            TAT_CALL_MULTIPLE_SMALL_N(2);
            TAT_CALL_MULTIPLE_SMALL_N(3);
            TAT_CALL_MULTIPLE_SMALL_N(4);
            TAT_CALL_MULTIPLE_SMALL_N(5);
            TAT_CALL_MULTIPLE_SMALL_N(6);
            TAT_CALL_MULTIPLE_SMALL_N(7);
            TAT_CALL_MULTIPLE_SMALL_N(8);
            TAT_CALL_MULTIPLE_SMALL_N(9);
            TAT_CALL_MULTIPLE_SMALL_N(10);
         default:
            for (Size a = 0; a < m; a++) {
               for (Size b = 0; b < k; b++) {
                  auto v = S[b];
                  const auto* data_source_block = &data_source[(a * k + b) * n];
                  auto* data_destination_block = &data_destination[(a * k + b) * n];
                  for (Size c = 0; c < n; c++) {
                     data_destination_block[c] = data_source_block[c] * v;
                  }
               }
            }
      }
#undef TAT_CALL_MULTIPLE_SMALL_N
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>
   Tensor<ScalarType, Symmetry, Name>::multiple(const SingularType& S, const Name& name, char direction, bool division) const {
      auto timer_guard = multiple_guard();
      auto pmr_guard = scope_resource<>();
      bool different_direction;
      if (direction == 'u' || direction == 'U') {
         different_direction = false;
      } else if (direction == 'v' || direction == 'V') {
         different_direction = true;
      } else {
         TAT_error("Direction invalid in multiple");
         return *this;
      }
      const auto found = name_to_index.find(name);
      if (found == name_to_index.end()) {
         TAT_warning_or_error_when_name_missing("Name not found in multiple");
         return *this;
      }
      auto result = same_shape();
      auto index = found->second;
      for (const auto& [symmetries, block_source] : core->blocks) {
         auto& block_destination = result.core->blocks.at(symmetries);
         auto symmetry_of_s = symmetries[index];
         if (different_direction) {
            symmetry_of_s = -symmetry_of_s;
         }
#ifdef TAT_USE_SINGULAR_MATRIX
         const auto& vector_in_S = S.core->blocks.at({-symmetry_of_s, symmetry_of_s});
         auto dimension = S.core->edges[1].map.at(symmetry_of_s);
         auto dimension_plus_one = dimension + 1;
#else
         const auto& vector_in_S = S.value.at(symmetry_of_s);
         auto dimension = vector_in_S.size();
#endif
         Rank i = 0;
         Size m = 1;
         for (; i < index; i++) {
            m *= core->edges[i].map.at(symmetries[i]);
         }
         Size k = core->edges[i].map.at(symmetries[i]);
         Size n = 1;
         for (i++; i < names.size(); i++) {
            n *= core->edges[i].map.at(symmetries[i]);
         }
         if (dimension != k) {
            TAT_error("Vector Size incompatible in Multiple with a tensor");
         }
         const auto* data_source = block_source.data();
         auto* data_destination = block_destination.data();

         using ScalarTypeS = typename std::remove_cv_t<std::remove_reference_t<decltype(vector_in_S)>>::value_type;
         vector<ScalarTypeS> realS(k);
         const auto* pointS = realS.data();
         if (division) {
            for (Size i = 0; i < k; i++) {
#ifdef TAT_USE_SINGULAR_MATRIX
               realS[i] = ScalarTypeS(1) / vector_in_S[i * dimension_plus_one];
#else
               realS[i] = ScalarTypeS(1) / vector_in_S[i];
#endif
            }
         } else {
#ifdef TAT_USE_SINGULAR_MATRIX
            for (Size i = 0; i < k; i++) {
               realS[i] = vector_in_S[i * dimension_plus_one];
            }
#else
            pointS = vector_in_S.data();
#endif
         }

         multiple_kernel(m, k, n, data_destination, data_source, pointS);
      }
      return result;
   }

   // TODO: conjugate和merge等操作不可交换，可能需要给Edge加上一个conjugated的flag
   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::conjugate() const {
      auto timer_guard = conjugate_guard();
      auto pmr_guard = scope_resource<>();
      if constexpr (std::is_same_v<Symmetry, NoSymmetry> && is_real_v<ScalarType>) {
         return *this;
      }
      auto result_edges = pmr::vector<Edge<Symmetry>>();
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
      auto transpose_flag = pmr::vector<Rank>(names.size(), 0);
      auto valid_flag = pmr::vector<bool>(1, true);
      auto result = Tensor<ScalarType, Symmetry, Name>(names, result_edges);
      for (const auto& [symmetries, block] : core->blocks) {
         auto result_symmetries = typename decltype(core->blocks)::key_type();
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

   /// \private
   template<typename ScalarType, typename VectorSize>
   void set_to_identity(ScalarType* pointer, const VectorSize& dimension, const VectorSize& leading, Rank rank) {
      auto current_index = VectorSize(rank, 0);
      while (true) {
         *pointer = 1;
         Rank active_position = rank - 1;

         current_index[active_position]++;
         pointer += leading[active_position];

         while (current_index[active_position] == dimension[active_position]) {
            current_index[active_position] = 0;
            pointer -= dimension[active_position] * leading[active_position];

            if (active_position == 0) {
               return;
            }
            active_position--;

            current_index[active_position]++;
            pointer += leading[active_position];
         }
      }
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   template<typename SetNameAndName>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::identity(const SetNameAndName& pairs) const {
      auto rank = names.size();
      auto half_rank = rank / 2;
      auto ordered_pair = pmr::vector<std::tuple<Name, Name>>();
      auto ordered_pair_index = pmr::vector<std::tuple<Rank, Rank>>();
      ordered_pair.reserve(half_rank);
      ordered_pair_index.reserve(half_rank);
      auto valid_index = pmr::vector<bool>(rank, true);
      for (Rank i = 0; i < rank; i++) {
         if (valid_index[i]) {
            const auto& name_to_find = names[i];
            const Name* name_correspond = nullptr;
            for (const auto& [name_1, name_2] : pairs) {
               if (name_1 == name_to_find) {
                  name_correspond = &name_2;
                  break;
               }
               if (name_2 == name_to_find) {
                  name_correspond = &name_1;
                  break;
               }
            }
            ordered_pair.push_back({name_to_find, *name_correspond});
            auto index_correspond = name_to_index.at(*name_correspond);
            ordered_pair_index.push_back({i, index_correspond});
            valid_index[index_correspond] = false;
         }
      }

      auto result = same_shape().zero();

      for (auto& [symmetries, block] : result.core->blocks) {
         auto dimension = pmr::vector<Size>(rank);
         auto leading = pmr::vector<Size>(rank);
         for (Rank i = rank; i-- > 0;) {
            dimension[i] = core->edges[i].map[symmetries[i]];
            if (i == rank - 1) {
               leading[i] = 1;
            } else {
               leading[i] = leading[i + 1] * dimension[i + 1];
            }
         }
         auto pair_dimension = pmr::vector<Size>();
         auto pair_leading = pmr::vector<Size>();
         pair_dimension.reserve(half_rank);
         pair_leading.reserve(half_rank);
         for (Rank i = 0; i < half_rank; i++) {
            pair_dimension.push_back(dimension[std::get<0>(ordered_pair_index[i])]);
            pair_leading.push_back(leading[std::get<0>(ordered_pair_index[i])] + leading[std::get<1>(ordered_pair_index[i])]);
            // ordered_pair_index使用较大的leading进行从大到小排序，所以pair_leading一定降序
         }
         set_to_identity(block.data(), pair_dimension, pair_leading, half_rank);
      }

      return result;
   }
} // namespace TAT
#endif
