/**
 * \file tensor_miscellaneous.hpp
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
#ifndef TAT_TENSOR_MISCELLANEOUS_HPP
#define TAT_TENSOR_MISCELLANEOUS_HPP

#include "tensor.hpp"
#include "timer.hpp"

namespace TAT {
   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>
   Tensor<ScalarType, Symmetry, Name>::multiple(const SingularType& S, const Name& name, char direction, bool division) const {
      auto guard = multiple_guard();
      bool different_direction;
      if (direction == 'u' || direction == 'U') {
         different_direction = false;
      } else if (direction == 'v' || direction == 'V') {
         different_direction = true;
      } else {
         return copy();
      }
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

         if (division) {
            for (Size a = 0; a < m; a++) {
               for (Size b = 0; b < k; b++) {
#ifdef TAT_USE_SINGULAR_MATRIX
                  auto v = vector_in_S[b * dimension_plus_one];
#else
                  auto v = vector_in_S[b];
#endif
                  for (Size c = 0; c < n; c++) {
                     *(data_destination++) = *(data_source++) / v;
                  }
               }
            }
         } else {
            for (Size a = 0; a < m; a++) {
               for (Size b = 0; b < k; b++) {
#ifdef TAT_USE_SINGULAR_MATRIX
                  auto v = vector_in_S[b * dimension_plus_one];
#else
                  auto v = vector_in_S[b];
#endif
                  for (Size c = 0; c < n; c++) {
                     *(data_destination++) = *(data_source++) * v;
                  }
               }
            }
         }
      }
      return result;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::conjugate() const {
      auto guard = conjugate_guard();
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
      auto result = Tensor<ScalarType, Symmetry, Name>(names, result_edges);
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

   /// \private
   template<typename ScalarType>
   void set_to_identity(ScalarType* pointer, const std::vector<Size>& dimension, const std::vector<Size>& leading, Rank rank) {
      auto current_index = std::vector<Size>(rank, 0);
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

   // identity应该是一个static的函数么？
   // 如果是static的, 他的接口应该类似于
   // Tensor::identity({"D", "A", "C", "B"}, {{"A", "B", 6}, {"C", "D", 9}}})
   // 因为我应该避免生成后还需要转置的行为, 所以vector<Name>必须保留
   // 而Edge方面, 不得不写成tuple<Name, Name, Edge>的形式, 太麻烦了
   // 所以像现在这样写成inplace的形式
   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>& Tensor<ScalarType, Symmetry, Name>::identity(const std::set<std::tuple<Name, Name>>& pairs) & {
      // 不需要check use_count, 因为zero调用了set, set调用了transform, transform会check
      zero();

      auto rank = names.size();
      auto half_rank = rank / 2;
      auto ordered_pair = std::vector<std::tuple<Name, Name>>();
      auto ordered_pair_index = std::vector<std::tuple<Rank, Rank>>();
      ordered_pair.reserve(half_rank);
      ordered_pair_index.reserve(half_rank);
      auto valid_index = std::vector<bool>(rank, true);
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
      for (auto& [symmetries, block] : core->blocks) {
         auto dimension = std::vector<Size>(rank);
         auto leading = std::vector<Size>(rank);
         for (Rank i = rank; i-- > 0;) {
            dimension[i] = core->edges[i].map[symmetries[i]];
            if (i == rank - 1) {
               leading[i] = 1;
            } else {
               leading[i] = leading[i + 1] * dimension[i + 1];
            }
         }
         auto pair_dimension = std::vector<Size>();
         auto pair_leading = std::vector<Size>();
         pair_dimension.reserve(half_rank);
         pair_leading.reserve(half_rank);
         for (Rank i = 0; i < half_rank; i++) {
            pair_dimension.push_back(dimension[std::get<0>(ordered_pair_index[i])]);
            pair_leading.push_back(leading[std::get<0>(ordered_pair_index[i])] + leading[std::get<1>(ordered_pair_index[i])]);
            // ordered_pair_index使用较大的leading进行从大到小排序，所以pair_leading一定降序
         }
         set_to_identity(block.data(), pair_dimension, pair_leading, half_rank);
      }

      return *this;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::exponential(const std::set<std::tuple<Name, Name>>& pairs, int step) const {
      real_base_t<ScalarType> norm_max = norm<-1>();
      auto temporary_tensor_rank = 0;
      real_base_t<ScalarType> temporary_tensor_parameter = 1;
      while (temporary_tensor_parameter * norm_max > 1) {
         temporary_tensor_rank += 1;
         temporary_tensor_parameter *= 1. / 2;
      }
      auto temporary_tensor = *this * temporary_tensor_parameter;

      auto result = same_shape().identity(pairs);

      auto power_of_temporary_tensor = Tensor<ScalarType, Symmetry, Name>();

      ScalarType series_parameter = 1;
      for (auto i = 1; i <= step; i++) {
         series_parameter /= i;
         if (i == 1) {
            result += temporary_tensor;
         } else if (i == 2) {
            power_of_temporary_tensor = temporary_tensor.contract(temporary_tensor, pairs);
            // result += series_parameter * power_of_temporary_tensor;
            result = series_parameter * power_of_temporary_tensor + result;
            // power_of_temporary_tensor相乘一次后边应该就会稳定, 这个时候将result放在+的右侧, 会使得result边的排列和左侧一样
            // 从而在 i>2 的时候减少转置
         } else {
            power_of_temporary_tensor = power_of_temporary_tensor.contract(temporary_tensor, pairs);
            result += series_parameter * power_of_temporary_tensor;
         }
      }

      for (auto i = 0; i < temporary_tensor_rank; i++) {
         result = result.contract(result, pairs);
      }
      return result;
   }
} // namespace TAT
#endif
