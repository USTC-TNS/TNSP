/**
 * \file transpose.hpp
 *
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_TRANSPOSE_HPP
#define TAT_TRANSPOSE_HPP

#include <tuple>

#include "misc.hpp"

namespace TAT {
   template<class ScalarType, bool parity>
   void matrix_transpose_kernel(
         const Size dimension_of_M,
         const Size dimension_of_N,
         const ScalarType* const __restrict data_source,
         const Size leading_source,
         ScalarType* const __restrict data_destination,
         const Size leading_destination,
         const Size line_size,
         const Size line_leading_source,
         const Size line_leading_destination) {
      for (Size i = 0; i < dimension_of_M; i++) {
         for (Size j = 0; j < dimension_of_N; j++) {
            auto line_destination = data_destination + j * leading_destination + i * line_leading_destination;
            auto line_source = data_source + i * leading_source + j * line_leading_source;
            for (Size k = 0; k < line_size; k++) {
               if constexpr (parity) {
                  line_destination[k] = -line_source[k];
               } else {
                  line_destination[k] = line_source[k];
               }
            }
            // TODO: 向量化
            // TODO: line size特化
         }
      }
   }

   template<class ScalarType, bool parity, Size cache_size, Size... other>
   void matrix_transpose(
         Size dimension_of_M,
         Size dimension_of_N,
         const ScalarType* const __restrict data_source,
         const Size leading_source,
         ScalarType* const __restrict data_destination,
         const Size leading_destination,
         const Size line_size,
         const Size line_leading_source,
         const Size line_leading_destination) {
      Size block_size = 1;
      // TODO: 是否应该乘以二做冗余？
      while (block_size * block_size * line_size * sizeof(ScalarType) * 2 < cache_size) {
         block_size <<= 1;
      }
      block_size >>= 1;
      for (Size i = 0; i < dimension_of_M; i += block_size) {
         for (Size j = 0; j < dimension_of_N; j += block_size) {
            auto block_dst = data_destination + j * leading_destination + i * line_leading_destination;
            auto block_src = data_source + i * leading_source + j * line_leading_source;
            auto little_dimension_of_M = dimension_of_M - i;
            auto little_dimension_of_N = dimension_of_N - j;
            little_dimension_of_M = (block_size <= little_dimension_of_M) ? block_size : little_dimension_of_M;
            little_dimension_of_N = (block_size <= little_dimension_of_N) ? block_size : little_dimension_of_N;

            if constexpr (sizeof...(other) == 0) {
               matrix_transpose_kernel<ScalarType, parity>(
                     little_dimension_of_M,
                     little_dimension_of_N,
                     block_src,
                     leading_source,
                     block_dst,
                     leading_destination,
                     line_size,
                     line_leading_source,
                     line_leading_destination);
            } else {
               matrix_transpose<ScalarType, parity, other...>(
                     little_dimension_of_M,
                     little_dimension_of_N,
                     block_src,
                     leading_source,
                     block_dst,
                     leading_destination,
                     line_size,
                     line_leading_source,
                     line_leading_destination);
            }
         }
      }
   }

   inline const Size l1_cache = 32768;
   inline const Size l2_cache = 262144;
   inline const Size l3_cache = 9437184;
   // TODO: 如何确定系统cache

   template<class ScalarType, bool parity>
   void block_transpose(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const vector<Rank>& plan_source_to_destination,
         const vector<Rank>& plan_destination_to_source,
         const vector<Size>& dimensions_source,
         const vector<Size>& dimensions_destination,
         const vector<Size>& leading_source,
         const vector<Size>& leading_destination,
         [[maybe_unused]] const Size& size,
         const Rank rank,
         const Size line_size) {
      // 调用者保证不会出现rank0, rank1的情况
      // 本程序做了许多小矩阵转置, 故要求最后一位不同, 若最后一位相同, 通过将他当作整个Scalar类型操作来避免问题
      // 如果没有leading问题, 则倒数第二位一定不同
      // 现在leading导致不能merge, 倒数第二位仍然相同
      //
      // 方案： 想把line size -> line size leading的bug处理好
      // 再把line size的生成上移
      // 然后将line size 替换为数组

      vector<Size> index_list_source(rank, 0);
      vector<Size> index_list_destination(rank, 0);
      Size offset_source = 0;
      Size offset_destination = 0;

      // M N => N M
      const Size dimension_of_N = dimensions_source[rank - 1];
      const Size dimension_of_M = dimensions_destination[rank - 1];
      const Rank position_of_M_in_source = plan_destination_to_source[rank - 1];
      const Rank position_of_N_in_destination = plan_source_to_destination[rank - 1];
      const Size leading_of_M_in_source = leading_source[position_of_M_in_source];
      const Size leading_of_N_in_destination = leading_destination[position_of_N_in_destination];
      const Size line_leading_source = leading_source[rank - 1];
      const Size line_leading_destination = leading_destination[rank - 1];

      while (true) {
         // TODO: l3太大了, 所以只按着l2和l1来划分, 这样合适么
         matrix_transpose<ScalarType, parity, l2_cache, l1_cache>(
               dimension_of_M,
               dimension_of_N,
               data_source + offset_source,
               leading_of_M_in_source,
               data_destination + offset_destination,
               leading_of_N_in_destination,
               line_size,
               line_leading_source,
               line_leading_destination);

         Rank active_position_in_destination = rank - 2;
         Rank active_position_in_source = plan_destination_to_source[active_position_in_destination];
         if (active_position_in_source == rank - 1) {
            if (active_position_in_destination == 0) {
               return;
            }
            active_position_in_destination -= 1;
            active_position_in_source = plan_destination_to_source[active_position_in_destination];
         }

         index_list_source[active_position_in_source] += 1;
         offset_source += leading_source[active_position_in_source];
         index_list_destination[active_position_in_destination] += 1;
         offset_destination += leading_destination[active_position_in_destination];

         while (index_list_destination[active_position_in_destination] == dimensions_destination[active_position_in_destination]) {
            index_list_source[active_position_in_source] = 0;
            offset_source -= dimensions_source[active_position_in_source] * leading_source[active_position_in_source];
            index_list_destination[active_position_in_destination] = 0;
            offset_destination -= dimensions_destination[active_position_in_destination] * leading_destination[active_position_in_destination];
            if (active_position_in_destination == 0) {
               return;
            }
            active_position_in_destination -= 1;
            active_position_in_source = plan_destination_to_source[active_position_in_destination];
            if (active_position_in_source == rank - 1) {
               if (active_position_in_destination == 0) {
                  return;
               }
               active_position_in_destination -= 1;
               active_position_in_source = plan_destination_to_source[active_position_in_destination];
            }
            index_list_source[active_position_in_source] += 1;
            offset_source += leading_source[active_position_in_source];
            index_list_destination[active_position_in_destination] += 1;
            offset_destination += leading_destination[active_position_in_destination];
         }
      }
   }

   inline auto cutting_for_transpose(
         const vector<Rank>& plan_source_to_destination,
         const vector<Rank>& plan_destination_to_source,
         const vector<Size>& dimensions_source,
         const vector<Size>& dimensions_destination,
         const vector<Size>& leading_source,
         const vector<Size>& leading_destination,
         const Rank& rank) {
      vector<bool> is_one_source(rank);
      vector<bool> is_one_destination(rank);
      for (Rank i = 0; i < rank; i++) {
         is_one_source[i] = dimensions_source[i] == 1;
      }
      for (Rank i = 0; i < rank; i++) {
         is_one_destination[i] = dimensions_destination[i] == 1;
      }
      vector<Rank> accumulated_one_source(rank);
      vector<Rank> accumulated_one_destination(rank);
      accumulated_one_source[0] = is_one_source[0];
      for (Rank i = 1; i < rank; i++) {
         accumulated_one_source[i] = accumulated_one_source[i - 1] + Rank(is_one_source[i]);
      }
      accumulated_one_destination[0] = is_one_destination[0];
      for (Rank i = 1; i < rank; i++) {
         accumulated_one_destination[i] = accumulated_one_destination[i - 1] + Rank(is_one_destination[i]);
      }

      vector<Rank> result_plan_source_to_destination;
      vector<Rank> result_plan_destination_to_source;
      for (Rank i = 0; i < rank; i++) {
         if (!is_one_source[i]) {
            result_plan_source_to_destination.push_back(plan_source_to_destination[i] - accumulated_one_destination[plan_source_to_destination[i]]);
         }
      }
      for (Rank i = 0; i < rank; i++) {
         if (!is_one_destination[i]) {
            result_plan_destination_to_source.push_back(plan_destination_to_source[i] - accumulated_one_source[plan_destination_to_source[i]]);
         }
      }
      auto result_rank = Rank(result_plan_destination_to_source.size());

      vector<Size> result_dimensions_source;
      vector<Size> result_dimensions_destination;
      vector<Size> result_leading_source;
      vector<Size> result_leading_destination;
      for (Rank i = 0; i < rank; i++) {
         if (dimensions_source[i] != 1) {
            result_dimensions_source.push_back(dimensions_source[i]);
            result_leading_source.push_back(leading_source[i]);
         }
      }
      for (Rank i = 0; i < rank; i++) {
         if (dimensions_destination[i] != 1) {
            result_dimensions_destination.push_back(dimensions_destination[i]);
            result_leading_destination.push_back(leading_destination[i]);
         }
      }
      return std::make_tuple(
            std::move(result_plan_source_to_destination),
            std::move(result_plan_destination_to_source),
            std::move(result_dimensions_source),
            std::move(result_dimensions_destination),
            std::move(result_leading_source),
            std::move(result_leading_destination),
            result_rank);
   }

   inline auto merging_for_transpose(
         const vector<Rank>& plan_source_to_destination,
         const vector<Rank>& plan_destination_to_source,
         const vector<Size>& dimensions_source,
         const vector<Size>& dimensions_destination,
         const vector<Size>& leading_source,
         const vector<Size>& leading_destination,
         const Rank& rank) {
      vector<bool> merging_source_to_destination(rank);
      vector<bool> merging_destination_to_source(rank);
      for (Rank i = 1; i < rank; i++) {
         if (Rank j = plan_source_to_destination[i]; i != 0 && j != 0 && j - 1 == plan_source_to_destination[i - 1] &&
                                                     leading_source[i - 1] == leading_source[i] * dimensions_source[i] &&
                                                     leading_destination[j - 1] == leading_destination[j] * dimensions_destination[j]) {
            merging_source_to_destination[i] = true;
            merging_destination_to_source[plan_source_to_destination[i]] = true;
         } else {
            merging_source_to_destination[i] = false;
            merging_destination_to_source[plan_source_to_destination[i]] = false;
         }
      }

      vector<Rank> accumulated_merging_source_to_destination(rank);
      vector<Rank> accumulated_merging_destination_to_source(rank);
      accumulated_merging_source_to_destination[0] = 0;
      for (Rank i = 1; i < rank; i++) {
         accumulated_merging_source_to_destination[i] = accumulated_merging_source_to_destination[i - 1] + Rank(merging_source_to_destination[i]);
      }
      accumulated_merging_destination_to_source[0] = 0;
      for (Rank i = 1; i < rank; i++) {
         accumulated_merging_destination_to_source[i] = accumulated_merging_destination_to_source[i - 1] + Rank(merging_destination_to_source[i]);
      }
      vector<Rank> result_plan_source_to_destination;
      vector<Rank> result_plan_destination_to_source;
      for (Rank i = 0; i < rank; i++) {
         if (!merging_source_to_destination[i]) {
            result_plan_source_to_destination.push_back(
                  plan_source_to_destination[i] - accumulated_merging_destination_to_source[plan_source_to_destination[i]]);
         }
      }
      for (Rank i = 0; i < rank; i++) {
         if (!merging_destination_to_source[i]) {
            result_plan_destination_to_source.push_back(
                  plan_destination_to_source[i] - accumulated_merging_source_to_destination[plan_destination_to_source[i]]);
         }
      }
      auto result_rank = Rank(result_plan_source_to_destination.size());
      vector<Size> result_dimensions_source(result_rank);
      vector<Size> result_dimensions_destination(result_rank);
      vector<Size> result_leading_source(result_rank);
      vector<Size> result_leading_destination(result_rank);
      for (Rank i = result_rank, j = rank; i-- > 0;) {
         result_leading_source[i] = leading_source[--j];
         result_dimensions_source[i] = dimensions_source[j];
         while (merging_source_to_destination[j]) {
            result_dimensions_source[i] *= dimensions_source[--j];
         }
      }
      for (Rank i = result_rank, j = rank; i-- > 0;) {
         result_leading_destination[i] = leading_destination[--j];
         result_dimensions_destination[i] = dimensions_destination[j];
         while (merging_destination_to_source[j]) {
            result_dimensions_destination[i] *= dimensions_destination[--j];
         }
      }

      return std::make_tuple(
            std::move(result_plan_source_to_destination),
            std::move(result_plan_destination_to_source),
            std::move(result_dimensions_source),
            std::move(result_dimensions_destination),
            std::move(result_leading_source),
            std::move(result_leading_destination),
            result_rank);
   }

   template<class ScalarType>
   void do_transpose(
         const ScalarType* data_source,
         ScalarType* data_destination,
         const vector<Rank>& plan_source_to_destination,
         const vector<Rank>& plan_destination_to_source,
         const vector<Size>& dimensions_source,
         const vector<Size>& dimensions_destination,
         const vector<Size>& leading_source,
         const vector<Size>& leading_destination,
         Rank rank,
         Size total_size,
         bool parity) {
      if (total_size == 0) {
         return;
      }
      if (total_size == 1) {
         if (parity) {
            *data_destination = -*data_source;
         } else {
            *data_destination = *data_source;
         }
         return;
      }

      auto [cutting_plan_source_to_destination,
            cutting_plan_destination_to_source,
            cutting_dimensions_source,
            cutting_dimensions_destination,
            cutting_leading_source,
            cutting_leading_destination,
            cutting_rank] =
            cutting_for_transpose(
                  plan_source_to_destination,
                  plan_destination_to_source,
                  dimensions_source,
                  dimensions_destination,
                  leading_source,
                  leading_destination,
                  rank);

      auto [merging_plan_source_to_destination,
            merging_plan_destination_to_source,
            merging_dimensions_source,
            merging_dimensions_destination,
            merging_leading_source,
            merging_leading_destination,
            merging_rank] =
            merging_for_transpose(
                  cutting_plan_source_to_destination,
                  cutting_plan_destination_to_source,
                  cutting_dimensions_source,
                  cutting_dimensions_destination,
                  cutting_leading_source,
                  cutting_leading_destination,
                  cutting_rank);

      // TODO
      // 上面的if merging rank = 1 是为了防止自己这边退掉最后的共同边后没有指标了的问题
      // 现在需要允许退掉多个共同边，他们有不同leading， 然后再做上面的判度胺
      Size line_size = 1;
      Rank effective_rank = merging_rank;
      if (merging_plan_source_to_destination.back() == merging_rank - 1) {
         effective_rank--;
         line_size *= merging_dimensions_destination.back();
      }

      if (effective_rank == 0) {
         if (parity) {
            for (Size k = 0; k < total_size; k++) {
               data_destination[k] = -data_source[k];
            }
         } else {
            for (Size k = 0; k < total_size; k++) {
               data_destination[k] = data_source[k];
            }
         }
         return;
      }

      // TODO: 需要考虑极端细致的情况
      if (parity) {
         block_transpose<ScalarType, true>(
               data_source,
               data_destination,
               merging_plan_source_to_destination,
               merging_plan_destination_to_source,
               merging_dimensions_source,
               merging_dimensions_destination,
               merging_leading_source,
               merging_leading_destination,
               total_size,
               effective_rank,
               line_size);
      } else {
         block_transpose<ScalarType, false>(
               data_source,
               data_destination,
               merging_plan_source_to_destination,
               merging_plan_destination_to_source,
               merging_dimensions_source,
               merging_dimensions_destination,
               merging_leading_source,
               merging_leading_destination,
               total_size,
               effective_rank,
               line_size);
      }
   }
} // namespace TAT
#endif
