/**
 * \file transpose.hpp
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
#ifndef TAT_TRANSPOSE_HPP
#define TAT_TRANSPOSE_HPP

#include <tuple>

#include "basic_type.hpp"

// TAT_USE_MKL_TRANSPOSE
namespace TAT {
   template<typename ScalarType>
   void line_copy(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const Rank rank,
         const std::vector<Size>& scalar_line_size,
         const std::vector<Size>& scalar_leading_source,
         const std::vector<Size>& scalar_leading_destination,
         const bool parity) {
      if (rank == 0) {
         data_destination[0] = parity ? -data_source[0] : data_source[0];
         return;
      }
      // leading[0] may not be 1
      if (rank == 1) {
         if (parity) {
            for (Rank i = 0, source_index = 0, destination_index = 0; i < scalar_line_size[0];
                 i++, source_index += scalar_leading_source[0], destination_index += scalar_leading_destination[0]) {
               data_destination[destination_index] = -data_source[source_index];
            }
         } else {
            for (Rank i = 0, source_index = 0, destination_index = 0; i < scalar_line_size[0];
                 i++, source_index += scalar_leading_source[0], destination_index += scalar_leading_destination[0]) {
               data_destination[destination_index] = data_source[source_index];
            }
         }
         return;
      }
      auto index_list = std::vector<Rank>(rank, 0);
      const ScalarType* current_source = data_source;
      ScalarType* current_destination = data_destination;
      while (true) {
         if (parity) {
            for (Rank i = 0, source_index = 0, destination_index = 0; i < scalar_line_size[0];
                 i++, source_index += scalar_leading_source[0], destination_index += scalar_leading_destination[0]) {
               current_destination[destination_index] = -current_source[source_index];
            }
         } else {
            for (Rank i = 0, source_index = 0, destination_index = 0; i < scalar_line_size[0];
                 i++, source_index += scalar_leading_source[0], destination_index += scalar_leading_destination[0]) {
               current_destination[destination_index] = current_source[source_index];
            }
         }
         auto current_position = 1;
         index_list[current_position]++;
         current_source += scalar_leading_source[current_position];
         current_destination += scalar_leading_destination[current_position];
         while (index_list[current_position] == scalar_line_size[current_position]) {
            index_list[current_position] = 0;
            current_source -= scalar_leading_source[current_position] * scalar_line_size[current_position];
            current_destination -= scalar_leading_destination[current_position] * scalar_line_size[current_position];

            current_position++;
            if (current_position == rank) {
               return;
            }
            index_list[current_position]++;
            current_source += scalar_leading_source[current_position];
            current_destination += scalar_leading_destination[current_position];
         }
      }
      // TODO: line size特化
   }

   template<typename ScalarType>
   void matrix_transpose_kernel(
         const Size dimension_of_M,
         const Size dimension_of_N,
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const Size leading_source,
         const Size leading_destination,
         const Size scalar_size_source,
         const Size scalar_size_destination,
         const Rank scalar_rank,
         const std::vector<Size>& scalar_line_size,
         const std::vector<Size>& scalar_leading_source,
         const std::vector<Size>& scalar_leading_destination,
         const bool parity) {
      if (scalar_rank == 0) {
         // 临时的优化
         if (parity) {
            for (Size i = 0; i < dimension_of_M; i++) {
               for (Size j = 0; j < dimension_of_N; j++) {
                  auto line_destination = data_destination + j * leading_destination + i * scalar_size_destination;
                  auto line_source = data_source + i * leading_source + j * scalar_size_source;
                  *line_destination = -*line_source;
               }
            }
         } else {
            for (Size i = 0; i < dimension_of_M; i++) {
               for (Size j = 0; j < dimension_of_N; j++) {
                  auto line_destination = data_destination + j * leading_destination + i * scalar_size_destination;
                  auto line_source = data_source + i * leading_source + j * scalar_size_source;
                  *line_destination = *line_source;
               }
            }
         }
      } else if (scalar_rank == 1) {
         // 临时的优化
         const auto scalar_leading_destination_head = scalar_leading_destination[0];
         const auto scalar_leading_source_head = scalar_leading_source[0];
         if (parity) {
            for (Size i = 0; i < dimension_of_M; i++) {
               for (Size j = 0; j < dimension_of_N; j++) {
                  auto line_destination = data_destination + j * leading_destination + i * scalar_size_destination;
                  auto line_source = data_source + i * leading_source + j * scalar_size_source;
                  for (Rank k = 0, source_index = 0, destination_index = 0; k < scalar_line_size[0];
                       k++, source_index += scalar_leading_source_head, destination_index += scalar_leading_destination_head) {
                     line_destination[destination_index] = -line_source[source_index];
                  }
               }
            }
         } else {
            for (Size i = 0; i < dimension_of_M; i++) {
               for (Size j = 0; j < dimension_of_N; j++) {
                  auto line_destination = data_destination + j * leading_destination + i * scalar_size_destination;
                  auto line_source = data_source + i * leading_source + j * scalar_size_source;
                  for (Rank k = 0, source_index = 0, destination_index = 0; k < scalar_line_size[0];
                       k++, source_index += scalar_leading_source_head, destination_index += scalar_leading_destination_head) {
                     line_destination[destination_index] = line_source[source_index];
                  }
               }
            }
         }

      } else {
         for (Size i = 0; i < dimension_of_M; i++) {
            for (Size j = 0; j < dimension_of_N; j++) {
               auto line_destination = data_destination + j * leading_destination + i * scalar_size_destination;
               auto line_source = data_source + i * leading_source + j * scalar_size_source;
               line_copy(line_source, line_destination, scalar_rank, scalar_line_size, scalar_leading_source, scalar_leading_destination, parity);
               // TODO: 向量化的转置
            }
         }
      }
   }

   template<typename ScalarType, Size cache_size, Size... other>
   void matrix_transpose(
         const Size dimension_of_M,
         const Size dimension_of_N,
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const Size leading_source,
         const Size leading_destination,
         const Size scalar_size_source,
         const Size scalar_size_destination,
         const Rank scalar_rank,
         const std::vector<Size>& scalar_line_size,
         const std::vector<Size>& scalar_leading_source,
         const std::vector<Size>& scalar_leading_destination,
         const bool parity) {
      Size block_size = 1;
      // TODO: 是否应该乘以二做冗余？
      while (block_size * block_size * (scalar_size_source + scalar_size_destination) * sizeof(ScalarType) < cache_size) {
         block_size <<= 1;
      }
      block_size >>= 1;
      for (Size i = 0; i < dimension_of_M; i += block_size) {
         for (Size j = 0; j < dimension_of_N; j += block_size) {
            auto block_dst = data_destination + j * leading_destination + i * scalar_size_destination;
            auto block_src = data_source + i * leading_source + j * scalar_size_source;
            auto little_dimension_of_M = dimension_of_M - i;
            auto little_dimension_of_N = dimension_of_N - j;
            little_dimension_of_M = (block_size <= little_dimension_of_M) ? block_size : little_dimension_of_M;
            little_dimension_of_N = (block_size <= little_dimension_of_N) ? block_size : little_dimension_of_N;

            if constexpr (sizeof...(other) == 0) {
               matrix_transpose_kernel<ScalarType>(
                     little_dimension_of_M,
                     little_dimension_of_N,
                     block_src,
                     block_dst,
                     leading_source,
                     leading_destination,
                     scalar_size_source,
                     scalar_size_destination,
                     scalar_rank,
                     scalar_line_size,
                     scalar_leading_source,
                     scalar_leading_destination,
                     parity);
            } else {
               matrix_transpose<ScalarType, other...>(
                     little_dimension_of_M,
                     little_dimension_of_N,
                     block_src,
                     block_dst,
                     leading_source,
                     leading_destination,
                     scalar_size_source,
                     scalar_size_destination,
                     scalar_rank,
                     scalar_line_size,
                     scalar_leading_source,
                     scalar_leading_destination,
                     parity);
            }
         }
      }
   }

   inline const Size l1_cache = 32768;
   inline const Size l2_cache = 262144;
   inline const Size l3_cache = 9437184;
   // TODO: 如何确定系统cache

   template<typename ScalarType>
   void block_transpose(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leading_source,
         const std::vector<Size>& leading_destination,
         [[maybe_unused]] const Size& size,
         const Rank rank,
         const std::vector<Size>& scalar_line_size,
         const std::vector<Size>& scalar_leading_source,
         const std::vector<Size>& scalar_leading_destination,
         const Rank line_rank,
         const bool parity) {
      std::vector<Size> index_list_source(rank, 0);
      std::vector<Size> index_list_destination(rank, 0);
      Size offset_source = 0;
      Size offset_destination = 0;

      // M N => N M
      const Size dimension_of_N = dimensions_source[rank - 1];
      const Size dimension_of_M = dimensions_destination[rank - 1];
      const Rank position_of_M_in_source = plan_destination_to_source[rank - 1];
      const Rank position_of_N_in_destination = plan_source_to_destination[rank - 1];
      const Size leading_of_M_in_source = leading_source[position_of_M_in_source];
      const Size leading_of_N_in_source = leading_source[rank - 1];
      const Size leading_of_N_in_destination = leading_destination[position_of_N_in_destination];
      const Size leading_of_M_in_destination = leading_destination[rank - 1];

      while (true) {
         // TODO: l3太大了, 所以只按着l2和l1来划分, 这样合适么
         matrix_transpose<ScalarType, l2_cache, l1_cache>(
               dimension_of_M,
               dimension_of_N,
               data_source + offset_source,
               data_destination + offset_destination,
               leading_of_M_in_source,
               leading_of_N_in_destination,
               leading_of_N_in_source,
               leading_of_M_in_destination,
               line_rank,
               scalar_line_size,
               scalar_leading_source,
               scalar_leading_destination,
               parity);

         Rank active_position_in_destination = rank - 2;
         Rank active_position_in_source = plan_destination_to_source[active_position_in_destination];
         if (active_position_in_source == rank - 1) {
            if (active_position_in_destination == 0) {
               return;
            }
            active_position_in_destination--;
            active_position_in_source = plan_destination_to_source[active_position_in_destination];
         }

         index_list_source[active_position_in_source]++;
         offset_source += leading_source[active_position_in_source];
         index_list_destination[active_position_in_destination]++;
         offset_destination += leading_destination[active_position_in_destination];

         while (index_list_destination[active_position_in_destination] == dimensions_destination[active_position_in_destination]) {
            index_list_source[active_position_in_source] = 0;
            offset_source -= dimensions_source[active_position_in_source] * leading_source[active_position_in_source];
            index_list_destination[active_position_in_destination] = 0;
            offset_destination -= dimensions_destination[active_position_in_destination] * leading_destination[active_position_in_destination];
            if (active_position_in_destination == 0) {
               return;
            }
            active_position_in_destination--;
            active_position_in_source = plan_destination_to_source[active_position_in_destination];
            if (active_position_in_source == rank - 1) {
               if (active_position_in_destination == 0) {
                  return;
               }
               active_position_in_destination--;
               active_position_in_source = plan_destination_to_source[active_position_in_destination];
            }
            index_list_source[active_position_in_source]++;
            offset_source += leading_source[active_position_in_source];
            index_list_destination[active_position_in_destination]++;
            offset_destination += leading_destination[active_position_in_destination];
         }
      }
   }

   inline auto cutting_for_transpose(
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leading_source,
         const std::vector<Size>& leading_destination,
         const Rank& rank) {
      std::vector<bool> is_one_source(rank);
      std::vector<bool> is_one_destination(rank);
      for (Rank i = 0; i < rank; i++) {
         is_one_source[i] = dimensions_source[i] == 1;
      }
      for (Rank i = 0; i < rank; i++) {
         is_one_destination[i] = dimensions_destination[i] == 1;
      }
      std::vector<Rank> accumulated_one_source(rank);
      std::vector<Rank> accumulated_one_destination(rank);
      accumulated_one_source[0] = is_one_source[0];
      for (Rank i = 1; i < rank; i++) {
         accumulated_one_source[i] = accumulated_one_source[i - 1] + Rank(is_one_source[i]);
      }
      accumulated_one_destination[0] = is_one_destination[0];
      for (Rank i = 1; i < rank; i++) {
         accumulated_one_destination[i] = accumulated_one_destination[i - 1] + Rank(is_one_destination[i]);
      }

      std::vector<Rank> result_plan_source_to_destination;
      std::vector<Rank> result_plan_destination_to_source;
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

      std::vector<Size> result_dimensions_source;
      std::vector<Size> result_dimensions_destination;
      std::vector<Size> result_leading_source;
      std::vector<Size> result_leading_destination;
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
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leading_source,
         const std::vector<Size>& leading_destination,
         const Rank& rank) {
      std::vector<bool> merging_source_to_destination(rank);
      std::vector<bool> merging_destination_to_source(rank);
      for (Rank i = 1; i < rank; i++) {
         if (const auto j = plan_source_to_destination[i]; i != 0 && j != 0 && j - 1 == plan_source_to_destination[i - 1] &&
                                                           leading_source[i - 1] == leading_source[i] * dimensions_source[i] &&
                                                           leading_destination[j - 1] == leading_destination[j] * dimensions_destination[j]) {
            merging_source_to_destination[i] = true;
            merging_destination_to_source[plan_source_to_destination[i]] = true;
         } else {
            merging_source_to_destination[i] = false;
            merging_destination_to_source[plan_source_to_destination[i]] = false;
         }
      }

      std::vector<Rank> accumulated_merging_source_to_destination(rank);
      std::vector<Rank> accumulated_merging_destination_to_source(rank);
      accumulated_merging_source_to_destination[0] = 0;
      for (Rank i = 1; i < rank; i++) {
         accumulated_merging_source_to_destination[i] = accumulated_merging_source_to_destination[i - 1] + Rank(merging_source_to_destination[i]);
      }
      accumulated_merging_destination_to_source[0] = 0;
      for (Rank i = 1; i < rank; i++) {
         accumulated_merging_destination_to_source[i] = accumulated_merging_destination_to_source[i - 1] + Rank(merging_destination_to_source[i]);
      }
      std::vector<Rank> result_plan_source_to_destination;
      std::vector<Rank> result_plan_destination_to_source;
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
      std::vector<Size> result_dimensions_source(result_rank);
      std::vector<Size> result_dimensions_destination(result_rank);
      std::vector<Size> result_leading_source(result_rank);
      std::vector<Size> result_leading_destination(result_rank);
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

   template<typename ScalarType>
   void do_transpose(
         const ScalarType* data_source,
         ScalarType* data_destination,
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leading_source,
         const std::vector<Size>& leading_destination,
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

      auto scalar_leading_source = std::vector<Size>();
      auto scalar_leading_destination = std::vector<Size>();
      auto scalar_line_size = std::vector<Size>();
      Rank effective_rank = merging_rank;
      Rank line_rank = 0;

      while (effective_rank != 0 && merging_plan_source_to_destination[effective_rank - 1] == effective_rank - 1) {
         scalar_line_size.push_back(merging_dimensions_destination[effective_rank - 1]);
         scalar_leading_source.push_back(merging_leading_source[effective_rank - 1]);
         scalar_leading_destination.push_back(merging_leading_destination[effective_rank - 1]);
         effective_rank--;
         line_rank++;
      }

      if (effective_rank == 0) {
         line_copy(data_source, data_destination, line_rank, scalar_line_size, scalar_leading_source, scalar_leading_destination, parity);
         return;
      }

      // TODO: 需要考虑极端细致的情况
      block_transpose(
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
            scalar_line_size,
            scalar_leading_source,
            scalar_leading_destination,
            line_rank,
            parity);
   }
} // namespace TAT
#endif
