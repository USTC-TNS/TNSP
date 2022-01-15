/**
 * \file transpose.hpp
 *
 * Copyright (C) 2019-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include "../utility/const_integral.hpp"

namespace TAT {

   inline timer transpose_kernel_core_guard("transpose_kernel_core");

   // It is the same to numpy, random read/linear write is better than linear read/random write.
   template<typename ScalarType, bool parity, bool loop_last = false, typename LineSizeType = int>
   void tensor_transpose_kernel(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const Size* const __restrict dimension,
         const Size* const __restrict leading_source,
         const Size* const __restrict leading_destination, // leading_destination is the proper order
         const Rank rank,
         const Rank line_rank = 0,
         const LineSizeType line_size = 0) {
      auto timer_guard = transpose_kernel_core_guard();

      const ScalarType* current_source = data_source;
      ScalarType* current_destination = data_destination;
      pmr::vector<Size> index_list(rank, 0);
      while (true) {
         Rank active_position = rank - 1;

         if constexpr (loop_last) {
            // come into this branch iff the last dimension is the same and its leading is 1.
            for (Size line_index = line_rank; line_index < rank - 1; line_index++) {
               index_list[line_index] = dimension[line_index] - 1;
            }
            index_list[rank - 1] = dimension[rank - 1];
            const Size line_size_value = line_size.value();
            for (Size i = 0; i < line_size_value; i++) {
               if constexpr (parity) {
                  *current_destination++ = -*current_source++;
               } else {
                  *current_destination++ = *current_source++;
               }
            }
         } else {
            if constexpr (parity) {
               *current_destination = -*current_source;
            } else {
               *current_destination = *current_source;
            }

            index_list[active_position]++;
            current_source += leading_source[active_position];
            current_destination += leading_destination[active_position];
         }

         while (index_list[active_position] == dimension[active_position]) {
            index_list[active_position] = 0;
            current_source -= dimension[active_position] * leading_source[active_position];
            current_destination -= dimension[active_position] * leading_destination[active_position];

            if (active_position == 0) {
               return;
            }
            active_position--;

            index_list[active_position]++;
            current_source += leading_source[active_position];
            current_destination += leading_destination[active_position];
         }
      }
   }

   template<typename ScalarType, bool parity>
   void simple_transpose(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const pmr::vector<Rank>& plan_source_to_destination,
         const pmr::vector<Rank>& plan_destination_to_source,
         const pmr::vector<Size>& dimensions_source,
         const pmr::vector<Size>& dimensions_destination,
         const pmr::vector<Size>& leadings_source,
         const pmr::vector<Size>& leadings_destination,
         const Rank rank) {
      auto leadings_source_by_destination = pmr::vector<Size>();
      leadings_source_by_destination.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         auto j = plan_destination_to_source[i];
         leadings_source_by_destination.push_back(leadings_source[j]);
      }

      if (leadings_source_by_destination[rank - 1] == 1 && leadings_destination[rank - 1] == 1) {
         Rank line_rank = rank - 1;
         Size expect_leading = 1;
         while (expect_leading *= dimensions_destination[line_rank],
                leadings_source_by_destination[line_rank - 1] == expect_leading && leadings_destination[line_rank - 1] == expect_leading) {
            if (line_rank == 0) {
               // totally linear copy
               break;
            }
            line_rank--;
         }
         const auto const_expect_leading_variant = to_const_integral<Size, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(expect_leading);
         std::visit(
               [&](const auto& const_expect_leading) {
                  tensor_transpose_kernel<ScalarType, parity, true>(
                        data_source,
                        data_destination,
                        dimensions_destination.data(),
                        leadings_source_by_destination.data(),
                        leadings_destination.data(),
                        rank,
                        line_rank,
                        const_expect_leading);
               },
               const_expect_leading_variant);
      } else {
         tensor_transpose_kernel<ScalarType, parity>(
               data_source,
               data_destination,
               dimensions_destination.data(),
               leadings_source_by_destination.data(),
               leadings_destination.data(),
               rank);
      }
   }

   inline timer transpose_kernel_guard("transpose_kernel");

   template<typename ScalarType>
   void do_transpose(
         const ScalarType* data_source,
         ScalarType* data_destination,
         const pmr::vector<Rank>& plan_source_to_destination,
         const pmr::vector<Rank>& plan_destination_to_source,
         const pmr::vector<Size>& dimensions_source,
         const pmr::vector<Size>& dimensions_destination,
         const pmr::vector<Size>& leadings_source,
         const pmr::vector<Size>& leadings_destination,
         Rank rank,
         Size total_size,
         bool parity) {
      auto timer_guard = transpose_kernel_guard();

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
      // rank != 0, dimension != 0

      if (parity) {
         simple_transpose<ScalarType, true>(
               data_source,
               data_destination,
               plan_source_to_destination,
               plan_destination_to_source,
               dimensions_source,
               dimensions_destination,
               leadings_source,
               leadings_destination,
               rank);
      } else {
         simple_transpose<ScalarType, false>(
               data_source,
               data_destination,
               plan_source_to_destination,
               plan_destination_to_source,
               dimensions_source,
               dimensions_destination,
               leadings_source,
               leadings_destination,
               rank);
      }
   }

   template<typename ScalarType>
   void matrix_transpose(Size m, Size n, const ScalarType* const source, ScalarType* const destination) {
      // auto dimension = pmr::vector<Size>{m, n};
      // auto leading_source = pmr::vector<Size>{n, 1};
      // auto leading_destination = pmr::vector<Size>{1, m};
      auto dimension = pmr::vector<Size>{n, m};
      auto leading_source = pmr::vector<Size>{1, n};
      auto leading_destination = pmr::vector<Size>{m, 1};
      tensor_transpose_kernel<ScalarType, false>(source, destination, dimension.data(), leading_source.data(), leading_destination.data(), 2);
   }
} // namespace TAT
#endif
