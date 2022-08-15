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

#include "../utility/allocator.hpp"
#include "../utility/common_variable.hpp"
#include "../utility/const_integral.hpp"
#include "../utility/timer.hpp"

void line_copy_interface(
      std::uint64_t line_number,
      const std::complex<double>** source_lines,
      std::complex<double>** destination_lines,
      std::uint64_t line_size_value,
      bool parity);

namespace TAT {

   inline timer transpose_kernel_core_guard("transpose_kernel_core");
   inline timer transpose_cuda_guard("transpose_cuda");

   // It is the same to numpy, random read/linear write is better than linear read/random write.
   template<typename ScalarType, bool parity, bool loop_last = false, typename LineSizeType = int>
   void tensor_transpose_kernel(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const Size* const __restrict dimension,
         const Size* const __restrict leading_source,
         const Size* const __restrict leading_destination, // leading_destination is the proper order
         const Size total_size,
         const Rank rank,
         const LineSizeType line_size = 0) {
      auto timer_guard = transpose_kernel_core_guard();

      Size line_number = 0;
      if constexpr (loop_last) {
         line_number = total_size / line_size.value();
      }
      const ScalarType** source_lines;
      cudaMallocManaged(&source_lines, line_number * sizeof(const ScalarType*));
      ScalarType** destination_lines;
      cudaMallocManaged(&destination_lines, line_number * sizeof(ScalarType*));
      Size line_index = 0;

      const ScalarType* current_source = data_source;
      ScalarType* current_destination = data_destination;
      pmr::vector<Size> index_list(rank, 0);
      while (true) {
         Rank active_position = rank - 1;

         if constexpr (loop_last) {
            // come into this branch iff the last dimension is the same and its leading is 1.
            index_list[active_position] = dimension[active_position];
            source_lines[line_index] = current_source;
            destination_lines[line_index] = current_destination;
            line_index++;
            const Size line_size_value = line_size.value();
            current_source += line_size_value;
            current_destination += line_size_value;
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
               if constexpr (loop_last) {
                  const Size line_size_value = line_size.value();
                  if constexpr (std::is_same_v<ScalarType, std::complex<double>>) {
                     auto timer_guard = transpose_cuda_guard();
                     line_copy_interface(line_number, source_lines, destination_lines, line_size_value, parity);
                  } else {
                     for (Size line = 0; line < line_number; line++) {
                        const ScalarType* __restrict source = source_lines[line];
                        ScalarType* __restrict destination = destination_lines[line];
                        for (Size i = 0; i < line_size_value; i++) {
                           if constexpr (parity) {
                              destination[i] = -source[i];
                           } else {
                              destination[i] = source[i];
                           }
                        }
                     }
                  }
               }
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
         const Size total_size,
         const Rank rank) {
      auto leadings_source_by_destination = pmr::vector<Size>();
      leadings_source_by_destination.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         auto j = plan_destination_to_source[i];
         leadings_source_by_destination.push_back(leadings_source[j]);
      }

      const auto& origin_dimensions = dimensions_destination;
      const auto& origin_leadings_source = leadings_source_by_destination;
      const auto& origin_leadings_destination = leadings_destination;

      auto result_dimensions = pmr::vector<Size>();
      auto result_leadings_source = pmr::vector<Size>();
      auto result_leadings_destination = pmr::vector<Size>();
      Rank result_rank = 0;
      Rank current_rank = 0;
      // If all dimension is 1, program will not run into this function.
      while (origin_dimensions[current_rank] == 1) {
         current_rank++;
      }
      while (current_rank < rank) {
         result_rank++;
         Size this_dimension = origin_dimensions[current_rank];
         Size this_leadings_destination = origin_leadings_destination[current_rank];
         Size this_leadings_source = origin_leadings_source[current_rank];
         current_rank++;
         while (current_rank < rank) {
            if (origin_dimensions[current_rank] == 1) {
               current_rank++;
            } else if (
                  (this_leadings_destination == origin_leadings_destination[current_rank] * origin_dimensions[current_rank]) &&
                  (this_leadings_source == origin_leadings_source[current_rank] * origin_dimensions[current_rank])) {
               this_dimension *= origin_dimensions[current_rank];
               this_leadings_destination = origin_leadings_destination[current_rank];
               this_leadings_source = origin_leadings_source[current_rank];
               current_rank++;
            } else {
               break;
            }
         }
         result_dimensions.push_back(this_dimension);
         result_leadings_destination.push_back(this_leadings_destination);
         result_leadings_source.push_back(this_leadings_source);
      }

      if (result_leadings_source.back() == 1 && result_leadings_destination.back() == 1) {
         Size line_size = result_dimensions.back();
         const auto const_line_size_variant = to_const_integral<Size, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(line_size);
         std::visit(
               [&](const auto& const_line_size) {
                  tensor_transpose_kernel<ScalarType, parity, true>(
                        data_source,
                        data_destination,
                        result_dimensions.data(),
                        result_leadings_source.data(),
                        result_leadings_destination.data(),
                        total_size,
                        result_rank,
                        const_line_size);
               },
               const_line_size_variant);
      } else {
         tensor_transpose_kernel<ScalarType, parity>(
               data_source,
               data_destination,
               result_dimensions.data(),
               result_leadings_source.data(),
               result_leadings_destination.data(),
               total_size,
               result_rank);
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
               total_size,
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
               total_size,
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
      cudaDeviceSynchronize();
   }
} // namespace TAT
#endif
