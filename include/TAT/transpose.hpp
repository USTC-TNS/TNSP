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

#ifdef TAT_USE_MKL_TRANSPOSE
extern "C" {
void mkl_somatcopy_(const char*, const char*, const int*, const int*, const float*, const float*, const int*, float*, const int*);
void mkl_domatcopy_(const char*, const char*, const int*, const int*, const double*, const double*, const int*, double*, const int*);
void mkl_comatcopy_(
      const char*,
      const char*,
      const int*,
      const int*,
      const std::complex<float>*,
      const std::complex<float>*,
      const int*,
      std::complex<float>*,
      const int*);
void mkl_zomatcopy_(
      const char*,
      const char*,
      const int*,
      const int*,
      const std::complex<double>*,
      const std::complex<double>*,
      const int*,
      const std::complex<double>*,
      const int*);
}
#endif

// TAT_USE_MKL_TRANSPOSE
namespace TAT {
   template<typename ScalarType>
   void mkl_transpose(
         int dimension_of_M,
         int dimension_of_N,
         const ScalarType* data_source,
         ScalarType* data_destination,
         int leading_source,
         int leading_destination,
         ScalarType alpha);

#ifdef TAT_USE_MKL_TRANSPOSE
   template<>
   void mkl_transpose<float>(
         const int dimension_of_M,
         const int dimension_of_N,
         const float* const data_source,
         float* const data_destination,
         const int leading_source,
         const int leading_destination,
         const float alpha) {
      mkl_somatcopy_("R", "T", &dimension_of_M, &dimension_of_N, &alpha, data_source, &leading_source, data_destination, &leading_destination);
   }
   template<>
   void mkl_transpose<double>(
         const int dimension_of_M,
         const int dimension_of_N,
         const double* const data_source,
         double* const data_destination,
         const int leading_source,
         const int leading_destination,
         const double alpha) {
      mkl_domatcopy_("R", "T", &dimension_of_M, &dimension_of_N, &alpha, data_source, &leading_source, data_destination, &leading_destination);
   }
   template<>
   void mkl_transpose<std::complex<float>>(
         const int dimension_of_M,
         const int dimension_of_N,
         const std::complex<float>* const data_source,
         std::complex<float>* const data_destination,
         const int leading_source,
         const int leading_destination,
         const std::complex<float> alpha) {
      mkl_comatcopy_("R", "T", &dimension_of_M, &dimension_of_N, &alpha, data_source, &leading_source, data_destination, &leading_destination);
   }
   template<>
   void mkl_transpose<std::complex<double>>(
         const int dimension_of_M,
         const int dimension_of_N,
         const std::complex<double>* const data_source,
         std::complex<double>* const data_destination,
         const int leading_source,
         const int leading_destination,
         const std::complex<double> alpha) {
      mkl_zomatcopy_("R", "T", &dimension_of_M, &dimension_of_N, &alpha, data_source, &leading_source, data_destination, &leading_destination);
   }
#endif

   // 这个是最简单的张量转置中实际搬运数据的部分，numpy也是这么写的，区别在于dimension和两个leading的顺序是可以一同交换的
   // numpy保证destination的leading是降序的， simple_transpose就是这么调用tensor_transpose_kernel的
   // 另外一个正在写的inturn_transpose是src dst轮流来, 可能会对cache更加友好, 日后还会根据cache大小split边，这样类似于矩阵转置中的预分块
   template<typename ScalarType, bool parity>
   void tensor_transpose_kernel(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const Size* const __restrict dimension,
         const Rank* const __restrict,
         const Size* const __restrict,
         const Size* const __restrict leading_source,
         const Size* const __restrict leading_destination,
         const Rank rank) {
      auto guard = transpose_kernel_core_guard();

      const ScalarType* current_source = data_source;
      ScalarType* current_destination = data_destination;
      std::vector<Size> index_list(rank, 0);
      while (true) {
         if constexpr (parity) {
            *current_destination = -*current_source;
         } else {
            *current_destination = *current_source;
         }

         Rank active_position = rank - 1;

         index_list[active_position]++;
         current_source += leading_source[active_position];
         current_destination += leading_destination[active_position];

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
   void tensor_transpose_kernel_with_block(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         Size* const __restrict dimension,
         const Rank* const __restrict checked_index,
         Size* const __restrict incomplete_dimension,
         const Size* const __restrict leading_source,
         const Size* const __restrict leading_destination,
         const Rank rank) {
      auto guard = transpose_kernel_core_guard();

      const ScalarType* current_source = data_source;
      ScalarType* current_destination = data_destination;
      std::vector<Size> index_list(rank, 0);
      while (true) {
         if constexpr (parity) {
            *current_destination = -*current_source;
         } else {
            *current_destination = *current_source;
         }

         Rank active_position = rank - 1;

         index_list[active_position]++;
         current_source += leading_source[active_position];
         current_destination += leading_destination[active_position];

         // index_list[active_position] == dimension[active_position]
         // 变成
         // active_checked_position = checked_index[active_position]
         // if active_checked_position == rank
         //    then index_list[active_position] == dimension[active_position]
         //    else if index_list[active_checked_position] == dimension[active_checked_position] - 1
         //         then index_list[active_position] == incomplete_dimension[active_position]
         //         else index_list[active_position] == dimension[active_position]

         while (index_list[active_position] == dimension[active_position]) {
            index_list[active_position] = 0;
            current_source -= dimension[active_position] * leading_source[active_position];
            current_destination -= dimension[active_position] * leading_destination[active_position];
            if (Rank active_checked_position = checked_index[active_position]; active_checked_position != rank) {
               auto temporary = dimension[active_checked_position];
               dimension[active_checked_position] = incomplete_dimension[active_checked_position];
               incomplete_dimension[active_checked_position] = temporary;
            }

            if (active_position == 0) {
               return;
            }
            active_position--;

            index_list[active_position]++;
            current_source += leading_source[active_position];
            current_destination += leading_destination[active_position];
         }

         if (index_list[active_position] == dimension[active_position] - 1) {
            if (Rank active_checked_position = checked_index[active_position]; active_checked_position != rank) {
               auto temporary = dimension[active_checked_position];
               dimension[active_checked_position] = incomplete_dimension[active_checked_position];
               incomplete_dimension[active_checked_position] = temporary;
            }
         }
      }
   }
   // TODO: l3太大了, 所以只按着l2和l1来划分, 这样合适么
   // 注意，现在这段代码被我暂时删掉了
   //
   // TODO 类似矩阵转置的优化方式优化张量转置
   // O...O...O.O..XXX
   //  X..X..X.....OOOO
   //              ^
   // 首先找到一段前后各自连续的部分，然后分块地做转置，分块的方案是cut掉这里的维度
   //
   // TODO
   // 按照cache来split边
   // 再交替iter dim
   // 即可兼容矩阵转置的优化方式

   inline auto simple_configure(
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leadings_source,
         const std::vector<Size>& leadings_destination,
         const Rank rank) {
      auto leadings_source_by_destination = std::vector<Size>();
      leadings_source_by_destination.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         auto j = plan_destination_to_source[i];
         leadings_source_by_destination.push_back(leadings_source[j]);
      }

      return std::make_tuple(dimensions_destination, leadings_source_by_destination, leadings_destination);
   }

   inline auto inturn_configure(
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leadings_source,
         const std::vector<Size>& leadings_destination,
         const Rank rank) {
      auto mask_source = std::vector<bool>(rank, false);
      auto mask_destination = std::vector<bool>(rank, false);
      auto real_dimensions = std::vector<Size>(rank);
      auto real_leadings_source = std::vector<Size>(rank);
      auto real_leadings_destination = std::vector<Size>(rank);

      bool source_exhausted = false;
      bool destination_exhausted = false;
      for (auto current_index = rank, current_index_source = rank, current_index_destination = rank; current_index-- > 0;) {
         if (current_index_destination != rank &&
             (current_index_source == rank || (destination_exhausted || (!source_exhausted && leadings_destination[current_index_destination] >
                                                                                                    leadings_source[current_index_source])))) {
            // add src
            do {
               current_index_source--;
            } while (mask_source[current_index_source]);
            auto response_index_destination = plan_source_to_destination[current_index_source];

            real_dimensions[current_index] = dimensions_source[current_index_source];
            real_leadings_source[current_index] = leadings_source[current_index_source];
            real_leadings_destination[current_index] = leadings_destination[response_index_destination];

            mask_destination[response_index_destination] = true;
            mask_source[current_index_source] = true;

            if (current_index_source == 0) {
               source_exhausted = true;
            }
         } else {
            // add dst
            do {
               current_index_destination--;
            } while (mask_destination[current_index_destination]);
            auto response_index_source = plan_destination_to_source[current_index_destination];

            real_dimensions[current_index] = dimensions_destination[current_index_destination];
            real_leadings_destination[current_index] = leadings_destination[current_index_destination];
            real_leadings_source[current_index] = leadings_source[response_index_source];

            mask_source[response_index_source] = true;
            mask_destination[current_index_destination] = true;

            if (current_index_destination == 0) {
               destination_exhausted = true;
            }
         }
      }

      return std::make_tuple(std::move(real_dimensions), std::move(real_leadings_source), std::move(real_leadings_destination));
   }

   template<typename ScalarType, bool parity>
   void simple_transpose(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leadings_source,
         const std::vector<Size>& leadings_destination,
         const Rank rank) {
      auto [dimension, leading_of_source, leading_of_destination] = simple_configure(
            plan_source_to_destination,
            plan_destination_to_source,
            dimensions_source,
            dimensions_destination,
            leadings_source,
            leadings_destination,
            rank);

      auto checked_index = std::vector<Rank>(rank, rank);
      auto incomplete_dimension = std::vector<Size>(rank, 0);

      tensor_transpose_kernel<ScalarType, parity>(
            data_source,
            data_destination,
            dimension.data(),
            checked_index.data(),
            incomplete_dimension.data(),
            leading_of_source.data(),
            leading_of_destination.data(),
            rank);
   }

   template<typename ScalarType, bool parity>
   void simple_transpose_with_block(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leadings_source,
         const std::vector<Size>& leadings_destination,
         const Rank rank) {
      auto [dimension, leading_of_source, leading_of_destination] = simple_configure(
            plan_source_to_destination,
            plan_destination_to_source,
            dimensions_source,
            dimensions_destination,
            leadings_source,
            leadings_destination,
            rank);

      auto checked_index = std::vector<Rank>(rank, rank);
      auto incomplete_dimension = std::vector<Size>(rank, 0);

      tensor_transpose_kernel_with_block<ScalarType, parity>(
            data_source,
            data_destination,
            dimension.data(),
            checked_index.data(),
            incomplete_dimension.data(),
            leading_of_source.data(),
            leading_of_destination.data(),
            rank);
   }

   template<typename ScalarType, bool parity>
   void inturn_transpose(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leadings_source,
         const std::vector<Size>& leadings_destination,
         const Rank rank) {
      auto [dimension, leading_of_source, leading_of_destination] = inturn_configure(
            plan_source_to_destination,
            plan_destination_to_source,
            dimensions_source,
            dimensions_destination,
            leadings_source,
            leadings_destination,
            rank);

      auto checked_index = std::vector<Rank>(rank, rank);
      auto incomplete_dimension = std::vector<Size>(rank, 0);

      tensor_transpose_kernel<ScalarType, parity>(
            data_source,
            data_destination,
            dimension.data(),
            checked_index.data(),
            incomplete_dimension.data(),
            leading_of_source.data(),
            leading_of_destination.data(),
            rank);
   }

   template<typename ScalarType, bool parity>
   void inturn_transpose_with_block(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leadings_source,
         const std::vector<Size>& leadings_destination,
         const Rank rank) {
      auto [dimension, leading_of_source, leading_of_destination] = inturn_configure(
            plan_source_to_destination,
            plan_destination_to_source,
            dimensions_source,
            dimensions_destination,
            leadings_source,
            leadings_destination,
            rank);

      auto checked_index = std::vector<Rank>(rank, rank);
      auto incomplete_dimension = std::vector<Size>(rank, 0);

      tensor_transpose_kernel_with_block<ScalarType, parity>(
            data_source,
            data_destination,
            dimension.data(),
            checked_index.data(),
            incomplete_dimension.data(),
            leading_of_source.data(),
            leading_of_destination.data(),
            rank);
   }

   /*
   inline auto find_in_leading(const std::vector<Size>& leading, Size size) {
      for (auto i = leading.size(); i-- > 0;) {
         if (leading[i] > size) {
            return Size(i);
         }
      }
      return Size(-1); // -1 means no split needed
   }

   // TODO 其实判断应该是下面的dimension乘上下面的leading而不是自己的leading，只不过稠密的时候他们相等
   // 这个部分也许应该放在noone和merge那一块
   template<typename ScalarType, bool parity>
   void block_transpose(
         const ScalarType* const __restrict data_source,
         ScalarType* const __restrict data_destination,
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leadings_source,
         const std::vector<Size>& leadings_destination,
         const Rank rank) {
      Size block_size = 2;
      while (block_size * block_size * sizeof(ScalarType) < l1_cache) {
         block_size <<= 1u;
      }
      block_size >>= 1u;
      auto source_split_index = find_in_leading(leadings_source, block_size);
      if (source_split_index != -1) {
         auto this_leading = leadings_source[source_split_index];
         auto next_leading = leadings_source[source_split_index + 1];
         auto this_dimension = this_leading / next_leading;
         auto this_dimension_2 = 2;
         while (next_leading * this_dimension_2 < block_size) {
            this_dimension_2 <<= 1u;
         }
         this_dimension_2 >>= 1u;
         auto this_dimension_1 = this_dimension / this_dimension_2;
         if (this_dimension_1 * this_dimension_2 == this_dimension) {
            auto middle_leading = this_dimension_2 * next_leading;
         }
      }
      // 这个plan需要重新分析，很烦
   }
    */

   // 去掉dimension = 1的边
   inline auto prune_for_transpose(
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leadings_source,
         const std::vector<Size>& leadings_destination,
         const Rank& rank) {
      std::vector<bool> is_one_source;
      std::vector<bool> is_one_destination;
      is_one_source.reserve(rank);
      is_one_destination.reserve(rank);
      for (Rank i = 0; i < rank; i++) {
         is_one_source.push_back(dimensions_source[i] == 1);
      }
      for (Rank i = 0; i < rank; i++) {
         is_one_destination.push_back(dimensions_destination[i] == 1);
      }
      std::vector<Rank> accumulated_one_source;
      std::vector<Rank> accumulated_one_destination;
      accumulated_one_source.reserve(rank);
      accumulated_one_destination.reserve(rank);
      accumulated_one_source.push_back(is_one_source.front());
      for (Rank i = 1; i < rank; i++) {
         accumulated_one_source.push_back(accumulated_one_source[i - 1] + Rank(is_one_source[i]));
      }
      accumulated_one_destination.push_back(is_one_destination.front());
      for (Rank i = 1; i < rank; i++) {
         accumulated_one_destination.push_back(accumulated_one_destination[i - 1] + Rank(is_one_destination[i]));
      }

      std::vector<Rank> result_plan_source_to_destination;
      std::vector<Rank> result_plan_destination_to_source;
      result_plan_source_to_destination.reserve(rank); // 会冗余, 无所谓
      result_plan_destination_to_source.reserve(rank);
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
      std::vector<Size> result_leadings_source;
      std::vector<Size> result_leadings_destination;
      result_dimensions_source.reserve(result_rank);
      result_dimensions_destination.reserve(result_rank);
      result_leadings_source.reserve(result_rank);
      result_leadings_destination.reserve(result_rank);
      for (Rank i = 0; i < rank; i++) {
         if (dimensions_source[i] != 1) {
            result_dimensions_source.push_back(dimensions_source[i]);
            result_leadings_source.push_back(leadings_source[i]);
         }
      }
      for (Rank i = 0; i < rank; i++) {
         if (dimensions_destination[i] != 1) {
            result_dimensions_destination.push_back(dimensions_destination[i]);
            result_leadings_destination.push_back(leadings_destination[i]);
         }
      }
      return std::make_tuple(
            std::move(result_plan_source_to_destination),
            std::move(result_plan_destination_to_source),
            std::move(result_dimensions_source),
            std::move(result_dimensions_destination),
            std::move(result_leadings_source),
            std::move(result_leadings_destination),
            result_rank);
   }

   inline auto merging_for_transpose(
         const std::vector<Rank>& plan_source_to_destination,
         const std::vector<Rank>& plan_destination_to_source,
         const std::vector<Size>& dimensions_source,
         const std::vector<Size>& dimensions_destination,
         const std::vector<Size>& leadings_source,
         const std::vector<Size>& leadings_destination,
         const Rank& rank) {
      std::vector<bool> merging_source_to_destination(rank, false);
      std::vector<bool> merging_destination_to_source(rank, false);
      for (Rank i = 1; i < rank; i++) {
         if (const auto j = plan_source_to_destination[i]; i != 0 && j != 0 && j - 1 == plan_source_to_destination[i - 1] &&
                                                           leadings_source[i - 1] == leadings_source[i] * dimensions_source[i] &&
                                                           leadings_destination[j - 1] == leadings_destination[j] * dimensions_destination[j]) {
            merging_source_to_destination[i] = true;
            merging_destination_to_source[plan_source_to_destination[i]] = true;
         }
      }

      std::vector<Rank> accumulated_merging_source_to_destination;
      std::vector<Rank> accumulated_merging_destination_to_source;
      accumulated_merging_source_to_destination.reserve(rank);
      accumulated_merging_destination_to_source.reserve(rank);
      accumulated_merging_source_to_destination.push_back(0);
      for (Rank i = 1; i < rank; i++) {
         accumulated_merging_source_to_destination.push_back(
               accumulated_merging_source_to_destination.back() + Rank(merging_source_to_destination[i]));
      }
      accumulated_merging_destination_to_source.push_back(0);
      for (Rank i = 1; i < rank; i++) {
         accumulated_merging_destination_to_source.push_back(
               accumulated_merging_destination_to_source.back() + Rank(merging_destination_to_source[i]));
      }
      std::vector<Rank> result_plan_source_to_destination;
      std::vector<Rank> result_plan_destination_to_source;
      result_plan_source_to_destination.reserve(rank); // 会冗余, 无所谓
      result_plan_destination_to_source.reserve(rank);
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
      std::vector<Size> result_leadings_source(result_rank);
      std::vector<Size> result_leadings_destination(result_rank);
      for (Rank i = result_rank, j = rank; i-- > 0;) {
         result_leadings_source[i] = leadings_source[--j];
         result_dimensions_source[i] = dimensions_source[j];
         while (merging_source_to_destination[j]) {
            result_dimensions_source[i] *= dimensions_source[--j];
         }
      }
      for (Rank i = result_rank, j = rank; i-- > 0;) {
         result_leadings_destination[i] = leadings_destination[--j];
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
            std::move(result_leadings_source),
            std::move(result_leadings_destination),
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
         const std::vector<Size>& leadings_source,
         const std::vector<Size>& leadings_destination,
         Rank rank,
         Size total_size,
         bool parity) {
      auto guard = transpose_kernel_guard();

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

      auto [prune_plan_source_to_destination,
            prune_plan_destination_to_source,
            prune_dimensions_source,
            prune_dimensions_destination,
            prune_leadings_source,
            prune_leadings_destination,
            prune_rank] =
            prune_for_transpose(
                  plan_source_to_destination,
                  plan_destination_to_source,
                  dimensions_source,
                  dimensions_destination,
                  leadings_source,
                  leadings_destination,
                  rank);

      auto [real_plan_source_to_destination,
            real_plan_destination_to_source,
            real_dimensions_source,
            real_dimensions_destination,
            real_leadings_source,
            real_leadings_destination,
            real_rank] =
            merging_for_transpose(
                  prune_plan_source_to_destination,
                  prune_plan_destination_to_source,
                  prune_dimensions_source,
                  prune_dimensions_destination,
                  prune_leadings_source,
                  prune_leadings_destination,
                  prune_rank);

      // TODO: 需要考虑极端细致的情况
      if (parity) {
         simple_transpose<ScalarType, true>(
               data_source,
               data_destination,
               real_plan_source_to_destination,
               real_plan_destination_to_source,
               real_dimensions_source,
               real_dimensions_destination,
               real_leadings_source,
               real_leadings_destination,
               real_rank);
      } else {
         simple_transpose<ScalarType, false>(
               data_source,
               data_destination,
               real_plan_source_to_destination,
               real_plan_destination_to_source,
               real_dimensions_source,
               real_dimensions_destination,
               real_leadings_source,
               real_leadings_destination,
               real_rank);
      }
   }

   template<typename ScalarType>
   void matrix_transpose(Size m, Size n, const ScalarType* const source, ScalarType* const destination) {
      auto dimension = std::vector<Size>{m, n};
      auto leading_source = std::vector<Size>{n, 1};
      auto leading_destination = std::vector<Size>{1, m};
      tensor_transpose_kernel<ScalarType, false>(
            source, destination, dimension.data(), nullptr, nullptr, leading_source.data(), leading_destination.data(), 2);
   }
} // namespace TAT
#endif
