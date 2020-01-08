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
         const Size M,
         const Size N,
         const ScalarType* const __restrict src,
         const Size leading_src,
         ScalarType* const __restrict dst,
         const Size leading_dst,
         const Size line_size) {
      for (Size i = 0; i < M; i++) {
         for (Size j = 0; j < N; j++) {
            auto line_dst = dst + (j * leading_dst + i) * line_size;
            auto line_src = src + (i * leading_src + j) * line_size;
            for (Size k = 0; k < line_size; k++) {
               if constexpr (parity) {
                  line_dst[k] = -line_src[k];
               } else {
                  line_dst[k] = line_src[k];
               }
            }
            // TODO: 向量化
         }
      }
   }

   template<class ScalarType, bool parity, Size cache_size, Size... other>
   void matrix_transpose(
         Size M,
         Size N,
         const ScalarType* const __restrict src,
         const Size leading_src,
         ScalarType* const __restrict dst,
         const Size leading_dst,
         const Size line_size) {
      Size block_size = 1;
      // TODO: 是否应该乘以二做冗余？
      while (block_size * block_size * line_size * sizeof(ScalarType) * 2 < cache_size) {
         block_size <<= 1;
      }
      block_size >>= 1;
      for (Size i = 0; i < M; i += block_size) {
         for (Size j = 0; j < N; j += block_size) {
            auto block_dst = dst + (j * leading_dst + i) * line_size;
            auto block_src = src + (i * leading_src + j) * line_size;
            auto m = M - i;
            auto n = N - j;
            m = (block_size <= m) ? block_size : m;
            n = (block_size <= n) ? block_size : n;

            if constexpr (sizeof...(other) == 0) {
               matrix_transpose_kernel<ScalarType, parity>(
                     m, n, block_src, leading_src, block_dst, leading_dst, line_size);
            } else {
               matrix_transpose<ScalarType, parity, other...>(
                     m, n, block_src, leading_src, block_dst, leading_dst, line_size);
            }
         }
      }
   }

   const Size l1_cache = 32768;
   const Size l2_cache = 262144;
   const Size l3_cache = 9437184;
   // TODO: 如何确定系统cache

   template<class ScalarType, bool parity>
   void block_transpose(
         const ScalarType* const __restrict src,
         ScalarType* const __restrict dst,
         const vector<Rank>& plan_src_to_dst,
         const vector<Rank>& plan_dst_to_src,
         const vector<Size>& dims_src,
         const vector<Size>& dims_dst,
         [[maybe_unused]] const Size& size,
         const Rank& rank,
         const Size& line_size) {
      vector<Size> step_src(rank);
      step_src[rank - 1] = 1;
      for (Rank i = rank - 1; i > 0; i--) {
         step_src[i - 1] = step_src[i] * dims_src[i];
      }
      vector<Size> step_dst(rank);
      step_dst[rank - 1] = 1;
      for (Rank i = rank - 1; i > 0; i--) {
         step_dst[i - 1] = step_dst[i] * dims_dst[i];
      }

      vector<Size> index_list_src(rank, 0);
      vector<Size> index_list_dst(rank, 0);
      Size index_src = 0;
      Size index_dst = 0;

      const Size dim_N = dims_src[rank - 1];
      const Size dim_M = dims_dst[rank - 1];
      const Rank pos_M = plan_dst_to_src[rank - 1];
      const Rank pos_N = plan_src_to_dst[rank - 1];
      const Size leading_M = step_src[pos_M];
      const Size leading_N = step_dst[pos_N];

      while (true) {
         // TODO: l3太大了, 所以只按着l2和l1来划分, 这样合适么
         matrix_transpose<ScalarType, parity, l2_cache, l1_cache>(
               dim_M,
               dim_N,
               src + index_src * line_size,
               leading_M,
               dst + index_dst * line_size,
               leading_N,
               line_size);

         Rank temp_rank_dst = rank - 2;
         Rank temp_rank_src = plan_dst_to_src[temp_rank_dst];
         if (temp_rank_src == rank - 1) {
            if (temp_rank_dst == 0) {
               return;
            }
            temp_rank_dst -= 1;
            temp_rank_src = plan_dst_to_src[temp_rank_dst];
         }

         index_list_src[temp_rank_src] += 1;
         index_src += step_src[temp_rank_src];
         index_list_dst[temp_rank_dst] += 1;
         index_dst += step_dst[temp_rank_dst];

         while (index_list_dst[temp_rank_dst] == dims_dst[temp_rank_dst]) {
            index_list_src[temp_rank_src] = 0;
            index_src -= dims_src[temp_rank_src] * step_src[temp_rank_src];
            index_list_dst[temp_rank_dst] = 0;
            index_dst -= dims_dst[temp_rank_dst] * step_dst[temp_rank_dst];
            if (temp_rank_dst == 0) {
               return;
            }
            temp_rank_dst -= 1;
            temp_rank_src = plan_dst_to_src[temp_rank_dst];
            if (temp_rank_src == rank - 1) {
               if (temp_rank_dst == 0) {
                  return;
               }
               temp_rank_dst -= 1;
               temp_rank_src = plan_dst_to_src[temp_rank_dst];
            }
            index_list_src[temp_rank_src] += 1;
            index_src += step_src[temp_rank_src];
            index_list_dst[temp_rank_dst] += 1;
            index_dst += step_dst[temp_rank_dst];
         }
      }
   }

   inline auto noone_in_transpose(
         const vector<Rank>& plan_src_to_dst,
         const vector<Rank>& plan_dst_to_src,
         const vector<Size>& dims_src,
         const vector<Size>& dims_dst,
         const Rank& rank) {
      vector<bool> isone_src(rank);
      vector<bool> isone_dst(rank);
      for (Rank i = 0; i < rank; i++) {
         isone_src[i] = dims_src[i] == 1;
      }
      for (Rank i = 0; i < rank; i++) {
         isone_dst[i] = dims_dst[i] == 1;
      }
      vector<Rank> accum_src(rank);
      vector<Rank> accum_dst(rank);
      accum_src[0] = isone_src[0];
      for (Rank i = 1; i < rank; i++) {
         accum_src[i] = accum_src[i - 1] + Rank(isone_src[i]);
      }
      accum_dst[0] = isone_dst[0];
      for (Rank i = 1; i < rank; i++) {
         accum_dst[i] = accum_dst[i - 1] + Rank(isone_dst[i]);
      }

      vector<Rank> noone_plan_src_to_dst;
      vector<Rank> noone_plan_dst_to_src;
      for (Rank i = 0; i < rank; i++) {
         if (!isone_src[i]) {
            noone_plan_src_to_dst.push_back(plan_src_to_dst[i] - accum_dst[plan_src_to_dst[i]]);
         }
      }
      for (Rank i = 0; i < rank; i++) {
         if (!isone_dst[i]) {
            noone_plan_dst_to_src.push_back(plan_dst_to_src[i] - accum_src[plan_dst_to_src[i]]);
         }
      }
      auto noone_rank = Rank(noone_plan_dst_to_src.size());

      vector<Size> noone_dims_src;
      vector<Size> noone_dims_dst;
      for (Rank i = 0; i < rank; i++) {
         if (dims_src[i] != 1) {
            noone_dims_src.push_back(dims_src[i]);
         }
      }
      for (Rank i = 0; i < rank; i++) {
         if (dims_dst[i] != 1) {
            noone_dims_dst.push_back(dims_dst[i]);
         }
      }
      return std::make_tuple(
            noone_plan_src_to_dst,
            noone_plan_dst_to_src,
            noone_dims_src,
            noone_dims_dst,
            noone_rank);
   }

   inline auto merge_in_transpose(
         const vector<Rank>& plan_src_to_dst,
         const vector<Rank>& plan_dst_to_src,
         const vector<Size>& dims_src,
         const vector<Size>& dims_dst,
         const Rank& rank) {
      vector<bool> merged_src_to_dst(rank);
      vector<bool> merged_dst_to_src(rank);
      merged_src_to_dst[0] = false;
      for (Rank i = 1; i < rank; i++) {
         if (plan_src_to_dst[i] == plan_src_to_dst[i - 1] + 1) {
            merged_src_to_dst[i] = true;
         } else {
            merged_src_to_dst[i] = false;
         }
      }
      merged_dst_to_src[0] = false;
      for (Rank i = 1; i < rank; i++) {
         if (plan_dst_to_src[i] == plan_dst_to_src[i - 1] + 1) {
            merged_dst_to_src[i] = true;
         } else {
            merged_dst_to_src[i] = false;
         }
      }
      vector<Rank> accum_src_to_dst(rank);
      vector<Rank> accum_dst_to_src(rank);
      accum_src_to_dst[0] = 0;
      for (Rank i = 1; i < rank; i++) {
         accum_src_to_dst[i] = accum_src_to_dst[i - 1] + Rank(merged_src_to_dst[i]);
      }
      accum_dst_to_src[0] = 0;
      for (Rank i = 1; i < rank; i++) {
         accum_dst_to_src[i] = accum_dst_to_src[i - 1] + Rank(merged_dst_to_src[i]);
      }
      vector<Rank> merged_plan_src_to_dst;
      vector<Rank> merged_plan_dst_to_src;
      for (Rank i = 0; i < rank; i++) {
         if (!merged_src_to_dst[i]) {
            merged_plan_src_to_dst.push_back(
                  plan_src_to_dst[i] - accum_dst_to_src[plan_src_to_dst[i]]);
         }
      }
      for (Rank i = 0; i < rank; i++) {
         if (!merged_dst_to_src[i]) {
            merged_plan_dst_to_src.push_back(
                  plan_dst_to_src[i] - accum_src_to_dst[plan_dst_to_src[i]]);
         }
      }
      auto merged_rank = Rank(merged_plan_src_to_dst.size());
      vector<Size> merged_dims_src(merged_rank);
      vector<Size> merged_dims_dst(merged_rank);
      Rank tmp_src_index = rank;
      for (Rank i = merged_rank; i-- > 0;) {
         merged_dims_src[i] = dims_src[--tmp_src_index];
         while (merged_src_to_dst[tmp_src_index]) {
            merged_dims_src[i] *= dims_src[--tmp_src_index];
         }
      }
      Rank tmp_dst_index = rank;
      for (Rank i = merged_rank; i-- > 0;) {
         merged_dims_dst[i] = dims_dst[--tmp_dst_index];
         while (merged_dst_to_src[tmp_dst_index]) {
            merged_dims_dst[i] *= dims_dst[--tmp_dst_index];
         }
      }
      return std::make_tuple(
            merged_plan_src_to_dst,
            merged_plan_dst_to_src,
            merged_dims_src,
            merged_dims_dst,
            merged_rank);
   }

   template<class ScalarType>
   void do_transpose(
         const vector<Rank>& plan_src_to_dst,
         const vector<Rank>& plan_dst_to_src,
         const ScalarType* src_data,
         ScalarType* dst_data,
         const vector<Size>& dims_src,
         const vector<Size>& dims_dst,
         Size block_size,
         Rank rank,
         bool parity) {
      if (block_size == 1) {
         if (parity) {
            *dst_data = -*src_data;
         } else {
            *dst_data = *src_data;
         }
         return;
      }

      auto [noone_plan_src_to_dst,
            noone_plan_dst_to_src,
            noone_dims_src,
            noone_dims_dst,
            noone_rank] =
            noone_in_transpose(plan_src_to_dst, plan_dst_to_src, dims_src, dims_dst, rank);

      auto [noone_merged_plan_src_to_dst,
            noone_merged_plan_dst_to_src,
            noone_merged_dims_src,
            noone_merged_dims_dst,
            noone_merged_rank] =
            merge_in_transpose(
                  noone_plan_src_to_dst,
                  noone_plan_dst_to_src,
                  noone_dims_src,
                  noone_dims_dst,
                  noone_rank);

      if (noone_merged_rank == 1) {
         if (parity) {
            for (Size k = 0; k < block_size; k++) {
               dst_data[k] = -src_data[k];
            }
         } else {
            for (Size k = 0; k < block_size; k++) {
               dst_data[k] = src_data[k];
            }
         }
      } else {
         Rank effective_rank = noone_merged_rank;
         Size line_size = 1;
         if (noone_merged_plan_src_to_dst[noone_merged_rank - 1] == noone_merged_rank - 1) {
            effective_rank--;
            line_size *= noone_merged_dims_dst[noone_merged_rank - 1];
         }
         // TODO: 需要考虑极端细致的情况
         if (parity) {
            block_transpose<ScalarType, true>(
                  src_data,
                  dst_data,
                  noone_merged_plan_src_to_dst,
                  noone_merged_plan_dst_to_src,
                  noone_merged_dims_src,
                  noone_merged_dims_dst,
                  block_size,
                  effective_rank,
                  line_size);
         } else {
            block_transpose<ScalarType, false>(
                  src_data,
                  dst_data,
                  noone_merged_plan_src_to_dst,
                  noone_merged_plan_dst_to_src,
                  noone_merged_dims_src,
                  noone_merged_dims_dst,
                  block_size,
                  effective_rank,
                  line_size);
         }
      }
   }
} // namespace TAT
#endif
