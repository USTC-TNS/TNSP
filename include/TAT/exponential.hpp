/**
 * \file exponential.hpp
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
#ifndef TAT_EXPONENTIAL_HPP
#define TAT_EXPONENTIAL_HPP

#include "tensor.hpp"
#include "timer.hpp"

namespace TAT {
   template<typename T>
   std::uint64_t max_of_zero_and_floor_of_log_2(T input) {
      std::uint64_t result = 0;
      std::uint64_t value = 2;
      while (value <= input) {
         result += 1;
         value *= 2;
      }
      return result;
   }

   template<typename ScalarType>
   void matrix_exponential(Size n, ScalarType* A, ScalarType* F, int step) {
      auto j = max_of_zero_and_floor_of_log_2(max_of_abs(A, n));
      for (Size i = 0; i < n * n; i++) {
         A[i] /= 1 << j;
         // TODO.. put it into log
      }
      // TODO... 需要把exp单独拿出来放在一个文件里
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::exponential(const std::set<std::tuple<Name, Name>>& pairs, int step) const {
      auto guard = exponential_guard();
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         // TODO
      }
      real_base_t<ScalarType> norm_max = norm<-1>();
      auto temporary_tensor_rank = 0;
      real_base_t<ScalarType> temporary_tensor_parameter = 1;
      while (temporary_tensor_parameter * norm_max > 1) {
         temporary_tensor_rank += 1;
         temporary_tensor_parameter *= 1. / 2;
      }
      auto temporary_tensor = *this * temporary_tensor_parameter;

      auto result = identity(pairs);

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
