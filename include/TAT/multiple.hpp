/**
 * \file multiple.hpp
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
#ifndef TAT_MULTIPLE_HPP
#define TAT_MULTIPLE_HPP

#include "const_integral.hpp"
#include "tensor.hpp"
#include "timer.hpp"

namespace TAT {
   template<typename ScalarType, typename ScalarTypeS>
   void multiple_kernel(Size m, Size k, Size n, ScalarType* data_destination, const ScalarType* data_source, const ScalarTypeS* S) {
      const auto const_n_variant = to_const_integral<Size, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(n);
      std::visit(
            [&](const auto& const_n) {
               const auto n = const_n.value();
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
            },
            const_n_variant);
   }

   template<typename ScalarType, typename Symmetry, typename Name, template<typename> class Allocator>
   Tensor<ScalarType, Symmetry, Name, Allocator>
   Tensor<ScalarType, Symmetry, Name, Allocator>::multiple(const SingularType& S, const Name& name, char direction, bool division) const {
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
} // namespace TAT
#endif
