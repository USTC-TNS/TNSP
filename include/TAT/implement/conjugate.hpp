/**
 * \file conjugate.hpp
 *
 * Copyright (C) 2020-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_CONJUGATE_HPP
#define TAT_CONJUGATE_HPP

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   inline timer conjugate_guard("conjugate");

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>
   Tensor<ScalarType, Symmetry, Name>::conjugate(bool default_is_physics_edge, const std::unordered_set<Name>& exclude_names) const {
      auto timer_guard = conjugate_guard();
      auto pmr_guard = scope_resource(default_buffer_size);
      if constexpr (Symmetry::length == 0) {
         if constexpr (is_real<ScalarType>) {
            return *this;
         } else if constexpr (is_complex<ScalarType>) {
            return map([](const auto& x) {
               return std::conj(x);
            });
         }
      }

      auto result_edges = std::vector<Edge<Symmetry>>();
      result_edges.reserve(rank());
      for (const auto& edge : edges()) {
         result_edges.push_back(edge.conjugated());
      }
      auto result = Tensor<ScalarType, Symmetry, Name>(names(), std::move(result_edges));

      // sign of block is full transpose ^ ^(parity of symmetry && is physics edge with arrow true)
      auto signs = mdspan<std::pair<Size, bool>, pmr::vector<Size>>(nullptr, {blocks().dimensions().begin(), blocks().dimensions().end()});
      auto signs_pool = pmr::vector<std::pair<Size, bool>>(signs.size()); // sum of parity and total parity
      signs.set_data(signs_pool.data());
      if constexpr (Symmetry::is_fermi_symmetry) {
         for (auto i = 0; i < rank(); i++) {
            const bool is_physics_edge = default_is_physics_edge ^ (exclude_names.find(names(i)) != exclude_names.end());
            // default is physics ^ it is not default = it is physics edge
            const bool is_physics_edge_with_arrow_true = edges(i).arrow() && is_physics_edge;
            Size self_size = signs.dimensions(i);
            Size in_size = signs.leadings(i);
            Size out_size = signs.size() == 0 ? 0 : signs.size() / (self_size * in_size);
            for (Size x = 0; x < out_size; x++) {
               auto offset_for_x = x;
               for (Size y = 0; y < self_size; y++) {
                  Symmetry symmetry = edges(i).segments(y).first;
                  if (!symmetry.parity()) {
                     continue;
                  }
                  auto offset_for_y = offset_for_x * self_size + y;
                  for (Size z = 0; z < in_size; z++) {
                     auto offset_for_z = offset_for_y * in_size + z;
                     // to get full transpose sign, calculate (it & 2) != 0 later
                     signs_pool[offset_for_z].first += 1;
                     // physics edge apply sign if arrow is true
                     signs_pool[offset_for_z].second ^= is_physics_edge_with_arrow_true;
                  }
               }
            }
         }
      }

      for (auto i = 0; i < signs.size(); i++) {
         if (!blocks().data()[i].has_value()) {
            continue;
         }
         const auto& block_source = blocks().data()[i].value();
         auto& block_destination = result.blocks().data()[i].value();

         const ScalarType* __restrict source = block_source.data();
         ScalarType* __restrict destination = block_destination.data();
         const Size total_size = block_source.size();

         bool parity = false;
         if constexpr (Symmetry::is_fermi_symmetry) {
            auto [sum_parity, total_parity] = signs_pool[i];
            parity = total_parity ^ ((sum_parity & 2) != 0);
         }

         if constexpr (is_complex<ScalarType>) {
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
} // namespace TAT
#endif
