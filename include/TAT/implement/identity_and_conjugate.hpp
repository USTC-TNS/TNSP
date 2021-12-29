/**
 * \file identity_and_conjugate.hpp
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
#ifndef TAT_IDENTITY_AND_CONJUGATE_HPP
#define TAT_IDENTITY_AND_CONJUGATE_HPP

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   inline timer conjugate_guard("conjugate");

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::conjugate(bool positive_contract) const {
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

      Rank rank = get_rank();

      auto transpose_flag = pmr::vector<Rank>(rank, 0);
      auto valid_flag = pmr::vector<bool>(1, true);

      auto result_edges = std::vector<Edge<Symmetry>>();
      result_edges.reserve(rank);
      for (const auto& edge : core->edges) {
         result_edges.push_back(edge.conjugated_edge());
      }
      auto result = Tensor<ScalarType, Symmetry, Name>(names, std::move(result_edges));
      for (const auto& [symmetries, block] : core->blocks) {
         auto result_symmetries = pmr::vector<Symmetry>();
         for (const auto& symmetry : symmetries) {
            result_symmetries.push_back(-symmetry);
         }
         const Size total_size = block.size();
         ScalarType* __restrict destination = result.blocks(result_symmetries).data();
         const ScalarType* __restrict source = block.data();
         bool parity = false;
         if constexpr (Symmetry::is_fermi_symmetry) {
            parity = Symmetry::get_split_merge_parity(symmetries, transpose_flag, valid_flag);
            // get full transpose sign
            if (positive_contract) {
               // true/false edge total parity
               for (Rank i = 0; i < rank; i++) {
                  if (edges(i).arrow) {
                     parity ^= symmetries[i].get_parity();
                  }
               }
            }
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

   namespace detail {
      template<bool parity, typename ScalarType, typename = std::enable_if_t<is_scalar<ScalarType>>>
      void set_to_identity(ScalarType* pointer, const pmr::vector<Size>& dimension, const pmr::vector<Size>& leading, Rank rank) {
         auto current_index = pmr::vector<Size>(rank, 0);
         while (true) {
            if constexpr (parity) {
               *pointer = -1;
            } else {
               *pointer = 1;
            }
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
   } // namespace detail

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>& Tensor<ScalarType, Symmetry, Name>::identity(const std::set<std::pair<Name, Name>>& pairs) & {
      // the order of fermi arrow should be (false true) before set to delta
      auto pmr_guard = scope_resource(default_buffer_size);
      auto rank = get_rank();
      auto half_rank = rank / 2;

      auto ordered_pair = pmr::vector<std::pair<Name, Name>>();
      auto ordered_pair_index = pmr::vector<std::pair<Rank, Rank>>();
      ordered_pair.reserve(half_rank);
      ordered_pair_index.reserve(half_rank);

      Rank destination_index = 0;
      no_initialize::pmr::vector<Rank> transpose_plan_source_to_destination;
      if constexpr (Symmetry::is_fermi_symmetry) {
         transpose_plan_source_to_destination.resize(rank);
      }

      auto valid_index = pmr::vector<bool>(rank, true);
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
            auto index_correspond = get_rank_from_name(*name_correspond);
            valid_index[index_correspond] = false;
            ordered_pair.push_back({name_to_find, *name_correspond});
            ordered_pair_index.push_back({i, index_correspond});
            if constexpr (Symmetry::is_fermi_symmetry) {
               if (edges(i).arrow == false) {
                  // i index_corresponding
                  transpose_plan_source_to_destination[i] = destination_index++;
                  transpose_plan_source_to_destination[index_correspond] = destination_index++;
               } else {
                  // index_corresponding i
                  transpose_plan_source_to_destination[index_correspond] = destination_index++;
                  transpose_plan_source_to_destination[i] = destination_index++;
               }
            }
         }
      }

      zero();

      for (auto& [symmetries, block] : core->blocks) {
         bool not_diagonal = false;
         for (const auto& [i0, i1] : ordered_pair_index) {
            not_diagonal = symmetries[i0] + symmetries[i1] != Symmetry();
            if (not_diagonal) {
               break;
            }
         }
         if (not_diagonal) {
            continue;
         }

         auto dimension = pmr::vector<Size>(rank);
         auto leading = pmr::vector<Size>(rank);
         for (Rank i = rank; i-- > 0;) {
            dimension[i] = edges(i).get_dimension_from_symmetry(symmetries[i]);
            if (i == rank - 1) {
               leading[i] = 1;
            } else {
               leading[i] = leading[i + 1] * dimension[i + 1];
            }
         }

         auto pair_dimension = pmr::vector<Size>();
         auto pair_leading = pmr::vector<Size>();
         pair_dimension.reserve(half_rank);
         pair_leading.reserve(half_rank);
         for (Rank i = 0; i < half_rank; i++) {
            pair_dimension.push_back(dimension[std::get<0>(ordered_pair_index[i])]);
            pair_leading.push_back(leading[std::get<0>(ordered_pair_index[i])] + leading[std::get<1>(ordered_pair_index[i])]);
            // ordered_pair_index order is from leading large to leading small so pair_leading is descreasing
         }
         bool parity = false;
         if constexpr (Symmetry::is_fermi_symmetry) {
            parity = Symmetry::get_transpose_parity(symmetries, transpose_plan_source_to_destination);
         }
         if (parity) {
            detail::set_to_identity<true>(block.data(), pair_dimension, pair_leading, half_rank);
         } else {
            detail::set_to_identity<false>(block.data(), pair_dimension, pair_leading, half_rank);
         }
      }

      return *this;
   }
} // namespace TAT
#endif
