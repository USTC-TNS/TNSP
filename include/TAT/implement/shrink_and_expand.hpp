/**
 * \file shrink_and_expand.hpp
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
#ifndef TAT_SHRINK_AND_EXPAND_HPP
#define TAT_SHRINK_AND_EXPAND_HPP

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   inline timer expand_guard("expand");

   // TODO these can be optimize, avoid to use contract
   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>
   Tensor<ScalarType, Symmetry, Name>::expand(const std::unordered_map<Name, EdgePointExpand>& configure, const Name& old_name) const {
      auto pmr_guard = scope_resource(default_buffer_size);
      auto timer_guard = expand_guard();
      // using EdgePointExpand = std::conditional_t<
      //       Symmetry::length == 0,
      //       std::tuple<Size, Size>,
      //       std::conditional_t<Symmetry::is_fermi_symmetry, std::tuple<Arrow, Symmetry, Size, Size>, std::tuple<Symmetry, Size, Size>>>;
      constexpr bool is_no_symmetry = Symmetry::length == 0;
      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
      if constexpr (is_fermi) {
         detail::warning("expand edge of fermi tensor is dangerous, please contract helper tensor manually");
      }

      // generator helper tensor names and edges
      // and an offset, the helper tensor have only one block
      auto new_names = std::vector<Name>();
      auto new_edges = std::vector<Edge<Symmetry>>();
      auto reserve_size = configure.size() + 1;
      new_names.reserve(reserve_size);
      new_edges.reserve(reserve_size);
      auto total_symmetry = Symmetry();
      Size total_offset = 0;
      for (const auto& [name, information] : configure) {
         new_names.push_back(name);
         if constexpr (is_no_symmetry) {
            const auto& [index, dimension] = information;
            total_offset *= dimension;
            total_offset += index;
            new_edges.push_back({{{Symmetry(), dimension}}});
         } else if constexpr (is_fermi) {
            const auto& [arrow, symmetry, index, dimension] = information;
            total_offset *= dimension;
            total_offset += index;
            new_edges.push_back({{{symmetry, dimension}}, arrow});
         } else {
            const auto& [symmetry, index, dimension] = information;
            total_offset *= dimension;
            total_offset += index;
            new_edges.push_back({{{symmetry, dimension}}});
         }
      }
      auto contract_names = std::unordered_set<std::pair<Name, Name>>();
      if (old_name != InternalName<Name>::No_Old_Name) {
         contract_names.insert({old_name, InternalName<Name>::No_Old_Name});
         new_names.push_back(InternalName<Name>::No_Old_Name);

         // add the last edge
         const auto& old_edge = edges(old_name);
         if constexpr (debug_mode) {
            if (old_edge.segment.size() != 1 || old_edge.segment.front().second != 1) {
               detail::error("Cannot Expand a Edge which dimension is not one");
            }
         }
         if constexpr (is_no_symmetry) {
            new_edges.push_back({{{Symmetry(), 1}}});
         } else {
            if constexpr (is_fermi) {
               new_edges.push_back({{{-total_symmetry, 1}}, !old_edge.arrow});
            } else {
               new_edges.push_back({{{-total_symmetry, 1}}});
            }
            if constexpr (debug_mode) {
               if (old_edge.segment.front().first != total_symmetry) {
                  detail::error("Cannot Expand to such Edges whose total Symmetry is not Compatible with origin Edge");
               }
            }
         }
      } else {
         if constexpr (debug_mode) {
            if constexpr (!is_no_symmetry) {
               if (total_symmetry != Symmetry()) {
                  detail::error("Cannot Expand to such Edges whose total Symmetry is not zero");
               }
            }
         }
      }
      auto helper = Tensor<ScalarType, Symmetry, Name>(std::move(new_names), std::move(new_edges));
      helper.zero();
      helper.storage()[total_offset] = 1;
      return contract(helper, contract_names);
   }

   inline timer shrink_guard("shrink");

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name>
   Tensor<ScalarType, Symmetry, Name>::shrink(const std::unordered_map<Name, EdgePointShrink>& configure, const Name& new_name, Arrow arrow) const {
      auto pmr_guard = scope_resource(default_buffer_size);
      auto timer_guard = shrink_guard();
      constexpr bool is_no_symmetry = Symmetry::length == 0;
      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
      if constexpr (is_fermi) {
         detail::warning("shrink edge of fermi tensor is dangerous, please contract helper tensor manually");
      }

      auto new_names = std::vector<Name>();
      auto new_edges = std::vector<Edge<Symmetry>>();
      auto reserve_size = configure.size() + 1;
      new_names.reserve(reserve_size);
      new_edges.reserve(reserve_size);
      auto total_symmetry = Symmetry();
      Size total_offset = 0;
      auto contract_names = std::unordered_set<std::pair<Name, Name>>();
      for (const auto& name : names) {
         if (auto found_position = configure.find(name); found_position != configure.end()) {
            // shrinking
            const auto& position = found_position->second;
            Symmetry symmetry;
            Size index;
            if constexpr (is_no_symmetry) {
               index = position;
            } else {
               symmetry = std::get<0>(position);
               index = std::get<1>(position);
               total_symmetry += symmetry;
            }
            const auto& this_edge = edges(name);
            Size dimension = this_edge.get_dimension_from_symmetry(symmetry);
            total_offset *= dimension;
            total_offset += index;
            new_names.push_back(name);
            contract_names.insert({name, name});
            if constexpr (is_fermi) {
               new_edges.push_back({{{-symmetry, dimension}}, !this_edge.arrow});
            } else {
               new_edges.push_back({{{-symmetry, dimension}}});
            }
         }
      }
      if (new_name != InternalName<Name>::No_New_Name) {
         new_names.push_back(new_name);
         if constexpr (is_fermi) {
            new_edges.push_back({{{total_symmetry, 1}}, arrow});
         } else {
            new_edges.push_back({{{total_symmetry, 1}}});
         }
      } else {
         if constexpr (debug_mode) {
            if constexpr (!is_no_symmetry) {
               if (total_symmetry != Symmetry()) {
                  detail::error("Need to Create a New Edge but Name not set in Slice");
               }
            }
         }
      }
      auto helper = Tensor<ScalarType, Symmetry, Name>(std::move(new_names), std::move(new_edges));
      helper.zero();
      helper.storage()[total_offset] = 1;
      return contract(helper, contract_names);
   }
} // namespace TAT
#endif
