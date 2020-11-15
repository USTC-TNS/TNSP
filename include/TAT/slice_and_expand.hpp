/**
 * \file slice_and_expand.hpp
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
#ifndef TAT_SLICE_AND_EXPAND_HPP
#define TAT_SLICE_AND_EXPAND_HPP

#include "tensor.hpp"
namespace TAT {
   // TODO expand的名字还不是很好听
   // TODO 这些都可以优化
   // TODO 复杂的slice -> 同时slice and expand
   template<typename ScalarType, typename Symmetry>
   Tensor<ScalarType, Symmetry>
   Tensor<ScalarType, Symmetry>::expand(const std::map<Name, EdgeInfoWithArrowForExpand>& configure, const Name& old_name) const {
      auto guard = expand_guard();
      // using EdgeInfoWithArrowForExpand = std::conditional_t<
      //            std::is_same_v<Symmetry, NoSymmetry>,
      //            std::tuple<Size, Size>,
      //            std::conditional_t<is_fermi_symmetry_v<Symmetry>, std::tuple<Arrow, Symmetry, Size, Size>, std::tuple<Symmetry, Size, Size>>>;
      constexpr bool is_no_symmetry = std::is_same_v<Symmetry, NoSymmetry>;
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
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
            new_edges.push_back({arrow, {{symmetry, dimension}}});
         } else {
            const auto& [symmetry, index, dimension] = information;
            total_offset *= dimension;
            total_offset += index;
            new_edges.push_back({{{symmetry, dimension}}});
         }
      }
      if (old_name != internal_name::Null) {
         new_names.push_back(old_name);
         // 调整使得可以缩并
         auto& old_edge = core->edges[name_to_index.at(old_name)];
         if (old_edge.map.size() != 1 || old_edge.map.begin()->second != 1) {
            TAT_error("Cannot Expand a Edge which dimension is not one");
         }
         if constexpr (is_no_symmetry) {
            new_edges.push_back({{{Symmetry(), 1}}});
         } else {
            if constexpr (is_fermi) {
               new_edges.push_back({!old_edge.arrow, {{-total_symmetry, 1}}});
            } else {
               new_edges.push_back({{{-total_symmetry, 1}}});
            }
            if (old_edge.map.begin()->first != total_symmetry) {
               TAT_error("Cannot Expand to such Edges whose total Symmetry is not Compatible with origin Edge");
            }
         }
      } else {
         if constexpr (!is_no_symmetry) {
            if (total_symmetry != Symmetry()) {
               TAT_error("Cannot Expand to such Edges whose total Symmetry is not zero");
            }
         }
      }
      auto helper = Tensor<ScalarType, Symmetry>(std::move(new_names), std::move(new_edges));
      helper.zero();
      helper.core->blocks.begin()->second[total_offset] = 1;
      return contract_all_edge(helper);
   }

   template<typename ScalarType, typename Symmetry>
   Tensor<ScalarType, Symmetry>
   Tensor<ScalarType, Symmetry>::slice(const std::map<Name, EdgeInfoForGetItem>& configure, const Name& new_name, Arrow arrow) const {
      auto guard = slice_guard();
      constexpr bool is_no_symmetry = std::is_same_v<Symmetry, NoSymmetry>;
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
      auto new_names = std::vector<Name>();
      auto new_edges = std::vector<Edge<Symmetry>>();
      auto reserve_size = configure.size() + 1;
      new_names.reserve(reserve_size);
      new_edges.reserve(reserve_size);
      auto total_symmetry = Symmetry();
      Size total_offset = 0;
      for (const auto& name : names) {
         if (auto found_position = configure.find(name); found_position != configure.end()) {
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
            const auto& this_edge = core->edges[name_to_index.at(name)];
            Size dimension = this_edge.map.at(symmetry);
            total_offset *= dimension;
            total_offset += index;
            new_names.push_back(name);
            if constexpr (is_fermi) {
               new_edges.push_back({!this_edge.arrow, {{-symmetry, dimension}}});
            } else {
               new_edges.push_back({{{-symmetry, dimension}}});
            }
         }
      }
      if (new_name != internal_name::Null) {
         new_names.push_back(new_name);
         if constexpr (is_fermi) {
            new_edges.push_back({arrow, {{total_symmetry, 1}}});
         } else {
            new_edges.push_back({{{total_symmetry, 1}}});
         }
      } else {
         if constexpr (!is_no_symmetry) {
            if (total_symmetry != Symmetry()) {
               TAT_error("Need to Create a New Edge but Name not set in Slice");
            }
         }
      }
      auto helper = Tensor<ScalarType, Symmetry>(std::move(new_names), std::move(new_edges));
      helper.zero();
      helper.core->blocks.begin()->second[total_offset] = 1;
      return contract_all_edge(helper);
   }
} // namespace TAT
#endif
