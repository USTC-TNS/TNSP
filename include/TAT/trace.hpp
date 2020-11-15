/**
 * \file trace.hpp
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
#ifndef TAT_TRACE_HPP
#define TAT_TRACE_HPP

#include "tensor.hpp"

namespace TAT {
   template<typename ScalarType, typename Symmetry>
   Tensor<ScalarType, Symmetry> Tensor<ScalarType, Symmetry>::trace(const std::set<std::tuple<Name, Name>>& trace_names) const {
      auto guard = trace_guard();
      // TODO to implement
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
      // 对于fermi的情况, 应是一进一出才合法
      auto traced_names = std::set<Name>();
      // TODO trace order maybe optimized
      auto trace_1_names = std::vector<Name>();
      auto trace_2_names = std::vector<Name>();
      for (const auto& i : trace_names) {
         // 对于费米子进行转向
         if constexpr (is_fermi) {
            if (core->edges[name_to_index.at(std::get<0>(i))].arrow) {
               trace_1_names.push_back(std::get<0>(i));
               trace_2_names.push_back(std::get<1>(i));
            } else {
               trace_1_names.push_back(std::get<1>(i));
               trace_2_names.push_back(std::get<0>(i));
            }
         } else {
            trace_1_names.push_back(std::get<0>(i));
            trace_2_names.push_back(std::get<1>(i));
         }
         // 统计traced names
         traced_names.insert(std::get<0>(i));
         traced_names.insert(std::get<1>(i));
      }
      // 寻找自由脚
      auto result_names = std::vector<Name>();
      auto reverse_names = std::set<Name>();
      auto split_plan = std::vector<std::tuple<Name, BoseEdge<Symmetry>>>();
      for (const auto& name : names) {
         if (auto found = traced_names.find(name); found == traced_names.end()) {
            const auto& this_edge = core->edges[name_to_index.at(name)];
            result_names.push_back(name);
            split_plan.push_back({name, {this_edge.map}});
            if constexpr (is_fermi) {
               if (this_edge.arrow) {
                  reverse_names.insert(name);
               }
            }
         }
      }
      auto merged_tensor = edge_operator(
            {},
            {},
            reverse_names,
            {{internal_name::Trace_1, trace_1_names}, {internal_name::Trace_2, trace_2_names}, {internal_name::Trace_3, result_names}},
            {internal_name::Trace_1, internal_name::Trace_2, internal_name::Trace_3},
            false,
            {{{}, {}, {}, {internal_name::Trace_1}}});
      auto traced_tensor = Tensor<ScalarType, Symmetry>({internal_name::Trace_3}, {merged_tensor.core->edges[2]}).zero();
      auto& destination_block = traced_tensor.core->blocks.begin()->second;
      const Size line_size = destination_block.size();

      for (const auto& [symmetry_1, dimension] : merged_tensor.core->edges[0].map) {
         auto symmetry_2 = -symmetry_1;
         auto source_block = merged_tensor.core->blocks.at({symmetry_1, symmetry_2, Symmetry()});
         for (Size i = 0; i < dimension; i++) {
            const ScalarType* __restrict source_data = source_block.data() + (dimension + 1) * i * line_size;
            ScalarType* __restrict destination_data = destination_block.data();
            for (Size j = 0; j < line_size; j++) {
               destination_data[j] += source_data[j];
            }
         }
      }
      auto result = traced_tensor.edge_operator({}, {{internal_name::Trace_3, split_plan}}, reverse_names, {}, result_names);
      return result;
   }
} // namespace TAT
#endif
