/**
 * \file trace.hpp
 *
 * Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   inline timer trace_guard("trace");

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::trace(const std::set<std::pair<Name, Name>>& trace_names) const {
      auto pmr_guard = scope_resource(default_buffer_size);
      auto timer_guard = trace_guard();

      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;

      auto rank = get_rank();
      auto trace_rank = trace_names.size();
      auto free_rank = rank - 2 * trace_rank;

      // transpose to a_i = b_{jji}, this is the most fast way to trace
      auto traced_names = pmr::set<Name>();
      auto trace_1_names = pmr::vector<Name>();
      auto trace_2_names = pmr::vector<Name>();
      trace_1_names.reserve(trace_rank);
      trace_2_names.reserve(trace_rank);

      // reverse before merge
      auto reverse_names = pmr::set<Name>();
      auto traced_reverse_flag = pmr::set<Name>();

      // delete empty corresponding segment of traced edge
      auto delete_map = pmr::map<Name, pmr::map<Symmetry, Size>>();

      // traced edge
      auto valid_index = pmr::vector<bool>(rank, true);
      for (auto i = rank; i-- > 0;) {
         // if possible, let names order unchanged
         if (valid_index[i]) {
            const auto& name_to_find = names[i];
            const Name* name_correspond = nullptr;
            for (const auto& [name_1, name_2] : trace_names) {
               if (name_1 == name_to_find) {
                  name_correspond = &name_2;
                  break;
               }
               if (name_2 == name_to_find) {
                  name_correspond = &name_1;
                  break;
               }
            }
            if (name_correspond) {
               // found in trace_names
               // trace_1 arrow true, trace_2 arrow false
               // so that (a b c)(b+ a+) = (c)
               if constexpr (is_fermi) {
                  if (edges(i).arrow) {
                     // need reversed
                     traced_reverse_flag.insert(name_to_find);
                     reverse_names.insert(name_to_find);
                     reverse_names.insert(*name_correspond);
                  }
               }
               trace_1_names.push_back(*name_correspond);
               trace_2_names.push_back(name_to_find);
               // trace_1 is in front of trace_2
               // name_correspond is in front of name_to_find

               traced_names.insert(name_to_find);
               traced_names.insert(*name_correspond);
               auto index_correspond = get_rank_from_name(*name_correspond);
               valid_index[index_correspond] = false;

               if constexpr (Symmetry::length != 0) {
                  // order of edge symmetry segment
                  const auto& name_1 = *name_correspond;
                  const auto& name_2 = name_to_find;
                  const auto& edge_1 = edges(index_correspond);
                  const auto& edge_2 = edges(i);
                  // same to contract delete dimension
                  auto delete_unused_dimension = [](const auto& edge_this, const auto& edge_other, const auto& name_this, auto& delete_this) {
                     if constexpr (debug_mode) {
                        if constexpr (is_fermi) {
                           if (edge_this.arrow == edge_other.arrow) {
                              detail::error("Different Fermi Arrow to Trace");
                           }
                        }
                     }
                     auto delete_map = pmr::map<Symmetry, Size>();
                     for (const auto& [symmetry, dimension] : edge_this.segment) {
                        auto found = edge_other.find_by_symmetry(-symmetry);
                        if (found != edge_other.segment.end()) {
                           // found
                           if constexpr (debug_mode) {
                              if (const auto dimension_other = found->second; dimension_other != dimension) {
                                 detail::error("Different Dimension to Trace");
                              }
                           }
                        } else {
                           // not found
                           delete_map[symmetry] = 0;
                           // pass delete map when merging, edge operator will delete the entire segment if it is zero
                        }
                     }
                     if (!delete_map.empty()) {
                        return delete_this.emplace(name_this, std::move(delete_map)).first;
                     } else {
                        return delete_this.end();
                     }
                  };
                  auto delete_map_edge_1_iterator = delete_unused_dimension(edge_1, edge_2, name_1, delete_map);
                  auto delete_map_edge_2_iterator = delete_unused_dimension(edge_2, edge_1, name_2, delete_map);
                  if constexpr (debug_mode) {
                     // check different order
                     auto empty_delete_map = pmr::map<Symmetry, Size>();
                     const auto& delete_map_edge_1 = [&]() -> const auto& {
                        if (delete_map_edge_1_iterator == delete_map.end()) {
                           return empty_delete_map;
                        } else {
                           return delete_map_edge_1_iterator->second;
                        }
                     }
                     ();
                     const auto& delete_map_edge_2 = [&]() -> const auto& {
                        if (delete_map_edge_2_iterator == delete_map.end()) {
                           return empty_delete_map;
                        } else {
                           return delete_map_edge_2_iterator->second;
                        }
                     }
                     ();
                     // it is impossible that one of i, j reach end and another not
                     for (auto [i, j] = std::tuple{edge_1.segment.begin(), edge_2.segment.begin()};
                          i != edge_1.segment.end() && j != edge_2.segment.end();
                          ++i, ++j) {
                        // i/j -> first :: Symmetry
                        while (delete_map_edge_1.find(i->first) != delete_map_edge_1.end()) {
                           ++i;
                        }
                        while (delete_map_edge_2.find(j->first) != delete_map_edge_2.end()) {
                           ++j;
                        }
                        if ((i->first + j->first) != Symmetry()) {
                           detail::error("Different symmetry segment order in trace");
                        }
                     }
                  }
               }
            }
         }
      }

      // free edge
      auto result_names = std::vector<Name>();
      // auto reverse_names = pmr::set<Name>(); // add to the former set
      auto split_plan = pmr::vector<std::tuple<Name, edge_segment_t<Symmetry>>>();
      result_names.reserve(free_rank);
      split_plan.reserve(free_rank);
      for (Rank i = 0; i < rank; i++) {
         const auto& name = names[i];
         if (auto found = traced_names.find(name); found == traced_names.end()) {
            // it is free name
            const auto& this_edge = edges(i);
            result_names.push_back(name);
            split_plan.push_back({name, {this_edge.segment}});
            if constexpr (is_fermi) {
               if (this_edge.arrow) {
                  reverse_names.insert(name);
               }
            }
         }
      }

      auto merged_tensor = edge_operator_implement(
            empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
            reverse_names,
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::Trace_1, std::move(trace_1_names)},
                  {InternalName<Name>::Trace_2, std::move(trace_2_names)},
                  {InternalName<Name>::Trace_3, {result_names.begin(), result_names.end()}}},
            std::vector<Name>{InternalName<Name>::Trace_1, InternalName<Name>::Trace_2, InternalName<Name>::Trace_3},
            false,
            empty_list<Name>(),                          // split
            traced_reverse_flag,                         // reverse
            empty_list<Name>(),                          // reverse
            pmr::set<Name>{InternalName<Name>::Trace_1}, // merge
            delete_map);
      // trace 1 is connected to trace_2, so one of then is applied sign, another is not
      // trace 3 will be reversed/splitted later, nothing changed
      auto traced_tensor = Tensor<ScalarType, Symmetry, Name>({InternalName<Name>::Trace_3}, {merged_tensor.edges(2)}).zero();
      auto& destination_block = traced_tensor.storage();
      // only one block here
      const Size line_size = destination_block.size();

      const auto const_line_size_variant = to_const_integral<Size, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(line_size);
      std::visit(
            [&](const auto& const_line_size) {
               const auto line_size = const_line_size.value();
               for (const auto& [symmetry_1, dimension] : merged_tensor.edges(0).segment) {
                  auto symmetry_2 = -symmetry_1;
                  // block symmetry_1 symmetry_2 0
                  auto source_block = merged_tensor.blocks(pmr::vector<Symmetry>{symmetry_1, symmetry_2, Symmetry()});
                  auto dimension_plus_one = dimension + 1;
                  for (Size i = 0; i < dimension; i++) {
                     const ScalarType* __restrict source_data = source_block.data() + dimension_plus_one * i * line_size;
                     ScalarType* __restrict destination_data = destination_block.data();
                     for (Size j = 0; j < line_size; j++) {
                        destination_data[j] += source_data[j];
                     }
                  }
               }
            },
            const_line_size_variant);

      auto result = traced_tensor.edge_operator_implement(
            pmr::map<Name, pmr::vector<std::tuple<Name, edge_segment_t<Symmetry>>>>{{InternalName<Name>::Trace_3, std::move(split_plan)}},
            reverse_names, // more than it have, it contains some traced edge, but it is ok
            empty_list<std::pair<Name, empty_list<Name>>>(),
            std::move(result_names),
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());
      return result;
   }
} // namespace TAT
#endif
