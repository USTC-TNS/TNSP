/**
 * \file edge_operator.hpp
 *
 * Copyright (C) 2019-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_EDGE_OPERATOR_HPP
#define TAT_EDGE_OPERATOR_HPP

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/const_hash_map.hpp"
#include "../utility/hash_for_list.hpp"
#include "../utility/timer.hpp"
#include "transpose.hpp"

namespace TAT {
   inline timer transpose_guard("transpose");

   template<typename ScalarType, typename Symmetry, typename Name>
   template<typename A, typename B, typename C, typename D, typename E, typename F, typename G, typename H>
   [[nodiscard]] auto Tensor<ScalarType, Symmetry, Name>::edge_operator_implement(
         const A& split_map,
         const B& reversed_name,
         const C& merge_map,
         std::vector<Name> new_names,
         const bool apply_parity,
         const D& parity_exclude_name_split,
         const E& parity_exclude_name_reversed_before_transpose,
         const F& parity_exclude_name_reversed_after_transpose,
         const G& parity_exclude_name_merge,
         const H& edge_and_symmetries_to_cut_before_all) const {
      // rename edge will not use edge operator, it will use special function which can also change the name type
      auto timer_guard = transpose_guard();
      auto hash_for_list = detail::hash_for_list();
      // step 1: cut
      // step 2: split
      // step 3: reverse
      // step 4: transpose
      // step 5: reverse before merge
      // step 6: merge

      // step 1,2,3 is doing up to down
      // step 4,5,6 need the result name to get the plan detail down to up
      // step 2,3,4,5,6 is symmetric along transpose

      // split map key is name before split
      // merge map key is name after merge

      // symmetry constexpr info
      constexpr auto is_fermi = Symmetry::is_fermi_symmetry;
      constexpr auto is_no_symmetry = Symmetry::length == 0;

      // Table of edge operation plan
      //
      //                     rank    name    edge
      // before_split        O       O       D
      // after_split         ↓       O       O
      // before_transpose    ↓       ↑       O
      // at_transpose        O       -       -
      // after_transpose     ↑       ↓       O
      // before_merge        ↑       O       O
      // after_merge         O       O       D
      //
      //                     flag                offset
      // split               rank[after_split]   (symmetry[]->(symmetry, offset))[before_split]
      // merge               rank[after_split]   (symmetry[]->(symmetry, offset))[after_merge]
      // symmetry[] is edges after split/before merge, it is mapped to symmetry and offset before split/after merge
      //
      // reversed            flag
      // before_transpose    bool[at_transpose]
      // after_transpose     bool[at_transpose]
      //
      // what maybe empty: two reverse flag and two offset map

      // before_split
      const Rank rank_before_split = get_rank();
      const auto& name_before_split = names;
      auto real_edge_before_split = std::vector<Edge<Symmetry>>();
      if (edge_and_symmetries_to_cut_before_all.size() != 0) {
         real_edge_before_split.reserve(rank_before_split);
         for (auto i = 0; i < rank_before_split; i++) {
            // edge_and_symmetries_to_cu_before_all :: Name -> Symmetry -> Size
            if (auto found = edge_and_symmetries_to_cut_before_all.find(names[i]); found != edge_and_symmetries_to_cut_before_all.end()) {
               // this edge need cut
               const auto& symmetry_to_cut_dimension = found->second;
               // symmetry_to_cut_dimension :: Symmetry -> Size
               auto& this_edge = real_edge_before_split.emplace_back(); // result edge
               if constexpr (is_fermi) {
                  this_edge.arrow = edges(i).arrow; // arrow not changed
               }
               for (const auto& [symmetry, dimension] : edges(i).segment) {
                  if (auto cut_iterator = symmetry_to_cut_dimension.find(symmetry); cut_iterator != symmetry_to_cut_dimension.end()) {
                     // this segment need cut
                     if (auto new_dimension = cut_iterator->second; new_dimension != 0) {
                        this_edge.segment.emplace_back(symmetry, new_dimension < dimension ? new_dimension : dimension);
                     }
                     // if new_dimension is zero, delete the entire segment
                     // cut edge will only change leading in future
                     // so later use core->edges to calculate offset, rather than edge_before_split
                  } else {
                     // this segment not need cut
                     this_edge.segment.emplace_back(symmetry, dimension);
                  }
               }
            } else {
               real_edge_before_split.push_back(edges(i));
            }
         }
      }
      const auto& edge_before_split = edge_and_symmetries_to_cut_before_all.size() != 0 ? real_edge_before_split : core->edges;

      // after_split
      // flag and offset is spilt plan
      auto split_flag = pmr::vector<Rank>();
      auto split_offset = pmr::vector<
            detail::const_hash_map<pmr::vector<Symmetry>, std::pair<Symmetry, Size>, detail::hash_for_list, detail::polymorphic_allocator>>();
      auto real_name_after_split = std::vector<Name>();
      auto edge_after_split = pmr::vector<EdgePointer<Symmetry>>();
      if (split_map.size() != 0) {
         // dont know rank after split(rank at transpose) now, but it can still reduce the realloc time
         split_flag.reserve(rank_before_split);            // rank_at_transpose
         split_offset.reserve(rank_before_split);          // no problem, it is rank before split
         real_name_after_split.reserve(rank_before_split); // rank_at_transpose
         edge_after_split.reserve(rank_before_split);      // rank_at_transpose
         for (Rank position_before_split = 0; position_before_split < rank_before_split; position_before_split++) {
            if (auto position = split_map.find(name_before_split[position_before_split]); position != split_map.end()) {
               // this edge is splitting
               const auto& this_split_begin_position_in_edge_after_split = edge_after_split.size();
               // the validity of edge after split is ensured by user
               Size this_offset_length = 1;
               for (const auto& [split_name, split_edge] : position->second) {
                  this_offset_length *= split_edge.segment.size();
                  real_name_after_split.push_back(split_name);
                  if constexpr (is_fermi) {
                     edge_after_split.push_back({split_edge.segment, edge_before_split[position_before_split].arrow});
                  } else {
                     edge_after_split.push_back({split_edge.segment});
                  }
                  split_flag.push_back(position_before_split);
               }
               // analyze split plan
               const auto edge_list_after_split = edge_after_split.data() + this_split_begin_position_in_edge_after_split;
               const auto split_rank = edge_after_split.size() - this_split_begin_position_in_edge_after_split;
               // loop between begin and end, get a map push_back into split_offset
               // this map is sym -> [sym] -> offset
               auto& this_offset = split_offset.emplace_back();
               this_offset.reserve(this_offset_length);
               auto offset_bank = pmr::unordered_map<Symmetry, Size>(unordered_parameter * edge_before_split[position_before_split].segment.size());
               // every sym contain several [sym], it is filled one by one
               for (const auto& [sym, dim] : edge_before_split[position_before_split].segment) {
                  offset_bank[sym] = 0;
               }
               auto accumulated_symmetries = pmr::vector<Symmetry>(split_rank);
               auto accumulated_dimensions = pmr::vector<Size>(split_rank);
               auto current_symmetries = pmr::vector<Symmetry>(split_rank);
               loop_edge<detail::polymorphic_allocator>(
                     edge_after_split.data() + this_split_begin_position_in_edge_after_split,
                     split_rank,
                     [&]() {
                        // this_offset[pmr::vector<Symmetry>{}] = {Symmetry(), 0};
                        this_offset.add(pmr::vector<Symmetry>{}, std::pair<Symmetry, Size>{Symmetry(), 0});
                     },
                     []() {},
                     [&](const auto& symmetry_iterator_list, Rank minimum_changed) {
                        for (auto i = minimum_changed; i < split_rank; i++) {
                           const auto& symmetry_iterator = symmetry_iterator_list[i];
                           accumulated_symmetries[i] = symmetry_iterator->first + (i ? accumulated_symmetries[i - 1] : Symmetry());
                           accumulated_dimensions[i] = symmetry_iterator->second * (i ? accumulated_dimensions[i - 1] : 1);
                           // do not check dim=0, because in constructor, it also didn't check
                           current_symmetries[i] = symmetry_iterator->first;
                        }
                        auto target_symmetry = accumulated_symmetries.back();
                        auto target_dimension = accumulated_dimensions.back();
                        // the target symmetry may not exist
                        if (auto found = offset_bank.find(target_symmetry); found != offset_bank.end()) {
                           // this_offset[current_symmetries] = {target_symmetry, found->second};
                           this_offset.add(current_symmetries, std::pair<Symmetry, Size>{target_symmetry, found->second});
                           found->second += target_dimension;
                        }
                        return split_rank;
                     });
               this_offset.sort();
            } else {
               // no split for this edge
               real_name_after_split.push_back(name_before_split[position_before_split]);
               if constexpr (is_fermi) {
                  edge_after_split.push_back({edge_before_split[position_before_split].segment, edge_before_split[position_before_split].arrow});
               } else {
                  edge_after_split.push_back({edge_before_split[position_before_split].segment});
               }
               split_flag.push_back(position_before_split);
               // auto& this_offset = split_offset.emplace_back();
               split_offset.emplace_back();
               // offset is not calculated, it will check no split so it will not use offset
               // for (const auto& [symmetry, dimension] : edge_before_split[position_before_split].segment) {
               //   this_offset[{symmetry}] = {symmetry, 0};
               // }
            }
         }
      } else {
         // no split for all edge
         edge_after_split.reserve(rank_before_split);
         split_flag.reserve(rank_before_split);
         // split_offset.reserve(rank_before_split);
         for (auto i = 0; i < rank_before_split; i++) {
            const auto& edge = edge_before_split[i];
            if constexpr (is_fermi) {
               edge_after_split.push_back({edge.segment, edge.arrow});
            } else {
               edge_after_split.push_back({edge.segment});
            }
            split_flag.push_back(i);
            // offset is not calculated, it will check no split so it will not use offset
            // even emplace back is not needed
            // auto& this_offset = split_offset.emplace_back();
            // for (const auto& [symmetry, dimension] : edge.segment) {
            //    this_offset[{symmetry}] = {symmetry, 0};
            // }
         }
      }
      const auto& name_after_split = split_map.size() != 0 ? real_name_after_split : name_before_split;
      const Rank rank_at_transpose = name_after_split.size();

      // before_transpose
      auto reversed_before_transpose_flag = pmr::vector<bool>(); // length = rank_at_transpose
      // this flag may be empty if no reverse happened
      auto fermi_edge_before_transpose = pmr::vector<EdgePointer<Symmetry>>();
      if constexpr (is_fermi) {
         if (reversed_name.size() != 0) {
            reversed_before_transpose_flag.reserve(rank_at_transpose);
            fermi_edge_before_transpose.reserve(rank_at_transpose);
            for (auto i = 0; i < rank_at_transpose; i++) {
               fermi_edge_before_transpose.push_back(edge_after_split[i]);
               if (reversed_name.find(name_after_split[i]) != reversed_name.end()) {
                  fermi_edge_before_transpose.back().arrow ^= true;
                  reversed_before_transpose_flag.push_back(true);
               } else {
                  reversed_before_transpose_flag.push_back(false);
               }
            }
         } else {
            reversed_before_transpose_flag = pmr::vector<bool>(rank_at_transpose, false);
         }
      }
      const auto& edge_before_transpose = is_fermi && reversed_name.size() != 0 ? fermi_edge_before_transpose : edge_after_split;

      // create res names
      auto result = Tensor<ScalarType, Symmetry, Name>();
      result.names = std::move(new_names);

      // analyze name info down to up
      const auto& name_after_merge = result.names;
      const Rank rank_after_merge = name_after_merge.size();
      auto merge_flag = pmr::vector<Rank>();
      auto real_name_before_merge = std::vector<Name>();
      if (merge_map.size() != 0) {
         merge_flag.reserve(rank_at_transpose);
         real_name_before_merge.reserve(rank_at_transpose);
         for (Rank position_after_merge = 0; position_after_merge < rank_after_merge; position_after_merge++) {
            const auto& merged_name = name_after_merge[position_after_merge];
            if (auto position = merge_map.find(merged_name); position != merge_map.end()) {
               for (const auto& merging_names : position->second) {
                  real_name_before_merge.push_back(merging_names);
                  merge_flag.push_back(position_after_merge);
               }
            } else {
               real_name_before_merge.push_back(merged_name);
               merge_flag.push_back(position_after_merge);
            }
         }
      } else {
         merge_flag.reserve(rank_after_merge);
         for (auto i = 0; i < rank_after_merge; i++) {
            merge_flag.push_back(i);
         }
      }
      const auto& name_before_merge = merge_map.size() != 0 ? real_name_before_merge : name_after_merge;
      if constexpr (debug_mode) {
         if (rank_at_transpose != name_before_merge.size()) {
            detail::error("Tensor to transpose with Different Rank");
         }
      }
      // what left: edge for after_transpose, before_merge and after merge

      // create plan of two way for transpose
      auto plan_source_to_destination = pmr::vector<Rank>(rank_at_transpose);
      auto plan_destination_to_source = pmr::vector<Rank>(rank_at_transpose);

      // and edge after transpose
      pmr::unordered_map<Name, int> name_after_split_to_index(unordered_parameter * rank_at_transpose);
      for (Rank i = 0; i < rank_at_transpose; i++) {
         name_after_split_to_index[name_after_split[i]] = i;
      }

      auto edge_after_transpose = pmr::vector<EdgePointer<Symmetry>>();
      edge_after_transpose.reserve(rank_at_transpose);
      for (auto i = 0; i < rank_at_transpose; i++) {
         auto found = name_after_split_to_index.find(name_before_merge[i]);
         if constexpr (debug_mode) {
            if (found == name_after_split_to_index.end()) {
               detail::error("Tensor to transpose with incompatible name list");
            }
         }
         plan_destination_to_source[i] = std::get<1>(*found);
         plan_source_to_destination[plan_destination_to_source[i]] = i;
         edge_after_transpose.push_back(edge_before_transpose[plan_destination_to_source[i]]);
      }

      // what plan left: merge offset, reversed flag

      // dealing with edge before merge and after merge together
      // prepare edge_before_merge
      auto fermi_edge_before_merge = pmr::vector<EdgePointer<Symmetry>>();
      if constexpr (is_fermi) {
         if (merge_map.size() != 0) {
            fermi_edge_before_merge.reserve(rank_at_transpose);
            for (const auto& edge : edge_after_transpose) {
               fermi_edge_before_merge.push_back(edge);
            }
         }
      }
      auto& edge_before_merge = is_fermi && merge_map.size() != 0 ? fermi_edge_before_merge : edge_after_transpose;

      // the last three things
      auto reversed_after_transpose_flag = pmr::vector<bool>();
      auto result_edge = std::vector<Edge<Symmetry>>();
      auto merge_offset = pmr::vector<
            detail::const_hash_map<pmr::vector<Symmetry>, std::pair<Symmetry, Size>, detail::hash_for_list, detail::polymorphic_allocator>>();
      if (merge_map.size() != 0) {
         if constexpr (is_fermi) {
            reversed_after_transpose_flag.reserve(rank_at_transpose);
         }
         result_edge.reserve(rank_after_merge);
         merge_offset.reserve(rank_after_merge);
         for (Rank position_after_merge = 0, start_of_merge = 0, end_of_merge = 0; position_after_merge < rank_after_merge; position_after_merge++) {
            // [start, end) need be merged into one edge
            Size this_offset_length = 1;
            while (end_of_merge < rank_at_transpose && merge_flag[end_of_merge] == position_after_merge) {
               this_offset_length *= edge_before_merge[end_of_merge].segment.size();
               end_of_merge++;
            }
            // arrow begin
            Arrow arrow;
            bool arrow_fixed = false;
            if constexpr (is_fermi) {
               if (start_of_merge == end_of_merge) {
                  // empty merge, merge zero edge into one edge, it will generate [(Symmetry(), 1)]
                  arrow = false;
                  // TODO it should be controled
               } else {
                  // normal merge
                  arrow = edge_before_merge[start_of_merge].arrow;
               }
               for (auto merge_group_position = start_of_merge; merge_group_position < end_of_merge; merge_group_position++) {
                  if (arrow == edge_before_merge[merge_group_position].arrow) {
                     reversed_after_transpose_flag.push_back(false);
                  } else {
                     edge_before_merge[merge_group_position].arrow ^= true;
                     reversed_after_transpose_flag.push_back(true);
                  }
               }
            }
            // arrow end

            // merge edge begin
            // result edge and offset
            auto merged_edge = std::vector<std::pair<Symmetry, Size>>();
            auto& this_offset = merge_offset.emplace_back();
            this_offset.reserve(this_offset_length);

            const Rank merge_rank = end_of_merge - start_of_merge;
            auto accumulated_symmetries = pmr::vector<Symmetry>(merge_rank);
            auto accumulated_dimensions = pmr::vector<Size>(merge_rank);
            auto current_symmetries = pmr::vector<Symmetry>(merge_rank);

            if (merge_rank != 1) {
               loop_edge<detail::polymorphic_allocator>(
                     edge_before_merge.data() + start_of_merge,
                     merge_rank,
                     [&]() {
                        merged_edge.push_back({Symmetry(), 1});
                        // this_offset[pmr::vector<Symmetry>{}] = {Symmetry(), 0};
                        this_offset.add(pmr::vector<Symmetry>{}, std::pair<Symmetry, Size>{Symmetry(), 0});
                     },
                     []() {},
                     [&](const auto& symmetry_iterator_list, const Rank minimum_changed) {
                        for (auto i = minimum_changed; i < merge_rank; i++) {
                           const auto& symmetry_iterator = symmetry_iterator_list[i];
                           accumulated_symmetries[i] = symmetry_iterator->first + (i ? accumulated_symmetries[i - 1] : Symmetry());
                           accumulated_dimensions[i] = symmetry_iterator->second * (i ? accumulated_dimensions[i - 1] : 1);
                           // do not check dim=0, because in constructor, i didn't check
                           current_symmetries[i] = symmetry_iterator->first;
                        }
                        auto target_symmetry = accumulated_symmetries.back();
                        auto found = std::find_if(merged_edge.begin(), merged_edge.end(), [&target_symmetry](auto x) {
                           return x.first == target_symmetry;
                        });
                        if (found == merged_edge.end()) {
                           merged_edge.push_back({target_symmetry, 0});
                           found = std::prev(merged_edge.end());
                        }
                        // this_offset[current_symmetries] = {target_symmetry, found->second};
                        this_offset.add(current_symmetries, std::pair<Symmetry, Size>{target_symmetry, found->second});
                        found->second += accumulated_dimensions.back();
                        return merge_rank;
                     });
               this_offset.sort();
               auto& real_merged_edge = result_edge.emplace_back(std::move(merged_edge));
               if constexpr (is_fermi) {
                  real_merged_edge.arrow = arrow;
               }
            } else {
               // not merge for this edge
               const auto& target_edge = edge_before_merge[start_of_merge];
               auto& real_merged_edge = result_edge.emplace_back(target_edge.segment);
               if constexpr (is_fermi) {
                  real_merged_edge.arrow = target_edge.arrow;
               }
               // it will check later so this_offset it not calculated
            }
            // merge edge end
            start_of_merge = end_of_merge;
         }
      } else {
         // no merge for all edge
         if constexpr (is_fermi) {
            reversed_after_transpose_flag = pmr::vector<bool>(rank_at_transpose, false);
            // no merge is no reverse before merge
         }
         result_edge.reserve(rank_after_merge);
         for (auto i = 0; i < rank_after_merge; i++) {
            const auto& edge = edge_before_merge[i];
            if constexpr (is_fermi) {
               result_edge.push_back({edge.segment, edge.arrow});
            } else {
               result_edge.push_back({edge.segment});
            }
            // it will check if no merge, so it will not use offset, so offset is not calculated
            // even emplace_back is not needed
            // auto& this_offset = merge_offset.emplace_back();
            // for (const auto& [symmetry, dimension] : edge.segment) {
            //    this_offset[{symmetry}] = {symmetry, 0};
            // }
         }
      }

      // put res_edge into res
      result.core = detail::shared_ptr<Core<ScalarType, Symmetry>>::make(std::move(result_edge));
      result.core->clear_unused_symmetry();
      if constexpr (debug_mode) {
         result.check_valid_name();
      }
      const auto& edge_after_merge = result.core->edges;

      // Table for analyze how data move
      //
      // before->source          symmetry[at_transpose]->(symmetry[before_split], offset[before_split])
      // after->destination      symmetry[at_transpose]->(symmetry[after_merge], offset[after_merge])
      //
      // marks:
      // split_flag_mark
      // reversed_before_transpose_flag_mark
      // reversed_after_transpose_flag_mark
      // merge_flag_mark

      using MapFromTransposeToSourceDestination = detail::const_hash_map<
            pmr::vector<Symmetry>,
            std::pair<pmr::vector<Symmetry>, pmr::vector<Size>>,
            detail::hash_for_list,
            detail::polymorphic_allocator>;
      // split part
      auto data_before_transpose_to_source = MapFromTransposeToSourceDestination();
      if (split_map.size() != 0) {
         // if some edge is cut, some symmetries list should not appear in the data_before_transpose_to_source
         // so main copy loop should be loop by data_after_transpose_to_destination
         auto map_pool = initialize_block_symmetries_with_check<detail::polymorphic_allocator>(edge_after_split.data(), edge_after_split.size());
         data_before_transpose_to_source.reserve(map_pool.size());
         for (auto& [symmetries_before_transpose, size] : map_pool) {
            // convert sym -> target_sym and offsets
            // and add to map
            auto symmetries = pmr::vector<Symmetry>();
            auto offsets = pmr::vector<Size>();
            symmetries.reserve(rank_before_split);
            offsets.reserve(rank_before_split);
            bool success = true;
            for (Rank position_before_split = 0, position_after_split = 0; position_before_split < rank_before_split; position_before_split++) {
               // get [start, end) which is splitted
               auto split_group_symmetries = pmr::vector<Symmetry>(); // single split edges
               while (position_after_split < rank_at_transpose && split_flag[position_after_split] == position_before_split) {
                  split_group_symmetries.push_back(symmetries_before_transpose[position_after_split]);
                  position_after_split++;
               }
               // if no split happened, do not use split_offset
               if (split_group_symmetries.size() != 1) {
                  // if it is empty split, split_group_symmetries = {}, it also work
                  if (auto found = split_offset[position_before_split].find(split_group_symmetries);
                      found != split_offset[position_before_split].end()) {
                     const auto& this_symmetry = found->second.first;
                     const auto& this_offset = found->second.second;
                     symmetries.push_back(this_symmetry);
                     offsets.push_back(this_offset);
                  } else {
                     success = false;
                     break; // no source block
                  }
               } else {
                  symmetries.push_back(split_group_symmetries.front());
                  offsets.push_back(0);
               }
            }
            if (success) {
               data_before_transpose_to_source.add(std::move(symmetries_before_transpose), std::pair{std::move(symmetries), std::move(offsets)});
            }
         }
      } else {
         // no split indeed
         data_before_transpose_to_source.reserve(core->blocks.size());
         for (const auto& [symmetries, block] : core->blocks) {
            data_before_transpose_to_source.add(
                  pmr::vector<Symmetry>{symmetries.begin(), symmetries.end()},
                  std::pair{pmr::vector<Symmetry>{symmetries.begin(), symmetries.end()}, pmr::vector<Size>(rank_before_split, 0)});
         }
      }
      data_before_transpose_to_source.sort();
      // merge part
      auto data_after_transpose_to_destination = MapFromTransposeToSourceDestination();
      if (merge_map.size() != 0) {
         auto map_pool = initialize_block_symmetries_with_check<detail::polymorphic_allocator>(edge_before_merge.data(), edge_before_merge.size());
         data_after_transpose_to_destination.reserve(map_pool.size());
         for (auto& [symmetries_after_transpose, size] : map_pool) {
            // convert sym -> target_sym and offsets
            // and add to map
            auto symmetries = pmr::vector<Symmetry>();
            auto offsets = pmr::vector<Size>();
            symmetries.reserve(rank_after_merge);
            offsets.reserve(rank_after_merge);
            bool success = true;
            for (Rank position_after_merge = 0, position_before_merge = 0; position_after_merge < rank_after_merge; position_after_merge++) {
               // find [start, end) which be merged
               auto merge_group_symmetries = pmr::vector<Symmetry>(); // single merge edges
               while (position_before_merge < rank_at_transpose && merge_flag[position_before_merge] == position_after_merge) {
                  merge_group_symmetries.push_back(symmetries_after_transpose[position_before_merge]);
                  position_before_merge++;
               }
               // if no merge happened, do not use merge_offset
               if (merge_group_symmetries.size() != 1) {
                  if (auto found = merge_offset[position_after_merge].find(merge_group_symmetries);
                      found != merge_offset[position_after_merge].end()) {
                     const auto& this_symmetry = found->second.first;
                     const auto& this_offset = found->second.second;
                     symmetries.push_back(this_symmetry);
                     offsets.push_back(this_offset);
                  } else {
                     success = false;
                     break; // no destination block
                  }
               } else {
                  symmetries.push_back(merge_group_symmetries.front());
                  offsets.push_back(0);
               }
            }
            if (success) {
               data_after_transpose_to_destination.add(std::move(symmetries_after_transpose), std::pair{std::move(symmetries), std::move(offsets)});
            }
         }
      } else {
         // no merge indeed
         data_after_transpose_to_destination.reserve(result.core->blocks.size());
         for (const auto& [symmetries, block] : result.core->blocks) {
            data_after_transpose_to_destination.add(
                  pmr::vector<Symmetry>{symmetries.begin(), symmetries.end()},
                  std::pair{pmr::vector<Symmetry>{symmetries.begin(), symmetries.end()}, pmr::vector<Size>(rank_after_merge, 0)});
         }
      }
      // not need to sort data_after_transpose_to_destination

      // marks
      auto split_flag_mark = pmr::vector<bool>();
      auto reversed_before_transpose_flag_mark = pmr::vector<bool>();
      auto reversed_after_transpose_flag_mark = pmr::vector<bool>();
      auto merge_flag_mark = pmr::vector<bool>();
      if constexpr (is_fermi) {
         split_flag_mark.reserve(rank_before_split);
         reversed_before_transpose_flag_mark.reserve(rank_at_transpose);
         reversed_after_transpose_flag_mark.reserve(rank_at_transpose);
         merge_flag_mark.reserve(rank_after_merge);
         // true => apply parity
         if (apply_parity) {
            // default apply, so apply == not in exclude, namely, find==end
            for (auto i = 0; i < rank_before_split; i++) {
               split_flag_mark.push_back(parity_exclude_name_split.find(name_before_split[i]) == parity_exclude_name_split.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_before_transpose_flag_mark.push_back(
                     parity_exclude_name_reversed_before_transpose.find(name_after_split[i]) == parity_exclude_name_reversed_before_transpose.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_after_transpose_flag_mark.push_back(
                     parity_exclude_name_reversed_after_transpose.find(name_before_merge[i]) == parity_exclude_name_reversed_after_transpose.end());
            }
            for (auto i = 0; i < rank_after_merge; i++) {
               merge_flag_mark.push_back(parity_exclude_name_merge.find(name_after_merge[i]) == parity_exclude_name_merge.end());
            }
         } else {
            for (auto i = 0; i < rank_before_split; i++) {
               split_flag_mark.push_back(parity_exclude_name_split.find(name_before_split[i]) != parity_exclude_name_split.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_before_transpose_flag_mark.push_back(
                     parity_exclude_name_reversed_before_transpose.find(name_after_split[i]) != parity_exclude_name_reversed_before_transpose.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_after_transpose_flag_mark.push_back(
                     parity_exclude_name_reversed_after_transpose.find(name_before_merge[i]) != parity_exclude_name_reversed_after_transpose.end());
            }
            for (auto i = 0; i < rank_after_merge; i++) {
               merge_flag_mark.push_back(parity_exclude_name_merge.find(name_after_merge[i]) != parity_exclude_name_merge.end());
            }
         }
      }
      // it maybe create no source block
      if constexpr (Symmetry::length != 0) {
         if (split_map.size() != 0) {
            // result.zero();
            auto& result_storage = result.storage();
            std::fill(result_storage.begin(), result_storage.end(), 0);
         }
      }

      // Main copy loop
      for (const auto& [symmetries_after_transpose, destination_symmetries_destination_offsets] : data_after_transpose_to_destination) {
         const auto& [destination_symmetries, destination_offsets] = destination_symmetries_destination_offsets;
         // Table of transpose info
         //                   source                     destination
         // symmetries        symmetry[before_split]     symmetry[after_merge]
         // offsets           offset[before_split]       offset[after_merge]
         //
         //                   before_transpose           after_transpose
         // symmetries        symmetry[at_transpose]     symmetry[at_transpose]
         // dimensions        dimension[at_transpose]    dimension[at_transpose]
         //
         //                   source         destination
         // total_offset      O              O
         // block             O              O
         //
         // leadings
         // leadings_of_source[before_split] -> leadings_before_transpose[at_transpose]
         // leadings_of_destination[after_merge] -> leadings_after_transpose[at_transpose]

         auto symmetries_before_transpose = pmr::vector<Symmetry>(rank_at_transpose);
         auto dimensions_after_transpose = pmr::vector<Size>(rank_at_transpose);
         auto dimensions_before_transpose = pmr::vector<Size>(rank_at_transpose);
         Size total_size = 1;
         for (auto i = 0; i < rank_at_transpose; i++) {
            auto dimension = edge_after_transpose[i].get_dimension_from_symmetry(symmetries_after_transpose[i]);
            dimensions_after_transpose[i] = dimension;
            dimensions_before_transpose[plan_destination_to_source[i]] = dimension;
            symmetries_before_transpose[plan_destination_to_source[i]] = symmetries_after_transpose[i];
            total_size *= dimension;
         }

         // split may generate a empty block
         auto found = data_before_transpose_to_source.find(symmetries_before_transpose);
         if (found == data_before_transpose_to_source.end()) {
            continue;
         }
         const auto& source_symmetries = found->second.first;
         const auto& source_offsets = found->second.second;

         // get block, offset and leadings
         auto found_source_block = detail::fake_map_find<true>(core->blocks, source_symmetries);
         if (found_source_block == core->blocks.end()) {
            continue;
         }
         const auto& source_block = found_source_block->second;
         auto found_destination_block = detail::fake_map_find<true>(result.core->blocks, destination_symmetries);
         if (found_destination_block == result.core->blocks.end()) {
            continue;
         }
         auto& destination_block = found_destination_block->second;

         Size total_source_offset = 0;
         for (auto i = 0; i < rank_before_split; i++) {
            // not use edge_before_split, use edges(i), since edge cut
            total_source_offset *= edges(i).get_dimension_from_symmetry(source_symmetries[i]);
            total_source_offset += source_offsets[i];
         }
         Size total_destination_offset = 0;
         for (auto i = 0; i < rank_after_merge; i++) {
            total_destination_offset *= edge_after_merge[i].get_dimension_from_symmetry(destination_symmetries[i]);
            total_destination_offset += destination_offsets[i];
         }

         auto leadings_of_source = pmr::vector<Size>(rank_before_split);
         for (auto i = rank_before_split; i-- > 0;) {
            if (i == rank_before_split - 1) {
               leadings_of_source[i] = 1;
            } else {
               // not use edge_before_split, use edges(i), since edge cut
               leadings_of_source[i] = leadings_of_source[i + 1] * edges(i + 1).get_dimension_from_symmetry(source_symmetries[i + 1]);
            }
         }
         auto leadings_before_transpose = pmr::vector<Size>(rank_at_transpose);
         for (auto i = rank_at_transpose; i-- > 0;) {
            if (i != rank_at_transpose - 1 && split_flag[i] == split_flag[i + 1]) {
               leadings_before_transpose[i] = leadings_before_transpose[i + 1] * dimensions_before_transpose[i + 1];
               // dimensions_before_transpose[i + 1] == edge_before_transpose[i + 1].segment.at(symmetries_before_transpose[i + 1]);
            } else {
               // no split leading
               leadings_before_transpose[i] = leadings_of_source[split_flag[i]];
            }
         }

         auto leadings_of_destination = pmr::vector<Size>(rank_after_merge);
         for (auto i = rank_after_merge; i-- > 0;) {
            if (i == rank_after_merge - 1) {
               leadings_of_destination[i] = 1;
            } else {
               leadings_of_destination[i] =
                     leadings_of_destination[i + 1] * edge_after_merge[i + 1].get_dimension_from_symmetry(destination_symmetries[i + 1]);
            }
         }
         auto leadings_after_transpose = pmr::vector<Size>(rank_at_transpose);
         for (auto i = rank_at_transpose; i-- > 0;) {
            if (i != rank_at_transpose - 1 && merge_flag[i] == merge_flag[i + 1]) {
               leadings_after_transpose[i] = leadings_after_transpose[i + 1] * dimensions_after_transpose[i + 1];
               // dimensions_after_transpose[i + 1] == edge_after_transpose[i + 1].segment.at(symmetries_after_transpose[i + 1]);
            } else {
               // no merge leading
               leadings_after_transpose[i] = leadings_of_destination[merge_flag[i]];
            }
         }

         // parity
         auto parity = false;
         if constexpr (is_fermi) {
            parity ^= Symmetry::get_transpose_parity(symmetries_before_transpose, plan_source_to_destination);

            parity ^= Symmetry::get_reverse_parity(symmetries_before_transpose, reversed_before_transpose_flag, reversed_before_transpose_flag_mark);
            parity ^= Symmetry::get_split_merge_parity(symmetries_before_transpose, split_flag, split_flag_mark);
            parity ^= Symmetry::get_reverse_parity(symmetries_after_transpose, reversed_after_transpose_flag, reversed_after_transpose_flag_mark);
            parity ^= Symmetry::get_split_merge_parity(symmetries_after_transpose, merge_flag, merge_flag_mark);
         }

         do_transpose(
               source_block.data() + total_source_offset,
               destination_block.data() + total_destination_offset,
               plan_source_to_destination,
               plan_destination_to_source,
               dimensions_before_transpose,
               dimensions_after_transpose,
               leadings_before_transpose,
               leadings_after_transpose,
               rank_at_transpose,
               total_size,
               parity);
      }

      return result;
   }
} // namespace TAT
#endif
