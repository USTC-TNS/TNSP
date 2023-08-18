/**
 * \file edge_operator.hpp
 *
 * Copyright (C) 2019-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <utility>

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/timer.hpp"

namespace TAT {
   inline timer transpose_guard("transpose");

   template<typename ScalarType, typename Symmetry, typename Name>
   template<typename A, typename B, typename C, typename D, typename E, typename F, typename G, typename H>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::edge_operator_implement(
         const A& split_map,
         const B& reversed_names,
         const C& merge_map,
         std::vector<Name> new_names,
         const bool apply_parity,
         const D& parity_exclude_names_split,
         const E& parity_exclude_names_reversed_before_transpose,
         const F& parity_exclude_names_reversed_after_transpose,
         const G& parity_exclude_names_merge,
         const H& edges_and_symmetries_to_cut_before_all) const {
      // rename edge will not use edge operator, it will use special function which can also change the name type
      auto timer_guard = transpose_guard();
      // step 1: cut
      // step 2: split
      // step 3: reverse
      // step 4: transpose
      // step 5: reverse before merge
      // step 6: merge

      // step 1,2,3 is doing up to down
      // step 4,5,6 need the result name to get the plan detail down to up
      // step 2,3,4,5,6 is symmetric along transpose

      // symmetry constexpr info
      constexpr auto is_fermi = Symmetry::is_fermi_symmetry;

      // Table of edge operation plan
      //
      //                     rank    names   edges
      // before_split        O       O       D
      // after_split         ↓       O       O
      // before_transpose    ↓       ↑       O
      // at_transpose        O       -       -
      // after_transpose     ↑       ↓       O
      // before_merge        ↑       O       O
      // after_merge         O       O       D
      //
      // To find original block and offset from block at transpose, we need
      // [symmetry] -> [(symmetry, offset)], the former is contiguous, the later share leading with the original block.
      // if no split happen, S -> (S, 0) where S:Symmetry, otherwise, a group of [S] splitted from single symmetry S.
      // For every edge splitted, need a pool, loop at split result [E], and sum every [S] in to S and add it to pool to get offset.

      //                     flags                 offsets
      // split               rank[at_transpose]   (symmetry[]->(total_symmetry, position_of_total_symmetry, size, offset, parity))[before_split]
      // merge               rank[at_transpose]   (symmetry[]->(total_symmetry, position_of_total_symmetry, size, offset, parity))[after_merge]
      // symmetry[] is edges after split/before merge, it is mapped to symmetry and offset before split/after merge
      //
      // reversed            flags
      // before_transpose    bool[at_transpose]
      // after_transpose     bool[at_transpose]
      //
      // what maybe empty: two reverse flag and two offset map

      // before_split
      const Rank rank_before_split = rank();
      const auto& names_before_split = names();
      auto real_edges_before_split = std::vector<Edge<Symmetry>>();
      if (edges_and_symmetries_to_cut_before_all.size() != 0) {
         real_edges_before_split.reserve(rank_before_split);
         for (auto index_before_split = 0; index_before_split < rank_before_split; index_before_split++) {
            // edge_and_symmetries_to_cut_before_all :: Name -> Symmetry -> Size
            if (auto found = edges_and_symmetries_to_cut_before_all.find(names(index_before_split));
                found != edges_and_symmetries_to_cut_before_all.end()) {
               // this edge need cut
               const auto& symmetry_to_cut_dimension = found->second;
               // symmetry_to_cut_dimension :: Symmetry -> Size
               std::vector<std::pair<Symmetry, Size>> segment;
               for (const auto& [symmetry, dimension] : edges(index_before_split).segments()) {
                  if (auto cut_iterator = symmetry_to_cut_dimension.find(symmetry); cut_iterator != symmetry_to_cut_dimension.end()) {
                     // this segment need cut
                     if (auto new_dimension = cut_iterator->second; new_dimension != 0) {
                        segment.emplace_back(symmetry, new_dimension < dimension ? new_dimension : dimension);
                     }
                     // if new_dimension is zero, delete the entire segment
                     // cut edge will only change leading in future
                     // so later use core->edges to calculate offset, rather than edge_before_split
                  } else {
                     // this segment not need cut
                     segment.emplace_back(symmetry, dimension);
                  }
               }
               real_edges_before_split.emplace_back(std::move(segment), edges(index_before_split).arrow()); // result edge
            } else {
               real_edges_before_split.push_back(edges(index_before_split));
            }
         }
      }
      const auto& edges_before_split = edges_and_symmetries_to_cut_before_all.size() != 0 ? real_edges_before_split : edges();

      // after_split
      // flag and offset is spilt plan
      auto real_names_after_split = std::vector<Name>();
      auto split_flags = pmr::vector<Rank>();
      auto edges_after_split = pmr::vector<EdgePointer<Symmetry>>();
      auto split_offsets_pool = pmr::vector<pmr::vector<std::tuple<Symmetry, Size, Size, Size, Size>>>();
      auto split_offsets = pmr::vector<std::optional<mdspan<std::tuple<Symmetry, Size, Size, Size, Size>, pmr::vector<Size>>>>();
      if (split_map.size() != 0) {
         // dont know rank after split(rank at transpose) now, but it can still reduce the realloc time
         real_names_after_split.reserve(rank_before_split); // rank_at_transpose
         split_flags.reserve(rank_before_split);            // rank_at_transpose
         edges_after_split.reserve(rank_before_split);      // rank_at_transpose
         split_offsets_pool.reserve(rank_before_split);     // no problem, it is rank before split
         split_offsets.reserve(rank_before_split);          // no problem, it is rank before split
         for (Rank index_before_split = 0; index_before_split < rank_before_split; index_before_split++) {
            if (auto found = split_map.find(names_before_split[index_before_split]); found != split_map.end()) {
               // this edge is splitting
               // the split begin, get it now because the pointer may change during puch_back
               const auto& this_split_begin_index_after_split = edges_after_split.size();
               auto this_split_shape = pmr::vector<Size>();
               this_split_shape.reserve(found->second.size());
               // the validity of edge after split is ensured by user
               for (const auto& [split_name, split_edge] : found->second) {
                  real_names_after_split.push_back(split_name);
                  split_flags.push_back(index_before_split);
                  edges_after_split.emplace_back(split_edge.segments(), edges_before_split[index_before_split].arrow());
                  this_split_shape.push_back(split_edge.segments_size());
               }
               const auto this_edges_after_split = &edges_after_split[this_split_begin_index_after_split];
               auto& this_offset = split_offsets.emplace_back(std::in_place, nullptr, std::move(this_split_shape)).value();
               auto& this_offset_pool =
                     split_offsets_pool.emplace_back(this_offset.size(), std::tuple<Symmetry, Size, Size, Size, Size>{{}, 0, 1, 0, 0});
               this_offset.set_data(this_offset_pool.data());
               // sum symmetry
               for (auto this_index_after_split = 0; this_index_after_split < this_offset.rank(); this_index_after_split++) {
                  Size self_size = this_offset.dimensions(this_index_after_split);
                  Size in_size = this_offset.leadings(this_index_after_split);
                  Size out_size = this_offset.size() == 0 ? 0 : this_offset.size() / (self_size * in_size);
                  for (Size x = 0; x < out_size; x++) {
                     auto offset_for_x = x;
                     for (Size y = 0; y < self_size; y++) {
                        const auto& [symmetry_here, dimension_here] = this_edges_after_split[this_index_after_split].segments(y);
                        auto offset_for_y = offset_for_x * self_size + y;
                        for (Size z = 0; z < in_size; z++) {
                           auto offset_for_z = offset_for_y * in_size + z;
                           std::get<0>(this_offset_pool[offset_for_z]) += symmetry_here;
                           std::get<2>(this_offset_pool[offset_for_z]) *= dimension_here;
                           std::get<4>(this_offset_pool[offset_for_z]) += symmetry_here.parity();
                        }
                     }
                  }
               }
               // accumulate offset
               auto offset_bank = pmr::vector<Size>(edges_before_split[index_before_split].segments_size());
               for (auto& [symmetry, position, size, offset, parity] : this_offset_pool) {
                  position = edges_before_split[index_before_split].position_by_symmetry(symmetry);
                  offset = offset_bank[position];
                  offset_bank[position] += size;
               }
            } else {
               // no split for this edge
               real_names_after_split.push_back(names_before_split[index_before_split]);
               split_flags.push_back(index_before_split);
               edges_after_split.emplace_back(edges_before_split[index_before_split].segments(), edges_before_split[index_before_split].arrow());
               split_offsets.emplace_back();
               // offset is not calculated, it will check no split so it will not use offset
            }
         }
      } else {
         // no split for all edge
         edges_after_split.reserve(rank_before_split);
         split_flags.reserve(rank_before_split);
         // split_offset.reserve(rank_before_split);
         for (auto index_before_split = 0; index_before_split < rank_before_split; index_before_split++) {
            const auto& edge = edges_before_split[index_before_split];
            split_flags.push_back(index_before_split);
            edges_after_split.emplace_back(edge.segments(), edge.arrow());
            // offset is not calculated, it will check no split so it will not use offset
            // even emplace back is not needed
         }
      }
      const auto& names_after_split = split_map.size() != 0 ? real_names_after_split : names_before_split;
      const Rank rank_at_transpose = names_after_split.size();

      // before_transpose
      auto reversed_before_transpose_flags = pmr::vector<bool>(); // length = rank_at_transpose
      // this flag may be empty if no reverse happened
      auto fermi_edges_before_transpose = pmr::vector<EdgePointer<Symmetry>>();
      if constexpr (is_fermi) {
         if (reversed_names.size() != 0) {
            reversed_before_transpose_flags.reserve(rank_at_transpose);
            fermi_edges_before_transpose.reserve(rank_at_transpose);
            for (auto index_after_split = 0; index_after_split < rank_at_transpose; index_after_split++) {
               auto& this_edge = fermi_edges_before_transpose.emplace_back(edges_after_split[index_after_split]);
               if (reversed_names.find(names_after_split[index_after_split]) != reversed_names.end()) {
                  this_edge.reverse_arrow();
                  reversed_before_transpose_flags.push_back(true);
               } else {
                  reversed_before_transpose_flags.push_back(false);
               }
            }
         } else {
            reversed_before_transpose_flags = pmr::vector<bool>(rank_at_transpose, false);
         }
      }
      const auto& edges_before_transpose = is_fermi && reversed_names.size() != 0 ? fermi_edges_before_transpose : edges_after_split;

      // analyze name info down to up
      const auto& names_after_merge = new_names;
      const Rank rank_after_merge = names_after_merge.size();

      auto merge_flags = pmr::vector<Rank>();
      auto real_names_before_merge = std::vector<Name>();
      if (merge_map.size() != 0) {
         merge_flags.reserve(rank_at_transpose);
         real_names_before_merge.reserve(rank_at_transpose);
         for (Rank index_after_merge = 0; index_after_merge < rank_after_merge; index_after_merge++) {
            const auto& merged_name = names_after_merge[index_after_merge];
            if (auto position = merge_map.find(merged_name); position != merge_map.end()) {
               for (const auto& merging_name : position->second) {
                  real_names_before_merge.push_back(merging_name);
                  merge_flags.push_back(index_after_merge);
               }
            } else {
               real_names_before_merge.push_back(merged_name);
               merge_flags.push_back(index_after_merge);
            }
         }
      } else {
         merge_flags.reserve(rank_after_merge);
         for (auto index_after_merge = 0; index_after_merge < rank_after_merge; index_after_merge++) {
            merge_flags.push_back(index_after_merge);
         }
      }
      const auto& names_before_merge = merge_map.size() != 0 ? real_names_before_merge : names_after_merge;

      if constexpr (debug_mode) {
         if (rank_at_transpose != names_before_merge.size()) {
            detail::error("Tensor to transpose with Different Rank");
         }
      }
      // what left: edge for after_transpose, before_merge and after merge

      // create plan of two way for transpose
      auto plan_source_to_destination = pmr::vector<Rank>(rank_at_transpose);
      auto plan_destination_to_source = pmr::vector<Rank>(rank_at_transpose);

      // and edge after transpose
      pmr::unordered_map<Name, Rank> name_after_split_to_index(unordered_parameter * rank_at_transpose);
      for (Rank i = 0; i < rank_at_transpose; i++) {
         name_after_split_to_index[names_after_split[i]] = i;
      }

      auto edges_after_transpose = pmr::vector<EdgePointer<Symmetry>>();
      edges_after_transpose.reserve(rank_at_transpose);
      for (auto index_before_merge = 0; index_before_merge < rank_at_transpose; index_before_merge++) {
         auto found = name_after_split_to_index.find(names_before_merge[index_before_merge]);
         if constexpr (debug_mode) {
            if (found == name_after_split_to_index.end()) {
               detail::error("Tensor to transpose with incompatible name list");
            }
         }
         auto index_after_split = plan_destination_to_source[index_before_merge] = found->second;
         plan_source_to_destination[index_after_split] = index_before_merge;
         edges_after_transpose.push_back(edges_before_transpose[index_after_split]);
      }

      // what plan left: merge offset, reversed flag

      // dealing with edge before merge and after merge together
      // prepare edge_before_merge, copy it to record reverse arrow for fermi symmetry.
      auto fermi_edges_before_merge = pmr::vector<EdgePointer<Symmetry>>();
      if constexpr (is_fermi) {
         if (merge_map.size() != 0) {
            fermi_edges_before_merge.reserve(rank_at_transpose);
            for (const auto& edge : edges_after_transpose) {
               fermi_edges_before_merge.push_back(edge);
            }
         }
      }
      auto& edges_before_merge = is_fermi && merge_map.size() != 0 ? fermi_edges_before_merge : edges_after_transpose;

      // the last 3.5 things
      // reversed_after_transpose_flag, merge_offset, edge_after_merge, and half edge_before_merge
      auto reversed_after_transpose_flags = pmr::vector<bool>();
      auto result_edges = std::vector<Edge<Symmetry>>();
      auto merge_offsets_pool = pmr::vector<pmr::vector<std::tuple<Symmetry, Size, Size, Size, Size>>>();
      auto merge_offsets = pmr::vector<std::optional<mdspan<std::tuple<Symmetry, Size, Size, Size, Size>, pmr::vector<Size>>>>();
      if (merge_map.size() != 0) {
         if constexpr (is_fermi) {
            reversed_after_transpose_flags.reserve(rank_at_transpose);
         }
         result_edges.reserve(rank_after_merge);
         merge_offsets_pool.reserve(rank_after_merge);
         merge_offsets.reserve(rank_after_merge);
         for (Rank index_after_merge = 0, start_of_merge = 0, end_of_merge = 0; index_after_merge < rank_after_merge; index_after_merge++) {
            // start/end_of_merge :: index_at_transpose
            // [start, end) need be merged into one edge
            auto this_merge_shape = pmr::vector<Size>();
            this_merge_shape.reserve(rank_at_transpose); // larger than needed
            while (end_of_merge < rank_at_transpose && merge_flags[end_of_merge] == index_after_merge) {
               this_merge_shape.push_back(edges_before_merge[end_of_merge].segments_size());
               end_of_merge++;
            }
            // arrow begin
            Arrow arrow;
            if constexpr (is_fermi) {
               if (start_of_merge == end_of_merge) {
                  // empty merge, merge zero edge into one edge, it will generate [(Symmetry(), 1)]
                  // For Symmetry(), arrow is not important.
                  arrow = false;
               } else {
                  // normal merge
                  arrow = edges_before_merge[start_of_merge].arrow();
               }
               for (auto this_index_before_merge = start_of_merge; this_index_before_merge < end_of_merge; this_index_before_merge++) {
                  auto& this_edge = edges_before_merge[this_index_before_merge];
                  if (arrow == this_edge.arrow()) {
                     reversed_after_transpose_flags.push_back(false);
                  } else {
                     this_edge.reverse_arrow();
                     reversed_after_transpose_flags.push_back(true);
                  }
               }
            }
            // arrow end
            if (end_of_merge != 1 + start_of_merge) {
               // merge edge begin
               // result edge and offset
               const auto this_edges_before_merge = &edges_before_merge[start_of_merge];
               auto& this_offset = merge_offsets.emplace_back(std::in_place, nullptr, std::move(this_merge_shape)).value();
               auto& this_offset_pool =
                     merge_offsets_pool.emplace_back(this_offset.size(), std::tuple<Symmetry, Size, Size, Size, Size>{{}, 0, 1, 0, 0});
               this_offset.set_data(this_offset_pool.data());
               // sum symmetry
               for (auto i = 0; i < this_offset.rank(); i++) {
                  Size self_size = this_offset.dimensions(i);
                  Size in_size = this_offset.leadings(i);
                  Size out_size = this_offset.size() == 0 ? 0 : this_offset.size() / (self_size * in_size);
                  for (Size x = 0; x < out_size; x++) {
                     auto offset_for_x = x;
                     for (Size y = 0; y < self_size; y++) {
                        const auto& [symmetry_here, dimension_here] = this_edges_before_merge[i].segments(y);
                        auto offset_for_y = offset_for_x * self_size + y;
                        for (Size z = 0; z < in_size; z++) {
                           auto offset_for_z = offset_for_y * in_size + z;
                           std::get<0>(this_offset_pool[offset_for_z]) += symmetry_here;
                           std::get<2>(this_offset_pool[offset_for_z]) *= dimension_here;
                           std::get<4>(this_offset_pool[offset_for_z]) += symmetry_here.parity();
                        }
                     }
                  }
               }
               // accumulate offset
               auto merged_edge = std::vector<std::pair<Symmetry, Size>>();
               for (auto& [symmetry, position, size, offset, parity] : this_offset_pool) {
                  auto found = std::find_if(merged_edge.begin(), merged_edge.end(), [&sym = symmetry](const auto& pair) {
                     return pair.first == sym;
                  });
                  std::pair<Symmetry, Size>* pointer;
                  if (found == merged_edge.end()) {
                     pointer = &merged_edge.emplace_back(symmetry, 0);
                  } else {
                     pointer = &*found;
                  }
                  position = pointer - merged_edge.data();
                  offset = pointer->second;
                  pointer->second += size;
               }
               result_edges.emplace_back(std::move(merged_edge), arrow);
            } else {
               // not merge for this edge
               const auto& target_edge = edges_before_merge[start_of_merge];
               result_edges.emplace_back(target_edge.segments(), target_edge.arrow());
               merge_offsets.emplace_back();
               // it will check later so this_offset it not calculated
            }
            // merge edge end
            start_of_merge = end_of_merge;
         }
      } else {
         // no merge for all edge
         if constexpr (is_fermi) {
            reversed_after_transpose_flags = pmr::vector<bool>(rank_at_transpose, false);
            // no merge is no reverse before merge
         }
         result_edges.reserve(rank_after_merge);
         for (auto index_after_merge = 0; index_after_merge < rank_after_merge; index_after_merge++) {
            const auto& edge = edges_before_merge[index_after_merge];
            result_edges.emplace_back(edge.segments(), edge.arrow());
            // it will check if no merge, so it will not use offset, so offset is not calculated
            // even emplace_back is not needed
         }
      }

      // put res_edge into res
      auto result = Tensor<ScalarType, Symmetry, Name>(std::move(new_names), std::move(result_edges));
      const auto& edges_after_merge = result.edges();

      // The previous one is moved.
      const auto& names_after_merge_new = result.names();
      const auto& names_before_merge_new = merge_map.size() != 0 ? real_names_before_merge : names_after_merge_new;

      // marks:
      // split_flag_mark
      // reversed_before_transpose_flag_mark
      // reversed_after_transpose_flag_mark
      // merge_flag_mark

      // marks
      auto split_flags_mark = pmr::vector<bool>();
      auto reversed_before_transpose_flags_mark = pmr::vector<bool>();
      auto reversed_after_transpose_flags_mark = pmr::vector<bool>();
      auto merge_flags_mark = pmr::vector<bool>();
      if constexpr (is_fermi) {
         split_flags_mark.reserve(rank_before_split);
         reversed_before_transpose_flags_mark.reserve(rank_at_transpose);
         reversed_after_transpose_flags_mark.reserve(rank_at_transpose);
         merge_flags_mark.reserve(rank_after_merge);
         // true => apply parity
         if (apply_parity) {
            // default apply, so apply == not in exclude, namely, find==end
            for (auto i = 0; i < rank_before_split; i++) {
               split_flags_mark.push_back(parity_exclude_names_split.find(names_before_split[i]) == parity_exclude_names_split.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_before_transpose_flags_mark.push_back(
                     parity_exclude_names_reversed_before_transpose.find(names_after_split[i]) ==
                     parity_exclude_names_reversed_before_transpose.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_after_transpose_flags_mark.push_back(
                     parity_exclude_names_reversed_after_transpose.find(names_before_merge_new[i]) ==
                     parity_exclude_names_reversed_after_transpose.end());
            }
            for (auto i = 0; i < rank_after_merge; i++) {
               merge_flags_mark.push_back(parity_exclude_names_merge.find(names_after_merge_new[i]) == parity_exclude_names_merge.end());
            }
         } else {
            for (auto i = 0; i < rank_before_split; i++) {
               split_flags_mark.push_back(parity_exclude_names_split.find(names_before_split[i]) != parity_exclude_names_split.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_before_transpose_flags_mark.push_back(
                     parity_exclude_names_reversed_before_transpose.find(names_after_split[i]) !=
                     parity_exclude_names_reversed_before_transpose.end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_after_transpose_flags_mark.push_back(
                     parity_exclude_names_reversed_after_transpose.find(names_before_merge_new[i]) !=
                     parity_exclude_names_reversed_after_transpose.end());
            }
            for (auto i = 0; i < rank_after_merge; i++) {
               merge_flags_mark.push_back(parity_exclude_names_merge.find(names_after_merge_new[i]) != parity_exclude_names_merge.end());
            }
         }
      }

      // Main loop
      auto edges_before_merge_shape = pmr::vector<Size>();
      edges_before_merge_shape.reserve(rank_at_transpose);
      for (const auto& edge : edges_before_merge) {
         edges_before_merge_shape.push_back(edge.segments_size());
      }
      // the total symmetry and total_parity
      auto main_loop = mdspan<std::pair<Symmetry, bool>, pmr::vector<Size>>(nullptr, std::move(edges_before_merge_shape));
      auto main_loop_pool = pmr::vector<std::pair<Symmetry, bool>>(main_loop.size());
      main_loop.set_data(main_loop_pool.data());
      for (auto index_before_merge = 0; index_before_merge < main_loop.rank(); index_before_merge++) {
         bool reversed_flag;
         if constexpr (is_fermi) {
            Rank index_after_split = plan_destination_to_source[index_before_merge];
            reversed_flag = (reversed_after_transpose_flags[index_before_merge] && reversed_after_transpose_flags_mark[index_before_merge]) ^
                            (reversed_before_transpose_flags[index_after_split] && reversed_before_transpose_flags_mark[index_after_split]);
         }
         Size self_size = main_loop.dimensions(index_before_merge);
         Size in_size = main_loop.leadings(index_before_merge);
         Size out_size = main_loop.size() == 0 ? 0 : main_loop.size() / (self_size * in_size);
         for (Size x = 0; x < out_size; x++) {
            auto offset_for_x = x;
            for (Size y = 0; y < self_size; y++) {
               const auto& [symmetry_here, dimension_here] = edges_before_merge[index_before_merge].segments(y);
               auto parity_here = symmetry_here.parity() && reversed_flag;
               auto offset_for_y = offset_for_x * self_size + y;
               for (Size z = 0; z < in_size; z++) {
                  auto offset_for_z = offset_for_y * in_size + z;
                  main_loop_pool[offset_for_z].first += symmetry_here;
                  main_loop_pool[offset_for_z].second ^= parity_here;
               }
            }
         }
      }
      if constexpr (is_fermi) {
         for (auto index1_before_merge = 0; index1_before_merge < main_loop.rank(); index1_before_merge++) {
            for (auto index2_before_merge = index1_before_merge + 1; index2_before_merge < main_loop.rank(); index2_before_merge++) {
               if (plan_destination_to_source[index1_before_merge] > plan_destination_to_source[index2_before_merge]) {
                  Size self1_size = main_loop.dimensions(index1_before_merge);
                  Size self2_size = main_loop.dimensions(index2_before_merge);
                  Size self1_leading = main_loop.leadings(index1_before_merge);
                  Size self2_leading = main_loop.leadings(index2_before_merge);
                  Size in_size = self2_leading;
                  Size mid_size = self1_leading == 0 ? 0 : self1_leading / (self2_size * self2_leading);
                  Size out_size = main_loop.size() == 0 ? 0 : main_loop.size() / (self1_size * self1_leading);
                  for (Size a = 0; a < out_size; a++) {
                     auto offset_for_a = a;
                     for (Size b = 0; b < self1_size; b++) {
                        if (edges_before_merge[index1_before_merge].segments(b).first.parity()) {
                           auto offset_for_b = (offset_for_a * self1_size) + b;
                           for (Size c = 0; c < mid_size; c++) {
                              auto offset_for_c = (offset_for_b * mid_size) + c;
                              for (Size d = 0; d < self2_size; d++) {
                                 if (edges_before_merge[index2_before_merge].segments(d).first.parity()) {
                                    auto offset_for_d = (offset_for_c * self2_size) + d;
                                    for (Size e = 0; e < in_size; e++) {
                                       auto offset_for_e = (offset_for_d * in_size) + e;
                                       main_loop_pool[offset_for_e].second ^= true;
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
      // reverse and transpose pairty in the main_loop value
      // merge split parity in the offsets pool
      for (auto it = main_loop.begin(); it.valid; ++it) {
         if (it->first != Symmetry()) {
            continue;
         }
         auto total_parity = it->second;
         const auto positions_before_merge = it.indices;

         auto positions_after_split = pmr::vector<Size>(rank_at_transpose);
         auto dimensions_before_merge = pmr::vector<Size>(rank_at_transpose);
         auto dimensions_after_split = pmr::vector<Size>(rank_at_transpose);
         for (auto index_before_merge = 0; index_before_merge < rank_at_transpose; index_before_merge++) {
            auto position = positions_before_merge[index_before_merge];
            auto index_after_split = plan_destination_to_source[index_before_merge];
            positions_after_split[index_after_split] = position;

            const auto& [symmetry, dimension] = edges_before_merge[index_before_merge].segments(position);
            dimensions_before_merge[index_before_merge] = dimension;
            dimensions_after_split[index_after_split] = dimension;
         }

         auto positions_after_merge = pmr::vector<Size>(rank_after_merge);
         auto offsets_after_merge = pmr::vector<Size>(rank_after_merge);
         for (auto index_after_merge = 0, index_before_merge = 0; index_after_merge < rank_after_merge; index_after_merge++) {
            auto this_merge_begin_index_before_merge = index_before_merge;
            while (index_before_merge < rank_at_transpose && merge_flags[index_before_merge] == index_after_merge) {
               index_before_merge++;
            }
            if (index_before_merge != 1 + this_merge_begin_index_before_merge) {
               // normal merge
               const auto& [symmetry, position, size, offset, parity] =
                     merge_offsets[index_after_merge]->at(&positions_before_merge[this_merge_begin_index_before_merge]);
               positions_after_merge[index_after_merge] = position;
               offsets_after_merge[index_after_merge] = offset;
               total_parity ^= ((parity & 2) != 0) && merge_flags_mark[index_after_merge];
            } else {
               // trivial merge
               positions_after_merge[index_after_merge] = positions_before_merge[index_before_merge - 1];
               offsets_after_merge[index_after_merge] = 0;
            }
         }

         auto positions_before_split = pmr::vector<Size>(rank_before_split);
         auto offsets_before_split = pmr::vector<Size>(rank_before_split);
         for (auto index_before_split = 0, index_after_split = 0; index_before_split < rank_before_split; index_before_split++) {
            auto this_split_begin_index_after_split = index_after_split;
            while (index_after_split < rank_at_transpose && split_flags[index_after_split] == index_before_split) {
               index_after_split++;
            }
            if (index_after_split != 1 + this_split_begin_index_after_split) {
               // normal split
               const auto& [symmetry, position, size, offset, parity] =
                     split_offsets[index_before_split]->at(&positions_after_split[this_split_begin_index_after_split]);
               positions_before_split[index_before_split] = position;
               offsets_before_split[index_before_split] = offset;
               total_parity ^= ((parity & 2) != 0) && split_flags_mark[index_before_split];
            } else {
               // trivial split
               positions_before_split[index_before_split] = positions_after_split[index_after_split - 1];
               offsets_before_split[index_before_split] = 0;
            }
         }

         auto real_positions_before_cut = pmr::vector<Size>();
         if (edges_and_symmetries_to_cut_before_all.size() != 0) {
            real_positions_before_cut.resize(rank_before_split);
            for (auto index_before_split = 0; index_before_split < rank_before_split; index_before_split++) {
               auto symmetry = edges_before_split[index_before_split].segments(positions_before_split[index_before_split]).first;
               const auto& edge = edges(index_before_split);
               real_positions_before_cut[index_before_split] = edge.find_by_symmetry(symmetry) - edge.segments().begin();
            }
         }
         const auto& positions_before_cut = edges_and_symmetries_to_cut_before_all.size() != 0 ? real_positions_before_cut : positions_before_split;

         auto leadings_after_merge = pmr::vector<Size>(rank_after_merge);
         for (auto index_after_merge = rank_after_merge; index_after_merge-- > 0;) {
            if (index_after_merge == rank_after_merge - 1) {
               leadings_after_merge[index_after_merge] = 1;
            } else {
               leadings_after_merge[index_after_merge] =
                     leadings_after_merge[index_after_merge + 1] *
                     edges_after_merge[index_after_merge + 1].segments(positions_after_merge[index_after_merge + 1]).second;
            }
         }
         auto leadings_before_merge = pmr::vector<Size>(rank_at_transpose);
         for (auto index_before_merge = rank_at_transpose; index_before_merge-- > 0;) {
            if (index_before_merge != rank_at_transpose - 1 && merge_flags[index_before_merge] == merge_flags[index_before_merge + 1]) {
               leadings_before_merge[index_before_merge] =
                     leadings_before_merge[index_before_merge + 1] * dimensions_before_merge[index_before_merge + 1];
            } else {
               leadings_before_merge[index_before_merge] = leadings_after_merge[merge_flags[index_before_merge]];
            }
         }
         auto leadings_before_split = pmr::vector<Size>(rank_before_split);
         for (auto index_before_split = rank_before_split; index_before_split-- > 0;) {
            if (index_before_split == rank_before_split - 1) {
               leadings_before_split[index_before_split] = 1;
            } else {
               leadings_before_split[index_before_split] =
                     leadings_before_split[index_before_split + 1] *
                     edges(index_before_split + 1).segments(positions_before_cut[index_before_split + 1]).second;
               // Use the original dimension before cut, because it is about leading.
            }
         }
         auto leadings_after_split = pmr::vector<Size>(rank_at_transpose);
         for (auto index_after_split = rank_at_transpose; index_after_split-- > 0;) {
            if (index_after_split != rank_at_transpose - 1 && split_flags[index_after_split] == split_flags[index_after_split + 1]) {
               leadings_after_split[index_after_split] = leadings_after_split[index_after_split + 1] * dimensions_after_split[index_after_split + 1];
            } else {
               leadings_after_split[index_after_split] = leadings_before_split[split_flags[index_after_split]];
            }
         }

         auto source_span = mdspan<const ScalarType, pmr::vector<Size>>(
               // The whole segment of some symmetry maybe removed directly, so positions is not usable.
               &blocks().at(positions_before_cut).value().at(offsets_before_split),
               std::move(dimensions_after_split),
               std::move(leadings_after_split));
         auto destination_span = mdspan<ScalarType, pmr::vector<Size>>(
               &result.blocks().at(positions_after_merge).value().at(offsets_after_merge),
               std::move(dimensions_before_merge),
               std::move(leadings_before_merge));

         if (total_parity) {
            mdspan_transform(source_span.transpose(plan_destination_to_source), destination_span, [](const auto& x) {
               return -x;
            });
         } else {
            mdspan_transform(source_span.transpose(plan_destination_to_source), destination_span, [](const auto& x) {
               return x;
            });
         }
      }

      return result;
   }
} // namespace TAT
#endif
