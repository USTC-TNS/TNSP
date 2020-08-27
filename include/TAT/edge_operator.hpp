/**
 * \file edge_operator.hpp
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
#ifndef TAT_EDGE_OPERATOR_HPP
#define TAT_EDGE_OPERATOR_HPP

#include "tensor.hpp"
#include "transpose.hpp"

namespace TAT {
   template<class ScalarType, class Symmetry>
   template<class T, class>
   [[nodiscard]] Tensor<ScalarType, Symmetry> Tensor<ScalarType, Symmetry>::edge_operator(
         const std::map<Name, Name>& rename_map,
         const std::map<Name, std::vector<std::tuple<Name, BoseEdge<Symmetry>>>>& split_map,
         const std::set<Name>& reversed_name,
         const std::map<Name, std::vector<Name>>& merge_map,
         T&& new_names,
         const bool apply_parity,
         const std::array<std::set<Name>, 4>& parity_exclude_name,
         const std::map<Name, std::map<Symmetry, Size>>& edge_and_symmetries_to_cut_before_all) const {
      // step 1: rename
      // step 2: split
      // step 3: reverse
      // step 4: transpose
      // step 5: reverse before merge
      // step 6: merge
      // 前面三步是因果顺序往下推, 第四步转置需要根据merge和最后的name来确定转置方案
      // 除了第一步, 剩下的步骤沿着transpose对称

      // 先写一个无任何优化的版本

      // parity_exclude_name的四部分分别是split, reverse, reverse_for_merge, merge
      // name 均为rename后的名称，split取split前， merge取merge后

      // is_fermi
      constexpr auto is_fermi = is_fermi_symmetry_v<Symmetry>;

      // 1. 先将rank, name, edge信息生成出来
      //
      // 1.1 先弄出来 status 0和status 1, 他们在关于transpose对称的操作之前
      //
      // 1.1.1 status 0
      // status 0 origin
      // rank_0 and name_0
      // create edge 0

      // 1.1.2 status 1
      // status 1 rename
      // rank_1 and edge_1
      const Rank rank_before_split = names.size();
      // create name_1
      auto name_before_split = std::vector<Name>(); // length = rank_before_split
      for (Rank i = 0; i < rank_before_split; i++) {
         auto name = names[i];
         if (auto position = rename_map.find(name); position != rename_map.end()) {
            name_before_split.push_back(position->second);
         } else {
            name_before_split.push_back(name);
         }
      }

      // 1.2 检查是否不需要做任何操作
      // check no change
      if (name_before_split == new_names && reversed_name.empty() && split_map.empty() && merge_map.empty()) {
         // share the core
         auto result = Tensor<ScalarType, Symmetry>();
         result.names = std::forward<T>(name_before_split);
         result.name_to_index = construct_name_to_index(result.names);
         result.core = core;
         // check_valid_name(result.names, result.core->edges.size());
         return result;
      }

      auto edge_before_split = std::vector<Edge<Symmetry>>();
      for (auto i = 0; i < rank_before_split; i++) {
         // 可能的cut, 这里应该是rename前的名称
         if (auto found = edge_and_symmetries_to_cut_before_all.find(names[i]); found != edge_and_symmetries_to_cut_before_all.end()) {
            const auto& symmetry_to_cut_dimension = found->second;
            auto& this_edge = edge_before_split.emplace_back();
            if constexpr (is_fermi) {
               this_edge.arrow = core->edges[i].arrow;
            }
            for (const auto& [symmetry, dimension] : core->edges[i].map) {
               if (auto cut_iterator = symmetry_to_cut_dimension.find(symmetry); cut_iterator != symmetry_to_cut_dimension.end()) {
                  if (auto new_dimension = cut_iterator->second; new_dimension != 0) {
                     // auto new_dimension = cut_iterator->second;
                     this_edge.map[symmetry] = new_dimension < dimension ? new_dimension : dimension;
                  }
                  // 如果new_dimension=0则直接去掉了这个symmetry
                  // 这个会影响leading, 故只需修改原来算offset处的代码即可, 替换回core->edges
               } else {
                  this_edge.map[symmetry] = dimension;
               }
            }
         } else {
            edge_before_split.push_back(core->edges[i]);
         }
      }

      // 1.3 对称的几个操作中transpose之前的 rank, name, edge
      // status 2 split
      // create name_2 and edge_2 and split_flag
      auto split_flag = std::vector<Rank>();
      auto split_offset = std::vector<std::map<std::vector<Symmetry>, std::tuple<Symmetry, Size>>>();
      auto name_after_split = std::vector<Name>();
      auto edge_after_split = std::vector<EdgePointer<Symmetry>>();
      for (Rank position_before_split = 0, total_split_index = 0; position_before_split < rank_before_split; position_before_split++) {
         if (auto position = split_map.find(name_before_split[position_before_split]); position != split_map.end()) {
            const auto& this_split_begin_position_in_edge_after_split = edge_after_split.size();
            for (const auto& [split_name, split_edge] : position->second) {
               name_after_split.push_back(split_name);
               if constexpr (is_fermi) {
                  edge_after_split.push_back({edge_before_split[position_before_split].arrow, split_edge.map});
               } else {
                  edge_after_split.push_back({split_edge.map});
               }
               split_flag.push_back(total_split_index);
            }
            const auto edge_list_after_split = edge_after_split.data() + this_split_begin_position_in_edge_after_split;
            const auto split_rank = edge_after_split.size() - this_split_begin_position_in_edge_after_split;
            // loop between begin and end, get a map push_Back into split_offset
            // this map is sym -> [sym] -> offset
            auto& this_offset = split_offset.emplace_back();
            auto offset_bank = std::map<Symmetry, Size>();
            for (const auto& [sym, dim] : edge_before_split[position_before_split].map) {
               // 只需要symmetry信息, edge_before_split和core->edge一样
               offset_bank[sym] = 0;
            }
            auto accumulated_symmetries = std::vector<Symmetry>(split_rank);
            auto accumulated_dimensions = std::vector<Size>(split_rank);
            auto current_symmetries = std::vector<Symmetry>(split_rank);
            loop_edge(
                  edge_list_after_split,
                  split_rank,
                  [&this_offset]() {
                     this_offset[std::vector<Symmetry>{}] = {Symmetry(), 0};
                  },
                  []() {},
                  [&](const MapIteratorList& symmetry_iterator_list, Rank minimum_changed) {
                     for (auto i = minimum_changed; i < split_rank; i++) {
                        const auto& symmetry_iterator = symmetry_iterator_list[i];
                        accumulated_symmetries[i] = symmetry_iterator->first + (i ? accumulated_symmetries[i - 1] : Symmetry());
                        accumulated_dimensions[i] = symmetry_iterator->second * (i ? accumulated_dimensions[i - 1] : 1);
                        // do not check dim=0, because in constructor, it didn't check
                        current_symmetries[i] = symmetry_iterator->first;
                     }
                     auto target_symmetry = accumulated_symmetries.back();
                     auto target_dimension = accumulated_dimensions.back();
                     // 这是可能不存在的, 不存在的情况是因为此target_symmetry没有有效block
                     // split后的各个symmetry指向了一个不存在的split前symmetry
                     if (auto found = offset_bank.find(target_symmetry); found != offset_bank.end()) {
                        this_offset[current_symmetries] = {target_symmetry, found->second};
                        found->second += target_dimension;
                     }
                     return split_rank;
                  });
            total_split_index++;
         } else {
            name_after_split.push_back(name_before_split[position_before_split]);
            if constexpr (is_fermi) {
               edge_after_split.push_back({edge_before_split[position_before_split].arrow, edge_before_split[position_before_split].map});
            } else {
               edge_after_split.push_back({edge_before_split[position_before_split].map});
            }
            split_flag.push_back(total_split_index++);
            auto& this_offset = split_offset.emplace_back();
            for (const auto& [symmetry, dimension] : edge_before_split[position_before_split].map) {
               this_offset[{symmetry}] = {symmetry, 0};
            }
         }
      }
      // rank_2
      const Rank rank_at_transpose = name_after_split.size();

      // status 3 reverse
      // rank_3 and name_3
      // create reversed_flag_src and edge_3
      auto reversed_before_transpose_flag = std::vector<bool>(); // length = rank_at_transpose
      auto fermi_edge_before_transpose = std::vector<EdgePointer<Symmetry>>();
      if constexpr (is_fermi) {
         for (auto i = 0; i < rank_at_transpose; i++) {
            fermi_edge_before_transpose.push_back(edge_after_split[i]);
            if (reversed_name.find(name_after_split[i]) != reversed_name.end()) {
               fermi_edge_before_transpose.back().arrow ^= true;
               reversed_before_transpose_flag.push_back(true);
            } else {
               reversed_before_transpose_flag.push_back(false);
            }
         }
      }
      const auto& edge_before_transpose = is_fermi ? fermi_edge_before_transpose : edge_after_split;

      // create res names
      auto result = Tensor<ScalarType, Symmetry>();
      result.names = std::forward<T>(new_names);
      result.name_to_index = construct_name_to_index(result.names);

      // 1.4 transpose之后的rank, name

      // status 4 transpose and status 6 merge and status 5 reverse before merge
      // 逆序确定name, 再顺序确定edge和data
      // name and rank
      // name_6 and rank_6
      const auto& name_after_merge = result.names;
      const Rank rank_after_merge = name_after_merge.size();
      // create merge_flag and name_5
      auto merge_flag = std::vector<Rank>();
      auto name_before_merge = std::vector<Name>();
      for (Rank position_after_merge = 0, total_merge_index = 0; position_after_merge < rank_after_merge; position_after_merge++) {
         const auto& merged_name = name_after_merge[position_after_merge];
         if (auto position = merge_map.find(merged_name); position != merge_map.end()) {
            for (const auto& merging_names : position->second) {
               name_before_merge.push_back(merging_names);
               merge_flag.push_back(total_merge_index);
            }
            total_merge_index++;
         } else {
            name_before_merge.push_back(merged_name);
            merge_flag.push_back(total_merge_index++);
         }
      }
      // rank_4 and name_5 and rank_5
      // name build by merge may contain some edge not exist
      if (rank_at_transpose != name_before_merge.size()) {
         warning_or_error("Different Rank When Transpose");
      }

      // 1.5 转置方案
      // to be easy, create name_to_index for name_3
      auto name_to_index_after_split = construct_name_to_index(name_after_split);
      // create plan of two way
      auto plan_source_to_destination = std::vector<Rank>(rank_at_transpose);
      auto plan_destination_to_source = std::vector<Rank>(rank_at_transpose);

      // edge
      // create edge_4
      auto edge_after_transpose = std::vector<EdgePointer<Symmetry>>();
      for (auto i = 0; i < rank_at_transpose; i++) {
         if (auto found = name_to_index_after_split.find(name_before_merge[i]); found != name_to_index_after_split.end()) {
            plan_destination_to_source[i] = found->second;
         } else {
            warning_or_error("Different Name When Transpose");
         }
         plan_source_to_destination[plan_destination_to_source[i]] = i;
         edge_after_transpose.push_back(edge_before_transpose[plan_destination_to_source[i]]);
      }
      // 1.6 考虑转置后的edge
      // reverse 和 merge 需要放在一起考虑, 如果fermi, 那么reverse的edge在前一个edge基础上修改, 否则直接是上一个引用

      // the following code is about merge
      // dealing with edge_5 and res_edge and reversed_flag_dst

      // prepare edge_5
      // if no merge, edge_5 is reference of edge_4, else is copy of edge_4
      auto fermi_edge_before_merge = std::vector<EdgePointer<Symmetry>>();
      if constexpr (is_fermi) {
         for (const auto& edge : edge_after_transpose) {
            fermi_edge_before_merge.push_back(edge);
         }
      }
      auto& edge_before_merge = is_fermi ? fermi_edge_before_merge : edge_after_transpose;

      // prepare res_edge
      // res_edge means edge_6 but type is different
      auto result_edge = std::vector<Edge<Symmetry>>();
      // prepare reversed_flag_dst
      auto reversed_after_transpose_flag = std::vector<bool>();

      auto merge_offset = std::vector<std::map<std::vector<Symmetry>, std::tuple<Symmetry, Size>>>();

      for (Rank position_after_merge = 0, start_of_merge = 0, end_of_merge = 0; position_after_merge < rank_after_merge; position_after_merge++) {
         // [start, end) need be merged
         while (end_of_merge < rank_at_transpose && merge_flag[end_of_merge] == position_after_merge) {
            end_of_merge++;
         }
         // arrow begin
         Arrow arrow;
         bool arrow_fixed = false;
         for (auto merge_group_position = start_of_merge; merge_group_position < end_of_merge; merge_group_position++) {
            if constexpr (is_fermi) {
               if (edge_before_merge[merge_group_position].arrow_valid()) {
                  if (arrow_fixed) {
                     if (arrow == edge_before_merge[merge_group_position].arrow) {
                        reversed_after_transpose_flag.push_back(false);
                     } else {
                        edge_before_merge[merge_group_position].arrow ^= true;
                        reversed_after_transpose_flag.push_back(true);
                     }
                  } else {
                     arrow_fixed = true;
                     arrow = edge_before_merge[merge_group_position].arrow;
                     reversed_after_transpose_flag.push_back(false);
                  }
               } else {
                  reversed_after_transpose_flag.push_back(false);
               }
            }
         }
         // arrow end

         // merge edge begin
         auto& merged_edge = result_edge.emplace_back();
         auto& this_offset = merge_offset.emplace_back();

         const Rank merge_rank = end_of_merge - start_of_merge;
         auto accumulated_symmetries = std::vector<Symmetry>(merge_rank);
         auto accumulated_dimensions = std::vector<Size>(merge_rank);
         auto current_symmetries = std::vector<Symmetry>(merge_rank);

         loop_edge(
               edge_before_merge.data() + start_of_merge,
               merge_rank,
               [&merged_edge, &this_offset]() {
                  merged_edge.map[Symmetry()] = 1;
                  this_offset[std::vector<Symmetry>{}] = {Symmetry(), 0};
               },
               []() {},
               [&](const MapIteratorList& symmetry_iterator_list, Rank minimum_changed) {
                  for (auto i = minimum_changed; i < merge_rank; i++) {
                     const auto& symmetry_iterator = symmetry_iterator_list[i];
                     accumulated_symmetries[i] = symmetry_iterator->first + (i ? accumulated_symmetries[i - 1] : Symmetry());
                     accumulated_dimensions[i] = symmetry_iterator->second * (i ? accumulated_dimensions[i - 1] : 1);
                     // do not check dim=0, because in constructor, i didn't check
                     current_symmetries[i] = symmetry_iterator->first;
                  }
                  auto target_symmetry = accumulated_symmetries.back();
                  this_offset[current_symmetries] = {target_symmetry, merged_edge.map[target_symmetry]};
                  merged_edge.map[target_symmetry] += accumulated_dimensions.back();
                  return merge_rank;
               });
         // merge edge end

         if constexpr (is_fermi) {
            merged_edge.arrow = arrow;
         }
         start_of_merge = end_of_merge;
      }
      // the code above is dealing with edge_5 and res_Edge and reversed_flag_dst

      // put res_edge into res
      result.core = std::make_shared<Core<ScalarType, Symmetry>>(std::move(result_edge));
      check_valid_name(result.names, result.core->edges.size());
      // edge_6
      const auto& edge_after_merge = result.core->edges;
      // 2. 开始分析data如何移动

      auto data_before_transpose_to_source = std::map<std::vector<Symmetry>, std::tuple<std::vector<Symmetry>, std::vector<Size>>>();
      for (auto& [symmetries_before_transpose, size] : initialize_block_symmetries_with_check(edge_after_split)) {
         // convert sym -> target_sym and offsets
         // and add to map
         auto symmetries = std::vector<Symmetry>();
         auto offsets = std::vector<Size>();
         bool success = true;
         for (Rank position_before_split = 0, position_after_split = 0; position_before_split < rank_before_split; position_before_split++) {
            // [start, end) be merged
            auto split_group_symmetries = std::vector<Symmetry>();
            while (position_after_split < rank_at_transpose && split_flag[position_after_split] == position_before_split) {
               split_group_symmetries.push_back(symmetries_before_transpose[position_after_split]);
               position_after_split++;
            }
            if (auto found = split_offset[position_before_split].find(split_group_symmetries); found != split_offset[position_before_split].end()) {
               const auto& [this_symmetry, this_offset] = found->second;
               symmetries.push_back(this_symmetry);
               offsets.push_back(this_offset);
            } else {
               success = false;
               break;
            }
         }
         if (success) {
            data_before_transpose_to_source[symmetries_before_transpose] = {std::move(symmetries), std::move(offsets)};
         }
      }
      auto data_after_transpose_to_destination = std::map<std::vector<Symmetry>, std::tuple<std::vector<Symmetry>, std::vector<Size>>>();
      for (auto& [symmetries_after_transpose, size] : initialize_block_symmetries_with_check(edge_before_merge)) {
         // convert sym -> target_sym and offsets
         // and add to map
         auto symmetries = std::vector<Symmetry>();
         auto offsets = std::vector<Size>();
         bool success = true;
         for (Rank position_after_merge = 0, position_before_merge = 0; position_after_merge < rank_after_merge; position_after_merge++) {
            // [start, end) be merged
            auto merge_group_symmetries = std::vector<Symmetry>();
            while (position_before_merge < rank_at_transpose && merge_flag[position_before_merge] == position_after_merge) {
               merge_group_symmetries.push_back(symmetries_after_transpose[position_before_merge]);
               position_before_merge++;
            }
            if (auto found = merge_offset[position_after_merge].find(merge_group_symmetries); found != merge_offset[position_after_merge].end()) {
               const auto& [this_symmetry, this_offset] = found->second;
               symmetries.push_back(this_symmetry);
               offsets.push_back(this_offset);
            } else {
               success = false;
               break;
            }
         }
         if (success) {
            data_after_transpose_to_destination[symmetries_after_transpose] = {std::move(symmetries), std::move(offsets)};
         }
      }

      // 3. 4 marks
      auto split_flag_mark = std::vector<bool>();
      auto reversed_before_transpose_flag_mark = std::vector<bool>();
      auto reversed_after_transpose_flag_mark = std::vector<bool>();
      auto merge_flag_mark = std::vector<bool>();
      if constexpr (is_fermi) {
         // true => 应用parity
         if (apply_parity) {
            // 默认应用, 故应用需不在exclude中, 即find==end
            for (auto i = 0; i < rank_before_split; i++) {
               split_flag_mark.push_back(parity_exclude_name[0].find(name_before_split[i]) == parity_exclude_name[0].end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_before_transpose_flag_mark.push_back(parity_exclude_name[1].find(name_after_split[i]) == parity_exclude_name[1].end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_after_transpose_flag_mark.push_back(parity_exclude_name[2].find(name_before_merge[i]) == parity_exclude_name[2].end());
            }
            for (auto i = 0; i < rank_after_merge; i++) {
               merge_flag_mark.push_back(parity_exclude_name[3].find(name_after_merge[i]) == parity_exclude_name[3].end());
            }
         } else {
            for (auto i = 0; i < rank_before_split; i++) {
               split_flag_mark.push_back(parity_exclude_name[0].find(name_before_split[i]) != parity_exclude_name[0].end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_before_transpose_flag_mark.push_back(parity_exclude_name[1].find(name_after_split[i]) != parity_exclude_name[1].end());
            }
            for (auto i = 0; i < rank_at_transpose; i++) {
               reversed_after_transpose_flag_mark.push_back(parity_exclude_name[2].find(name_before_merge[i]) != parity_exclude_name[2].end());
            }
            for (auto i = 0; i < rank_after_merge; i++) {
               merge_flag_mark.push_back(parity_exclude_name[3].find(name_after_merge[i]) != parity_exclude_name[3].end());
            }
         }
      }
      // 可能会产生空的无源分块
      result.zero();
      // 缩并时在这里产生无源分块是很正常的事, 这正是对称性ansatz所假设的, 每个张量都是对称守恒的
      // 但是缩并后, 他便可以表示两个对称性不守恒的张量的乘积, 或者说缩并后表达能力变强了, 所以会产生多余的0
      // 5. main copy loop
      for (const auto& [symmetries_before_transpose, source_symmetries_and_offsets] : data_before_transpose_to_source) {
         const auto& [source_symmetries, source_offsets] = source_symmetries_and_offsets;

         auto symmetries_after_transpose = std::vector<Symmetry>(rank_at_transpose);
         auto dimensions_before_transpose = std::vector<Size>(rank_at_transpose);
         auto dimensions_after_transpose = std::vector<Size>(rank_at_transpose);
         Size total_size = 1;
         for (auto i = 0; i < rank_at_transpose; i++) {
            auto dimension = edge_before_transpose[i].map.at(symmetries_before_transpose[i]);
            dimensions_before_transpose[i] = dimension;
            dimensions_after_transpose[plan_source_to_destination[i]] = dimension;
            symmetries_after_transpose[plan_source_to_destination[i]] = symmetries_before_transpose[i];
            total_size *= dimension;
         }

         const auto& [destination_symmetries, destination_offsets] = data_after_transpose_to_destination.at(symmetries_after_transpose);

         const auto& source_block = core->blocks.at(source_symmetries);
         auto& destination_block = result.core->blocks.at(destination_symmetries);

         Size total_source_offset = 0;
         for (auto i = 0; i < rank_before_split; i++) {
            // 这里将edge_before_split换为core->edges
            total_source_offset *= core->edges[i].map.at(source_symmetries[i]);
            total_source_offset += source_offsets[i];
         }
         Size total_destination_offset = 0;
         for (auto i = 0; i < rank_after_merge; i++) {
            total_destination_offset *= edge_after_merge[i].map.at(destination_symmetries[i]);
            total_destination_offset += destination_offsets[i];
         }

         auto leading_of_source = std::vector<Size>(rank_before_split);
         for (auto i = rank_before_split; i-- > 0;) {
            if (i == rank_before_split - 1) {
               leading_of_source[i] = 1;
            } else {
               // 这里将edge_before_split换为core->edges
               leading_of_source[i] = leading_of_source[i + 1] * core->edges[i + 1].map.at(source_symmetries[i + 1]);
            }
         }
         auto leading_before_transpose = std::vector<Size>(rank_at_transpose);
         for (auto i = rank_at_transpose; i-- > 0;) {
            if (i != rank_at_transpose - 1 && split_flag[i] == split_flag[i + 1]) {
               leading_before_transpose[i] = leading_before_transpose[i + 1] * dimensions_before_transpose[i + 1];
               // dimensions_before_transpose[i + 1] == edge_before_transpose[i + 1].map.at(symmetries_before_transpose[i + 1]);
            } else {
               leading_before_transpose[i] = leading_of_source[split_flag[i]];
            }
         }

         auto leading_of_destination = std::vector<Size>(rank_after_merge);
         for (auto i = rank_after_merge; i-- > 0;) {
            if (i == rank_after_merge - 1) {
               leading_of_destination[i] = 1;
            } else {
               leading_of_destination[i] = leading_of_destination[i + 1] * edge_after_merge[i + 1].map.at(destination_symmetries[i + 1]);
            }
         }
         auto leading_after_transpose = std::vector<Size>(rank_at_transpose);
         for (auto i = rank_at_transpose; i-- > 0;) {
            if (i != rank_at_transpose - 1 && merge_flag[i] == merge_flag[i + 1]) {
               leading_after_transpose[i] = leading_after_transpose[i + 1] * dimensions_after_transpose[i + 1];
               // dimensions_after_transpose[i + 1] == edge_after_transpose[i + 1].map.at(symmetries_after_transpose[i + 1]);
            } else {
               leading_after_transpose[i] = leading_of_destination[merge_flag[i]];
            }
         }

         // parity
         auto parity = false;
         if constexpr (is_fermi) {
            parity = Symmetry::get_transpose_parity(symmetries_before_transpose, plan_source_to_destination);

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
               leading_before_transpose,
               leading_after_transpose,
               rank_at_transpose,
               total_size,
               parity);
      }

      return result;
   }
} // namespace TAT
#endif
