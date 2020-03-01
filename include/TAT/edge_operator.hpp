/**
 * \file edge_operator.hpp
 *
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include "tensor.hpp"
#include "transpose.hpp"

namespace TAT {
   template<class ScalarType, class Symmetry>
   template<class T, class>
   [[nodiscard]] Tensor<ScalarType, Symmetry> Tensor<ScalarType, Symmetry>::edge_operator(
         const std::map<Name, Name>& rename_map,
         const std::map<Name, vector<std::tuple<Name, BoseEdge<Symmetry>>>>& split_map,
         const std::set<Name>& reversed_name,
         const std::map<Name, vector<Name>>& merge_map,
         T&& new_names,
         const bool apply_parity,
         const std::array<std::set<Name>, 4>& parity_exclude_name) const {
      // step 1: rename
      // step 2: split
      // step 3: reverse
      // step 4: transpose
      // step 5: reverse before merge
      // step 6: merge
      // 前面三步是因果顺序往下推, 第四步转置需要根据merge和最后的name来确定转置方案

      // parity_exclude_name的四部分分别是split, reverse, reverse_for_merge, merge
      // name 均为rename后的名称，split取split前， merge取merge后

      // is_fermi
      constexpr auto is_fermi = is_fermi_symmetry_v<Symmetry>;
      // status 0 origin
      // rank_0 and name_0
      const auto rank_0 = names.size();
      const auto& name_0 = names;
      // create edge 0
      auto edge_0 = vector<PtrEdge<Symmetry>>();
      for (const auto& e : core->edges) {
         if constexpr (is_fermi) {
            edge_0.push_back({e.arrow, &e.map});
         } else {
            edge_0.push_back({&e.map});
         }
      }
      // create data_0
      auto data_0 = std::map<vector<Symmetry>, const ScalarType*>();
      for (const auto& [sym, vec] : core->blocks) {
         data_0[sym] = vec.data();
      }

      // status 1 rename
      // rank_1 and edge_1 and data_1
      const auto& rank_1 = rank_0;
      const auto& edge_1 = edge_0;
      const auto& data_1 = data_0;
      // create name_1
      auto true_name_1 = vector<Name>();
      auto ptr_name_1 = &name_0;
      if (!rename_map.empty()) {
         true_name_1.resize(rank_1);
         for (auto i = 0; i < rank_1; i++) {
            auto name = name_0[i];
            auto pos = rename_map.find(name);
            if (pos != rename_map.end()) {
               name = pos->second;
            }
            true_name_1[i] = name;
         }
         ptr_name_1 = &true_name_1;
      }
      const auto& name_1 = *ptr_name_1;

      // check no change
      if (name_1 == new_names && reversed_name.empty() && split_map.empty() && merge_map.empty()) {
         // share the core
         auto res = Tensor<ScalarType, Symmetry>();
         res.names = std::forward<T>(new_names);
         res.name_to_index = construct_name_to_index(res.names);
         res.core = core;
         return res;
      }

      // status 2 split
      // create name_2 and edge_2 and split_flag
      auto split_flag = vector<Rank>();
      auto true_name_2 = vector<Name>();
      auto true_edge_2 = vector<PtrEdge<Symmetry>>();
      auto ptr_name_2 = &name_1;
      auto ptr_edge_2 = &edge_1;
      if (!split_map.empty()) {
         Rank total_split_index = 0;
         for (auto i = 0; i < rank_1; i++) {
            auto pos = split_map.find(name_1[i]);
            if (pos != split_map.end()) {
               for (const auto& [n, e] : pos->second) {
                  true_name_2.push_back(n);
                  if constexpr (is_fermi) {
                     true_edge_2.push_back({edge_1[i].arrow, &e.map});
                  } else {
                     true_edge_2.push_back({&e.map});
                  }
                  split_flag.push_back(total_split_index);
               }
               total_split_index++;
            } else {
               true_name_2.push_back(name_1[i]);
               true_edge_2.push_back(edge_1[i]);
               split_flag.push_back(total_split_index++);
            }
         }
         ptr_name_2 = &true_name_2;
         ptr_edge_2 = &true_edge_2;
      } else {
         split_flag.resize(rank_1);
         for (auto i = 0; i < rank_1; i++) {
            split_flag[i] = i;
         }
      }
      const auto& name_2 = *ptr_name_2;
      const auto& edge_2 = *ptr_edge_2;
      // rank_2
      const auto rank_2 = name_2.size();
      // create data_2
      auto true_data_2 = std::map<vector<Symmetry>, const ScalarType*>();
      auto ptr_data_2 = &data_1;
      if (!split_map.empty()) {
         auto offset_src = data_1;
         auto symmetries_src = initialize_block_symmetries_with_check(edge_2);
         for (auto& [sym, size] : symmetries_src) {
            auto target_symmetry = vector<Symmetry>(rank_1);
            for (auto i = 0; i < rank_1; i++) {
               target_symmetry[i] = Symmetry();
            }
            for (auto i = 0; i < rank_2; i++) {
               target_symmetry[split_flag[i]] += sym[i];
            }
            true_data_2[std::move(sym)] = offset_src[target_symmetry];
            offset_src[std::move(target_symmetry)] += size;
         }
         ptr_data_2 = &true_data_2;
      }
      const auto& data_2 = *ptr_data_2;

      // status 3 reverse
      // rank_3 and name_3 and data_3
      const auto& rank_3 = rank_2;
      const auto& name_3 = name_2;
      const auto& data_3 = data_2;
      // create reversed_flag_src and edge_3
      auto reversed_flag_src = vector<bool>();
      auto ptr_edge_3 = &edge_2;
      auto true_edge_3 = vector<PtrEdge<Symmetry>>();
      if constexpr (is_fermi) {
         if (!reversed_name.empty()) {
            reversed_flag_src.resize(rank_3);
            true_edge_3 = edge_2;
            for (auto i = 0; i < rank_3; i++) {
               if (reversed_name.find(name_3[i]) != reversed_name.end()) {
                  true_edge_3[i].arrow ^= true;
                  reversed_flag_src[i] = true;
               } else {
                  reversed_flag_src[i] = false;
               }
            }
            ptr_edge_3 = &true_edge_3;
         } else {
            reversed_flag_src.resize(rank_3);
            for (auto i = 0; i < rank_3; i++) {
               reversed_flag_src[i] = false;
            }
         }
      }
      const auto& edge_3 = *ptr_edge_3;

      // create res names
      auto res = Tensor<ScalarType, Symmetry>();
      res.names = std::forward<T>(new_names);
      res.name_to_index = construct_name_to_index(res.names);
      // status 4 transpose and status 6 merge and status 5 reverse before merge
      // 逆序确定name, 再顺序确定edge和data
      // name and rank
      // name_6 and rank_6
      const auto& name_6 = res.names;
      const auto& rank_6 = name_6.size();
      // create merge_flag and name_4
      auto merge_flag = vector<Rank>();
      auto true_name_4 = vector<Name>();
      auto ptr_name_4 = &name_6;
      if (!merge_map.empty()) {
         Rank total_merge_index = 0;
         for (const auto& n : name_6) {
            auto pos = merge_map.find(n);
            if (pos != merge_map.end()) {
               for (const auto& i : pos->second) {
                  true_name_4.push_back(i);
                  merge_flag.push_back(total_merge_index);
               }
               total_merge_index++;
            } else {
               true_name_4.push_back(n);
               merge_flag.push_back(total_merge_index++);
            }
         }
         ptr_name_4 = &true_name_4;
      } else {
         merge_flag.resize(rank_6);
         for (auto i = 0; i < rank_6; i++) {
            merge_flag[i] = i;
         }
      }
      const auto& name_4 = *ptr_name_4;
      // rank_4 and name_5 and rank_5
      const auto rank_4 = name_4.size();
      const auto& name_5 = name_4;
      const auto& rank_5 = rank_4;

      // to be easy, create name_to_index for name_3
      auto name_to_index_3 = construct_name_to_index(name_3);
      // create plan of two way
      auto plan_src_to_dst = vector<Rank>(rank_4);
      auto plan_dst_to_src = vector<Rank>(rank_4);
      // edge
      // create edge_4
      auto edge_4 = vector<PtrEdge<Symmetry>>(rank_4);
      for (auto i = 0; i < rank_4; i++) {
         plan_dst_to_src[i] = name_to_index_3.at(name_4[i]);
         plan_src_to_dst[plan_dst_to_src[i]] = i;
         edge_4[i] = edge_3[plan_dst_to_src[i]];
      }

      // the following code is about merge
      // dealing with edge_5 and res_edge and reversed_flag_dst

      // prepare edge_5
      // if no merge, edge_5 is reference of edge_4, else is copy of edge_4
      auto ptr_edge_5 = &edge_4;
      auto true_edge_5 = vector<PtrEdge<Symmetry>>();
      if (!merge_map.empty()) {
         if constexpr (is_fermi) {
            true_edge_5 = edge_4;
            ptr_edge_5 = &true_edge_5;
         }
      }
      auto& edge_5 = *ptr_edge_5;
      // prepare res_edge
      // res_edge means edge_6 but type is different
      auto res_edge = vector<Edge<Symmetry>>();
      // prepare reversed_flag_dst
      auto reversed_flag_dst = vector<bool>();
      if (!merge_map.empty()) {
         // edge_5 is copy of edge_4 and here is will change
         auto start_of_merge = 0;
         auto end_of_merge = 0;
         for (auto i = 0; i < rank_6; i++) {
            // [start, end) need be merged
            while (end_of_merge < rank_4 && merge_flag[end_of_merge] == i) {
               end_of_merge++;
            }
            Arrow arrow;
            bool arrow_fixed = false;
            auto edge_to_merge = vector<PtrEdge<Symmetry>>();
            for (auto j = start_of_merge; j < end_of_merge; j++) {
               if constexpr (is_fermi) {
                  if (edge_5[j].arrow_valid()) {
                     if (arrow_fixed) {
                        if (arrow == edge_5[j].arrow) {
                           reversed_flag_dst.push_back(false);
                        } else {
                           edge_5[j].arrow ^= true;
                           reversed_flag_dst.push_back(true);
                        }
                     } else {
                        arrow_fixed = true;
                        arrow = edge_5[j].arrow;
                        reversed_flag_dst.push_back(false);
                     }
                  } else {
                     reversed_flag_dst.push_back(false);
                  }
               }
               edge_to_merge.push_back(edge_5[j]);
            }
            auto merged_edge = get_merged_edge(edge_to_merge);
            if constexpr (is_fermi) {
               merged_edge.arrow = arrow;
            }
            res_edge.push_back(std::move(merged_edge));
            start_of_merge = end_of_merge;
         }
      } else {
         // if no merge, res_edge is just edge_5, and reversed_flag_dst is all false
         // edge_5 is reference of edge_4 since no reverse
         res_edge.resize(rank_4);
         reversed_flag_dst.resize(rank_4);
         for (auto i = 0; i < rank_4; i++) {
            reversed_flag_dst[i] = false;
            if constexpr (is_fermi) {
               res_edge[i] = {edge_4[i].arrow, *edge_4[i].map};
            } else {
               res_edge[i] = {*edge_4[i].map};
            }
         }
      }
      // the code above is dealing with edge_5 and res_Edge and reversed_flag_dst

      // put res_edge into res
      res.core = std::make_shared<Core<ScalarType, Symmetry>>(std::move(res_edge));
      if (!is_valid_name(res.names, res.core->edges.size())) {
         TAT_WARNING("Invalid Names");
      }
      // edge_6
      const auto& edge_6 = res.core->edges;

      // data
      // data_6
      auto data_6 = std::map<vector<Symmetry>, ScalarType*>();
      for (auto& [sym, vec] : res.core->blocks) {
         data_6[sym] = vec.data();
      };
      // create data_5
      auto true_data_5 = std::map<vector<Symmetry>, ScalarType*>();
      auto ptr_data_5 = &data_6;
      if (!merge_map.empty()) {
         auto offset_dst = data_6;
         auto symmetries_dst = initialize_block_symmetries_with_check(edge_5);
         for (auto& [sym, size] : symmetries_dst) {
            auto target_symmetry = vector<Symmetry>(rank_6);
            for (auto i = 0; i < rank_6; i++) {
               target_symmetry[i] = Symmetry();
            }
            for (auto i = 0; i < rank_4; i++) {
               target_symmetry[merge_flag[i]] += sym[i];
            }
            true_data_5[std::move(sym)] = offset_dst.at(target_symmetry);
            offset_dst[std::move(target_symmetry)] += size;
         }
         ptr_data_5 = &true_data_5;
      }
      const auto& data_5 = *ptr_data_5;
      // data_4
      const auto& data_4 = data_5;

      // 4 marks
      auto split_flag_mark = vector<bool>();
      auto reversed_flag_src_mark = vector<bool>();
      auto reversed_flag_dst_mark = vector<bool>();
      auto merge_flag_mark = vector<bool>();
      if constexpr (is_fermi) {
         if (apply_parity) {
            split_flag_mark.resize(rank_1);
            for (auto i = 0; i < rank_1; i++) {
               split_flag_mark[i] =
                     parity_exclude_name[0].find(name_1[i]) != parity_exclude_name[0].end();
            }
            reversed_flag_src_mark.resize(rank_2);
            for (auto i = 0; i < rank_2; i++) {
               reversed_flag_src_mark[i] =
                     parity_exclude_name[1].find(name_2[i]) != parity_exclude_name[1].end();
            }
            reversed_flag_dst_mark.resize(rank_5);
            for (auto i = 0; i < rank_5; i++) {
               reversed_flag_dst_mark[i] =
                     parity_exclude_name[2].find(name_5[i]) != parity_exclude_name[2].end();
            }
            merge_flag_mark.resize(rank_6);
            for (auto i = 0; i < rank_6; i++) {
               merge_flag_mark[i] =
                     parity_exclude_name[3].find(name_6[i]) != parity_exclude_name[3].end();
            }
         }
      }
      // main copy loop
      for (const auto& [sym_src, ptr_src] : data_3) {
         auto sym_dst = vector<Symmetry>(rank_3);
         auto dim_src = vector<Size>(rank_3);
         auto dim_dst = vector<Size>(rank_3);
         Size total_size = 1;
         for (auto i = 0; i < rank_3; i++) {
            auto dim = edge_3[i].map->at(sym_src[i]);
            total_size *= dim;
            dim_src[i] = dim;
            dim_dst[plan_src_to_dst[i]] = dim;
            sym_dst[plan_src_to_dst[i]] = sym_src[i];
         }
         auto ptr_dst = data_4.at(sym_dst);

         bool parity = false;
         if constexpr (is_fermi) {
            parity = Symmetry::get_transpose_parity(sym_src, plan_src_to_dst);

            if (apply_parity) {
               parity ^= Symmetry::get_reverse_parity(
                     sym_src, reversed_flag_src, reversed_flag_src_mark);
               parity ^= Symmetry::get_split_merge_parity(sym_src, split_flag, split_flag_mark);
               parity ^= Symmetry::get_reverse_parity(
                     sym_dst, reversed_flag_dst, reversed_flag_dst_mark);
               parity ^= Symmetry::get_split_merge_parity(sym_dst, merge_flag, merge_flag_mark);
            }
         }

         do_transpose(
               plan_src_to_dst,
               plan_dst_to_src,
               ptr_src,
               ptr_dst,
               dim_src,
               dim_dst,
               total_size,
               rank_3,
               parity);
      }

      return res;
   }
} // namespace TAT
#endif
