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
         const bool apply_parity) const {
      // step 1: rename
      // step 2: split
      // step 3: reverse
      // step 4: transpose
      // step 5: reverse before merge
      // step 6: merge
      // 前面三步是因果顺序往下推, 第四步转置需要根据merge和最后的name来确定转置方案

      using ConstDataType = std::map<vector<Symmetry>, const ScalarType*>;
      using DataType = std::map<vector<Symmetry>, ScalarType*>;

      // status 0
      const auto rank_0 = names.size();
      const auto& name_0 = names;
      auto edge_0 = vector<PtrEdge<Symmetry>>();
      for (const auto& e : core->edges) {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            edge_0.push_back({e.arrow, &e.map});
         } else {
            edge_0.push_back({&e.map});
         }
      }
      auto data_0 = ConstDataType();
      for (const auto& [sym, vec] : core->blocks) {
         data_0[sym] = vec.data();
      }

      // status 1 rename
      const auto& rank_1 = rank_0;
      const auto& edge_1 = edge_0;
      const auto& data_1 = data_0;
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
         auto res = Tensor<ScalarType, Symmetry>();
         res.names = std::forward<T>(new_names);
         res.name_to_index = construct_name_to_index(res.names);
         res.core = core;
         return res;
      }

      // status 2 split
      auto split_flag = vector<Rank>();
      Rank total_split_index = 0;
      auto true_name_2 = vector<Name>();
      auto true_edge_2 = vector<PtrEdge<Symmetry>>();
      auto ptr_name_2 = &name_1;
      auto ptr_edge_2 = &edge_1;
      if (!split_map.empty()) {
         for (auto i = 0; i < rank_1; i++) {
            auto pos = split_map.find(name_1[i]);
            if (pos != split_map.end()) {
               for (const auto& [n, e] : pos->second) {
                  true_name_2.push_back(n);
                  if constexpr (is_fermi_symmetry_v<Symmetry>) {
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

      const auto rank_2 = name_2.size();
      auto true_data_2 = ConstDataType();
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
      [[maybe_unused]] auto reversed_flag_src = vector<bool>();
      const auto& rank_3 = rank_2;
      const auto& name_3 = name_2;
      const auto& data_3 = data_2;
      auto ptr_edge_3 = &edge_2;
      auto true_edge_3 = vector<PtrEdge<Symmetry>>();
      if constexpr (is_fermi_symmetry_v<Symmetry>) {
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

      auto res = Tensor<ScalarType, Symmetry>();
      res.names = std::forward<T>(new_names);
      res.name_to_index = construct_name_to_index(res.names);
      // status 4 transpose and status 6 merge and status 5 reverse before merge
      // 先根据merge何name_6确定name_4, 再确定根据edge_3确定edge_4
      // name and rank
      const auto& name_6 = res.names;
      const auto& rank_6 = name_6.size();

      auto merge_flag = vector<Rank>();
      Rank total_merge_index = 0;
      auto true_name_4 = vector<Name>();
      auto ptr_name_4 = &name_6;
      if (!merge_map.empty()) {
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
      const auto rank_4 = name_4.size();
      const auto& name_5 = name_4;
      const auto& rank_5 = rank_4;

      // edge
      auto name_to_index_3 = construct_name_to_index(name_3);
      auto plan_src_to_dst = vector<Rank>(rank_4);
      auto plan_dst_to_src = vector<Rank>(rank_4);
      auto edge_4 = vector<PtrEdge<Symmetry>>(rank_4);
      for (auto i = 0; i < rank_4; i++) {
         plan_dst_to_src[i] = name_to_index_3.at(name_4[i]);
         plan_src_to_dst[plan_dst_to_src[i]] = i;
         edge_4[i] = edge_3[plan_dst_to_src[i]];
      }

      [[maybe_unused]] auto reversed_flag_dst = vector<bool>();
      auto res_edge = vector<Edge<Symmetry>>();
      auto start_of_merge = 0;
      auto end_of_merge = 0;
      auto ptr_edge_5 = &edge_4;
      auto true_edge_5 = vector<PtrEdge<Symmetry>>();
      if (!merge_map.empty()) {
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            true_edge_5 = edge_4;
            ptr_edge_5 = &true_edge_5;
         }
      }
      auto& edge_5 = *ptr_edge_5;

      if (!merge_map.empty()) {
         for (auto i = 0; i < rank_6; i++) {
            while (end_of_merge < rank_4 && merge_flag[end_of_merge] == i) {
               end_of_merge++;
            }
            [[maybe_unused]] Arrow arrow;
            [[maybe_unused]] bool arrow_fixed = false;
            auto edge_to_merge = vector<PtrEdge<Symmetry>>();
            for (auto j = start_of_merge; j < end_of_merge; j++) {
               if constexpr (is_fermi_symmetry_v<Symmetry>) {
                  if (edge_5[j].arrow_valid()) {
                     auto this_arrow = edge_5[j].arrow;
                     if (arrow_fixed) {
                        if (arrow == this_arrow) {
                           reversed_flag_dst.push_back(false);
                        } else {
                           edge_5[j].arrow ^= true;
                           reversed_flag_dst.push_back(true);
                        }
                     } else {
                        arrow_fixed = true;
                        arrow = this_arrow;
                        reversed_flag_dst.push_back(false);
                     }
                  } else {
                     reversed_flag_dst.push_back(false);
                  }
               }
               edge_to_merge.push_back(edge_5[j]);
            }
            auto merged_edge = get_merged_edge(edge_to_merge);
            if constexpr (is_fermi_symmetry_v<Symmetry>) {
               merged_edge.arrow = arrow;
            }
            res_edge.push_back(std::move(merged_edge));
            start_of_merge = end_of_merge;
         }
      } else {
         res_edge.resize(rank_4);
         reversed_flag_dst.resize(rank_4);
         for (auto i = 0; i < rank_4; i++) {
            reversed_flag_dst[i] = false;
            if constexpr (is_fermi_symmetry_v<Symmetry>) {
               res_edge[i] = {edge_4[i].arrow, *edge_4[i].map};
            } else {
               res_edge[i] = {*edge_4[i].map};
            }
         }
      }

      res.core = std::make_shared<Core<ScalarType, Symmetry>>(std::move(res_edge), false);
      if (!is_valid_name(res.names, res.core->edges.size())) {
         TAT_WARNING("Invalid Names");
      }
      const auto& edge_6 = res.core->edges;

      // data
      auto data_6 = DataType();
      for (auto& [sym, vec] : res.core->blocks) {
         data_6[sym] = vec.data();
      };

      auto true_data_5 = DataType();
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
      const auto& data_4 = data_5;

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
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            parity = Symmetry::get_transpose_parity(sym_src, plan_src_to_dst);

            if (apply_parity) {
               parity ^= Symmetry::get_reverse_parity(sym_src, reversed_flag_src);
               parity ^= Symmetry::get_split_merge_parity(sym_src, split_flag);
               parity ^= Symmetry::get_reverse_parity(sym_dst, reversed_flag_dst);
               parity ^= Symmetry::get_split_merge_parity(sym_dst, merge_flag);
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
