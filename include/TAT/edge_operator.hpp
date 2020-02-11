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

//TODO: 实现edge op
namespace TAT {
   template<class ScalarType, class Symmetry>
   template<class T, class>
   [[nodiscard]] Tensor<ScalarType, Symmetry> Tensor<ScalarType, Symmetry>::edge_operator(
         //void Tensor<ScalarType, Symmetry>::edge_operator(
         const std::map<Name, Name>& rename_map,
         const std::map<Name, vector<std::tuple<Name, Edge<Symmetry>>>>& split_map,
         const std::set<Name>& reversed_name,
         const std::map<Name, vector<Name>>& merge_map,
         T&& new_names,
         const bool apply_parity) const {
      // step 1: rename
      // step 2: split
      // step 3: reverse
      // step 4: transpose
      // step 5: merge
      // 前面三步是因果顺序往下推, 第四步转置需要根据merge和最后的name来确定转置方案

      using ConstDataType = std::map<vector<Symmetry>, const ScalarType*>;
      using DataType = std::map<vector<Symmetry>, ScalarType*>;

      const auto rank_0 = names.size();
      const auto& name_0 = names;
      auto edge_0 = vector<PtrEdge<Symmetry>>();
      for (const auto& e : core->edges) {
         edge_0.push_back(convert_to_ptr_edge(e));
      }
      auto data_0 = ConstDataType();
      for (const auto& [sym, vec] : core->blocks) {
         data_0[sym] = vec.data();
      };

      const auto& rank_1 = rank_0;
      const auto& edge_1 = edge_0;
      const auto& data_1 = data_0;
      auto name_1 = vector<Name>(rank_1);
      for (auto i = 0; i < rank_1; i++) {
         auto name = name_0[i];
         auto pos = rename_map.find(name);
         if (pos != rename_map.end()) {
            name = pos->second;
         }
         name_1[i] = name;
      }

      auto split_flag = vector<Rank>();
      Rank total_split_index = 0;
      auto name_2 = vector<Name>();
      auto edge_2 = vector<PtrEdge<Symmetry>>();
      for (auto i = 0; i < rank_1; i++) {
         auto pos = split_map.find(name_1[i]);
         if (pos != split_map.end()) {
            for (const auto& [n, e] : pos->second) {
               name_2.push_back(n);
               edge_2.push_back(convert_to_ptr_edge(e));
               split_flag.push_back(total_split_index);
            }
            total_split_index++;
         } else {
            name_2.push_back(name_1[i]);
            edge_2.push_back(edge_1[i]);
            split_flag.push_back(total_split_index++);
         }
      }
      const auto rank_2 = name_2.size();
      auto data_2 = ConstDataType();
      auto offset_src = data_1;
      auto symmetries_src = initialize_block_symmetries_with_check(edge_2);
      for (const auto& [sym, size] : symmetries_src) {
         auto target_symmetry = vector<Symmetry>(rank_1);
         for (auto i = 0; i < rank_1; i++) {
            target_symmetry[i] = Symmetry();
         }
         for (auto i = 0; i < rank_2; i++) {
            target_symmetry[split_flag[i]] += sym[i];
         }
         data_2[sym] = offset_src[target_symmetry];
         offset_src[target_symmetry] += size;
      }

      [[maybe_unused]] auto reversed_flag = vector<bool>();
      const auto& rank_3 = rank_2;
      const auto& name_3 = name_2;
      const auto& data_3 = data_2;
      auto edge_3 = edge_2;
      if constexpr (is_fermi_symmetry_v<Symmetry>) {
         if (apply_parity) {
            reversed_flag.resize(rank_3);
            for (auto i = 0; i < rank_3; i++) {
               if (reversed_name.find(name_3[i]) != reversed_name.end()) {
                  edge_3[i].arrow ^= true;
                  reversed_flag[i] = true;
               } else {
                  reversed_flag[i] = false;
               }
            }
         }
      }

      const auto& tmp_name_5 = new_names;
      const auto& rank_5 = tmp_name_5.size();
      auto name_4 = vector<Name>();
      auto merge_flag = vector<Rank>();
      Rank total_merge_index = 0;
      for (const auto& n : tmp_name_5) {
         auto pos = merge_map.find(n);
         if (pos != merge_map.end()) {
            for (const auto& i : pos->second) {
               name_4.push_back(i);
               merge_flag.push_back(total_merge_index);
            }
            total_merge_index++;
         } else {
            name_4.push_back(n);
            merge_flag.push_back(total_merge_index++);
         }
      }
      auto rank_4 = name_4.size();

      auto name_to_index_3 = construct_name_to_index(name_3);
      auto plan_src_to_dst = vector<Rank>(rank_4);
      auto plan_dst_to_src = vector<Rank>(rank_4);
      auto edge_4 = vector<PtrEdge<Symmetry>>(rank_4);
      for (auto i = 0; i < rank_4; i++) {
         plan_dst_to_src[i] = name_to_index_3.at(name_4[i]);
         plan_src_to_dst[plan_dst_to_src[i]] = i;
         edge_4[i] = edge_3[plan_dst_to_src[i]];
      }

      auto res_edge = vector<Edge<Symmetry>>();
      auto tmp_edge_list = vector<PtrEdge<Symmetry>>();
      auto current_res_edge_index = 0;
      for (auto i = 0; i < rank_4; i++) {
         if (merge_flag[i] != current_res_edge_index) {
            res_edge.push_back(get_merged_edge(tmp_edge_list));
            tmp_edge_list.clear();
            tmp_edge_list.push_back(edge_4[i]);
            current_res_edge_index++;
         } else {
            tmp_edge_list.push_back(edge_4[i]);
         }
      }
      res_edge.push_back(get_merged_edge(tmp_edge_list));

      auto res = Tensor<ScalarType, Symmetry>{std::forward<T>(new_names), std::move(res_edge)};
      const auto name_5 = res.names;
      const auto& edge_5 = res.core->edges;

      auto data_5 = DataType();
      for (auto& [sym, vec] : res.core->blocks) {
         data_5[sym] = vec.data();
      };

      auto data_4 = DataType();
      auto offset_dst = data_5;
      auto symmetries_dst = initialize_block_symmetries_with_check(edge_4);
      for (const auto& [sym, size] : symmetries_dst) {
         auto target_symmetry = vector<Symmetry>(rank_5);
         for (auto i = 0; i < rank_5; i++) {
            target_symmetry[i] = Symmetry();
         }
         for (auto i = 0; i < rank_4; i++) {
            target_symmetry[merge_flag[i]] += sym[i];
         }
         data_4[sym] = offset_dst[target_symmetry];
         offset_dst[target_symmetry] += size;
      }

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
               parity ^= Symmetry::get_reverse_parity(sym_src, reversed_flag);
               parity ^= Symmetry::get_split_merge_parity(sym_src, split_flag);
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
   } // namespace TAT
} // namespace TAT
#endif
