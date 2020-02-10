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
         const std::map<Name, Name>& rename_map,
         const std::map<Name, vector<NameWithEdge>>& split_map,
         const std::vector<Name>& reversed_name,
         const std::map<Name, vector<Name>>& merge_map,
         T&& new_names,
         const bool apply_parity) const {
#if 1
#else
      // merge split的过程中, 会产生半个符号, 所以需要限定这一个tensor是否拥有这个符号
      vector<std::tuple<Rank, Rank>> split_list;
      vector<std::tuple<Rank, Rank>> merge_list;

      auto original_name = vector<Name>();
      auto splitted_name = vector<Name>();
      const vector<Edge<Symmetry>>& original_edge = core->edges;
      auto splitted_edge = vector<const Edge<Symmetry>*>();
      for (auto& i : names) {
         auto renamed_name = i;
         auto it1 = rename_map.find(i);
         if (it1 != rename_map.end()) {
            renamed_name = it1->second;
         }
         original_name.push_back(renamed_name);

         const Edge<Symmetry>& e = core->edges[name_to_index.at(i)];
         auto it2 = split_map.find(renamed_name);
         if (it2 != split_map.end()) {
            auto edge_list = vector<const Edge<Symmetry>*>();
            Rank split_start = splitted_name.size();
            for (const auto& k : it2->second) {
               splitted_name.push_back(k.name);
               splitted_edge.push_back(&k.edge);
               edge_list.push_back(&k.edge);
            }
            Rank split_end = splitted_name.size();
            split_list.push_back({split_start, split_end});

            auto expected_cover_edge = get_merged_edge(edge_list);
            for (const auto& [key, value] : e) {
               if (value != expected_cover_edge.at(key)) {
                  TAT_WARNING("Invalid Edge Split");
               }
            }
         } else {
            splitted_name.push_back(renamed_name);
            splitted_edge.push_back(&e);
         }
      }

      const Rank original_rank = names.size();
      // const Rank splitted_rank = splitted_name.size();

      auto splitted_name_to_index = construct_name_to_index(splitted_name);

      vector<Rank> fine_plan_dst;
      vector<Rank> plan_dst_to_src;

      vector<Name> merged_name = std::forward<T>(new_names);
      auto transposed_name = vector<Name>();
      auto merged_edge = vector<Edge<Symmetry>>();
      auto transposed_edge = vector<const Edge<Symmetry>*>();

      Rank fine_dst_tmp = 0;
      for (const auto& i : merged_name) {
         auto it1 = merge_map.find(i);
         if (it1 != merge_map.end()) {
            auto edge_list = vector<const Edge<Symmetry>*>();
            Rank merge_start = transposed_name.size();
            for (const auto& k : it1->second) {
               transposed_name.push_back(k);
               auto idx = splitted_name_to_index.at(k);
               plan_dst_to_src.push_back(idx);
               fine_plan_dst.push_back(fine_dst_tmp);
               auto ep = splitted_edge[idx];
               transposed_edge.push_back(ep);
               edge_list.push_back(ep);
            }
            Rank merge_end = transposed_name.size();
            merge_list.push_back({merge_start, merge_end});
            merged_edge.push_back(get_merged_edge(edge_list));
         } else {
            transposed_name.push_back(i);
            auto idx = splitted_name_to_index.at(i);
            plan_dst_to_src.push_back(idx);
            fine_plan_dst.push_back(fine_dst_tmp);
            auto ep = splitted_edge[idx];
            transposed_edge.push_back(ep);
            merged_edge.push_back(*ep);
         }
         fine_dst_tmp++;
      }

      const Rank transposed_rank = transposed_name.size();
      const Rank merged_rank = merged_name.size();

      vector<Rank> fine_plan_src(transposed_rank);
      vector<Rank> plan_src_to_dst(transposed_rank);

      for (Rank i = 0; i < transposed_rank; i++) {
         plan_src_to_dst[plan_dst_to_src[i]] = i;
      }

      Rank fine_src_tmp = 0;
      for (const auto& i : original_name) {
         auto it1 = split_map.find(i);
         if (it1 != split_map.end()) {
            for (const auto& j : it1->second) {
               fine_plan_src[plan_src_to_dst[splitted_name_to_index.at(j.name)]] = fine_src_tmp;
            }
         } else {
            fine_plan_src[plan_src_to_dst[splitted_name_to_index.at(i)]] = fine_src_tmp;
         }
         fine_src_tmp++;
      }

      if (merged_rank == original_rank) {
         auto need_operator = false;
         for (Rank i = 0; i < merged_rank; i++) {
            auto name1 = original_name[i];
            auto it1 = split_map.find(name1);
            if (it1 != split_map.end()) {
               if (it1->second.size() != 1) {
                  need_operator = true;
                  break;
               }
               name1 = it1->second[0].name;
            }
            auto name2 = merged_name[i];
            auto it2 = merge_map.find(name2);
            if (it2 != merge_map.end()) {
               if (it2->second.size() != 1) {
                  need_operator = true;
                  break;
               }
               name2 = it2->second[0];
            }
            if (name1 != name2) {
               need_operator = true;
               break;
            }
         }
         if (!need_operator) {
            auto res = Tensor<ScalarType, Symmetry>{};
            res.names = std::move(merged_name);
            res.name_to_index = construct_name_to_index(res.names);
            res.core = core;
            return res;
         }
      }

      auto res = Tensor<ScalarType, Symmetry>(std::move(merged_name), std::move(merged_edge));

      auto src_rank = Rank(names.size());
      auto dst_rank = Rank(res.names.size());
      auto src_block_number = Nums(core->blocks.size());
      auto dst_block_number = Nums(res.core->blocks.size());
      vector<Size> src_offset(src_block_number, 0);
      vector<Size> dst_offset(dst_block_number, 0);

      using PosType = vector<typename Edge<Symmetry>::const_iterator>;

      vector<std::tuple<PosType, Nums, Size, Size, bool>> src_position;

      vector<Size> total_size_list(transposed_rank);

      loop_edge(
            splitted_edge,
            // rank0
            []() {},
            // check
            []([[maybe_unused]] const PosType& pos) {
               auto sum = Symmetry();
               for (const auto& i : pos) {
                  sum += i->first;
               }
               return sum == Symmetry();
            },
            // append
            [&]([[maybe_unused]] const PosType& pos) {
               vector<Symmetry> src_pos(src_rank, Symmetry());
               for (Rank i = 0; i < pos.size(); i++) {
                  src_pos[fine_plan_src[plan_src_to_dst[i]]] += pos[i]->first;
               }
               auto src_block_index = core->find_block(src_pos);

               auto total_size = total_size_list[pos.size() - 1];

               auto src_parity = false;
               if (apply_parity) {
                  src_parity ^= Symmetry::get_parity(src_pos, split_list);
               }

               src_position.push_back(
                     {pos, src_block_index, src_offset[src_block_index], total_size, src_parity});
               src_offset[src_block_index] += total_size;
            },
            // update
            [&total_size_list]([[maybe_unused]] const PosType& pos, [[maybe_unused]] Rank ptr) {
               for (auto i = ptr; i < pos.size(); i++) {
                  if (i == 0) {
                     total_size_list[0] = pos[0]->second;
                  } else {
                     total_size_list[i] = total_size_list[i - 1] * pos[i]->second;
                  }
               }
            });

      vector<Size> dst_dim(transposed_rank);
      vector<Size> src_dim(transposed_rank);

      loop_edge(
            transposed_edge,
            // rank0
            [&]() {
               res.core->blocks[0].raw_data[0] = core->blocks[0].raw_data[0];
               // rank=0则parity一定为1
            },
            // check
            []([[maybe_unused]] const PosType& pos) {
               auto sum = Symmetry();
               for (const auto& i : pos) {
                  sum += i->first;
               }
               return sum == Symmetry();
            },
            // append
            [&]([[maybe_unused]] const PosType& pos) {
               vector<Symmetry> dst_pos(dst_rank, Symmetry());
               for (auto i = 0; i < transposed_rank; i++) {
                  dst_pos[fine_plan_dst[i]] += pos[i]->first;
               }
               auto dst_block_index = res.core->find_block(dst_pos);
               ScalarType* dst_data = res.core->blocks[dst_block_index].raw_data.data() +
                                      dst_offset[dst_block_index];

               const ScalarType* src_data;
               Size total_size;
               auto total_parity = Symmetry::get_parity(dst_pos, plan_dst_to_src);
               if (apply_parity) {
                  total_parity ^= Symmetry::get_parity(dst_pos, merge_list);
               }

               auto found_src = false;
               for (const auto& [src_pos_info, src_index, src_offset, src_total_size, src_parity] :
                    src_position) {
                  auto difference = false;
                  for (auto j = 0; j < pos.size(); j++) {
                     if (src_pos_info[j]->first != pos[plan_src_to_dst[j]]->first) {
                        difference = true;
                        break;
                     }
                  }
                  if (!difference) {
                     src_data = core->blocks[src_index].raw_data.data() + src_offset;
                     total_size = src_total_size;
                     total_parity ^= src_parity;
                     found_src = true;
                     break;
                  }
               }
               if (!found_src) {
                  TAT_WARNING("Source Block Not Found");
               }

               do_transpose(
                     plan_src_to_dst,
                     plan_dst_to_src,
                     src_data,
                     dst_data,
                     src_dim,
                     dst_dim,
                     total_size,
                     transposed_rank,
                     total_parity);

               dst_offset[dst_block_index] += total_size;
            },
            // update
            [&src_dim, &dst_dim, &plan_dst_to_src](
                  [[maybe_unused]] const PosType& pos, [[maybe_unused]] Rank ptr) {
               for (auto i = ptr; i < pos.size(); i++) {
                  src_dim[plan_dst_to_src[i]] = dst_dim[i] = pos[i]->second;
               }
            });

      return res;
#endif
   }
} // namespace TAT
#endif
