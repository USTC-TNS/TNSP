/**
 * \file tensor.hpp
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
#ifndef TAT_TENSOR_HPP
#define TAT_TENSOR_HPP

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <tuple>

#include "core.hpp"
#include "name.hpp"
#include "symmetry.hpp"
#include "transpose.hpp"

// #include <mpi.h>
// TODO: MPI 这个最后弄, 不然valgrind一大堆报错

namespace TAT {
   template<class ScalarType = double, class Symmetry = NoSymmetry>
   struct Tensor {
      using scalar_valid = std::enable_if_t<is_scalar_v<ScalarType>>;
      using symmetry_valid = std::enable_if_t<is_symmetry_v<Symmetry>>;

      // initialize
      vector<Name> names;
      std::map<Name, Rank> name_to_index;
      std::shared_ptr<Core<ScalarType, Symmetry>> core;

      template<
            class U = vector<Name>,
            class T = vector<map<Symmetry, Size>>,
            class = std::enable_if_t<std::is_convertible_v<U, vector<Name>>>,
            class = std::enable_if_t<std::is_convertible_v<T, vector<map<Symmetry, Size>>>>>
      Tensor(U&& n, T&& e) :
            names(std::forward<U>(n)),
            name_to_index(construct_name_to_index(names)),
            core(std::make_shared<Core<ScalarType, Symmetry>>(std::forward<T>(e))) {
         if (!is_valid_name(names, core->edges.size())) {
            TAT_WARNING("Invalid Names");
         }
      }

      [[nodiscard]] Tensor<ScalarType, Symmetry> copy() const {
         Tensor<ScalarType, Symmetry> res;
         res.names = names;
         res.name_to_index = name_to_index;
         res.core = std::make_shared<Core<ScalarType, Symmetry>>(*core);
         return res;
      }

      Tensor() = default;
      Tensor(const Tensor& other) = default;
      Tensor(Tensor&& other) = default;
      Tensor& operator=(const Tensor& other) = default;
      Tensor& operator=(Tensor&& other) = default;
      ~Tensor() = default;

      Tensor(ScalarType num) : Tensor({}, {}) {
         core->blocks[0].raw_data[0] = num;
      }

      operator ScalarType() const {
         if (!names.empty()) {
            TAT_WARNING("Conversion From Multiple Rank Tensor To Scalar");
         }
         return core->blocks[0].raw_data[0];
      }

      template<class Generator>
      Tensor<ScalarType, Symmetry>& set(Generator&& generator) & {
         if (core.use_count() != 1) {
            TAT_WARNING("Set Tensor Shared");
         }
         for (auto& i : core->blocks) {
            std::generate(i.raw_data.begin(), i.raw_data.end(), generator);
         }
         return *this;
      }
      template<class Generator>
      Tensor<ScalarType, Symmetry> set(Generator&& generator) && {
         return std::move(set(generator));
      }

      Tensor<ScalarType, Symmetry>& zero() & {
         return set([]() { return 0; });
      }
      Tensor<ScalarType, Symmetry> zero() && {
         return std::move(zero());
      }

      // get item
      [[nodiscard]] const auto& at(const std::map<Name, Symmetry>& position) const& {
         using is_not_nosymmetry = std::enable_if_t<!std::is_same_v<Symmetry, NoSymmetry>>;
         auto sym = get_pos_for_at(position, name_to_index, *core);
         return core->blocks[sym].raw_data;
      }

      [[nodiscard]] const ScalarType& at(const std::map<Name, Size>& position) const& {
         using is_nosymmetry = std::enable_if_t<std::is_same_v<Symmetry, NoSymmetry>>;
         auto pos = get_pos_for_at(position, name_to_index, *core);
         return core->blocks[0].raw_data[pos];
      }

      [[nodiscard]] ScalarType& at(const std::map<Name, Size>& position) & {
         using is_nosymmetry = std::enable_if_t<std::is_same_v<Symmetry, NoSymmetry>>;
         auto pos = get_pos_for_at(position, name_to_index, *core);
         return core->blocks[0].raw_data[pos];
      }

      [[nodiscard]] ScalarType at(const std::map<Name, Size>& position) && {
         using is_nosymmetry = std::enable_if_t<std::is_same_v<Symmetry, NoSymmetry>>;
         auto pos = get_pos_for_at(position, name_to_index, *core);
         return core->blocks[0].raw_data[pos];
      }

      [[nodiscard]] const ScalarType&
      at(const std::map<Name, std::tuple<Symmetry, Size>>& position) const& {
         auto [sym, pos] = get_pos_for_at(position, name_to_index, *core);
         return core->blocks[sym].raw_data[pos];
      }

      [[nodiscard]] ScalarType& at(const std::map<Name, std::tuple<Symmetry, Size>>& position) & {
         auto [sym, pos] = get_pos_for_at(position, name_to_index, *core);
         return core->blocks[sym].raw_data[pos];
      }

      [[nodiscard]] ScalarType at(const std::map<Name, std::tuple<Symmetry, Size>>& position) && {
         auto [sym, pos] = get_pos_for_at(position, name_to_index, *core);
         return core->blocks[sym].raw_data[pos];
      }

      // conversion
      template<class OtherScalarType, class = std::enable_if_t<is_scalar_v<OtherScalarType>>>
      [[nodiscard]] Tensor<OtherScalarType, Symmetry> to() const {
         if constexpr (std::is_same_v<ScalarType, OtherScalarType>) {
            auto res = Tensor<ScalarType, Symmetry>{};
            res.names = names;
            res.name_to_index = name_to_index;
            res.core = core;
            return res;
         } else {
            auto res = Tensor<OtherScalarType, Symmetry>{};
            res.names = names;
            res.name_to_index = name_to_index;
            res.core = std::make_shared<Core<OtherScalarType, Symmetry>>(core->edges);
            for (Nums i = 0; i < core->blocks.size(); i++) {
               auto& dst = res.core->blocks[i].raw_data;
               const auto& src = core->blocks[i].raw_data;
               for (Size j = 0; j < src.size(); j++) {
                  dst[j] = static_cast<OtherScalarType>(src[j]);
               }
            }
            return res;
         }
      }

      // norm
      template<int p = 2>
      [[nodiscard]] Tensor<real_base_t<ScalarType>, Symmetry> norm() const {
         real_base_t<ScalarType> res = 0;
         if constexpr (p == -1) {
            for (const auto& block : core->blocks) {
               for (const auto& j : block.raw_data) {
                  auto tmp = std::abs(j);
                  if (tmp > res) {
                     res = tmp;
                  }
               }
            }
         } else if constexpr (p == 0) {
            for (const auto& block : core->blocks) {
               res += real_base_t<ScalarType>(block.raw_data.size());
            }
         } else {
            for (const auto& block : core->blocks) {
               for (const auto& j : block.raw_data) {
                  if constexpr (p == 1) {
                     res += std::abs(j);
                  } else if constexpr (p == 2) {
                     if constexpr (std::is_same_v<ScalarType, real_base_t<ScalarType>>) {
                        auto tmp = j;
                        res += tmp * tmp;
                     } else {
                        auto tmp = std::abs(j);
                        res += tmp * tmp;
                     }
                  } else {
                     if constexpr (
                           p % 2 == 0 && std::is_same_v<ScalarType, real_base_t<ScalarType>>) {
                        res += std::pow(j, p);
                     } else {
                        res += std::pow(std::abs(j), p);
                     }
                  }
               }
            }
            return res = std::pow(res, 1. / p);
         }
         return res;
      }

      // edge rename
      [[nodiscard]] Tensor<ScalarType, Symmetry>
      edge_rename(const std::map<Name, Name>& dict) const {
         auto res = Tensor<ScalarType, Symmetry>{};
         res.core = core;
         std::transform(
               names.begin(), names.end(), std::back_inserter(res.names), [&dict](Name name) {
                  auto pos = dict.find(name);
                  if (pos == dict.end()) {
                     return name;
                  } else {
                     return pos->second;
                  }
               });
         res.name_to_index = construct_name_to_index(res.names);
         return res;
      }

      // edge operators
      template<
            class T = vector<Name>,
            class = std::enable_if_t<std::is_convertible_v<T, vector<Name>>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry> transpose(T&& target_names) const {
         return edge_operator({}, {}, {}, std::forward<T>(target_names));
      }

      [[nodiscard]] Tensor<ScalarType, Symmetry>
      merge_edge(const std::map<Name, vector<Name>>& merge, const bool apply_parity = false) const {
         vector<Name> target_name;
         for (auto it = names.rbegin(); it != names.rend(); ++it) {
            auto found_in_merge = false;
            for (const auto& [key, value] : merge) {
               auto vit = std::find(value.begin(), value.end(), *it);
               if (vit != value.end()) {
                  if (vit == value.end() - 1) {
                     target_name.push_back(key);
                  }
                  found_in_merge = true;
                  break;
               }
            }
            if (!found_in_merge) {
               target_name.push_back(*it);
            }
         }
         for (const auto& [key, value] : merge) {
            if (value.empty()) {
               target_name.push_back(key);
            }
         }
         reverse(target_name.begin(), target_name.end());
         return edge_operator({}, {}, merge, std::move(target_name), apply_parity);
      }

      [[nodiscard]] Tensor<ScalarType, Symmetry> split_edge(
            const std::map<Name, vector<NameWithEdge<Symmetry>>>& split,
            const bool apply_parity = false) const {
         vector<Name> target_name;
         for (const auto& n : names) {
            auto it = split.find(n);
            if (it != split.end()) {
               for (const auto& sn : it->second) {
                  target_name.push_back(sn.name);
               }
            } else {
               target_name.push_back(n);
            }
         }
         return edge_operator({}, split, {}, std::move(target_name), apply_parity);
      }

      // edge operator core
      template<
            class T = vector<Name>,
            class = std::enable_if_t<std::is_convertible_v<T, vector<Name>>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry> edge_operator(
            const std::map<Name, Name>& rename_map,
            const std::map<Name, vector<NameWithEdge<Symmetry>>>& split_map,
            const std::map<Name, vector<Name>>& merge_map,
            T&& new_names,
            const bool apply_parity = false) const {
         // merge split的过程中, 会产生半个符号, 所以需要限定这一个tensor是否拥有这个符号
         vector<std::tuple<Rank, Rank>> split_list;
         vector<std::tuple<Rank, Rank>> merge_list;

         auto original_name = vector<Name>();
         auto splitted_name = vector<Name>();
         const vector<map<Symmetry, Size>>& original_edge = core->edges;
         auto splitted_edge = vector<const map<Symmetry, Size>*>();
         for (auto& i : names) {
            auto renamed_name = i;
            auto it1 = rename_map.find(i);
            if (it1 != rename_map.end()) {
               renamed_name = it1->second;
            }
            original_name.push_back(renamed_name);

            const map<Symmetry, Size>& e = core->edges[name_to_index.at(i)];
            auto it2 = split_map.find(renamed_name);
            if (it2 != split_map.end()) {
               auto edge_list = vector<const map<Symmetry, Size>*>();
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
         auto merged_edge = vector<map<Symmetry, Size>>();
         auto transposed_edge = vector<const map<Symmetry, Size>*>();

         Rank fine_dst_tmp = 0;
         for (const auto& i : merged_name) {
            auto it1 = merge_map.find(i);
            if (it1 != merge_map.end()) {
               auto edge_list = vector<const map<Symmetry, Size>*>();
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

         using PosType = vector<typename map<Symmetry, Size>::const_iterator>;

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

                  src_position.push_back({pos,
                                          src_block_index,
                                          src_offset[src_block_index],
                                          total_size,
                                          src_parity});
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
      }
      // TODO: symmetry的设计漏洞， 需要指定方向和symmetry
      // 最后的一个edge应该是这样的 std::tuple<bool, std::map<Symmetry, Size>>
      // 现在的参数是vector<Name>, vector<map<Symmetry, Size>>
      // bool表示方向， 在逆转方向时， symmetry也会逆转， 但是对于fermi symmetry
      // 还会多一个parity （一个边一个， 所以一个张量半个， 和merge时的效果一样）
      // 方向应该由各个函数各自维护， 在edge operator中， 如果merge的方向不一样， 需要自行逆转
      // 并通过apply parity判断是否应用

      // TODO: contract
      // 调用 merge ， 这样就不用考虑contract特有的符号问题了
      static Tensor<ScalarType, Symmetry> contract(
            const Tensor<ScalarType, Symmetry>& tensor1,
            const Tensor<ScalarType, Symmetry>& tensor2,
            const vector<Name>& names1,
            const vector<Name>& names2) {
         // TODO: 不转置成矩阵直接乘积的可能
         // names1 names2 需要 check order, 这个无法再merge中判断
         // merge 需要按照原样的顺序进行转置
         auto merged_tensor1 = tensor1.merge_edge({{names1, Contract1}}, false, true);
         auto merged_tensor2 = tensor2.merge_edge({{names2, Contract2}}, true, true);
         // check which one in 8 and do matrix contract
      }

      // TODO: converge
      // multiple 的扩展版
      // 考虑到PESS可能和对称性不兼容, 那么还是恢复原来的multiple模式
      // 因为这个东西唯一出现的地方就是svd了
      static Tensor<ScalarType, Symmetry> converge(
            const Tensor<ScalarType, Symmetry>& tensor1,
            const Tensor<ScalarType, Symmetry>& tensor2,
            const vector<Name>& names1,
            const vector<Name>& names2,
            const vector<Name>& names_res,
            const Size order) {
         // 这个东西在svd写好后再弄
      }

      struct svd_res {
         Tensor<ScalarType, Symmetry> U;
         Tensor<real_base_t<ScalarType>, Symmetry> S;
         Tensor<ScalarType, Symmetry> V;
      };

      // TODO: SVD
      // 根据情况选择转置方式
      // 或许可以称作cutted orthogonalize
      // 需要先merge再split, 不然不能保证分块依然分块
      svd_res svd(const vector<Name>& u_edges, Name u_new_name, Name v_new_name) const {
         // auto merged_tensor = merge_edge({{u_edges, SVD1}, {v_edges, SVD2}});
         // svd
         // auto split...
      }

      struct orthogonalize_res {
         Tensor<ScalarType, Symmetry> U;
         Tensor<ScalarType, Symmetry> T;
      };

      // TODO: QR LQ
      // 除了svd的考虑外， 需要考虑使用lq还是qr等
      orthogonalize_res orthogonalize(
            const vector<Name>& given_edges,
            Name given_new_name,
            Name other_new_name,
            bool u_edges_given = true) const {
         //
      }

      // TODO: 代码review
      // TODO: 更美观的IO

      const Tensor<ScalarType, Symmetry>& meta_put(std::ostream&) const;
      const Tensor<ScalarType, Symmetry>& data_put(std::ostream&) const;
      Tensor<ScalarType, Symmetry>& meta_get(std::istream&);
      Tensor<ScalarType, Symmetry>& data_get(std::istream&);
   }; // namespace TAT

   // TODO: lazy framework
   // 看一下idris是如何做的
   // 需要考虑深搜不可行的问题
   // 支持inplace操作

   // GPU and so on
} // namespace TAT

#endif
