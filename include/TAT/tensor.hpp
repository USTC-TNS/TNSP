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
#include "symmetry.hpp"

// #include <mpi.h>
// TODO: MPI 这个最后弄, 不然valgrind一大堆报错

namespace TAT {
   /**
    * \brief TAT is A Tensor library!
    * \tparam ScalarType 张量内的标量类型
    * \tparam Symmetry 张量所满足的对称性
    */
   template<class ScalarType = double, class Symmetry = NoSymmetry>
   struct Tensor {
      using scalar_valid = std::enable_if_t<is_scalar_v<ScalarType>>;
      using symmetry_valid = std::enable_if_t<is_symmetry_v<Symmetry>>;

      /**
       * \brief 张量的边的名称
       * \see Name
       */
      vector<Name> names;
      /**
       * \brief 张量边名称到边的序号的映射表
       */
      std::map<Name, Rank> name_to_index;
      /**
       * \brief 张量中出了边名称外的其他数据
       * \see Core
       * \note 因为重命名边的操作很常见, 为了避免复制, 使用shared_ptr封装Core
       */
      std::shared_ptr<Core<ScalarType, Symmetry>> core;

      /**
       * \brief 根据张量边的名称和形状构造张量, 分块将自动根据对称性进行处理
       * \param names_init 边的名称
       * \param edges_init 边的形状
       * \param auto_reverse 费米对称性是否自动根据是否有负值整个反转
       * \see Core
       */
      template<
            class U = vector<Name>,
            class T = vector<Edge<Symmetry>>,
            class = std::enable_if_t<std::is_convertible_v<U, vector<Name>>>,
            class = std::enable_if_t<std::is_convertible_v<T, vector<Edge<Symmetry>>>>>
      Tensor(U&& names_init, T&& edges_init, const bool auto_reverse = false) :
            names(std::forward<U>(names_init)),
            name_to_index(construct_name_to_index(names)),
            core(std::make_shared<Core<ScalarType, Symmetry>>(
                  std::forward<T>(edges_init),
                  auto_reverse)) {
         if (!is_valid_name(names, core->edges.size())) {
            TAT_WARNING("Invalid Names");
         }
      }

      /**
       * \brief 张量的复制, 默认的赋值和复制初始化不会拷贝数据，而会共用core
       * \return 复制的结果
       * \see core
       */
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

      /**
       * \brief 创建秩为零的张量
       * \param num 秩为零的张量拥有的唯一一个元素的值
       */
      Tensor(ScalarType num) : Tensor({}, {}) {
         core->blocks.begin()->second[0] = num;
      }

      /**
       * \brief 秩为一的张量转化为其中唯一一个元素的标量类型
       */
      operator ScalarType() const {
         if (!names.empty()) {
            TAT_WARNING("Conversion From Multiple Rank Tensor To Scalar");
         }
         return core->blocks.begin()->second[0];
      }

      /**
       * \brief 通过一个生成器设置一个张量内的数据
       * \param generator 生成器, 一般来说是一个无参数的函数, 返回值为标量, 多次调用填充张量
       * \return 张量自身
       * \note 参见std::generate
       */
      template<class Generator>
      Tensor<ScalarType, Symmetry>& set(Generator&& generator) & {
         if (core.use_count() != 1) {
            TAT_WARNING("Set Tensor Shared");
         }
         for (auto& [_, i] : core->blocks) {
            std::generate(i.begin(), i.end(), generator);
         }
         return *this;
      }
      template<class Generator>
      Tensor<ScalarType, Symmetry> set(Generator&& generator) && {
         return std::move(set(generator));
      }

      /**
       * \brief 将张量内的数据全部设置为零
       * \return 张量自身
       * \see set
       */
      Tensor<ScalarType, Symmetry>& zero() & {
         return set([]() { return 0; });
      }
      Tensor<ScalarType, Symmetry> zero() && {
         return std::move(zero());
      }

      /**
       * \brief 将张量内的数据设置为便于测试的值
       * \return 张量自身
       * \see set
       */
      Tensor<ScalarType, Symmetry>& test(ScalarType first = 0, ScalarType step = 1) & {
         return set([&first, step]() {
            auto res = first;
            first += step;
            return res;
         });
      }
      Tensor<ScalarType, Symmetry> test(ScalarType first = 0, ScalarType step = 1) && {
         return std::move(test(first, step));
      }

      /**
       * \brief 获取张量的某个分块
       * \param position 分块每个子边对应的对称性值
       * \return 一个不可变的一维数组
       * \see get_block_for_get_item
       */
      [[nodiscard]] const auto& block(const std::map<Name, Symmetry>& position) const& {
         using has_symmetry = std::enable_if_t<!std::is_same_v<Symmetry, NoSymmetry>>;
         auto sym = get_block_for_get_item(position, name_to_index, *core);
         return core->blocks.at(sym);
      }

      using EdgeInfoForGetItem = std::
            conditional_t<std::is_same_v<Symmetry, NoSymmetry>, Size, std::tuple<Symmetry, Size>>;

      /**
       * \brief 获取张量中某个分块内的某个元素
       * \param position 分块每个子边对应的对称性值以及元素在此子边上的位置
       * \see get_offset_for_get_item, get_block_and_offset_for_get_item
       */
      [[nodiscard]] ScalarType at(const std::map<Name, EdgeInfoForGetItem>& position) const& {
         if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
            auto pos = get_offset_for_get_item(position, name_to_index, *core);
            return core->blocks.begin()->second[pos];
         } else {
            auto [sym, pos] = get_block_and_offset_for_get_item(position, name_to_index, *core);
            return core->blocks.at(sym)[pos];
         }
      }
      [[nodiscard]] ScalarType& at(const std::map<Name, EdgeInfoForGetItem>& position) & {
         if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
            auto pos = get_offset_for_get_item(position, name_to_index, *core);
            return core->blocks.begin()->second[pos];
         } else {
            auto [sym, pos] = get_block_and_offset_for_get_item(position, name_to_index, *core);
            return core->blocks.at(sym)[pos];
         }
      }

      /**
       * \brief 不同标量类型的张量之间的转换函数
       * \tparam OtherScalarType 目标张量的基础标量类型 
       * \return 转换后的张量
       */
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
            res.core = std::make_shared<Core<OtherScalarType, Symmetry>>();
            res.core->edges = core->edges;
            for (const auto& [i, j] : core->blocks) {
               auto tmp = vector<OtherScalarType>(j.size());
               for (auto k = 0; k < j.size(); k++) {
                  if constexpr (is_complex_v<ScalarType> && is_real_v<OtherScalarType>) {
                     tmp[k] = static_cast<OtherScalarType>(j[k].real());
                  } else {
                     tmp[k] = static_cast<OtherScalarType>(j[k]);
                  }
               }
               res.core->blocks[i] = std::move(tmp);
            }
            return res;
         }
      }

      /**
       * \brief 求张量的模, 是拉平看作向量的模, 并不是矩阵模之类的东西
       * \tparam p 所求的模是张量的p-模, 如果p=-1, 则意味着最大模即p=inf
       * \return 标量类型的模
       */
      template<int p = 2>
      [[nodiscard]] Tensor<real_base_t<ScalarType>, Symmetry> norm() const {
         real_base_t<ScalarType> res = 0;
         if constexpr (p == -1) {
            for (const auto& [_, block] : core->blocks) {
               for (const auto& j : block) {
                  auto tmp = std::abs(j);
                  if (tmp > res) {
                     res = tmp;
                  }
               }
            }
         } else if constexpr (p == 0) {
            for (const auto& [_, block] : core->blocks) {
               res += real_base_t<ScalarType>(block.size());
            }
         } else {
            for (const auto& [_, block] : core->blocks) {
               for (const auto& j : block) {
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

      /**
       * \brief 对张量边的名称进行重命名
       * \param dict 重命名方案的映射表
       * \return 仅仅改变了边的名称的张量, 与原张量共享Core
       * \note 虽然功能蕴含于edge_operator中, 但是edge_rename操作很常用, 所以并没有调用会稍微慢的edge_operator
       * 而是实现一个小功能的edge_rename
       */
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

      /**
       * \brief 对张量的边进行操作的中枢函数, 对边依次做重命名, 分裂, 费米箭头取反, 合并, 转置的操作,
       * \param rename_map 重命名边的名称的映射表
       * \param split_map 分裂一些边的数据, 需要包含分裂后边的形状, 不然分裂不唯一
       * \param reversed_name 将要取反费米箭头的边的名称列表
       * \param merge_map 合并一些边的名称列表
       * \param new_names 最后进行的转置操作后的边的名称顺序列表
       * \param apply_parity 控制费米对称性中费米性质产生的符号是否应用在结果张量上
       * \return 进行了一系列操作后的结果张量
       * \note 反转不满足和合并操作的条件时, 将在合并前再次反转需要反转的边, 方向对齐第一个有方向的边
       * \note 因为费米箭头在反转和合并分裂时会产生半个符号, 所以需要扔给一方张量, 另一方张量不变号
       * \note 但是转置部分时产生一个符号的, 所以这一部分无视apply_parity
       */
      template<
            class T = vector<Name>,
            class = std::enable_if_t<std::is_convertible_v<T, vector<Name>>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry> edge_operator(
            const std::map<Name, Name>& rename_map,
            const std::map<Name, vector<std::tuple<Name, BoseEdge<Symmetry>>>>& split_map,
            const std::set<Name>& reversed_name,
            const std::map<Name, vector<Name>>& merge_map,
            T&& new_names,
            const bool apply_parity = false) const;

      /**
       * \brief 对张量进行转置
       * \param target_names 转置后的目标边的名称顺序
       * \return 转置后的结果张量
       */
      template<
            class T = vector<Name>,
            class = std::enable_if_t<std::is_convertible_v<T, vector<Name>>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry> transpose(T&& target_names) const {
         return edge_operator({}, {}, {}, {}, std::forward<T>(target_names));
      }

      /**
       * \brief 将费米张量的一些边进行反转
       * \param reversed_name 反转的边的集合
       * \param apply_parity 是否应用反转产生的符号
       * \return 反转后的结果张量
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry>
      reverse_edge(const std::set<Name>& reversed_name, const bool apply_parity = false) const {
         return edge_operator({}, {}, reversed_name, {}, names, apply_parity);
      }

      /**
       * \brief 合并张量的一些边
       * \param merge 合并的边的名称的映射表
       * \param apply_parity 是否应用合并边产生的符号
       * \return 合并边后的结果张量
       * \note 合并前转置的策略是将一组合并的边按照合并时的顺序移动到这组合并边中最后的一个边前, 其他边位置不变
       */
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
         return edge_operator({}, {}, {}, merge, std::move(target_name), apply_parity);
      }

      /**
       * \brief 分裂张量的一些边
       * \param split 分裂的边的名称的映射表
       * \param apply_parity 是否应用分裂边产生的符号
       * \return 分裂边后的结果张量
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry> split_edge(
            const std::map<Name, vector<std::tuple<Name, BoseEdge<Symmetry>>>>& split,
            const bool apply_parity = false) const {
         vector<Name> target_name;
         for (const auto& n : names) {
            auto it = split.find(n);
            if (it != split.end()) {
               for (const auto& sn : it->second) {
                  target_name.push_back(std::get<0>(sn));
               }
            } else {
               target_name.push_back(n);
            }
         }
         return edge_operator({}, split, {}, {}, std::move(target_name), apply_parity);
      }

      // TODO: contract
      // 调用 merge ， 这样就不用考虑contract特有的符号问题了
#if 0
      static Tensor<ScalarType, Symmetry> contract(
            const Tensor<ScalarType, Symmetry>& tensor1,
            const Tensor<ScalarType, Symmetry>& tensor2,
            const vector<Name>& names1,
            const vector<Name>& names2) {
         // TODO: 不转置成矩阵直接乘积的可能, 当然， 这是几乎不可能的
         // names1 names2 需要 check order, 这个无法再merge中判断
         // merge 需要按照原样的顺序进行转置
         auto merged_tensor1 = tensor1.merge_edge({{names1, Contract1}}, false, true);
         auto merged_tensor2 = tensor2.merge_edge({{names2, Contract2}}, true, true);
         // check which one in 8 and do matrix contract
      }
#endif
#if 0


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
#endif

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
