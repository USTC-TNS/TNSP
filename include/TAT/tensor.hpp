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
#include <array>
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
            core(std::make_shared<Core<ScalarType, Symmetry>>(std::forward<T>(edges_init), auto_reverse)) {
         if (!is_valid_name(names, core->edges.size())) {
            warning_or_error("Invalid Names");
         }
      }

      /**
       * \brief 张量的复制, 默认的赋值和复制初始化不会拷贝数据，而会共用core
       * \return 复制的结果
       * \see core
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry> copy() const {
         Tensor<ScalarType, Symmetry> result;
         result.names = names;
         result.name_to_index = name_to_index;
         result.core = std::make_shared<Core<ScalarType, Symmetry>>(*core);
         return result;
      }

      Tensor() = default;
      Tensor(const Tensor& other) {
         names = other.names;
         name_to_index = other.name_to_index;
         core = other.core;
         warning_or_error("Why Copy");
      };
      Tensor(Tensor&& other) = default;
      Tensor& operator=(const Tensor& other) {
         names = other.names;
         name_to_index = other.name_to_index;
         core = other.core;
         warning_or_error("Why Copy");
         return *this;
      };
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
            warning_or_error("Conversion From Multiple Rank Tensor To Scalar");
         }
         return core->blocks.begin()->second[0];
      }

      /**
       * \brief 产生一个与自己形状一样的张量
       * \return 一个未初始化数据内容的张量
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry> same_shape() const {
         return Tensor<ScalarType, Symmetry>(names, core->edges);
      }
      /**
       * \brief 对张量的每个数据元素做同样的非原地的变换
       * \param function 变换的函数
       * \return 张量自身
       * \note 参见std::transform
       * \see transform
       */
      template<class Transform>
      [[nodiscard]] Tensor<ScalarType, Symmetry> map(Transform&& function) const {
         auto result = same_shape();
         for (auto& [symmetries, block] : core->blocks) {
            std::transform(block.begin(), block.end(), result.core->blocks.at(symmetries).begin(), function);
         }
         return result;
      }

      /**
       * \brief 对张量的每个数据元素做同样的原地的变换
       * \param function 变换的函数
       * \return 张量自身
       * \note 参见std::transform
       * \see map
       */
      template<class Transform>
      Tensor<ScalarType, Symmetry>& transform(Transform&& function) & {
         if (core.use_count() != 1) {
            warning_or_error("Set Tensor Shared");
         }
         for (auto& [_, block] : core->blocks) {
            std::transform(block.begin(), block.end(), block.begin(), function);
         }
         return *this;
      }
      template<class Transform>
      Tensor<ScalarType, Symmetry> transform(Transform&& function) && {
         return std::move(transform(function));
      }

      /**
       * \brief 通过一个生成器设置一个张量内的数据
       * \param generator 生成器, 一般来说是一个无参数的函数, 返回值为标量, 多次调用填充张量
       * \return 张量自身
       * \see transform
       */
      template<class Generator>
      Tensor<ScalarType, Symmetry>& set(Generator&& generator) & {
         transform([&](ScalarType _) { return generator(); });
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
            auto result = first;
            first += step;
            return result;
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
      [[nodiscard]] const auto& block(const std::map<Name, Symmetry>& position) const&;

      [[nodiscard]] auto& block(const std::map<Name, Symmetry>& position) &;

      using EdgeInfoForGetItem = std::conditional_t<std::is_same_v<Symmetry, NoSymmetry>, Size, std::tuple<Symmetry, Size>>;

      /**
       * \brief 获取张量中某个分块内的某个元素
       * \param position 分块每个子边对应的对称性值以及元素在此子边上的位置
       * \see get_offset_for_get_item, get_block_and_offset_for_get_item
       */
      [[nodiscard]] ScalarType at(const std::map<Name, EdgeInfoForGetItem>& position) const&;

      [[nodiscard]] ScalarType& at(const std::map<Name, EdgeInfoForGetItem>& position) &;

      /**
       * \brief 不同标量类型的张量之间的转换函数
       * \tparam OtherScalarType 目标张量的基础标量类型 
       * \return 转换后的张量
       */
      template<class OtherScalarType, class = std::enable_if_t<is_scalar_v<OtherScalarType>>>
      [[nodiscard]] Tensor<OtherScalarType, Symmetry> to() const {
         if constexpr (std::is_same_v<ScalarType, OtherScalarType>) {
            auto result = Tensor<ScalarType, Symmetry>{};
            result.names = names;
            result.name_to_index = name_to_index;
            result.core = core;
            return result;
         } else {
            auto result = Tensor<OtherScalarType, Symmetry>{};
            result.names = names;
            result.name_to_index = name_to_index;
            result.core = std::make_shared<Core<OtherScalarType, Symmetry>>();
            result.core->edges = core->edges;
            for (const auto& [symmetries, block] : core->blocks) {
               auto [iterator, success] = result.core->blocks.emplace(symmetries, block.size());
               auto& this_block = iterator->second;
               for (auto i = 0; i < block.size(); i++) {
                  if constexpr (is_complex_v<ScalarType> && is_real_v<OtherScalarType>) {
                     this_block[i] = static_cast<OtherScalarType>(block[i].real());
                  } else {
                     this_block[i] = static_cast<OtherScalarType>(block[i]);
                  }
               }
            }
            return result;
         }
      }

      /**
       * \brief 求张量的模, 是拉平看作向量的模, 并不是矩阵模之类的东西
       * \tparam p 所求的模是张量的p-模, 如果p=-1, 则意味着最大模即p=inf
       * \return 标量类型的模
       */
      template<int p = 2>
      [[nodiscard]] Tensor<real_base_t<ScalarType>, Symmetry> norm() const {
         real_base_t<ScalarType> result = 0;
         if constexpr (p == -1) {
            for (const auto& [_, block] : core->blocks) {
               for (const auto& number : block) {
                  if (auto absolute_value = std::abs(number); absolute_value > result) {
                     result = absolute_value;
                  }
               }
            }
         } else if constexpr (p == 0) {
            for (const auto& [_, block] : core->blocks) {
               result += real_base_t<ScalarType>(block.size());
            }
         } else {
            for (const auto& [_, block] : core->blocks) {
               for (const auto& number : block) {
                  if constexpr (p == 1) {
                     result += std::abs(number);
                  } else if constexpr (p == 2) {
                     result += std::norm(number);
                  } else {
                     if constexpr (p % 2 == 0 && is_real_v<ScalarType>) {
                        result += std::pow(number, p);
                     } else {
                        result += std::pow(std::abs(number), p);
                     }
                  }
               }
            }
            result = std::pow(result, 1. / p);
         }
         return result;
      }

      using MapIteratorList = vector<typename std::map<Symmetry, Size>::const_iterator>;

      /**
       * \brief 对张量边的名称进行重命名
       * \param dictionary 重命名方案的映射表
       * \return 仅仅改变了边的名称的张量, 与原张量共享Core
       * \note 虽然功能蕴含于edge_operator中, 但是edge_rename操作很常用, 所以并没有调用会稍微慢的edge_operator
       * 而是实现一个小功能的edge_rename
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry> edge_rename(const std::map<Name, Name>& dictionary) const {
         auto result = Tensor<ScalarType, Symmetry>{};
         result.core = core;
         std::transform(names.begin(), names.end(), std::back_inserter(result.names), [&dictionary](Name name) {
            if (auto position = dictionary.find(name); position == dictionary.end()) {
               return name;
            } else {
               return position->second;
            }
         });
         result.name_to_index = construct_name_to_index(result.names);
         return result;
      }

      /**
       * \brief 对张量的边进行操作的中枢函数, 对边依次做重命名, 分裂, 费米箭头取反, 合并, 转置的操作,
       * \param rename_map 重命名边的名称的映射表
       * \param split_map 分裂一些边的数据, 需要包含分裂后边的形状, 不然分裂不唯一
       * \param reversed_name 将要取反费米箭头的边的名称列表
       * \param merge_map 合并一些边的名称列表
       * \param new_names 最后进行的转置操作后的边的名称顺序列表
       * \param apply_parity 控制费米对称性中费米性质产生的符号是否应用在结果张量上的默认行为
       * \param parity_exclude_name 是否产生符号这个问题上行为与默认行为相反的操作的边的名称, 四部分分别是split, reverse, reverse_before_merge, merge
       * \return 进行了一系列操作后的结果张量
       * \note 反转不满足和合并操作的条件时, 将在合并前再次反转需要反转的边, 方向对齐第一个有方向的边
       * \note 因为费米箭头在反转和合并分裂时会产生半个符号, 所以需要扔给一方张量, 另一方张量不变号
       * \note 但是转置部分时产生一个符号的, 所以这一部分无视apply_parity
       */
      template<class T = vector<Name>, class = std::enable_if_t<std::is_convertible_v<T, vector<Name>>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry> edge_operator(
            const std::map<Name, Name>& rename_map,
            const std::map<Name, vector<std::tuple<Name, BoseEdge<Symmetry>>>>& split_map,
            const std::set<Name>& reversed_name,
            const std::map<Name, vector<Name>>& merge_map,
            T&& new_names,
            const bool apply_parity = false,
            const std::array<std::set<Name>, 4>& parity_exclude_name = {{{}, {}, {}, {}}},
            const std::map<Name, std::map<Symmetry, Size>>& edge_and_symmetries_to_cut_before_all = {}) const;

      /**
       * \brief 对张量进行转置
       * \param target_names 转置后的目标边的名称顺序
       * \return 转置后的结果张量
       */
      template<class T = vector<Name>, class = std::enable_if_t<std::is_convertible_v<T, vector<Name>>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry> transpose(T&& target_names) const {
         return edge_operator({}, {}, {}, {}, std::forward<T>(target_names));
      }

      /**
       * \brief 将费米张量的一些边进行反转
       * \param reversed_name 反转的边的集合
       * \param apply_parity 是否应用反转产生的符号
       * \return 反转后的结果张量
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry> reverse_edge(const std::set<Name>& reversed_name, const bool apply_parity = false) const {
         return edge_operator({}, {}, reversed_name, {}, names, apply_parity);
      }

      /**
       * \brief 合并张量的一些边
       * \param merge 合并的边的名称的映射表
       * \param apply_parity 是否应用合并边产生的符号
       * \return 合并边后的结果张量
       * \note 合并前转置的策略是将一组合并的边按照合并时的顺序移动到这组合并边中最后的一个边前, 其他边位置不变
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry> merge_edge(const std::map<Name, vector<Name>>& merge, const bool apply_parity = false) const {
         vector<Name> target_name;
         for (auto iterator = names.rbegin(); iterator != names.rend(); ++iterator) {
            auto found_in_merge = false;
            for (const auto& [name_after_merge, names_before_merge] : merge) {
               if (auto position_in_group = std::find(names_before_merge.begin(), names_before_merge.end(), *iterator);
                   position_in_group != names_before_merge.end()) {
                  if (position_in_group == names_before_merge.end() - 1) {
                     target_name.push_back(name_after_merge);
                  }
                  found_in_merge = true;
                  break;
               }
            }
            if (!found_in_merge) {
               target_name.push_back(*iterator);
            }
         }
         for (const auto& [name_after_merge, names_before_merge] : merge) {
            if (names_before_merge.empty()) {
               target_name.push_back(name_after_merge);
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
      [[nodiscard]] Tensor<ScalarType, Symmetry>
      split_edge(const std::map<Name, vector<std::tuple<Name, BoseEdge<Symmetry>>>>& split, const bool apply_parity = false) const {
         vector<Name> target_name;
         for (const auto& n : names) {
            if (auto found = split.find(n); found != split.end()) {
               for (const auto& edge_after_split : found->second) {
                  target_name.push_back(std::get<0>(edge_after_split));
               }
            } else {
               target_name.push_back(n);
            }
         }
         return edge_operator({}, split, {}, {}, std::move(target_name), apply_parity);
      }

      // TODO: 不转置成矩阵直接乘积的可能, 当然， 这是几乎不可能的
      /**
       * \brief 两个张量的缩并运算
       * \param tensor_1 参与缩并的第一个张量
       * \param tensor_2 参与缩并的第二个张量
       * \param contract_names_1 第一个张量将要缩并掉的边的名称
       * \param contract_names_2 第二个张量将要缩并掉的边的名称
       * \return 缩并后的张量
       */
      static Tensor<ScalarType, Symmetry> contract(
            const Tensor<ScalarType, Symmetry>& tensor_1,
            const Tensor<ScalarType, Symmetry>& tensor_2,
            const vector<Name>& contract_names_1,
            const vector<Name>& contract_names_2);

      Tensor<ScalarType, Symmetry>
      contract(const Tensor<ScalarType, Symmetry>& tensor_2, const vector<Name>& contract_names_1, const vector<Name>& contract_names_2) const {
         return Tensor<ScalarType, Symmetry>::contract(*this, tensor_2, contract_names_1, contract_names_2);
      }

      /**
       * \brief 生成张量的共轭张量
       * \note 如果为对称性张量, 量子数取反, 如果为费米张量, 箭头取反, 如果为复张量, 元素取共轭
       */
      Tensor<ScalarType, Symmetry> conjugate() const {
         if constexpr (std::is_same_v<Symmetry, NoSymmetry> && is_real_v<ScalarType>) {
            return *this;
         }
         auto result_edges = vector<Edge<Symmetry>>();
         for (const auto& edge : core->edges) {
            auto& result_edge = result_edges.emplace_back();
            if constexpr (is_fermi_symmetry_v<Symmetry>) {
               result_edge.arrow = !edge.arrow;
            }
            for (const auto& [symmetry, dimension] : edge.map) {
               result_edge.map[-symmetry] = dimension;
            }
         }
         auto result = Tensor<ScalarType, Symmetry>(names, result_edges);
         for (const auto& [symmetries, block] : core->blocks) {
            auto result_symmetries = vector<Symmetry>();
            for (const auto& symmetry : symmetries) {
               result_symmetries.push_back(-symmetry);
            }
            // result.core->blocks.at(result_symmetries) <- block
            Size total_size = block.size();
            ScalarType* destination = result.core->blocks.at(result_symmetries).data();
            const ScalarType* source = block.data();
            if constexpr (is_complex_v<ScalarType>) {
               for (Size i = 0; i < total_size; i++) {
                  destination[i] = std::conj(source[i]);
               }
            } else {
               for (Size i = 0; i < total_size; i++) {
                  destination[i] = source[i];
               }
            }
         }
         return result;
      }

      /**
       * \brief 张量svd的结果类型
       * \note S的的对称性是有方向的, 用来标注如何对齐, 向U对齐
       */
      struct svd_result {
         Tensor<ScalarType, Symmetry> U;
         std::map<Symmetry, vector<real_base_t<ScalarType>>> S;
         Tensor<ScalarType, Symmetry> V;
      };

      template<class OtherScalarType>
      Tensor<ScalarType, Symmetry>&
      multiple(const std::map<Symmetry, vector<OtherScalarType>>& S, const Name& name, bool different_direction = false) {
         if (core.use_count() != 1) {
            warning_or_error("Set Tensor Shared");
            warning_or_error("You Can Use tensor.copy().multiple(...)");
         }
         auto index = name_to_index.at(name);
         for (auto& [symmetries, block] : core->blocks) {
            auto symmetry_of_s = symmetries[index];
            if (different_direction) {
               symmetry_of_s = -symmetry_of_s;
            }
            const auto& vector_in_S = S.at(symmetry_of_s);
            auto i = 0;
            Size m = 1;
            for (; i < index; i++) {
               m *= core->edges[i].map.at(symmetries[i]);
            }
            Size k = core->edges[i].map.at(symmetries[i]);
            Size n = 1;
            for (i++; i < names.size(); i++) {
               n *= core->edges[i].map.at(symmetries[i]);
            }
            if (vector_in_S.size() != k) {
               warning_or_error("Vector Size Invalid in Multiple");
            }
            auto* data = block.data();
            for (Size a = 0; a < m; a++) {
               for (Size b = 0; b < k; b++) {
                  OtherScalarType v = vector_in_S[b];
                  for (Size c = 0; c < n; c++) {
                     *(data++) *= v;
                  }
               }
            }
         }
         return *this;
      }

      /**
       * \brief 对张量进行svd分解
       * \param free_name_set_u svd分解中u的边的名称集合
       * \param common_name_u 分解后u新产生的边的名称
       * \param common_name_v 分解后v新产生的边的名称
       * \param cut 需要截断的维度数目
       * \return svd的结果
       * \see svd_result
       * \note 对于对称性张量, S需要有对称性, S对称性与V的公共边配对, 与U的公共边相同
       */
      svd_result svd(const std::set<Name>& free_name_set_u, Name common_name_u, Name common_name_v, Size cut = -1) const;

      const Tensor<ScalarType, Symmetry>& meta_put(std::ostream&) const;
      const Tensor<ScalarType, Symmetry>& data_put(std::ostream&) const;
      Tensor<ScalarType, Symmetry>& meta_get(std::istream&);
      Tensor<ScalarType, Symmetry>& data_get(std::istream&);
   }; // namespace TAT

   // TODO: middle 用edge operator表示一个待计算的张量, 在contract中用到
   // 因为contract的操作是这样的
   // merge gemm split
   // 上一次split可以和下一次的merge合并
   // 比较重要， 可以大幅减少对称性张量的分块
   // 需要先把svd写出来
   /*
   template<class ScalarType, class Symmetry>
   struct QuasiTensor {
      Tensor<ScalarType, Symmetry> tensor;
      std::map<Name, vector<std::tuple<Name, BoseEdge<Symmetry>>>> split_map;
      std::set<Name> reversed_set;
      vector<Name> res_name;

      QuasiTensor

      operator Tensor<ScalarType, Symmetry>() && {
         return tensor.edge_operator({}, split_map, reversed_set, {}, std::move(res_name));
      }
      operator Tensor<ScalarType, Symmetry>() const& {
         return tensor.edge_operator({}, split_map, reversed_set, {}, res_name);
      }

      Tensor<ScalarType, Symmetry> merge_again(
            const std::set<Name>& merge_reversed_set,
            const std::map<Name, vector<Name>>& merge_map,
            vector<Name>&& merge_res_name,
            std::set<Name>& split_parity_mark,
            std::set<Name>& merge_parity_mark) {
         auto total_reversed_set = reversed_set; // merge_reversed_set
         return tensor.edge_operator(
               {},
               split_map,
               total_reversed_set,
               merge_map,
               merge_res_name,
               false,
               {{{}, split_parity_mark, {}, merge_parity_mark}});
      }
      QuasiTensor<ScalarType, Symmetry>
   };
   */

   // TODO: lazy framework
   // 看一下idris是如何做的
   // 需要考虑深搜不可行的问题
   // 支持inplace操作
   // TODO: use it
   // TODO: python bind

} // namespace TAT

#endif
