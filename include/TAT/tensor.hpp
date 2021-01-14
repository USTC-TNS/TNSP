/**
 * \file tensor.hpp
 *
 * Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include "basic_type.hpp"
#include "core.hpp"
#include "edge.hpp"
#include "name.hpp"
#include "pmr_resource.hpp"
#include "symmetry.hpp"

namespace TAT {
   /**
    * \defgroup Singular
    * @{
    */

   /**
    * 张量看成矩阵后做svd分解后得到的奇异值类型, 为对角矩阵形式的张量
    *
    * \see Tensor::svd
    */
   template<typename ScalarType = double, typename Symmetry = NoSymmetry, typename Name = DefaultName>
   struct Singular {
      using normal_map = std::map<Symmetry, vector<real_base_t<ScalarType>>>;
      using fake_singular_map = fake_map<Symmetry, vector<real_base_t<ScalarType>>>;
#ifdef TAT_USE_SIMPLE_NOSYMMETRY
      using singular_map = std::conditional_t<std::is_same_v<Symmetry, NoSymmetry>, fake_singular_map, normal_map>;
#else
      using singular_map = normal_map;
#endif
      singular_map value;

      template<int p>
      [[nodiscard]] real_base_t<ScalarType> norm() const {
         if constexpr (p == -1) {
            real_base_t<ScalarType> maximum = 0;
            for (const auto& [symmetry, singulars] : value) {
               for (const auto& element : singulars) {
                  auto absolute = std::abs(element);
                  maximum = maximum < absolute ? absolute : maximum;
               }
            }
            return maximum;
         } else if constexpr (p == 1) {
            real_base_t<ScalarType> summation = 0;
            for (const auto& [symmetry, singulars] : value) {
               for (const auto& element : singulars) {
                  auto absolute = std::abs(element);
                  summation += absolute;
               }
            }
            return summation;
         } else {
            TAT_error("Not Implement For Singulars Normalize Kind, Only +1 and -1 supported now");
            return 0;
         }
      }

      [[nodiscard]] std::string show() const;
      [[nodiscard]] std::string dump() const;
      Singular<ScalarType, Symmetry, Name>& load(const std::string&) &;
      Singular<ScalarType, Symmetry, Name>&& load(const std::string& string) && {
         return std::move(load(string));
      };

      [[nodiscard]] Singular<ScalarType, Symmetry, Name> copy() const {
         return Singular<ScalarType, Symmetry, Name>{value};
      }
   };

   /**@}*/
   /**
    * \defgroup Tensor
    * @{
    */

   /// \private
   template<typename ScalarType, typename Symmetry, typename Name>
   struct TensorShape;

   /**
    * 张量类型
    *
    * 张量类型中含有元信息和数据两部分. 元信息包括秩, 以及秩个边
    * 每个边含有一个Name信息以及形状信息, 对于无对称性的张量, 边的形状使用一个数字描述, 即此边的维度.
    * 对于其他类型的对称性, 边的形状为一个该类型对称性(应该是该对称性的量子数, 这里简称对称性)到数的映射,
    * 表示某量子数下的维度. 而张量数据部分为若干个秩维矩块, 对于无对称性张量, 仅有唯一一个矩块.
    *
    * \tparam ScalarType 张量内的标量类型
    * \tparam Symmetry 张量所满足的对称性
    * \tparam Name 张量的边的名称类型
    */
   template<typename ScalarType = double, typename Symmetry = NoSymmetry, typename Name = DefaultName>
   struct Tensor {
      using scalar_valid = std::enable_if_t<is_scalar_v<ScalarType>>;
      using symmetry_valid = std::enable_if_t<is_symmetry_v<Symmetry>>;
      using name_valid = std::enable_if_t<is_name_v<Name>>;
      // TODO: private访问控制

      using scalar_t = ScalarType;
      using symmetry_t = Symmetry;
      using name_t = Name;
      using edge_t = Edge<Symmetry>;

      /**
       * 张量的边的名称
       * \see Name
       */
      std::vector<Name> names;
      /**
       * 张量边名称到边的序号的映射表
       * \note 虽然可能因为内存分配效率会不高, 但是在边很多的时候会很有用, 比如费米张量的w(s)处
       */
      std::map<Name, Rank> name_to_index;
      /**
       * 张量中出了边名称外的其他数据
       * \see Core
       * \note 因为重命名边的操作很常见, 为了避免复制, 使用shared_ptr封装Core
       */
      std::shared_ptr<Core<ScalarType, Symmetry>> core;

      TensorShape<ScalarType, Symmetry, Name> shape() {
         return {this};
      }

      /**
       * 根据张量边的名称和形状构造张量, 分块将自动根据对称性进行处理
       * \param names_init 边的名称
       * \param edges_init 边的形状
       * \param auto_reverse 费米对称性是否自动根据是否有负值整个反转
       * \see Core
       */
      template<typename VectorName = pmr::vector<Name>, typename VectorEdge = pmr::vector<Edge<Symmetry>>>
      Tensor(const VectorName& names_init, const VectorEdge& edges_init, const bool auto_reverse = false) :
            names(names_init.begin(), names_init.end()),
            name_to_index(construct_name_to_index<decltype(name_to_index)>(names)),
            core(std::make_shared<Core<ScalarType, Symmetry>>(edges_init, auto_reverse)) {
         check_valid_name(names, core->edges.size());
      }

      /**
       * 张量的复制, 默认的赋值和复制初始化不会拷贝数据，而会共用core
       * \return 复制的结果
       * \see core
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> copy() const {
         Tensor<ScalarType, Symmetry, Name> result;
         result.names = names;
         result.name_to_index = name_to_index;
         result.core = std::make_shared<Core<ScalarType, Symmetry>>(*core);
         return result;
      }

      [[nodiscard]] bool is_valid() const {
         return bool(core);
      }

#ifdef TAT_USE_VALID_DEFAULT_TENSOR
      Tensor() : Tensor(1){};
#else
      Tensor() = default;
#endif
      Tensor(const Tensor& other) = default;
      Tensor(Tensor&& other) noexcept = default;
      Tensor& operator=(const Tensor& other) = default;
      Tensor& operator=(Tensor&& other) noexcept = default;
      ~Tensor() = default;

      /**
       * 创建秩为零的张量
       * \param number 秩为零的张量拥有的唯一一个元素的值
       */
      Tensor(ScalarType number) : Tensor({}, {}) {
         core->blocks.begin()->second.front() = number;
      }

      /**
       * 创建高秩但是元素只有一个的张量
       * \param number 秩为零的张量拥有的唯一一个元素的值
       * \param names_init 边的名称
       * \param edge_symmetry 如果系统含有对称性, 则需要设置此值
       * \param edge_arrow 如果系统对称性为fermi对称性, 则需要设置此值
       */
      template<typename VectorName = pmr::vector<Name>, typename VectorSymmetry = pmr::vector<Symmetry>, typename VectorArrow = pmr::vector<Arrow>>
      [[nodiscard]] static Tensor<ScalarType, Symmetry, Name>
      one(ScalarType number, const VectorName& names_init, const VectorSymmetry& edge_symmetry = {}, const VectorArrow& edge_arrow = {}) {
         auto rank = names_init.size();
         auto result = Tensor(names_init, get_edge_from_edge_symmetry_and_arrow(edge_symmetry, edge_arrow, rank));
         result.core->blocks.begin()->second.front() = number;
         return result;
      }

      [[nodiscard]] bool is_scalar() const {
         return core->blocks.size() == 1 && core->blocks.begin()->second.size() == 1;
      }

      /**
       * 秩为一的张量转化为其中唯一一个元素的标量类型
       */
      operator ScalarType() const {
         if (!is_scalar()) {
            TAT_error("Try to get the only element of the tensor which contains more than one element");
         }
         return core->blocks.begin()->second.front();
      }

      using EdgePoint = std::conditional_t<std::is_same_v<Symmetry, NoSymmetry>, Size, std::tuple<Symmetry, Size>>;
      using EdgePointWithArrow = std::conditional_t<
            std::is_same_v<Symmetry, NoSymmetry>,
            std::tuple<Size, Size>,
            std::conditional_t<is_fermi_symmetry_v<Symmetry>, std::tuple<Arrow, Symmetry, Size, Size>, std::tuple<Symmetry, Size, Size>>>;

      template<typename ExpandConfigure = pmr::map<Name, EdgePointWithArrow>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      expand(const ExpandConfigure& configure, const Name& old_name = InternalName<Name>::No_Old_Name) const;

      template<typename ShrinkConfigure = pmr::map<Name, EdgePoint>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      shrink(const ShrinkConfigure& configure, const Name& new_name = InternalName<Name>::No_New_Name, Arrow arrow = false) const;

      [[deprecated("Use shrink instead, slice will be remove in v0.2.0")]] [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      slice(const pmr::map<Name, EdgePoint>& configure, const Name& new_name = InternalName<Name>::No_New_Name, Arrow arrow = false) const {
         return shrink(configure, new_name, arrow);
      }

      /**
       * 产生一个与自己形状一样的张量
       * \return 一个未初始化数据内容的张量
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> same_shape() const {
         return Tensor<ScalarType, Symmetry, Name>(names, core->edges);
      }
      /**
       * 对张量的每个数据元素做同样的非原地的变换
       * \param function 变换的函数
       * \return 张量自身
       * \note 参见std::transform
       * \see transform
       */
      template<typename Function>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> map(Function&& function) const {
         auto result = same_shape();
         for (auto& [symmetries, block] : core->blocks) {
            std::transform(block.begin(), block.end(), result.core->blocks.at(symmetries).begin(), function);
         }
         return result;
      }

      /**
       * 对张量的每个数据元素做同样的原地的变换
       * \param function 变换的函数
       * \return 张量自身
       * \note 参见std::transform
       * \see map
       */
      template<typename Function>
      Tensor<ScalarType, Symmetry, Name>& transform(Function&& function) & {
         if (core.use_count() != 1) {
            core = std::make_shared<Core<ScalarType, Symmetry>>(*core);
            TAT_warning_or_error_when_copy_shared("Set tensor shared, copy happened here");
         }
         for (auto& [_, block] : core->blocks) {
            std::transform(block.begin(), block.end(), block.begin(), function);
         }
         return *this;
      }
      template<typename Function>
      Tensor<ScalarType, Symmetry, Name>&& transform(Function&& function) && {
         return std::move(transform(function));
      }

      /**
       * 通过一个生成器设置一个张量内的数据
       * \param generator 生成器, 一般来说是一个无参数的函数, 返回值为标量, 多次调用填充张量
       * \return 张量自身
       * \see transform
       */
      template<typename Generator>
      Tensor<ScalarType, Symmetry, Name>& set(Generator&& generator) & {
         transform([&](ScalarType _) { return generator(); });
         return *this;
      }
      template<typename Generator>
      Tensor<ScalarType, Symmetry, Name>&& set(Generator&& generator) && {
         return std::move(set(generator));
      }

      /**
       * 将张量内的数据全部设置为零
       * \return 张量自身
       * \see set
       */
      Tensor<ScalarType, Symmetry, Name>& zero() & {
         return set([]() { return 0; });
      }
      Tensor<ScalarType, Symmetry, Name>&& zero() && {
         return std::move(zero());
      }

      /**
       * 将张量内的数据设置为便于测试的值
       * \return 张量自身
       * \see set
       */
      Tensor<ScalarType, Symmetry, Name>& test(ScalarType first = 0, ScalarType step = 1) & {
         return set([&first, step]() {
            auto result = first;
            first += step;
            return result;
         });
      }
      Tensor<ScalarType, Symmetry, Name>&& test(ScalarType first = 0, ScalarType step = 1) && {
         return std::move(test(first, step));
      }

      /**
       * 获取张量的某个分块
       * \param position 分块每个子边对应的对称性值
       * \return 一个不可变的一维数组
       * \see get_block_for_get_item
       */
      template<typename MapNameSymmetry = pmr::map<Name, Symmetry>>
      [[nodiscard]] const auto& block(const MapNameSymmetry& position = {}) const& {
         return const_block(position);
      }

      template<typename MapNameSymmetry = pmr::map<Name, Symmetry>>
      [[nodiscard]] auto& block(const MapNameSymmetry& position = {}) &;

      template<typename MapNameSymmetry = pmr::map<Name, Symmetry>>
      [[nodiscard]] const auto& const_block(const MapNameSymmetry& position = {}) const&;

      /**
       * 获取张量中某个分块内的某个元素
       * \param position 分块每个子边对应的对称性值以及元素在此子边上的位置
       * \note position对于无对称性张量, 为边名到维度的映射表, 对于有对称性的张量, 是边名到对称性和相应维度的映射表
       */
      template<typename MapNameEdgePoint = pmr::map<Name, EdgePoint>>
      [[nodiscard]] const ScalarType& at(const MapNameEdgePoint& position) const& {
         return const_at(position);
      }

      template<typename MapNameEdgePoint = pmr::map<Name, EdgePoint>>
      [[nodiscard]] ScalarType& at(const MapNameEdgePoint& position) &;

      template<typename MapNameEdgePoint = pmr::map<Name, EdgePoint>>
      [[nodiscard]] const ScalarType& const_at(const MapNameEdgePoint& position) const&;

      /**
       * 不同标量类型的张量之间的转换函数
       * \tparam OtherScalarType 目标张量的基础标量类型
       * \return 转换后的张量
       */
      template<typename OtherScalarType, typename = std::enable_if_t<is_scalar_v<OtherScalarType>>>
      [[nodiscard]] Tensor<OtherScalarType, Symmetry, Name> to() const {
         if constexpr (std::is_same_v<ScalarType, OtherScalarType>) {
            auto result = Tensor<ScalarType, Symmetry, Name>{};
            result.names = names;
            result.name_to_index = name_to_index;
            result.core = core;
            return result;
         } else {
            auto result = Tensor<OtherScalarType, Symmetry, Name>{};
            result.names = names;
            result.name_to_index = name_to_index;
            result.core = std::make_shared<Core<OtherScalarType, Symmetry>>();
            result.core->edges = core->edges;
            for (const auto& [symmetries, block] : core->blocks) {
               auto [iterator, success] = result.core->blocks.emplace(symmetries, block.size());
               auto& this_block = iterator->second;
               for (Size i = 0; i < block.size(); i++) {
                  if constexpr (is_complex_v<ScalarType> && is_real_v<OtherScalarType>) {
                     this_block[i] = OtherScalarType(block[i].real());
                  } else {
                     this_block[i] = OtherScalarType(block[i]);
                  }
               }
            }
            return result;
         }
      }

      /**
       * 求张量的模, 是拉平看作向量的模, 并不是矩阵模之类的东西
       * \tparam p 所求的模是张量的p-模, 如果p=-1, 则意味着最大模即p=inf
       * \return 标量类型的模
       */
      template<int p = 2>
      [[nodiscard]] real_base_t<ScalarType> norm() const {
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

      /**
       * 对张量的边进行操作的中枢函数, 对边依次做重命名, 分裂, 费米箭头取反, 合并, 转置的操作,
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
       * \note 本函数对转置外不标准的腿的输入是脆弱的
       */
      template<
            typename MapNameName = pmr::map<Name, Name>,
            typename MapNameVectorNameAndEdge = pmr::map<Name, std::vector<std::tuple<Name, BoseEdge<Symmetry>>>>,
            typename SetName1 = pmr::set<Name>,
            typename MapNameVectorName = pmr::map<Name, std::vector<Name>>,
            typename VectorName = pmr::vector<Name>,
            typename SetName2 = pmr::set<Name>,
            typename MapNameMapSymmetrySize = std::map<Name, std::map<Symmetry, Size>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> edge_operator(
            const MapNameName& rename_map,
            const MapNameVectorNameAndEdge& split_map,
            const SetName1& reversed_name,
            const MapNameVectorName& merge_map,
            const VectorName& new_names,
            bool apply_parity = false,
            const std::array<SetName2, 4>& parity_exclude_name = {{{}, {}, {}, {}}},
            const MapNameMapSymmetrySize& edge_and_symmetries_to_cut_before_all = {}) const;

      /**
       * 对张量边的名称进行重命名
       * \param dictionary 重命名方案的映射表
       * \return 仅仅改变了边的名称的张量, 与原张量共享Core
       * \note 虽然功能蕴含于edge_operator中, 但是edge_rename操作很常用, 所以并没有调用会稍微慢的edge_operator, 而是实现一个小功能的edge_rename
       */
      template<typename MapNameName = pmr::map<Name, Name>>
      [[nodiscard]] auto edge_rename(const MapNameName& dictionary) const;

      /**
       * 对张量进行转置
       * \param target_names 转置后的目标边的名称顺序
       * \return 转置后的结果张量
       */
      template<typename VectorName = pmr::vector<Name>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> transpose(const VectorName& target_names) const;

      /**
       * 将费米张量的一些边进行反转
       * \param reversed_name 反转的边的集合
       * \param apply_parity 是否应用反转产生的符号
       * \param parity_exclude_name 与apply_parity行为相反的边名集合
       * \return 反转后的结果张量
       */
      template<typename SetName1 = pmr::set<Name>, typename SetName2 = pmr::set<Name>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      reverse_edge(const SetName1& reversed_name, bool apply_parity = false, SetName2&& parity_exclude_name = {}) const;

      /**
       * 合并张量的一些边
       * \param merge 合并的边的名称的映射表
       * \param apply_parity 是否应用合并边产生的符号
       * \param parity_exclude_name_merge merge过程中与apply_parity不符的例外
       * \param parity_exclude_name_reverse merge前不得不做的reverse过程中与apply_parity不符的例外
       * \return 合并边后的结果张量
       * \note 合并前转置的策略是将一组合并的边按照合并时的顺序移动到这组合并边中最后的一个边前, 其他边位置不变
       */
      template<typename MapNameVectorName = pmr::map<Name, std::vector<Name>>, typename SetName = pmr::set<Name>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> merge_edge(
            MapNameVectorName merge,
            bool apply_parity = false,
            SetName&& parity_exclude_name_merge = {},
            SetName&& parity_exclude_name_reverse = {}) const;

      /**
       * 分裂张量的一些边
       * \param split 分裂的边的名称的映射表
       * \param apply_parity 是否应用分裂边产生的符号
       * \param parity_exclude_name_split split过程中与apply_parity不符的例外
       * \return 分裂边后的结果张量
       */
      template<
            typename MapNameVectorNameAndEdge = pmr::map<Name, std::vector<std::tuple<Name, BoseEdge<Symmetry>>>>,
            typename SetName = pmr::set<Name>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      split_edge(MapNameVectorNameAndEdge split, bool apply_parity = false, SetName&& parity_exclude_name_split = {}) const;

      // 可以考虑不转置成矩阵直接乘积的可能, 但这个最多优化N^2的常数次, 只需要转置不调用多次就不会产生太大的问题
      /**
       * 两个张量的缩并运算
       * \param tensor_1 参与缩并的第一个张量
       * \param tensor_2 参与缩并的第二个张量
       * \param contract_names 两个张量将要缩并掉的边的名称
       * \return 缩并后的张量
       */
      template<typename SetNameAndName = pmr::set<std::tuple<Name, Name>>>
      [[nodiscard]] static Tensor<ScalarType, Symmetry, Name> contract(
            const Tensor<ScalarType, Symmetry, Name>& tensor_1,
            const Tensor<ScalarType, Symmetry, Name>& tensor_2,
            SetNameAndName&& contract_names);

      template<typename ScalarType1, typename ScalarType2, typename SetNameAndName = pmr::set<std::tuple<Name, Name>>>
      [[nodiscard]] static auto contract(
            const Tensor<ScalarType1, Symmetry, Name>& tensor_1,
            const Tensor<ScalarType2, Symmetry, Name>& tensor_2,
            SetNameAndName&& contract_names) {
         using ResultScalarType = std::common_type_t<ScalarType1, ScalarType2>;
         using ResultTensor = Tensor<ResultScalarType, Symmetry, Name>;
         if constexpr (std::is_same_v<ResultScalarType, ScalarType1>) {
            if constexpr (std::is_same_v<ResultScalarType, ScalarType2>) {
               return ResultTensor::contract(tensor_1, tensor_2, std::forward<SetNameAndName>(contract_names));
            } else {
               return ResultTensor::contract(tensor_1, tensor_2.template to<ResultScalarType>(), std::forward<SetNameAndName>(contract_names));
            }
         } else {
            if constexpr (std::is_same_v<ResultScalarType, ScalarType2>) {
               return ResultTensor::contract(tensor_1.template to<ResultScalarType>(), tensor_2, std::forward<SetNameAndName>(contract_names));
            } else {
               return ResultTensor::contract(
                     tensor_1.template to<ResultScalarType>(),
                     tensor_2.template to<ResultScalarType>(),
                     std::forward<SetNameAndName>(contract_names));
            }
         }
      }

      template<typename OtherScalarType, typename SetNameAndName = pmr::set<std::tuple<Name, Name>>>
      [[nodiscard]] auto contract(const Tensor<OtherScalarType, Symmetry, Name>& tensor_2, SetNameAndName&& contract_names) const {
         return contract(*this, tensor_2, std::forward<SetNameAndName>(contract_names));
      }

      /**
       * 将一个张量与另一个张量的所有相同名称的边进行缩并
       * \param other 另一个张量
       * \return 缩并后的结果
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> contract_all_edge(const Tensor<ScalarType, Symmetry, Name>& other) const {
         auto pmr_guard = scope_resource<1 << 10>();
         // other不含有的边会在contract中自动删除
         auto contract_names = pmr::set<std::tuple<Name, Name>>();
         for (const auto& i : names) {
            contract_names.insert({i, i});
         }
         return contract(other, std::move(contract_names));
      }

      /**
       * 张量与自己的共轭进行尽可能的缩并
       * \return 缩并后的结果
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> contract_all_edge() const {
         return contract_all_edge(conjugate());
      }

      /**
       * 生成相同形状的单位张量
       * \param pairs 看作矩阵时边的配对方案
       */
      template<typename SetNameAndName = pmr::set<std::tuple<Name, Name>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> identity(const SetNameAndName& pairs) const;

      /**
       * 看作矩阵后求出矩阵指数
       * \param pairs 边的配对方案
       * \param step 迭代步数
       */
      template<typename SetNameAndName = pmr::set<std::tuple<Name, Name>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> exponential(const SetNameAndName& pairs, int step = 2) const;

      /**
       * 生成张量的共轭张量
       * \note 如果为对称性张量, 量子数取反, 如果为费米张量, 箭头取反, 如果为复张量, 元素取共轭
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> conjugate() const;

      template<typename SetNameAndName = pmr::set<std::tuple<Name, Name>>>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> trace(const SetNameAndName& trace_names) const;

      using SingularType =
#ifdef TAT_USE_SINGULAR_MATRIX
            Tensor<ScalarType, Symmetry, Name>
#else
            Singular<ScalarType, Symmetry, Name>
#endif
            ;
      /**
       * 张量svd的结果类型
       * \note S的的对称性是有方向的, 用来标注如何对齐, 向U对齐
       */
      struct svd_result {
         Tensor<ScalarType, Symmetry, Name> U;
         SingularType S;
         Tensor<ScalarType, Symmetry, Name> V;
      };

      /**
       * 张量qr的结果类型
       */
      struct qr_result {
         Tensor<ScalarType, Symmetry, Name> Q;
         Tensor<ScalarType, Symmetry, Name> R;
      };

      /**
       * 张量缩并上SVD产生的奇异值数据, 就地操作
       * \param S 奇异值
       * \param name 张量与奇异值缩并的边名
       * \param direction 奇异值是含有一个方向的, SVD的结果中U还是V将与S相乘在这里被选定
       * \param division 如果为真, 则进行除法而不是乘法
       * \return 缩并的结果
       */
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name> multiple(const SingularType& S, const Name& name, char direction, bool division = false) const;

      /**
       * 对张量进行svd分解
       * \param free_name_set_u svd分解中u的边的名称集合
       * \param common_name_u 分解后u新产生的边的名称
       * \param common_name_v 分解后v新产生的边的名称
       * \param cut 需要截断的维度数目
       * \return svd的结果
       * \see svd_result
       * \note 对于对称性张量, S需要有对称性, S对称性与V的公共边配对, 与U的公共边相同
       */
      template<typename SetName = pmr::set<Name>>
      [[nodiscard]] svd_result
      svd(const SetName& free_name_set_u,
          const Name& common_name_u,
          const Name& common_name_v,
          Size cut = Size(-1),
          const Name& singular_name_u = InternalName<Name>::SVD_U,
          const Name& singular_name_v = InternalName<Name>::SVD_V) const;

      /**
       * 对张量进行qr分解
       * \param free_name_direction free_name_set取的方向, 为'Q'或'R'
       * \param free_name_set qr分解中某一侧的边的名称集合
       * \param common_name_q 分解后q新产生的边的名称
       * \param common_name_r 分解后r新产生的边的名称
       * \return qr的结果
       * \see qr_result
       */
      template<typename SetName = pmr::set<Name>>
      [[nodiscard]] qr_result qr(char free_name_direction, const SetName& free_name_set, const Name& common_name_q, const Name& common_name_r) const;

#ifdef TAT_USE_MPI
      /**
       * source调用此函数, 向destination发送一个张量
       */
      [[deprecated("TAT::Tensor::send deprecated, use TAT::mpi.send instead")]] void send(int destination) const;
      /**
       * destination调用此函数, 从source接受一个张量
       */
      [[deprecated("TAT::Tensor::receive deprecated, use TAT::mpi.receive instead")]] static Tensor<ScalarType, Symmetry, Name> receive(int source);
      /**
       * 像简单类型一样使用mpi但send和receive, 调用后, 一个destination返回source调用时输入tensor, 其他进程返回空张量
       */
      [[deprecated("TAT::Tensor::send_receive deprecated, use TAT::mpi.send_receive instead")]] Tensor<ScalarType, Symmetry, Name>
      send_receive(int source, int destination) const;
      /**
       * 从root进程分发张量, 使用简单的树形分发, 必须所有进程一起调用这个函数
       */
      [[deprecated("TAT::Tensor::broadcast deprecated, use TAT::mpi.broadcast instead")]] Tensor<ScalarType, Symmetry, Name>
      broadcast(int root) const;
      /**
       * 向root进程reduce张量, 使用简单的树形reduce, 必须所有进程一起调用这个函数, 最后root进程返回全部reduce的结果, 其他进程为中间结果一般无意义
       */
      template<typename Func>
      [[deprecated("TAT::Tensor::reduce deprecated, use TAT::mpi.reduce instead")]] Tensor<ScalarType, Symmetry, Name>
      reduce(int root, Func&& function) const;
      /**
       * mpi进程间同步
       */
      [[deprecated("TAT::Tensor::barrier deprecated, use TAT::mpi.barrier instead")]] static void barrier();
      /*
       * 对各个进程但张量通过求和进行reduce
       */
      [[deprecated("TAT::Tensor::summary deprecated, reduce directly")]] Tensor<ScalarType, Symmetry, Name> summary(const int root) const {
         return reduce(root, [](const auto& tensor_1, const auto& tensor_2) { return tensor_1 + tensor_2; });
      };
#endif

      const Tensor<ScalarType, Symmetry, Name>& meta_put(std::ostream&) const;
      const Tensor<ScalarType, Symmetry, Name>& data_put(std::ostream&) const;
      Tensor<ScalarType, Symmetry, Name>& meta_get(std::istream&);
      Tensor<ScalarType, Symmetry, Name>& data_get(std::istream&);

      [[nodiscard]] std::string show() const;
      [[nodiscard]] std::string dump() const;
      Tensor<ScalarType, Symmetry, Name>& load(const std::string&) &;
      Tensor<ScalarType, Symmetry, Name>&& load(const std::string& string) && {
         return std::move(load(string));
      };
   };
   /**@}*/

   /// \private
   template<typename Tensor1, typename Tensor2, typename SetNameAndName = pmr::set<std::tuple<typename Tensor1::name_t, typename Tensor2::name_t>>>
   [[nodiscard]] auto contract(const Tensor1& tensor_1, const Tensor2& tensor_2, SetNameAndName&& contract_names) {
      return tensor_1.contract(tensor_2, std::forward<SetNameAndName>(contract_names));
   }

   /// \private
   template<typename ScalarType, typename Symmetry, typename Name>
   struct TensorShape {
      Tensor<ScalarType, Symmetry, Name>* owner;
   };

   // TODO: middle 用edge operator表示一个待计算的张量, 在contract中用到
   // 因为contract的操作是这样的
   // merge gemm split
   // 上一次split可以和下一次的merge合并
   // 比较重要， 可以大幅减少对称性张量的分块
   /*
   template<typename ScalarType, typename Symmetry, typename Name>
   struct QuasiTensor {
      Tensor<ScalarType, Symmetry, Name> tensor;
      std::map<Name, std::vector<std::tuple<Name, BoseEdge<Symmetry>>>> split_map;
      std::set<Name> reversed_set;
      std::vector<Name> res_name;

      QuasiTensor

      operator Tensor<ScalarType, Symmetry, Name>() && {
         return tensor.edge_operator({}, split_map, reversed_set, {}, std::move(res_name));
      }
      operator Tensor<ScalarType, Symmetry, Name>() const& {
         return tensor.edge_operator({}, split_map, reversed_set, {}, res_name);
      }

      Tensor<ScalarType, Symmetry, Name> merge_again(
            const std::set<Name>& merge_reversed_set,
            const std::map<Name, std::vector<Name>>& merge_map,
            std::vector<Name>&& merge_res_name,
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
      QuasiTensor<ScalarType, Symmetry, Name>
   };
   */

   // TODO: lazy framework
   // 看一下idris是如何做的
   // 需要考虑深搜不可行的问题
   // 支持inplace操作

} // namespace TAT
#endif
