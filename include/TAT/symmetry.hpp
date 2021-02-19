/**
 * \file symmetry.hpp
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
#ifndef TAT_SYMMETRY_HPP
#define TAT_SYMMETRY_HPP

#include <tuple>

#include "basic_type.hpp"

namespace TAT {
   /**
    * \defgroup Symmetry
    * 对称性模块
    *
    * 如需自定义新的对称性类型CustomizedSymmetry, 需继承自TAT::bose_symmetry<CustomizedSymmetry>
    * 或者TAT::fermi_symmetry<CustomizedSymmetry>并实现比较运算以及至少`operator+`, `operator+=`, `operator-`三个运算符
    * @{
    */

   template<typename T>
   struct fermi_wrap {};

   template<typename T>
   struct fermi_unwrap : type_identity<T> {};
   template<typename T>
   struct fermi_unwrap<fermi_wrap<T>> : type_identity<T> {};
   template<typename T>
   using fermi_unwrap_t = typename fermi_unwrap<T>::type;

   template<typename T>
   struct fermi_wrapped : std::bool_constant<false> {};
   template<typename T>
   struct fermi_wrapped<fermi_wrap<T>> : std::bool_constant<true> {};
   template<typename T>
   constexpr bool fermi_wrapped_v = fermi_wrapped<T>::value;

   template<typename... T>
   struct symmetry_t : std::tuple<fermi_unwrap_t<T>...> {
   private:
      using self_t = symmetry_t<T...>;

   public:
      using base_tuple = std::tuple<fermi_unwrap_t<T>...>;
      static constexpr int length = sizeof...(T);
      static constexpr std::array<bool, length> is_fermi_item = {fermi_wrapped_v<T>...};
      static constexpr bool is_fermi_symmetry = (fermi_wrapped_v<T> || ...);
      using index_sequence = std::index_sequence_for<T...>;

   private:
      template<typename... Args>
      base_tuple construct_base_tuple(const Args&... args) {
         if constexpr (sizeof...(Args) == length) {
            return base_tuple(args...);
         } else {
            return construct_base_tuple(args..., 0);
         }
      }

   public:
      template<typename... Args, std::enable_if_t<sizeof...(Args) <= length && (std::is_integral_v<Args> && ...), int> = 0>
      symmetry_t(const Args&... args) : base_tuple(construct_base_tuple(args...)) {}

   private:
      template<typename Item>
      static Item& inplace_plus_item(Item& a, const Item& b) {
         if constexpr (std::is_same_v<Item, bool>) {
            return a ^= b;
         } else {
            return a += b;
         }
      }
      template<std::size_t... Is>
      static self_t& inplace_plus_symmetry(self_t& symmetry_1, const self_t& symmetry_2, std::index_sequence<Is...>) {
         (inplace_plus_item(std::get<Is>(symmetry_1), std::get<Is>(symmetry_2)), ...);
         return symmetry_1;
      }

   public:
      self_t& operator+=(const self_t& other_symmetry) & {
         return inplace_plus_symmetry(*this, other_symmetry, index_sequence());
      }

   private:
      template<typename Item>
      static Item plus_item(const Item& a, const Item& b) {
         if constexpr (std::is_same_v<Item, bool>) {
            return a ^ b;
         } else {
            return a + b;
         }
      }
      template<size_t... Is>
      static self_t plus_symmetry(const self_t& symmetry_1, const self_t& symmetry_2, std::index_sequence<Is...>) {
         return self_t(plus_item(std::get<Is>(symmetry_1), std::get<Is>(symmetry_2))...);
      }

   public:
      self_t operator+(const self_t& other_symmetry) const& {
         return plus_symmetry(*this, other_symmetry, index_sequence());
      }

   private:
      template<typename Item>
      static Item minus_item(const Item& a) {
         if constexpr (std::is_same_v<Item, bool>) {
            return a;
         } else {
            return -a;
         }
      }
      template<size_t... Is>
      static self_t minus_symmetry(const self_t& symmetry, std::index_sequence<Is...>) {
         return self_t(minus_item(std::get<Is>(symmetry))...);
      }

   public:
      self_t operator-() const& {
         return minus_symmetry(*this, index_sequence());
      }

   public:
      template<std::size_t Index>
      bool get_item_parity() const {
         if constexpr (is_fermi_item[Index]) {
            const auto quantum_number = std::get<Index>(*this);
            if constexpr (std::is_same_v<decltype(quantum_number), bool>) {
               return quantum_number;
            } else {
               return bool(quantum_number % 2);
            }
         } else {
            return false;
         }
      }

   private:
      template<std::size_t Index>
      static void update_symmetry_result_single_item(bool& result, const self_t& symmetry) {
         if constexpr (is_fermi_item[Index]) {
            result ^= symmetry.template get_item_parity<Index>();
         }
      }
      template<std::size_t... Is>
      static void update_symmetry_result(bool& result, const self_t& symmetry, std::index_sequence<Is...>) {
         (update_symmetry_result_single_item<Is>(result, symmetry), ...);
      }

   public:
      /**
       * 给出翻转各边时产生的parity
       *
       * \param symmetries 某分块在各个边的对称性情况
       * \param reverse_flag 各个边是否需要翻转的列表
       * \param valid_mark 各个边翻转的有效性
       *
       * 在edge_operator中, 反转边的时候, 所有奇性边会产生一个符号, 本函数求得总的符号,
       * 即统计symmetries中为奇, reverse_flag中为true, valid_mark中为true的数目的奇偶性
       * \see Tensor::edge_operator
       */
      template<
            typename VectorSymmetry,
            typename VectorBool1,
            typename VectorBool2,
            typename = std::enable_if_t<is_list_of_v<VectorSymmetry, self_t> && is_list_of_v<VectorBool1, bool> && is_list_of_v<VectorBool2, bool>>>
      [[nodiscard]] static bool get_reverse_parity(const VectorSymmetry& symmetries, const VectorBool1& reverse_flag, const VectorBool2& valid_mark) {
         auto result = false;
         for (Rank i = 0; i < symmetries.size(); i++) {
            if (reverse_flag[i] && valid_mark[i]) {
               update_symmetry_result(result, symmetries[i], index_sequence());
            }
         }
         return result;
      }

   private:
      template<std::size_t Index>
      static void update_symmetry_result_single_item(bool& result, const self_t& symmetry_1, const self_t& symmetry_2) {
         if constexpr (is_fermi_item[Index]) {
            result ^= symmetry_1.get_item_parity<Index>() ^ symmetry_2.get_item_parity<Index>();
         }
      }
      template<std::size_t... Is>
      static void update_symmetry_result(bool& result, const self_t& symmetry_1, const self_t& symmetry_2, std::index_sequence<Is...>) {
         (update_symmetry_result_single_item<Is>(result, symmetry_1, symmetry_2), ...);
      }

   public:
      /**
       * 给出转置时产生的parity
       *
       * \param symmetries 某分块在各个边的对称性情况
       * \param transpose_plan 转置方案
       *
       * 转置的parity总是有效的, 而翻转和split, merge涉及的两个张量只会有一侧有效, 毕竟这是单个张量的操作
       * \see Tensor::edge_operator
       */
      template<
            typename VectorSymmetry,
            typename VectorRank,
            typename = std::enable_if_t<is_list_of_v<VectorSymmetry, self_t> && is_list_of_v<VectorRank, Rank>>>
      [[nodiscard]] static bool get_transpose_parity(const VectorSymmetry& symmetries, const VectorRank& transpose_plan) {
         auto result = false;
         for (Rank i = 0; i < symmetries.size(); i++) {
            for (Rank j = i + 1; j < symmetries.size(); j++) {
               if (transpose_plan[i] > transpose_plan[j]) {
                  update_symmetry_result(result, symmetries[i], symmetries[j], index_sequence());
               }
            }
         }
         return result;
      }

   private:
      template<std::size_t Index, typename VectorSymmetry, typename = std::enable_if_t<is_list_of_v<VectorSymmetry, self_t>>>
      static void update_symmetry_result_single_item(
            bool& result,
            const VectorSymmetry& symmetries,
            Rank split_merge_begin_position,
            Rank split_merge_end_position) {
         if constexpr (is_fermi_item[Index]) {
            auto sum_of_parity = 0l;
            auto sum_of_parity_square = 0l;
            for (auto position_in_group = split_merge_begin_position; position_in_group < split_merge_end_position; position_in_group++) {
               auto this_parity = std::get<Index>(symmetries[position_in_group]);
               sum_of_parity += this_parity;
               sum_of_parity_square += this_parity * this_parity;
            }
            result ^= bool(((sum_of_parity * sum_of_parity - sum_of_parity_square) / 2) % 2);
         }
      }
      template<typename VectorSymmetry, std::size_t... Is, typename = std::enable_if_t<is_list_of_v<VectorSymmetry, self_t>>>
      static void update_symmetry_result(
            bool& result,
            const VectorSymmetry& symmetries,
            Rank split_merge_begin_position,
            Rank split_merge_end_position,
            std::index_sequence<Is...>) {
         (update_symmetry_result_single_item<Is>(result, symmetries, split_merge_begin_position, split_merge_end_position), ...);
      }

   public:
      /**
       * 给出merge或split时产生的parity
       *
       * \param symmetries 某分块在各个边的对称性情况
       * \param split_merge_flag merge或split的方案
       * \param valid_mark 各个merge或split的有效性
       *
       * \note 实际上每一个merge或split操作都是一个全翻转,
       * 而\f$\sum_{i\neq j} s_i s_j = \frac{(\sum s_i)^2 - \sum s_i^2}{2}\f$, 所以可以更简单的实现
       */
      template<
            typename VectorSymmetry,
            typename VectorRank,
            typename VectorBool,
            typename = std::enable_if_t<is_list_of_v<VectorSymmetry, self_t> && is_list_of_v<VectorRank, Rank> && is_list_of_v<VectorBool, bool>>>
      [[nodiscard]] static bool get_split_merge_parity(
            const VectorSymmetry& symmetries,   // before merge length
            const VectorRank& split_merge_flag, // before merge length
            const VectorBool& valid_mark) {     // after merge length
         auto result = false;
         for (Rank split_merge_group_position = 0, split_merge_begin_position = 0, split_merge_end_position = 0;
              split_merge_group_position < valid_mark.size();
              split_merge_group_position++) {
            // split_merge_group_position point to after merge position
            // begin_position and end_position point to before merge position
            while (split_merge_end_position < symmetries.size() && split_merge_flag[split_merge_end_position] == split_merge_group_position) {
               split_merge_end_position++;
            }
            if (valid_mark[split_merge_group_position]) {
               update_symmetry_result(result, symmetries, split_merge_begin_position, split_merge_end_position, index_sequence());
            }
            split_merge_begin_position = split_merge_end_position;
         }
         return result;
      }

   private:
      template<std::size_t Head, std::size_t... Tail>
      auto loop_to_get_first_parity_item() const {
         if constexpr (is_fermi_item[Head]) {
            return std::get<Head>(*this);
         } else {
            return loop_to_get_first_parity_item<Tail...>();
         }
      }
      template<std::size_t... Is>
      auto loop_to_get_first_parity(std::index_sequence<Is...>) const {
         return loop_to_get_first_parity_item<Is...>();
      }

   public:
      auto get_first_parity() const {
         return loop_to_get_first_parity(index_sequence());
      }

   private:
      template<std::size_t... Is>
      auto loop_to_get_total_parity(std::index_sequence<Is...>) const {
         return (get_item_parity<Is>() ^ ...);
      }

   public:
      // TODO: 多个fermion时应该如何做?
      auto get_total_parity() const {
         return loop_to_get_total_parity(index_sequence());
      }
   };

   using NoSymmetry = symmetry_t<>;
   using Z2Symmetry = symmetry_t<Z2>;
   using U1Symmetry = symmetry_t<U1>;
   using FermiSymmetry = symmetry_t<fermi_wrap<U1>>;
   using FermiZ2Symmetry = symmetry_t<fermi_wrap<U1>, Z2>;
   using FermiU1Symmetry = symmetry_t<fermi_wrap<U1>, U1>;

   /**
    * 判断一个类型是否是对称性类型
    *
    * \tparam T 如果`T`是对称性类型, 则`value`为`true`
    * \see is_symmetry_v
    */
   template<typename T>
   struct is_symmetry : std::bool_constant<false> {};
   template<typename... T>
   struct is_symmetry<symmetry_t<T...>> : std::bool_constant<true> {};
   template<typename T>
   constexpr bool is_symmetry_v = is_symmetry<T>::value;

   /**
    * 判断一个类型是否是玻色对称性类型
    *
    * \tparam T 如果`T`是玻色对称性类型, 则`value`为`true`
    * \see is_bose_symmetry_v
    */
   template<typename T>
   struct is_bose_symmetry : std::bool_constant<!T::is_fermi_symmetry> {};
   template<typename T>
   constexpr bool is_bose_symmetry_v = is_bose_symmetry<T>::value;

   /**
    * 判断一个类型是否是费米对称性类型
    *
    * \tparam T 如果`T`是费米对称性类型, 则`value`为`true`
    * \see is_fermi_symmetry_v
    */
   template<typename T>
   struct is_fermi_symmetry : std::bool_constant<T::is_fermi_symmetry> {};
   template<typename T>
   constexpr bool is_fermi_symmetry_v = is_fermi_symmetry<T>::value;
   /**@}*/
} // namespace TAT
#endif
