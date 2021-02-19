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
   struct general_symmetry : std::tuple<fermi_unwrap_t<T>...> {
      using self_class = general_symmetry<T...>;
      using base_class = std::tuple<fermi_unwrap_t<T>...>;
      static constexpr int length = sizeof...(T);
      static constexpr std::array<bool, length> is_fermi = {fermi_wrapped_v<T>...};
      static constexpr bool is_fermi_symmetry = (fermi_wrapped_v<T> || ...);
      using index_sequence = std::index_sequence_for<T...>;

      template<typename... Args>
      base_class construct_symmetry_tuple(const Args&... args) {
         if constexpr (sizeof...(Args) == length) {
            return base_class(args...);
         } else {
            return construct_symmetry_tuple(args..., 0);
         }
      }
      template<typename... Args, std::enable_if_t<sizeof...(Args) <= length && (std::is_integral_v<Args> && ...), int> = 0>
      general_symmetry(const Args&... args) : base_class(construct_symmetry_tuple(args...)) {}

      template<typename Item>
      static Item& inplace_plus_item(Item& a, const Item& b) {
         if constexpr (std::is_same_v<Item, bool>) {
            return a ^= b;
         } else {
            return a += b;
         }
      }
      template<std::size_t... Is>
      static self_class& inplace_plus_symmetry(self_class& symmetry_1, const self_class& symmetry_2, std::index_sequence<Is...>) {
         (inplace_plus_item(std::get<Is>(symmetry_1), std::get<Is>(symmetry_2)), ...);
         return symmetry_1;
      }
      self_class& operator+=(const self_class& other_symmetry) & {
         return inplace_plus_symmetry(*this, other_symmetry, index_sequence());
      }

      template<typename Item>
      static Item plus_item(const Item& a, const Item& b) {
         if constexpr (std::is_same_v<Item, bool>) {
            return a ^ b;
         } else {
            return a + b;
         }
      }
      template<size_t... Is>
      static self_class plus_symmetry(const self_class& symmetry_1, const self_class& symmetry_2, std::index_sequence<Is...>) {
         return self_class(plus_item(std::get<Is>(symmetry_1), std::get<Is>(symmetry_2))...);
      }
      self_class operator+(const self_class& other_symmetry) const& {
         return plus_symmetry(*this, other_symmetry, index_sequence());
      }

      template<typename Item>
      static Item minus_item(const Item& a) {
         if constexpr (std::is_same_v<Item, bool>) {
            return a;
         } else {
            return -a;
         }
      }
      template<size_t... Is>
      static self_class minus_symmetry(const self_class& symmetry, std::index_sequence<Is...>) {
         return self_class(minus_item(std::get<Is>(symmetry))...);
      }
      self_class operator-() const& {
         return minus_symmetry(*this, index_sequence());
      }

      template<typename Number, typename = std::enable_if_t<std::is_integral_v<Number>>>
      static bool fermi_to_bool(Number number) {
         return bool(number % 2);
      }
      static bool fermi_to_bool(bool number) {
         return number;
      }

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
      template<typename VectorSymmetry, typename VectorBool1, typename VectorBool2>
      [[nodiscard]] static bool get_reverse_parity(const VectorSymmetry& symmetries, const VectorBool1& reverse_flag, const VectorBool2& valid_mark) {
         auto result = false;
         for (Rank i = 0; i < symmetries.size(); i++) {
            if (reverse_flag[i] && valid_mark[i]) {
               result ^= fermi_to_bool(symmetries[i].fermi);
            }
         }
         return result;
      }
      /**
       * 给出转置时产生的parity
       *
       * \param symmetries 某分块在各个边的对称性情况
       * \param transpose_plan 转置方案
       *
       * 转置的parity总是有效的, 而翻转和split, merge涉及的两个张量只会有一侧有效, 毕竟这是单个张量的操作
       * \see Tensor::edge_operator
       */
      template<typename VectorSymmetry, typename VectorRank>
      [[nodiscard]] static bool get_transpose_parity(const VectorSymmetry& symmetries, const VectorRank& transpose_plan) {
         auto result = false;
         for (Rank i = 0; i < symmetries.size(); i++) {
            for (Rank j = i + 1; j < symmetries.size(); j++) {
               if (transpose_plan[i] > transpose_plan[j]) {
                  result ^= (fermi_to_bool(symmetries[i].fermi) && fermi_to_bool(symmetries[j].fermi));
               }
            }
         }
         return result;
      }
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
      template<typename VectorSymmetry, typename VectorRank, typename VectorBool>
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
               auto sum_of_parity = 0l;
               auto sum_of_parity_square = 0l;
               for (auto position_in_group = split_merge_begin_position; position_in_group < split_merge_end_position; position_in_group++) {
                  auto this_parity = symmetries[position_in_group].fermi;
                  sum_of_parity += this_parity;
                  sum_of_parity_square += this_parity * this_parity;
               }
               result ^= bool(((sum_of_parity * sum_of_parity - sum_of_parity_square) / 2) % 2);
            }
            split_merge_begin_position = split_merge_end_position;
         }
         return result;
      }
   };

   using NoSymmetry = general_symmetry<>;
   using Z2Symmetry = general_symmetry<Z2>;
   using U1Symmetry = general_symmetry<U1>;
   using FermiSymmetry = general_symmetry<fermi_wrap<U1>>;
   using FermiZ2Symmetry = general_symmetry<fermi_wrap<U1>, Z2>;
   using FermiU1Symmetry = general_symmetry<fermi_wrap<U1>, U1>;

   /**
    * 判断一个类型是否是对称性类型
    *
    * \tparam T 如果`T`是对称性类型, 则`value`为`true`
    * \see is_symmetry_v
    */
   template<typename T>
   struct is_symmetry : std::bool_constant<false> {};
   template<typename... T>
   struct is_symmetry<general_symmetry<T...>> : std::bool_constant<true> {};
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
