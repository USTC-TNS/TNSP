/**
 * \file symmetry.hpp
 *
 * Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

namespace TAT {
   /**
    * \brief 所有对称性类型的基类, 用来判断一个类型是否是对称性类型
    */
   struct symmetry_base {};
   /**
    * \brief 所有费米对称性的基类, 用来判断一个类型是否是玻色对称性
    */
   struct bose_symmetry_base : symmetry_base {};
   /**
    * \brief 所有费米对称性的基类, 用来判断一个类型是否是费米对称性
    */
   struct fermi_symmetry_base : symmetry_base {};

   /**
    * \brief 判断一个类型是否是对称性类型
    * \tparam T 如果T是对称性类型, 则value为true
    */
   template<class T>
   struct is_symmetry : std::is_base_of<symmetry_base, T> {};
   template<class T>
   constexpr bool is_symmetry_v = is_symmetry<T>::value;

   /**
    * \brief 判断一个类型是否是玻色对称性类型
    * \tparam T 如果T是玻色对称性类型, 则value为true
    */
   template<class T>
   struct is_bose_symmetry : std::is_base_of<bose_symmetry_base, T> {};
   template<class T>
   constexpr bool is_bose_symmetry_v = is_bose_symmetry<T>::value;

   /**
    * \brief 判断一个类型是否是费米对称性类型
    * \tparam T 如果T是费米对称性类型, 则value为true
    */
   template<class T>
   struct is_fermi_symmetry : std::is_base_of<fermi_symmetry_base, T> {};
   template<class T>
   constexpr bool is_fermi_symmetry_v = is_fermi_symmetry<T>::value;

   template<class Derived>
   struct bose_symmetry : bose_symmetry_base {};

   template<class Derived>
   struct fermi_symmetry : fermi_symmetry_base {
      /**
       * \brief 根据对称性列表和各边是否需要翻转的情况和parity有效性给出总的parity
       *
       * 在edge_operator中, 反转边的时候, 所有奇性边会产生一个符号, 本函数求得总的符号,
       * 即统计symmetries中为奇, reverse_flag中为true, valid_mark中为true的数目的奇偶性
       * \see Tensor::edge_operator
       */
      [[nodiscard]] static bool
      get_reverse_parity(const std::vector<Derived>& symmetries, const std::vector<bool>& reverse_flag, const std::vector<bool>& valid_mark) {
         auto result = false;
         for (auto i = 0; i < symmetries.size(); i++) {
            if (reverse_flag[i] && valid_mark[i]) {
               result ^= static_cast<bool>(symmetries[i].fermi % 2);
            }
         }
         return result;
      }
      /**
       * \brief 根据对称性列表和各边的转置方案给出总parity
       *
       * 转置的parity总是有效的, 不像翻转和split, merge只会有一侧的张量有效, 毕竟这是单个张量的操作
       * \see Tensor::edge_operator
       */
      [[nodiscard]] static bool get_transpose_parity(const std::vector<Derived>& symmetries, const std::vector<Rank>& transpose_plan) {
         auto result = false;
         for (auto i = 0; i < symmetries.size(); i++) {
            for (auto j = i + 1; j < symmetries.size(); j++) {
               if (transpose_plan[i] > transpose_plan[j]) {
                  result ^= (static_cast<bool>(symmetries[i].fermi % 2) && static_cast<bool>(symmetries[j].fermi % 2));
               }
            }
         }
         return result;
      }
      /**
       * \brief 根据对称性列表和split或merge的方案以及parity有效性给出总的parity
       *
       * \note sum_{i!=j} s_i s_j = ((sum s_i)^2 - sum s_i^2)/2
       */
      [[nodiscard]] static bool get_split_merge_parity(
            const std::vector<Derived>& symmetries,    // before merge length
            const std::vector<Rank>& split_merge_flag, // before merge length
            const std::vector<bool>& valid_mark) {     // after merge length
         auto result = false;
         for (auto split_merge_group_position = 0, split_merge_begin_position = 0, split_merge_end_position = 0;
              split_merge_group_position < valid_mark.size();
              split_merge_group_position++) {
            // split_merge_group_position point to after merge position
            // begin_position and end_position point to before merge position
            while (split_merge_end_position < symmetries.size() && split_merge_flag[split_merge_end_position] == split_merge_group_position) {
               split_merge_end_position++;
            }
            if (valid_mark[split_merge_group_position]) {
               auto sum_of_parity = 0;
               auto sum_of_parity_square = 0;
               for (auto position_in_group = split_merge_begin_position; position_in_group < split_merge_end_position; position_in_group++) {
                  auto this_parity = symmetries[position_in_group].fermi;
                  sum_of_parity += this_parity;
                  sum_of_parity_square += this_parity * this_parity;
               }
               result ^= static_cast<bool>(((sum_of_parity * sum_of_parity - sum_of_parity_square) / 2) % 2);
            }
            split_merge_begin_position = split_merge_end_position;
         }
         return result;
      }
   };

   /**
    * \brief 无对称性
    */
   struct NoSymmetry : bose_symmetry<NoSymmetry> {
      [[nodiscard]] auto information() const {
         return 0;
      }
   };
   inline NoSymmetry operator+(const NoSymmetry&, const NoSymmetry&) {
      return NoSymmetry();
   }
   inline NoSymmetry& operator+=(NoSymmetry& symmetry, const NoSymmetry&) {
      return symmetry;
   }
   inline NoSymmetry operator-(const NoSymmetry&) {
      return NoSymmetry();
   }

   /**
    * \brief Z2对称性
    */
   struct Z2Symmetry : bose_symmetry<Z2Symmetry> {
      Z2 z2;

      Z2Symmetry(const Z2 z2 = false) : z2(z2) {}

      [[nodiscard]] auto information() const {
         return z2;
      }
   };
   inline Z2Symmetry operator+(const Z2Symmetry& symmetry_1, const Z2Symmetry& symmetry_2) {
      return Z2Symmetry(symmetry_1.z2 ^ symmetry_2.z2);
   }
   inline Z2Symmetry& operator+=(Z2Symmetry& symmetry_1, const Z2Symmetry& symmetry_2) {
      symmetry_1.z2 ^= symmetry_2.z2;
      return symmetry_1;
   }
   inline Z2Symmetry operator-(const Z2Symmetry& symmetry) {
      return Z2Symmetry(-symmetry.z2);
   }

   /**
    * \brief U1对称性
    */
   struct U1Symmetry : bose_symmetry<U1Symmetry> {
      U1 u1;

      U1Symmetry(const U1 u1 = 0) : u1(u1) {}

      [[nodiscard]] auto information() const {
         return u1;
      }
   };
   inline U1Symmetry operator+(const U1Symmetry& symmetry_1, const U1Symmetry& symmetry_2) {
      return U1Symmetry(symmetry_1.u1 + symmetry_2.u1);
   }
   inline U1Symmetry& operator+=(U1Symmetry& symmetry_1, const U1Symmetry& symmetry_2) {
      symmetry_1.u1 += symmetry_2.u1;
      return symmetry_1;
   }
   inline U1Symmetry operator-(const U1Symmetry& symmetry) {
      return U1Symmetry(-symmetry.u1);
   }

   /**
    * \brief 费米的无对称性
    */
   struct FermiSymmetry : fermi_symmetry<FermiSymmetry> {
      Fermi fermi;

      FermiSymmetry(const Fermi fermi = 0) : fermi(fermi) {}

      [[nodiscard]] auto information() const {
         return fermi;
      }
   };
   inline FermiSymmetry operator+(const FermiSymmetry& symmetry_1, const FermiSymmetry& symmetry_2) {
      return FermiSymmetry(symmetry_1.fermi + symmetry_2.fermi);
   }
   inline FermiSymmetry& operator+=(FermiSymmetry& symmetry_1, const FermiSymmetry& symmetry_2) {
      symmetry_1.fermi += symmetry_2.fermi;
      return symmetry_1;
   }
   inline FermiSymmetry operator-(const FermiSymmetry& symmetry) {
      return FermiSymmetry(-symmetry.fermi);
   }

   /**
    * \brief 费米的Z2对称性
    */
   struct FermiZ2Symmetry : fermi_symmetry<FermiZ2Symmetry> {
      Fermi fermi;
      Z2 z2;

      FermiZ2Symmetry(const Fermi fermi = 0, const Z2 z2 = false) : fermi(fermi), z2(z2) {}

      [[nodiscard]] auto information() const {
         return std::tie(fermi, z2);
      }
   };
   inline FermiZ2Symmetry operator+(const FermiZ2Symmetry& symmetry_1, const FermiZ2Symmetry& symmetry_2) {
      return FermiZ2Symmetry(symmetry_1.fermi + symmetry_2.fermi, symmetry_1.z2 ^ symmetry_2.z2);
   }
   inline FermiZ2Symmetry& operator+=(FermiZ2Symmetry& symmetry_1, const FermiZ2Symmetry& symmetry_2) {
      symmetry_1.fermi += symmetry_2.fermi;
      symmetry_1.z2 ^= symmetry_2.z2;
      return symmetry_1;
   }
   inline FermiZ2Symmetry operator-(const FermiZ2Symmetry& symmetry) {
      return FermiZ2Symmetry(-symmetry.fermi, -symmetry.z2);
   }

   /**
    * \brief 费米的U1对称性
    */
   struct FermiU1Symmetry : fermi_symmetry<FermiU1Symmetry> {
      Fermi fermi;
      U1 u1;

      FermiU1Symmetry(const Fermi fermi = 0, const U1 u1 = 0) : fermi(fermi), u1(u1) {}

      [[nodiscard]] auto information() const {
         return std::tie(fermi, u1);
      }
   };
   inline FermiU1Symmetry operator+(const FermiU1Symmetry& symmetry_1, const FermiU1Symmetry& symmetry_2) {
      return FermiU1Symmetry(symmetry_1.fermi + symmetry_2.fermi, symmetry_1.u1 + symmetry_2.u1);
   }
   inline FermiU1Symmetry& operator+=(FermiU1Symmetry& symmetry_1, const FermiU1Symmetry& symmetry_2) {
      symmetry_1.fermi += symmetry_2.fermi;
      symmetry_1.u1 += symmetry_2.u1;
      return symmetry_1;
   }
   inline FermiU1Symmetry operator-(const FermiU1Symmetry& symmetry) {
      return FermiU1Symmetry(-symmetry.fermi, -symmetry.u1);
   }

   // 此处将可被c++20的operator<=>替换
   // 生成每个对称性的对称性的比较运算符重载
#define TAT_DEFINE_SINGLE_SYMMETRY_OPERATOR(SYM, OP, EXP)         \
   inline bool OP(const SYM& symmetry_1, const SYM& symmetry_2) { \
      return EXP;                                                 \
   }
#define TAT_DEFINE_SYMMETRY_ALL_OPERATOR(SYM)                                                                 \
   TAT_DEFINE_SINGLE_SYMMETRY_OPERATOR(SYM, operator==, symmetry_1.information() == symmetry_2.information()) \
   TAT_DEFINE_SINGLE_SYMMETRY_OPERATOR(SYM, operator!=, symmetry_1.information() != symmetry_2.information()) \
   TAT_DEFINE_SINGLE_SYMMETRY_OPERATOR(SYM, operator>=, symmetry_1.information() >= symmetry_2.information()) \
   TAT_DEFINE_SINGLE_SYMMETRY_OPERATOR(SYM, operator<=, symmetry_1.information() <= symmetry_2.information()) \
   TAT_DEFINE_SINGLE_SYMMETRY_OPERATOR(SYM, operator>, symmetry_1.information() > symmetry_2.information())   \
   TAT_DEFINE_SINGLE_SYMMETRY_OPERATOR(SYM, operator<, symmetry_1.information() < symmetry_2.information())

   TAT_DEFINE_SYMMETRY_ALL_OPERATOR(NoSymmetry)
   TAT_DEFINE_SYMMETRY_ALL_OPERATOR(Z2Symmetry)
   TAT_DEFINE_SYMMETRY_ALL_OPERATOR(U1Symmetry)
   TAT_DEFINE_SYMMETRY_ALL_OPERATOR(FermiSymmetry)
   TAT_DEFINE_SYMMETRY_ALL_OPERATOR(FermiZ2Symmetry)
   TAT_DEFINE_SYMMETRY_ALL_OPERATOR(FermiU1Symmetry)
#undef TAT_DEFINE_SYMMETRY_ALL_OPERATOR
#undef TAT_DEFINE_SINGLE_SYMMETRY_OPERATOR
} // namespace TAT
#endif
