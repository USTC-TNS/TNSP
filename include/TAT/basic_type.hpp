/**
 * \file basic_type.hpp
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
#ifndef TAT_BASIC_TYPE_HPP
#define TAT_BASIC_TYPE_HPP

#include <complex>
#include <type_traits>

namespace TAT {
   /**
    * \brief 张量的秩的类型
    */
   using Rank = unsigned short;
   /**
    * \brief 张量数据维度大小和数据本身大小的类型
    */
   using Size = unsigned long;

   /**
    * \brief Z2对称性的类型
    */
   using Z2 = bool;
   /**
    * \brief U1对称性的类型
    */
   using U1 = int;
   /**
    * \brief 费米子数目的类型
    */
   using Fermi = short;

   /**
    * \brief 费米箭头方向的类型, false和true分别表示出入
    */
   using Arrow = bool;

   /**
    * \brief 判断一个类型是否是标量类型, 修复了std::scalar不能判断std::complex的问题
    * \tparam T 如果T是标量类型, 则value为true
    * \see is_scalar_v
    */
   template<typename T>
   struct is_scalar : std::is_scalar<T> {};
   template<typename T>
   struct is_scalar<std::complex<T>> : std::is_scalar<T> {};
   template<typename T>
   constexpr bool is_scalar_v = is_scalar<T>::value;

   /**
    * \brief 取对应的实数类型, 在svd, norm等地方会用到
    * \tparam T 如果T是std::complex<S>, 则type为S, 若T为其他标量类型, 则type为T本身, 否则为void
    * \see real_base_t
    */
   template<typename T>
   struct real_base : std::conditional<is_scalar<T>::value, T, void> {};
   template<typename T>
   struct real_base<std::complex<T>> : std::conditional<is_scalar<T>::value, T, void> {};
   template<typename T>
   using real_base_t = typename real_base<T>::type;

   /**
    * \brief 判断是否是复数类型
    * \see is_complex_v
    */
   template<typename T>
   struct is_complex : std::is_same<T, std::complex<real_base_t<T>>> {};
   template<typename T>
   constexpr bool is_complex_v = is_complex<T>::value;
   /**
    * \brief 判断是否是实数类型
    * \see is_real_v
    */
   template<typename T>
   struct is_real : std::is_same<T, real_base_t<T>> {};
   template<typename T>
   constexpr bool is_real_v = is_real<T>::value;
} // namespace TAT
#endif
