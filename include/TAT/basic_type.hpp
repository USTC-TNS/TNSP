/**
 * \file basic_type.hpp
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
#ifndef TAT_BASIC_TYPE_HPP
#define TAT_BASIC_TYPE_HPP

#include <complex>
#include <cstdint>
#include <type_traits>

namespace TAT {
   /**
    * \defgroup Miscellaneous
    * @{
    */

   // 下面三个类型原本是short, int, long
   // 在linux(lp64)下分别是16, 32, 64
   // 但是windows(llp64)中是16, 32, 32
   // 在io中会出现windows和linux的输入输出互相不可读的问题
   // 所以显式写成uintxx_t的格式
   // TAT中还有一些地方会出现int, 一般为调用blas和lapack的地方
   // 为32位, 在64位系统中(ilp64)和32位系统中(lp32)分别是64为和16位
   // 所以底层的blas和lapack库不可以是ilp64或者lp32版本
   /**
    * 张量的秩的类型
    */
   using Rank = std::uint16_t;
   /**
    * 张量分块数目和一个边上对称性数目的类型
    */
   using Nums = std::uint32_t;
   /**
    * 张量数据维度大小和数据本身大小的类型
    */
   using Size = std::uint64_t;

   /**
    * Z2对称性的类型
    */
   using Z2 = bool;
   /**
    * U1对称性的类型
    */
   using U1 = std::int32_t;
   /**
    * 费米子数目的类型
    */
   using Fermi = std::int16_t;

   /**
    * 费米箭头方向的类型, `false`和`true`分别表示出入
    */
   using Arrow = bool;

   /**
    * 判断一个类型是否是标量类型, 修复了`std::scalar`不能判断`std::complex`的问题
    * \tparam T 如果`T`是标量类型, 则`value`为`true`
    * \see is_scalar_v
    */
   template<typename T>
   struct is_scalar : std::is_scalar<T> {};
   /// \private
   template<typename T>
   struct is_scalar<std::complex<T>> : std::is_scalar<T> {};
   template<typename T>
   constexpr bool is_scalar_v = is_scalar<T>::value;

   /**
    * 取对应的实数类型, 在svd, norm等地方会用到
    * \tparam T 如果`T`是`std::complex<S>`, 则`type`为`S`, 若`T`为其他标量类型, 则`type`为`T`本身, 否则为`void`
    * \see real_base_t
    */
   template<typename T>
   struct real_base : std::conditional<is_scalar<T>::value, T, void> {};
   /// \private
   template<typename T>
   struct real_base<std::complex<T>> : std::conditional<is_scalar<T>::value, T, void> {};
   template<typename T>
   using real_base_t = typename real_base<T>::type;

   /**
    * 判断是否是复数类型
    * \tparam T 如果`T`为复数标量类型则`value`为`true`, 否则为`false`
    * \see is_complex_v
    */
   template<typename T>
   struct is_complex : std::is_same<T, std::complex<real_base_t<T>>> {};
   template<typename T>
   constexpr bool is_complex_v = is_complex<T>::value;
   /**
    * 判断是否是实数类型
    * \tparam T 如果`T`为实数标量类型则`value`为`true`, 否则为`false`
    * \see is_real_v
    */
   template<typename T>
   struct is_real : std::is_same<T, real_base_t<T>> {};
   template<typename T>
   constexpr bool is_real_v = is_real<T>::value;

   /**@}*/
} // namespace TAT
#endif
