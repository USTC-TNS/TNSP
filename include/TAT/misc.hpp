/**
 * \file misc.hpp
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
#ifndef TAT_MISC_HPP
#define TAT_MISC_HPP

#include <complex>
#include <type_traits>
#include <vector>

/**
 * \brief TAT is A Tensor library
 */
namespace TAT {
   /**
    * \brief 是否在非windows系统中始终输出有颜色的文本
    */
#ifdef TAT_ALWAYS_COLOR
   constexpr bool always_color = true;
#ifdef _WIN32
#warning Cannot Always Output Color in Windows
#endif
#else
   constexpr bool always_color = false;
#endif

   /**
    * \brief TAT的版本号
    */
   const std::string TAT_VERSION = "0.0.4";

   /**
    * \brief Debug模式中, 将在程序末尾打印一行友情提示
    */
   struct Evil {
      ~Evil();
   };
   const Evil evil;

   /**
    * \brief 打印警告, 有时也可能是错误, 但在非debug模式中不做事
    * \param message 待打印的话
    */
   inline void warning_or_error([[maybe_unused]] const std::string& message);

   /**
    * \brief 张量的秩的大小的类型
    */
   using Rank = unsigned short;
   /**
    * \brief 张量的分块的数目的类型
    */
   using Nums = unsigned int;
   /**
    * \brief 张量的数据维度大小和数据本身大小的类型
    */
   using Size = unsigned long;

   /**
    * \brief Z2对称性的表示的类型
    */
   using Z2 = bool;
   /**
    * \brief U1对称性的表示的类型
    */
   using U1 = long;
   /**
    * \brief 费米子数目的类型
    */
   using Fermi = int;

   /**
    * \brief 费米箭头的方向的类型, false和true分别表示出入
    */
   using Arrow = bool;

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

   /**
    * \brief 判断一个类型是否是标量类型, 修复了std::scalar不能判断std::complex的问题
    * \tparam T 如果T是标量类型, 则value为true
    */
   template<class T>
   struct is_scalar : std::is_scalar<T> {};
   /**
    * \brief 对std::complex的特殊处理
    */
   template<class T>
   struct is_scalar<std::complex<T>> : std::is_scalar<T> {};
   template<class T>
   constexpr bool is_scalar_v = is_scalar<T>::value;

   /**
    * \brief c++20的type_identity
    * \tparam T type的类型
    */
   template<class T>
   struct type_identity {
      using type = T;
   };
   template<class T>
   using type_identity_t = typename type_identity<T>::type;

   /**
    * \brief 取对应的实数类型, 在svd, norm等地方会用到
    * \tparam T 如果T是std::complex<S>, 则type为S, 否则为T本身
    */
   template<class T>
   struct real_base : type_identity<T> {};
   /**
    * \brief 对std::complex进行特化
    */
   template<class T>
   struct real_base<std::complex<T>> : type_identity<T> {};
   template<class T>
   using real_base_t = typename real_base<T>::type;

   template<class T>
   struct is_complex : std::is_same<T, std::complex<real_base_t<T>>> {};
   template<class T>
   constexpr bool is_complex_v = is_complex<T>::value;
   template<class T>
   struct is_real : std::is_same<T, real_base_t<T>> {};
   template<class T>
   constexpr bool is_real_v = is_real<T>::value;

   /**
    * \brief 用于不初始化的vector的allocator
    */
   template<class T>
   struct allocator_without_initialize : std::allocator<T> {
      template<class U>
      struct rebind {
         using other = allocator_without_initialize<U>;
      };

      /**
       * \brief 初始化函数, 如果没有参数, 且类型T可以被平凡的析构, 则不做任何初始化操作, 否则进行正常的就地初始化
       * \tparam Args 初始化的参数类型
       * \param pointer 被初始化的值的地址
       * \param arguments 初始化的参数
       */
      template<class... Args>
      void construct([[maybe_unused]] T* pointer, Args&&... arguments) {
         if constexpr (!((sizeof...(arguments) == 0) && (std::is_trivially_destructible_v<T>))) {
            new (pointer) T(arguments...);
         }
      }

      allocator_without_initialize() = default;
      template<class U>
      explicit allocator_without_initialize(allocator_without_initialize<U>) {}
   };

   /**
    * \brief 尽可能不做初始化的vector容器
    * \see allocator_without_initialize
    * \note 为了兼容性, 仅在张量的数据处使用
    */
   template<class T>
   using vector = std::vector<T, allocator_without_initialize<T>>;
} // namespace TAT
#endif
