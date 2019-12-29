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
#ifndef TAT_MISC_HPP_
#   define TAT_MISC_HPP_

#define TAT_VERSION 0.0.4

#ifdef NDEBUG
#   define TAT_WARNING(msg) std::clog
#else
#   define TAT_WARNING(msg) std::clog << msg << std::endl
#endif

#include <complex>
#include <iostream>

namespace TAT {
   using Rank = unsigned int;
   using Nums = unsigned long;
   using Size = unsigned long long;
   using Z2 = bool;
   using U1 = long;
   using Fermi = int;

   template<class T>
   struct is_scalar : std::is_scalar<T> {};
   template<class T>
   struct is_scalar<std::complex<T>> : std::is_scalar<T> {};
   template<class T>
   static constexpr bool is_scalar_v = is_scalar<T>::value;

   template<class T>
   struct type_identity {
      using type = T;
   };
   template<class T>
   using type_identity_t = typename type_identity<T>::type;

   template<class T>
   struct real_base : type_identity<T> {};
   template<class T>
   struct real_base<std::complex<T>> : type_identity<T> {};
   template<class T>
   using real_base_t = typename real_base<T>::type;

   template<class T>
   struct remove_cvref : std::remove_cv<typename std::remove_reference<T>::type> {};
   template<class T>
   using remove_cvref_t = typename remove_cvref<T>::type;

   template<class T, class U>
   struct is_same_nocvref : std::is_same<remove_cvref_t<T>, remove_cvref_t<U>> {};
   template<class T, class U>
   static constexpr bool is_same_nocvref_v = is_same_nocvref<T, U>::value;

   template<class T>
   struct allocator_without_initialize : std::allocator<T> {
      template<class U>
      struct rebind {
         using other = allocator_without_initialize<U>;
      };

      template<class... Args>
      void construct([[maybe_unused]] T* p, Args&&... args) {
         if constexpr (!((sizeof...(args) == 0) && (std::is_trivially_destructible_v<T>))) {
            new (p) T(args...);
         }
      }

      allocator_without_initialize() = default;
      template<class U>
      allocator_without_initialize(allocator_without_initialize<U>) {}
   };

   template<class T>
   struct vector : public std::vector<T, allocator_without_initialize<T>> {
      using std::vector<T, allocator_without_initialize<T>>::vector;
   };

   template<class T>
   std::ostream& operator<<(std::ostream& out, const vector<T>& vec);
   template<class T>
   std::ostream& operator<=(std::ostream& out, const vector<T>& vec);
   template<class T>
   std::istream& operator>=(std::istream& in, vector<T>& vec);
} // namespace TAT

#endif
