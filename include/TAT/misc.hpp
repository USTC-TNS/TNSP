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
#include <iostream>
#include <map>
#include <type_traits>
#include <vector>

namespace TAT {
   const std::string TAT_VERSION = "0.0.4";

   inline void TAT_WARNING([[maybe_unused]] const std::string& msg) {
#ifndef NDEBUG
      std::cerr << msg << std::endl;
#endif
   }

   using Rank = unsigned short;
   using Nums = unsigned int;
   using Size = unsigned long;
   using Z2 = bool;
   using U1 = long;
   using Fermi = int;

   struct symmetry_base {};
   struct fermi_symmetry_base : symmetry_base {};
   template<class T>
   struct is_symmetry : std::is_base_of<symmetry_base, T> {};
   template<class T>
   constexpr bool is_symmetry_v = is_symmetry<T>::value;
   template<class T>
   struct is_fermi_symmetry : std::is_base_of<fermi_symmetry_base, T> {};
   template<class T>
   constexpr bool is_fermi_symmetry_v = is_fermi_symmetry<T>::value;

   template<class T>
   struct is_scalar : std::is_scalar<T> {};
   template<class T>
   struct is_scalar<std::complex<T>> : std::is_scalar<T> {};
   template<class T>
   constexpr bool is_scalar_v = is_scalar<T>::value;

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

   template<class Key, class T>
   struct map : public std::map<Key, T> {
      using std::map<Key, T>::map;

      map(const T s) : std::map<Key, T>({{Key(), s}}) {}
   };

   template<class Key, class T>
   std::ostream& operator<<(std::ostream& out, const map<Key, T>& edge);
   template<class Key, class T>
   std::ostream& operator<=(std::ostream& out, const map<Key, T>& edge);
   template<class Key, class T>
   std::istream& operator>=(std::istream& in, map<Key, T>& edge);

   template<class T>
   auto std_begin(const T& v) {
      if constexpr (std::is_pointer_v<T>) {
         return std::begin(*v);
      } else {
         return std::begin(v);
      }
   }

   template<class T>
   auto std_end(const T& v) {
      if constexpr (std::is_pointer_v<T>) {
         return std::end(*v);
      } else {
         return std::end(v);
      }
   }
} // namespace TAT

#endif
