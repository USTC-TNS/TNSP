/**
 * \file symmetry.hpp
 *
 * Copyright (C) 2019-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#include <utility>

#include "../utility/common_variable.hpp"

namespace TAT {
   // A symmetry is a tuple of several int or bool, which may be fermi or bose
   // fermi is used to select fermi, bose is used to select bose, otherwise bose by default.
   template<typename T>
   struct fermi {};
   template<typename T>
   using bose = T;

   namespace detail {
      template<typename T>
      struct symmetry_unwrap_helper : type_identity<T> {};
      template<typename T>
      struct symmetry_unwrap_helper<fermi<T>> : type_identity<T> {};
      template<typename T>
      using symmetry_unwrap = typename symmetry_unwrap_helper<T>::type;

      template<typename T>
      struct is_fermi_helper : std::false_type {};
      template<typename T>
      struct is_fermi_helper<fermi<T>> : std::true_type {};
      template<typename T>
      constexpr bool is_fermi = is_fermi_helper<T>::value;
   } // namespace detail

   namespace detail {
      template<typename Result = void, typename Func, typename Tuple, std::size_t... Is>
      auto map_on_tuple(Func&& func, Tuple&& tuple, std::index_sequence<Is...>) {
         if constexpr (std::is_same_v<Result, void>) {
            (func(std::get<Is>(std::forward<Tuple>(tuple)), std::integral_constant<std::size_t, Is>()), ...);
            return 0;
         } else {
            return Result(func(std::get<Is>(std::forward<Tuple>(tuple)), std::integral_constant<std::size_t, Is>())...);
         }
      }

      template<typename Result = void, typename Func, typename Tuple1, typename Tuple2, std::size_t... Is>
      auto map_on_2tuple(Func&& func, Tuple1&& tuple1, Tuple2&& tuple2, std::index_sequence<Is...>) {
         if constexpr (std::is_same_v<Result, void>) {
            (func(std::get<Is>(std::forward<Tuple1>(tuple1)), std::get<Is>(std::forward<Tuple2>(tuple2)), std::integral_constant<std::size_t, Is>()),
             ...);
            return 0;
         } else {
            return Result(
                  func(std::get<Is>(std::forward<Tuple1>(tuple1)),
                       std::get<Is>(std::forward<Tuple2>(tuple2)),
                       std::integral_constant<std::size_t, Is>())...);
         }
      }
   } // namespace detail

   /**
    * General symmetry type, used to mark edge index the different transform rule when some symmetric operation is applied.
    *
    * For example, Symmetry<int> is U1 symmetry and Symmetry<bool> is Z2 symmetry.
    * While Symmetry<int, femri_wrap<int>> represent the conservation of particle number of two particle, one is boson and the other is fermion.
    *
    * A symmetry have a bool property to represent its parity.
    *
    * \param T integral type, such as bool or int, or use fermi<...> to mark the fermi symmetry
    */
   template<typename... T>
   struct Symmetry : std::tuple<detail::symmetry_unwrap<T>...> {
      static_assert((std::is_integral_v<detail::symmetry_unwrap<T>> && ...));

    public:
      using self_t = Symmetry<T...>; // used freq too high so alias it
      using base_tuple_t = std::tuple<detail::symmetry_unwrap<T>...>;
      using index_sequence_t = std::index_sequence_for<T...>;
      static constexpr int length = sizeof...(T);
      static constexpr std::array<bool, length> is_fermi_item = {detail::is_fermi<T>...};
      static constexpr bool is_fermi_symmetry = (detail::is_fermi<T> || ...);

    private:
      template<typename... Args>
      base_tuple_t construct_base_tuple(const Args&... args) {
         if constexpr (sizeof...(Args) < length) {
            return construct_base_tuple(args..., 0);
         } else {
            return base_tuple_t(args...);
         }
      }

    public:
      template<typename... Args, typename = std::enable_if_t<(sizeof...(Args) <= length) && (std::is_integral_v<Args> && ...)>>
      Symmetry(const Args&... args) : base_tuple_t(construct_base_tuple(args...)) {}

    private:
      const base_tuple_t& obtain_base() const {
         return static_cast<const base_tuple_t&>(*this);
      }
      base_tuple_t& obtain_base() {
         return static_cast<base_tuple_t&>(*this);
      }

    public:
      self_t& operator+=(const self_t& other_symmetry) & {
         detail::map_on_2tuple(
               [](auto& a, const auto& b, const auto&) {
                  if constexpr (std::is_same_v<remove_cvref_t<decltype(a)>, bool>) {
                     a ^= b;
                  } else {
                     a += b;
                  }
               },
               obtain_base(),
               other_symmetry.obtain_base(),
               index_sequence_t());
         return *this;
      }

      self_t& operator-=(const self_t& other_symmetry) & {
         detail::map_on_2tuple(
               [](auto& a, const auto& b, const auto&) {
                  if constexpr (std::is_same_v<remove_cvref_t<decltype(a)>, bool>) {
                     a ^= b;
                  } else {
                     a -= b;
                  }
               },
               obtain_base(),
               other_symmetry.obtain_base(),
               index_sequence_t());
         return *this;
      }

      [[nodiscard]] self_t operator+(const self_t& other_symmetry) const& {
         return detail::map_on_2tuple<self_t>(
               [](auto& a, const auto& b, const auto&) {
                  if constexpr (std::is_same_v<remove_cvref_t<decltype(a)>, bool>) {
                     return a ^ b;
                  } else {
                     return a + b;
                  }
               },
               obtain_base(),
               other_symmetry.obtain_base(),
               index_sequence_t());
      }

      [[nodiscard]] self_t operator-(const self_t& other_symmetry) const& {
         return detail::map_on_2tuple<self_t>(
               [](auto& a, const auto& b, const auto&) {
                  if constexpr (std::is_same_v<remove_cvref_t<decltype(a)>, bool>) {
                     return a ^ b;
                  } else {
                     return a - b;
                  }
               },
               obtain_base(),
               other_symmetry.obtain_base(),
               index_sequence_t());
      }
      [[nodiscard]] self_t operator-() const& {
         return detail::map_on_tuple<self_t>(
               [](const auto& a, const auto&) {
                  if constexpr (std::is_same_v<remove_cvref_t<decltype(a)>, bool>) {
                     return a;
                  } else {
                     return -a;
                  }
               },
               obtain_base(),
               index_sequence_t());
      }

      // hash
      [[nodiscard]] std::size_t hash() const {
         std::size_t seed = length;
         detail::map_on_tuple(
               [&seed](const auto& a, const auto&) {
                  using A = remove_cvref_t<decltype(a)>;
                  hash_absorb(seed, std::hash<A>{}(a));
               },
               obtain_base(),
               index_sequence_t());
         return seed;
      }

      // parity
      /**
       * Get the total parity, whether fermion or boson
       */
      [[nodiscard]] bool parity() const {
         bool result = false;
         detail::map_on_tuple(
               [&result](const auto& a, const auto& integral_constant) {
                  if constexpr (is_fermi_item[remove_cvref_t<decltype(integral_constant)>::value]) {
                     if constexpr (std::is_same_v<remove_cvref_t<decltype(a)>, bool>) {
                        result ^= a;
                     } else {
                        result ^= bool(a % 2);
                     }
                  }
               },
               obtain_base(),
               index_sequence_t());
         return result;
      }
   };

   namespace detail {
      template<typename T>
      struct is_symmetry_helper : std::false_type {};

      template<typename... T>
      struct is_symmetry_helper<Symmetry<T...>> : std::true_type {};
   } // namespace detail

   /**
    * Check whether a type is a symmetry type
    *
    * Only type specialized by Symmetry is symmetry type
    *
    * \see Symmetry
    */
   template<typename T>
   constexpr bool is_symmetry = detail::is_symmetry_helper<T>::value;

   // common used symmetries alias
   /**
    * Z2 Symmetry
    */
   using Z2 = bool;
   /**
    * U1 symmetry
    */
   using U1 = std::int32_t;

   using NoSymmetry = Symmetry<>;
   using Z2Symmetry = Symmetry<bose<Z2>>;
   using U1Symmetry = Symmetry<bose<U1>>;
   using FermiSymmetry = Symmetry<fermi<U1>>;
   using FermiZ2Symmetry = Symmetry<fermi<U1>, bose<Z2>>;
   using FermiU1Symmetry = Symmetry<fermi<U1>, bose<U1>>;
   using ParitySymmetry = Symmetry<fermi<Z2>>;
} // namespace TAT

namespace std {
   template<typename... T>
   struct hash<TAT::Symmetry<T...>> {
      size_t operator()(const TAT::Symmetry<T...>& symmetry) const {
         return symmetry.hash();
      }
   };
} // namespace std
#endif
