/**
 * \file symmetry.hpp
 *
 * Copyright (C) 2019-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include "../utility/allocator.hpp"
#include "../utility/common_variable.hpp"

namespace TAT {
   // A symmetry is a tuple of several int or bool, which may be fermi or bose
   // fermi_wrap is used to select fermi, otherwise bose by default.
   template<typename T>
   struct fermi_wrap {};

   namespace detail {
      template<typename T>
      struct fermi_unwrap_helper : type_identity<T> {};
      template<typename T>
      struct fermi_unwrap_helper<fermi_wrap<T>> : type_identity<T> {};
      template<typename T>
      using fermi_unwrap = typename fermi_unwrap_helper<T>::type;

      template<typename T>
      struct fermi_wrapped_helper : std::false_type {};
      template<typename T>
      struct fermi_wrapped_helper<fermi_wrap<T>> : std::true_type {};
      template<typename T>
      constexpr bool fermi_wrapped = fermi_wrapped_helper<T>::value;
   } // namespace detail

   namespace detail {
      template<typename Result, typename Func, typename Tuple, std::size_t... Is>
      auto map_on_tuple_helper(Func&& func, Tuple&& tuple, std::index_sequence<Is...>) {
         if constexpr (std::is_same_v<Result, void>) {
            (func(std::get<Is>(std::forward<Tuple>(tuple)), std::integral_constant<std::size_t, Is>()), ...);
            return 0;
         } else {
            return Result(func(std::get<Is>(std::forward<Tuple>(tuple)), std::integral_constant<std::size_t, Is>())...);
         }
      }
      template<typename Result = void, typename Func, typename Tuple>
      auto map_on_tuple(Func&& func, Tuple&& tuple) {
         using index_sequence_t = std::make_index_sequence<std::tuple_size_v<remove_cvref_t<Tuple>>>;
         return map_on_tuple_helper<Result>(std::forward<Func>(func), std::forward<Tuple>(tuple), index_sequence_t());
      }

      template<typename Result, typename Func, typename Tuple1, typename Tuple2, std::size_t... Is>
      auto map_on_2tuple_helper(Func&& func, Tuple1&& tuple1, Tuple2&& tuple2, std::index_sequence<Is...>) {
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
      template<typename Result = void, typename Func, typename Tuple1, typename Tuple2>
      auto map_on_2tuple(Func&& func, Tuple1&& tuple1, Tuple2&& tuple2) {
         using index_sequence_t = std::make_index_sequence<std::tuple_size_v<remove_cvref_t<Tuple1>>>;
         return map_on_2tuple_helper<Result>(
               std::forward<Func>(func),
               std::forward<Tuple1>(tuple1),
               std::forward<Tuple2>(tuple2),
               index_sequence_t());
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
    * \param T integral type, such as bool or int, or use fermi_wrap<...> to mark the fermi symmetry
    */
   template<typename... T>
   struct Symmetry : std::tuple<detail::fermi_unwrap<T>...> {
      static_assert((std::is_integral_v<detail::fermi_unwrap<T>> && ...));

      using self_t = Symmetry<T...>; // used freq too high so alias it

      using base_tuple_t = std::tuple<detail::fermi_unwrap<T>...>;
      static constexpr int length = sizeof...(T);
      static constexpr std::array<bool, length> is_fermi_item = {detail::fermi_wrapped<T>...};
      static constexpr bool is_fermi_symmetry = (detail::fermi_wrapped<T> || ...);
      using index_sequence_t = std::index_sequence_for<T...>;

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
      Symmetry(const Args&... args) noexcept : base_tuple_t(construct_base_tuple(args...)) {}

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
               static_cast<base_tuple_t&>(*this),
               static_cast<const base_tuple_t&>(other_symmetry));
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
               static_cast<base_tuple_t&>(*this),
               static_cast<const base_tuple_t&>(other_symmetry));
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
               static_cast<const base_tuple_t&>(*this),
               static_cast<const base_tuple_t&>(other_symmetry));
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
               static_cast<const base_tuple_t&>(*this),
               static_cast<const base_tuple_t&>(other_symmetry));
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
               static_cast<const base_tuple_t&>(*this));
      }

      // hash
      std::size_t hash() const {
         std::size_t seed = length;
         detail::map_on_tuple(
               [&seed](const auto& a, const auto&) {
                  using A = remove_cvref_t<decltype(a)>;
                  seed ^= std::hash<A>()(a) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
               },
               static_cast<const base_tuple_t&>(*this));
         return seed;
      }

      // parity
      /**
       * Get the total parity, whether fermion or boson
       */
      [[nodiscard]] bool get_parity() const {
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
               static_cast<const base_tuple_t>(*this));
         return result;
      }

    public:
      /**
       * Give the parity when reverse some edge
       *
       * \param symmetries symmetry list for each rank of some block
       * \param reverse_flag flags marking whether to reverse for each rank
       * \param valid_mark flags marking the validity for each rank
       *
       * when reversing edge in edge_operator, all reversed edge when parity is odd,
       * and this function will give the total parity summation of all the edge.
       * \see Tensor::edge_operator
       */
      template<typename SymList, typename BoolList1, typename BoolList2>
      [[nodiscard]] static bool get_reverse_parity(const SymList& symmetries, const BoolList1& reverse_flag, const BoolList2& valid_mark) {
         auto result = false;
         for (Rank i = 0; i < symmetries.size(); i++) {
            if (reverse_flag[i] && valid_mark[i]) {
               result ^= symmetries[i].get_parity();
            }
         }
         return result;
      }

    public:
      /**
       * Give the parity of transposation
       *
       * \param symmetries symmetry list for each rank of some block
       * \param transpose_plan plan of transposition
       *
       * There is no valid mark like reversing edge, since the parity of transposition is always valid, because
       * transposition is the operation of a single tensor.
       * \see Tensor::edge_operator
       */
      template<typename SymList, typename RankList>
      [[nodiscard]] static bool get_transpose_parity(const SymList& symmetries, const RankList& transpose_plan) {
         // auto result = false;
         // for (Rank i = 0; i < symmetries.size(); i++) {
         //    for (Rank j = i + 1; j < symmetries.size(); j++) {
         //       if (transpose_plan[i] > transpose_plan[j]) {
         //          result ^= symmetries[i].get_parity() && symmetries[j].get_parity();
         //       }
         //    }
         // }
         // return result;
         const Rank rank = symmetries.size();
         pmr::vector<Rank> fermi_plan;
         fermi_plan.reserve(rank);
         for (Rank i = 0; i < rank; i++) {
            if (symmetries[i].get_parity()) {
               fermi_plan.push_back(transpose_plan[i]);
            }
         }
         const int fermi_rank = fermi_plan.size();
         if (fermi_rank != 0) {
            return calculate_inversion(fermi_plan.data(), fermi_plan.data() + fermi_rank - 1);
         } else {
            return false;
         }
      }

    private:
      static bool calculate_inversion(Rank* begin, Rank* end) {
         bool result = false;
         if (begin < end) {
            Rank reference = begin[(end - begin) / 2];
            Rank* left = begin;
            Rank* right = end;
            while (true) {
               while (*left < reference) {
                  ++left;
               }
               while (*right > reference) {
                  --right;
               }
               if (left < right) {
                  result ^= true;
                  Rank temporary = *left;
                  *left = *right;
                  *right = temporary;
                  ++left;
                  --right;
               } else if (left == right) {
                  ++left;
                  --right;
                  break;
               } else {
                  break;
               }
            }
            result ^= calculate_inversion(begin, right);
            result ^= calculate_inversion(left, end);
         }
         return result;
      }

    public:
      /**
       * Give the parity when merging or spliting edges.
       *
       * \param symmetries symmetry list for each rank of some block
       * \param split_merge_flag The plan of merging or splitting
       * \param valid_mark validity of each merging or splitting operation.
       *
       * \note This is equivalant to total transposition over the merging list or splitting list,
       * while \f$\sum_{i\le j} s_i s_j = \frac{(\sum s_i)^2 - \sum s_i^2}{2}\f$, so it can be simplified.
       */
      template<typename SymList, typename RankList, typename BoolList>
      [[nodiscard]] static bool get_split_merge_parity(
            const SymList& symmetries,        // before merge length
            const RankList& split_merge_flag, // before merge length
            const BoolList& valid_mark) {     // after merge length
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
               // (sum(x)^2 - sum(x^2)) / 2 % 2 = (sum(x)^2 - sum(x)) & 2 != 0 = sum(x) & 2 != 0
               auto sum_of_parity = 0l;
               for (auto position_in_group = split_merge_begin_position; position_in_group < split_merge_end_position; position_in_group++) {
                  sum_of_parity += symmetries[position_in_group].get_parity();
               }
               result ^= ((sum_of_parity & 2) != 0);
            }
            split_merge_begin_position = split_merge_end_position;
         }
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
   using Z2Symmetry = Symmetry<Z2>;
   using U1Symmetry = Symmetry<U1>;
   using FermiSymmetry = Symmetry<fermi_wrap<U1>>;
   using FermiZ2Symmetry = Symmetry<fermi_wrap<U1>, Z2>;
   using FermiU1Symmetry = Symmetry<fermi_wrap<U1>, U1>;
   using ParitySymmetry = Symmetry<fermi_wrap<Z2>>;
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
