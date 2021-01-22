/**
 * \file const_integral.hpp
 *
 * Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_CONST_INTEGRAL_HPP
#define TAT_CONST_INTEGRAL_HPP

#include <limits>
#include <variant>

namespace TAT {
   template<auto StaticValue, typename DynamicType = void>
   struct Integer {
      using value_type = DynamicType;
      value_type m_value;
      Integer() = delete;
      Integer(value_type v) : m_value(v) {}
      value_type value() const {
         return m_value;
      }
      static constexpr bool is_static = false;
      static constexpr bool is_dynamic = true;
   };

   template<auto StaticValue>
   struct Integer<StaticValue, void> {
      using value_type = decltype(StaticValue);
      Integer() {}
      Integer(value_type v) {}
      static value_type value() {
         return StaticValue;
      }
      static constexpr bool is_static = true;
      static constexpr bool is_dynamic = false;
   };

   template<typename T>
   Integer(T v) -> Integer<0, T>;

   template<typename T, T... possible_value>
   using to_const_result = std::variant<Integer<0, T>, Integer<possible_value>...>;

   template<typename R, typename T>
   R to_const_helper(T value) {
      return Integer(value);
   }
   template<typename R, typename T, T first_value, T... possible_value>
   R to_const_helper(T value) {
      if (first_value == value) {
         return Integer<first_value>();
      } else {
         return to_const_helper<R, T, possible_value...>(value);
      }
   }

   template<typename T, T... possible_value>
   auto to_const(T value) {
      using result_type = to_const_result<T, possible_value...>;
      return to_const_helper<result_type, T, possible_value...>(value);
   }
} // namespace TAT

#endif
