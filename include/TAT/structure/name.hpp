/**
 * \file name.hpp
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
#ifndef TAT_NAME_HPP
#define TAT_NAME_HPP

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../utility/common_variable.hpp"

namespace TAT {
   /**
    * FastName as tensor edge name
    *
    * Use hash to represent a string via dataset map
    */
   struct FastName {
      using hash_t = std::size_t;

      // Singleton
      struct dataset_t {
         std::unordered_map<hash_t, std::string> hash_to_name;
         std::hash<std::string_view> hash_function;
      };
      inline static auto dataset = dataset_t();
      inline static auto unknown_prefix = "UnknownName";

      hash_t hash;

      FastName() : hash(0) {} // A Temporary name

      // Specify name by its hash directly
      explicit FastName(const hash_t hash) : hash(hash) {}

      template<
            typename String,
            typename = std::enable_if_t<std::is_convertible_v<String, std::string_view> && !std::is_same_v<remove_cvref_t<String>, FastName>>>
      FastName(String&& name) : hash(dataset.hash_function(name)) {
         auto found = dataset.hash_to_name.find(hash);
         if (found == dataset.hash_to_name.end()) {
            dataset.hash_to_name[hash] = name;
         }
      }

      operator const std::string() const {
         auto found = dataset.hash_to_name.find(hash);
         if (found == dataset.hash_to_name.end()) {
            return unknown_prefix + std::to_string(hash);
         } else {
            return found->second;
         }
      }

#define TAT_DEFINE_FASTNAME_COMPARE(OP, EVAL) \
   inline bool OP(const FastName& other) const { \
      const auto& a = hash; \
      const auto& b = other.hash; \
      return EVAL; \
   }
      TAT_DEFINE_FASTNAME_COMPARE(operator==, a == b)
      TAT_DEFINE_FASTNAME_COMPARE(operator!=, a != b)
      TAT_DEFINE_FASTNAME_COMPARE(operator<, a < b)
      TAT_DEFINE_FASTNAME_COMPARE(operator>, a > b)
      TAT_DEFINE_FASTNAME_COMPARE(operator<=, a <= b)
      TAT_DEFINE_FASTNAME_COMPARE(operator>=, a >= b)
   };
#undef TAT_DEFINE_FASTNAME_COMPARE

   using DefaultName =
#ifdef TAT_USE_FAST_NAME
         FastName
#else
         std::string
#endif
         ;

   /**
    * For every Name type, some internal name is needed.
    *
    * It would be better to define all name to get better debug experience.
    * You can also define only three `Default_x`
    */
   template<typename Name>
   struct InternalName {
#define TAT_DEFINE_ALL_INTERNAL_NAME(x) static const Name x;
      TAT_DEFINE_ALL_INTERNAL_NAME(Default_0)
      TAT_DEFINE_ALL_INTERNAL_NAME(Default_1)
      TAT_DEFINE_ALL_INTERNAL_NAME(Default_2)
#undef TAT_DEFINE_ALL_INTERNAL_NAME
#define TAT_DEFINE_ALL_INTERNAL_NAME(x) static const Name& x;
      TAT_DEFINE_ALL_INTERNAL_NAME(Contract_0)  // Used in contract temporary tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(Contract_1)  // Used in contract temporary tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(Contract_2)  // used in contract temporary tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(SVD_U)       // used in svd temporary tensor and singular matrix tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(SVD_V)       // used in svd temporary tensor and singular matrix tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(QR_1)        // used in qr temporary tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(QR_2)        // used in qr temporary tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(Trace_1)     // used in trace temporary tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(Trace_2)     // used in trace temporary tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(Trace_3)     // used in trace temporary tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(No_Old_Name) // used in expand configuration
      TAT_DEFINE_ALL_INTERNAL_NAME(No_New_Name) // used in shrink configuration
      TAT_DEFINE_ALL_INTERNAL_NAME(Exp_1)       // used in exponential temporary tensor
      TAT_DEFINE_ALL_INTERNAL_NAME(Exp_2)       // used in exponential temporary tensor
#undef TAT_DEFINE_ALL_INTERNAL_NAME
   };
#define TAT_DEFINE_DEFAULT_INTERNAL_NAME(x, n) \
   template<typename Name> \
   const Name& InternalName<Name>::x = InternalName<Name>::Default_##n;
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Contract_0, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Contract_1, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Contract_2, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(SVD_U, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(SVD_V, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(QR_1, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(QR_2, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_1, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_2, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_3, 2)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(No_Old_Name, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(No_New_Name, 0)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Exp_1, 1)
   TAT_DEFINE_DEFAULT_INTERNAL_NAME(Exp_2, 2)
#undef TAT_DEFINE_DEFAULT_INTERNAL_NAME
#define TAT_DEFINE_INTERNAL_NAME(x) \
   template<> \
   inline const FastName& InternalName<FastName>::x = "__" #x; \
   template<> \
   inline const std::string& InternalName<std::string>::x = "__" #x;
   TAT_DEFINE_INTERNAL_NAME(Contract_0)
   TAT_DEFINE_INTERNAL_NAME(Contract_1)
   TAT_DEFINE_INTERNAL_NAME(Contract_2)
   TAT_DEFINE_INTERNAL_NAME(SVD_U)
   TAT_DEFINE_INTERNAL_NAME(SVD_V)
   TAT_DEFINE_INTERNAL_NAME(QR_1)
   TAT_DEFINE_INTERNAL_NAME(QR_2)
   TAT_DEFINE_INTERNAL_NAME(Trace_1)
   TAT_DEFINE_INTERNAL_NAME(Trace_2)
   TAT_DEFINE_INTERNAL_NAME(Trace_3)
   TAT_DEFINE_INTERNAL_NAME(No_Old_Name)
   TAT_DEFINE_INTERNAL_NAME(No_New_Name)
   TAT_DEFINE_INTERNAL_NAME(Exp_1)
   TAT_DEFINE_INTERNAL_NAME(Exp_2)
#undef TAT_DEFINE_INTERNAL_NAME

   template<typename Name>
   using name_out_operator_t = std::ostream& (*)(std::ostream&, const Name&);
   template<typename Name>
   using name_in_operator_t = std::istream& (*)(std::istream&, Name&);

   /**
    * Name type also need input and output method
    *
    * Specialize NameTraits to define write, read, print, scan
    * Their type are name_out_operator_t<Name> and name_in_operator_t<Name>
    */
   template<typename Name>
   struct NameTraits {};

   namespace detail {
      // It is hard implement in c++17
      template<typename T>
      using name_trait_write_checker = decltype(&NameTraits<T>::write);
      template<typename T>
      using name_trait_read_checker = decltype(&NameTraits<T>::read);
      template<typename T>
      using name_trait_print_checker = decltype(&NameTraits<T>::print);
      template<typename T>
      using name_trait_scan_checker = decltype(&NameTraits<T>::scan);
   } // namespace detail

   /**
    * Check whether a type is a Name type
    *
    * If any one of those four function is defined, it is a Name type.
    * Because it is considered that user want to use it as Name for some operator
    */
   template<typename Name>
   constexpr bool is_name = is_detected_v<detail::name_trait_write_checker, Name> || is_detected_v<detail::name_trait_read_checker, Name> ||
                            is_detected_v<detail::name_trait_print_checker, Name> || is_detected_v<detail::name_trait_scan_checker, Name>;
} // namespace TAT

namespace std {
   template<>
   struct hash<TAT::FastName> {
      size_t operator()(const TAT::FastName& name) const {
         return name.hash;
      }
   };

   template<>
   struct hash<pair<TAT::FastName, TAT::FastName>> {
      size_t operator()(const pair<TAT::FastName, TAT::FastName>& names) const {
         const auto& [name_1, name_2] = names;
         auto hash_1 = hash<TAT::FastName>()(name_1);
         auto hash_2 = hash<TAT::FastName>()(name_2);
         auto seed = hash_1;
         auto v = hash_2;
         seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
         return seed;
      }
   };

   template<>
   struct hash<pair<string, string>> {
      size_t operator()(const pair<string, string>& names) const {
         const auto& [name_1, name_2] = names;
         auto hash_1 = hash<string>()(name_1);
         auto hash_2 = hash<string>()(name_2);
         auto seed = hash_1;
         auto v = hash_2;
         seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
         return seed;
      }
   };
} // namespace std
#endif
