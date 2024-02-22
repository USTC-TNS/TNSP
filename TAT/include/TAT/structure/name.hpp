/**
 * \file name.hpp
 *
 * Copyright (C) 2019-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "../utility/common_variable.hpp"

namespace TAT {
    /**
     * FastName as tensor edge name
     *
     * Use hash to represent a string via dataset map
     */
    class FastName {
      private:
        using hash_t = std::size_t;

        hash_t hash;

        friend std::hash<FastName>;

      private:
        // Singleton
        struct dataset_t {
            std::unordered_map<hash_t, std::string> hash_to_name = {};
            const std::hash<std::string_view> hash_function = {};

            static dataset_t& instance() {
                static const auto pointer = std::make_unique<dataset_t>();
                return *pointer;
            }
        };
        static auto& dataset() {
            return dataset_t::instance();
        }

      public:
        FastName() : FastName("") { }

        template<
            typename String,
            typename = std::enable_if_t<std::is_convertible_v<String, std::string_view> && !std::is_same_v<remove_cvref_t<String>, FastName>>>
        FastName(String&& name) : hash(dataset().hash_function(name)) {
            if (auto found = dataset().hash_to_name.find(hash); found == dataset().hash_to_name.end()) {
                dataset().hash_to_name[hash] = name;
            }
        }

      public:
        operator const std::string&() const {
            // hash must be in dataset, which has been added into dataset when construct FastName.
            auto found = dataset().hash_to_name.find(hash);
            return found->second;
        }

        // This is not needed by TAT, but maybe needed by user for put it in map or set.
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
#undef TAT_DEFINE_FASTNAME_COMPARE
    };

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
        TAT_DEFINE_ALL_INTERNAL_NAME(Default_3)
        TAT_DEFINE_ALL_INTERNAL_NAME(Default_4)
#undef TAT_DEFINE_ALL_INTERNAL_NAME
#define TAT_DEFINE_ALL_INTERNAL_NAME(x) static const Name& x;
        TAT_DEFINE_ALL_INTERNAL_NAME(Contract_0) // Used in contract temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(Contract_1) // Used in contract temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(Contract_2) // used in contract temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(SVD_U) // used in svd temporary tensor and singular matrix tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(SVD_V) // used in svd temporary tensor and singular matrix tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(QR_1) // used in qr temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(QR_2) // used in qr temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(Trace_1) // used in trace temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(Trace_2) // used in trace temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(Trace_3) // used in trace temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(Trace_4) // used in trace temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(Trace_5) // used in trace temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(No_Old_Name) // used in expand configuration
        TAT_DEFINE_ALL_INTERNAL_NAME(No_New_Name) // used in shrink configuration
        TAT_DEFINE_ALL_INTERNAL_NAME(Exp_1) // used in exponential temporary tensor
        TAT_DEFINE_ALL_INTERNAL_NAME(Exp_2) // used in exponential temporary tensor
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
    TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_4, 3)
    TAT_DEFINE_DEFAULT_INTERNAL_NAME(Trace_5, 4)
    TAT_DEFINE_DEFAULT_INTERNAL_NAME(No_Old_Name, 0)
    TAT_DEFINE_DEFAULT_INTERNAL_NAME(No_New_Name, 0)
    TAT_DEFINE_DEFAULT_INTERNAL_NAME(Exp_1, 1)
    TAT_DEFINE_DEFAULT_INTERNAL_NAME(Exp_2, 2)
#undef TAT_DEFINE_DEFAULT_INTERNAL_NAME
#define TAT_DEFINE_INTERNAL_NAME(x) \
    inline const FastName target_preset_name_of_fastname_##x = "__" #x; \
    inline const std::string target_preset_name_of_string_##x = "__" #x; \
    template<> \
    inline const FastName& InternalName<FastName>::x = target_preset_name_of_fastname_##x; \
    template<> \
    inline const std::string& InternalName<std::string>::x = target_preset_name_of_string_##x;
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
    TAT_DEFINE_INTERNAL_NAME(Trace_4)
    TAT_DEFINE_INTERNAL_NAME(Trace_5)
    TAT_DEFINE_INTERNAL_NAME(No_Old_Name)
    TAT_DEFINE_INTERNAL_NAME(No_New_Name)
    TAT_DEFINE_INTERNAL_NAME(Exp_1)
    TAT_DEFINE_INTERNAL_NAME(Exp_2)
#undef TAT_DEFINE_INTERNAL_NAME

    template<typename Name>
    using out_operator_t = std::ostream& (*)(std::ostream&, const Name&);
    template<typename Name>
    using in_operator_t = std::istream& (*)(std::istream&, Name&);

    /**
     * Name type also need input and output method
     *
     * Specialize NameTraits to define write, read, print, scan
     * Their type are out_operator_t<Name> and in_operator_t<Name>
     */
    template<typename Name>
    struct NameTraits { };

    namespace detail {
        template<typename T>
        using name_write_checker = decltype(NameTraits<T>::write(std::declval<std::ostream&>(), std::declval<const T&>()));
        template<typename T>
        using name_read_checker = decltype(NameTraits<T>::read(std::declval<std::istream&>(), std::declval<T&>()));
        template<typename T>
        using name_print_checker = decltype(NameTraits<T>::print(std::declval<std::ostream&>(), std::declval<const T&>()));
        template<typename T>
        using name_scan_checker = decltype(NameTraits<T>::scan(std::declval<std::istream&>(), std::declval<T&>()));
    } // namespace detail

    /**
     * Check whether a type is a Name type
     *
     * If any one of those four function is defined, it is a Name type.
     * Because it is considered that user want to use it as Name for only some operator.
     */
    template<typename Name>
    constexpr bool is_name = is_detected_v<detail::name_write_checker, Name> || is_detected_v<detail::name_read_checker, Name> ||
                             is_detected_v<detail::name_print_checker, Name> || is_detected_v<detail::name_scan_checker, Name>;
} // namespace TAT

namespace std {
    template<>
    struct hash<TAT::FastName> {
        size_t operator()(const TAT::FastName& name) const {
            return name.hash;
        }
    };

    template<typename Name>
    struct hash<pair<Name, Name>> {
        size_t operator()(const pair<Name, Name>& names) const {
            const auto& [name_1, name_2] = names;
            auto hash_1 = hash<Name>()(name_1);
            auto hash_2 = hash<Name>()(name_2);
            auto seed = hash_1;
            auto v = hash_2;
            TAT::hash_absorb(seed, v);
            return seed;
        }
    };
} // namespace std
#endif
