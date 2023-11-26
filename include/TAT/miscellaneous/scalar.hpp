/**
 * \file scalar.hpp
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
#ifndef TAT_SCALAR_HPP
#define TAT_SCALAR_HPP

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"

namespace TAT {
#define TAT_DEFINE_SCALAR_OPERATOR(OP, EVAL) \
    template< \
        typename ScalarType1, \
        typename ScalarType2, \
        typename = std::enable_if_t< \
            (is_complex<ScalarType1> || is_complex<ScalarType2>)&&(!std::is_same_v<real_scalar<ScalarType1>, real_scalar<ScalarType2>>)>> \
    inline auto OP(const ScalarType1& a, const ScalarType2& b) { \
        using t = std::common_type_t<ScalarType1, ScalarType2>; \
        return EVAL; \
    }
    TAT_DEFINE_SCALAR_OPERATOR(operator+, t(a) + t(b))
    TAT_DEFINE_SCALAR_OPERATOR(operator-, t(a) - t(b))
    TAT_DEFINE_SCALAR_OPERATOR(operator*, t(a) * t(b))
    TAT_DEFINE_SCALAR_OPERATOR(operator/, t(a) / t(b))
#undef TAT_DEFINE_SCALAR_OPERATOR

    inline timer scalar_outplace_guard("scalar_outplace");

#define TAT_DEFINE_SCALAR_OPERATOR(OP, EVAL) \
    template< \
        typename ScalarType1, \
        typename ScalarType2, \
        typename Symmetry, \
        typename Name, \
        typename = std::enable_if_t<is_scalar<ScalarType1> && is_scalar<ScalarType2> && is_symmetry<Symmetry> && is_name<Name>>> \
    [[nodiscard]] auto OP(const Tensor<ScalarType1, Symmetry, Name>& tensor_1, const Tensor<ScalarType2, Symmetry, Name>& tensor_2) { \
        auto timer_guard = scalar_outplace_guard(); \
        using ScalarType = std::common_type_t<ScalarType1, ScalarType2>; \
        return tensor_1.template zip_map<ScalarType>(tensor_2, [](const auto& x, const auto& y) { return EVAL; }); \
    } \
    template< \
        typename ScalarType1, \
        typename ScalarType2, \
        typename Symmetry, \
        typename Name, \
        typename = std::enable_if_t<is_scalar<ScalarType1> && is_scalar<ScalarType2> && is_symmetry<Symmetry> && is_name<Name>>> \
    [[nodiscard]] auto OP(const Tensor<ScalarType1, Symmetry, Name>& tensor_1, const ScalarType2& number_2) { \
        auto timer_guard = scalar_outplace_guard(); \
        using ScalarType = std::common_type_t<ScalarType1, ScalarType2>; \
        return tensor_1.template map<ScalarType>([&y = number_2](const auto& x) { return EVAL; }); \
    } \
    template< \
        typename ScalarType1, \
        typename ScalarType2, \
        typename Symmetry, \
        typename Name, \
        typename = std::enable_if_t<is_scalar<ScalarType1> && is_scalar<ScalarType2> && is_symmetry<Symmetry> && is_name<Name>>> \
    [[nodiscard]] auto OP(const ScalarType1& number_1, const Tensor<ScalarType2, Symmetry, Name>& tensor_2) { \
        auto timer_guard = scalar_outplace_guard(); \
        using ScalarType = std::common_type_t<ScalarType1, ScalarType2>; \
        return tensor_2.template map<ScalarType>([&x = number_1](const auto& y) { return EVAL; }); \
    }

    TAT_DEFINE_SCALAR_OPERATOR(operator+, x + y)
    TAT_DEFINE_SCALAR_OPERATOR(operator-, x - y)
    TAT_DEFINE_SCALAR_OPERATOR(operator*, x* y)
    TAT_DEFINE_SCALAR_OPERATOR(operator/, x / y)
#undef TAT_DEFINE_SCALAR_OPERATOR

    inline timer scalar_inplace_guard("scalar_inplace");

#define TAT_DEFINE_SCALAR_OPERATOR(OP, EVAL) \
    template< \
        typename ScalarType1, \
        typename ScalarType2, \
        typename Symmetry, \
        typename Name, \
        typename = std::enable_if_t<is_scalar<ScalarType1> && is_scalar<ScalarType2> && is_symmetry<Symmetry> && is_name<Name>>, \
        typename = std::enable_if_t<is_complex<ScalarType1> || is_real<ScalarType2>>> \
    Tensor<ScalarType1, Symmetry, Name>& OP(Tensor<ScalarType1, Symmetry, Name>& tensor_1, const Tensor<ScalarType2, Symmetry, Name>& tensor_2) { \
        auto timer_guard = scalar_inplace_guard(); \
        tensor_1.acquire_data_ownership("Inplace operator on tensor shared, copy happened here"); \
        return tensor_1.zip_transform_(tensor_2, [](const auto& x, const auto& y) { return EVAL; }); \
    } \
    template< \
        typename ScalarType1, \
        typename ScalarType2, \
        typename Symmetry, \
        typename Name, \
        typename = std::enable_if_t<is_scalar<ScalarType1> && is_scalar<ScalarType2> && is_symmetry<Symmetry> && is_name<Name>>, \
        typename = std::enable_if_t<is_complex<ScalarType1> || is_real<ScalarType2>>> \
    Tensor<ScalarType1, Symmetry, Name>& OP(Tensor<ScalarType1, Symmetry, Name>& tensor_1, const ScalarType2& number_2) { \
        auto timer_guard = scalar_inplace_guard(); \
        tensor_1.acquire_data_ownership("Inplace operator on tensor shared, copy happened here"); \
        return tensor_1.transform_([&y = number_2](const auto& x) { return EVAL; }); \
    }
    TAT_DEFINE_SCALAR_OPERATOR(operator+=, x + y)
    TAT_DEFINE_SCALAR_OPERATOR(operator-=, x - y)
    TAT_DEFINE_SCALAR_OPERATOR(operator*=, x* y)
    TAT_DEFINE_SCALAR_OPERATOR(operator/=, x / y)
#undef TAT_DEFINE_SCALAR_OPERATOR
} // namespace TAT
#endif
