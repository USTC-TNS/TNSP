/**
 * \file scalar.hpp
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
#ifndef TAT_SCALAR_HPP
#define TAT_SCALAR_HPP

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"

namespace TAT {
    inline timer scalar_outplace_guard("scalar_outplace");
    inline timer scalar_inplace_guard("scalar_inplace");

    namespace detail {
        enum arithmetic_type {
            addition,
            subtraction,
            multiplication,
            division,
        };

#ifdef TAT_USE_CUDA
        template<typename ScalarType1, typename ScalarType2, arithmetic_type type>
        TAT_CUDA_HOST TAT_CUDA_DEVICE auto arithmetic(const cuda::thrust_complex<ScalarType1>& x, const cuda::thrust_complex<ScalarType2>& y) {
            using ScalarType = cuda::thrust_complex<std::common_type_t<ScalarType1, ScalarType2>>;
            if constexpr (type == addition) {
                return ScalarType(x) + ScalarType(y);
            } else if constexpr (type == subtraction) {
                return ScalarType(x) - ScalarType(y);
            } else if constexpr (type == multiplication) {
                return ScalarType(x) * ScalarType(y);
            } else {
                return ScalarType(x) / ScalarType(y);
            }
        }

        template<typename ScalarType1, typename ScalarType2, arithmetic_type type>
        struct ArithmeticTN {
            using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;
            cuda::thrust_complex<ScalarType2> y;
            ArithmeticTN(const ScalarType2& number) : y(number) { }
            TAT_CUDA_HOST TAT_CUDA_DEVICE cuda::thrust_complex<ScalarType> operator()(const cuda::thrust_complex<ScalarType1>& x) {
                return arithmetic<ScalarType1, ScalarType2, type>(x, y);
            }
        };
        template<typename ScalarType1, typename ScalarType2, arithmetic_type type>
        struct ArithmeticNT {
            using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;
            cuda::thrust_complex<ScalarType1> x;
            ArithmeticNT(const ScalarType1& number) : x(number) { }
            TAT_CUDA_HOST TAT_CUDA_DEVICE cuda::thrust_complex<ScalarType> operator()(const cuda::thrust_complex<ScalarType2>& y) {
                return arithmetic<ScalarType1, ScalarType2, type>(x, y);
            }
        };
        template<typename ScalarType1, typename ScalarType2, arithmetic_type type>
        struct ArithmeticTT {
            using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;
            TAT_CUDA_HOST TAT_CUDA_DEVICE cuda::thrust_complex<ScalarType>
            operator()(const cuda::thrust_complex<ScalarType1>& x, const cuda::thrust_complex<ScalarType2>& y) {
                return arithmetic<ScalarType1, ScalarType2, type>(x, y);
            }
        };
#else
        template<typename ScalarType1, typename ScalarType2, arithmetic_type type>
        auto arithmetic(const ScalarType1& x, const ScalarType2& y) {
            using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;
            if constexpr (type == addition) {
                return ScalarType(x) + ScalarType(y);
            } else if constexpr (type == subtraction) {
                return ScalarType(x) - ScalarType(y);
            } else if constexpr (type == multiplication) {
                return ScalarType(x) * ScalarType(y);
            } else {
                return ScalarType(x) / ScalarType(y);
            }
        }

        template<typename ScalarType1, typename ScalarType2, arithmetic_type type>
        struct ArithmeticTN {
            using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;
            ScalarType2 y;
            ArithmeticTN(const ScalarType2& number) : y(number) { }
            ScalarType operator()(const ScalarType1& x) {
                return arithmetic<ScalarType1, ScalarType2, type>(x, y);
            }
        };
        template<typename ScalarType1, typename ScalarType2, arithmetic_type type>
        struct ArithmeticNT {
            using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;
            ScalarType1 x;
            ArithmeticNT(const ScalarType1& number) : x(number) { }
            ScalarType operator()(const ScalarType2& y) {
                return arithmetic<ScalarType1, ScalarType2, type>(x, y);
            }
        };
        template<typename ScalarType1, typename ScalarType2, arithmetic_type type>
        struct ArithmeticTT {
            using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;
            ScalarType operator()(const ScalarType1& x, const ScalarType2& y) {
                return arithmetic<ScalarType1, ScalarType2, type>(x, y);
            }
        };
#endif
    } // namespace detail

#define TAT_DEFINE_SCALAR_OPERATOR(OP, type) \
    template< \
        typename ScalarType1, \
        typename ScalarType2, \
        typename Symmetry, \
        typename Name, \
        typename = std::enable_if_t<is_scalar<ScalarType1> && is_scalar<ScalarType2> && is_symmetry<Symmetry> && is_name<Name>>> \
    [[nodiscard]] auto OP(const Tensor<ScalarType1, Symmetry, Name>& tensor_1, const Tensor<ScalarType2, Symmetry, Name>& tensor_2) { \
        auto timer_guard = scalar_outplace_guard(); \
        using ScalarType = std::common_type_t<ScalarType1, ScalarType2>; \
        return tensor_1.template zip_map<ScalarType>(tensor_2, detail::ArithmeticTT<ScalarType1, ScalarType2, type>()); \
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
        return tensor_1.template map<ScalarType>(detail::ArithmeticTN<ScalarType1, ScalarType2, type>(number_2)); \
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
        return tensor_2.template map<ScalarType>(detail::ArithmeticNT<ScalarType1, ScalarType2, type>(number_1)); \
    }

    TAT_DEFINE_SCALAR_OPERATOR(operator+, detail::addition)
    TAT_DEFINE_SCALAR_OPERATOR(operator-, detail::subtraction)
    TAT_DEFINE_SCALAR_OPERATOR(operator*, detail::multiplication)
    TAT_DEFINE_SCALAR_OPERATOR(operator/, detail::division)
#undef TAT_DEFINE_SCALAR_OPERATOR

#define TAT_DEFINE_SCALAR_OPERATOR(OP, type) \
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
        return tensor_1.zip_transform_(tensor_2, detail::ArithmeticTT<ScalarType1, ScalarType2, type>()); \
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
        return tensor_1.transform_(detail::ArithmeticTN<ScalarType1, ScalarType2, type>(number_2)); \
    }
    TAT_DEFINE_SCALAR_OPERATOR(operator+=, detail::addition)
    TAT_DEFINE_SCALAR_OPERATOR(operator-=, detail::subtraction)
    TAT_DEFINE_SCALAR_OPERATOR(operator*=, detail::multiplication)
    TAT_DEFINE_SCALAR_OPERATOR(operator/=, detail::division)
#undef TAT_DEFINE_SCALAR_OPERATOR
} // namespace TAT
#endif
