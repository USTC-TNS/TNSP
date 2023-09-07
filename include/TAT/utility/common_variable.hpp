/**
 * \file common_variable.hpp
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
#ifndef TAT_COMMON_VARIABLE_HPP
#define TAT_COMMON_VARIABLE_HPP

// Macros options
// - TAT_USE_MPI: define to enable mpi support, cmake can configure it
// - TAT_USE_MKL_GEMM_BATCH: define to use mkl ?gemm_batch, cmake can configure it

// - TAT_USE_FAST_NAME: define to use TAT::FastName as default name instead of std::string
// - TAT_USE_TIMER: define to add timers for some common operator

// - TAT_ERROR_BITS: throw exception for different situations, rather than print warning
// - TAT_NOTHING_BITS: keep silent for different situations, rather than print warning

/**
 * TAT is A Tensor library
 */
namespace TAT {
#ifndef TAT_VERSION
#define TAT_VERSION "0.3.9"
#endif

    /**
     * TAT version
     */
    inline const char* version = TAT_VERSION;

    /**
     * Debug flag
     */
    inline constexpr bool debug_mode =
#ifdef NDEBUG
        false
#else
        true
#endif
        ;

    /**
     * TAT informations about compiler and license
     */
    inline const char* information = "TAT " TAT_VERSION " ("
#ifdef TAT_BUILD_TYPE
                                     "" TAT_BUILD_TYPE ", "
#endif
                                     "" __DATE__ ", " __TIME__
#ifdef TAT_COMPILER_INFORMATION
                                     ", " TAT_COMPILER_INFORMATION
#endif
                                     ")\n"
                                     "Copyright (C) 2019-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>\n"
                                     "This is free software; see the source for copying conditions.  There is NO\n"
                                     "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.";

    // evil
    namespace detail {
        /**
         * Singleton, control color ansi in windows
         */
        struct evil_t {
            evil_t();
            ~evil_t();
        };
        inline const evil_t evil;
    } // namespace detail

    // log and warning
    namespace detail {
        /**
         * Print log for TAT
         */
        inline void log(const char* message);

        /**
         * Print warning
         *
         * Share the same interface with `nothing` and `error`.
         * Used for different situation, control by macro TAT_NOTHING_BITS and TAT_ERROR_BITS
         *
         * \param message warning message
         *
         * \see nothing, error
         */
        inline void warning(const char* message);

        /**
         * Do nothing
         *
         * \see warning, error
         */
        inline void nothing(const char*) { }

        /**
         * Throw runtime exception
         *
         * \param message exception message
         *
         * \see warning, nothing
         */
        [[noreturn]] inline void error(const char* message);

#ifndef TAT_ERROR_BITS
#define TAT_ERROR_BITS 0
#endif
#ifndef TAT_NOTHING_BITS
#define TAT_NOTHING_BITS 0
#endif

        constexpr auto what_if_lapack_error = TAT_ERROR_BITS & 1 ? error : TAT_NOTHING_BITS & 1 ? nothing : warning;
        constexpr auto what_if_copy_shared = TAT_ERROR_BITS & 2 ? error : TAT_NOTHING_BITS & 2 ? nothing : warning;
    } // namespace detail
} // namespace TAT

#include <cstdint>

namespace TAT {
    // type alias

    // The most common used integral type `short`, `int`, `long` have different size in different platform
    // in linux, they are 16, 32, 64(lp64)
    // in windows, they are 16, 32, 32(llp64)
    // So use uintxx_t explicitly to avoid it incompatible when import data exported in another platform
    // In TAT, there is also `int` type common used, especially when calling blas or lapack function
    // It is all 32 bit in most platform currently.
    // But it is 64bit in ilp64 and 16bit in lp32
    // So please do not link blas lapack using ilp64 or lp32
    /**
     * Tensor rank type
     */
    using Rank = std::uint16_t;
    /**
     * Tensor block number, or dimension segment number type
     */
    using Nums = std::uint32_t;
    /**
     * Tensor content data size, or dimension size type
     */
    using Size = std::uint64_t;

    /**
     * Fermi arrow type
     *
     * \note For connected two edge, EPR pair is \f$a^\dagger b^\dagger\f$
     * then, tensor owning edge a have arrow=false, and tensor owning edge b has arrow=true.
     * namely, the EPR pair arrow order is (false true)
     */
    using Arrow = bool;
} // namespace TAT

#include <complex>
#include <type_traits>

namespace TAT {
    // traits about scalar
    template<typename T>
    constexpr bool is_real = std::is_scalar_v<T>;

    namespace detail {
        template<typename T>
        struct is_complex_helper : std::bool_constant<false> { };
        template<typename T>
        struct is_complex_helper<std::complex<T>> : std::bool_constant<true> { };
    } // namespace detail
    template<typename T>
    constexpr bool is_complex = detail::is_complex_helper<T>::value;

    template<typename T>
    constexpr bool is_scalar = is_real<T> || is_complex<T>;

    namespace detail {
        template<typename T>
        struct real_scalar_helper : std::conditional<is_real<T>, T, void> { };
        template<typename T>
        struct real_scalar_helper<std::complex<T>> : std::conditional<is_real<T>, T, void> { };
    } // namespace detail
    /**
     * Get corresponding real type, used in svd and norm
     *
     * \tparam T if T is complex type, return corresponding basic real type, if it is real type, return itself, otherwise, return void
     */
    template<typename T>
    using real_scalar = typename detail::real_scalar_helper<T>::type;

    // type traits from c++latest
    namespace detail {
        template<typename AlwaysVoid, template<typename...> class Op, typename... Args>
        struct detector : std::false_type { };

        template<template<typename...> class Op, typename... Args>
        struct detector<std::void_t<Op<Args...>>, Op, Args...> : std::true_type { };

    } // namespace detail
    template<template<typename...> class Op, typename... Args>
    using is_detected = typename detail::detector<void, Op, Args...>;
    template<template<typename...> class Op, typename... Args>
    constexpr bool is_detected_v = is_detected<Op, Args...>::value;

    template<typename T>
    struct remove_cvref {
        using type = std::remove_cv_t<std::remove_reference_t<T>>;
    };
    template<typename T>
    using remove_cvref_t = typename remove_cvref<T>::type;

    template<typename T>
    struct type_identity {
        using type = T;
    };
    template<typename T>
    using type_identity_t = typename type_identity<T>::type;
} // namespace TAT

#include <array>

namespace TAT {
    template<typename T>
    struct empty_list : std::array<T, 0> {
        template<typename U>
        auto find(const U&) const {
            return this->end();
        }
    };

    template<typename... Fs>
    struct overloaded : Fs... {
        using Fs::operator()...;
    };
    template<typename... Fs>
    overloaded(Fs...) -> overloaded<Fs...>;

    constexpr std::size_t unordered_parameter = 4;

    inline std::size_t& hash_absorb(std::size_t& seed, std::size_t value) {
        // copy from boost
        return seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
} // namespace TAT

#endif
