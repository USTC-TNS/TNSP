/**
 * \file cuda.hpp
 *
 * Copyright (C) 2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_CUDA_HPP
#define TAT_CUDA_HPP

#ifdef TAT_USE_CUDA

#define TAT_CUDA_DEVICE __device__
#define TAT_CUDA_HOST __host__

#include <complex>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/complex.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#include "common_variable.hpp"

namespace TAT {
    namespace cuda {
        inline void check_cuda(cudaError_t code) {
            if (code != cudaSuccess) {
                throw thrust::system_error(code, thrust::cuda_category());
            }
        }

        template<typename T>
        struct thrust_complex_helper : type_identity<T> { };
        template<typename T>
        struct thrust_complex_helper<std::complex<T>> : type_identity<thrust::complex<T>> { };
        template<typename T>
        using thrust_complex = typename thrust_complex_helper<T>::type;

        template<typename T>
        auto thrust_complex_wrap(T* value) {
            if constexpr (std::is_const_v<T>) {
                return reinterpret_cast<const thrust_complex<std::remove_const_t<T>>*>(value);
            } else {
                return reinterpret_cast<thrust_complex<T>*>(value);
            }
        }

        /**
         * Allocator on gpu.
         */
        template<typename T>
        struct allocator : std::allocator<T> {
            using std::allocator<T>::allocator;

            allocator(const allocator& other) = default;
            template<typename U>
            allocator(const allocator<U>& other) : allocator() { }

            allocator<T> select_on_container_copy_construction() const {
                return allocator<T>();
            }

            // It is useless, but base class has it so derived class must have it.
            template<typename U>
            struct rebind {
                using other = allocator<U>;
            };

            template<typename U, typename... Args>
            void construct([[maybe_unused]] U* p, Args&&... args) {
                if constexpr (!((sizeof...(args) == 0) && (std::is_trivially_destructible_v<T>))) {
                    detail::error("cuda allocator cannot allocate non trivial type.");
                }
            }

            T* allocate(std::size_t n) {
                return thrust::raw_pointer_cast(thrust::device_malloc<T>(n));
            }

            void deallocate(T* p, std::size_t) {
                thrust::device_free(thrust::device_pointer_cast(p));
            }
        };

        template<typename T>
        using vector = std::vector<T, allocator<T>>;

        template<typename T>
        TAT_CUDA_HOST TAT_CUDA_DEVICE auto abs(const thrust_complex<T>& x) {
            if constexpr (is_complex<T>) {
                return thrust::abs(x);
            } else {
                return x < 0 ? -x : +x;
            }
        }

        template<typename T>
        struct abs_op {
            TAT_CUDA_HOST TAT_CUDA_DEVICE auto operator()(const thrust_complex<T>& x) const {
                return abs<T>(x);
            }
        };

        template<typename T>
        TAT_CUDA_HOST TAT_CUDA_DEVICE auto pow(const T& x, int p) {
            // T must be a real number here, float or double
            if constexpr (std::is_same_v<T, double>) {
                return ::pow(x, p);
            } else {
                return ::powf(x, p);
            }
        }

        template<typename T, int p>
        struct pow_op {
            TAT_CUDA_HOST TAT_CUDA_DEVICE auto operator()(const thrust_complex<T>& number) const {
                if constexpr (p == 1) {
                    return abs(number);
                } else if constexpr (p == 2) {
                    if constexpr (is_complex<T>) {
                        return thrust::norm(number);
                    } else {
                        return number * number;
                    }
                } else {
                    if constexpr (p % 2 == 0 && is_real<T>) {
                        return pow(number, p);
                    } else {
                        return pow(abs(number), p);
                    }
                }
            }
        };

        template<typename T>
        struct identity_op {
            TAT_CUDA_HOST TAT_CUDA_DEVICE auto operator()(const thrust_complex<T>& value) const {
                return value;
            }
        };

        template<typename T>
        struct zero_op {
            TAT_CUDA_HOST TAT_CUDA_DEVICE auto operator()() const {
                return thrust_complex<T>(0);
            }
        };

        template<typename From, typename To>
        struct convertion_op {
            TAT_CUDA_HOST TAT_CUDA_DEVICE thrust_complex<To> operator()(thrust_complex<From> value) const {
                if constexpr (is_complex<From> && is_real<To>) {
                    return value.real();
                } else {
                    return value;
                }
            }
        };

        template<typename T>
        struct range_op {
            thrust_complex<T> first;
            thrust_complex<T> step;

            range_op(thrust_complex<T> first, thrust_complex<T> step) : first(first), step(step) { }

            TAT_CUDA_HOST TAT_CUDA_DEVICE auto operator()(const Size& index) const {
                return first + step * index;
            }
        };

        template<typename T>
        constexpr auto gemm = nullptr;
        template<>
        inline constexpr auto gemm<float> = cublasSgemm;
        template<>
        inline constexpr auto gemm<double> = cublasDgemm;
        template<>
        inline constexpr auto gemm<std::complex<float>> = cublasCgemm;
        template<>
        inline constexpr auto gemm<std::complex<double>> = cublasZgemm;
    } // namespace cuda
} // namespace TAT

#else
#define TAT_CUDA_DEVICE
#define TAT_CUDA_HOST
#endif

#endif
