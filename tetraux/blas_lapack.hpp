/**
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

#ifndef BLAS_LAPACK_HPP
#define BLAS_LAPACK_HPP

#include <complex>
#include <hipblas.h>
#include <mpi.h>
#include <stdexcept>
#include <type_traits>

template<typename T>
struct type_identity {
    using type = T;
};

template<typename Scalar>
struct real_base_helper : type_identity<Scalar> { };
template<typename Scalar>
struct real_base_helper<std::complex<Scalar>> : type_identity<Scalar> { };

template<typename Scalar>
using real_base = typename real_base_helper<Scalar>::type;

void check_hiperror(hipError_t error) {
    if (error != hipSuccess) {
        printf("hipError_t: %s(%s)\n", hipGetErrorName(error), hipGetErrorString(error));
        throw std::runtime_error("hip error");
    }
}

void check_hipblas_status(hipblasStatus_t error) {
    if (error != HIPBLAS_STATUS_SUCCESS) {
        printf("hipblasStatus_t: %s\n", hipblasStatusToString(error));
        throw std::runtime_error("hip blas error");
    }
}

template<typename T>
auto erase_pointer(T v) {
    if constexpr (std::is_pointer_v<T>) {
        using Tv = std::remove_pointer_t<T>;
        using Tr = std::remove_const_t<Tv>;
        using Tb = std::conditional_t<
            std::is_same_v<Tr, std::complex<float>>,
            hipblasComplex,
            std::conditional_t<std::is_same_v<Tr, std::complex<double>>, hipblasDoubleComplex, Tr>>;
        // hipblas use a different complex type other than std::complex<T>
        // although they share the same layout, but c++ does not allow pass it as another
        // reinterpret the pointer here
        return reinterpret_cast<std::conditional_t<std::is_const_v<Tv>, const Tb, Tb>*>(v);
    } else {
        return v;
    }
}

template<typename Scalar>
constexpr auto gemv_ = nullptr;
template<>
inline constexpr auto gemv_<float> = hipblasSgemv;
template<>
inline constexpr auto gemv_<double> = hipblasDgemv;
template<>
inline constexpr auto gemv_<std::complex<float>> = hipblasCgemv;
template<>
inline constexpr auto gemv_<std::complex<double>> = hipblasZgemv;
template<typename Scalar, typename... Args>
auto gemv(Args... args) {
    auto status = gemv_<Scalar>(erase_pointer(args)...);
    check_hipblas_status(status);
}

template<typename Scalar>
constexpr auto nrm2_ = nullptr;
template<>
inline constexpr auto nrm2_<float> = hipblasSnrm2;
template<>
inline constexpr auto nrm2_<double> = hipblasDnrm2;
template<>
inline constexpr auto nrm2_<std::complex<float>> = hipblasScnrm2;
template<>
inline constexpr auto nrm2_<std::complex<double>> = hipblasDznrm2;
template<typename Scalar, typename... Args>
auto nrm2(Args... args) {
    auto status = nrm2_<Scalar>(erase_pointer(args)...);
    check_hipblas_status(status);
}

template<typename Scalar>
constexpr auto scal_ = nullptr;
template<>
inline constexpr auto scal_<float> = hipblasSscal;
template<>
inline constexpr auto scal_<double> = hipblasDscal;
template<>
inline constexpr auto scal_<std::complex<float>> = hipblasCscal;
template<>
inline constexpr auto scal_<std::complex<double>> = hipblasZscal;
template<typename Scalar, typename... Args>
auto scal(Args... args) {
    auto status = scal_<Scalar>(erase_pointer(args)...);
    check_hipblas_status(status);
}

template<typename Scalar>
constexpr auto copy_ = nullptr;
template<>
inline constexpr auto copy_<float> = hipblasScopy;
template<>
inline constexpr auto copy_<double> = hipblasDcopy;
template<>
inline constexpr auto copy_<std::complex<float>> = hipblasCcopy;
template<>
inline constexpr auto copy_<std::complex<double>> = hipblasZcopy;
template<typename Scalar, typename... Args>
auto copy(Args... args) {
    auto status = copy_<Scalar>(erase_pointer(args)...);
    check_hipblas_status(status);
}

template<typename Scalar>
constexpr auto axpy_ = nullptr;
template<>
inline constexpr auto axpy_<float> = hipblasSaxpy;
template<>
inline constexpr auto axpy_<double> = hipblasDaxpy;
template<>
inline constexpr auto axpy_<std::complex<float>> = hipblasCaxpy;
template<>
inline constexpr auto axpy_<std::complex<double>> = hipblasZaxpy;
template<typename Scalar, typename... Args>
auto axpy(Args... args) {
    auto status = axpy_<Scalar>(erase_pointer(args)...);
    check_hipblas_status(status);
}

// gpu

struct gpu_handle {
    hipblasHandle_t m_handle;
    gpu_handle(int device) {
        int gpu_count;
        check_hiperror(hipGetDeviceCount(&gpu_count));
        check_hiperror(hipSetDevice(device % gpu_count));
        check_hipblas_status(hipblasCreate(&m_handle));
    }
    ~gpu_handle() {
        check_hipblas_status(hipblasDestroy(m_handle));
    }
    auto& get() {
        return m_handle;
    }
};

template<typename Scalar>
struct gpu_array {
    Scalar* m_pointer;
    int m_size;
    gpu_array(int size) {
        check_hiperror(hipMalloc(&m_pointer, size * sizeof(Scalar)));
        m_size = size;
    }
    ~gpu_array() {
        check_hiperror(hipFree(m_pointer));
    }
    auto& get() {
        return m_pointer;
    }
    void to_host(Scalar* p) {
        check_hipblas_status(hipblasGetVector(m_size, sizeof(Scalar), m_pointer, 1, p, 1));
    }
    void from_host(const Scalar* p) {
        check_hipblas_status(hipblasSetVector(m_size, sizeof(Scalar), p, 1, m_pointer, 1));
    }
};

inline constexpr auto blas_op_n = HIPBLAS_OP_N;
inline constexpr auto blas_op_t = HIPBLAS_OP_T;
inline constexpr auto blas_op_c = HIPBLAS_OP_C;

#endif
