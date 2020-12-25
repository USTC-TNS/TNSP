/**
 * \file exponential.hpp
 *
 * Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_EXPONENTIAL_HPP
#define TAT_EXPONENTIAL_HPP

#include <algorithm>
#include <cmath>

#include "contract.hpp"
#include "tensor.hpp"
#include "timer.hpp"
#include "transpose.hpp"

#ifndef TAT_DOXYGEN_SHOULD_SKIP_THIS
extern "C" {
void sgesv_(const int* n, const int* nrhs, float* A, const int* lda, int* ipiv, float* B, const int* ldb, int* info);
void dgesv_(const int* n, const int* nrhs, double* A, const int* lda, int* ipiv, double* B, const int* ldb, int* info);
void cgesv_(const int* n, const int* nrhs, std::complex<float>* A, const int* lda, int* ipiv, std::complex<float>* B, const int* ldb, int* info);
void zgesv_(const int* n, const int* nrhs, std::complex<double>* A, const int* lda, int* ipiv, std::complex<double>* B, const int* ldb, int* info);
}
#endif

namespace TAT {
#ifndef TAT_DOXYGEN_SHOULD_SKIP_THIS
   template<typename ScalarType>
   constexpr void (*gesv)(const int* n, const int* nrhs, ScalarType* A, const int* lda, int* ipiv, ScalarType* B, const int* ldb, int* info) =
         nullptr;
   template<>
   inline auto gesv<float> = sgesv_;
   template<>
   inline auto gesv<double> = dgesv_;
   template<>
   inline auto gesv<std::complex<float>> = cgesv_;
   template<>
   inline auto gesv<std::complex<double>> = zgesv_;

   template<typename ScalarType>
   void linear_solve(int n, ScalarType* A, int nrhs, ScalarType* B) {
      // A: n*n
      // B: n*nrhs
      vector<ScalarType> AT(n * n);
      matrix_transpose(n, n, A, AT.data());
      vector<ScalarType> BT(n * nrhs);
      matrix_transpose(n, nrhs, B, BT.data());
      vector<int> ipiv(n);
      int result;
      gesv<ScalarType>(&n, &nrhs, AT.data(), &n, ipiv.data(), BT.data(), &n, &result);
      if (result != 0) {
         TAT_warning_or_error_when_lapack_error("error in GESV");
      }
      matrix_transpose(nrhs, n, BT.data(), B);
   }

   template<typename ScalarType>
   auto max_of_abs(const ScalarType* data, Size n) {
      real_base_t<ScalarType> result = 0;
      for (Size i = 0; i < n * n; i++) {
         auto here = std::abs(data[i]);
         result = result < here ? here : result;
      }
      return result;
   }

   template<typename ScalarType>
   void initialize_identity_matrix(ScalarType* data, Size n) {
      for (Size i = 0; i < n - 1; i++) {
         *(data++) = 1;
         for (Size j = 0; j < n; j++) {
            *(data++) = 0;
         }
      }
      *data = 1;
   }

   template<typename ScalarType>
   void matrix_exponential(Size n, ScalarType* A, ScalarType* F, int q) {
      // j = max(0, 1+floor(log2(|A|_inf)))
      auto j = std::max(0, 1 + int(std::log2(max_of_abs(A, n))));
      // A = A/2^j
      ScalarType parameter = ScalarType(1) / (1 << j);
      for (Size i = 0; i < n * n; i++) {
         A[i] *= parameter;
      }
      // D=I, N=I, X=I, c=1
      vector<ScalarType> D(n * n);
      initialize_identity_matrix(D.data(), n);
      vector<ScalarType> N(n * n);
      initialize_identity_matrix(N.data(), n);
      vector<ScalarType> X1(n * n);
      initialize_identity_matrix(X1.data(), n);
      vector<ScalarType> X2(n * n);
      ScalarType c = 1;
      // for k=1:q
      const ScalarType alpha = 1;
      const ScalarType beta = 0;
      for (auto k = 1; k <= q; k++) {
         // c = (c*(q-k+1))/((2*q-k+1)*k)
         c = (c * (q - k + 1)) / ((2 * q - k + 1) * k);
         // X = A@X, N=N+c*X, D=D+(-1)^k*c*X
         auto& X_old = k % 2 == 1 ? X1 : X2;
         auto& X_new = k % 2 == 0 ? X1 : X2;
         // new = A @ old
         // new.T = old.T @ A.T
         gemm<ScalarType>("N", "N", &n, &n, &n, &alpha, X_old.data(), &n, A, &n, &beta, X_new.data(), &n);
         ScalarType d = k % 2 == 0 ? c : -c;
         for (Size i = 0; i < n * n; i++) {
            auto x = X_new[i];
            N[i] += c * x;
            D[i] += d * x;
         }
      }
      // solve D@F=N for F
      vector<ScalarType> F1(n * n);
      vector<ScalarType> F2(n * n);
      auto& R = j == 0 ? F : F1;
      // D@R=N
      linear_solve<ScalarType>(n, D.data(), n, N.data());
      R = std::move(N);
      // for k=1:j
      for (auto k = 1; k <= j; j++) {
         // F = F@F
         auto& F_old = k % 2 == 1 ? F1 : F2;
         auto& F_new = k == j ? F : k % 2 == 0 ? F1 : F2;
         // new = old * old
         // new.T = old.T * old.T
         gemm<ScalarType>("N", "N", &n, &n, &n, &alpha, F_old.data(), &n, F_old.data(), &n, &beta, F_new.data(), &n);
      }
   }
#endif

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::exponential(const std::set<std::tuple<Name, Name>>& pairs, int step) const {
      auto guard = exponential_guard();
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         // TODO
      }
      real_base_t<ScalarType> norm_max = norm<-1>();
      auto temporary_tensor_rank = 0;
      real_base_t<ScalarType> temporary_tensor_parameter = 1;
      while (temporary_tensor_parameter * norm_max > 1) {
         temporary_tensor_rank += 1;
         temporary_tensor_parameter *= 1. / 2;
      }
      auto temporary_tensor = *this * temporary_tensor_parameter;

      auto result = identity(pairs);

      auto power_of_temporary_tensor = Tensor<ScalarType, Symmetry, Name>();

      ScalarType series_parameter = 1;
      for (auto i = 1; i <= step; i++) {
         series_parameter /= i;
         if (i == 1) {
            result += temporary_tensor;
         } else if (i == 2) {
            power_of_temporary_tensor = temporary_tensor.contract(temporary_tensor, pairs);
            // result += series_parameter * power_of_temporary_tensor;
            result = series_parameter * power_of_temporary_tensor + result;
            // power_of_temporary_tensor相乘一次后边应该就会稳定, 这个时候将result放在+的右侧, 会使得result边的排列和左侧一样
            // 从而在 i>2 的时候减少转置
         } else {
            power_of_temporary_tensor = power_of_temporary_tensor.contract(temporary_tensor, pairs);
            result += series_parameter * power_of_temporary_tensor;
         }
      }

      for (auto i = 0; i < temporary_tensor_rank; i++) {
         result = result.contract(result, pairs);
      }
      return result;
   }
} // namespace TAT
#endif
