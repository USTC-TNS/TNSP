/**
 * \file exponential.hpp
 *
 * Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
   void linear_solve(int n, ScalarType* A, int nrhs, ScalarType* B, ScalarType* X) {
      // AX=B
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
      matrix_transpose(nrhs, n, BT.data(), X);
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
      int int_n = n;
      // j = max(0, 1+floor(log2(|A|_inf)))
      auto j = std::max(0, 1 + int(std::log2(max_of_abs(A, n))));
      // A = A/2^j
      ScalarType parameter = ScalarType(1) / ScalarType(1 << j);
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
         c = (c * ScalarType(q - k + 1)) / ScalarType((2 * q - k + 1) * k);
         // X = A@X, N=N+c*X, D=D+(-1)^k*c*X
         auto& X_old = k % 2 == 1 ? X1 : X2;
         auto& X_new = k % 2 == 0 ? X1 : X2;
         // new = A @ old
         // new.T = old.T @ A.T
         gemm<ScalarType>("N", "N", &int_n, &int_n, &int_n, &alpha, X_old.data(), &int_n, A, &int_n, &beta, X_new.data(), &int_n);
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
      auto* R = j == 0 ? F : F1.data();
      // D@R=N
      linear_solve<ScalarType>(n, D.data(), n, N.data(), R);
      // for k=1:j
      for (auto k = 1; k <= j; k++) {
         // F = F@F
         const auto* F_old = k % 2 == 1 ? F1.data() : F2.data();
         auto* F_new = k == j ? F : k % 2 == 0 ? F1.data() : F2.data();
         // new = old * old
         // new.T = old.T * old.T
         gemm<ScalarType>("N", "N", &int_n, &int_n, &int_n, &alpha, F_old, &int_n, F_old, &int_n, &beta, F_new, &int_n);
      }
   }
#endif

   template<typename ScalarType, typename Symmetry, typename Name>
   template<typename SetNameAndName>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::exponential(const SetNameAndName& pairs, int step) const {
      auto timer_guard = exponential_guard();
      auto pmr_guard = scope_resource<>();

      Rank rank = names.size();
      Rank half_rank = rank / 2;
      auto merge_map = pmr::map<Name, pmr::vector<Name>>();
      auto& merge_1 = merge_map[InternalName<Name>::Exp_1];
      merge_1.resize(half_rank);
      auto& merge_2 = merge_map[InternalName<Name>::Exp_2];
      merge_2.resize(half_rank);
      auto split_map_result = pmr::map<Name, pmr::vector<std::tuple<Name, BoseEdge<Symmetry>>>>();
      auto& split_1 = split_map_result[InternalName<Name>::Exp_1];
      split_1.resize(half_rank);
      auto& split_2 = split_map_result[InternalName<Name>::Exp_2];
      split_2.resize(half_rank);

      auto valid_index = pmr::vector<bool>(rank, true);
      Rank current_index = half_rank;
      for (Rank i = rank; i-- > 0;) {
         if (valid_index[i]) {
            const auto& name_to_found = names[i];
            for (const auto& [a, b] : pairs) {
               if (a == name_to_found) {
                  auto ia = name_to_index.at(a);
                  auto ib = name_to_index.at(b);
                  valid_index[ib] = false;
                  current_index--;
                  merge_1[current_index] = a;
                  merge_2[current_index] = b;
                  split_1[current_index] = {a, core->edges[ia].map};
                  split_2[current_index] = {b, core->edges[ib].map};
                  break;
               }
               if (b == name_to_found) {
                  auto ia = name_to_index.at(a);
                  auto ib = name_to_index.at(b);
                  valid_index[ia] = false;
                  current_index--;
                  merge_1[current_index] = a;
                  merge_2[current_index] = b;
                  split_1[current_index] = {a, core->edges[ia].map};
                  split_2[current_index] = {b, core->edges[ib].map};
                  break;
               }
            }
         }
      }
      auto reverse_set = pmr::set<Name>();
      if constexpr (is_fermi_symmetry_v<Symmetry>) {
         for (Rank i = 0; i < rank; i++) {
            if (core->edges[i].arrow) {
               reverse_set.insert(names[i]);
            }
         }
      }
      auto merged_names = pmr::vector<Name>();
      merged_names.reserve(2);
      auto result_names = pmr::vector<Name>();
      result_names.reserve(rank);
      if (names.empty() || names.back() == merge_1.back()) {
         // 2 1
         merged_names.push_back(InternalName<Name>::Exp_2);
         merged_names.push_back(InternalName<Name>::Exp_1);
         for (const auto& i : merge_2) {
            result_names.push_back(i);
         }
         for (const auto& i : merge_1) {
            result_names.push_back(i);
         }
      } else {
         // 1 2
         merged_names.push_back(InternalName<Name>::Exp_1);
         merged_names.push_back(InternalName<Name>::Exp_2);
         for (const auto& i : merge_1) {
            result_names.push_back(i);
         }
         for (const auto& i : merge_2) {
            result_names.push_back(i);
         }
      }
      auto tensor_merged = edge_operator({}, {}, reverse_set, merge_map, merged_names);
      auto result = tensor_merged.same_shape();
      for (auto& [symmetries, data_source] : tensor_merged.core->blocks) {
         auto& data_destination = result.core->blocks.at(symmetries);
         auto n = tensor_merged.core->edges[0].map.at(symmetries[0]);
         matrix_exponential(n, data_source.data(), data_destination.data(), step);
      }
      return result.edge_operator({}, split_map_result, reverse_set, {}, result_names);
   }
} // namespace TAT
#endif
