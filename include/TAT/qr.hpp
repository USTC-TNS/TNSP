/**
 * \file qr.hpp
 *
 * Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_QR_HPP
#define TAT_QR_HPP

#include "tensor.hpp"

extern "C" {
void sgeqrf_(const int* m, const int* n, float* A, const int* lda, float* tau, float* work, const int* lwork, int* info);
void dgeqrf_(const int* m, const int* n, double* A, const int* lda, double* tau, double* work, const int* lwork, int* info);
void cgeqrf_(
      const int* m,
      const int* n,
      std::complex<float>* A,
      const int* lda,
      std::complex<float>* tau,
      std::complex<float>* work,
      const int* lwork,
      int* info);
void zgeqrf_(
      const int* m,
      const int* n,
      std::complex<double>* A,
      const int* lda,
      std::complex<double>* tau,
      std::complex<double>* work,
      const int* lwork,
      int* info);
void sgelqf_(const int* m, const int* n, float* A, const int* lda, float* tau, float* work, const int* lwork, int* info);
void dgelqf_(const int* m, const int* n, double* A, const int* lda, double* tau, double* work, const int* lwork, int* info);
void cgelqf_(
      const int* m,
      const int* n,
      std::complex<float>* A,
      const int* lda,
      std::complex<float>* tau,
      std::complex<float>* work,
      const int* lwork,
      int* info);
void zgelqf_(
      const int* m,
      const int* n,
      std::complex<double>* A,
      const int* lda,
      std::complex<double>* tau,
      std::complex<double>* work,
      const int* lwork,
      int* info);
void sorgqr_(const int* m, const int* n, const int* k, float* A, const int* lda, float const* tau, float* work, const int* lwork, int* info);
void dorgqr_(const int* m, const int* n, const int* k, double* A, const int* lda, double const* tau, double* work, const int* lwork, int* info);
void cungqr_(
      const int* m,
      const int* n,
      const int* k,
      std::complex<float>* A,
      const int* lda,
      std::complex<float> const* tau,
      std::complex<float>* work,
      const int* lwork,
      int* info);
void zungqr_(
      const int* m,
      const int* n,
      const int* k,
      std::complex<double>* A,
      const int* lda,
      std::complex<double> const* tau,
      std::complex<double>* work,
      const int* lwork,
      int* info);
void sorglq_(const int* m, const int* n, const int* k, float* A, const int* lda, float const* tau, float* work, const int* lwork, int* info);
void dorglq_(const int* m, const int* n, const int* k, double* A, const int* lda, double const* tau, double* work, const int* lwork, int* info);
void cunglq_(
      const int* m,
      const int* n,
      const int* k,
      std::complex<float>* A,
      const int* lda,
      std::complex<float> const* tau,
      std::complex<float>* work,
      const int* lwork,
      int* info);
void zunglq_(
      const int* m,
      const int* n,
      const int* k,
      std::complex<double>* A,
      const int* lda,
      std::complex<double> const* tau,
      std::complex<double>* work,
      const int* lwork,
      int* info);
}

namespace TAT {
   // template<class ScalarType>
   // void geqrf(const int* m, const int* n, ScalarType* A, const int* lda, ScalarType* tau, ScalarType* work, const int* lwork, int* info);
   // template<class ScalarType>
   // void gelqf(const int* m, const int* n, ScalarType* A, const int* lda, ScalarType* tau, ScalarType* work, const int* lwork, int* info);
   // template<class ScalarType>
   // void
   // orgqr(const int* m, const int* n, const int* k, ScalarType* A, const int* lda, ScalarType* tau, ScalarType* work, const int* lwork, int* info);
   // template<class ScalarType>
   // void
   // orglq(const int* m, const int* n, const int* k, ScalarType* A, const int* lda, ScalarType* tau, ScalarType* work, const int* lwork, int* info);

   template<class ScalarType>
   constexpr void (
         *geqrf)(const int* m, const int* n, ScalarType* A, const int* lda, ScalarType* tau, ScalarType* work, const int* lwork, int* info) = nullptr;
   template<>
   auto geqrf<float> = sgeqrf_;
   template<>
   auto geqrf<double> = dgeqrf_;
   template<>
   auto geqrf<std::complex<float>> = cgeqrf_;
   template<>
   auto geqrf<std::complex<double>> = zgeqrf_;
   template<class ScalarType>
   constexpr void (
         *gelqf)(const int* m, const int* n, ScalarType* A, const int* lda, ScalarType* tau, ScalarType* work, const int* lwork, int* info) = nullptr;
   template<>
   auto gelqf<float> = sgelqf_;
   template<>
   auto gelqf<double> = dgelqf_;
   template<>
   auto gelqf<std::complex<float>> = cgelqf_;
   template<>
   auto gelqf<std::complex<double>> = zgelqf_;
   template<class ScalarType>
   constexpr void (*orgqr)(
         const int* m,
         const int* n,
         const int* k,
         ScalarType* A,
         const int* lda,
         ScalarType* tau,
         ScalarType* work,
         const int* lwork,
         int* info) = nullptr;
   template<>
   auto orgqr<float> = sorgqr_;
   template<>
   auto orgqr<double> = dorgqr_;
   template<>
   auto orgqr<std::complex<float>> = cungqr_;
   template<>
   auto orgqr<std::complex<double>> = zungqr_;
   template<class ScalarType>
   constexpr void (*orglq)(
         const int* m,
         const int* n,
         const int* k,
         ScalarType* A,
         const int* lda,
         ScalarType* tau,
         ScalarType* work,
         const int* lwork,
         int* info) = nullptr;
   template<>
   auto orglq<float> = sorglq_;
   template<>
   auto orglq<double> = dorglq_;
   template<>
   auto orglq<std::complex<float>> = cunglq_;
   template<>
   auto orglq<std::complex<double>> = zunglq_;

   template<class ScalarType>
   void calculate_qr(
         const int& m,
         const int& n,
         const int& min,
         const int& max,
         ScalarType* __restrict data,
         ScalarType* __restrict data_1,
         ScalarType* __restrict data_2,
         bool use_qr_not_lq) {
      // m*n c matrix at data do lq
      // n*m fortran matrix at data do qr
      if (use_qr_not_lq) {
         // c qr -> fortran lq
         // LQ
         //
         // XX   X        XQ
         // XX   XX XX    XX
         // XX = XX XX -> XX
         //
         // XXX   X  XXX    XQQ
         // XXX = XX XXX -> XXQ
         int result;
         int lwork = 64 * max;
         auto work = vector<ScalarType>(lwork);
         auto tau = vector<ScalarType>(min);
         gelqf<ScalarType>(&n, &m, data, &n, tau.data(), work.data(), &lwork, &result);
         if (result != 0) {
            TAT_error("Error in LQ");
         }
         // Q matrix
         // data n*m
         // data_1 min*m
         for (auto i = 0; i < m; i++) {
            std::copy(data + i * n, data + i * n + min, data_1 + i * min);
         }
         orglq<ScalarType>(&min, &m, &min, data_1, &min, tau.data(), work.data(), &lwork, &result);
         if (result != 0) {
            TAT_error("Error in LQ");
         }
         // L matrix
         for (auto i = 0; i < min; i++) {
            std::fill(data_2 + i * n, data_2 + i * n + i, 0);
            std::copy(data + i * n + i, data + i * n + n, data_2 + i * n + i);
         }
      } else {
         // c lq -> fortran qr
         // QR
         //
         // XX   XX       XX
         // XX   XX XX    QX
         // XX = XX  X -> QQ
         //
         // XXX   XX XXX    XXX
         // XXX = XX  XX -> QXX
         int result;
         int lwork = 64 * max;
         auto work = vector<ScalarType>(lwork);
         auto tau = vector<ScalarType>(min);
         geqrf<ScalarType>(&n, &m, data, &n, tau.data(), work.data(), &lwork, &result);
         if (result != 0) {
            TAT_error("Error in QR");
         }
         // Q matrix
         std::copy(data, data + n * min, data_2); // 多复制了无用的上三角部分
         // fortran
         // data n*m
         // data_2 n*min
         orgqr<ScalarType>(&n, &min, &min, data_2, &n, tau.data(), work.data(), &lwork, &result);
         // same size of lwork
         if (result != 0) {
            TAT_error("Error in QR");
         }
         // R matrix
         for (auto i = 0; i < min; i++) {
            std::copy(data + n * i, data + n * i + i + 1, data_1 + min * i);
            std::fill(data_1 + min * i + i + 1, data_1 + min * i + min, 0);
         }
         std::copy(data + n * min, data + n * m, data_1 + min * min);
         // 若为第一种, 则这个copy不做事
      }
   }

   template<class ScalarType, class Symmetry>
   typename Tensor<ScalarType, Symmetry>::qr_result
   Tensor<ScalarType, Symmetry>::qr(char free_name_direction, const std::set<Name>& free_name_set, Name common_name_q, Name common_name_r) const {
      // free_name_set不需要做特殊处理即可自动处理不准确的边名
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
      const auto rank = names.size();
      // 判断使用lq还是qr
      bool use_r_name;
      if (free_name_direction == 'r' || free_name_direction == 'R') {
         use_r_name = true;
      } else if (free_name_direction == 'q' || free_name_direction == 'Q') {
         use_r_name = false;
      } else {
         TAT_error("Invalid direction in QR");
      };
      bool use_qr_not_lq = names.empty() || ((free_name_set.find(names.back()) != free_name_set.end()) == use_r_name);
      // merge
      auto free_name_1 = std::vector<Name>();
      auto free_name_2 = std::vector<Name>();
      auto reversed_set_1 = std::set<Name>();
      auto reversed_set_2 = std::set<Name>();
      auto reversed_set_origin = std::set<Name>();
      auto result_name_1 = std::vector<Name>();
      auto result_name_2 = std::vector<Name>();
      auto free_names_and_edges_1 = std::vector<std::tuple<Name, BoseEdge<Symmetry, true>>>();
      auto free_names_and_edges_2 = std::vector<std::tuple<Name, BoseEdge<Symmetry, true>>>();
      free_name_1.reserve(rank);
      free_name_2.reserve(rank);
      result_name_1.reserve(rank + 1);
      result_name_2.reserve(rank + 1);
      free_names_and_edges_1.reserve(rank);
      free_names_and_edges_2.reserve(rank);
      result_name_2.push_back(use_qr_not_lq ? common_name_r : common_name_q);
      for (auto i = 0; i < names.size(); i++) {
         const auto& n = names[i];
         // set.find() != set.end() => n in the set
         // (!=) == use_r_name => n in the r name
         // (!=) == use_r_name == use_qr_not_lq => in the second name
         if ((free_name_set.find(n) != free_name_set.end()) == use_r_name == use_qr_not_lq) {
            free_name_2.push_back(n);
            result_name_2.push_back(n);
            free_names_and_edges_2.push_back({n, {core->edges[i].map}});
            if constexpr (is_fermi) {
               if (core->edges[i].arrow) {
                  reversed_set_2.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         } else {
            free_name_1.push_back(n);
            result_name_1.push_back(n);
            free_names_and_edges_1.push_back({n, {core->edges[i].map}});
            if constexpr (is_fermi) {
               if (core->edges[i].arrow) {
                  reversed_set_1.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         }
      }
      result_name_1.push_back(use_qr_not_lq ? common_name_q : common_name_r);
      auto tensor_merged = edge_operator({}, {}, reversed_set_origin, {{QR_1, free_name_1}, {QR_2, free_name_2}}, {QR_1, QR_2});
      // call lapack
      auto common_edge_1 = Edge<Symmetry>();
      auto common_edge_2 = Edge<Symmetry>();
      for (const auto& [sym, _] : tensor_merged.core->blocks) {
         auto m = tensor_merged.core->edges[0].map.at(sym[0]);
         auto n = tensor_merged.core->edges[1].map.at(sym[1]);
         auto k = m > n ? n : m;
         common_edge_1.map[sym[1]] = k;
         common_edge_2.map[sym[0]] = k;
      }
      auto tensor_1 = Tensor<ScalarType, Symmetry>{{QR_1, QR_2}, {std::move(tensor_merged.core->edges[0]), std::move(common_edge_1)}};
      auto tensor_2 = Tensor<ScalarType, Symmetry>{{QR_1, QR_2}, {std::move(common_edge_2), std::move(tensor_merged.core->edges[1])}};
      for (auto& [symmetries, block] : tensor_merged.core->blocks) {
         auto* data_1 = tensor_1.core->blocks.at(symmetries).data();
         auto* data_2 = tensor_2.core->blocks.at(symmetries).data();
         auto* data = block.data();
         const int m = tensor_1.core->edges[0].map.at(symmetries[0]);
         const int n = tensor_2.core->edges[1].map.at(symmetries[1]);
         const int k = m > n ? n : m;
         const int max = m > n ? m : n;
         if (m * n != 0) {
            calculate_qr<ScalarType>(m, n, k, max, data, data_1, data_2, use_qr_not_lq);
         }
      }
      const auto& tensor_q = use_qr_not_lq ? tensor_1 : tensor_2;
      const auto& tensor_r = use_qr_not_lq ? tensor_2 : tensor_1;
      // reversed_set_1.insert(common_name_q); ?? TODO ??
      // cut 可以放在这里自动进行 TODO
      auto new_tensor_1 = tensor_1.template edge_operator<true>(
            {{QR_2, use_qr_not_lq ? common_name_q : common_name_r}}, {{QR_1, free_names_and_edges_1}}, reversed_set_1, {}, result_name_1);
      auto new_tensor_2 = tensor_2.template edge_operator<true>(
            {{QR_1, use_qr_not_lq ? common_name_r : common_name_q}}, {{QR_2, free_names_and_edges_2}}, reversed_set_2, {}, result_name_2);
      return {std::move(use_qr_not_lq ? new_tensor_1 : new_tensor_2), std::move(use_qr_not_lq ? new_tensor_2 : new_tensor_1)};
   }
} // namespace TAT
#endif
