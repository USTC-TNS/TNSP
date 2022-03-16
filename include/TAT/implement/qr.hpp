/**
 * \file qr.hpp
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
#ifndef TAT_QR_HPP
#define TAT_QR_HPP

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"
#include "transpose.hpp"

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

   inline timer qr_kernel_guard("qr_kernel");

   namespace detail {
      template<typename ScalarType>
      constexpr void (
            *geqrf)(const int* m, const int* n, ScalarType* A, const int* lda, ScalarType* tau, ScalarType* work, const int* lwork, int* info) =
            nullptr;
      template<>
      inline auto geqrf<float> = sgeqrf_;
      template<>
      inline auto geqrf<double> = dgeqrf_;
      template<>
      inline auto geqrf<std::complex<float>> = cgeqrf_;
      template<>
      inline auto geqrf<std::complex<double>> = zgeqrf_;
      template<typename ScalarType>
      constexpr void (
            *gelqf)(const int* m, const int* n, ScalarType* A, const int* lda, ScalarType* tau, ScalarType* work, const int* lwork, int* info) =
            nullptr;
      template<>
      inline auto gelqf<float> = sgelqf_;
      template<>
      inline auto gelqf<double> = dgelqf_;
      template<>
      inline auto gelqf<std::complex<float>> = cgelqf_;
      template<>
      inline auto gelqf<std::complex<double>> = zgelqf_;
      template<typename ScalarType>
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
      inline auto orgqr<float> = sorgqr_;
      template<>
      inline auto orgqr<double> = dorgqr_;
      template<>
      inline auto orgqr<std::complex<float>> = cungqr_;
      template<>
      inline auto orgqr<std::complex<double>> = zungqr_;
      template<typename ScalarType>
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
      inline auto orglq<float> = sorglq_;
      template<>
      inline auto orglq<double> = dorglq_;
      template<>
      inline auto orglq<std::complex<float>> = cunglq_;
      template<>
      inline auto orglq<std::complex<double>> = zunglq_;

      template<typename ScalarType>
      int to_int(const ScalarType& value) {
         if constexpr (is_complex<ScalarType>) {
            return int(value.real());
         } else {
            return int(value);
         }
      }

      template<typename ScalarType>
      void calculate_qr_kernel(
            const int& m,
            const int& n,
            const int& min,
            const int& max,
            ScalarType* __restrict data,
            ScalarType* __restrict data_1,
            ScalarType* __restrict data_2,
            bool use_qr_not_lq) {
         // m*n c matrix
         // n*m fortran matrix
         if (use_qr_not_lq) {
            // c qr -> fortran lq
            // LQ
            //
            // here X means matrix content, and Q means the data to generate Q matrix
            //
            // XX   X        XQ
            // XX   XX XX    XX
            // XX = XX XX -> XX
            //
            // XXX   X  XXX    XQQ
            // XXX = XX XXX -> XXQ
            int result;
            auto tau = no_initialize::pmr::vector<ScalarType>(min);
            const int lwork_query = -1;
            ScalarType float_lwork;
            gelqf<ScalarType>(&n, &m, data, &n, tau.data(), &float_lwork, &lwork_query, &result);
            if (result != 0) {
               detail::what_if_lapack_error("Error in LQ");
            }
            const int lwork = to_int(float_lwork);
            auto work = no_initialize::pmr::vector<ScalarType>(lwork);
            gelqf<ScalarType>(&n, &m, data, &n, tau.data(), work.data(), &lwork, &result);
            if (result != 0) {
               detail::what_if_lapack_error("Error in LQ");
            }
            // Q matrix
            // data n*m
            // data_1 min*m
            for (auto i = 0; i < m; i++) {
               // it does not copy entire matrix for the first situation
               // but it still copy useless lower triangular part
               std::copy(data + i * n, data + i * n + min, data_1 + i * min);
            }
            orglq<ScalarType>(&min, &m, &min, data_1, &min, tau.data(), work.data(), &lwork, &result);
            // WRONG -> orglq<ScalarType>(&min, &min, &min, data_1, &min, tau.data(), work.data(), &lwork, &result);
            if (result != 0) {
               detail::what_if_lapack_error("Error in LQ");
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
            auto tau = no_initialize::pmr::vector<ScalarType>(min);
            const int lwork_query = -1;
            ScalarType float_lwork;
            geqrf<ScalarType>(&n, &m, data, &n, tau.data(), &float_lwork, &lwork_query, &result);
            if (result != 0) {
               detail::what_if_lapack_error("Error in QR");
            }
            const int lwork = to_int(float_lwork);
            auto work = no_initialize::pmr::vector<ScalarType>(lwork);
            geqrf<ScalarType>(&n, &m, data, &n, tau.data(), work.data(), &lwork, &result);
            if (result != 0) {
               detail::what_if_lapack_error("Error in QR");
            }
            // Q matrix
            // it does copy the entire matrix for the both situation, it is different to the c qr branch
            std::copy(data, data + n * min, data_2); // this copy useless upper triangular part
            // fortran
            // data n*m
            // data_2 n*min
            orgqr<ScalarType>(&n, &min, &min, data_2, &n, tau.data(), work.data(), &lwork, &result);
            // WRONG -> orgqr<ScalarType>(&min, &min, &min, data_2, &n, tau.data(), work.data(), &lwork, &result);
            // same size of lwork
            if (result != 0) {
               detail::what_if_lapack_error("Error in QR");
            }
            // R matrix
            for (auto i = 0; i < min; i++) {
               std::copy(data + n * i, data + n * i + i + 1, data_1 + min * i);
               std::fill(data_1 + min * i + i + 1, data_1 + min * i + min, 0);
            }
            std::copy(data + n * min, data + n * m, data_1 + min * min);
            // for the first situation, min == m, this copy do nothing
         }
      }

      template<typename ScalarType>
      void calculate_qr(
            const int& m,
            const int& n,
            const int& min,
            const int& max,
            ScalarType* __restrict data,
            ScalarType* __restrict data_1,
            ScalarType* __restrict data_2,
            bool use_qr_not_lq) {
         auto kernel_guard = qr_kernel_guard();
         // sometimes, transpose before qr/lq is faster, there is a simular operation in svd
         // by testing, m > n is better
         if (m > n) {
            auto new_data = no_initialize::pmr::vector<ScalarType>(n * m);
            auto old_data_1 = no_initialize::pmr::vector<ScalarType>(n * min);
            auto old_data_2 = no_initialize::pmr::vector<ScalarType>(min * m);
            matrix_transpose(m, n, data, new_data.data());
            calculate_qr_kernel(n, m, min, max, new_data.data(), old_data_1.data(), old_data_2.data(), !use_qr_not_lq);
            matrix_transpose(n, min, old_data_1.data(), data_2);
            matrix_transpose(min, m, old_data_2.data(), data_1);
         } else {
            calculate_qr_kernel(m, n, min, max, data, data_1, data_2, use_qr_not_lq);
         }
      }
   } // namespace detail

   inline timer qr_guard("qr");

   template<typename ScalarType, typename Symmetry, typename Name>
   typename Tensor<ScalarType, Symmetry, Name>::qr_result Tensor<ScalarType, Symmetry, Name>::qr(
         char free_name_direction,
         const std::set<Name>& free_name_set,
         const Name& common_name_q,
         const Name& common_name_r) const {
      auto pmr_guard = scope_resource(default_buffer_size);
      auto timer_guard = qr_guard();

      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
      const auto rank = get_rank();

      if constexpr (debug_mode) {
         // check free_name_set is valid
         for (const auto& name : free_name_set) {
            if (auto found = find_rank_from_name(name); found == names.end()) {
               detail::error("Missing name in qr");
            }
         }
      }

      // determine do LQ or QR
      bool use_r_name;
      if (free_name_direction == 'r' || free_name_direction == 'R') {
         use_r_name = true;
      } else if (free_name_direction == 'q' || free_name_direction == 'Q') {
         use_r_name = false;
      } else {
         detail::error("Invalid direction in QR");
      };
      bool use_qr_not_lq = names.empty() || ((free_name_set.find(names.back()) != free_name_set.end()) == use_r_name);

      // merge plan
      auto free_name_1 = pmr::vector<Name>(); // part of merge map
      auto free_name_2 = pmr::vector<Name>(); // part of merge map
      auto reversed_set_origin = pmr::unordered_set<Name>();
      // result name is trivial

      // split plan
      auto reversed_set_1 = pmr::unordered_set<Name>();
      auto reversed_set_2 = pmr::unordered_set<Name>();
      auto result_name_1 = std::vector<Name>();
      auto result_name_2 = std::vector<Name>();
      auto free_names_and_edges_1 = pmr::vector<std::tuple<Name, edge_segment_t<Symmetry, true>>>(); // part of split map
      auto free_names_and_edges_2 = pmr::vector<std::tuple<Name, edge_segment_t<Symmetry, true>>>(); // part of split map

      free_name_1.reserve(rank);
      free_name_2.reserve(rank);
      result_name_1.reserve(rank + 1);
      result_name_2.reserve(rank + 1);
      free_names_and_edges_1.reserve(rank);
      free_names_and_edges_2.reserve(rank);

      result_name_2.push_back(use_qr_not_lq ? common_name_r : common_name_q);
      for (Rank i = 0; i < get_rank(); i++) {
         const auto& n = names[i];
         // set.find() != set.end() => n in the set
         // (!=) == use_r_name => n in the r name
         // (!=) == use_r_name == use_qr_not_lq => in the second name
         if ((free_name_set.find(n) != free_name_set.end()) == use_r_name == use_qr_not_lq) {
            // tensor_2 side
            free_name_2.push_back(n);
            result_name_2.push_back(n);
            free_names_and_edges_2.push_back({n, {edges(i).segment}});
            if constexpr (is_fermi) {
               if (edges(i).arrow) {
                  reversed_set_2.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         } else {
            // tensor_1 side
            free_name_1.push_back(n);
            result_name_1.push_back(n);
            free_names_and_edges_1.push_back({n, {edges(i).segment}});
            if constexpr (is_fermi) {
               if (edges(i).arrow) {
                  reversed_set_1.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         }
      }
      result_name_1.push_back(use_qr_not_lq ? common_name_q : common_name_r);

      auto tensor_merged = edge_operator_implement(
            empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
            reversed_set_origin,
            pmr::unordered_map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::QR_1, std::move(free_name_1)},
                  {InternalName<Name>::QR_2, std::move(free_name_2)}},
            std::vector<Name>{InternalName<Name>::QR_1, InternalName<Name>::QR_2},
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());

      // prepare result tensor
      auto common_edge_1 = Edge<Symmetry>();
      auto common_edge_2 = Edge<Symmetry>();
      // arrow all false currently, arrow check is processed at the last
      for (const auto& [syms, block] : tensor_merged.core->blocks) {
         auto m = tensor_merged.edges(0).get_dimension_from_symmetry(syms[0]);
         auto n = tensor_merged.edges(1).get_dimension_from_symmetry(syms[1]);
         auto k = m > n ? n : m;
         common_edge_1.segment.emplace_back(syms[1], k);
         common_edge_2.segment.emplace_back(syms[0], k);
      }
      auto tensor_1 = Tensor<ScalarType, Symmetry, Name>{
            {InternalName<Name>::QR_1, use_qr_not_lq ? common_name_q : common_name_r},
            {std::move(tensor_merged.edges(0)), std::move(common_edge_1)}};
      auto tensor_2 = Tensor<ScalarType, Symmetry, Name>{
            {use_qr_not_lq ? common_name_r : common_name_q, InternalName<Name>::QR_2},
            {std::move(common_edge_2), std::move(tensor_merged.edges(1))}};

      // call lapack
      for (auto& [symmetries, block] : tensor_merged.core->blocks) {
         auto* data_1 = tensor_1.blocks(symmetries).data();
         auto* data_2 = tensor_2.blocks(symmetries).data();
         auto* data = block.data();
         const int m = tensor_1.edges(0).get_dimension_from_symmetry(symmetries[0]);
         const int n = tensor_2.edges(1).get_dimension_from_symmetry(symmetries[1]);
         const int k = m > n ? n : m;
         const int max = m > n ? m : n;
         if (m * n != 0) {
            detail::calculate_qr<ScalarType>(m, n, k, max, data, data_1, data_2, use_qr_not_lq);
         }
      }

      // this is simular to svd
      //
      // no matter whether it is qr or lq
      // tensor 1 common edge should be true
      // tensor 2 common edge should be false
      //
      // what need to do is: tensor_1 not_apply_reverse
      //
      // what did in the following code:
      // QR:
      // tensor_1 not_apply_reverse
      // LQ:
      // tensor_2 apply_reverse

      // tensor_1 not_apply_reverse -> tensor_2 apply_reverse operation is:
      // tensor_1 not_apply_reverse and tensor_2 apply_reverse, it is a conserved operation
      // so now, it maintains that tensor Q always have a true edge
      if constexpr (is_fermi) {
         (use_qr_not_lq ? reversed_set_1 : reversed_set_2).insert(common_name_q);
      }
      auto new_tensor_1 = tensor_1.edge_operator_implement(
            pmr::unordered_map<Name, pmr::vector<std::tuple<Name, edge_segment_t<Symmetry, true>>>>{
                  {InternalName<Name>::QR_1, std::move(free_names_and_edges_1)}},
            reversed_set_1,
            empty_list<std::pair<Name, empty_list<Name>>>(),
            std::move(result_name_1),
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());
      auto new_tensor_2 = tensor_2.edge_operator_implement(
            pmr::unordered_map<Name, pmr::vector<std::tuple<Name, edge_segment_t<Symmetry, true>>>>{
                  {InternalName<Name>::QR_2, std::move(free_names_and_edges_2)}},
            reversed_set_2,
            empty_list<std::pair<Name, empty_list<Name>>>(),
            std::move(result_name_2),
            false,
            empty_list<Name>(),
            use_qr_not_lq ? pmr::unordered_set<Name>{} : pmr::unordered_set<Name>{common_name_q},
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());
      return {std::move(use_qr_not_lq ? new_tensor_1 : new_tensor_2), std::move(use_qr_not_lq ? new_tensor_2 : new_tensor_1)};
   }
} // namespace TAT
#endif
