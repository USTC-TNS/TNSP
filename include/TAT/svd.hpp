/**
 * \file svd.hpp
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
#ifndef TAT_SVD_HPP
#define TAT_SVD_HPP

#include "tensor.hpp"
#include "timer.hpp"
#include "transpose.hpp"

extern "C" {
int sgesvd_(
      const char* job_u,
      const char* job_vt,
      const int* m,
      const int* n,
      const float* a,
      const int* ld_a,
      float* s,
      float* u,
      const int* ld_u,
      float* vt,
      const int* ld_vt,
      float* work,
      const int* lwork,
      int* info);
int dgesvd_(
      const char* job_u,
      const char* job_vt,
      const int* m,
      const int* n,
      const double* a,
      const int* ld_a,
      double* s,
      double* u,
      const int* ld_u,
      double* vt,
      const int* ld_vt,
      double* work,
      const int* lwork,
      int* info);
int cgesvd_(
      const char* job_u,
      const char* job_vt,
      const int* m,
      const int* n,
      const std::complex<float>* a,
      const int* ld_a,
      float* s,
      std::complex<float>* u,
      const int* ld_u,
      std::complex<float>* vt,
      const int* ld_vt,
      std::complex<float>* work,
      const int* lwork,
      float* rwork,
      int* info);
int zgesvd_(
      const char* job_u,
      const char* job_vt,
      const int* m,
      const int* n,
      const std::complex<double>* a,
      const int* ld_a,
      double* s,
      std::complex<double>* u,
      const int* ld_u,
      std::complex<double>* vt,
      const int* ld_vt,
      std::complex<double>* work,
      const int* lwork,
      double* rwork,
      int* info);
}

namespace TAT {
   template<typename ScalarType, typename Symmetry>
   [[nodiscard]] Tensor<ScalarType, Symmetry> singular_to_tensor(const std::map<Symmetry, vector<real_base_t<ScalarType>>>& singular) {
      auto symmetries = std::vector<Edge<Symmetry>>(2);
      for (const auto& [symmetry, values] : singular) {
         auto dimension = values.size();
         symmetries[0].map[-symmetry] = dimension;
         symmetries[1].map[symmetry] = dimension;
      }
      if constexpr (is_fermi_symmetry_v<Symmetry>) {
         symmetries[0].arrow = false;
         symmetries[1].arrow = true;
      }
      auto result = Tensor<ScalarType, Symmetry>({internal_name::SVD_U, internal_name::SVD_V}, std::move(symmetries));
      for (auto& [symmetries, data_destination] : result.core->blocks) {
         const auto& data_source = singular.at(symmetries[1]);
         auto dimension = data_source.size();
         auto dimension_plus_one = dimension + 1;
         std::fill(data_destination.begin(), data_destination.end(), 0);
         for (auto i = 0; i < data_source.size(); i++) {
            data_destination[i * dimension_plus_one] = data_source[i];
         }
      }
      return result;
   }

   template<typename ScalarType>
   void calculate_svd_kernel(
         const int& m,
         const int& n,
         const int& min,
         const int& max,
         const ScalarType* a,
         ScalarType* u,
         real_base_t<ScalarType>* s,
         ScalarType* vt);

   template<typename ScalarType>
   void calculate_svd(
         const int& m,
         const int& n,
         const int& min,
         const int& max,
         const ScalarType* a,
         ScalarType* u,
         real_base_t<ScalarType>* s,
         ScalarType* vt) {
      auto kernel_guard = svd_kernel_guard();
      if (m > n) {
         auto new_a = vector<ScalarType>(n * m);
         auto old_u = vector<ScalarType>(n * min);
         auto old_vt = vector<ScalarType>(min * m);
         // new_a = a^T
         // u s vt = a
         // vt^T s u^T = a^T
         // old_u = vt^T
         // old_vt = u^T
         matrix_transpose(m, n, a, new_a.data()); // m*n -> n*m
         calculate_svd_kernel(n, m, min, max, new_a.data(), old_u.data(), s, old_vt.data());
         matrix_transpose(n, min, old_u.data(), vt); // n*min -> min*n
         matrix_transpose(min, m, old_vt.data(), u); // min*m -> m*min
      } else {
         calculate_svd_kernel(m, n, min, max, a, u, s, vt);
      }
   }

   template<>
   inline void
   calculate_svd_kernel<float>(const int& m, const int& n, const int& min, const int& max, const float* a, float* u, float* s, float* vt) {
      int result;
      const int lwork_query = -1;
      float float_lwork;
      sgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, &float_lwork, &lwork_query, &result);
      const int lwork = int(float_lwork);
      auto work = vector<float>(lwork);
      sgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, &result);
      if (result != 0) {
         TAT_warning_or_error_when_lapack_error("Error in GESVD");
      }
   }
   template<>
   inline void
   calculate_svd_kernel<double>(const int& m, const int& n, const int& min, const int& max, const double* a, double* u, double* s, double* vt) {
      int result;
      const int lwork_query = -1;
      double float_lwork;
      dgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, &float_lwork, &lwork_query, &result);
      const int lwork = int(float_lwork);
      auto work = vector<double>(lwork);
      dgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, &result);
      if (result != 0) {
         TAT_warning_or_error_when_lapack_error("Error in GESVD");
      }
   }
   template<>
   inline void calculate_svd_kernel<std::complex<float>>(
         const int& m,
         const int& n,
         const int& min,
         const int& max,
         const std::complex<float>* a,
         std::complex<float>* u,
         float* s,
         std::complex<float>* vt) {
      int result;
      auto rwork = vector<float>(5 * min);
      const int lwork_query = -1;
      std::complex<float> float_lwork;
      cgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, &float_lwork, &lwork_query, rwork.data(), &result);
      const int lwork = int(float_lwork.real());
      auto work = vector<std::complex<float>>(lwork);
      cgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, rwork.data(), &result);
      if (result != 0) {
         TAT_warning_or_error_when_lapack_error("Error in GESVD");
      }
   }
   template<>
   inline void calculate_svd_kernel<std::complex<double>>(
         const int& m,
         const int& n,
         const int& min,
         const int& max,
         const std::complex<double>* a,
         std::complex<double>* u,
         double* s,
         std::complex<double>* vt) {
      int result;
      auto rwork = vector<double>(5 * min);
      const int lwork_query = -1;
      std::complex<double> float_lwork;
      zgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, &float_lwork, &lwork_query, rwork.data(), &result);
      const int lwork = int(float_lwork.real());
      auto work = vector<std::complex<double>>(lwork);
      zgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, rwork.data(), &result);
      if (result != 0) {
         TAT_warning_or_error_when_lapack_error("Error in GESVD");
      }
   }

   template<typename ScalarType, typename Symmetry>
   typename Tensor<ScalarType, Symmetry>::svd_result
   Tensor<ScalarType, Symmetry>::svd(const std::set<Name>& free_name_set_u, const Name& common_name_u, const Name& common_name_v, Size cut) const {
      auto guard = svd_guard();
      // free_name_set_u不需要做特殊处理即可自动处理不准确的边名
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
      const auto rank = names.size();
      // merge
      auto free_name_u = std::vector<Name>();
      auto free_name_v = std::vector<Name>();
      auto reversed_set_u = std::set<Name>();
      auto reversed_set_v = std::set<Name>();
      auto reversed_set_origin = std::set<Name>();
      auto result_name_u = std::vector<Name>();
      auto result_name_v = std::vector<Name>();
      auto free_names_and_edges_u = std::vector<std::tuple<Name, BoseEdge<Symmetry, true>>>();
      auto free_names_and_edges_v = std::vector<std::tuple<Name, BoseEdge<Symmetry, true>>>();
      free_name_u.reserve(rank);
      free_name_v.reserve(rank);
      result_name_u.reserve(rank + 1);
      result_name_v.reserve(rank + 1);
      free_names_and_edges_u.reserve(rank);
      free_names_and_edges_v.reserve(rank);
      result_name_v.push_back(common_name_v);
      for (auto i = 0; i < names.size(); i++) {
         const auto& n = names[i];
         if (free_name_set_u.find(n) != free_name_set_u.end()) {
            free_name_u.push_back(n);
            result_name_u.push_back(n);
            free_names_and_edges_u.push_back({n, {core->edges[i].map}});
            if constexpr (is_fermi) {
               if (core->edges[i].arrow) {
                  reversed_set_u.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         } else {
            free_name_v.push_back(n);
            result_name_v.push_back(n);
            free_names_and_edges_v.push_back({n, {core->edges[i].map}});
            if constexpr (is_fermi) {
               if (core->edges[i].arrow) {
                  reversed_set_v.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         }
      }
      result_name_u.push_back(common_name_u);
      const bool put_v_right = free_name_v.empty() || free_name_v.back() == names.back();
      auto tensor_merged = edge_operator(
            {},
            {},
            reversed_set_origin,
            {{internal_name::SVD_U, free_name_u}, {internal_name::SVD_V, free_name_v}},
            put_v_right ? std::vector<Name>{internal_name::SVD_U, internal_name::SVD_V} :
                          std::vector<Name>{internal_name::SVD_V, internal_name::SVD_U});
      // tensor -> SVD_U -O- SVD_V
      // call GESVD
      auto common_edge_1 = Edge<Symmetry>();
      auto common_edge_2 = Edge<Symmetry>();
      // arrow always false
      for (const auto& [sym, _] : tensor_merged.core->blocks) {
         auto m = tensor_merged.core->edges[0].map.at(sym[0]);
         auto n = tensor_merged.core->edges[1].map.at(sym[1]);
         auto k = m > n ? n : m;
         common_edge_1.map[sym[1]] = k;
         common_edge_2.map[sym[0]] = k;
      }
      auto tensor_1 = Tensor<ScalarType, Symmetry>{
            put_v_right ? std::vector<Name>{internal_name::SVD_U, internal_name::SVD_V} :
                          std::vector<Name>{internal_name::SVD_V, internal_name::SVD_U},
            {std::move(tensor_merged.core->edges[0]), std::move(common_edge_1)}};
      auto tensor_2 = Tensor<ScalarType, Symmetry>{
            put_v_right ? std::vector<Name>{internal_name::SVD_U, internal_name::SVD_V} :
                          std::vector<Name>{internal_name::SVD_V, internal_name::SVD_U},
            {std::move(common_edge_2), std::move(tensor_merged.core->edges[1])}};
      auto result_s = std::map<Symmetry, vector<real_base_t<ScalarType>>>();
      for (const auto& [symmetries, block] : tensor_merged.core->blocks) {
         auto* data_u = tensor_1.core->blocks.at(symmetries).data();
         auto* data_v = tensor_2.core->blocks.at(symmetries).data();
         const auto* data = block.data();
         const int m = tensor_1.core->edges[0].map.at(symmetries[0]);
         const int n = tensor_2.core->edges[1].map.at(symmetries[1]);
         const int k = m > n ? n : m;
         const int max = m > n ? m : n;
         auto s = vector<real_base_t<ScalarType>>(k);
         auto* s_data = s.data();
         if (m * n != 0) {
            calculate_svd<ScalarType>(m, n, k, max, data, data_u, s_data, data_v);
         }
         result_s[symmetries[put_v_right]] = std::move(s);
      }

      // 分析cut方案
      Size total_dimension = 0;
      for (const auto& [symmetry, vector_s] : result_s) {
         total_dimension += vector_s.size();
      }
      auto remain_dimension_u = std::map<Symmetry, Size>();
      auto remain_dimension_v = std::map<Symmetry, Size>();
      if (cut != -1 && cut < total_dimension) {
         // auto remain_dimension = std::map<Symmetry, Size>();
         for (const auto& [symmetry, vector_s] : result_s) {
            remain_dimension_u[symmetry] = 0;
            remain_dimension_v[-symmetry] = 0;
         }
         for (Size i = 0; i < cut; i++) {
            Symmetry maximum_position;
            real_base_t<ScalarType> maximum_singular = 0;
            for (const auto& [symmetry, vector_s] : result_s) {
               if (auto& this_remain = remain_dimension_u.at(symmetry); this_remain != vector_s.size()) {
                  if (auto this_singular = vector_s[this_remain]; this_singular > maximum_singular) {
                     maximum_singular = this_singular;
                     maximum_position = symmetry;
                  }
               }
            }
            remain_dimension_u.at(maximum_position) += 1;
            remain_dimension_v.at(-maximum_position) += 1;
         }

         for (const auto& [symmetry, this_remain] : remain_dimension_u) {
            if (this_remain == 0) {
               result_s.erase(symmetry);
            } else {
               result_s.at(symmetry).resize(this_remain);
            }
         }
      }

      const auto& tensor_u = put_v_right ? tensor_1 : tensor_2;
      const auto& tensor_v = put_v_right ? tensor_2 : tensor_1;
      // 始终在tensor_1处无符号reverse, 然后判断是否再在tensor_u和tensor_v中分别有无符号翻转
      // tensor_1 == tensor_u -> u nr // put_v_right
      // tensor_1 == tensor_v -> v nr v nr u yr -> u yr
      if constexpr (is_fermi) {
         reversed_set_u.insert(common_name_u);
      }
      // 这里会自动cut
      auto u = tensor_u.template edge_operator<true>(
            {{internal_name::SVD_V, common_name_u}},
            {{internal_name::SVD_U, free_names_and_edges_u}},
            reversed_set_u,
            {},
            result_name_u,
            false,
            {{{}, put_v_right ? std::set<Name>{} : std::set<Name>{common_name_u}, {}, {}}},
            {{internal_name::SVD_V, remain_dimension_u}});
      auto v = tensor_v.template edge_operator<true>(
            {{internal_name::SVD_U, common_name_v}},
            {{internal_name::SVD_V, free_names_and_edges_v}},
            reversed_set_v,
            {},
            result_name_v,
            false,
            {{{}, {}, {}, {}}},
            {{internal_name::SVD_U, remain_dimension_v}});
      return {
            std::move(u),
#ifdef TAT_USE_SINGULAR_MATRIX
            singular_to_tensor<ScalarType, Symmetry>(result_s),
#else
            {std::move(result_s)},
#endif
            std::move(v)};
   }
} // namespace TAT
#endif
