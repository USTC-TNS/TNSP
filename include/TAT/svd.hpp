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
      const int* l_work,
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
      const int* l_work,
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
      const int* l_work,
      float* r_work,
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
      const int* l_work,
      double* r_work,
      int* info);
}

namespace TAT {
   template<class ScalarType>
   void calculate_svd(const int& m, const int& n, const int& min, const ScalarType* a, ScalarType* u, real_base_t<ScalarType>* s, ScalarType* vt);

   template<>
   inline void calculate_svd<float>(const int& m, const int& n, const int& min, const float* a, float* u, float* s, float* vt) {
      int result;
      const int max = m > n ? m : n;
      const int l_work = 2 * (5 * min + max);
      auto work = std::vector<float>(l_work);
      sgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &l_work, &result);
      if (result != 0) {
         warning_or_error("Error in GESVD");
      }
   }
   template<>
   inline void calculate_svd<double>(const int& m, const int& n, const int& min, const double* a, double* u, double* s, double* vt) {
      int result;
      const int max = m > n ? m : n;
      const int l_work = 2 * (5 * min + max);
      auto work = std::vector<double>(l_work);
      dgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &l_work, &result);
      if (result != 0) {
         warning_or_error("Error in GESVD");
      }
   }
   template<>
   inline void calculate_svd<std::complex<float>>(
         const int& m,
         const int& n,
         const int& min,
         const std::complex<float>* a,
         std::complex<float>* u,
         float* s,
         std::complex<float>* vt) {
      int result;
      const int max = m > n ? m : n;
      const int l_work = 2 * (5 * min + max);
      auto work = std::vector<std::complex<float>>(l_work);
      auto r_work = std::vector<float>(5 * min);
      cgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &l_work, r_work.data(), &result);
      if (result != 0) {
         warning_or_error("Error in GESVD");
      }
   }
   template<>
   inline void calculate_svd<std::complex<double>>(
         const int& m,
         const int& n,
         const int& min,
         const std::complex<double>* a,
         std::complex<double>* u,
         double* s,
         std::complex<double>* vt) {
      int result;
      const int max = m > n ? m : n;
      const int l_work = 2 * (5 * min + max);
      auto work = std::vector<std::complex<double>>(l_work);
      auto r_work = std::vector<double>(5 * min);
      zgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &l_work, r_work.data(), &result);
      if (result != 0) {
         warning_or_error("Error in GESVD");
      }
   }

   template<class ScalarType, class Symmetry>
   typename Tensor<ScalarType, Symmetry>::svd_result
   Tensor<ScalarType, Symmetry>::svd(const std::set<Name>& free_name_set_u, Name common_name_u, Name common_name_v, Size cut) const {
      // free_name_set_u不需要做特殊处理即可自动处理不准确的边名
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
      // merge
      auto free_name_u = std::vector<Name>();
      auto free_name_v = std::vector<Name>();
      auto reversed_set_u = std::set<Name>();
      auto reversed_set_v = std::set<Name>();
      auto reversed_set_origin = std::set<Name>();
      auto result_name_u = std::vector<Name>();
      auto result_name_v = std::vector<Name>();
      auto free_names_and_edges_u = std::vector<std::tuple<Name, BoseEdge<Symmetry>>>();
      auto free_names_and_edges_v = std::vector<std::tuple<Name, BoseEdge<Symmetry>>>();
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
            {{SVD1, free_name_u}, {SVD2, free_name_v}},
            put_v_right ? std::vector<Name>{SVD1, SVD2} : std::vector<Name>{SVD2, SVD1});
      // call GESVD
      auto common_edge_1 = Edge<Symmetry>();
      auto common_edge_2 = Edge<Symmetry>();
      for (const auto& [sym, _] : tensor_merged.core->blocks) {
         auto m = tensor_merged.core->edges[0].map.at(sym[0]);
         auto n = tensor_merged.core->edges[1].map.at(sym[1]);
         auto k = m > n ? n : m;
         common_edge_1.map[sym[1]] = k;
         common_edge_2.map[sym[0]] = k;
      }
      auto tensor_1 = Tensor<ScalarType, Symmetry>{
            put_v_right ? std::vector<Name>{SVD1, SVD2} : std::vector<Name>{SVD2, SVD1},
            {std::move(tensor_merged.core->edges[0]), std::move(common_edge_1)}};
      auto tensor_2 = Tensor<ScalarType, Symmetry>{
            put_v_right ? std::vector<Name>{SVD1, SVD2} : std::vector<Name>{SVD2, SVD1},
            {std::move(common_edge_2), std::move(tensor_merged.core->edges[1])}};
      auto result_s = std::map<Symmetry, vector<real_base_t<ScalarType>>>();
      for (const auto& [symmetries, block] : tensor_merged.core->blocks) {
         auto* data_u = tensor_1.core->blocks.at(symmetries).data();
         auto* data_v = tensor_2.core->blocks.at(symmetries).data();
         const auto* data = block.data();
         const int m = tensor_1.core->edges[0].map.at(symmetries[0]);
         const int n = tensor_2.core->edges[1].map.at(symmetries[1]);
         const int k = m > n ? n : m;
         auto s = vector<real_base_t<ScalarType>>(k);
         auto* s_data = s.data();
         if (m * n != 0) {
            calculate_svd<ScalarType>(m, n, k, data, data_u, s_data, data_v);
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
      reversed_set_u.insert(common_name_u);
      auto u = tensor_u.edge_operator(
            {{SVD2, common_name_u}},
            {{SVD1, free_names_and_edges_u}},
            reversed_set_u,
            {},
            result_name_u,
            false,
            {{{}, {}, {}, {}}},
            {{SVD2, remain_dimension_u}});
      auto v = tensor_v.edge_operator(
            {{SVD1, common_name_v}},
            {{SVD2, free_names_and_edges_v}},
            reversed_set_v,
            {},
            result_name_v,
            false,
            {{{}, {}, {}, {}}},
            {{SVD1, remain_dimension_v}});
      return {std::move(u), {std::move(result_s)}, std::move(v)};
   }
} // namespace TAT
#endif
