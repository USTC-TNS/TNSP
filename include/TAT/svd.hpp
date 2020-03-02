/**
 * \file svd.hpp
 *
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#define LAPACK_COMPLEX_CUSTOM
using lapack_complex_float = std::complex<float>;
using lapack_complex_double = std::complex<double>;
//#include "lapacke.h"

extern "C" {
void sgesvd_(
      const char* jobu,
      const char* jobvt,
      const int* m,
      const int* n,
      const float* a,
      const int* lda,
      float* s,
      float* u,
      const int* ldu,
      float* vt,
      const int* ldvt,
      float* work,
      const int* lwork,
      int* info);
void dgesvd_(
      const char* jobu,
      const char* jobvt,
      const int* m,
      const int* n,
      const double* a,
      const int* lda,
      double* s,
      double* u,
      const int* ldu,
      double* vt,
      const int* ldvt,
      double* work,
      const int* lwork,
      int* info);
void cgesvd_(
      const char* jobu,
      const char* jobvt,
      const int* m,
      const int* n,
      const std::complex<float>* a,
      const int* lda,
      float* s,
      std::complex<float>* u,
      const int* ldu,
      std::complex<float>* vt,
      const int* ldvt,
      std::complex<float>* work,
      const int* lwork,
      float* rwork,
      int* info);
void zgesvd_(
      const char* jobu,
      const char* jobvt,
      const int* m,
      const int* n,
      const std::complex<double>* a,
      const int* lda,
      double* s,
      std::complex<double>* u,
      const int* ldu,
      std::complex<double>* vt,
      const int* ldvt,
      std::complex<double>* work,
      const int* lwork,
      double* rwork,
      int* info);
}

namespace TAT {
   template<class ScalarType>
   void calculate_svd(
         const int& m,
         const int& n,
         const int& min,
         const ScalarType* a,
         ScalarType* u,
         real_base_t<ScalarType>* s,
         ScalarType* vt);

   template<>
   void calculate_svd<float>(
         const int& m,
         const int& n,
         const int& min,
         const float* a,
         float* u,
         float* s,
         float* vt) {
      int result;
      int max = m > n ? m : n;
      int lwork = 2 * (5 * min + max);
      auto work = vector<float>(lwork);
      sgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, &result);
      if (result != 0) {
         TAT_WARNING("Error in GESVD");
      }
   }
   template<>
   void calculate_svd<double>(
         const int& m,
         const int& n,
         const int& min,
         const double* a,
         double* u,
         double* s,
         double* vt) {
      int result;
      int max = m > n ? m : n;
      int lwork = 2 * (5 * min + max);
      auto work = vector<double>(lwork);
      dgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, &result);
      if (result != 0) {
         TAT_WARNING("Error in GESVD");
      }
   }
   template<>
   void calculate_svd<std::complex<float>>(
         const int& m,
         const int& n,
         const int& min,
         const std::complex<float>* a,
         std::complex<float>* u,
         float* s,
         std::complex<float>* vt) {
      int result;
      int max = m > n ? m : n;
      int lwork = 2 * (5 * min + max);
      auto work = vector<std::complex<float>>(lwork);
      auto rwork = vector<float>(5 * min);
      cgesvd_(
            "S",
            "S",
            &n,
            &m,
            a,
            &n,
            s,
            vt,
            &n,
            u,
            &min,
            work.data(),
            &lwork,
            rwork.data(),
            &result);
      if (result != 0) {
         TAT_WARNING("Error in GESVD");
      }
   }
   template<>
   void calculate_svd<std::complex<double>>(
         const int& m,
         const int& n,
         const int& min,
         const std::complex<double>* a,
         std::complex<double>* u,
         double* s,
         std::complex<double>* vt) {
      int result;
      int max = m > n ? m : n;
      int lwork = 2 * (5 * min + max);
      auto work = vector<std::complex<double>>(lwork);
      auto rwork = vector<double>(5 * min);
      zgesvd_(
            "S",
            "S",
            &n,
            &m,
            a,
            &n,
            s,
            vt,
            &n,
            u,
            &min,
            work.data(),
            &lwork,
            rwork.data(),
            &result);
      if (result != 0) {
         TAT_WARNING("Error in GESVD");
      }
   }

   template<class ScalarType, class Symmetry>
   typename Tensor<ScalarType, Symmetry>::svd_result Tensor<ScalarType, Symmetry>::svd(
         const std::set<Name>& u_free_names_set,
         Name u_common_name,
         Name v_common_name,
         Size cut) const {
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
      // merge
      auto u_free_names = vector<Name>();
      auto v_free_names = vector<Name>();
      auto reversed_set_u = std::set<Name>();
      auto reversed_set_v = std::set<Name>();
      auto reversed_set = std::set<Name>();
      auto res_u_names = vector<Name>();
      auto res_v_names = vector<Name>();
      auto u_free_names_and_edges = vector<std::tuple<Name, BoseEdge<Symmetry>>>();
      auto v_free_names_and_edges = vector<std::tuple<Name, BoseEdge<Symmetry>>>();
      res_v_names.push_back(v_common_name);
      for (auto i = 0; i < names.size(); i++) {
         const auto& n = names[i];
         if (u_free_names_set.find(n) != u_free_names_set.end()) {
            u_free_names.push_back(n);
            res_u_names.push_back(n);
            u_free_names_and_edges.push_back({n, {core->edges[i].map}});
            if constexpr (is_fermi) {
               if (core->edges[i].arrow) {
                  reversed_set_u.insert(n);
                  reversed_set.insert(n);
               }
            }
         } else {
            v_free_names.push_back(n);
            res_v_names.push_back(n);
            v_free_names_and_edges.push_back({n, {core->edges[i].map}});
            if constexpr (is_fermi) {
               if (core->edges[i].arrow) {
                  reversed_set_v.insert(n);
                  reversed_set.insert(n);
               }
            }
         }
      }
      res_u_names.push_back(u_common_name);
      const bool v_right = v_free_names.back() == names.back();
      auto tensor_merged = edge_operator(
            {},
            {},
            reversed_set,
            {{SVD1, u_free_names}, {SVD2, v_free_names}},
            v_right ? vector<Name>{SVD1, SVD2} : vector<Name>{SVD2, SVD1});
      // gesvd
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
            v_right ? vector<Name>{SVD1, SVD2} : vector<Name>{SVD2, SVD1},
            {std::move(tensor_merged.core->edges[0]), std::move(common_edge_1)}};
      auto tensor_2 = Tensor<ScalarType, Symmetry>{
            v_right ? vector<Name>{SVD1, SVD2} : vector<Name>{SVD2, SVD1},
            {std::move(common_edge_2), std::move(tensor_merged.core->edges[1])}};
      auto res_s = std::map<Symmetry, vector<real_base_t<ScalarType>>>();
      for (const auto& [sym, vec] : tensor_merged.core->blocks) {
         auto* u_data = tensor_1.core->blocks.at(sym).data();
         auto* v_data = tensor_2.core->blocks.at(sym).data();
         const auto* data = vec.data();
         const int m = tensor_1.core->edges[0].map.at(sym[0]);
         const int n = tensor_2.core->edges[1].map.at(sym[1]);
         const int k = m > n ? n : m;
         auto s = vector<real_base_t<ScalarType>>(k);
         auto* s_data = s.data();
         calculate_svd<ScalarType>(m, n, k, data, u_data, s_data, v_data);
         res_s[sym[!v_right]] = std::move(s);
      }
      const auto* u_tensor = &tensor_1;
      const auto* v_tensor = &tensor_2;
      if (!v_right) {
         u_tensor = &tensor_2;
         v_tensor = &tensor_1;
      }
      reversed_set_u.insert(u_common_name);
      auto u = u_tensor->edge_operator(
            {{SVD2, u_common_name}},
            {{SVD1, u_free_names_and_edges}},
            reversed_set_u,
            {},
            res_u_names);
      auto v = v_tensor->edge_operator(
            {{SVD1, v_common_name}},
            {{SVD2, v_free_names_and_edges}},
            reversed_set_v,
            {},
            res_v_names);
      return {std::move(u), std::move(res_s), std::move(v)};
   }
} // namespace TAT
#endif
