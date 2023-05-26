/**
 * \file svd.hpp
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
#ifndef TAT_SVD_HPP
#define TAT_SVD_HPP

#include <variant>

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/timer.hpp"

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

   inline timer svd_kernel_guard("svd_kernel");

   namespace detail {
      template<typename ScalarType>
      void calculate_svd_kernel(
            const int& m,
            const int& n,
            const int& min,
            const ScalarType* a,
            ScalarType* u,
            real_scalar<ScalarType>* s,
            ScalarType* vt);

      template<>
      inline void calculate_svd_kernel<float>(const int& m, const int& n, const int& min, const float* a, float* u, float* s, float* vt) {
         int result;
         const int lwork_query = -1;
         float float_lwork;
         sgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, &float_lwork, &lwork_query, &result);
         if (result != 0) {
            detail::what_if_lapack_error("Error in GESVD");
         }
         const int lwork = int(float_lwork);
         auto work = no_initialize::pmr::vector<float>(lwork);
         sgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, &result);
         if (result != 0) {
            detail::what_if_lapack_error("Error in GESVD");
         }
      }
      template<>
      inline void calculate_svd_kernel<double>(const int& m, const int& n, const int& min, const double* a, double* u, double* s, double* vt) {
         int result;
         const int lwork_query = -1;
         double float_lwork;
         dgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, &float_lwork, &lwork_query, &result);
         if (result != 0) {
            detail::what_if_lapack_error("Error in GESVD");
         }
         const int lwork = int(float_lwork);
         auto work = no_initialize::pmr::vector<double>(lwork);
         dgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, &result);
         if (result != 0) {
            detail::what_if_lapack_error("Error in GESVD");
         }
      }
      template<>
      inline void calculate_svd_kernel<std::complex<float>>(
            const int& m,
            const int& n,
            const int& min,
            const std::complex<float>* a,
            std::complex<float>* u,
            float* s,
            std::complex<float>* vt) {
         int result;
         auto rwork = no_initialize::pmr::vector<float>(5 * min);
         const int lwork_query = -1;
         std::complex<float> float_lwork;
         cgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, &float_lwork, &lwork_query, rwork.data(), &result);
         if (result != 0) {
            detail::what_if_lapack_error("Error in GESVD");
         }
         const int lwork = int(float_lwork.real());
         auto work = no_initialize::pmr::vector<std::complex<float>>(lwork);
         cgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, rwork.data(), &result);
         if (result != 0) {
            detail::what_if_lapack_error("Error in GESVD");
         }
      }
      template<>
      inline void calculate_svd_kernel<std::complex<double>>(
            const int& m,
            const int& n,
            const int& min,
            const std::complex<double>* a,
            std::complex<double>* u,
            double* s,
            std::complex<double>* vt) {
         int result;
         auto rwork = no_initialize::pmr::vector<double>(5 * min);
         const int lwork_query = -1;
         std::complex<double> float_lwork;
         zgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, &float_lwork, &lwork_query, rwork.data(), &result);
         if (result != 0) {
            detail::what_if_lapack_error("Error in GESVD");
         }
         const int lwork = int(float_lwork.real());
         auto work = no_initialize::pmr::vector<std::complex<double>>(lwork);
         zgesvd_("S", "S", &n, &m, a, &n, s, vt, &n, u, &min, work.data(), &lwork, rwork.data(), &result);
         if (result != 0) {
            detail::what_if_lapack_error("Error in GESVD");
         }
      }

      template<typename ScalarType>
      void calculate_svd(const int& m, const int& n, const int& min, const ScalarType* a, ScalarType* u, real_scalar<ScalarType>* s, ScalarType* vt) {
         auto kernel_guard = svd_kernel_guard();
         // after testing: m>n is better than m<n and false, true is obviously worse
         if (m > n) {
            auto new_a = no_initialize::pmr::vector<ScalarType>(n * m);
            auto old_u = no_initialize::pmr::vector<ScalarType>(n * min);
            auto old_vt = no_initialize::pmr::vector<ScalarType>(min * m);
            // new_a = a^T
            // u s vt = a
            // vt^T s u^T = a^T
            // old_u = vt^T
            // old_vt = u^T
            matrix_transpose<pmr::vector<Size>>(m, n, a, new_a.data()); // m*n -> n*m
            calculate_svd_kernel(n, m, min, new_a.data(), old_u.data(), s, old_vt.data());
            matrix_transpose<pmr::vector<Size>>(n, min, old_u.data(), vt); // n*min -> min*n
            matrix_transpose<pmr::vector<Size>>(min, m, old_vt.data(), u); // min*m -> m*min
         } else {
            calculate_svd_kernel(m, n, min, a, u, s, vt);
         }
      }

      template<typename ScalarType, typename Symmetry, typename Name, typename SingularValue>
      Tensor<ScalarType, Symmetry, Name>
      singular_to_tensor(const SingularValue& singular, const Name& singular_name_u, const Name& singular_name_v, const bool put_v_right) {
         // singular : [(sym, vector<Scalar>)]
         // (... U sym true) (-sym false S sym true) (-sym false V ...)
         auto segments_u = std::vector<std::pair<Symmetry, Size>>();
         auto segments_v = std::vector<std::pair<Symmetry, Size>>();
         segments_u.reserve(singular.size());
         segments_v.reserve(singular.size());
         for (const auto& [symmetry, values] : singular) {
            auto dimension = values.size();
            segments_u.emplace_back(-symmetry, dimension);
            segments_v.emplace_back(symmetry, dimension);
         }
         auto result = Tensor<ScalarType, Symmetry, Name>(
               {singular_name_u, singular_name_v},
               {{std::move(segments_u), false}, {std::move(segments_v), true}});
         result.zero();
         for (const auto& [symmetry, values] : singular) {
            const auto* data_source = values.data();
            auto* data_destination = result.blocks(pmr::vector<Symmetry>{-symmetry, symmetry}).data();
            auto dimension = values.size();
            auto dimension_plus_one = dimension + 1;
            bool parity = false;
            if constexpr (Symmetry::is_fermi_symmetry) {
               if (!put_v_right) {
                  parity = symmetry.parity();
               }
            }
            if (parity) {
               for (Size i = 0; i < dimension; i++) {
                  data_destination[i * dimension_plus_one] = -data_source[i];
               }
            } else {
               for (Size i = 0; i < dimension; i++) {
                  data_destination[i * dimension_plus_one] = data_source[i];
               }
            }
         }
         return result;
      }
   } // namespace detail

   inline timer svd_guard("svd");

   template<typename ScalarType, typename Symmetry, typename Name>
   typename Tensor<ScalarType, Symmetry, Name>::svd_result Tensor<ScalarType, Symmetry, Name>::svd(
         const std::unordered_set<Name>& free_names_u,
         const Name& common_name_u,
         const Name& common_name_v,
         const Name& singular_name_u,
         const Name& singular_name_v,
         Cut cut) const {
      auto pmr_guard = scope_resource(default_buffer_size);
      auto timer_guard = svd_guard();

      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;

      if constexpr (debug_mode) {
         // check free_name_set_u is valid
         for (const auto& name : free_names_u) {
            if (auto found = find_by_name(name); found == names().end()) {
               detail::error("Missing name in svd");
            }
         }
      }
      // now check done

      // merge to matrix
      // svd it with generating middle bond
      // split original edge

      // merge_plan -> all edge into free_name_u and free_name_v
      // and collect reversed edges

      // split and apply reversed edges, and use the free_name_x of merge
      // result name is also needed

      // merge
      auto free_name_list_u = pmr::vector<Name>();                                        // merge plan
      auto free_name_list_v = pmr::vector<Name>();                                        // merge plan
      auto reversed_names_input = pmr::unordered_set<Name>(unordered_parameter * rank()); // merge plan

      auto reversed_names_u = pmr::unordered_set<Name>(unordered_parameter * rank());                // split plan
      auto reversed_names_v = pmr::unordered_set<Name>(unordered_parameter * rank());                // split plan
      auto result_names_u = std::vector<Name>();                                                     // split plan
      auto result_names_v = std::vector<Name>();                                                     // split plan
      auto free_names_and_edges_u = pmr::vector<std::pair<Name, edge_segments_t<Symmetry, true>>>(); // split plan
      auto free_names_and_edges_v = pmr::vector<std::pair<Name, edge_segments_t<Symmetry, true>>>(); // split plan

      free_name_list_u.reserve(rank());
      free_name_list_v.reserve(rank());
      result_names_u.reserve(rank() + 1);
      result_names_v.reserve(rank() + 1);
      free_names_and_edges_u.reserve(rank());
      free_names_and_edges_v.reserve(rank());

      const bool put_v_right = names().empty() || free_names_u.find(names().back()) == free_names_u.end();
      // The last names not in free_names_u -> last names in free_names_v -> put v right

      if (put_v_right) {
         result_names_v.push_back(common_name_v);
      } else {
         result_names_u.push_back(common_name_u);
      }
      for (Rank i = 0; i < rank(); i++) {
         const auto& n = names(i);
         if (free_names_u.find(n) != free_names_u.end()) {
            // U side
            free_name_list_u.push_back(n);
            result_names_u.push_back(n);
            free_names_and_edges_u.push_back({n, {edges(i).segments()}});
            if constexpr (is_fermi) {
               if (edges(i).arrow()) {
                  reversed_names_u.insert(n);
                  reversed_names_input.insert(n);
               }
            }
         } else {
            // V side
            free_name_list_v.push_back(n);
            result_names_v.push_back(n);
            free_names_and_edges_v.push_back({n, {edges(i).segments()}});
            if constexpr (is_fermi) {
               if (edges(i).arrow()) {
                  reversed_names_v.insert(n);
                  reversed_names_input.insert(n);
               }
            }
         }
      }
      if (put_v_right) {
         result_names_u.push_back(common_name_u);
      } else {
         result_names_v.push_back(common_name_v);
      }

      auto tensor_merged = edge_operator_implement(
            {},
            reversed_names_input,
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::SVD_U, std::move(free_name_list_u)},
                  {InternalName<Name>::SVD_V, std::move(free_name_list_v)}},
            put_v_right ? std::vector<Name>{InternalName<Name>::SVD_U, InternalName<Name>::SVD_V} :
                          std::vector<Name>{InternalName<Name>::SVD_V, InternalName<Name>::SVD_U},
            false,
            {},
            {},
            {},
            {},
            {});

      // tensor -> SVD_U -O- SVD_V    when put_v_right = true
      //        or SVD_V -O- SVD_U    when put_v_right = false
      // put_v_right = true
      //     SVD_U -O- common_1 . common_2 -O- SVD_V
      // put_v_right = false
      //     SVD_V -O- common_1 . common_1 -O- SVD_U

      const auto& edge_0_input = tensor_merged.edges(0);
      const auto& edge_1_input = tensor_merged.edges(1);
      // prepare result tensor
      auto common_edge_segments_1 = std::vector<std::pair<Symmetry, Size>>();
      auto common_edge_segments_2 = std::vector<std::pair<Symmetry, Size>>();
      common_edge_segments_1.reserve(edge_0_input.segments_size());
      common_edge_segments_2.reserve(edge_0_input.segments_size());
      // arrow all false currently, arrow check is processed at the last
      for (const auto& [symmetry_0, dimension_0] : edge_0_input.segments()) {
         auto symmetry_1 = -symmetry_0;
         if (auto found = edge_1_input.find_by_symmetry(symmetry_1); found != edge_1_input.segments().end()) {
            // This block does exist
            auto m = dimension_0;
            auto n = found->second;
            auto k = m > n ? n : m;
            common_edge_segments_1.emplace_back(symmetry_1, k);
            common_edge_segments_2.emplace_back(symmetry_0, k);
         }
      }
      auto tensor_1 = Tensor<ScalarType, Symmetry, Name>{
            put_v_right ? std::vector<Name>{InternalName<Name>::SVD_U, common_name_u} : std::vector<Name>{InternalName<Name>::SVD_V, common_name_v},
            {tensor_merged.edges(0), std::move(common_edge_segments_1)}};
      auto tensor_2 = Tensor<ScalarType, Symmetry, Name>{
            put_v_right ? std::vector<Name>{common_name_v, InternalName<Name>::SVD_V} : std::vector<Name>{common_name_u, InternalName<Name>::SVD_U},
            {std::move(common_edge_segments_2), tensor_merged.edges(1)}};
      auto result_s = pmr::list<std::pair<Symmetry, pmr::vector<real_scalar<ScalarType>>>>(); // use list for easy delete
      // const auto& edge_common_1 = tensor_1.edges(1);
      const auto& edge_common_2 = tensor_2.edges(0);

      // call lapack
      for (const auto& [symmetry, _] : edge_0_input.segments()) {
         Size position_0_input = edge_0_input.find_by_symmetry(symmetry) - edge_0_input.segments().begin();
         Size position_1_input = edge_1_input.find_by_symmetry(-symmetry) - edge_1_input.segments().begin();
         if (position_1_input == edge_1_input.segments_size()) {
            continue;
         }
         Size position_common = edge_common_2.find_by_symmetry(symmetry) - edge_common_2.segments().begin();
         const int m = edge_0_input.segments(position_0_input).second;
         const int n = edge_1_input.segments(position_1_input).second;
         const int k = edge_common_2.segments(position_common).second;

         auto* data_1 = tensor_1.blocks(pmr::vector<Size>{position_0_input, position_common}).data();
         auto* data_2 = tensor_2.blocks(pmr::vector<Size>{position_common, position_1_input}).data();
         const auto* data = tensor_merged.blocks(pmr::vector<Size>{position_0_input, position_1_input}).data();

         auto s = pmr::vector<real_scalar<ScalarType>>(k);
         auto* data_s = s.data();

         if (m * n != 0) {
            detail::calculate_svd<ScalarType>(m, n, k, data, data_1, data_s, data_2);
         }
         result_s.emplace_back(put_v_right ? -symmetry : symmetry, std::move(s)); // symmetry of s is always the same to common edge of tensor U
      }

      // analyze how to cut
      Size total_dimension = 0;
      for (const auto& [symmetry, vector_s] : result_s) {
         total_dimension += vector_s.size();
      }
      auto remain_dimension_u = pmr::unordered_map<Symmetry, Size>(unordered_parameter * result_s.size());
      auto remain_dimension_v = pmr::unordered_map<Symmetry, Size>(unordered_parameter * result_s.size());

      real_scalar<ScalarType> total_maximum_singular = 0;
      for (const auto& [symmetry, vector_s] : result_s) {
         for (const auto& this_singular : vector_s) {
            if (this_singular > total_maximum_singular) {
               total_maximum_singular = this_singular;
            }
         }
      }
      const real_scalar<ScalarType> cut_threshold = cut.relative_cut * total_maximum_singular;
      const Size remain_cut = cut.remain_cut < total_dimension ? cut.remain_cut : total_dimension;

      for (const auto& [symmetry, vector_s] : result_s) {
         remain_dimension_u[symmetry] = 0;
         remain_dimension_v[-symmetry] = 0;
      }
      for (Size i = 0; i < remain_cut; i++) {
         Symmetry maximum_symmetry;
         real_scalar<ScalarType> maximum_singular = 0;
         for (const auto& [symmetry, vector_s] : result_s) {
            if (auto& this_remain = remain_dimension_u.at(symmetry); this_remain != vector_s.size()) {
               if (auto this_singular = vector_s[this_remain]; this_singular > maximum_singular) {
                  maximum_singular = this_singular;
                  maximum_symmetry = symmetry;
               }
            }
         }
         if (maximum_singular > cut_threshold) {
            // If the singular is too small, do not remain it.
            remain_dimension_u.at(maximum_symmetry) += 1;
            remain_dimension_v.at(-maximum_symmetry) += 1;
         } else {
            break;
         }
      }
      // delete element of tensor S
      for (auto it = result_s.begin(); it != result_s.end();) {
         const auto& symmetry = it->first;
         const auto& this_remain = remain_dimension_u.at(symmetry);
         if (this_remain == 0) {
            it = result_s.erase(it);
         } else {
            it->second.resize(this_remain);
            ++it;
         }
      }
      // cut analyze done

      const auto& tensor_u = put_v_right ? tensor_1 : tensor_2;
      const auto& tensor_v = put_v_right ? tensor_2 : tensor_1;
      // tensor 1 common edge should be true
      // tensor 2 common edge should be false
      // S tensor 1 side should be false
      // S tensor 2 side should be true

      // if put_v_right, it should be
      // (... U sym true) (-sym false S sym true) (-sym false V ...)
      // if !put_v_right, it should be
      // (... V -sym true) (sym false S -sym true) (sym false U ...)
      // which is same to
      // (... V -sym false) (sym true S -sym false) (sym true U ...)
      // aka (sym true U ...) (sym true v_S_u -sym false) (... V -sym false)
      // so always put reverse tensor_u common name
      // if put_v_right: (-sym false u_S_v sym true)
      // if !put_v_right: (sym true v_S_u -sym false) -> (-sym false u_S_v sym true) with a transpose sign
      // -> always set v_edge in s to sym true
      if constexpr (is_fermi) {
         reversed_names_u.insert(common_name_u);
      }
      // cut happened here
      auto u = tensor_u.edge_operator_implement(
            pmr::map<Name, pmr::vector<std::pair<Name, edge_segments_t<Symmetry, true>>>>{
                  {InternalName<Name>::SVD_U, std::move(free_names_and_edges_u)}},
            reversed_names_u,
            {},
            std::move(result_names_u),
            false,
            {},
            {},
            {},
            {},
            pmr::map<Name, pmr::unordered_map<Symmetry, Size>>{{common_name_u, std::move(remain_dimension_u)}});
      auto v = tensor_v.edge_operator_implement(
            pmr::map<Name, pmr::vector<std::pair<Name, edge_segments_t<Symmetry, true>>>>{
                  {InternalName<Name>::SVD_V, std::move(free_names_and_edges_v)}},
            reversed_names_v,
            {},
            std::move(result_names_v),
            false,
            {},
            {},
            {},
            {},
            pmr::map<Name, pmr::unordered_map<Symmetry, Size>>{{common_name_v, std::move(remain_dimension_v)}});
      return {
            std::move(u),
            // it should be noticed that, singular value content of fermi tensor is not always positive,
            // since it is not even valid to talk about sign of tensor content for fermi tensor
            detail::singular_to_tensor<ScalarType, Symmetry, Name>(result_s, singular_name_u, singular_name_v, put_v_right),
            std::move(v)};
   }
} // namespace TAT
#endif
