/**
 * \file svd.hpp
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
#ifndef TAT_SVD_HPP
#define TAT_SVD_HPP

#include <variant>

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/timer.hpp"
#include "transpose.hpp"

extern "C" {
   void sgesvd_(
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
   void dgesvd_(
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
   void cgesvd_(
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
   void zgesvd_(
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
      template<typename ScalarType, typename Symmetry, typename Name, typename SingularValue>
      [[nodiscard]] Tensor<ScalarType, Symmetry, Name>
      singular_to_tensor(const SingularValue& singular, const Name& singular_name_u, const Name& singular_name_v, const bool need_transpose) {
         // singular : [(sym, vector<Scalar>)]
         // (... U sym true) (-sym false S sym true) (-sym false V ...)
         auto symmetries = std::vector<Edge<Symmetry>>(2);
         for (const auto& [symmetry, values] : singular) {
            auto dimension = values.size();
            symmetries[0].segment.emplace_back(-symmetry, dimension);
            symmetries[1].segment.emplace_back(symmetry, dimension);
         }
         if constexpr (Symmetry::is_fermi_symmetry) {
            symmetries[0].arrow = false;
            symmetries[1].arrow = true;
         }
         auto result = Tensor<ScalarType, Symmetry, Name>({singular_name_u, singular_name_v}, std::move(symmetries));
         for (const auto& [symmetry, data_source] : singular) {
            auto& data_destination = result.blocks(std::array<Symmetry, 2>{-symmetry, symmetry});
            auto dimension = data_source.size();
            auto dimension_plus_one = dimension + 1;
            std::fill(data_destination.begin(), data_destination.end(), 0);
            bool parity = false;
            if constexpr (Symmetry::is_fermi_symmetry) {
               if (need_transpose) {
                  parity = symmetry.get_parity();
               }
            }
            if (parity) {
               for (Size i = 0; i < data_source.size(); i++) {
                  data_destination[i * dimension_plus_one] = -data_source[i];
               }
            } else {
               for (Size i = 0; i < data_source.size(); i++) {
                  data_destination[i * dimension_plus_one] = data_source[i];
               }
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
            real_scalar<ScalarType>* s,
            ScalarType* vt);

      template<typename ScalarType>
      void calculate_svd(
            const int& m,
            const int& n,
            const int& min,
            const int& max,
            const ScalarType* a,
            ScalarType* u,
            real_scalar<ScalarType>* s,
            ScalarType* vt) {
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
      inline void
      calculate_svd_kernel<double>(const int& m, const int& n, const int& min, const int& max, const double* a, double* u, double* s, double* vt) {
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
            const int& max,
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
            const int& max,
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
   } // namespace detail

   inline timer svd_guard("svd");

   template<typename ScalarType, typename Symmetry, typename Name>
   typename Tensor<ScalarType, Symmetry, Name>::svd_result Tensor<ScalarType, Symmetry, Name>::svd(
         const std::set<Name>& free_name_set_u,
         const Name& common_name_u,
         const Name& common_name_v,
         const Name& singular_name_u,
         const Name& singular_name_v,
         Cut cut) const {
      auto pmr_guard = scope_resource(default_buffer_size);
      auto timer_guard = svd_guard();

      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
      const auto rank = get_rank();

      if constexpr (debug_mode) {
         // check free_name_set_u is valid
         for (const auto& name : free_name_set_u) {
            if (auto found = find_rank_from_name(name); found == names.end()) {
               detail::error("Missing name in svd");
            }
         }
      }
      // now check done

      // merge to matrix
      // svd generate middle bond
      // split original edge

      // merge_plan -> free_name_u and free_name_v
      // and collect reversed edges
      // result name is also needed

      // svd need to pay attension to new generated edge symmetries

      // split just apply reversed edges, and use the free_name_x of merge
      // result name is also needed

      // merge
      auto free_name_u = pmr::vector<Name>();      // merge plan
      auto free_name_v = pmr::vector<Name>();      // merge plan
      auto reversed_set_origin = pmr::set<Name>(); // merge plan

      auto reversed_set_u = pmr::set<Name>();                                                       // split plan
      auto reversed_set_v = pmr::set<Name>();                                                       // split plan
      auto result_name_u = std::vector<Name>();                                                     // split plan
      auto result_name_v = std::vector<Name>();                                                     // split plan
      auto free_names_and_edges_u = pmr::vector<std::pair<Name, edge_segment_t<Symmetry, true>>>(); // split plan
      auto free_names_and_edges_v = pmr::vector<std::pair<Name, edge_segment_t<Symmetry, true>>>(); // split plan

      free_name_u.reserve(rank);
      free_name_v.reserve(rank);
      result_name_u.reserve(rank + 1);
      result_name_v.reserve(rank + 1);
      free_names_and_edges_u.reserve(rank);
      free_names_and_edges_v.reserve(rank);

      result_name_v.push_back(common_name_v);
      for (Rank i = 0; i < get_rank(); i++) {
         const auto& n = names[i];
         if (free_name_set_u.find(n) != free_name_set_u.end()) {
            // U side
            free_name_u.push_back(n);
            result_name_u.push_back(n);
            free_names_and_edges_u.push_back({n, {edges(i).segment}});
            if constexpr (is_fermi) {
               if (edges(i).arrow) {
                  reversed_set_u.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         } else {
            // V side
            free_name_v.push_back(n);
            result_name_v.push_back(n);
            free_names_and_edges_v.push_back({n, {edges(i).segment}});
            if constexpr (is_fermi) {
               if (edges(i).arrow) {
                  reversed_set_v.insert(n);
                  reversed_set_origin.insert(n);
               }
            }
         }
      }
      result_name_u.push_back(common_name_u);

      const bool put_v_right = free_name_v.empty() || free_name_v.back() == names.back();
      auto tensor_merged = edge_operator_implement(
            empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
            reversed_set_origin,
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::SVD_U, std::move(free_name_u)},
                  {InternalName<Name>::SVD_V, std::move(free_name_v)}},
            put_v_right ? std::vector<Name>{InternalName<Name>::SVD_U, InternalName<Name>::SVD_V} :
                          std::vector<Name>{InternalName<Name>::SVD_V, InternalName<Name>::SVD_U},
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());

      // tensor -> SVD_U -O- SVD_V

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
            put_v_right ? std::vector<Name>{InternalName<Name>::SVD_U, common_name_u} : std::vector<Name>{InternalName<Name>::SVD_V, common_name_v},
            {std::move(tensor_merged.edges(0)), std::move(common_edge_1)}};
      auto tensor_2 = Tensor<ScalarType, Symmetry, Name>{
            put_v_right ? std::vector<Name>{common_name_v, InternalName<Name>::SVD_V} : std::vector<Name>{common_name_u, InternalName<Name>::SVD_U},
            {std::move(common_edge_2), std::move(tensor_merged.edges(1))}};
      auto result_s = pmr::list<std::pair<Symmetry, pmr::vector<real_scalar<ScalarType>>>>();

      // call lapack
      for (const auto& [symmetries, block] : tensor_merged.core->blocks) {
         auto* data_u = tensor_1.blocks(symmetries).data();
         auto* data_v = tensor_2.blocks(symmetries).data();
         const auto* data = block.data();
         const int m = tensor_1.edges(0).get_dimension_from_symmetry(symmetries[0]);
         const int n = tensor_2.edges(1).get_dimension_from_symmetry(symmetries[1]);
         const int k = m > n ? n : m;
         const int max = m > n ? m : n;
         auto s = pmr::vector<real_scalar<ScalarType>>(k);
         auto* s_data = s.data();
         if (m * n != 0) {
            detail::calculate_svd<ScalarType>(m, n, k, max, data, data_u, s_data, data_v);
         }
         result_s.emplace_back(symmetries[put_v_right ? 1 : 0], std::move(s)); // sym is always the same to common edge of tensor U
      }

      // analyze how to cut
      Size total_dimension = 0;
      for (const auto& [symmetry, vector_s] : result_s) {
         total_dimension += vector_s.size();
      }
      auto remain_dimension_u = pmr::map<Symmetry, Size>();
      auto remain_dimension_v = pmr::map<Symmetry, Size>();
      if (auto cut_value = std::get_if<RemainCut>(&cut)) {
         Size cut_i = cut_value->value;
         if (cut_i < total_dimension) {
            for (const auto& [symmetry, vector_s] : result_s) {
               remain_dimension_u[symmetry] = 0;
               remain_dimension_v[-symmetry] = 0;
            }
            for (Size i = 0; i < cut_i; i++) {
               Symmetry maximum_position;
               real_scalar<ScalarType> maximum_singular = 0;
               for (const auto& [symmetry, vector_s] : result_s) {
                  if (auto& this_remain = remain_dimension_u.at(symmetry); this_remain != vector_s.size()) {
                     if (auto this_singular = vector_s[this_remain]; this_singular > maximum_singular) {
                        maximum_singular = this_singular;
                        maximum_position = symmetry;
                     }
                  }
               }
               if (maximum_singular != 0) {
                  // sometimes non zero singular number is less than cut_i
                  // this if check the validity of maximum_position
                  remain_dimension_u.at(maximum_position) += 1;
                  remain_dimension_v.at(-maximum_position) += 1;
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
         }
      } else if (auto cut_value = std::get_if<RelativeCut>(&cut)) {
         real_scalar<ScalarType> maximum_singular = 0;
         for (const auto& [symmetry, vector_s] : result_s) {
            for (const auto& this_singular : vector_s) {
               if (this_singular > maximum_singular) {
                  maximum_singular = this_singular;
               }
            }
         }
         real_scalar<ScalarType> threshold = cut_value->value * maximum_singular;
         for (const auto& [symmetry, vector_s] : result_s) {
            auto current_size = vector_s.size();
            for (auto i = 0; i < current_size; i++) {
               if (vector_s[i] < threshold) {
                  remain_dimension_u[symmetry] = i;
                  remain_dimension_v[-symmetry] = i;
                  break;
               }
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
      }
      // cut analyze done

      const auto& tensor_u = put_v_right ? tensor_1 : tensor_2;
      const auto& tensor_v = put_v_right ? tensor_2 : tensor_1;
      // tensor 1 common edge should be true
      // tensor 2 common edge should be false
      // S tensor 1 side should be false
      // S tensor 2 side should be true
      // so we need to reverse tensor_1 common edge without applying sign

      // aims:
      // tensor_1 == tensor_u -> u not_apply_reverse
      // tensor_1 == tensor_v -> v not_apply_reverse

      // what did for S
      // while sym of S is always the same to common edge of tensor U
      // tensor_1 == tensor_u -> sym of S correct -> nothing
      // tensor_1 == tensor_v -> sym of S incorrect -> need to reverse two EPR pair -> need to do more u apply_reverse & v apply_reverse

      // summary
      // tensor_1 == tensor_u -> u not_apply_reverse
      // tensor_1 == tensor_v -> u not_apply_reverse

      // however, tensor S transpose generate a sign, this should be pay attension to, when tensor_1 == tensor_v, namely put_v_right == false

      if constexpr (is_fermi) {
         reversed_set_u.insert(common_name_u);
      }
      // cut happened here
      auto u = tensor_u.edge_operator_implement(
            pmr::map<Name, pmr::vector<std::pair<Name, edge_segment_t<Symmetry, true>>>>{
                  {InternalName<Name>::SVD_U, std::move(free_names_and_edges_u)}},
            reversed_set_u,
            empty_list<std::pair<Name, empty_list<Name>>>(),
            std::move(result_name_u),
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            pmr::map<Name, pmr::map<Symmetry, Size>>{{common_name_u, std::move(remain_dimension_u)}});
      auto v = tensor_v.edge_operator_implement(
            pmr::map<Name, pmr::vector<std::pair<Name, edge_segment_t<Symmetry, true>>>>{
                  {InternalName<Name>::SVD_V, std::move(free_names_and_edges_v)}},
            reversed_set_v,
            empty_list<std::pair<Name, empty_list<Name>>>(),
            std::move(result_name_v),
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            pmr::map<Name, pmr::map<Symmetry, Size>>{{common_name_v, std::move(remain_dimension_v)}});
      // (... U sym true) (-sym false S sym true) (-sym false V ...)
      return {
            std::move(u),
            // it should be noticed that, singular value content of fermi tensor is not always positive,
            // since it is not even valid to talk about sign of tensor content for fermi tensor
            detail::singular_to_tensor<ScalarType, Symmetry, Name>(result_s, singular_name_u, singular_name_v, put_v_right == false),
            std::move(v)};
   }
} // namespace TAT
#endif
