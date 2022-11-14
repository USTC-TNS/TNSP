/**
 * \file contract.hpp
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
#ifndef TAT_CONTRACT_HPP
#define TAT_CONTRACT_HPP

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/timer.hpp"

extern "C" {
   int sgemm_(
         const char* transpose_a,
         const char* transpose_b,
         const int* m,
         const int* n,
         const int* k,
         const float* alpha,
         const float* a,
         const int* lda,
         const float* b,
         const int* ldb,
         const float* beta,
         float* c,
         const int* ldc);
   int dgemm_(
         const char* transpose_a,
         const char* transpose_b,
         const int* m,
         const int* n,
         const int* k,
         const double* alpha,
         const double* a,
         const int* lda,
         const double* b,
         const int* ldb,
         const double* beta,
         double* c,
         const int* ldc);
   int cgemm_(
         const char* transpose_a,
         const char* transpose_b,
         const int* m,
         const int* n,
         const int* k,
         const std::complex<float>* alpha,
         const std::complex<float>* a,
         const int* lda,
         const std::complex<float>* b,
         const int* ldb,
         const std::complex<float>* beta,
         std::complex<float>* c,
         const int* ldc);
   int zgemm_(
         const char* transpose_a,
         const char* transpose_b,
         const int* m,
         const int* n,
         const int* k,
         const std::complex<double>* alpha,
         const std::complex<double>* a,
         const int* lda,
         const std::complex<double>* b,
         const int* ldb,
         const std::complex<double>* beta,
         std::complex<double>* c,
         const int* ldc);

#ifdef TAT_USE_MKL_GEMM_BATCH
   int sgemm_batch_(
         const char* transpose_a,
         const char* transpose_b,
         const int* m,
         const int* n,
         const int* k,
         const float* alpha,
         const float** a,
         const int* lda,
         const float** b,
         const int* ldb,
         const float* beta,
         float** c,
         const int* ldc,
         const int* group_count,
         const int* group_size);
   int dgemm_batch_(
         const char* transpose_a,
         const char* transpose_b,
         const int* m,
         const int* n,
         const int* k,
         const double* alpha,
         const double** a,
         const int* lda,
         const double** b,
         const int* ldb,
         const double* beta,
         double** c,
         const int* ldc,
         const int* group_count,
         const int* group_size);
   int cgemm_batch_(
         const char* transpose_a,
         const char* transpose_b,
         const int* m,
         const int* n,
         const int* k,
         const std::complex<float>* alpha,
         const std::complex<float>** a,
         const int* lda,
         const std::complex<float>** b,
         const int* ldb,
         const std::complex<float>* beta,
         std::complex<float>** c,
         const int* ldc,
         const int* group_count,
         const int* group_size);
   int zgemm_batch_(
         const char* transpose_a,
         const char* transpose_b,
         const int* m,
         const int* n,
         const int* k,
         const std::complex<double>* alpha,
         const std::complex<double>** a,
         const int* lda,
         const std::complex<double>** b,
         const int* ldb,
         const std::complex<double>* beta,
         std::complex<double>** c,
         const int* ldc,
         const int* group_count,
         const int* group_size);
#endif
}

namespace TAT {
   namespace detail {
      template<typename ScalarType>
      constexpr int (*gemm)(
            const char* transpose_a,
            const char* transpose_b,
            const int* m,
            const int* n,
            const int* k,
            const ScalarType* alpha,
            const ScalarType* a,
            const int* lda,
            const ScalarType* b,
            const int* ldb,
            const ScalarType* beta,
            ScalarType* c,
            const int* ldc) = nullptr;

      template<>
      inline auto gemm<float> = sgemm_;
      template<>
      inline auto gemm<double> = dgemm_;
      template<>
      inline auto gemm<std::complex<float>> = cgemm_;
      template<>
      inline auto gemm<std::complex<double>> = zgemm_;

      template<typename ScalarType>
      constexpr int (*mkl_gemm_batch)(
            const char* transpose_a,
            const char* transpose_b,
            const int* m,
            const int* n,
            const int* k,
            const ScalarType* alpha,
            const ScalarType** a,
            const int* lda,
            const ScalarType** b,
            const int* ldb,
            const ScalarType* beta,
            ScalarType** c,
            const int* ldc,
            const int* group_count,
            const int* group_size) = nullptr;

#ifdef TAT_USE_MKL_GEMM_BATCH
      template<>
      inline auto mkl_gemm_batch<float> = sgemm_batch_;
      template<>
      inline auto mkl_gemm_batch<double> = dgemm_batch_;
      template<>
      inline auto mkl_gemm_batch<std::complex<float>> = cgemm_batch_;
      template<>
      inline auto mkl_gemm_batch<std::complex<double>> = zgemm_batch_;
#endif
   } // namespace detail

   inline timer contract_kernel_guard("contract_kernel");

   namespace detail {
      template<typename ScalarType, bool same_shape>
      void gemm_batch(
            const char* transpose_a,
            const char* transpose_b,
            const int* m,
            const int* n,
            const int* k,
            const ScalarType* alpha,
            const ScalarType** a,
            const int* lda,
            const ScalarType** b,
            const int* ldb,
            const ScalarType* beta,
            ScalarType** c,
            const int* ldc,
            const int& batch_size) {
         auto kernel_guard = contract_kernel_guard();
         if (batch_size == 1) {
            gemm<ScalarType>(transpose_a, transpose_b, m, n, k, alpha, a[0], lda, b[0], ldb, beta, c[0], ldc);
         } else {
#ifdef TAT_USE_MKL_GEMM_BATCH
            if constexpr (same_shape) {
               int group_count = 1;
               mkl_gemm_batch<ScalarType>(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, &group_count, &batch_size);
            } else {
               pmr::vector<int> group_size(batch_size, 1);
               mkl_gemm_batch<ScalarType>(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, &batch_size, group_size.data());
            }
#else
            if constexpr (same_shape) {
               for (auto i = 0; i < batch_size; i++) {
                  gemm<ScalarType>(transpose_a, transpose_b, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i], ldc);
               }
            } else {
               for (auto i = 0; i < batch_size; i++) {
                  gemm<ScalarType>(
                        &transpose_a[i],
                        &transpose_b[i],
                        &m[i],
                        &n[i],
                        &k[i],
                        &alpha[i],
                        a[i],
                        &lda[i],
                        b[i],
                        &ldb[i],
                        &beta[i],
                        c[i],
                        &ldc[i]);
               }
            }
#endif
         }
      }

      template<typename Name, typename SetNameName>
      auto generate_contract_map(const SetNameName& contract_pairs) {
         auto size = contract_pairs.size();
         auto contract_names_1_2 = pmr::unordered_map<Name, Name>(unordered_parameter * size);
         auto contract_names_2_1 = pmr::unordered_map<Name, Name>(unordered_parameter * size);
         for (const auto& [name_1, name_2] : contract_pairs) {
            contract_names_1_2[name_1] = name_2;
            contract_names_2_1[name_2] = name_1;
         }
         return std::make_pair(std::move(contract_names_1_2), std::move(contract_names_2_1));
      }

      template<typename ScalarType, typename Symmetry, typename Name>
      void check_valid_contract_plan(
            const Tensor<ScalarType, Symmetry, Name>& tensor_1,
            const Tensor<ScalarType, Symmetry, Name>& tensor_2,
            const std::unordered_set<std::pair<Name, Name>>& contract_pairs,
            const pmr::unordered_map<Name, Name>& contract_names_1_2,
            const pmr::unordered_map<Name, Name>& contract_names_2_1,
            const std::unordered_set<Name>& fuse_names = {}) {
         // check if some missing name in contract
         for (const auto& [name_1, name_2] : contract_pairs) {
            if (auto found = tensor_1.find_by_name(name_1) == tensor_1.names().end()) {
               detail::error("Name missing in contract");
            }
            if (auto found = tensor_2.find_by_name(name_2) == tensor_2.names().end()) {
               detail::error("Name missing in contract");
            }
         }
         // check if some duplicated name in pairs
         for (auto i = contract_pairs.begin(); i != contract_pairs.end(); ++i) {
            for (auto j = std::next(i); j != contract_pairs.end(); ++j) {
               if (i->first == j->first) {
                  detail::error("Duplicated name in contract_names");
               }
               if (i->second == j->second) {
                  detail::error("Duplicated name in contract_names");
               }
            }
         }
         // check if some duplicated name in two tensor except fuse_names and contract_names
         for (const auto& name_1 : tensor_1.names()) {
            for (const auto& name_2 : tensor_2.names()) {
               if ((name_1 == name_2) && (fuse_names.find(name_1) == fuse_names.end()) &&
                   (contract_names_1_2.find(name_1) == contract_names_1_2.end() && contract_names_2_1.find(name_2) == contract_names_2_1.end())) {
                  detail::error("Duplicated name in two contracting tensor but not fused or contracted");
               }
            }
         }
      }
   } // namespace detail

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> contract_without_fuse(
         const Tensor<ScalarType, Symmetry, Name>& tensor_1,
         const Tensor<ScalarType, Symmetry, Name>& tensor_2,
         const std::unordered_set<std::pair<Name, Name>>& contract_pairs) {
      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
      const Rank rank_1 = tensor_1.rank();
      const Rank rank_2 = tensor_2.rank();
      const Rank common_rank = contract_pairs.size();
      const Rank free_rank_1 = rank_1 - common_rank;
      const Rank free_rank_2 = rank_2 - common_rank;
      const auto contract_map = detail::generate_contract_map<Name>(contract_pairs); // lambda cannot capture structure binding
      const auto& contract_names_1_2 = contract_map.first;
      const auto& contract_names_2_1 = contract_map.second;
      if constexpr (debug_mode) {
         detail::check_valid_contract_plan(tensor_1, tensor_2, contract_pairs, contract_names_1_2, contract_names_2_1);
      }
      // reverse -> merge -> product -> split -> reverse
      // reverse free name all two false and recovery them at last, both not apply sign
      // reverse common name to different arrow, one of these two apply sign
      // when merge edge, one of two common edges apply sign, and free edge not apply
      // when splitting edge at last, also not apply
      // so
      // let all edge except tensor_1 common edge arrow as false

      // what need to prepare:
      // reverse set 1
      // reverse set 2
      // merge plan 1 -> free name 1 & common name 1
      // merge plan 2 -> free name 2 & common name 2
      // split plan
      // reverse set result
      // and three result edge name order
      //    -> two put what right and name_result
      auto reversed_names_1 = pmr::unordered_set<Name>(unordered_parameter * free_rank_1);
      auto free_names_1 = pmr::vector<Name>();                                                    // used for merge
      auto common_names_1 = pmr::vector<Name>();                                                  // used for merge
      auto common_reversed_names_1 = pmr::unordered_set<Name>(unordered_parameter * common_rank); // used for reverse merge flag
      free_names_1.reserve(free_rank_1);
      common_names_1.reserve(common_rank); // this will be set later

      auto reversed_names_2 = pmr::unordered_set<Name>(unordered_parameter * free_rank_2);
      auto free_names_2 = pmr::vector<Name>();   // used for merge
      auto common_names_2 = pmr::vector<Name>(); // used for merge
      free_names_2.reserve(free_rank_2);
      common_names_2.reserve(common_rank); // this will be set later

      auto split_map_result = pmr::unordered_map<Name, pmr::vector<std::pair<Name, edge_segments_t<Symmetry, true>>>>(unordered_parameter * 3);
      auto reversed_names_result = pmr::unordered_set<Name>(unordered_parameter * (free_rank_1 + free_rank_2 + common_rank));
      auto names_result = std::vector<Name>();
      auto& split_map_result_part_1 = split_map_result[InternalName<Name>::Contract_1];
      auto& split_map_result_part_2 = split_map_result[InternalName<Name>::Contract_2];
      names_result.reserve(free_rank_1 + free_rank_2);
      split_map_result_part_1.reserve(free_rank_1);
      split_map_result_part_2.reserve(free_rank_2);

      // put name in (free names, names result, split map result part | common name), [reversed_names], [reversed_names_result]
      // common name need to be update later for the proper order
      for (Rank i = 0; i < rank_1; i++) {
         const auto& n = tensor_1.names(i);
         if (contract_names_1_2.find(n) == contract_names_1_2.end()) {
            // it is free name
            free_names_1.push_back(n);
            split_map_result_part_1.push_back({n, {tensor_1.edges(i).segments()}});
            names_result.push_back(n);
            if constexpr (is_fermi) {
               // to false
               if (tensor_1.edges(i).arrow()) {
                  reversed_names_1.insert(n);
                  reversed_names_result.insert(n);
               }
            }
         } else {
            // it is common name
            if constexpr (is_fermi) {
               // to true
               if (!tensor_1.edges(i).arrow()) {
                  reversed_names_1.insert(n);
                  common_reversed_names_1.insert(n);
               }
            }
         }
      }
      for (Rank i = 0; i < rank_2; i++) {
         const auto& n = tensor_2.names(i);
         if (contract_names_2_1.find(n) == contract_names_2_1.end()) {
            // it is free name
            free_names_2.push_back(n);
            split_map_result_part_2.push_back({n, {tensor_2.edges(i).segments()}});
            names_result.push_back(n);
            if constexpr (is_fermi) {
               if (tensor_2.edges(i).arrow()) {
                  reversed_names_2.insert(n);
                  reversed_names_result.insert(n);
               }
            }
         } else {
            // it is common name
            if constexpr (is_fermi) {
               // to false
               if (tensor_2.edges(i).arrow()) {
                  reversed_names_2.insert(n);
               }
            }
         }
      }

      // common name need reorder
      bool put_common_1_right;
      bool put_common_2_right;
      auto fit_tensor_1_common_edges = [&]() {
         for (const auto& n : tensor_1.names()) {
            if (auto position = contract_names_1_2.find(n); position != contract_names_1_2.end()) {
               common_names_1.push_back(position->first);
               common_names_2.push_back(position->second);
            }
         }
      };
      auto fit_tensor_2_common_edges = [&]() {
         for (const auto& n : tensor_2.names()) {
            if (auto position = contract_names_2_1.find(n); position != contract_names_2_1.end()) {
               common_names_2.push_back(position->first);
               common_names_1.push_back(position->second);
            }
         }
      };
      // determine common edge order depend on what
      // to ensure less transpose need to be done
      if (free_rank_1 == 0) {
         put_common_1_right = true;
         fit_tensor_2_common_edges(); // tensor 1 is smaller, so let tensor 2 has better order
         put_common_2_right = common_names_2.empty() || common_names_2.back() == tensor_2.names().back();
      } else if (free_rank_2 == 0) {
         put_common_2_right = true;
         fit_tensor_1_common_edges(); // tensor 2 is smaller, so let tensor 1 has better order
         put_common_1_right = common_names_1.empty() || common_names_1.back() == tensor_1.names().back();
      } else if (free_names_1.back() != tensor_1.names().back()) { // last name in tensor_1 is common name
         put_common_1_right = true;
         fit_tensor_1_common_edges();
         put_common_2_right = common_names_2.empty() || common_names_2.back() == tensor_2.names().back();
      } else if (free_names_2.back() != tensor_2.names().back()) { // last name in tensor_2 is common name
         put_common_2_right = true;
         fit_tensor_2_common_edges();
         put_common_1_right = common_names_1.empty() || common_names_1.back() == tensor_1.names().back();
      } else {
         put_common_1_right = false;
         put_common_2_right = false;
         // fit the larger tensor
         if (tensor_1.storage().size() > tensor_2.storage().size()) {
            fit_tensor_1_common_edges();
         } else {
            fit_tensor_2_common_edges();
         }
      }

      // check edge validity
      if constexpr (debug_mode) {
         // contract edges segments correct
         for (const auto& [name_1, name_2] : contract_pairs) {
            if (tensor_1.edges(name_1).conjugated() != tensor_2.edges(name_2)) {
               detail::error("Incompatible edge segments in contract");
            }
         }
      }

      // merge
      // only apply sign to common edge reverse and merge
      auto tensor_1_merged = tensor_1.edge_operator_implement(
            {},
            reversed_names_1,
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::Contract_1, std::move(free_names_1)},
                  {InternalName<Name>::Contract_2, std::move(common_names_1)}},
            put_common_1_right ? std::vector<Name>{InternalName<Name>::Contract_1, InternalName<Name>::Contract_2} :
                                 std::vector<Name>{InternalName<Name>::Contract_2, InternalName<Name>::Contract_1},
            false,
            {},
            common_reversed_names_1,
            {},
            pmr::set<Name>{InternalName<Name>::Contract_2},
            {});
      auto tensor_2_merged = tensor_2.edge_operator_implement(
            {},
            reversed_names_2,
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::Contract_2, std::move(free_names_2)},
                  {InternalName<Name>::Contract_1, std::move(common_names_2)}},
            put_common_2_right ? std::vector<Name>{InternalName<Name>::Contract_2, InternalName<Name>::Contract_1} :
                                 std::vector<Name>{InternalName<Name>::Contract_1, InternalName<Name>::Contract_2},
            false,
            {},
            {},
            {},
            {},
            {});

      // calculate_product
      auto product_result = Tensor<ScalarType, Symmetry, Name>(
            {InternalName<Name>::Contract_1, InternalName<Name>::Contract_2},
            {tensor_1_merged.edges(put_common_1_right ? 0 : 1), tensor_2_merged.edges(put_common_2_right ? 0 : 1)});

      const auto& edge_0_result = product_result.edges(0);
      const auto& edge_1_result = product_result.edges(1);
      // const auto& edge_common_1 = tensor_1_merged.edges(put_common_1_right ? 1 : 0);
      const auto& edge_common_2 = tensor_2_merged.edges(put_common_2_right ? 1 : 0);

      auto max_batch_size = edge_common_2.segments_size();
      pmr::vector<char> transpose_a_list(max_batch_size), transpose_b_list(max_batch_size);
      pmr::vector<int> m_list(max_batch_size), n_list(max_batch_size), k_list(max_batch_size), lda_list(max_batch_size), ldb_list(max_batch_size),
            ldc_list(max_batch_size);
      pmr::vector<ScalarType> alpha_list(max_batch_size), beta_list(max_batch_size);
      pmr::vector<const ScalarType*> a_list(max_batch_size), b_list(max_batch_size);
      pmr::vector<ScalarType*> c_list(max_batch_size);
      int batch_size = 0;

      for (const auto& [symmetry, _] : edge_0_result.segments()) {
         // m k n
         Size position_0_result = edge_0_result.find_by_symmetry(symmetry) - edge_0_result.segments().begin();
         if (position_0_result == edge_0_result.segments_size()) {
            continue;
         }
         const int m = edge_0_result.segments(position_0_result).second;
         Size position_1_result = edge_1_result.find_by_symmetry(-symmetry) - edge_1_result.segments().begin();
         if (position_1_result == edge_1_result.segments_size()) {
            continue;
         }
         const int n = edge_1_result.segments(position_1_result).second;
         Size position_common = edge_common_2.find_by_symmetry(symmetry) - edge_common_2.segments().begin();
         if (position_common == edge_common_2.segments_size()) {
            const auto positions_result = pmr::vector<Size>{position_0_result, position_1_result};
            auto& data = product_result.blocks(positions_result);
            std::fill(data.data(), data.data() + data.size(), 0);
            continue;
         }
         const int k = edge_common_2.segments(position_common).second;

         const auto positions_result = pmr::vector<Size>{position_0_result, position_1_result};
         const auto positions_1 =
               put_common_1_right ? pmr::vector<Size>{position_0_result, position_common} : pmr::vector<Size>{position_common, position_0_result};
         const auto positions_2 =
               put_common_2_right ? pmr::vector<Size>{position_1_result, position_common} : pmr::vector<Size>{position_common, position_1_result};

         auto& data = product_result.blocks(positions_result);
         const auto& data_1 = tensor_1_merged.blocks(positions_1);
         const auto& data_2 = tensor_2_merged.blocks(positions_2);

         ScalarType alpha = 1;
         if constexpr (is_fermi) {
            // the standard arrow is
            // (false true) (false false)
            // namely
            // (a b) (c d) (c+ b+) = (a d)
            // EPR pair order is (false true)
            if ((put_common_1_right ^ !put_common_2_right) && symmetry.parity()) {
               alpha = -1;
            }
         }
         const ScalarType beta = 0;
         if (m && n && k) {
            transpose_a_list[batch_size] = put_common_2_right ? 'T' : 'N';
            transpose_b_list[batch_size] = put_common_1_right ? 'N' : 'T';
            m_list[batch_size] = n;
            n_list[batch_size] = m;
            k_list[batch_size] = k;
            alpha_list[batch_size] = alpha;
            a_list[batch_size] = data_2.data();
            lda_list[batch_size] = put_common_2_right ? k : n;
            b_list[batch_size] = data_1.data();
            ldb_list[batch_size] = put_common_1_right ? k : m;
            beta_list[batch_size] = beta;
            c_list[batch_size] = data.data();
            ldc_list[batch_size] = n;
            batch_size++;
         } else if (m && n) {
            std::fill(data.data(), data.data() + data.size(), 0);
         }
      }
      detail::gemm_batch<ScalarType, false>(
            transpose_a_list.data(),
            transpose_b_list.data(),
            m_list.data(),
            n_list.data(),
            k_list.data(),
            alpha_list.data(),
            a_list.data(),
            lda_list.data(),
            b_list.data(),
            ldb_list.data(),
            beta_list.data(),
            c_list.data(),
            ldc_list.data(),
            batch_size);

      return product_result.edge_operator_implement(split_map_result, reversed_names_result, {}, std::move(names_result), false, {}, {}, {}, {}, {});
   }

   template<typename ScalarType, typename Name>
   Tensor<ScalarType, Symmetry<>, Name> contract_with_fuse(
         const Tensor<ScalarType, Symmetry<>, Name>& tensor_1,
         const Tensor<ScalarType, Symmetry<>, Name>& tensor_2,
         const std::unordered_set<std::pair<Name, Name>>& contract_pairs,
         const std::unordered_set<Name>& fuse_names) {
      const Rank rank_1 = tensor_1.rank();
      const Rank rank_2 = tensor_2.rank();
      const Rank common_rank = contract_pairs.size();
      const Rank fuse_rank = fuse_names.size();
      const Rank free_rank_1 = rank_1 - common_rank - fuse_rank;
      const Rank free_rank_2 = rank_2 - common_rank - fuse_rank;
      const auto contract_map = detail::generate_contract_map<Name>(contract_pairs); // lambda cannot capture structure binding
      const auto& contract_names_1_2 = contract_map.first;
      const auto& contract_names_2_1 = contract_map.second;
      if constexpr (debug_mode) {
         detail::check_valid_contract_plan(tensor_1, tensor_2, contract_pairs, contract_names_1_2, contract_names_2_1, fuse_names);
      }
      // merge -> product -> split
      // merge to two rank 3 tensor

      // what need to prepare:
      // merge plan 1 -> free name 1 & common name 1 & fuse name
      // merge plan 2 -> free name 2 & common name 2 & fuse name
      // not split plan here, just set name and edge directly
      // so need edge result
      // and three result edge name order
      //    -> two put what right and name_result
      // always put fuse name at first to use gemm_batch
      // so detail order of fuse name is not important
      auto free_names_1 = pmr::vector<Name>();   // used for merge
      auto common_names_1 = pmr::vector<Name>(); // used for merge
      free_names_1.reserve(free_rank_1);
      common_names_1.reserve(common_rank);

      auto free_names_2 = pmr::vector<Name>();   // used for merge
      auto common_names_2 = pmr::vector<Name>(); // used for merge
      free_names_2.reserve(free_rank_2);
      common_names_2.reserve(common_rank);

      auto names_result = std::vector<Name>();
      auto edges_result = std::vector<Edge<Symmetry<>>>();
      names_result.reserve(free_rank_1 + free_rank_2 + fuse_rank);
      edges_result.reserve(free_rank_1 + free_rank_2 + fuse_rank);

      pmr::vector<Name> fuse_names_list;
      fuse_names_list.reserve(fuse_rank);
      for (const auto& name : fuse_names) { // the order of fuse names is not important
         names_result.push_back(name);
         fuse_names_list.push_back(name);
         const auto& edge_1 = tensor_1.edges(name);
         const auto& edge_2 = tensor_2.edges(name);
         if constexpr (debug_mode) {
            if (!(edge_1 == edge_2)) {
               detail::error("Cannot fuse two edge with different shape");
            }
         }
         edges_result.push_back(edge_1);
      }

      for (Rank i = 0; i < rank_1; i++) {
         const auto& n = tensor_1.names(i);
         if (contract_names_1_2.find(n) == contract_names_1_2.end()) {
            // it is free or fuse
            if (fuse_names.find(n) == fuse_names.end()) {
               // it is free
               free_names_1.push_back(n);
               names_result.push_back(n);
               edges_result.push_back(tensor_1.edges(i));
            }
         }
      }
      for (Rank i = 0; i < rank_2; i++) {
         const auto& n = tensor_2.names(i);
         if (contract_names_2_1.find(n) == contract_names_2_1.end()) {
            // it is free or fuse
            if (fuse_names.find(n) == fuse_names.end()) {
               // it is free
               free_names_2.push_back(n);
               names_result.push_back(n);
               edges_result.push_back(tensor_2.edges(i));
            }
         }
      }

      // common name need reorder
      bool put_common_1_right;
      bool put_common_2_right;
      auto fit_tensor_1_common_edges = [&]() {
         for (const auto& n : tensor_1.names()) {
            if (auto position = contract_names_1_2.find(n); position != contract_names_1_2.end()) {
               common_names_1.push_back(position->first);
               common_names_2.push_back(position->second);
            }
         }
      };
      auto fit_tensor_2_common_edges = [&]() {
         for (const auto& n : tensor_2.names()) {
            if (auto position = contract_names_2_1.find(n); position != contract_names_2_1.end()) {
               common_names_2.push_back(position->first);
               common_names_1.push_back(position->second);
            }
         }
      };
      // determine common edge order depend on what
      // to ensure less transpose need to be done
      if (free_rank_1 == 0) {
         put_common_1_right = true;
         fit_tensor_2_common_edges(); // tensor 1 is smaller, so let tensor 2 has better order
         put_common_2_right = common_names_2.empty() || common_names_2.back() == tensor_2.names().back();
      } else if (free_rank_2 == 0) {
         put_common_2_right = true;
         fit_tensor_1_common_edges(); // tensor 2 is smaller, so let tensor 1 has better order
         put_common_1_right = common_names_1.empty() || common_names_1.back() == tensor_1.names().back();
      } else if (free_names_1.back() != tensor_1.names().back()) { // last name in tensor_1 is common name
         put_common_1_right = true;
         fit_tensor_1_common_edges();
         put_common_2_right = common_names_2.empty() || common_names_2.back() == tensor_2.names().back();
      } else if (free_names_2.back() != tensor_2.names().back()) { // last name in tensor_2 is common name
         put_common_2_right = true;
         fit_tensor_2_common_edges();
         put_common_1_right = common_names_1.empty() || common_names_1.back() == tensor_1.names().back();
      } else {
         put_common_1_right = false;
         put_common_2_right = false;
         // fit the larger tensor
         if (tensor_1.storage().size() > tensor_2.storage().size()) {
            fit_tensor_1_common_edges();
         } else {
            fit_tensor_2_common_edges();
         }
      }

      // check edge validity
      if constexpr (debug_mode) {
         // contract name dimension correct
         for (const auto& [name_1, name_2] : contract_pairs) {
            if (tensor_1.edges(name_1) != tensor_2.edges(name_2)) {
               detail::error("Contracting two edge with different dimension");
            }
         }
      }

      // merge
      auto tensor_1_merged = tensor_1.edge_operator_implement(
            {},
            {},
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::Contract_1, std::move(free_names_1)},
                  {InternalName<Name>::Contract_2, std::move(common_names_1)},
                  {InternalName<Name>::Contract_0, fuse_names_list}},
            put_common_1_right ? std::vector<Name>{InternalName<Name>::Contract_0, InternalName<Name>::Contract_1, InternalName<Name>::Contract_2} :
                                 std::vector<Name>{InternalName<Name>::Contract_0, InternalName<Name>::Contract_2, InternalName<Name>::Contract_1},
            false,
            {},
            {},
            {},
            {},
            {});
      auto tensor_2_merged = tensor_2.edge_operator_implement(
            {},
            {},
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::Contract_2, std::move(free_names_2)},
                  {InternalName<Name>::Contract_1, std::move(common_names_2)},
                  {InternalName<Name>::Contract_0, std::move(fuse_names_list)}},
            put_common_2_right ? std::vector<Name>{InternalName<Name>::Contract_0, InternalName<Name>::Contract_2, InternalName<Name>::Contract_1} :
                                 std::vector<Name>{InternalName<Name>::Contract_0, InternalName<Name>::Contract_1, InternalName<Name>::Contract_2},
            false,
            {},
            {},
            {},
            {},
            {});
      // calculate_product
      const int l = tensor_1_merged.edges(0).total_dimension();
      const int m = tensor_1_merged.edges(put_common_1_right ? 1 : 2).total_dimension();
      const int n = tensor_2_merged.edges(put_common_2_right ? 1 : 2).total_dimension();
      const int k = tensor_1_merged.edges(put_common_1_right ? 2 : 1).total_dimension();
      const ScalarType alpha = 1;
      const ScalarType beta = 0;

      auto result = Tensor<ScalarType, Symmetry<>, Name>(std::move(names_result), std::move(edges_result));

      ScalarType* data = result.storage().data();
      const ScalarType* data_1 = tensor_1_merged.storage().data();
      const ScalarType* data_2 = tensor_2_merged.storage().data();
      if (m && n && k) {
         pmr::vector<const ScalarType*> a_list(l), b_list(l);
         pmr::vector<ScalarType*> c_list(l);
         const auto kn = k * n;
         const auto mk = m * k;
         const auto mn = m * n;
         for (auto i = 0; i < l; i++) {
            a_list[i] = data_2 + kn * i;
            b_list[i] = data_1 + mk * i;
            c_list[i] = data + mn * i;
         }
         detail::gemm_batch<ScalarType, true>(
               put_common_2_right ? "T" : "N",
               put_common_1_right ? "N" : "T",
               &n,
               &m,
               &k,
               &alpha,
               a_list.data(),
               put_common_2_right ? &k : &n,
               b_list.data(),
               put_common_1_right ? &k : &m,
               &beta,
               c_list.data(),
               &n,
               l);
      } else if (m && n) {
         std::fill(result.storage().begin(), result.storage().end(), 0);
      }
      return result;
   }

   inline timer contract_guard("contract");

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::contract_implement(
         const Tensor<ScalarType, Symmetry, Name>& tensor_1,
         const Tensor<ScalarType, Symmetry, Name>& tensor_2,
         const std::unordered_set<std::pair<Name, Name>>& contract_pairs,
         const std::unordered_set<Name>& fuse_names) {
      auto timer_guard = contract_guard();
      auto pmr_guard = scope_resource(default_buffer_size);
      if constexpr (Symmetry::length == 0) {
         return contract_with_fuse(tensor_1, tensor_2, contract_pairs, fuse_names);
      } else {
         if constexpr (debug_mode) {
            if (fuse_names.size() != 0) {
               detail::error("Cannot fuse edge of symmetric tensor");
            }
         }
         return contract_without_fuse(tensor_1, tensor_2, contract_pairs);
      }
   }
} // namespace TAT
#endif
