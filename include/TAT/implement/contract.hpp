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

#include <cublas_v2.h>

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
}

namespace TAT {
   namespace detail {

      template<typename ScalarType>
      using cuda_complex_wrap = std::conditional_t<
            is_complex<ScalarType>,
            std::conditional_t<std::is_same_v<ScalarType, std::complex<double>>, cuDoubleComplex, cuFloatComplex>,
            ScalarType>;

      template<typename T>
      auto cuda_complex_wrap_value(T* value) {
         if constexpr (std::is_const_v<T>) {
            return reinterpret_cast<const cuda_complex_wrap<std::remove_const_t<T>>*>(value);
         } else {
            return reinterpret_cast<cuda_complex_wrap<T>*>(value);
         }
      }

      template<typename ScalarType>
      constexpr cublasStatus_t (*cuda_gemm)(
            cublasHandle_t,
            cublasOperation_t transpose_a,
            cublasOperation_t transpose_b,
            const int m,
            const int n,
            const int k,
            const cuda_complex_wrap<ScalarType>* alpha,
            const cuda_complex_wrap<ScalarType>* a,
            const int lda,
            const cuda_complex_wrap<ScalarType>* b,
            const int ldb,
            const cuda_complex_wrap<ScalarType>* beta,
            cuda_complex_wrap<ScalarType>* c,
            const int ldc) = nullptr;

      template<>
      inline auto cuda_gemm<float> = cublasSgemm;
      template<>
      inline auto cuda_gemm<double> = cublasDgemm;
      template<>
      inline auto cuda_gemm<std::complex<float>> = cublasCgemm;
      template<>
      inline auto cuda_gemm<std::complex<double>> = cublasZgemm;

      template<typename ScalarType>
      std::pair<cudaStream_t, cublasHandle_t>
      gemm(const char* transpose_a,
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
           const int* ldc) {
         cudaStream_t stream;
         cudaStreamCreate(&stream);
         cublasHandle_t handle;
         cublasCreate(&handle);
         cublasSetStream(handle, stream);
         cuda_gemm<ScalarType>(
               handle,
               'N' == *transpose_a ? CUBLAS_OP_N : CUBLAS_OP_T,
               'N' == *transpose_b ? CUBLAS_OP_N : CUBLAS_OP_T,
               *m,
               *n,
               *k,
               cuda_complex_wrap_value(alpha),
               cuda_complex_wrap_value(a),
               *lda,
               cuda_complex_wrap_value(b),
               *ldb,
               cuda_complex_wrap_value(beta),
               cuda_complex_wrap_value(c),
               *ldc);
         return {stream, handle};
      }
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
         pmr::vector<std::pair<cudaStream_t, cublasHandle_t>> handles;
         if (batch_size == 1) {
            handles.push_back(gemm<ScalarType>(transpose_a, transpose_b, m, n, k, alpha, a[0], lda, b[0], ldb, beta, c[0], ldc));
         } else {
            if constexpr (same_shape) {
               for (auto i = 0; i < batch_size; i++) {
                  handles.push_back(gemm<ScalarType>(transpose_a, transpose_b, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i], ldc));
               }
            } else {
               for (auto i = 0; i < batch_size; i++) {
                  handles.push_back(gemm<ScalarType>(
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
                        &ldc[i]));
               }
            }
         }
         cudaDeviceSynchronize();
         for (auto& [stream, handle] : handles) {
            cublasDestroy(handle);
            cudaStreamDestroy(stream);
         }
      }

      template<typename Name, typename SetNameName>
      auto generate_contract_map(const SetNameName& contract_names) {
         auto size = contract_names.size();
         auto contract_names_1_2 = pmr::unordered_map<Name, Name>(unordered_parameter * size);
         auto contract_names_2_1 = pmr::unordered_map<Name, Name>(unordered_parameter * size);
         for (const auto& [name_1, name_2] : contract_names) {
            contract_names_1_2[name_1] = name_2;
            contract_names_2_1[name_2] = name_1;
         }
         return std::make_tuple(std::move(contract_names_1_2), std::move(contract_names_2_1));
      }

      template<typename ScalarType, typename Symmetry, typename Name>
      void check_valid_contract_plan(
            const Tensor<ScalarType, Symmetry, Name>& tensor_1,
            const Tensor<ScalarType, Symmetry, Name>& tensor_2,
            const std::unordered_set<std::pair<Name, Name>>& contract_names,
            const pmr::unordered_map<Name, Name>& contract_names_1_2,
            const pmr::unordered_map<Name, Name>& contract_names_2_1,
            const std::unordered_set<Name>& fuse_names = {}) {
         // check if some missing name in contract
         for (const auto& [name_1, name_2] : contract_names) {
            if (auto found = tensor_1.find_rank_from_name(name_1) == tensor_1.names.end()) {
               detail::error("Name missing in contract");
            }
            if (auto found = tensor_2.find_rank_from_name(name_2) == tensor_2.names.end()) {
               detail::error("Name missing in contract");
            }
         }
         // check if some duplicated name in pairs
         for (auto i = contract_names.begin(); i != contract_names.end(); ++i) {
            for (auto j = std::next(i); j != contract_names.end(); ++j) {
               if (i->first == j->first) {
                  detail::error("Duplicated name in contract_names");
               }
               if (i->second == j->second) {
                  detail::error("Duplicated name in contract_names");
               }
            }
         }
         // check if some duplicated name in two tensor except fuse_names
         for (const auto& name_1 : tensor_1.names) {
            for (const auto& name_2 : tensor_2.names) {
               if ((name_1 == name_2) && (fuse_names.find(name_1) == fuse_names.end()) &&
                   (contract_names_1_2.find(name_1) == contract_names_1_2.end() && contract_names_2_1.find(name_2) == contract_names_2_1.end())) {
                  detail::error("Duplicated name in two contracting tensor but not fusing");
               }
            }
         }
      }
   } // namespace detail

   template<typename ScalarType, typename Name, typename = std::enable_if_t<is_scalar<ScalarType> && is_name<Name>>>
   Tensor<ScalarType, Symmetry<>, Name> contract_with_fuse(
         const Tensor<ScalarType, Symmetry<>, Name>& tensor_1,
         const Tensor<ScalarType, Symmetry<>, Name>& tensor_2,
         const std::unordered_set<std::pair<Name, Name>>& contract_names,
         const std::unordered_set<Name>& fuse_names);

   template<
         typename ScalarType,
         typename Symmetry,
         typename Name,
         typename = std::enable_if_t<is_scalar<ScalarType> && is_symmetry<Symmetry> && is_name<Name>>>
   Tensor<ScalarType, Symmetry, Name> contract_without_fuse(
         const Tensor<ScalarType, Symmetry, Name>& tensor_1,
         const Tensor<ScalarType, Symmetry, Name>& tensor_2,
         const std::unordered_set<std::pair<Name, Name>>& contract_names);

   inline timer contract_guard("contract");

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::contract_implement(
         const Tensor<ScalarType, Symmetry, Name>& tensor_1,
         const Tensor<ScalarType, Symmetry, Name>& tensor_2,
         const std::unordered_set<std::pair<Name, Name>>& contract_names,
         const std::unordered_set<Name>& fuse_names) {
      auto timer_guard = contract_guard();
      auto pmr_guard = scope_resource(default_buffer_size);
      if constexpr (Symmetry::length == 0) {
         return contract_with_fuse(tensor_1, tensor_2, contract_names, fuse_names);
      } else {
         if constexpr (debug_mode) {
            if (fuse_names.size() != 0) {
               detail::error("Cannot fuse edge of symmetric tensor");
            }
         }
         return contract_without_fuse(tensor_1, tensor_2, contract_names);
      }
   }

   template<typename ScalarType, typename Symmetry, typename Name, typename>
   Tensor<ScalarType, Symmetry, Name> contract_without_fuse(
         const Tensor<ScalarType, Symmetry, Name>& tensor_1,
         const Tensor<ScalarType, Symmetry, Name>& tensor_2,
         const std::unordered_set<std::pair<Name, Name>>& contract_names) {
      constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
      const Rank rank_1 = tensor_1.get_rank();
      const Rank rank_2 = tensor_2.get_rank();
      const Rank common_rank = contract_names.size();
      const Rank free_rank_1 = rank_1 - common_rank;
      const Rank free_rank_2 = rank_2 - common_rank;
      const auto& contract_names_maps = detail::generate_contract_map<Name>(contract_names);
      const auto& contract_names_1_2 = std::get<0>(contract_names_maps);
      const auto& contract_names_2_1 = std::get<1>(contract_names_maps);
      if constexpr (debug_mode) {
         detail::check_valid_contract_plan(tensor_1, tensor_2, contract_names, contract_names_1_2, contract_names_2_1);
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
      auto reversed_set_1 = pmr::unordered_set<Name>(unordered_parameter * free_rank_1);
      auto free_name_1 = pmr::vector<Name>();                                                  // used for merge
      auto common_name_1 = pmr::vector<Name>();                                                // used for merge
      auto common_reverse_set_1 = pmr::unordered_set<Name>(unordered_parameter * common_rank); // used for reverse merge flag
      free_name_1.reserve(free_rank_1);
      common_name_1.reserve(common_rank); // this will be set later

      auto reversed_set_2 = pmr::unordered_set<Name>(unordered_parameter * free_rank_2);
      auto free_name_2 = pmr::vector<Name>();   // used for merge
      auto common_name_2 = pmr::vector<Name>(); // used for merge
      free_name_2.reserve(free_rank_2);
      common_name_2.reserve(common_rank); // this will be set later

      auto split_map_result = pmr::unordered_map<Name, pmr::vector<std::pair<Name, edge_segment_t<Symmetry>>>>(unordered_parameter * 3);
      auto reversed_set_result = pmr::unordered_set<Name>(unordered_parameter * (free_rank_1 + free_rank_2 + common_rank));
      auto name_result = std::vector<Name>();
      auto& split_map_result_part_1 = split_map_result[InternalName<Name>::Contract_1];
      auto& split_map_result_part_2 = split_map_result[InternalName<Name>::Contract_2];
      name_result.reserve(free_rank_1 + free_rank_2);
      split_map_result_part_1.reserve(free_rank_1);
      split_map_result_part_2.reserve(free_rank_2);

      for (Rank i = 0; i < rank_1; i++) {
         const auto& n = tensor_1.names[i];
         if (contract_names_1_2.find(n) == contract_names_1_2.end()) {
            // it is free name
            free_name_1.push_back(n);
            split_map_result_part_1.push_back({n, {tensor_1.edges(i).segment}});
            name_result.push_back(n);
            if constexpr (is_fermi) {
               // to false
               if (tensor_1.edges(i).arrow) {
                  reversed_set_1.insert(n);
                  reversed_set_result.insert(n);
               }
            }
         } else {
            // it is common name
            if constexpr (is_fermi) {
               // to true
               if (!tensor_1.edges(i).arrow) {
                  reversed_set_1.insert(n);
                  common_reverse_set_1.insert(n);
               }
            }
         }
      }
      for (Rank i = 0; i < rank_2; i++) {
         const auto& n = tensor_2.names[i];
         if (contract_names_2_1.find(n) == contract_names_2_1.end()) {
            // it is free name
            free_name_2.push_back(n);
            split_map_result_part_2.push_back({n, {tensor_2.edges(i).segment}});
            name_result.push_back(n);
            if constexpr (is_fermi) {
               if (tensor_2.edges(i).arrow) {
                  reversed_set_2.insert(n);
                  reversed_set_result.insert(n);
               }
            }
         } else {
            // it is common name
            if constexpr (is_fermi) {
               // to false
               if (tensor_2.edges(i).arrow) {
                  reversed_set_2.insert(n);
               }
            }
         }
      }

      // common name need reorder
      bool put_common_1_right;
      bool put_common_2_right;
      auto fit_tensor_1_common_edge = [&]() {
         for (const auto& n : tensor_1.names) {
            if (auto position = contract_names_1_2.find(n); position != contract_names_1_2.end()) {
               common_name_1.push_back(std::get<0>(*position));
               common_name_2.push_back(std::get<1>(*position));
            }
         }
      };
      auto fit_tensor_2_common_edge = [&]() {
         for (const auto& n : tensor_2.names) {
            if (auto position = contract_names_2_1.find(n); position != contract_names_2_1.end()) {
               common_name_1.push_back(std::get<1>(*position));
               common_name_2.push_back(std::get<0>(*position));
            }
         }
      };
      // determine common edge order depend on what
      if (free_rank_1 == 0) {
         put_common_1_right = true;
         fit_tensor_2_common_edge();
         put_common_2_right = common_name_2.empty() || common_name_2.back() == tensor_2.names.back();
      } else if (free_rank_2 == 0) {
         put_common_2_right = true;
         fit_tensor_1_common_edge();
         put_common_1_right = common_name_1.empty() || common_name_1.back() == tensor_1.names.back();
      } else if (free_name_1.back() != tensor_1.names.back()) {
         put_common_1_right = true;
         fit_tensor_1_common_edge();
         put_common_2_right = common_name_2.empty() || common_name_2.back() == tensor_2.names.back();
      } else if (free_name_2.back() != tensor_2.names.back()) {
         put_common_2_right = true;
         fit_tensor_2_common_edge();
         put_common_1_right = common_name_1.empty() || common_name_1.back() == tensor_1.names.back();
      } else {
         put_common_1_right = false;
         put_common_2_right = false;
         // fit the larger tensor
         if (tensor_1.storage().size() > tensor_2.storage().size()) {
            fit_tensor_1_common_edge();
         } else {
            fit_tensor_2_common_edge();
         }
      }

      // delete uncommon symmetry
      // and check symmetry order
      auto delete_1 = pmr::unordered_map<Name, pmr::unordered_map<Symmetry, Size>>(unordered_parameter * common_rank);
      auto delete_2 = pmr::unordered_map<Name, pmr::unordered_map<Symmetry, Size>>(unordered_parameter * common_rank);
      for (Rank i = 0; i < common_rank; i++) {
         const auto& name_1 = common_name_1[i];
         const auto& name_2 = common_name_2[i];
         const auto& edge_1 = tensor_1.edges(name_1);
         const auto& edge_2 = tensor_2.edges(name_2);
         // same to trace delete dimension
         auto delete_unused_dimension = [](const auto& edge_this, const auto& edge_other, const auto& name_this, auto& delete_this) {
            constexpr bool is_fermi = Symmetry::is_fermi_symmetry; // MSVC require capture it even it is constexpr
            if constexpr (debug_mode) {
               if constexpr (is_fermi) {
                  if (edge_this.arrow == edge_other.arrow) {
                     detail::error("Different Fermi Arrow to Contract");
                  }
               }
            }
            auto delete_map = pmr::unordered_map<Symmetry, Size>(unordered_parameter * edge_this.segment.size());
            for (const auto& [symmetry, dimension] : edge_this.segment) {
               auto found = edge_other.find_by_symmetry(-symmetry);
               if (found != edge_other.segment.end()) {
                  // found
                  if constexpr (debug_mode) {
                     if (const auto dimension_other = found->second; dimension_other != dimension) {
                        detail::error("Different Dimension to Contract");
                     }
                  }
               } else {
                  // not found
                  delete_map[symmetry] = 0;
                  // pass delete map when merging, edge operator will delete the entire segment if it is zero
               }
            }
            if (!delete_map.empty()) {
               return delete_this.emplace(name_this, std::move(delete_map)).first;
            } else {
               return delete_this.end();
            }
         };
         auto delete_map_edge_1_iterator = delete_unused_dimension(edge_1, edge_2, name_1, delete_1);
         auto delete_map_edge_2_iterator = delete_unused_dimension(edge_2, edge_1, name_2, delete_2);
         if constexpr (debug_mode) {
            // check different order
            auto empty_delete_map = pmr::unordered_map<Symmetry, Size>();
            const auto& delete_map_edge_1 = [&]() -> const auto& {
               if (delete_map_edge_1_iterator == delete_1.end()) {
                  return empty_delete_map;
               } else {
                  return delete_map_edge_1_iterator->second;
               }
            }
            ();
            const auto& delete_map_edge_2 = [&]() -> const auto& {
               if (delete_map_edge_2_iterator == delete_2.end()) {
                  return empty_delete_map;
               } else {
                  return delete_map_edge_2_iterator->second;
               }
            }
            ();
            // it is impossible that one of i, j reach end and another not
            for (auto [i, j] = std::tuple{edge_1.segment.begin(), edge_2.segment.begin()}; i != edge_1.segment.end() && j != edge_2.segment.end();
                 ++i, ++j) {
               // i/j -> first :: Symmetry
               while (delete_map_edge_1.find(i->first) != delete_map_edge_1.end()) {
                  ++i;
               }
               while (delete_map_edge_2.find(j->first) != delete_map_edge_2.end()) {
                  ++j;
               }
               if ((i->first + j->first) != Symmetry()) {
                  detail::error("Different symmetry segment order in contract");
               }
            }
         }
      }

      // merge
      // only apply sign to common edge reverse and merge
      auto tensor_1_merged = tensor_1.edge_operator_implement(
            empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
            reversed_set_1,
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::Contract_1, std::move(free_name_1)},
                  {InternalName<Name>::Contract_2, std::move(common_name_1)}},
            put_common_1_right ? std::vector<Name>{InternalName<Name>::Contract_1, InternalName<Name>::Contract_2} :
                                 std::vector<Name>{InternalName<Name>::Contract_2, InternalName<Name>::Contract_1},
            false,
            empty_list<Name>(),
            common_reverse_set_1,
            empty_list<Name>(),
            pmr::set<Name>{InternalName<Name>::Contract_2},
            delete_1);
      auto tensor_2_merged = tensor_2.edge_operator_implement(
            empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
            reversed_set_2,
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::Contract_2, std::move(free_name_2)},
                  {InternalName<Name>::Contract_1, std::move(common_name_2)}},
            put_common_2_right ? std::vector<Name>{InternalName<Name>::Contract_2, InternalName<Name>::Contract_1} :
                                 std::vector<Name>{InternalName<Name>::Contract_1, InternalName<Name>::Contract_2},
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            delete_2);

      // calculate_product
      auto product_result = Tensor<ScalarType, Symmetry, Name>(
            {InternalName<Name>::Contract_1, InternalName<Name>::Contract_2},
            {std::move(tensor_1_merged.edges(put_common_1_right ? 0 : 1)), std::move(tensor_2_merged.edges(put_common_2_right ? 0 : 1))});
      const auto& common_edge = tensor_1_merged.edges(put_common_1_right ? 1 : 0); // tensor 1 common edge

      auto max_batch_size = product_result.core->blocks.size();
      no_initialize::pmr::vector<char> transpose_a_list(max_batch_size), transpose_b_list(max_batch_size);
      no_initialize::pmr::vector<int> m_list(max_batch_size), n_list(max_batch_size), k_list(max_batch_size), lda_list(max_batch_size),
            ldb_list(max_batch_size), ldc_list(max_batch_size);
      no_initialize::pmr::vector<ScalarType> alpha_list(max_batch_size), beta_list(max_batch_size);
      no_initialize::pmr::vector<const ScalarType*> a_list(max_batch_size), b_list(max_batch_size);
      no_initialize::pmr::vector<ScalarType*> c_list(max_batch_size);
      int batch_size = 0;

      for (auto& [symmetries, data] : product_result.core->blocks) {
         // m k n
         auto reversed_symmetries = std::vector<Symmetry>{symmetries[1], symmetries[0]};
         const auto& symmetries_1 = put_common_1_right ? symmetries : reversed_symmetries;
         const auto& symmetries_2 = put_common_2_right ? reversed_symmetries : symmetries;
         const auto& data_1 = tensor_1_merged.blocks(symmetries_1);
         const auto& data_2 = tensor_2_merged.blocks(symmetries_2);
         const int m = product_result.edges(0).get_dimension_from_symmetry(symmetries[0]);
         const int n = product_result.edges(1).get_dimension_from_symmetry(symmetries[1]);
         const int k = common_edge.get_dimension_from_symmetry(symmetries[1]);
         ScalarType alpha = 1;
         if constexpr (is_fermi) {
            // the standard arrow is
            // (false true) (false false)
            // namely
            // (a b) (c d) (c+ b+) = (a d)
            // EPR pair order is (false true)
            if ((put_common_2_right ^ !put_common_1_right) && symmetries[0].get_parity()) {
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
            std::fill(data.begin(), data.end(), 0);
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

      return product_result.edge_operator_implement(
            split_map_result,
            reversed_set_result,
            empty_list<std::pair<Name, empty_list<Name>>>(),
            std::move(name_result),
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());
   }

   template<typename ScalarType, typename Name, typename>
   Tensor<ScalarType, Symmetry<>, Name> contract_with_fuse(
         const Tensor<ScalarType, Symmetry<>, Name>& tensor_1,
         const Tensor<ScalarType, Symmetry<>, Name>& tensor_2,
         const std::unordered_set<std::pair<Name, Name>>& contract_names,
         const std::unordered_set<Name>& fuse_names) {
      const Rank rank_1 = tensor_1.get_rank();
      const Rank rank_2 = tensor_2.get_rank();
      const Rank common_rank = contract_names.size();
      const Rank fuse_rank = fuse_names.size();
      const Rank free_rank_1 = rank_1 - common_rank - fuse_rank;
      const Rank free_rank_2 = rank_2 - common_rank - fuse_rank;
      const auto& contract_names_maps = detail::generate_contract_map<Name>(contract_names);
      const auto& contract_names_1_2 = std::get<0>(contract_names_maps);
      const auto& contract_names_2_1 = std::get<1>(contract_names_maps);
      if constexpr (debug_mode) {
         detail::check_valid_contract_plan(tensor_1, tensor_2, contract_names, contract_names_1_2, contract_names_2_1, fuse_names);
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
      // always put fuse name at first
      // so detail order of fuse name is not important
      auto free_name_1 = pmr::vector<Name>();   // used for merge
      auto common_name_1 = pmr::vector<Name>(); // used for merge
      free_name_1.reserve(free_rank_1);
      common_name_1.reserve(common_rank);

      auto free_name_2 = pmr::vector<Name>();   // used for merge
      auto common_name_2 = pmr::vector<Name>(); // used for merge
      free_name_2.reserve(free_rank_2);
      common_name_2.reserve(common_rank);

      auto name_result = std::vector<Name>();
      auto edge_result = std::vector<Edge<Symmetry<>>>();
      name_result.reserve(free_rank_1 + free_rank_2 + fuse_rank);
      edge_result.reserve(free_rank_1 + free_rank_2 + fuse_rank);

      pmr::vector<Name> fuse_names_list;
      fuse_names_list.reserve(fuse_rank);
      for (const auto& name : fuse_names) {
         name_result.push_back(name);
         fuse_names_list.push_back(name);
         const auto& edge_1 = tensor_1.edges(name);
         const auto& edge_2 = tensor_2.edges(name);
         if constexpr (debug_mode) {
            if (!(edge_1 == edge_2)) {
               detail::error("Cannot fuse two edge with different shape");
            }
         }
         edge_result.push_back(edge_1);
      }

      for (Rank i = 0; i < rank_1; i++) {
         const auto& n = tensor_1.names[i];
         if (contract_names_1_2.find(n) == contract_names_1_2.end()) {
            // it is free or fuse
            if (fuse_names.find(n) == fuse_names.end()) {
               // it is free
               name_result.push_back(n);
               free_name_1.push_back(n);
               edge_result.push_back(tensor_1.edges(i));
            }
         }
      }
      for (Rank i = 0; i < rank_2; i++) {
         const auto& n = tensor_2.names[i];
         if (contract_names_2_1.find(n) == contract_names_2_1.end()) {
            // it is free or fuse
            if (fuse_names.find(n) == fuse_names.end()) {
               // it is free
               name_result.push_back(n);
               free_name_2.push_back(n);
               edge_result.push_back(tensor_2.edges(i));
            }
         }
      }

      // common name need reorder
      bool put_common_1_right;
      bool put_common_2_right;
      auto fit_tensor_1_common_edge = [&]() {
         for (const auto& n : tensor_1.names) {
            if (auto position = contract_names_1_2.find(n); position != contract_names_1_2.end()) {
               common_name_1.push_back(std::get<0>(*position));
               common_name_2.push_back(std::get<1>(*position));
            }
         }
      };
      auto fit_tensor_2_common_edge = [&]() {
         for (const auto& n : tensor_2.names) {
            if (auto position = contract_names_2_1.find(n); position != contract_names_2_1.end()) {
               common_name_1.push_back(std::get<1>(*position));
               common_name_2.push_back(std::get<0>(*position));
            }
         }
      };
      // determine common edge order depend on what
      if (free_rank_1 == 0) {
         put_common_1_right = true;
         fit_tensor_2_common_edge();
         put_common_2_right = common_name_2.empty() || common_name_2.back() == tensor_2.names.back();
      } else if (free_rank_2 == 0) {
         put_common_2_right = true;
         fit_tensor_1_common_edge();
         put_common_1_right = common_name_1.empty() || common_name_1.back() == tensor_1.names.back();
      } else if (free_name_1.back() != tensor_1.names.back()) {
         put_common_1_right = true;
         fit_tensor_1_common_edge();
         put_common_2_right = common_name_2.empty() || common_name_2.back() == tensor_2.names.back();
      } else if (free_name_2.back() != tensor_2.names.back()) {
         put_common_2_right = true;
         fit_tensor_2_common_edge();
         put_common_1_right = common_name_1.empty() || common_name_1.back() == tensor_1.names.back();
      } else {
         put_common_1_right = false;
         put_common_2_right = false;
         // fit the larger tensor
         if (tensor_1.storage().size() > tensor_2.storage().size()) {
            fit_tensor_1_common_edge();
         } else {
            fit_tensor_2_common_edge();
         }
      }

      // check edge validity
      if constexpr (debug_mode) {
         // contract name dimension correct
         for (const auto& [name_1, name_2] : contract_names) {
            if (tensor_1.edges(name_1) != tensor_2.edges(name_2)) {
               detail::error("Contracting two edge with different dimension");
            }
         }
      }

      // merge
      auto tensor_1_merged = tensor_1.edge_operator_implement(
            empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segment_t<Symmetry<>>>>>>(),
            empty_list<Name>(),
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::Contract_1, std::move(free_name_1)},
                  {InternalName<Name>::Contract_2, std::move(common_name_1)},
                  {InternalName<Name>::Contract_0, fuse_names_list}},
            put_common_1_right ? std::vector<Name>{InternalName<Name>::Contract_0, InternalName<Name>::Contract_1, InternalName<Name>::Contract_2} :
                                 std::vector<Name>{InternalName<Name>::Contract_0, InternalName<Name>::Contract_2, InternalName<Name>::Contract_1},
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, empty_list<std::pair<Symmetry<>, Size>>>>());
      auto tensor_2_merged = tensor_2.edge_operator_implement(
            empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segment_t<Symmetry<>>>>>>(),
            empty_list<Name>(),
            pmr::map<Name, pmr::vector<Name>>{
                  {InternalName<Name>::Contract_2, std::move(free_name_2)},
                  {InternalName<Name>::Contract_1, std::move(common_name_2)},
                  {InternalName<Name>::Contract_0, std::move(fuse_names_list)}},
            put_common_2_right ? std::vector<Name>{InternalName<Name>::Contract_0, InternalName<Name>::Contract_2, InternalName<Name>::Contract_1} :
                                 std::vector<Name>{InternalName<Name>::Contract_0, InternalName<Name>::Contract_1, InternalName<Name>::Contract_2},
            false,
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, empty_list<std::pair<Symmetry<>, Size>>>>());
      // calculate_product
      const int l = tensor_1_merged.edges(0).segment.front().second;
      const int m = tensor_1_merged.edges(put_common_1_right ? 1 : 2).segment.front().second;
      const int n = tensor_2_merged.edges(put_common_2_right ? 1 : 2).segment.front().second;
      const int k = tensor_1_merged.edges(put_common_1_right ? 2 : 1).segment.front().second;
      const ScalarType alpha = 1;
      const ScalarType beta = 0;

      auto result = Tensor<ScalarType, Symmetry<>, Name>(std::move(name_result), std::move(edge_result));

      ScalarType* data = result.storage().data();
      const ScalarType* data_1 = tensor_1_merged.storage().data();
      const ScalarType* data_2 = tensor_2_merged.storage().data();
      if (m && n && k) {
         pmr::vector<const ScalarType*> a_list(l), b_list(l);
         pmr::vector<ScalarType*> c_list(l);
         for (auto i = 0; i < l; i++) {
            a_list[i] = data_2 + k * n * i;
            b_list[i] = data_1 + m * k * i;
            c_list[i] = data + m * n * i;
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
} // namespace TAT
#endif
