/**
 * \file contract.hpp
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
#ifndef TAT_CONTRACT_HPP
#define TAT_CONTRACT_HPP

#include "tensor.hpp"

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
   template<class ScalarType>
   void calculate_product(
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
         const int* ldc);

   template<>
   inline void calculate_product<float>(
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
         const int* ldc) {
      sgemm_(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }
   template<>
   inline void calculate_product<double>(
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
         const int* ldc) {
      dgemm_(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }
   template<>
   inline void calculate_product<std::complex<float>>(
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
         const int* ldc) {
      cgemm_(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }
   template<>
   inline void calculate_product<std::complex<double>>(
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
         const int* ldc) {
      zgemm_(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }

   template<int i>
   auto find_in_contract_names(const std::set<std::tuple<Name, Name>>& contract_names, const Name& name) {
      auto iterator = contract_names.begin();
      for (; iterator != contract_names.end(); ++iterator) {
         if (std::get<i>(*iterator) == name) {
            return iterator;
         }
      }
      return iterator;
   }

   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry> Tensor<ScalarType, Symmetry>::contract(
         const Tensor<ScalarType, Symmetry>& tensor_1,
         const Tensor<ScalarType, Symmetry>& tensor_2,
         std::set<std::tuple<Name, Name>> contract_names) {
      // 为未来split做准备
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
      constexpr bool is_no_symmetry = std::is_same_v<Symmetry, NoSymmetry>;
      const Rank rank_1 = tensor_1.names.size();
      const Rank rank_2 = tensor_2.names.size();
      // 删除不存在的名称
      for (auto iterator = contract_names.begin(); iterator != contract_names.end();) {
         auto found_1 = tensor_1.name_to_index.find(std::get<0>(*iterator));
         auto found_2 = tensor_2.name_to_index.find(std::get<1>(*iterator));
         if (found_1 == tensor_1.name_to_index.end() || found_2 == tensor_2.name_to_index.end()) {
            iterator = contract_names.erase(iterator);
         } else {
            ++iterator;
         }
      }
      // Rank contract_origin_rank = contract_names_1.size();
      // for (Rank i = 0; i < contract_origin_rank; i++) {
      // }
      // 需要反转成 - + - -
      // 事后恢复两侧的边
      auto reversed_set_1 = std::set<Name>();           // 第一个张量merge时反转表
      auto reversed_set_2 = std::set<Name>();           // 第二个张量merge时反转表
      auto edge_result = std::vector<Edge<Symmetry>>(); // 无对称性的时候不需要split方案直接获取最后的edge
      auto split_map_result = std::map<Name, std::vector<std::tuple<Name, BoseEdge<Symmetry>>>>(); // split方案
      auto reversed_set_result = std::set<Name>();                                                 // 最后split时的反转标
      auto name_result = std::vector<Name>();                                                      // 最后split后的name
      split_map_result[Contract1];
      split_map_result[Contract2];
      auto free_name_1 = std::vector<Name>(); // 第一个张量的自由边, merge时使用
      for (Rank i = 0; i < rank_1; i++) {
         const auto& n = tensor_1.names[i];
         if (find_in_contract_names<0>(contract_names, n) == contract_names.end()) {
            free_name_1.push_back(n);
            if constexpr (is_no_symmetry) {
               edge_result.push_back(tensor_1.core->edges[i]);
            } else {
               split_map_result.at(Contract1).push_back({n, {tensor_1.core->edges[i].map}});
            }
            name_result.push_back(n);
            if constexpr (is_fermi) {
               if (tensor_1.core->edges[i].arrow) {
                  reversed_set_1.insert(n);
                  reversed_set_result.insert(n);
               }
            }
         } else {
            if constexpr (is_fermi) {
               if (!tensor_1.core->edges[i].arrow) {
                  reversed_set_1.insert(n);
               }
            }
         }
      }
      const auto free_rank_1 = free_name_1.size();
      auto free_name_2 = std::vector<Name>(); // 第二个张量的自由边, merge时使用
      for (Rank i = 0; i < rank_2; i++) {
         const auto& n = tensor_2.names[i];
         if (find_in_contract_names<1>(contract_names, n) == contract_names.end()) {
            free_name_2.push_back(n);
            if constexpr (is_no_symmetry) {
               edge_result.push_back(tensor_2.core->edges[i]);
            } else {
               split_map_result.at(Contract2).push_back({n, {tensor_2.core->edges[i].map}});
            }
            name_result.push_back(n);
            if constexpr (is_fermi) {
               if (tensor_2.core->edges[i].arrow) {
                  reversed_set_2.insert(n);
                  reversed_set_result.insert(n);
               }
            }
         } else {
            if constexpr (is_fermi) {
               if (tensor_2.core->edges[i].arrow) {
                  reversed_set_2.insert(n);
               }
            }
         }
      }
      const auto free_rank_2 = free_name_2.size();
      // 确定转置方案
      Rank common_rank = 0;
      auto common_name_1 = std::vector<Name>(); // 第一个张量的公共边, merge时使用
      auto common_name_2 = std::vector<Name>(); // 第二个张量的公共边, merge时使用
      bool put_common_1_right;
      bool put_common_2_right;
      // 这会自动删掉两边都不在name中的情况
      auto fit_tensor_1_common_edge = [&]() {
         for (const auto& n : tensor_1.names) {
            if (auto position = find_in_contract_names<0>(contract_names, n); position != contract_names.end()) {
               common_name_1.push_back(std::get<0>(*position));
               common_name_2.push_back(std::get<1>(*position));
               common_rank++;
            }
         }
      };
      auto fit_tensor_2_common_edge = [&]() {
         for (const auto& n : tensor_2.names) {
            if (auto position = find_in_contract_names<1>(contract_names, n); position != contract_names.end()) {
               common_name_1.push_back(std::get<0>(*position));
               common_name_2.push_back(std::get<1>(*position));
               common_rank++;
            }
         }
      };
      // 确定方案
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
         fit_tensor_2_common_edge();
         // 所以尽量大张量放在后侧
      }
      // 确定交错的对称性
      auto delete_1 = std::map<Name, std::map<Symmetry, Size>>();
      auto delete_2 = std::map<Name, std::map<Symmetry, Size>>();
      for (Rank i = 0; i < common_rank; i++) {
         auto name_1 = common_name_1[i];
         auto name_2 = common_name_2[i];
         auto edge_1 = tensor_1.core->edges[tensor_1.name_to_index.at(name_1)];
         auto edge_2 = tensor_2.core->edges[tensor_2.name_to_index.at(name_2)];
         auto this_delete_1 = std::map<Symmetry, Size>();
         for (const auto& [symmetry_1, dimension_1] : edge_1.map) {
            auto symmetry_2 = -symmetry_1;
            auto found = edge_2.map.find(symmetry_2);
            if (found != edge_2.map.end()) {
               // found
               if (found->second != dimension_1) {
                  warning_or_error("Different Dimension to Contract");
               }
            } else {
               // not found
               this_delete_1[symmetry_1] = 0;
            }
         }
         if (!this_delete_1.empty()) {
            delete_1[name_1] = std::move(this_delete_1);
         }
         auto this_delete_2 = std::map<Symmetry, Size>();
         for (const auto& [symmetry_2, dimension_2] : edge_2.map) {
            auto symmetry_1 = -symmetry_2;
            auto found = edge_1.map.find(symmetry_1);
            if (found != edge_1.map.end()) {
               // found
               if (found->second != dimension_2) {
                  warning_or_error("Different Dimension to Contract");
               }
            } else {
               // not found
               this_delete_2[symmetry_2] = 0;
            }
         }
         if (!this_delete_2.empty()) {
            delete_2[name_2] = std::move(this_delete_2);
         }
      }
      // merge
      // 仅对第一个张量的公共边的reverse和merge做符号
      auto tensor_1_merged = tensor_1.edge_operator(
            {},
            {},
            reversed_set_1,
            {{Contract1, free_name_1}, {Contract2, common_name_1}},
            put_common_1_right ? std::vector<Name>{Contract1, Contract2} : std::vector<Name>{Contract2, Contract1},
            false,
            {{{}, std::set<Name>(common_name_1.begin(), common_name_1.end()), {}, {Contract2}}},
            delete_1);
      auto tensor_2_merged = tensor_2.edge_operator(
            {},
            {},
            reversed_set_2,
            {{Contract2, free_name_2}, {Contract1, common_name_2}},
            put_common_2_right ? std::vector<Name>{Contract2, Contract1} : std::vector<Name>{Contract1, Contract2},
            false,
            {{{}, {}, {}, {}}},
            delete_2);
#if 0
      std::cout << "T1:" << tensor_1 << "\n";
      std::cout << "M1:" << tensor_1_merged << "\n";
      std::cout << "T2:" << tensor_2 << "\n";
      std::cout << "M2:" << tensor_2_merged << "\n";
#endif
      // calculate_product
      auto product_result = Tensor<ScalarType, Symmetry>(
            {Contract1, Contract2},
            {std::move(tensor_1_merged.core->edges[!put_common_1_right]), std::move(tensor_2_merged.core->edges[!put_common_2_right])});
      // 因取了T1和T2的edge，所以会自动去掉merge后仍然存在的交错边
      auto common_edge = std::move(tensor_1_merged.core->edges[put_common_1_right]);
      for (auto& [symmetries, data] : product_result.core->blocks) {
         // m k n
         auto symmetries_1 = put_common_1_right ? symmetries : std::vector<Symmetry>{symmetries[1], symmetries[0]};
         auto symmetries_2 = put_common_2_right ? std::vector<Symmetry>{symmetries[1], symmetries[0]} : symmetries;
         const auto& data_1 = tensor_1_merged.core->blocks.at(symmetries_1);
         const auto& data_2 = tensor_2_merged.core->blocks.at(symmetries_2);
         const int m = product_result.core->edges[0].map.at(symmetries[0]);
         const int n = product_result.core->edges[1].map.at(symmetries[1]);
         const int k = common_edge.map.at(symmetries[1]);
         ScalarType alpha = 1;
         if constexpr (is_fermi) {
            if ((put_common_2_right ^ !put_common_1_right) && bool(symmetries[0].fermi % 2)) {
#if 0
               std::cout << "R\n";
#endif
               alpha = -1;
            }
#if 0
            else {
               std::cout << "N\n";
            }
#endif
         }
         const ScalarType beta = 0;
         if (m * n * k != 0) {
            calculate_product<ScalarType>(
                  put_common_2_right ? "C" : "N",
                  put_common_1_right ? "N" : "C",
                  &n,
                  &m,
                  &k,
                  &alpha,
                  data_2.data(),
                  put_common_2_right ? &k : &n,
                  data_1.data(),
                  put_common_1_right ? &k : &m,
                  &beta,
                  data.data(),
                  &n);
         } else if (m * n != 0) {
            std::fill(data.begin(), data.end(), 0);
         }
      }
      if constexpr (is_no_symmetry) {
         auto result = Tensor<ScalarType, Symmetry>{std::move(name_result), std::move(edge_result)};
         result.core->blocks.begin()->second = std::move(product_result.core->blocks.begin()->second);
         return result;
      } else {
         auto result = product_result.edge_operator({}, split_map_result, reversed_set_result, {}, std::move(name_result));
#if 0
         std::cout << "R:" << product_result << "\n";
         std::cout << "S:" << result << "\n";
#endif
         return result;
      }
   }
} // namespace TAT
#endif
