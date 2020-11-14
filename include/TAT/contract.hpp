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
#include "timer.hpp"

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
   template<typename ScalarType>
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
      auto kernel_guard = contract_kernel_guard();
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
      auto kernel_guard = contract_kernel_guard();
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
      auto kernel_guard = contract_kernel_guard();
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
      auto kernel_guard = contract_kernel_guard();
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

   template<typename ScalarType>
   Tensor<ScalarType, NoSymmetry> contract_with_fuse(
         const Tensor<ScalarType, NoSymmetry>& tensor_1,
         const Tensor<ScalarType, NoSymmetry>& tensor_2,
         std::set<std::tuple<Name, Name>> contract_names);

   template<typename ScalarType, typename Symmetry>
   Tensor<ScalarType, Symmetry> contract_without_fuse(
         const Tensor<ScalarType, Symmetry>& tensor_1,
         const Tensor<ScalarType, Symmetry>& tensor_2,
         std::set<std::tuple<Name, Name>> contract_names);

   template<typename ScalarType, typename Symmetry>
   Tensor<ScalarType, Symmetry> Tensor<ScalarType, Symmetry>::contract(
         const Tensor<ScalarType, Symmetry>& tensor_1,
         const Tensor<ScalarType, Symmetry>& tensor_2,
         std::set<std::tuple<Name, Name>> contract_names) {
      auto guard = contract_guard();
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         return contract_with_fuse(tensor_1, tensor_2, std::move(contract_names));
      } else {
         return contract_without_fuse(tensor_1, tensor_2, std::move(contract_names));
      }
   }

   template<typename ScalarType, typename Symmetry>
   Tensor<ScalarType, Symmetry> contract_without_fuse(
         const Tensor<ScalarType, Symmetry>& tensor_1,
         const Tensor<ScalarType, Symmetry>& tensor_2,
         std::set<std::tuple<Name, Name>> contract_names) {
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
      constexpr bool is_no_symmetry = std::is_same_v<Symmetry, NoSymmetry>;
      // 为未来split做准备
      const Rank rank_1 = tensor_1.names.size();
      const Rank rank_2 = tensor_2.names.size();
      // 删除不存在的名称, 即在name tuple list中但不在names中
      for (auto iterator = contract_names.begin(); iterator != contract_names.end();) {
         auto found_1 = tensor_1.name_to_index.find(std::get<0>(*iterator));
         auto found_2 = tensor_2.name_to_index.find(std::get<1>(*iterator));
         if (found_1 == tensor_1.name_to_index.end() || found_2 == tensor_2.name_to_index.end()) {
            iterator = contract_names.erase(iterator);
         } else {
            ++iterator;
         }
      }
      const auto common_rank = contract_names.size();
      std::set<Name> contract_names_1;
      std::set<Name> contract_names_2;
      for (const auto& [name_1, name_2] : contract_names) {
         contract_names_1.insert(name_1);
         contract_names_2.insert(name_2);
      }
      if (contract_names_1.size() != common_rank || contract_names_2.size() != common_rank) {
         TAT_error("Duplicated Contract Name");
      }
      // Rank contract_origin_rank = contract_names_1.size();
      // for (Rank i = 0; i < contract_origin_rank; i++) {
      // }
      // 需要反转成 - + - -
      // 事后恢复两侧的边
      auto reversed_set_1 = std::set<Name>();           // 第一个张量merge时反转表
      auto reversed_set_2 = std::set<Name>();           // 第二个张量merge时反转表
      auto edge_result = std::vector<Edge<Symmetry>>(); // 无对称性的时候不需要split方案直接获取最后的edge
      if constexpr (is_no_symmetry) {
         edge_result.reserve(rank_1 + rank_2 - 2 * common_rank);
      }
      auto split_map_result = std::map<Name, std::vector<std::tuple<Name, BoseEdge<Symmetry>>>>(); // split方案
      auto reversed_set_result = std::set<Name>();                                                 // 最后split时的反转标
      auto name_result = std::vector<Name>();                                                      // 最后split后的name
      name_result.reserve(rank_1 + rank_2 - 2 * common_rank);
      split_map_result[internal_name::Contract_1].reserve(rank_1 - common_rank);
      split_map_result[internal_name::Contract_2].reserve(rank_2 - common_rank);
      auto free_name_1 = std::vector<Name>(); // 第一个张量的自由边, merge时使用
      free_name_1.reserve(rank_1 - common_rank);
      for (Rank i = 0; i < rank_1; i++) {
         const auto& n = tensor_1.names[i];
         if (find_in_contract_names<0>(contract_names, n) == contract_names.end()) {
            // 是free name
            free_name_1.push_back(n);
            if constexpr (is_no_symmetry) {
               edge_result.push_back(tensor_1.core->edges[i]);
            } else {
               split_map_result.at(internal_name::Contract_1).push_back({n, {tensor_1.core->edges[i].map}});
            }
            name_result.push_back(n);
            if constexpr (is_fermi) {
               if (tensor_1.core->edges[i].arrow) {
                  reversed_set_1.insert(n);
                  reversed_set_result.insert(n);
               }
            }
         } else {
            // 将被contract掉
            if constexpr (is_fermi) {
               if (!tensor_1.core->edges[i].arrow) {
                  reversed_set_1.insert(n);
               }
            }
         }
      }
      const auto free_rank_1 = free_name_1.size();
      auto free_name_2 = std::vector<Name>(); // 第二个张量的自由边, merge时使用
      free_name_2.reserve(rank_2 - common_rank);
      for (Rank i = 0; i < rank_2; i++) {
         const auto& n = tensor_2.names[i];
         if (find_in_contract_names<1>(contract_names, n) == contract_names.end()) {
            // 是free name
            free_name_2.push_back(n);
            if constexpr (is_no_symmetry) {
               edge_result.push_back(tensor_2.core->edges[i]);
            } else {
               split_map_result.at(internal_name::Contract_2).push_back({n, {tensor_2.core->edges[i].map}});
            }
            name_result.push_back(n);
            if constexpr (is_fermi) {
               if (tensor_2.core->edges[i].arrow) {
                  reversed_set_2.insert(n);
                  reversed_set_result.insert(n);
               }
            }
         } else {
            // 将被contract掉
            if constexpr (is_fermi) {
               if (tensor_2.core->edges[i].arrow) {
                  reversed_set_2.insert(n);
               }
            }
         }
      }
      const auto free_rank_2 = free_name_2.size();
      // 确定转置方案
      auto common_name_1 = std::vector<Name>(); // 第一个张量的公共边, merge时使用
      auto common_name_2 = std::vector<Name>(); // 第二个张量的公共边, merge时使用
      common_name_1.reserve(common_rank);
      common_name_2.reserve(common_rank);
      bool put_common_1_right;
      bool put_common_2_right;
      auto fit_tensor_1_common_edge = [&]() {
         for (const auto& n : tensor_1.names) {
            if (auto position = find_in_contract_names<0>(contract_names, n); position != contract_names.end()) {
               common_name_1.push_back(std::get<0>(*position));
               common_name_2.push_back(std::get<1>(*position));
            }
         }
      };
      auto fit_tensor_2_common_edge = [&]() {
         for (const auto& n : tensor_2.names) {
            if (auto position = find_in_contract_names<1>(contract_names, n); position != contract_names.end()) {
               common_name_1.push_back(std::get<0>(*position));
               common_name_2.push_back(std::get<1>(*position));
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
      if constexpr (!is_no_symmetry) {
         for (Rank i = 0; i < common_rank; i++) {
            auto name_1 = common_name_1[i];
            auto name_2 = common_name_2[i];
            auto edge_1 = tensor_1.core->edges[tensor_1.name_to_index.at(name_1)];
            auto edge_2 = tensor_2.core->edges[tensor_2.name_to_index.at(name_2)];
            auto delete_unused_dimension = [](const auto& edge_this, const auto& edge_other, const auto& name_this, auto& delete_this) {
#ifndef _MSVC_LANG
               // 2020.10.27 现在版本的msvc认为constexpr需要捕获, 等待新版本msvc支持此功能, 才可以删掉这个判断
               if constexpr (is_fermi) {
                  if (edge_this.arrow == edge_other.arrow) {
                     TAT_error("Different Fermi Arrow to Contract");
                  }
               }
#endif
               auto delete_map = std::map<Symmetry, Size>();
               for (const auto& [symmetry, dimension] : edge_this.map) {
                  auto found = edge_other.map.find(-symmetry);
                  if (found != edge_other.map.end()) {
                     // found
                     if (const auto dimension_other = found->second; dimension_other != dimension) {
                        TAT_error("Different Dimension to Contract");
                     }
                  } else {
                     // not found
                     delete_map[symmetry] = 0;
                     // 用于merge时cut, cut成0会自动删除
                  }
               }
               if (!delete_map.empty()) {
                  delete_this[name_this] = std::move(delete_map);
               }
            };
            delete_unused_dimension(edge_1, edge_2, name_1, delete_1);
            delete_unused_dimension(edge_2, edge_1, name_2, delete_2);
         }
      }
      // merge
      // 仅对第一个张量的公共边的reverse和merge做符号
      auto tensor_1_merged = tensor_1.edge_operator(
            {},
            {},
            reversed_set_1,
            {{internal_name::Contract_1, free_name_1}, {internal_name::Contract_2, common_name_1}},
            put_common_1_right ? std::vector<Name>{internal_name::Contract_1, internal_name::Contract_2} :
                                 std::vector<Name>{internal_name::Contract_2, internal_name::Contract_1},
            false,
            {{{}, std::set<Name>(common_name_1.begin(), common_name_1.end()), {}, {internal_name::Contract_2}}},
            delete_1);
      auto tensor_2_merged = tensor_2.edge_operator(
            {},
            {},
            reversed_set_2,
            {{internal_name::Contract_2, free_name_2}, {internal_name::Contract_1, common_name_2}},
            put_common_2_right ? std::vector<Name>{internal_name::Contract_2, internal_name::Contract_1} :
                                 std::vector<Name>{internal_name::Contract_1, internal_name::Contract_2},
            false,
            {{{}, {}, {}, {}}},
            delete_2);
      // calculate_product
      auto product_result = Tensor<ScalarType, Symmetry>(
            {internal_name::Contract_1, internal_name::Contract_2},
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
            // 因为并非标准- + - -产生的符号
            if ((put_common_2_right ^ !put_common_1_right) && bool(symmetries[0].fermi % 2)) {
               alpha = -1;
            }
         }
         const ScalarType beta = 0;
         if (m * n * k != 0) {
            calculate_product<ScalarType>(
                  put_common_2_right ? "T" : "N",
                  put_common_1_right ? "N" : "T",
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
         return result;
      }
   }

   template<typename ScalarType>
   Tensor<ScalarType, NoSymmetry> contract_with_fuse(
         const Tensor<ScalarType, NoSymmetry>& tensor_1,
         const Tensor<ScalarType, NoSymmetry>& tensor_2,
         std::set<std::tuple<Name, Name>> contract_names) {
      const Rank rank_1 = tensor_1.names.size();
      const Rank rank_2 = tensor_2.names.size();
      // 删除不存在的名称, 即在name tuple list中但不在names中
      for (auto iterator = contract_names.begin(); iterator != contract_names.end();) {
         auto found_1 = tensor_1.name_to_index.find(std::get<0>(*iterator));
         auto found_2 = tensor_2.name_to_index.find(std::get<1>(*iterator));
         if (found_1 == tensor_1.name_to_index.end() || found_2 == tensor_2.name_to_index.end()) {
            iterator = contract_names.erase(iterator);
         } else {
            ++iterator;
         }
      }
      const auto common_rank = contract_names.size();
      std::set<Name> contract_names_1;
      std::set<Name> contract_names_2;
      for (const auto& [name_1, name_2] : contract_names) {
         contract_names_1.insert(name_1);
         contract_names_2.insert(name_2);
      }
      if (contract_names_1.size() != common_rank || contract_names_2.size() != common_rank) {
         TAT_error("Duplicated Contract Name");
      }
      // 确认fuse name即相同名称的边
      std::set<Name> fuse_names;
      for (const auto& name : tensor_1.names) {
         const auto in_tensor_2 = tensor_2.name_to_index.find(name) != tensor_2.name_to_index.end();
         const auto in_contract_1 = contract_names_1.find(name) != contract_names_1.end();
         const auto in_contract_2 = contract_names_2.find(name) != contract_names_2.end();
         if (in_tensor_2 && !in_contract_1 && !in_contract_2) {
            fuse_names.insert(name);
         }
      }
      const auto fuse_rank = fuse_names.size();
      // 准备方案
      auto edge_result = std::vector<Edge<NoSymmetry>>(); // 无对称性的时候不需要split方案直接获取最后的edge
      edge_result.reserve(rank_1 + rank_2 - 2 * common_rank - fuse_rank);
      auto name_result = std::vector<Name>(); // 最后split后的name
      name_result.reserve(rank_1 + rank_2 - 2 * common_rank - fuse_rank);

      // 首先安排fuse name到结果的最前面, 这里并没有考虑顺序，也许这样不好，但是维度很大应该没问题
      std::vector<Name> fuse_names_list;
      fuse_names_list.reserve(fuse_rank);
      for (const auto& name : fuse_names) {
         name_result.push_back(name);
         fuse_names_list.push_back(name);
         const auto& edge_1 = tensor_1.core->edges[tensor_1.name_to_index.at(name)];
         const auto& edge_2 = tensor_2.core->edges[tensor_2.name_to_index.at(name)];
         if (!(edge_1 == edge_2)) {
            TAT_error("Cannot fuse two edge with different shape");
         }
         edge_result.push_back(edge_1);
      }

      auto free_name_1 = std::vector<Name>(); // 第一个张量的自由边, merge时使用
      free_name_1.reserve(rank_1 - common_rank - fuse_rank);
      for (Rank i = 0; i < rank_1; i++) {
         const auto& n = tensor_1.names[i];
         if (find_in_contract_names<0>(contract_names, n) == contract_names.end()) {
            // 是free name或fuse name
            if (fuse_names.find(n) == fuse_names.end()) {
               // 不是fuse name, 是free name
               free_name_1.push_back(n);
               edge_result.push_back(tensor_1.core->edges[i]);
               name_result.push_back(n);
            }
         }
      }
      const auto free_rank_1 = free_name_1.size();
      auto free_name_2 = std::vector<Name>(); // 第二个张量的自由边, merge时使用
      free_name_2.reserve(rank_2 - common_rank);
      for (Rank i = 0; i < rank_2; i++) {
         const auto& n = tensor_2.names[i];
         if (find_in_contract_names<1>(contract_names, n) == contract_names.end()) {
            // 是free name或fuse name
            if (fuse_names.find(n) == fuse_names.end()) {
               // 不是fuse name, 是free name
               free_name_2.push_back(n);
               edge_result.push_back(tensor_2.core->edges[i]);
               name_result.push_back(n);
            }
         }
      }
      const auto free_rank_2 = free_name_2.size();
      // 确定转置方案
      auto common_name_1 = std::vector<Name>(); // 第一个张量的公共边, merge时使用
      auto common_name_2 = std::vector<Name>(); // 第二个张量的公共边, merge时使用
      common_name_1.reserve(common_rank);
      common_name_2.reserve(common_rank);
      bool put_common_1_right;
      bool put_common_2_right;
      auto fit_tensor_1_common_edge = [&]() {
         for (const auto& n : tensor_1.names) {
            if (auto position = find_in_contract_names<0>(contract_names, n); position != contract_names.end()) {
               // 是contract name
               common_name_1.push_back(std::get<0>(*position));
               common_name_2.push_back(std::get<1>(*position));
            }
         }
      };
      auto fit_tensor_2_common_edge = [&]() {
         for (const auto& n : tensor_2.names) {
            if (auto position = find_in_contract_names<1>(contract_names, n); position != contract_names.end()) {
               // 是contract name
               common_name_1.push_back(std::get<0>(*position));
               common_name_2.push_back(std::get<1>(*position));
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

      // merge
      // 仅对第一个张量的公共边的reverse和merge做符号
      auto tensor_1_merged = tensor_1.edge_operator(
            {},
            {},
            {},
            {{internal_name::Contract_1, free_name_1}, {internal_name::Contract_2, common_name_1}, {internal_name::Contract_0, fuse_names_list}},
            put_common_1_right ? std::vector<Name>{internal_name::Contract_0, internal_name::Contract_1, internal_name::Contract_2} :
                                 std::vector<Name>{internal_name::Contract_0, internal_name::Contract_2, internal_name::Contract_1});
      auto tensor_2_merged = tensor_2.edge_operator(
            {},
            {},
            {},
            {{internal_name::Contract_2, free_name_2}, {internal_name::Contract_1, common_name_2}, {internal_name::Contract_0, fuse_names_list}},
            put_common_2_right ? std::vector<Name>{internal_name::Contract_0, internal_name::Contract_2, internal_name::Contract_1} :
                                 std::vector<Name>{internal_name::Contract_0, internal_name::Contract_1, internal_name::Contract_2});
      // calculate_product
      auto product_result = Tensor<ScalarType, NoSymmetry>(
            {internal_name::Contract_0, internal_name::Contract_1, internal_name::Contract_2},
            {std::move(tensor_1_merged.core->edges[0]),
             std::move(tensor_1_merged.core->edges[1 + !put_common_1_right]),
             std::move(tensor_2_merged.core->edges[1 + !put_common_2_right])});

      auto common_edge = std::move(tensor_1_merged.core->edges[1 + put_common_1_right]);

      const int l = product_result.core->edges[0].map.begin()->second;
      const int m = product_result.core->edges[1].map.begin()->second;
      const int n = product_result.core->edges[2].map.begin()->second;
      const int k = common_edge.map.begin()->second;
      const ScalarType alpha = 1;
      const ScalarType beta = 0;

      ScalarType* data = product_result.core->blocks.begin()->second.data();
      const ScalarType* data_1 = tensor_1_merged.core->blocks.begin()->second.data();
      const ScalarType* data_2 = tensor_2_merged.core->blocks.begin()->second.data();
      if (m * n * k != 0) {
         for (auto i = 0; i < l; i++) {
            calculate_product<ScalarType>(
                  put_common_2_right ? "T" : "N",
                  put_common_1_right ? "N" : "T",
                  &n,
                  &m,
                  &k,
                  &alpha,
                  data_2 + k * n * i,
                  put_common_2_right ? &k : &n,
                  data_1 + m * k * i,
                  put_common_1_right ? &k : &m,
                  &beta,
                  data + m * n * i,
                  &n);
         }
      } else if (m * n != 0) {
         std::fill(data, data + m * n * l, 0);
      }

      auto result = Tensor<ScalarType, NoSymmetry>{std::move(name_result), std::move(edge_result)};
      result.core->blocks.begin()->second = std::move(product_result.core->blocks.begin()->second);
      return result;
   }
} // namespace TAT
#endif
