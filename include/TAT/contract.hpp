/**
 * \file contract.hpp
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
#ifndef TAT_CONTRACT_HPP
#define TAT_CONTRACT_HPP

#include "tensor.hpp"

extern "C" {
void sgemm_(
      const char* transA,
      const char* transB,
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
void dgemm_(
      const char* transA,
      const char* transB,
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
void cgemm_(
      const char* transA,
      const char* transB,
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
void zgemm_(
      const char* transA,
      const char* transB,
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
         const char* transA,
         const char* transB,
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
         const char* transA,
         const char* transB,
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
      sgemm_(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }
   template<>
   inline void calculate_product<double>(
         const char* transA,
         const char* transB,
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
      dgemm_(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }
   template<>
   inline void calculate_product<std::complex<float>>(
         const char* transA,
         const char* transB,
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
      cgemm_(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }
   template<>
   inline void calculate_product<std::complex<double>>(
         const char* transA,
         const char* transB,
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
      zgemm_(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }

   template<class ScalarType, class Symmetry>
   Tensor<ScalarType, Symmetry> Tensor<ScalarType, Symmetry>::contract(
         const Tensor<ScalarType, Symmetry>& tensor1,
         const Tensor<ScalarType, Symmetry>& tensor2,
         const vector<Name>& names1,
         const vector<Name>& names2) {
      // 为未来split做准备
      constexpr bool is_fermi = is_fermi_symmetry_v<Symmetry>;
      constexpr bool is_no_sym = std::is_same_v<Symmetry, NoSymmetry>;
      auto rank1 = tensor1.names.size();
      auto rank2 = tensor2.names.size();
      // 需要反转成 - + - -
      // 事后恢复两侧的边
      auto reversed_set1 = std::set<Name>(); // 第一个张量merge时反转表
      auto reversed_set2 = std::set<Name>(); // 第二个张量merge时反转表
      auto res_edge = vector<Edge<Symmetry>>(); // 无对称性的时候不需要split方案直接获取最后的edge
      auto split_map = std::map<Name, vector<std::tuple<Name, BoseEdge<Symmetry>>>>(); // split方案
      auto reversed_set = std::set<Name>(); // 最后split时的反转标
      auto res_name = vector<Name>();       // 最后split后的name
      split_map[Contract1];
      split_map[Contract2];
      auto free_name1 = vector<Name>(); // 第一个张量的自由边, merge时使用
      for (auto i = 0; i < rank1; i++) {
         const auto& n = tensor1.names[i];
         if (std::find(names1.begin(), names1.end(), n) == names1.end()) {
            free_name1.push_back(n);
            if constexpr (is_no_sym) {
               res_edge.push_back(tensor1.core->edges[i]);
            } else {
               split_map.at(Contract1).push_back({n, {tensor1.core->edges[i].map}});
            }
            res_name.push_back(n);
            if constexpr (is_fermi) {
               if (tensor1.core->edges[i].arrow) {
                  reversed_set1.insert(n);
                  reversed_set.insert(n);
               }
            }
         } else {
            if constexpr (is_fermi) {
               if (!tensor1.core->edges[i].arrow) {
                  reversed_set1.insert(n);
               }
            }
         }
      }
      const auto free_rank1 = free_name1.size();
      auto free_name2 = vector<Name>(); // 第二个张量的自由边, merge时使用
      for (auto i = 0; i < rank2; i++) {
         const auto& n = tensor2.names[i];
         if (std::find(names2.begin(), names2.end(), n) == names2.end()) {
            free_name2.push_back(n);
            if constexpr (is_no_sym) {
               res_edge.push_back(tensor2.core->edges[i]);
            } else {
               split_map.at(Contract2).push_back({n, {tensor2.core->edges[i].map}});
            }
            res_name.push_back(n);
            if constexpr (is_fermi) {
               if (tensor2.core->edges[i].arrow) {
                  reversed_set2.insert(n);
                  reversed_set.insert(n);
               }
            }
         } else {
            if constexpr (is_fermi) {
               if (tensor2.core->edges[i].arrow) {
                  reversed_set2.insert(n);
               }
            }
         }
      }
      const auto free_rank2 = free_name2.size();
      // 确定转置方案
      auto common_name1 = vector<Name>(); // 第一个张量的公共边, merge时使用
      auto common_name2 = vector<Name>(); // 第二个张量的公共边, merge时使用
      bool put_right1;
      bool put_right2;
      auto fit_tensor1 = [&]() {
         for (const auto& n : tensor1.names) {
            auto pos = std::find(names1.begin(), names1.end(), n);
            if (pos != names1.end()) {
               common_name1.push_back(*pos);
               common_name2.push_back(names2[pos - names1.begin()]);
            }
         }
      };
      auto fit_tensor2 = [&]() {
         for (const auto& n : tensor2.names) {
            auto pos = std::find(names2.begin(), names2.end(), n);
            if (pos != names2.end()) {
               common_name2.push_back(*pos);
               common_name1.push_back(names1[pos - names2.begin()]);
            }
         }
      };
      // 确定方案
      if (free_rank1 == 0) {
         put_right1 = true;
         fit_tensor2();
         put_right2 = common_name2.back() == tensor2.names.back();
      } else if (free_rank2 == 0) {
         put_right2 = true;
         fit_tensor1();
         put_right1 = common_name1.back() == tensor1.names.back();
      } else if (free_name1.back() != tensor1.names.back()) {
         put_right1 = true;
         fit_tensor1();
         put_right2 = common_name2.back() == tensor2.names.back();
      } else if (free_name2.back() != tensor2.names.back()) {
         put_right2 = true;
         fit_tensor2();
         put_right1 = common_name1.back() == tensor1.names.back();
      } else {
         put_right1 = false;
         put_right2 = false;
         fit_tensor2();
         // 所以尽量大张量放在后侧
      }
      // merge
      // 仅对第一个张量的公共边的reverse和merge做符号
      auto tensor1_merged = tensor1.edge_operator(
            {},
            {},
            reversed_set1,
            {{Contract1, free_name1}, {Contract2, common_name1}},
            put_right1 ? vector<Name>{Contract1, Contract2} : vector<Name>{Contract2, Contract1},
            false,
            {{{}, std::set<Name>(common_name1.begin(), common_name1.end()), {}, {Contract2}}});
      auto tensor2_merged = tensor2.edge_operator(
            {},
            {},
            reversed_set2,
            {{Contract2, free_name2}, {Contract1, common_name2}},
            put_right2 ? vector<Name>{Contract2, Contract1} : vector<Name>{Contract1, Contract2});
      // calculate_product
      auto product_res = Tensor<ScalarType, Symmetry>(
            {Contract1, Contract2},
            {std::move(tensor1_merged.core->edges[!put_right1]),
             std::move(tensor2_merged.core->edges[!put_right2])});
      auto common_edge = std::move(tensor1_merged.core->edges[put_right1]);
      for (auto& [sym, data] : product_res.core->blocks) {
         // m k n
         auto sym1 = put_right1 ? sym : vector<Symmetry>{sym[1], sym[0]};
         auto sym2 = put_right2 ? vector<Symmetry>{sym[1], sym[0]} : sym;
         const auto& data1 = tensor1_merged.core->blocks.at(sym1);
         const auto& data2 = tensor2_merged.core->blocks.at(sym2);
         const int m = product_res.core->edges[0].map.at(sym[0]);
         const int n = product_res.core->edges[1].map.at(sym[1]);
         const int k = common_edge.map.at(sym[1]);
         const ScalarType alpha = 1;
         const ScalarType beta = 0;
         calculate_product<ScalarType>(
               put_right2 ? "C" : "N",
               put_right1 ? "N" : "C",
               &n,
               &m,
               &k,
               &alpha,
               data2.data(),
               put_right2 ? &k : &n,
               data1.data(),
               put_right1 ? &k : &m,
               &beta,
               data.data(),
               &n);
      }
      if constexpr (is_no_sym) {
         auto res = Tensor<ScalarType, Symmetry>{std::move(res_name), std::move(res_edge)};
         res.core->blocks.begin()->second = std::move(product_res.core->blocks.begin()->second);
         return res;
      } else {
         return product_res.edge_operator({}, split_map, reversed_set, {}, std::move(res_name));
      }
   }
} // namespace TAT
#endif
