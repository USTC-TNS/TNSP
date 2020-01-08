
/**
 * \file scalar.hpp
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
#ifndef TAT_SCALAR_HPP
#define TAT_SCALAR_HPP

#include "tensor.hpp"

namespace TAT {
#define DEF_SCALAR_OP(OP, EVAL1, EVAL2, EVAL3)                                                 \
   template<class ScalarType1, class ScalarType2, class Symmetry>                              \
   auto OP(const Tensor<ScalarType1, Symmetry>& t1, const Tensor<ScalarType2, Symmetry>& t2) { \
      using ScalarType = std::common_type_t<ScalarType1, ScalarType2>;                         \
      if (t1.names.empty()) {                                                                  \
         const auto& x = t1.core->blocks[0].raw_data[0];                                       \
         auto res = Tensor<ScalarType, Symmetry>{t2.names, t2.core->edges};                    \
         for (Nums i = 0; i < res.core->blocks.size(); i++) {                                  \
            const ScalarType2* __restrict b = t2.core->blocks[i].raw_data.data();              \
            ScalarType* __restrict c = res.core->blocks[i].raw_data.data();                    \
            for (Size j = 0; j < res.core->blocks[i].size; j++) {                              \
               EVAL1;                                                                          \
            }                                                                                  \
         }                                                                                     \
         return res;                                                                           \
      } else if (t2.names.empty()) {                                                           \
         const auto& y = t2.core->blocks[0].raw_data[0];                                       \
         auto res = Tensor<ScalarType, Symmetry>{t1.names, t1.core->edges};                    \
         for (Nums i = 0; i < res.core->blocks.size(); i++) {                                  \
            const ScalarType1* __restrict a = t1.core->blocks[i].raw_data.data();              \
            ScalarType* __restrict c = res.core->blocks[i].raw_data.data();                    \
            for (Size j = 0; j < res.core->blocks[i].size; j++) {                              \
               EVAL2;                                                                          \
            }                                                                                  \
         }                                                                                     \
         return res;                                                                           \
      } else {                                                                                 \
         if (!((t1.names == t2.names) && (t1.core->edges == t2.core->edges))) {                \
            TAT_WARNING("Scalar Operator In Different Shape Tensor");                          \
         }                                                                                     \
         auto res = Tensor<ScalarType, Symmetry>{t1.names, t1.core->edges};                    \
         for (Nums i = 0; i < res.core->blocks.size(); i++) {                                  \
            const ScalarType1* __restrict a = t1.core->blocks[i].raw_data.data();              \
            const ScalarType2* __restrict b = t2.core->blocks[i].raw_data.data();              \
            ScalarType* __restrict c = res.core->blocks[i].raw_data.data();                    \
            for (Size j = 0; j < res.core->blocks[i].size; j++) {                              \
               EVAL3;                                                                          \
            }                                                                                  \
         }                                                                                     \
         return res;                                                                           \
      }                                                                                        \
   }                                                                                           \
   template<                                                                                   \
         class ScalarType1,                                                                    \
         class ScalarType2,                                                                    \
         class Symmetry,                                                                       \
         class = std::enable_if_t<is_scalar_v<ScalarType2>>>                                   \
   auto OP(const Tensor<ScalarType1, Symmetry>& t1, const ScalarType2& n2) {                   \
      return OP(t1, Tensor<ScalarType2, Symmetry>{n2});                                        \
   }                                                                                           \
   template<                                                                                   \
         class ScalarType1,                                                                    \
         class ScalarType2,                                                                    \
         class Symmetry,                                                                       \
         class = std::enable_if_t<is_scalar_v<ScalarType1>>>                                   \
   auto OP(const ScalarType1& n1, const Tensor<ScalarType2, Symmetry>& t2) {                   \
      return OP(Tensor<ScalarType1, Symmetry>{n1}, t2);                                        \
   }

   DEF_SCALAR_OP(operator+, c[j] = x + b[j], c[j] = a[j] + y, c[j] = a[j] + b[j])
   DEF_SCALAR_OP(operator-, c[j] = x - b[j], c[j] = a[j] - y, c[j] = a[j] - b[j])
   DEF_SCALAR_OP(operator*, c[j] = x* b[j], c[j] = a[j]* y, c[j] = a[j]* b[j])
   DEF_SCALAR_OP(operator/, c[j] = x / b[j], c[j] = a[j] / y, c[j] = a[j] / b[j])
#undef DEF_SCALAR_OP

#define DEF_SCALAR_OP(OP, EVAL1, EVAL2)                                                          \
   template<class ScalarType1, class ScalarType2, class Symmetry>                                \
   Tensor<ScalarType1, Symmetry>& OP(                                                            \
         Tensor<ScalarType1, Symmetry>& t1, const Tensor<ScalarType2, Symmetry>& t2) {           \
      if (t1.core.use_count() != 1) {                                                            \
         TAT_WARNING("Inplace Operator On Tensor Shared");                                       \
      }                                                                                          \
      if (t2.names.empty()) {                                                                    \
         const auto& y = t2.core->blocks[0].raw_data[0];                                         \
         for (Nums i = 0; i < t1.core->blocks.size(); i++) {                                     \
            ScalarType1* __restrict a = t1.core->blocks[i].raw_data.data();                      \
            for (Size j = 0; j < t1.core->blocks[i].size; j++) {                                 \
               EVAL1;                                                                            \
            }                                                                                    \
         }                                                                                       \
      } else {                                                                                   \
         if (!((t1.names == t2.names) && (t1.core->edges == t2.core->edges))) {                  \
            TAT_WARNING("Scalar Operator In Different Shape Tensor");                            \
         }                                                                                       \
         for (Nums i = 0; i < t1.core->blocks.size(); i++) {                                     \
            ScalarType1* __restrict a = t1.core->blocks[i].raw_data.data();                      \
            const ScalarType2* __restrict b = t2.core->blocks[i].raw_data.data();                \
            for (Size j = 0; j < t1.core->blocks[i].size; j++) {                                 \
               EVAL2;                                                                            \
            }                                                                                    \
         }                                                                                       \
      }                                                                                          \
      return t1;                                                                                 \
   }                                                                                             \
   template<                                                                                     \
         class ScalarType1,                                                                      \
         class ScalarType2,                                                                      \
         class Symmetry,                                                                         \
         class = std::enable_if_t<is_scalar_v<ScalarType2>>>                                     \
   Tensor<ScalarType1, Symmetry>& OP(Tensor<ScalarType1, Symmetry>& t1, const ScalarType2& n2) { \
      return OP(t1, Tensor<ScalarType2, Symmetry>{n2});                                          \
   }
   DEF_SCALAR_OP(operator+=, a[j] += y, a[j] += b[j])
   DEF_SCALAR_OP(operator-=, a[j] -= y, a[j] -= b[j])
   DEF_SCALAR_OP(operator*=, a[j] *= y, a[j] *= b[j])
   DEF_SCALAR_OP(operator/=, a[j] /= y, a[j] /= b[j])
#undef DEF_SCALAR_OP
} // namespace TAT
#endif
