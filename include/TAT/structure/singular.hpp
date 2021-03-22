/**
 * \file singular.hpp
 *
 * Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_SINGULAR_HPP
#define TAT_SINGULAR_HPP

namespace TAT {
   /**
    * \defgroup Singular
    * @{
    */

   /**
    * 张量看成矩阵后做svd分解后得到的奇异值类型, 为对角矩阵形式的张量
    *
    * \see Tensor::svd
    */
   template<is_scalar ScalarType = double, is_symmetry Symmetry = Symmetry<>, is_name Name = DefaultName>
   struct Singular {
      using scalar_vector_t = std::vector<real_scalar<ScalarType>>;
      using singular_map = std::map<Symmetry, scalar_vector_t>;
      singular_map value;

      template<int p>
      [[nodiscard]] real_scalar<ScalarType> norm() const {
         if constexpr (p == -1) {
            real_scalar<ScalarType> maximum = 0;
            for (const auto& [symmetry, singulars] : value) {
               for (const auto& element : singulars) {
                  auto absolute = std::abs(element);
                  maximum = maximum < absolute ? absolute : maximum;
               }
            }
            return maximum;
         } else if constexpr (p == 1) {
            real_scalar<ScalarType> summation = 0;
            for (const auto& [symmetry, singulars] : value) {
               for (const auto& element : singulars) {
                  auto absolute = std::abs(element);
                  summation += absolute;
               }
            }
            return summation;
         } else {
            TAT_error("Not Implement For Singulars Normalize Kind, Only +1 and -1 supported now");
            return 0;
         }
      }

      [[nodiscard]] std::string show() const;
      [[nodiscard]] std::string dump() const;
      Singular<ScalarType, Symmetry, Name>& load(const std::string&) &;
      Singular<ScalarType, Symmetry, Name>&& load(const std::string& string) && {
         return std::move(load(string));
      };

      [[nodiscard]] Singular<ScalarType, Symmetry, Name> copy() const {
         return Singular<ScalarType, Symmetry, Name>{value};
      }
   };
} // namespace TAT
#endif
