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

namespace TAT {
   template<class ScalarType, class Symmetry>
   typename Tensor<ScalarType, Symmetry>::svd_result Tensor<ScalarType, Symmetry>::svd(
         const std::set<Name>& u_edges,
         Name u_new_name,
         Name v_new_name) const {
      // merge, svd, split
      // TODO: svd
   }
} // namespace TAT
#endif
