/* TAT/Tensor/norm.hpp
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

#ifndef TAT_Tensor_Norm_HPP_
#define TAT_Tensor_Norm_HPP_

#include "../Tensor.hpp"

namespace TAT {
  namespace tensor {
    template<Device device, class Base>
    template<int n>
    Tensor<device, Base> Tensor<device, Base>::norm() const {
      Tensor<device, Base> res({}, {});
      res.node = node.template norm<n>();
      return std::move(res);
    } // norm
  } // namespace tensor
} // namespace TAT

#endif // TAT_Tensor_Norm_HPP_
