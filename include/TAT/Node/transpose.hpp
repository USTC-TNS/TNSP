/* TAT/Node/transpose.hpp
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

#ifndef TAT_Node_Transpose_HPP_
#define TAT_Node_Transpose_HPP_

#include "../Node.hpp"

namespace TAT {
  namespace node {
    namespace transpose {
      void plan(std::vector<Size>& new_dims, const std::vector<Size>& dims, const std::vector<Rank>& plan) {
        for (const auto& i : plan) {
          new_dims.push_back(dims[i]);
        } // for i
      } // plan
    } // namespace node::transpose

    template<Device device, class Base>
    Node<device, Base> Node<device, Base>::transpose(const std::vector<Rank>& plan) const {
      Node<device, Base> res;
      transpose::plan(res.dims, dims, plan);
      assert(plan.size()==dims.size());
      assert(get_size(res.dims)==data.size);
      res.data = data.transpose(dims, plan);
      return std::move(res);
    } // transpose
  } // namespace node
} // namespace TAT

#endif // TAT_Node_Transpose_HPP_
