/* TAT/Node/multiple.hpp
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

#ifndef TAT_Node_Multiple_HPP_
#define TAT_Node_Multiple_HPP_

#include "../Node.hpp"

namespace TAT {
  namespace node {
    namespace multiple {
      void plan(Size& a, Size& b, Size& c, const std::vector<Size>& dims, const Rank& index) {
        Rank i=0, rank=dims.size();
        for (; i<index; i++) {
          a *= dims[i];
        } // for i
        b = dims[i];
        i++;
        for (; i<rank; i++) {
          c *= dims[i];
        } // for
      } // plan
    } // namespace node::multiple

    template<Device device, class Base>
    Node<device, Base> Node<device, Base>::multiple(const Node<device, Base>& other, const Rank& index) const {
      Node<device, Base> res;
      res.dims = dims;
      Size a=1, b=1, c=1;
      multiple::plan(a, b, c, dims, index);
      assert(other.dims.size()==1);
      assert(b==other.dims[0]);
      res.data = data.multiple(other.data, a, b, c);
      return std::move(res);
    } // multiple
  } // namespace node
} // namespace TAT

#endif // TAT_Node_Multple_HPP_
