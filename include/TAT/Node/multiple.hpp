/** TAT/Node/multiple.hpp
 * @file
 * @author  Hao Zhang <zh970204@mail.ustc.edu.cn>
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
    template<Device device, class Base>
    Node<device, Base> Node<device, Base>::multiple(const Node<device, Base>& other, const Legs& position) const {
      Node<device, Base> res;
      assert(other.legs.size()==1);
      res.legs = legs;
      auto pos = std::find(legs.begin(), legs.end(), position);
      if (pos==legs.end()) {
        return *this;
      } // if not multiple
      Rank index = std::distance(legs.begin(), pos);
      res.tensor = tensor.multiple(other.tensor, index);
      return std::move(res);
    } // multiple
  } // namespace node
} // namespace TAT

#endif // TAT_Node_Multiple_HPP_
