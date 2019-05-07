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
      void plan(std::vector<Rank>& plan, const std::vector<Legs>& new_legs, const std::vector<Legs>& legs) {
        const Rank& rank = legs.size();
        for (Rank i=0; i<rank; i++) {
          for (Rank j=0; j<rank; j++) {
            if (new_legs[i]==legs[j]) {
              plan.push_back(j);
              break;
            } // if
          } // for j
        } // for i
      } // plan
    } // namespace node::transpose

    template<Device device, class Base>
    std::shared_ptr<Node<device, Base>> Node<device, Base>::transpose(const std::vector<Legs>& new_legs) const {
      auto res=std::make_shared<Node<device, Base>>();
      res->legs = internal::in_and_in(new_legs, legs);
      assert(legs.size()==res->legs.size());
#ifndef NDEBUG
      auto set_new = std::set<Legs>(res->legs.begin(), res->legs.end());
      assert(set_new.size()==res->legs.size());
      set_new.insert(legs.begin(), legs.end());
      assert(set_new.size()==res->legs.size());
#endif // NDEBUG
      std::vector<Rank> plan;
      transpose::plan(plan, res->legs, legs);
      assert(res->legs.size()==legs.size());
      assert(plan.size()==legs.size());
      res->func = [&, plan](){return tensor.transpose(plan);};
      downstream.push_back(res);
      return res;
    } // transpose
  } // namespace node
} // namespace TAT

#endif // TAT_Node_Transpose_HPP_
