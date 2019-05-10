/** TAT/Node/contract.hpp
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

#ifndef TAT_Node_Contract_HPP_
#define TAT_Node_Contract_HPP_

#include "../Node.hpp"

namespace TAT {
  namespace node {
    namespace contract {
      void plan(Rank& contract_num,
                std::vector<Legs>& legs,
                std::vector<Legs>& new_legs1,
                std::vector<Legs>& new_legs2,
                const std::vector<Legs>& total_legs1,
                const std::vector<Legs>& total_legs2,
                const std::vector<Legs>& legs1,
                const std::vector<Legs>& legs2,
                const std::map<Legs, Legs>& map1,
                const std::map<Legs, Legs>& map2) {
        auto filt_legs1 = internal::in_and_not_in(total_legs1, legs1);
        internal::append(new_legs1, filt_legs1);
        internal::append(legs, internal::map_or_not(filt_legs1, map1));

        auto tmp_legs1 = internal::in_and_in(legs1, total_legs1);
        internal::append(new_legs1, tmp_legs1);

        auto tmp_legs2 = internal::in_and_in(legs2, total_legs2);
        internal::append(new_legs2, tmp_legs2);

        auto filt_legs2 = internal::in_and_not_in(total_legs2, legs2);
        internal::append(new_legs2, filt_legs2);
        internal::append(legs, internal::map_or_not(filt_legs2, map2));

        assert(tmp_legs1.size()==tmp_legs2.size());
        contract_num = tmp_legs1.size();
      } // plan
    } // namespace node::contract

    template<Device device, class Base>
    Node<device, Base> Node<device, Base>::contract(const Node<device, Base>& node1,
        const Node<device, Base>& node2,
        const std::vector<Legs>& legs1,
        const std::vector<Legs>& legs2,
        const std::map<Legs, Legs>& map1,
        const std::map<Legs, Legs>& map2) {
      Node<device, Base> res;
      std::vector<Legs> new_legs1, new_legs2;
      std::vector<Rank> plan1, plan2;
      Rank contract_num;
      assert(legs1.size()==legs2.size());
      contract::plan(contract_num, res.legs, new_legs1, new_legs2, node1.legs, node2.legs, legs1, legs2, map1, map2);
      transpose::plan(plan1, new_legs1, node1.legs);
      transpose::plan(plan2, new_legs2, node2.legs);
      assert(new_legs1.size()==node1.legs.size());
      assert(plan1.size()==node1.legs.size());
      assert(new_legs2.size()==node2.legs.size());
      assert(plan2.size()==node2.legs.size());
      res.tensor = Tensor<device, Base>::contract(node1.tensor, node2.tensor, plan1, plan2, contract_num);
      return std::move(res);
    } // contract
  } // namespace node
} // namespace TAT

#endif // TAT_Node_Contract_HPP_
