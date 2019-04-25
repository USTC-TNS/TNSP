/* TAT/Node/contract.hpp
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

#ifndef TAT_Node_Contract_HPP_
#define TAT_Node_Contract_HPP_

#include "../Node.hpp"

namespace TAT {
  namespace node {
    namespace contract {
      void plan(std::vector<Size>& dims, Size& m, Size& k, Size& n,
                const::std::vector<Size>& dims1,
                const::std::vector<Size>& dims2,
                const std::vector<Rank>& plan1,
                const std::vector<Rank>& plan2,
                const Rank& contract_num) {
        Rank i, tmp=dims1.size()-contract_num, rank2=dims2.size();
        for (i=0; i<tmp; i++) {
          const Size& t = dims1[plan1[i]];
          m *= t;
          dims.push_back(t);
        } // for i
        for (i=0; i<contract_num; i++) {
          k *= dims1[plan1[i+tmp]];
          assert(dims1[plan1[i+tmp]]==dims2[plan2[i]]);
        } // for i
        for (; i<rank2; i++) {
          const Size& t = dims2[plan2[i]];
          n *= t;
          dims.push_back(t);
        } // for i
      } // plan
    } // namespace node::contract

    template<Device device, class Base>
    Node<device, Base> Node<device, Base>::contract(const Node<device, Base>& node1,
        const Node<device, Base>& node2,
        const std::vector<Rank>& plan1,
        const std::vector<Rank>& plan2,
        const Rank& contract_num) {
      Node<device, Base> res;
      Size m=1, k=1, n=1;
      contract::plan(res.dims, m, k, n, node1.dims, node2.dims, plan1, plan2, contract_num);
      res.data = Data<device, Base>::contract(node1.data, node2.data, node1.dims, node2.dims, plan1, plan2, m, k, n);
      return std::move(res);
    } // contract
  } // namespace node
} // namespace TAT

#endif // TAT_Node_Contract_HPP_
