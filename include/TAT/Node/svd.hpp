/** TAT/Node/svd.hpp
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

#ifndef TAT_Node_Svd_HPP_
#define TAT_Node_Svd_HPP_

#include "../Node.hpp"

namespace TAT {
  namespace node {
    namespace svd {
      void plan(std::vector<Legs>& U_legs,
                std::vector<Legs>& V_legs,
                std::vector<Legs>& tmp_legs,
                Rank& u_rank,
                const std::vector<Legs>& total_legs,
                const std::vector<Legs>& u_legs,
                const Legs& new_u_legs,
                const Legs& new_v_legs) {
        u_rank = u_legs.size();
        V_legs.push_back(new_v_legs);
        for (const auto& i : total_legs) {
          if (internal::in(i, u_legs)) {
            U_legs.push_back(i);
          } else {
            V_legs.push_back(i);
          } // if
        } // for
        U_legs.push_back(new_u_legs);
        tmp_legs.insert(tmp_legs.end(), U_legs.begin(), U_legs.end()-1);
        tmp_legs.insert(tmp_legs.end(), V_legs.begin()+1, V_legs.end());
      } // plan
    } // namespace node::svd

    template<Device device, class Base>
    typename Node<device, Base>::svd_res Node<device, Base>::svd(const std::vector<Legs>& input_u_legs, const Legs& new_u_legs, const Legs& new_v_legs, const Rank& cut) const {
      std::vector<Legs> u_legs = internal::in_and_in(legs, input_u_legs);
      svd_res res;
      std::vector<Legs> tmp_legs;
      std::vector<Rank> plan;
      Rank u_rank;
      svd::plan(res.U.legs, res.V.legs, tmp_legs, u_rank, legs, u_legs, new_u_legs, new_v_legs);
      transpose::plan(plan, tmp_legs, legs);
      auto tensor_res = tensor.svd(plan, u_rank, cut);
      res.S.legs = {new_u_legs};// new_u_legs or new_v_legs
      res.U.tensor = std::move(tensor_res.U);
      res.S.tensor = std::move(tensor_res.S);
      res.V.tensor = std::move(tensor_res.V);
      return std::move(res);
    } // svd
  } // namespace node
} // namespace TAT

#endif // TAT_Node_Svd_HPP_
