/** TAT/Node/qr.hpp
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

#ifndef TAT_Node_Qr_HPP_
#define TAT_Node_Qr_HPP_

#include "../Node.hpp"

namespace TAT {
  namespace node {
    template<Device device, class Base>
    typename Node<device, Base>::qr_res Node<device, Base>::qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const {
      std::vector<Legs> q_legs = internal::in_and_in(legs, input_q_legs);
      qr_res res;
      std::vector<Legs> tmp_legs;
      std::vector<Rank> plan;
      Rank q_rank;
      svd::plan(res.Q.legs, res.R.legs, tmp_legs, q_rank, legs, q_legs, new_q_legs, new_r_legs);
      transpose::plan(plan, tmp_legs, legs);
      auto tensor_res = tensor.qr(plan, q_rank);
      res.Q.tensor = std::move(tensor_res.Q);
      res.R.tensor = std::move(tensor_res.R);
      return std::move(res);
    } // qr
  } // namespace node
} // namespace TAT

#endif // TAT_Node_Qr_HPP_
