/* TAT/Node/qr.hpp
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

#ifndef TAT_Node_Qr_HPP_
#define TAT_Node_Qr_HPP_

#include "../Node.hpp"

namespace TAT {
  namespace node {
    template<Device device, class Base>
    typename Node<device, Base>::qr_res Node<device, Base>::qr(const std::vector<Rank>& plan, const Rank& q_rank) const {
      qr_res res;
      Size q_size=1;
      std::vector<Size> tmp_dims;
      transpose::plan(tmp_dims, dims, plan);
      svd::plan(q_size, q_rank, tmp_dims);
      auto mid = tmp_dims.begin()+q_rank;
      Size r_size=data.size/q_size;
      Size min_size = (q_size<r_size)?q_size:r_size;
      auto data_res = data.qr(dims, plan, q_size, r_size, min_size);
      res.Q.dims.insert(res.Q.dims.end(), tmp_dims.begin(), mid);
      res.Q.dims.push_back(min_size);
      res.R.dims.push_back(min_size);
      res.R.dims.insert(res.R.dims.end(), mid, tmp_dims.end());
      res.Q.data = std::move(data_res.Q);
      res.R.data = std::move(data_res.R);
      return std::move(res);
    } // qr
  } // namespace node
} // namespace TAT

#endif // TAT_Node_Qr_HPP_
