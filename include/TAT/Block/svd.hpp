/* TAT/Block/svd.hpp
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

#ifndef TAT_Block_Svd_HPP_
#define TAT_Block_Svd_HPP_

#include "../Block.hpp"

namespace TAT {
  namespace block {
    namespace svd {
      void plan(Size& u_size, const Rank& u_rank, const std::vector<Size>& dims) {
        for (Rank i=0; i<u_rank; i++) {
          u_size *= dims[i];
        } // for i
      } // plan
    } // namespace block::svd

    template<Device device, class Base>
    typename Block<device, Base>::svd_res Block<device, Base>::svd(const std::vector<Rank>& plan, const Rank& u_rank, const Size& cut) const {
      svd_res res;
      Size u_size=1;
      std::vector<Size> tmp_dims;
      transpose::plan(tmp_dims, dims, plan);
      svd::plan(u_size, u_rank, tmp_dims);
      Size v_size = size()/u_size;
      Size min_mn = (u_size<v_size)?u_size:v_size;
      auto data_res = data.svd(dims, plan, u_size, v_size, min_mn, cut);
      auto mid = tmp_dims.begin()+u_rank;
      res.U.dims.insert(res.U.dims.end(), tmp_dims.begin(), mid);
      res.U.dims.push_back(data_res.S.size);
      res.S.dims.push_back(data_res.S.size);
      res.V.dims.push_back(data_res.S.size);
      res.V.dims.insert(res.V.dims.end(), mid, tmp_dims.end());
      res.U.data = std::move(data_res.U);
      res.S.data = std::move(data_res.S);
      res.V.data = std::move(data_res.V);
      return std::move(res);
    } // svd
  } // namespace block
} // namespace TAT

#endif // TAT_Block_Svd_HPP_
