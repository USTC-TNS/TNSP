/* TAT/Data/CPU/transpose.hpp
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

#ifndef TAT_Data_CPU_Transpose_HPP_
#define TAT_Data_CPU_Transpose_HPP_

#include "../../Data.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
    namespace CPU {
      namespace transpose {
        template<class Base>
        void run(const std::vector<Rank>& plan, const std::vector<Size>& dims, const Base* src, Base* dst) {
          std::vector<int> int_plan(plan.begin(), plan.end());
          std::vector<int> int_dims(dims.begin(), dims.end());
          hptt::create_plan(int_plan.data(), int_plan.size(),
                            1, src, int_dims.data(), NULL,
                            0, dst, NULL,
                            hptt::ESTIMATE, 1, NULL, 1)->execute();
        } // run
      } // namespace data::CPU::transpose

      template<class Base>
      Data<Base> Data<Base>::transpose(const std::vector<Size>& dims,
                                       const std::vector<Rank>& plan) const {
        Data<Base> res(size);
        assert(dims.size()==plan.size());
        transpose::run(plan, dims, get(), res.get());
        return std::move(res);
      } // transpose
    } // namespace data::CPU
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

#endif // TAT_Data_CPU_Transpose_HPP_
