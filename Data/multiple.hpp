/* TAT/Data/multiple.hpp
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

#ifndef TAT_Data_Multiple_HPP_
#define TAT_Data_Multiple_HPP_

#include "../Data.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
    namespace CPU {
      namespace multiple {
        template<class Base>
        void run(Base* res_data, const Base* src_data, const Base* other_data, const Size& a, const Size& b, const Size& c) {
          for (Size i=0; i<a; i++) {
            for (Size j=0; j<b; j++) {
              Base v = other_data[j];
              for (Size k=0; k<c; k++) {
                *(res_data++) = *(src_data++) * v;
              } // for k
            } // for j
          } // for i
        } // run
      } // namespace data::CPU::multiple

      template<class Base>
      Data<Base> Data<Base>::multiple(const Data<Base>& other, const Size& a, const Size& b, const Size& c) const {
        Data<Base> res(size);
        assert(b==other.size);
        assert(a*b*c==size);
        multiple::run<Base>(res.get(), get(), other.get(), a, b, c);
        return std::move(res);
      } // multiple
    } // namespace data::CPU
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

#endif // TAT_Data_Multiple_HPP_
