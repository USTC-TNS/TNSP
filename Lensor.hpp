/* TAT/Lensor.hpp
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

#ifndef TAT_Lensor_HPP_
#define TAT_Lensor_HPP_

#include "TAT.hpp"

namespace TAT {
  /*
  namespace lensor {
    template<Device device=Device::CPU, class Base=double>
    class Lensor;
  } // namespace lensor
  using lensor::Lensor;
  */
  /*
  namespace lensor {
    template<Device device, class Base>
    class LensorData {
     public:
      Tensor<device, Base> tensor;
      bool flag = false;
      std::function<Tensor<device, Base>()> func;
      std::vector<std::weak_ptr<LensorData>> downstream;

      void reset() {
        if (flag) {
          flag = false;
          for (const auto& ds : downstream) {
            if (!ds.expired()) {
              ds.lock()->reset();
            } // need reset
          } // downstream
        } // if flag
      } // reset
    }; // class LensorData

    template<Device device, class Base>
    class Lensor : public std::shared_ptr<LensorData<device, Base>> {
     public:
      Lensor() : std::shared_ptr<LensorData<device, Base>>(std::make_shared<LensorData<device, Base>>()) {}

      void reset() {
        operator*().reset();
      } // reset
    }; // class LazyTensor
  } // namespace lensor
  */
} // namespace TAT

#endif // TAT_Lensor_HPP_
