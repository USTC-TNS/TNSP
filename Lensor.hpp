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
  // using T = std::shared_ptr<Lensor<.., ..>>
  // T a = Tensor(.., ..).set_...()
  // T b = Tensor(.., ..).set_...()
  // T c = a.contract(b, ...)
  // std::cout << c()
  // auto d = c.svd(...)
  // T e = d.U/S/V
  // std::cout << e()
  namespace lensor {
    namespace transpose {
      template<Device device, class Base>
      Tensor<device, Base> run(std::shared_ptr<Lensor<device, Base>> tensor, const std::vector<Legs>& legs) {
        return tensor->value().transpose(legs);
      } // run
    } // namespace transpsoe

    template<Device device, class Base>
    class Lensor : public std::enable_shared_from_this<Lensor<device, Base>> {
     public:
      Tensor<device, Base> tensor;
      bool flag = false; // whether tensor is valid
      std::function<Tensor<device, Base>()> func;
      std::vector<std::weak_ptr<Lensor>> downstream;

      void reset() {
        if (flag) {
          flag = false;
          for (const auto& ds : downstream) {
            if (!ds.expired()) {
              ds.lock()->reset();
            } // need reset
          } // downstream
        } // if flag
      } // reset this lensor and all its children

      std::shared_ptr<Lensor<device, Base>> shared_from_this() {
        return std::enable_shared_from_this<Lensor<device, Base>>::shared_from_this();
      } // shared_from_this

      template<class ... Args>
      std::shared_ptr<Lensor<device, Base>> set_lensor(Args&& ... args) {
        reset(); // since new tensor set, lensor need reset
        tensor = std::move(Tensor<device, Base>(std::forward<Args>(args) ...));
        flag = true;
        return shared_from_this();
      } // set_lensor, update tensor

      template<class ... Args>
      static std::shared_ptr<Lensor<device, Base>> make_lensor(Args&& ... args) {
        auto res = std::make_shared<Lensor<device, Base>>();
        res->set_lensor(std::forward<Args>(args) ...);
        return res;
      } // make_lensor by initial tensor

      const Tensor<device, Base>& value() {
        if (!flag) {
          tensor = func();
          flag = true;
        }
        return tensor;
      } // calc

      std::shared_ptr<Lensor<device, Base>> transpose(const std::vector<Legs>& new_legs) {
        auto res = std::make_shared<Lensor<device, Base>>();
        res->func = std::bind(transpose::run<device, Base>, shared_from_this(), new_legs);
        downstream.push_back(res);
        return res;
      } // transpose
    }; // class Lensor
  } // namespace lensor
} // namespace TAT

#endif // TAT_Lensor_HPP_
