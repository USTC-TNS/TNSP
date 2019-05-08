/* TAT/Lazy.hpp
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

#ifndef TAT_Lazy_HPP_
#define TAT_Lazy_HPP_

#include "../TAT.hpp"

namespace TAT {
  /*
  namespace lazy {
    template<Device device=Device::CPU, class Base=double>
    class Lazy;
  } // namespace lazy
  using lazy::Lazy;
  */
  // using T = std::shared_ptr<Lazy<.., ..>>
  // auto a = T::make_lazy({}, {})->set...()
  // auto b = T::make_lazy({}, {})->set...()
  // auto c = a->contract(b, ...)
  // std::cout << c.value()
  // auto d = c.svd(...)
  // T e = d.U/S/V
  // std::cout << e.value()
  namespace lazy {
    namespace legs_rename {
      template<Device device, class Base>
      Node<device, Base> run(std::function<Node<device, Base>()> tensor, const std::map<Legs, Legs>& dict) {
        return std::move(tensor().legs_rename(dict));
      } // run
    } // legs_rename

    namespace normalize {
      template<Device device, class Base, int n>
      Node<device, Base> run(std::function<Node<device, Base>()> tensor) {
        auto tmp = tensor();
        return tmp/tmp.template norm<n>();
      } // run
    } // legs_rename

    namespace transpose {
      template<Device device, class Base>
      Node<device, Base> run(std::shared_ptr<Lazy<device, Base>> tensor, const std::vector<Legs>& legs) {
        return tensor->value().transpose(legs);
      } // run
    } // namespace transpsoe

    template<Device device, class Base>
    class Lazy : public std::enable_shared_from_this<Lazy<device, Base>> {
     public:
      Node<device, Base> tensor;
      bool flag = false; // whether tensor is valid
      std::function<Node<device, Base>()> func;
      std::vector<std::weak_ptr<Lazy>> downstream;

      std::shared_ptr<Lazy<device, Base>> reset() {
        if (flag) {
          flag = false;
          for (const auto& ds : downstream) {
            if (!ds.expired()) {
              ds.lock()->reset();
            } // need reset
          } // downstream
        } // if flag
        return shared_from_this();
      } // reset this lazy and all its children

      std::shared_ptr<Lazy<device, Base>> shared_from_this() {
        return std::enable_shared_from_this<Lazy<device, Base>>::shared_from_this();
      } // shared_from_this

      template<class ... Args>
      std::shared_ptr<Lazy<device, Base>> set(Args&& ... args) {
        reset(); // since new tensor set, lazy need reset
        tensor = std::move(Node<device, Base>(std::forward<Args>(args) ...));
        flag = true;
        return shared_from_this();
      } // set_lazy, update tensor

      template<class T1=std::vector<Legs>, class T2=std::vector<Size>>
      std::shared_ptr<Lazy<device, Base>> set(T1&& _legs, T2&& _dims) {
        reset(); // since new tensor set, lazy need reset
        tensor = std::move(Node<device, Base>(std::forward<T1>(_legs), std::forward<T2>(_dims)));
        flag = true;
        return shared_from_this();
      } // set_lazy, update tensor

      template<class ... Args>
      static std::shared_ptr<Lazy<device, Base>> make(Args&& ... args) {
        auto res = std::make_shared<Lazy<device, Base>>();
        res->set(std::forward<Args>(args) ...);
        return res;
      } // make_lazy by initial tensor

      template<class T1=std::vector<Legs>, class T2=std::vector<Size>>
      static std::shared_ptr<Lazy<device, Base>> make(T1&& _legs, T2&& _dims) {
        auto res = std::make_shared<Lazy<device, Base>>();
        res->set(std::forward<T1>(_legs), std::forward<T2>(_dims));
        return res;
      } // make_lazy by initial tensor

      std::shared_ptr<Lazy<device, Base>> set_test() {
        reset();
        tensor.set_test();
        flag = true;
        return shared_from_this();
      } // set_test
      std::shared_ptr<Lazy<device, Base>> set_zero() {
        reset();
        tensor.set_zero();
        flag = true;
        return shared_from_this();
      } // set_zero
      std::shared_ptr<Lazy<device, Base>> set_random(const std::function<Base()>& random) {
        reset();
        tensor.set_random(random);
        flag = true;
        return shared_from_this();
      } // set_random
      std::shared_ptr<Lazy<device, Base>> set_constant(Base num) {
        reset();
        tensor.set_constant(num);
        flag = true;
        return shared_from_this();
      } // set_constant

      const Node<device, Base>& value() {
        if (!flag) {
          tensor = func();
          flag = true;
        }
        return tensor;
      } // calc

      std::shared_ptr<Lazy<device, Base>> legs_rename(const std::map<Legs, Legs>& dict) {
        reset();
        func = std::bind(legs_rename::run<device, Base>, std::move(func), dict);
        return shared_from_this();
      } // legs_rename

      template<int n>
      std::shared_ptr<Lazy<device, Base>> normalize() {
        reset();
        func = std::bind(normalize::run<device, Base, n>, std::move(func));
        return shared_from_this();
      } // normalize

      // no to<n> function since downstream need same type
      // the function above is lazy inplace, the below is not

      std::shared_ptr<Lazy<device, Base>> transpose(const std::vector<Legs>& new_legs) {
        auto res = std::make_shared<Lazy<device, Base>>();
        res->func = std::bind(transpose::run<device, Base>, shared_from_this(), new_legs);
        downstream.push_back(res);
        return res;
      } // transpose
    }; // class Lazy
  } // namespace lazy
} // namespace TAT

#endif // TAT_Lazy_HPP_
