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
    template<Device device, class Base>
    class Lazy final : public BaseLazy {
     public:
      std::unique_ptr<Node<device, Base>> node;
      std::function<Node<device, Base>()> func;
      std::vector<std::weak_ptr<BaseLazy>> downstream;

      void reset(bool release_itself=true) override {
        if (node) {
          if(release_itself) {
            delete node.release();
          }
          for (const auto& ds : downstream) {
            if (!ds.expired()) {
              ds.lock()->reset();
            } // need reset
          } // downstream
        } // if node has value
      } // reset this lazy and all its children

      std::shared_ptr<Lazy<device, Base>> shared_from_this() {
        return std::dynamic_pointer_cast<Lazy<device, Base>>(std::enable_shared_from_this<BaseLazy>::shared_from_this());
      } // shared_from_this

      template<class ... Args>
      std::shared_ptr<Lazy<device, Base>> set(Args&& ... args) {
        reset(); // since new node set, lazy need reset
        node = std::unique_ptr<Node<device, Base>>(new Node<device, Base>(std::forward<Args>(args) ...));
        return shared_from_this();
      } // set_lazy, update node

      template<class T1=std::vector<Legs>, class T2=std::vector<Size>>
      std::shared_ptr<Lazy<device, Base>> set(T1&& _legs, T2&& _dims) {
        reset(); // since new node set, lazy need reset
        node = std::unique_ptr<Node<device, Base>>(new Node<device, Base>(std::forward<T1>(_legs), std::forward<T2>(_dims)));
        return shared_from_this();
      } // set_lazy, update node

      template<class ... Args>
      static std::shared_ptr<Lazy<device, Base>> make(Args&& ... args) {
        auto res = std::make_shared<Lazy<device, Base>>();
        res->set(std::forward<Args>(args) ...);
        return res;
      } // make_lazy by initial node

      template<class T1=std::vector<Legs>, class T2=std::vector<Size>>
      static std::shared_ptr<Lazy<device, Base>> make(T1&& _legs, T2&& _dims) {
        auto res = std::make_shared<Lazy<device, Base>>();
        res->set(std::forward<T1>(_legs), std::forward<T2>(_dims));
        return res;
      } // make_lazy by initial node

      std::shared_ptr<Lazy<device, Base>> set_test() {
        reset(false);
        node->set_test();
        return shared_from_this();
      } // set_test
      std::shared_ptr<Lazy<device, Base>> set_zero() {
        reset(false);
        node->set_zero();
        return shared_from_this();
      } // set_zero
      std::shared_ptr<Lazy<device, Base>> set_random(const std::function<Base()>& random) {
        reset(false);
        node->set_random(random);
        return shared_from_this();
      } // set_random
      std::shared_ptr<Lazy<device, Base>> set_constant(Base num) {
        reset(false);
        node->set_constant(num);
        return shared_from_this();
      } // set_constant

      std::shared_ptr<Lazy<device, Base>> calc() {
        if (!node) {
          set(std::move(func()));
        } // has no value
        return shared_from_this();
      } // calc

      const Node<device, Base>& value() {
        calc();
        return *node;
      } // value

      // several inplace op

      std::shared_ptr<Lazy<device, Base>> legs_rename(const std::map<Legs, Legs>& dict) {
        reset(false);
        if (node) {
          node->legs_rename(dict);
        } else {
          auto&& tmp = std::move(func);
          func = [=](){
            auto res = tmp();
            res.legs_rename(dict);
            return std::move(res);
          };
        } // flag
        return shared_from_this();
      } // legs_rename

      template<int n>
      std::shared_ptr<Lazy<device, Base>> normalize() {
        reset(false);
        if (node) {
          *node /= node->template norm<n>();
        } else {
          auto&& tmp = std::move(func);
          func = [=](){
            auto res=tmp();
            res/=res.template norm<n>();
            return std::move(res);
          };
        } // has value
        return shared_from_this();
      } // normalize

      // no to<n> function since downstream need same type
      // the function above is lazy inplace, the below is not

      std::shared_ptr<Lazy<device, Base>> transpose(const std::vector<Legs>& new_legs) {
        auto res = std::make_shared<Lazy<device, Base>>();
        auto origin = shared_from_this();
        res->func = [=](){
          return origin->value().transpose(new_legs);
        };
        downstream.push_back(std::dynamic_pointer_cast<BaseLazy>(res));
        return res;
      } // transpose

      template<class Base2, ENABLE_IF(is_scalar<Base2>)>
      std::shared_ptr<Lazy<device, Base2>> to() {
        auto res = std::make_shared<Lazy<device, Base>>();
        auto origin = shared_from_this();
        res->func = [=](){
          return origin->value().template to<Base2>();
        };
        downstream.push_back(std::dynamic_pointer_cast<BaseLazy>(res));
        return res;
      }
    }; // class Lazy
  } // namespace lazy
} // namespace TAT

#endif // TAT_Lazy_HPP_
