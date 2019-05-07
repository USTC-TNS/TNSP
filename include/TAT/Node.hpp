/* TAT/Node.hpp
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

#ifndef TAT_Node_HPP_
#define TAT_Node_HPP_

#include "../TAT.hpp"

namespace TAT {
  namespace node {
    namespace internal {
      template<class T>
      bool in(const T& i, const std::vector<T>& b) {
        auto pos = std::find(b.begin(), b.end(), i);
        return pos!=b.end();
      } // in

      template<class T>
      std::vector<T> in_and_in(const std::vector<T>& a, const std::vector<T>& b) {
        std::vector<T> res;
        for (const auto& i : a) {
          if (in(i, b)) {
            res.push_back(i);
          }
        }
        return res;
      } // in_and_in

      template<class T>
      std::vector<T> in_and_not_in(const std::vector<T>& a, const std::vector<T>& b) {
        std::vector<T> res;
        for (const auto& i : a) {
          if (!in(i, b)) {
            res.push_back(i);
          }
        }
        return res;
      } // in_and_not_in

      template<class T>
      void append(std::vector<T>& a, const std::vector<T>& b) {
        a.insert(a.end(), b.begin(), b.end());
      } // append

      template<class T>
      std::vector<T> map_or_not(const std::vector<T>& a, const std::map<T, T>& b) {
        std::vector<T> res;
        for (const auto& i : a) {
          try {
            res.push_back(b.at(i));
          } catch (const std::out_of_range& e) {
            res.push_back(i);
          } // try
        } // for
        return res;
      } // map_or_not
    } // namespace node::internal

    template<Device device, class Base>
    class Node : public std::enable_shared_from_this<Node<device, Base>> {
     public:
      Node() : legs({}), tensor() {}

      std::vector<Legs> legs;
      Tensor<device, Base> tensor;
      bool flag = false; // whether tensor is valid
      std::function<Tensor<device, Base>()> func;
      std::vector<std::weak_ptr<Node<device, Base>>> downstream;

      // graph op

      using NodePtr = std::shared_ptr<Node<device, Base>>;

      NodePtr shared_from_this() {
        return std::enable_shared_from_this<Node<device, Base>>::shared_from_this();
      } // shared_from_this

      NodePtr reset() {
        if (flag) {
          flag = false;
          for (const auto& ds : downstream) {
            if (!ds.expired()) {
              ds.lock()->reset();
            } // need reset
          } // downstream
        } // if flag
        return shared_from_this();
      } // reset this node and all its children

      template<class ... Args>
      static NodePtr make(Args&& ... args) {
        auto res = std::make_shared<Node<device, Base>>(std::forward<Args>(args) ...);
        return res;
      } // make_node by initial node

      template<class T1=std::vector<Legs>, class T2=std::vector<Size>>
      static NodePtr make(T1&& _legs, T2&& _dims) {
        auto res = std::make_shared<Node<device, Base>>(std::forward<T1>(_legs), std::forward<T2>(_dims));
        return res;
      } // make_node by initial node

      template<class T=std::vector<Legs>>
      static NodePtr make(T&& _legs) {
        auto res = std::make_shared<Node<device, Base>>(std::forward<T>(_legs));
        return res;
      } // make_node by initial node

      NodePtr calc() {
        if (!flag) {
          tensor = func();
          flag = true;
        }
        return shared_from_this();
      } // calc

      // node op

      ~Node() = default;
      Node(Node<device, Base>&& other) = default;
      Node(const Node<device, Base>& other) = default;
      Node<device, Base>& operator=(Node<device, Base>&& other) = default;
      Node<device, Base>& operator=(const Node<device, Base>& other) = default;
      template<class T=std::vector<Legs>>
      Node(T&& _legs) : legs(std::forward<T>(_legs)) {}
      template<class T1=std::vector<Legs>, class T2=std::vector<Size>>
      Node(T1&& _legs, T2&& _dims) : legs(std::forward<T1>(_legs)), tensor(std::forward<T2>(_dims)) {
        assert(legs.size()==tensor.dims.size());
        assert(std::set<Legs>(legs.begin(), legs.end()).size()==legs.size());
      }
      Node(const Base& num) : legs({}), tensor(num) {}

      const std::vector<Size>& dims() const {
        return tensor.dims;
      } // dims
      const Size& size() const {
        return tensor.size();
      } // size
      const Base* get() const {
        return tensor.get();
      } // get
      Base* get() {
        return tensor.get();
      } // get

      NodePtr set_test() {
        reset();
        tensor.set_test();
        flag = true;
        return shared_from_this();
      } // set_test
      NodePtr set_zero() {
        reset();
        tensor.set_zero();
        flag = true;
        return shared_from_this();
      } // set_zero
      NodePtr set_random(const std::function<Base()>& random) {
        reset();
        tensor.set_random(random);
        flag = true;
        return shared_from_this();
      } // set_random
      NodePtr set_constant(Base num) {
        reset();
        tensor.set_constant(num);
        flag = true;
        return shared_from_this();
      } // set_constant

      Node<device, Base>& legs_rename(const std::map<Legs, Legs>& dict) {
        for (auto& i : legs) {
          auto where = dict.find(i);
          if (where!=dict.end()) {
            i = where->second;
          } // if map
        } // for leg
        return *this;
      } // legs_rename

      template<class Base2, ENABLE_IF(is_scalar<Base2>)>
      Node<device, Base2> to() const {
        Node<device, Base2> res;
        res.legs = legs;
        res.tensor = tensor.template to<Base2>();
        return std::move(res);
      } // to

      template<int n>
      Node<device, Base> norm() const;

      Node<device, Base> transpose(const std::vector<Legs>& new_legs) const;

      static Node<device, Base> contract(const Node<device, Base>& node1,
                                           const Node<device, Base>& node2,
                                           const std::vector<Legs>& legs1,
                                           const std::vector<Legs>& legs2,
                                           const std::map<Legs, Legs>& map1,
                                           const std::map<Legs, Legs>& map2);

      Node<device, Base> contract(const Node<device, Base>& node2,
                                    const std::vector<Legs>& legs1,
                                    const std::vector<Legs>& legs2,
                                    const std::map<Legs, Legs>& map1 = {},
                                    const std::map<Legs, Legs>& map2 = {}) const {
        return std::move(Node<device, Base>::contract(*this, node2, legs1, legs2, map1, map2));
      } // contract

      Node<device, Base> multiple(const Node<device, Base>& other, const Legs& position) const;

      friend class svd_res;
      class svd_res {
       public:
        Node<device, Base> U;
        Node<device, Base> S;
        Node<device, Base> V;
      }; // class svd_res

      svd_res svd(const std::vector<Legs>& input_u_legs, const Legs& new_u_legs, const Legs& new_v_legs, const Rank& cut=-1) const;

      friend class qr_res;
      class qr_res {
       public:
        Node<device, Base> Q;
        Node<device, Base> R;
      }; // class qr_res

      qr_res qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const;
    }; // class Node
  } // namespace node
} // namespace TAT

#include "Node/norm.hpp"
#include "Node/transpose.hpp"
#include "Node/contract.hpp"
#include "Node/multiple.hpp"
#include "Node/svd.hpp"
#include "Node/qr.hpp"
#include "Node/scalar.hpp"
#include "Node/io.hpp"

#endif // TAT_Node_HPP_
