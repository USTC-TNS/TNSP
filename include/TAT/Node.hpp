/** TAT/Node.hpp
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

    template<class Base>
    class Node {
     public:
      Node() : legs({}), tensor() {}

      std::vector<Legs> legs;
      Tensor<Base> tensor;

      ~Node() = default;
      Node(Node<Base>&& other) = default;
      Node(const Node<Base>& other) = default;
      Node<Base>& operator=(Node<Base>&& other) = default;
      Node<Base>& operator=(const Node<Base>& other) = default;
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

      Node<Base>& set_test() {
        tensor.set_test();
        return *this;
      } // set_test
      Node<Base>& set_zero() {
        tensor.set_zero();
        return *this;
      } // set_zero
      Node<Base>& set_random(const std::function<Base()>& random) {
        tensor.set_random(random);
        return *this;
      } // set_random
      Node<Base>& set_constant(Base num) {
        tensor.set_constant(num);
        return *this;
      } // set_constant

      Node<Base>& legs_rename(const std::map<Legs, Legs>& dict) {
        for (auto& i : legs) {
          auto where = dict.find(i);
          if (where!=dict.end()) {
            i = where->second;
          } // if map
        } // for leg
        return *this;
      } // legs_rename

      template<class Base2, ENABLE_IF(scalar_tools::is_scalar<Base2>)>
      Node<Base2> to() const {
        Node<Base2> res;
        res.legs = legs;
        res.tensor = tensor.template to<Base2>();
        return std::move(res);
      } // to

      template<int n>
      Node<Base> norm() const;

      Node<Base> transpose(const std::vector<Legs>& new_legs) const;

      static Node<Base> contract(const Node<Base>& node1,
                                         const Node<Base>& node2,
                                         const std::vector<Legs>& legs1,
                                         const std::vector<Legs>& legs2,
                                         const std::map<Legs, Legs>& map1,
                                         const std::map<Legs, Legs>& map2);

      Node<Base> contract(const Node<Base>& node2,
                                  const std::vector<Legs>& legs1,
                                  const std::vector<Legs>& legs2,
                                  const std::map<Legs, Legs>& map1 = {},
                                  const std::map<Legs, Legs>& map2 = {}) const {
        return std::move(Node<Base>::contract(*this, node2, legs1, legs2, map1, map2));
      } // contract

      Node<Base> multiple(const Node<Base>& other, const Legs& position) const;

      friend class svd_res;
      class svd_res {
       public:
        Node<Base> U;
        Node<Base> S;
        Node<Base> V;
      }; // class svd_res

      svd_res svd(const std::vector<Legs>& input_u_legs, const Legs& new_u_legs, const Legs& new_v_legs, const Rank& cut=-1) const;

      friend class qr_res;
      class qr_res {
       public:
        Node<Base> Q;
        Node<Base> R;
      }; // class qr_res

      qr_res qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const;
    }; // class Node
  } // namespace node
} // namespace TAT

#endif // TAT_Node_HPP_
