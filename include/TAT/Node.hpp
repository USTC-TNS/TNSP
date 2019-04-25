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
    template<Device device, class Base>
    class Node {
     public:
      Node() : dims({}), data() {}

      std::vector<Size> dims;
      Data<device, Base> data;

      ~Node() = default;
      Node(Node<device, Base>&& other) = default;
      Node(const Node<device, Base>& other) = default;
      Node<device, Base>& operator=(Node<device, Base>&& other) = default;
      Node<device, Base>& operator=(const Node<device, Base>& other) = default;
      static Size get_size(const std::vector<Size>& _dims) {
        Size res = 1;
        for (const auto& i : _dims) {
          res *= i;
        } // for i
        return res;
      } // get_size
      template<class T=std::vector<Size>>
      Node(T&& _dims) : data(get_size(_dims)) {
        dims = std::forward<T>(_dims);
      }
      Node(const Base& num) : dims({}), data(num) {}

      const Size& size() const {
        return data.size;
      } // size
      const Base* get() const {
        return data.get();
      } // get
      Base* get() {
        return data.get();
      } // get

      Node<device, Base>& set_test() {
        data.set_test();
        return *this;
      } // set_test
      Node<device, Base>& set_zero() {
        data.set_zero();
        return *this;
      } // set_zero
      Node<device, Base>& set_random(const std::function<Base()>& random) {
        data.set_random(random);
        return *this;
      } // set_random
      Node<device, Base>& set_constant(Base num) {
        data.set_constant(num);
        return *this;
      } // set_constant

      template<class Base2, ENABLE_IF(is_scalar<Base2>)>
      Node<device, Base2> to() const {
        Node<device, Base2> res;
        res.dims = dims;
        res.data = data.template to<Base2>();
        return std::move(res);
      } // to

      template<int n>
      Node<device, Base> norm() const;

      Node<device, Base> transpose(const std::vector<Rank>& plan) const;

      static Node<device, Base> contract(const Node<device, Base>& node1,
                                         const Node<device, Base>& node2,
                                         const std::vector<Rank>& plan1,
                                         const std::vector<Rank>& plan2,
                                         const Rank& contract_num);

      Node<device, Base> contract(const Node<device, Base>& node2,
                                  const std::vector<Rank>& plan1,
                                  const std::vector<Rank>& plan2,
                                  const Rank& contract_num) const {
        return std::move(Node<device, Base>::contract(*this, node2, plan1, plan2, contract_num));
      } // contract

      Node<device, Base> multiple(const Node<device, Base>& other, const Rank& index) const;

      class svd_res {
       public:
        Node<device, Base> U;
        Node<device, Base> S;
        Node<device, Base> V;
      }; // class svd_res

      svd_res svd(const std::vector<Rank>& plan, const Rank& u_rank, const Size& cut) const;

      class qr_res {
       public:
        Node<device, Base> Q;
        Node<device, Base> R;
      }; // class qr_res

      qr_res qr(const std::vector<Rank>& plan, const Rank& q_rank) const;
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
