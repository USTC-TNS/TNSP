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

#include "TAT.hpp"

namespace TAT {
  namespace node {
    namespace transpose {
      void plan(std::vector<Size>& new_dims, const std::vector<Size>& dims, const std::vector<Rank>& plan) {
        for (const auto& i : plan) {
          new_dims.push_back(dims[i]);
        } // for i
      } // plan
    } // namespace node::transpose

    namespace contract {
      void plan(std::vector<Size>& dims, Size& m, Size& k, Size& n,
                const::std::vector<Size>& dims1,
                const::std::vector<Size>& dims2,
                const std::vector<Rank>& plan1,
                const std::vector<Rank>& plan2,
                const Rank& contract_num) {
        Rank i, tmp=dims1.size()-contract_num, rank2=dims2.size();
        for (i=0; i<tmp; i++) {
          const Size& t = dims1[plan1[i]];
          m *= t;
          dims.push_back(t);
        } // for i
        for (i=0; i<contract_num; i++) {
          k *= dims1[plan1[i+tmp]];
          assert(dims1[plan1[i+tmp]]==dims2[plan2[i]]);
        } // for i
        for (; i<rank2; i++) {
          const Size& t = dims2[plan2[i]];
          n *= t;
          dims.push_back(t);
        } // for i
      } // plan
    } // namespace node::contract

    namespace multiple {
      void plan(Size& a, Size& b, Size& c, const std::vector<Size>& dims, const Rank& index) {
        Rank i=0, rank=dims.size();
        for (; i<index; i++) {
          a *= dims[i];
        } // for i
        b = dims[i];
        i++;
        for (; i<rank; i++) {
          c *= dims[i];
        } // for
      } // plan
    } // namespace node::multiple

    namespace svd {
      void plan(Size& u_size, const Rank& u_rank, const std::vector<Size>& dims) {
        for (Rank i=0; i<u_rank; i++) {
          u_size *= dims[i];
        } // for i
      } // plan
    } // namespace node::svd

    namespace qr {}

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

      template<int n>
      Node<device, Base> norm() const {
        Node<device, Base> res({});
        res.data = data.template norm<n>();
        return std::move(res);
      } // norm

      template<class Base2, ENABLE_IF(is_scalar<Base2>)>
      Node<device, Base2> to() const {
        Node<device, Base2> res;
        res.dims = dims;
        res.data = data.template to<Base2>();
        return std::move(res);
      } // to

      Node<device, Base> transpose(const std::vector<Rank>& plan) const {
        Node<device, Base> res;
        transpose::plan(res.dims, dims, plan);
        assert(plan.size()==dims.size());
        assert(get_size(res.dims)==data.size);
        res.data = data.transpose(dims, plan);
        return std::move(res);
      } // transpose

      static Node<device, Base> contract(const Node<device, Base>& node1,
                                         const Node<device, Base>& node2,
                                         const std::vector<Rank>& plan1,
                                         const std::vector<Rank>& plan2,
                                         const Rank& contract_num) {
        Node<device, Base> res;
        Size m=1, k=1, n=1;
        contract::plan(res.dims, m, k, n, node1.dims, node2.dims, plan1, plan2, contract_num);
        res.data = Data<device, Base>::contract(node1.data, node2.data, node1.dims, node2.dims, plan1, plan2, m, k, n);
        return std::move(res);
      } // contract

      Node<device, Base> contract(const Node<device, Base>& node2,
                                  const std::vector<Rank>& plan1,
                                  const std::vector<Rank>& plan2,
                                  const Rank& contract_num) const {
        return std::move(Node<device, Base>::contract(*this, node2, plan1, plan2, contract_num));
      } // contract

      Node<device, Base> multiple(const Node<device, Base>& other, const Rank& index) const {
        Node<device, Base> res;
        res.dims = dims;
        Size a=1, b=1, c=1;
        multiple::plan(a, b, c, dims, index);
        assert(other.dims.size()==1);
        assert(b==other.dims[0]);
        res.data = data.multiple(other.data, a, b, c);
        return std::move(res);
      } // multiple

      friend class svd_res;
      class svd_res {
       public:
        Node<device, Base> U;
        Node<device, Base> S;
        Node<device, Base> V;
      }; // class svd_res

      svd_res svd(const std::vector<Rank>& plan, const Rank& u_rank, const Size& cut) const {
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

      friend class qr_res;
      class qr_res {
       public:
        Node<device, Base> Q;
        Node<device, Base> R;
      }; // class qr_res

      qr_res qr(const std::vector<Rank>& plan, const Rank& q_rank) const {
        qr_res res;
        Size q_size=1;
        std::vector<Size> tmp_dims;
        transpose::plan(tmp_dims, dims, plan);
        svd::plan(q_size, q_rank, tmp_dims);
        auto mid = tmp_dims.begin()+q_rank;
        Size r_size=data.size/q_size;
        Size min_size = (q_size<r_size)?q_size:r_size;
        auto data_res = data.qr(dims, plan, q_size, r_size, min_size);
        res.Q.dims.insert(res.Q.dims.end(), tmp_dims.begin(), mid);
        res.Q.dims.push_back(min_size);
        res.R.dims.push_back(min_size);
        res.R.dims.insert(res.R.dims.end(), mid, tmp_dims.end());
        res.Q.data = std::move(data_res.Q);
        res.R.data = std::move(data_res.R);
        return std::move(res);
      } // qr
    }; // class Node

    inline namespace scalar {
      bool operator==(const std::vector<Size>& a, const std::vector<Size>& b) {
        if (a.size()!=b.size()) {
          return false;
        } // if size
        Rank size=a.size();
        for (Rank i=0; i<size; i++) {
          if (a[i]!=b[i]) {
            return false;
          } // if
        } // for i
        return true;
      } // operator==

      template<Device device, class Base>
      Node<device, Base>& operator*=(Node<device, Base>& a, const Node<device, Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data *= b.data;
        return a;
      } // operator*=

      template<Device device, class Base>
      Node<device, Base> operator*(const Node<device, Base>& a, const Node<device, Base>& b) {
        Node<device, Base> res;
        if (b.dims.size()==0) {
          res.dims = a.dims;
        } else if (a.dims.size()==0) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data * b.data;
        return std::move(res);
      } // operator*

      template<Device device, class Base>
      Node<device, Base>& operator/=(Node<device, Base>& a, const Node<device, Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data /= b.data;
        return a;
      } // operator/=

      template<Device device, class Base>
      Node<device, Base> operator/(const Node<device, Base>& a, const Node<device, Base>& b) {
        Node<device, Base> res;
        if (b.dims.size()==0) {
          res.dims = a.dims;
        } else if (a.dims.size()==0) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data / b.data;
        return std::move(res);
      } // operator/

      template<Device device, class Base>
      Node<device, Base> operator+(const Node<device, Base>& a) {
        Node<device, Base> res;
        res.dims = a.dims;
        res.data = + a.data;
        return std::move(res);
      } // operator+

      template<Device device, class Base>
      Node<device, Base> operator+(Node<device, Base>&& a) {
        Node<device, Base> res;
        res.dims = std::move(a.dims);
        res.data = + std::move(a.data);
        return std::move(res);
      } // operator+

      template<Device device, class Base>
      Node<device, Base>& operator+=(Node<device, Base>& a, const Node<device, Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data += b.data;
        return a;
      } // operator+=

      template<Device device, class Base>
      Node<device, Base> operator+(const Node<device, Base>& a, const Node<device, Base>& b) {
        Node<device, Base> res;
        if (b.dims.size()==0) {
          res.dims = a.dims;
        } else if (a.dims.size()==0) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data + b.data;
        return std::move(res);
      } // operator+

      template<Device device, class Base>
      Node<device, Base> operator-(const Node<device, Base>& a) {
        Node<device, Base> res;
        res.dims = a.dims;
        res.data = - a.data;
        return std::move(res);
      } // operator-

      template<Device device, class Base>
      Node<device, Base>& operator-=(Node<device, Base>& a, const Node<device, Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data -= b.data;
        return a;
      } // operator-=

      template<Device device, class Base>
      Node<device, Base> operator-(const Node<device, Base>& a, const Node<device, Base>& b) {
        Node<device, Base> res;
        if (b.dims.size()==0) {
          res.dims = a.dims;
        } else if (a.dims.size()==0) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data - b.data;
        return std::move(res);
      } // operator-
    } // namespace node::scalar

    inline namespace io {
      std::ostream& operator<<(std::ostream& out, const std::vector<Size>& value) {
        Rank size=value.size();
        out << "[";
        for (Rank i=0; i<size-1; i++) {
          out << value[i] << ", ";
        } // for i
        if (size!=0) {
          out << value[size-1];
        } // if
        out << "]";
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ostream& operator<<(std::ostream& out, const Node<device, Base>& value) {
        return out << "{" << rang::fg::magenta << "\"dims\": " << value.dims << rang::fg::reset << ", \"data\": " << value.data << "}";
      } // operator<<

      template<Device device, class Base>
      std::ofstream& operator<<(std::ofstream& out, const Node<device, Base>& value) {
        Rank rank = value.dims.size();
        out.write((char*)&rank, sizeof(Rank));
        out.write((char*)value.dims.data(), rank*sizeof(Size));
        out << value.data;
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ifstream& operator>>(std::ifstream& in, Node<device, Base>& value) {
        Rank rank;
        in.read((char*)&rank, sizeof(Rank));
        value.dims.resize(rank);
        in.read((char*)value.dims.data(), rank*sizeof(Size));
        in >> value.data;
        return in;
      } // operator<<
    } // namespace node::io
  } // namespace node
} // namespace TAT

#endif // TAT_Node_HPP_
