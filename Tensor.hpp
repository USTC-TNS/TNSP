/* TAT/Tensor.hpp
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

#ifndef TAT_Tensor_HPP_
#define TAT_Tensor_HPP_

#include "TAT.hpp"

namespace TAT {
  namespace tensor {
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
    } // namespace tensor::internal

    namespace transpose {
      void plan(std::vector<Rank>& plan, const std::vector<Legs>& new_legs, const std::vector<Legs>& legs) {
        const Rank& rank = legs.size();
        for (Rank i=0; i<rank; i++) {
          for (Rank j=0; j<rank; j++) {
            if (new_legs[i]==legs[j]) {
              plan.push_back(j);
              break;
            } // if
          } // for j
        } // for i
      } // plan
    } // namespace tensor::transpose

    namespace contract {
      void plan(Rank& contract_num,
                std::vector<Legs>& legs,
                std::vector<Legs>& new_legs1,
                std::vector<Legs>& new_legs2,
                const std::vector<Legs>& total_legs1,
                const std::vector<Legs>& total_legs2,
                const std::vector<Legs>& legs1,
                const std::vector<Legs>& legs2,
                const std::map<Legs, Legs>& map1,
                const std::map<Legs, Legs>& map2) {
        auto filt_legs1 = internal::in_and_not_in(total_legs1, legs1);
        internal::append(new_legs1, filt_legs1);
        internal::append(legs, internal::map_or_not(filt_legs1, map1));

        auto tmp_legs1 = internal::in_and_in(legs1, total_legs1);
        internal::append(new_legs1, tmp_legs1);

        auto tmp_legs2 = internal::in_and_in(legs2, total_legs2);
        internal::append(new_legs2, tmp_legs2);

        auto filt_legs2 = internal::in_and_not_in(total_legs2, legs2);
        internal::append(new_legs2, filt_legs2);
        internal::append(legs, internal::map_or_not(filt_legs2, map2));

        assert(tmp_legs1.size()==tmp_legs2.size());
        contract_num = tmp_legs1.size();
      } // plan
    } // namespace tensor::contract

    namespace multiple {}

    namespace svd {
      void plan(std::vector<Legs>& U_legs,
                std::vector<Legs>& V_legs,
                std::vector<Legs>& tmp_legs,
                Rank& u_rank,
                const std::vector<Legs>& total_legs,
                const std::vector<Legs>& u_legs,
                const Legs& new_u_legs,
                const Legs& new_v_legs) {
        u_rank = u_legs.size();
        V_legs.push_back(new_v_legs);
        for (const auto& i : total_legs) {
          if (internal::in(i, u_legs)) {
            U_legs.push_back(i);
          } else {
            V_legs.push_back(i);
          } // if
        } // for
        U_legs.push_back(new_u_legs);
        tmp_legs.insert(tmp_legs.end(), U_legs.begin(), U_legs.end()-1);
        tmp_legs.insert(tmp_legs.end(), V_legs.begin()+1, V_legs.end());
      } // plan
    } // namespace tensor::svd

    namespace qr {}

    template<Device device, class Base>
    class Tensor {
     public:
      Tensor() : legs({}), node() {}

      std::vector<Legs> legs;
      Node<device, Base> node;

      ~Tensor() = default;
      Tensor(Tensor<device, Base>&& other) = default;
      Tensor(const Tensor<device, Base>& other) = default;
      Tensor<device, Base>& operator=(Tensor<device, Base>&& other) = default;
      Tensor<device, Base>& operator=(const Tensor<device, Base>& other) = default;
      template<class T1=std::vector<Legs>, class T2=std::vector<Size>>
      Tensor(T1&& _legs, T2&& _dims) : legs(std::forward<T1>(_legs)), node(std::forward<T2>(_dims)) {
        assert(legs.size()==node.dims.size());
        assert(std::set<Legs>(legs.begin(), legs.end()).size()==legs.size());
      }
      Tensor(const Base& num) : legs({}), node(num) {}

      const std::vector<Size>& dims() const {
        return node.dims;
      } // dims
      const Size& size() const {
        return node.size();
      } // size
      const Base* get() const {
        return node.get();
      } // get
      Base* get() {
        return node.get();
      } // get

      Tensor<device, Base>& set_test() {
        node.set_test();
        return *this;
      } // set_test
      Tensor<device, Base>& set_zero() {
        node.set_zero();
        return *this;
      } // set_zero
      Tensor<device, Base>& set_random(const std::function<Base()>& random) {
        node.set_random(random);
        return *this;
      } // set_random
      Tensor<device, Base>& set_constant(Base num) {
        node.set_constant(num);
        return *this;
      } // set_constant

      Tensor<device, Base>& legs_rename(const std::map<Legs, Legs>& dict) {
        for (auto& i : legs) {
          auto where = dict.find(i);
          if (where!=dict.end()) {
            i = where->second;
          } // if map
        } // for leg
        return *this;
      } // legs_rename

      template<int n>
      Tensor<device, Base> norm() const {
        Tensor<device, Base> res({}, {});
        res.node = node.template norm<n>();
        return std::move(res);
      } // norm

      template<class Base2, ENABLE_IF(is_scalar<Base2>)>
      Tensor<device, Base2> to() const {
        Tensor<device, Base2> res;
        res.legs = legs;
        res.node = node.template to<Base2>();
        return std::move(res);
      } // to

      Tensor<device, Base> transpose(const std::vector<Legs>& new_legs) const {
        Tensor<device, Base> res;
        res.legs = internal::in_and_in(new_legs, legs);
        assert(legs.size()==res.legs.size());
#ifndef NDEBUG
        auto set_new = std::set<Legs>(res.legs.begin(), res.legs.end());
        assert(set_new.size()==res.legs.size());
        set_new.insert(legs.begin(), legs.end());
        assert(set_new.size()==res.legs.size());
#endif // NDEBUG
        std::vector<Rank> plan;
        transpose::plan(plan, res.legs, legs);
        assert(res.legs.size()==legs.size());
        assert(plan.size()==legs.size());
        res.node = node.transpose(plan);
        return std::move(res);
      } // transpose

      static Tensor<device, Base> contract(const Tensor<device, Base>& tensor1,
                                           const Tensor<device, Base>& tensor2,
                                           const std::vector<Legs>& legs1,
                                           const std::vector<Legs>& legs2,
                                           const std::map<Legs, Legs>& map1 = {},
                                           const std::map<Legs, Legs>& map2 = {}) {
        Tensor<device, Base> res;
        std::vector<Legs> new_legs1, new_legs2;
        std::vector<Rank> plan1, plan2;
        Rank contract_num;
        assert(legs1.size()==legs2.size());
        contract::plan(contract_num, res.legs, new_legs1, new_legs2, tensor1.legs, tensor2.legs, legs1, legs2, map1, map2);
        transpose::plan(plan1, new_legs1, tensor1.legs);
        transpose::plan(plan2, new_legs2, tensor2.legs);
        assert(new_legs1.size()==tensor1.legs.size());
        assert(plan1.size()==tensor1.legs.size());
        assert(new_legs2.size()==tensor2.legs.size());
        assert(plan2.size()==tensor2.legs.size());
        res.node = Node<device, Base>::contract(tensor1.node, tensor2.node, plan1, plan2, contract_num);
        return std::move(res);
      } // contract

      Tensor<device, Base> contract(const Tensor<device, Base>& tensor2,
                                    const std::vector<Legs>& legs1,
                                    const std::vector<Legs>& legs2,
                                    const std::map<Legs, Legs>& map1 = {},
                                    const std::map<Legs, Legs>& map2 = {}) const {
        return std::move(Tensor<device, Base>::contract(*this, tensor2, legs1, legs2, map1, map2));
      } // contract

      Tensor<device, Base> multiple(const Tensor<device, Base>& other, const Legs& position) const {
        Tensor<device, Base> res;
        assert(other.legs.size()==1);
        res.legs = legs;
        auto pos = std::find(legs.begin(), legs.end(), position);
        if (pos==legs.end()) {
          return *this;
        } // if not multiple
        Rank index = std::distance(legs.begin(), pos);
        res.node = node.multiple(other.node, index);
        return std::move(res);
      } // multiple

      friend class svd_res;
      class svd_res {
       public:
        Tensor<device, Base> U;
        Tensor<device, Base> S;
        Tensor<device, Base> V;
      }; // class svd_res

      svd_res svd(const std::vector<Legs>& input_u_legs, const Legs& new_u_legs, const Legs& new_v_legs, const Rank& cut=-1) const {
        std::vector<Legs> u_legs = internal::in_and_in(legs, input_u_legs);
        svd_res res;
        std::vector<Legs> tmp_legs;
        std::vector<Rank> plan;
        Rank u_rank;
        svd::plan(res.U.legs, res.V.legs, tmp_legs, u_rank, legs, u_legs, new_u_legs, new_v_legs);
        transpose::plan(plan, tmp_legs, legs);
        auto node_res = node.svd(plan, u_rank, cut);
        res.S.legs = {new_u_legs};// new_u_legs or new_v_legs
        res.U.node = std::move(node_res.U);
        res.S.node = std::move(node_res.S);
        res.V.node = std::move(node_res.V);
        return std::move(res);
      } // svd

      friend class qr_res;
      class qr_res {
       public:
        Tensor<device, Base> Q;
        Tensor<device, Base> R;
      }; // class qr_res

      qr_res qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const {
        std::vector<Legs> q_legs = internal::in_and_in(legs, input_q_legs);
        qr_res res;
        std::vector<Legs> tmp_legs;
        std::vector<Rank> plan;
        Rank q_rank;
        svd::plan(res.Q.legs, res.R.legs, tmp_legs, q_rank, legs, q_legs, new_q_legs, new_r_legs);
        transpose::plan(plan, tmp_legs, legs);
        auto node_res = node.qr(plan, q_rank);
        res.Q.node = std::move(node_res.Q);
        res.R.node = std::move(node_res.R);
        return std::move(res);
      } // qr
    }; // class Tensor

    inline namespace scalar {
      bool operator==(const std::vector<Legs>& a, const std::vector<Legs>& b) {
        if (a.size()!=b.size()) {
          return false;
        } // if size
        Rank size=a.size();
        for (Rank i=0; i<size; i++) {
          if (a[i]!=b[i]) {
            return false;
          } // if i
        } // for
        return true;
      } // operator==

      template<Device device, class Base>
      Tensor<device, Base>& operator*=(Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        if (b.legs.size()!=0) {
          assert(a.legs==b.legs);
        }
        a.node *= b.node;
        return a;
      } // operator*=

      template<Device device, class Base>
      Tensor<device, Base> operator*(const Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        Tensor<device, Base> res;
        if (b.legs.size()==0) {
          res.legs = a.legs;
        } else if (a.legs.size()==0) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.node = a.node * b.node;
        return std::move(res);
      } // operator*

      template<Device device, class Base>
      Tensor<device, Base>& operator/=(Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        if (b.legs.size()!=0) {
          assert(a.legs==b.legs);
        } // if
        a.node /= b.node;
        return a;
      } // operator/=

      template<Device device, class Base>
      Tensor<device, Base> operator/(const Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        Tensor<device, Base> res;
        if (b.legs.size()==0) {
          res.legs = a.legs;
        } else if (a.legs.size()==0) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.node = a.node / b.node;
        return std::move(res);
      } // operator/

      template<Device device, class Base>
      Tensor<device, Base> operator+(const Tensor<device, Base>& a) {
        Tensor<device, Base> res;
        res.legs = a.legs;
        res.node = + a.node;
        return std::move(res);
      } // operator+

      template<Device device, class Base>
      Tensor<device, Base> operator+(Tensor<device, Base>&& a) {
        Tensor<device, Base> res;
        res.legs = std::move(a.legs);
        res.node = + std::move(a.node);
        return std::move(res);
      } // operator+

      template<Device device, class Base>
      Tensor<device, Base>& operator+=(Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        if (b.legs.size()!=0) {
          assert(a.legs==b.legs);
        } // if
        a.node += b.node;
        return a;
      } // operator+=

      template<Device device, class Base>
      Tensor<device, Base> operator+(const Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        Tensor<device, Base> res;
        if (b.legs.size()==0) {
          res.legs = a.legs;
        } else if (a.legs.size()==0) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.node = a.node + b.node;
        return std::move(res);
      } // operator+

      template<Device device, class Base>
      Tensor<device, Base> operator-(const Tensor<device, Base>& a) {
        Tensor<device, Base> res;
        res.legs = a.legs;
        res.node = - a.node;
        return std::move(res);
      } // operator-

      template<Device device, class Base>
      Tensor<device, Base>& operator-=(Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        if (b.legs.size()!=0) {
          assert(a.legs==b.legs);
        } // if
        a.node -= b.node;
        return a;
      } // operator-=

      template<Device device, class Base>
      Tensor<device, Base> operator-(const Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        Tensor<device, Base> res;
        if (b.legs.size()==0) {
          res.legs = a.legs;
        } else if (a.legs.size()==0) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.node = a.node - b.node;
        return std::move(res);
      } // operator-

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base>& operator*=(Tensor<device, Base>& a, const B& b) {
        return a*=Tensor<device, Base>(b);
      } // operator*=

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base> operator*(const Tensor<device, Base>& a, const B& b) {
        return a*Tensor<device, Base>(b);
      } // operator*

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base> operator*(const B& b, const Tensor<device, Base>& a) {
        return Tensor<device, Base>(b)*a;
      } // operator*

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base>& operator/=(Tensor<device, Base>& a, const B& b) {
        return a/=Tensor<device, Base>(b);
      } // operator/=

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base> operator/(const Tensor<device, Base>& a, const B& b) {
        return a/Tensor<device, Base>(b);
      } // operator/

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base> operator/(const B& b, const Tensor<device, Base>& a) {
        return Tensor<device, Base>(b)/a;
      } // operator/

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base>& operator+=(Tensor<device, Base>& a, const B& b) {
        return a+=Tensor<device, Base>(b);
      } // operator+

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base> operator+(const Tensor<device, Base>& a, const B& b) {
        return a+Tensor<device, Base>(b);
      } // operator+

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base> operator+(const B& b, const Tensor<device, Base>& a) {
        return Tensor<device, Base>(b)+a;
      } // operator+

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base>& operator-=(Tensor<device, Base>& a, const B& b) {
        return a-=Tensor<device, Base>(b);
      } // operator-=

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base> operator-(const Tensor<device, Base>& a, const B& b) {
        return a-Tensor<device, Base>(b);
      } // operator-

      template<Device device, class Base, class B, ENABLE_IF(is_scalar<B>)>
      Tensor<device, Base> operator-(const B& b, const Tensor<device, Base>& a) {
        return Tensor<device, Base>(b)-a;
      } // operator-
    } // namespace tensor::scalar

    inline namespace io {
      std::ostream& operator<<(std::ostream& out, const std::vector<Legs>& value) {
        Rank size=value.size();
        out << "[";
        for (Rank i=0; i<size-1; i++) {
          out << "\"" << value[i] << "\", ";
        } // for i
        if (size!=0) {
          out << "\"" << value[size-1] << "\"";
        } // if
        out << "]";
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ostream& operator<<(std::ostream& out, const Tensor<device, Base>& value) {
        return out << "{" << rang::fgB::yellow << "\"rank\": " << value.legs.size() << rang::fg::reset << ", " << rang::fgB::blue << "\"legs\": " << value.legs << rang::fg::reset << ", \"node\": " << value.node << "}";
      } // operator<<

      template<Device device, class Base>
      std::ofstream& operator<<(std::ofstream& out, const Tensor<device, Base>& value) {
        Rank rank = value.legs.size();
        out.write((char*)&rank, sizeof(Rank));
        out.write((char*)value.legs.data(), rank*sizeof(Legs));
        out << value.node;
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ifstream& operator>>(std::ifstream& in, Tensor<device, Base>& value) {
        Rank rank;
        in.read((char*)&rank, sizeof(Rank));
        value.legs.resize(rank);
        in.read((char*)value.legs.data(), rank*sizeof(Legs));
        in >> value.node;
        return in;
      } // operator<<
    } // namespace tensor::io
  } // namespace tensor
} // namespace TAT

#endif // TAT_Tensor_HPP_
