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

#include "../TAT.hpp"

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

      template<class Base2, ENABLE_IF(is_scalar<Base2>)>
      Tensor<device, Base2> to() const {
        Tensor<device, Base2> res;
        res.legs = legs;
        res.node = node.template to<Base2>();
        return std::move(res);
      } // to

      template<int n>
      Tensor<device, Base> norm() const;

      Tensor<device, Base> transpose(const std::vector<Legs>& new_legs) const;

      static Tensor<device, Base> contract(const Tensor<device, Base>& tensor1,
                                           const Tensor<device, Base>& tensor2,
                                           const std::vector<Legs>& legs1,
                                           const std::vector<Legs>& legs2,
                                           const std::map<Legs, Legs>& map1,
                                           const std::map<Legs, Legs>& map2);

      Tensor<device, Base> contract(const Tensor<device, Base>& tensor2,
                                    const std::vector<Legs>& legs1,
                                    const std::vector<Legs>& legs2,
                                    const std::map<Legs, Legs>& map1 = {},
                                    const std::map<Legs, Legs>& map2 = {}) const {
        return std::move(Tensor<device, Base>::contract(*this, tensor2, legs1, legs2, map1, map2));
      } // contract

      Tensor<device, Base> multiple(const Tensor<device, Base>& other, const Legs& position) const;

      friend class svd_res;
      class svd_res {
       public:
        Tensor<device, Base> U;
        Tensor<device, Base> S;
        Tensor<device, Base> V;
      }; // class svd_res

      svd_res svd(const std::vector<Legs>& input_u_legs, const Legs& new_u_legs, const Legs& new_v_legs, const Rank& cut=-1) const;

      friend class qr_res;
      class qr_res {
       public:
        Tensor<device, Base> Q;
        Tensor<device, Base> R;
      }; // class qr_res

      qr_res qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const;
    }; // class Tensor
  } // namespace tensor
} // namespace TAT

#include "Tensor/norm.hpp"
#include "Tensor/transpose.hpp"
#include "Tensor/contract.hpp"
#include "Tensor/multiple.hpp"
#include "Tensor/svd.hpp"
#include "Tensor/qr.hpp"
#include "Tensor/scalar.hpp"
#include "Tensor/io.hpp"

#endif // TAT_Tensor_HPP_
