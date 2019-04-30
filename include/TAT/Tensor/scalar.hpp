/* TAT/Tensor/scalar.hpp
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

#ifndef TAT_Tensor_Scalar_HPP_
#define TAT_Tensor_Scalar_HPP_

#include "../Tensor.hpp"

namespace TAT {
  namespace tensor {
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
  } // namespace tensor
} // namespace TAT

#endif // TAT_Tensor_Scalar_HPP_
