/** TAT/scalar.hpp
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

#ifndef TAT_Scalar_HPP_
#define TAT_Scalar_HPP_

#include "../TAT.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
      inline namespace scalar {
        template<class Base>
        void vLinearFrac(const Size& n, const Base* a, const Base* b,
                         const Base& sa, const Base& oa, const Base& sb, const Base& ob,
                         Base* y);
        // y = (a*sa + oa)/(b*sb + ob)

        template<>
        void vLinearFrac<float>(const Size& n, const float* a, const float* b,
                                const float& sa, const float& oa, const float& sb, const float& ob,
                                float* y) {
          vsLinearFrac(n, a, b, sa, oa, sb, ob, y);
        } // vLinearFrac

        template<>
        void vLinearFrac<double>(const Size& n, const double* a, const double* b,
                                 const double& sa, const double& oa, const double& sb, const double& ob,
                                 double* y) {
          vdLinearFrac(n, a, b, sa, oa, sb, ob, y);
        } // vLinearFrac

        template<>
        void vLinearFrac<std::complex<float>>(const Size& n, const std::complex<float>* a, const std::complex<float>* b,
                                              const std::complex<float>& sa, const std::complex<float>& oa, const std::complex<float>& sb, const std::complex<float>& ob,
        std::complex<float>* y) {
          //vcLinearFrac(n, a, b, sa, oa, sb, ob, y);
          for (Size i=0; i<n; i++) {
            y[i] = (a[i]*sa + oa)/(b[i]*sb + ob);
          } // for
        } // vLinearFrac

        template<>
        void vLinearFrac<std::complex<double>>(const Size& n, const std::complex<double>* a, const std::complex<double>* b,
                                               const std::complex<double>& sa, const std::complex<double>& oa, const std::complex<double>& sb, const std::complex<double>& ob,
        std::complex<double>* y) {
          //vzLinearFrac(n, a, b, sa, oa, sb, ob, y);
          for (Size i=0; i<n; i++) {
            y[i] = (a[i]*sa + oa)/(b[i]*sb + ob);
          } // for
        } // vLinearFrac

        template<class Base>
        void LinearFrac(const Data<Base>& src, Data<Base>& dst,
                        const Base& sa, const Base& oa, const Base& sb, const Base& ob) {
          assert(src.size==dst.size);
          vLinearFrac<Base>(src.size, src.get(), src.get(), sa, oa, sb, ob, dst.get());
        } // LinearFrac

        template<class Base>
        void vAdd(const Size& n, const Base* a, const Base* b, Base* y);

        template<>
        void vAdd<float>(const Size& n, const float* a, const float* b, float* y) {
          vsAdd(n, a, b, y);
        } // vAdd

        template<>
        void vAdd<double>(const Size& n, const double* a, const double* b, double* y) {
          vdAdd(n, a, b, y);
        } // vAdd

        template<>
        void vAdd<std::complex<float>>(const Size& n, const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) {
          vcAdd(n, a, b, y);
        } // vAdd

        template<>
        void vAdd<std::complex<double>>(const Size& n, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) {
          vzAdd(n, a, b, y);
        } // vAdd

        template<class Base>
        void Add(const Data<Base>& a, const Data<Base>& b, Data<Base>& y) {
          assert(a.size==b.size);
          vAdd<Base>(a.size, a.get(), b.get(), y.get());
        } // Add

        template<class Base>
        void vSub(const Size& n, const Base* a, const Base* b, Base* y);

        template<>
        void vSub<float>(const Size& n, const float* a, const float* b, float* y) {
          vsSub(n, a, b, y);
        } // vSub

        template<>
        void vSub<double>(const Size& n, const double* a, const double* b, double* y) {
          vdSub(n, a, b, y);
        } // vSub

        template<>
        void vSub<std::complex<float>>(const Size& n, const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) {
          vcSub(n, a, b, y);
        } // vSub

        template<>
        void vSub<std::complex<double>>(const Size& n, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) {
          vzSub(n, a, b, y);
        } // vSub

        template<class Base>
        void Sub(const Data<Base>& a, const Data<Base>& b, Data<Base>& y) {
          assert(a.size==b.size);
          vSub<Base>(a.size, a.get(), b.get(), y.get());
        } // Sub

        template<class Base>
        void vMul(const Size& n, const Base* a, const Base* b, Base* y);

        template<>
        void vMul<float>(const Size& n, const float* a, const float* b, float* y) {
          vsMul(n, a, b, y);
        } // vMul

        template<>
        void vMul<double>(const Size& n, const double* a, const double* b, double* y) {
          vdMul(n, a, b, y);
        } // vMul

        template<>
        void vMul<std::complex<float>>(const Size& n, const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) {
          vcMul(n, a, b, y);
        } // vMul

        template<>
        void vMul<std::complex<double>>(const Size& n, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) {
          vzMul(n, a, b, y);
        } // vMul

        template<class Base>
        void Mul(const Data<Base>& a, const Data<Base>& b, Data<Base>& y) {
          assert(a.size==b.size);
          vMul<Base>(a.size, a.get(), b.get(), y.get());
        } // Mul

        template<class Base>
        void vDiv(const Size& n, const Base* a, const Base* b, Base* y);

        template<>
        void vDiv<float>(const Size& n, const float* a, const float* b, float* y) {
          vsDiv(n, a, b, y);
        } // vDiv

        template<>
        void vDiv<double>(const Size& n, const double* a, const double* b, double* y) {
          vdDiv(n, a, b, y);
        } // vDiv

        template<>
        void vDiv<std::complex<float>>(const Size& n, const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) {
          vcDiv(n, a, b, y);
        } // vDiv

        template<>
        void vDiv<std::complex<double>>(const Size& n, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) {
          vzDiv(n, a, b, y);
        } // vDiv

        template<class Base>
        void Div(const Data<Base>& a, const Data<Base>& b, Data<Base>& y) {
          assert(a.size==b.size);
          vDiv<Base>(a.size, a.get(), b.get(), y.get());
        } // Div

        template<class Base>
        Data<Base>& operator*=(Data<Base>& a, const Data<Base>& b) {
          if (b.size==1) {
            LinearFrac<Base>(a, a, *b.get(), 0, 0, 1);
          } else {
            Mul<Base>(a, b, a);
          } // if
          return a;
        } // operator*=

        template<class Base>
        Data<Base> operator*(const Data<Base>& a, const Data<Base>& b) {
          if (a.size==1) {
            Data<Base> res(b.size);
            LinearFrac<Base>(b, res, *a.get(), 0, 0, 1);
            return std::move(res);
          } // if
          if (b.size==1) {
            Data<Base> res(a.size);
            LinearFrac<Base>(a, res, *b.get(), 0, 0, 1);
            return std::move(res);
          } // if
          Data<Base> res(a.size);
          Mul<Base>(a, b, res);
          return std::move(res);
        } // operator*

        template<class Base>
        Data<Base>& operator/=(Data<Base>& a, const Data<Base>& b) {
          if (b.size==1) {
            LinearFrac<Base>(a, a, 1, 0, 0, *b.get());
          } else {
            Div<Base>(a, b, a);
          } // if
          return a;
        } // operator/=

        template<class Base>
        Data<Base> operator/(const Data<Base>& a, const Data<Base>& b) {
          if (a.size==1) {
            Data<Base> res(b.size);
            LinearFrac<Base>(b, res, 0, *a.get(), 1, 0);
            return std::move(res);
          } // if
          if (b.size==1) {
            Data<Base> res(a.size);
            LinearFrac<Base>(a, res, 1, 0, 0, *b.get());
            return std::move(res);
          } // if
          Data<Base> res(a.size);
          Div<Base>(a, b, res);
          return std::move(res);
        } // operator/

        template<class Base>
        Data<Base> operator+(const Data<Base>& a) {
          return Data<Base>(a);
        } // operator+

        template<class Base>
        Data<Base> operator+(Data<Base>&& a) {
          return Data<Base>(std::move(a));
        } // operator+

        template<class Base>
        Data<Base>& operator+=(Data<Base>& a, const Data<Base>& b) {
          if (b.size==1) {
            LinearFrac<Base>(a, a, 1, *b.get(), 0, 1);
          } else {
            Add<Base>(a, b, a);
          } // if
          return a;
        } // operator+=

        template<class Base>
        Data<Base> operator+(const Data<Base>& a, const Data<Base>& b) {
          if (a.size==1) {
            Data<Base> res(b.size);
            LinearFrac<Base>(b, res, 1, *a.get(), 0, 1);
            return std::move(res);
          } // if
          if (b.size==1) {
            Data<Base> res(a.size);
            LinearFrac<Base>(a, res, 1, *b.get(), 0, 1);
            return std::move(res);
          } // if
          Data<Base> res(a.size);
          Add<Base>(a, b, res);
          return std::move(res);
        } // operator+

        template<class Base>
        Data<Base> operator-(const Data<Base>& a) {
          Data<Base> res(a.size);
          LinearFrac<Base>(a, res, -1, 0, 0, 1);
          return std::move(res);
        } // operator-

        template<class Base>
        Data<Base>& operator-=(Data<Base>& a, const Data<Base>& b) {
          if (b.size==1) {
            LinearFrac<Base>(a, a, 1, -*b.get(), 0, 1);
          } else {
            Sub<Base>(a, b, a);
          } // if
          return a;
        } // operator-=

        template<class Base>
        Data<Base> operator-(const Data<Base>& a, const Data<Base>& b) {
          if (a.size==1) {
            Data<Base> res(b.size);
            LinearFrac<Base>(b, res, -1, *a.get(), 0, 1);
            return std::move(res);
          } // if
          if (b.size==1) {
            Data<Base> res(a.size);
            LinearFrac<Base>(a, res, 1, -*b.get(), 0, 1);
            return std::move(res);
          } // if
          Data<Base> res(a.size);
          Sub<Base>(a, b, res);
          return std::move(res);
        } // operator-
      } // namespace data::CPU::scalar
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

namespace TAT {
  namespace block {
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

      template<class Base>
      Block<Base>& operator*=(Block<Base>& a, const Block<Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data *= b.data;
        return a;
      } // operator*=

      template<class Base>
      Block<Base> operator*(const Block<Base>& a, const Block<Base>& b) {
        Block<Base> res;
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

      template<class Base>
      Block<Base>& operator/=(Block<Base>& a, const Block<Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data /= b.data;
        return a;
      } // operator/=

      template<class Base>
      Block<Base> operator/(const Block<Base>& a, const Block<Base>& b) {
        Block<Base> res;
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

      template<class Base>
      Block<Base> operator+(const Block<Base>& a) {
        Block<Base> res;
        res.dims = a.dims;
        res.data = + a.data;
        return std::move(res);
      } // operator+

      template<class Base>
      Block<Base> operator+(Block<Base>&& a) {
        Block<Base> res;
        res.dims = std::move(a.dims);
        res.data = + std::move(a.data);
        return std::move(res);
      } // operator+

      template<class Base>
      Block<Base>& operator+=(Block<Base>& a, const Block<Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data += b.data;
        return a;
      } // operator+=

      template<class Base>
      Block<Base> operator+(const Block<Base>& a, const Block<Base>& b) {
        Block<Base> res;
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

      template<class Base>
      Block<Base> operator-(const Block<Base>& a) {
        Block<Base> res;
        res.dims = a.dims;
        res.data = - a.data;
        return std::move(res);
      } // operator-

      template<class Base>
      Block<Base>& operator-=(Block<Base>& a, const Block<Base>& b) {
        if (b.dims.size()!=0) {
          assert(a.dims==b.dims);
        } // if
        a.data -= b.data;
        return a;
      } // operator-=

      template<class Base>
      Block<Base> operator-(const Block<Base>& a, const Block<Base>& b) {
        Block<Base> res;
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
    } // namespace block::scalar
  } // namespace block
} // namespace TAT

namespace TAT {
  namespace node {
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

      template<class Base>
      Node<Base>& operator*=(Node<Base>& a, const Node<Base>& b) {
        if (b.legs.size()!=0) {
          assert(a.legs==b.legs);
        }
        a.tensor *= b.tensor;
        return a;
      } // operator*=

      template<class Base>
      Node<Base> operator*(const Node<Base>& a, const Node<Base>& b) {
        Node<Base> res;
        if (b.legs.size()==0) {
          res.legs = a.legs;
        } else if (a.legs.size()==0) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.tensor = a.tensor * b.tensor;
        return std::move(res);
      } // operator*

      template<class Base>
      Node<Base>& operator/=(Node<Base>& a, const Node<Base>& b) {
        if (b.legs.size()!=0) {
          assert(a.legs==b.legs);
        } // if
        a.tensor /= b.tensor;
        return a;
      } // operator/=

      template<class Base>
      Node<Base> operator/(const Node<Base>& a, const Node<Base>& b) {
        Node<Base> res;
        if (b.legs.size()==0) {
          res.legs = a.legs;
        } else if (a.legs.size()==0) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.tensor = a.tensor / b.tensor;
        return std::move(res);
      } // operator/

      template<class Base>
      Node<Base> operator+(const Node<Base>& a) {
        Node<Base> res;
        res.legs = a.legs;
        res.tensor = + a.tensor;
        return std::move(res);
      } // operator+

      template<class Base>
      Node<Base> operator+(Node<Base>&& a) {
        Node<Base> res;
        res.legs = std::move(a.legs);
        res.tensor = + std::move(a.tensor);
        return std::move(res);
      } // operator+

      template<class Base>
      Node<Base>& operator+=(Node<Base>& a, const Node<Base>& b) {
        if (b.legs.size()!=0) {
          assert(a.legs==b.legs);
        } // if
        a.tensor += b.tensor;
        return a;
      } // operator+=

      template<class Base>
      Node<Base> operator+(const Node<Base>& a, const Node<Base>& b) {
        Node<Base> res;
        if (b.legs.size()==0) {
          res.legs = a.legs;
        } else if (a.legs.size()==0) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.tensor = a.tensor + b.tensor;
        return std::move(res);
      } // operator+

      template<class Base>
      Node<Base> operator-(const Node<Base>& a) {
        Node<Base> res;
        res.legs = a.legs;
        res.tensor = - a.tensor;
        return std::move(res);
      } // operator-

      template<class Base>
      Node<Base>& operator-=(Node<Base>& a, const Node<Base>& b) {
        if (b.legs.size()!=0) {
          assert(a.legs==b.legs);
        } // if
        a.tensor -= b.tensor;
        return a;
      } // operator-=

      template<class Base>
      Node<Base> operator-(const Node<Base>& a, const Node<Base>& b) {
        Node<Base> res;
        if (b.legs.size()==0) {
          res.legs = a.legs;
        } else if (a.legs.size()==0) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.tensor = a.tensor - b.tensor;
        return std::move(res);
      } // operator-

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base>& operator*=(Node<Base>& a, const B& b) {
        return a*=Node<Base>(b);
      } // operator*=

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base> operator*(const Node<Base>& a, const B& b) {
        return a*Node<Base>(b);
      } // operator*

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base> operator*(const B& b, const Node<Base>& a) {
        return Node<Base>(b)*a;
      } // operator*

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base>& operator/=(Node<Base>& a, const B& b) {
        return a/=Node<Base>(b);
      } // operator/=

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base> operator/(const Node<Base>& a, const B& b) {
        return a/Node<Base>(b);
      } // operator/

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base> operator/(const B& b, const Node<Base>& a) {
        return Node<Base>(b)/a;
      } // operator/

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base>& operator+=(Node<Base>& a, const B& b) {
        return a+=Node<Base>(b);
      } // operator+

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base> operator+(const Node<Base>& a, const B& b) {
        return a+Node<Base>(b);
      } // operator+

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base> operator+(const B& b, const Node<Base>& a) {
        return Node<Base>(b)+a;
      } // operator+

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base>& operator-=(Node<Base>& a, const B& b) {
        return a-=Node<Base>(b);
      } // operator-=

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base> operator-(const Node<Base>& a, const B& b) {
        return a-Node<Base>(b);
      } // operator-

      template<class Base, class B, ENABLE_IF(scalar_tools::is_scalar<B>)>
      Node<Base> operator-(const B& b, const Node<Base>& a) {
        return Node<Base>(b)-a;
      } // operator-
    } // namespace node::scalar
  } // namespace node
} // namespace TAT

#endif // TAT_Scalar_HPP_
