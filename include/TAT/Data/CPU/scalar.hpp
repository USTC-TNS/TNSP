/* TAT/Data/CPU_scalar.hpp
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

#ifndef TAT_Data_CPU_Scalar_HPP_
#define TAT_Data_CPU_Scalar_HPP_

#include "../CPU.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
    namespace CPU {
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
    } // namespace data::CPU
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

#endif // TAT_Data_CPU_Scalar_HPP_
