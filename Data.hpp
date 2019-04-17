/* TAT/Data.hpp
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

#ifndef TAT_Data_HPP_
#define TAT_Data_HPP_

#include "TAT.hpp"

namespace TAT {
  namespace data {
    namespace CPU {
#ifdef TAT_USE_CPU
      template<class Base>
      class RealBaseClass;

      template<>
      class RealBaseClass<float> {
       public:
        using type=float;
      };

      template<>
      class RealBaseClass<double> {
       public:
        using type=double;
      };

      template<>
      class RealBaseClass<std::complex<float>> {
       public:
        using type=float;
      };

      template<>
      class RealBaseClass<std::complex<double>> {
       public:
        using type=double;
      };

      template<class Base>
      using RealBase = typename RealBaseClass<Base>::type;

      namespace transpose {
        template<class Base>
        void run(const std::vector<Rank>& plan, const std::vector<Size>& dims, const Base* src, Base* dst) {
          std::vector<int> int_plan(plan.begin(), plan.end());
          std::vector<int> int_dims(dims.begin(), dims.end());
          hptt::create_plan(int_plan.data(), int_plan.size(),
                            1, src, int_dims.data(), NULL,
                            0, dst, NULL,
                            hptt::ESTIMATE, 1, NULL, 1)->execute();
        } // run
      } // namespace data::CPU::transpose

      namespace contract {
        template<class Base>
        void run(Base* data,
                 const Base* data1,
                 const Base* data2,
                 const Size& m,
                 const Size& n,
                 const Size& k);

        template<>
        void run<float>(float* data,
                        const float* data1,
                        const float* data2,
                        const Size& m,
                        const Size& n,
                        const Size& k) {
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      m, n, k,
                      1, const_cast<float*>(data1), k, const_cast<float*>(data2), n,
                      0, data, n);
        } // run<float>

        template<>
        void run<double>(double* data,
                         const double* data1,
                         const double* data2,
                         const Size& m,
                         const Size& n,
                         const Size& k) {
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      m, n, k,
                      1, const_cast<double*>(data1), k, const_cast<double*>(data2), n,
                      0, data, n);
        } // run<double>

        template<>
        void run<std::complex<float>>(std::complex<float>* data,
                                      const std::complex<float>* data1,
                                      const std::complex<float>* data2,
                                      const Size& m,
                                      const Size& n,
        const Size& k) {
          std::complex<float> alpha = 1;
          std::complex<float> beta = 0;
          cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      m, n, k,
                      &alpha, const_cast<std::complex<float>*>(data1), k, const_cast<std::complex<float>*>(data2), n,
                      &beta, data, n);
        } // run<std::complex<float>>

        template<>
        void run<std::complex<double>>(std::complex<double>* data,
                                       const std::complex<double>* data1,
                                       const std::complex<double>* data2,
                                       const Size& m,
                                       const Size& n,
        const Size& k) {
          std::complex<double> alpha = 1;
          std::complex<double> beta = 0;
          cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      m, n, k,
                      &alpha, const_cast<std::complex<double>*>(data1), k, const_cast<std::complex<double>*>(data2), n,
                      &beta, data, n);
        } // run<std::complex<double>>
      } // namespace data::CPU::contract

      namespace multiple {
        template<class Base>
        void run(Base* res_data, const Base* src_data, const Base* other_data, const Size& a, const Size& b, const Size& c) {
          for (Size i=0; i<a; i++) {
            for (Size j=0; j<b; j++) {
              Base v = other_data[j];
              for (Size k=0; k<c; k++) {
                *(res_data++) = *(src_data++) * v;
              } // for k
            } // for j
          } // for i
        } // run
      } // namespace data::CPU::multiple

      namespace svd {
#if (defined TAT_USE_GESVD) || (defined TAT_USE_GESDD)
        template<class Base>
        void run(const Size& m, const Size& n, const Size& min, Base* a, Base* u, Base* s, Base* vt);

        template<>
        void run<float>(const Size& m, const Size& n, const Size& min, float* a, float* u, float* s, float* vt) {
#ifdef TAT_USE_GESDD
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S', m, n, a, n, s, u, min, vt, n);
#endif // TAT_USE_GESDD
#ifdef TAT_USE_GESVD
          auto superb = new float[min-1];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, a, n, s, u, min, vt, n, superb);
          delete[] superb;
#endif // TAT_USE_GESVD
          assert(res==0);
        } // run<float>

        template<>
        void run<double>(const Size& m, const Size& n, const Size& min, double* a, double* u, double* s, double* vt) {
#ifdef TAT_USE_GESDD
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', m, n, a, n, s, u, min, vt, n);
#endif // TAT_USE_GESDD
#ifdef TAT_USE_GESVD
          auto superb = new double[min-1];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, a, n, s, u, min, vt, n, superb);
          delete[] superb;
#endif // TAT_USE_GESVD
          assert(res==0);
        } // run<double>

        template<>
        void run<std::complex<float>>(const Size& m, const Size& n, const Size& min, std::complex<float>* a, std::complex<float>* u, std::complex<float>* s, std::complex<float>* vt) {
#ifdef TAT_USE_GESDD
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_cgesdd(LAPACK_ROW_MAJOR, 'S', m, n, a, n, s, u, min, vt, n);
#endif // TAT_USE_GESDD
#ifdef TAT_USE_GESVD
          auto superb = new float[min-1];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_cgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, a, n, reinterpret_cast<float*>(s), u, min, vt, n, superb);
          //for (int i=min-1; i>=0; i--) {
          //  s[i] = reinterpret_cast<float*>(s)[i];
          //} // for S
          delete[] superb;
#endif // TAT_USE_GESVD
          assert(res==0);
        } // run<std::complex<float>>

        template<>
        void run<std::complex<double>>(const Size& m, const Size& n, const Size& min, std::complex<double>* a, std::complex<double>* u, std::complex<double>* s, std::complex<double>* vt) {
#ifdef TAT_USE_GESDD
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_zgesdd(LAPACK_ROW_MAJOR, 'S', m, n, a, n, s, u, min, vt, n);
#endif // TAT_USE_GESDD
#ifdef TAT_USE_GESVD
          auto superb = new double[min-1];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, a, n, reinterpret_cast<double*>(s), u, min, vt, n, superb);
          //for (int i=min-1; i>=0; i--) {
          //  s[i] = reinterpret_cast<double*>(s)[i];
          //} // for S
          delete[] superb;
#endif // TAT_USE_GESVD
          assert(res==0);
        } // run<std::complex<double>>

        template<class Base>
        Data<Base> cut(const Data<Base>& other,
                       const Size& m1,
                       const Size& n1,
                       const Size& m2,
                       const Size& n2) {
          (void)m1; // avoid warning of unused when NDEBUG
          Data<Base> res(m2*n2);
          assert(n2<=n1);
          assert(m2<=m1);
          if (n2==n1) {
            std::memcpy(res.get(), other.get(), n2*m2*sizeof(Base));
          } else {
            Base* dst = res.get();
            const Base* src = other.get();
            Size size = n2*sizeof(Base);
            for (Size i=0; i<m2; i++) {
              std::memcpy(dst, src, size);
              dst += n2;
              src += n1;
            } // for i
          } // if
          return std::move(res);
        } // cut

        template<class Base>
        Data<Base> cutS(const Data<RealBase<Base>>& other,
                        const Size& n1,
                        const Size& n2) {
          Data<Base> res(n2);
          assert(n2<=n1);
          Base* dst = res.get();
          const RealBase<Base>* src = other.get();
          for (Size i=0; i<n2; i++) {
            dst[i] = src[i];
          } // for i
          return std::move(res);
        } // cutS
#endif // TAT_USE_GESVD TAT_USE_GESDD

#ifdef TAT_USE_GESVDX
        template<class Base>
        void run(const Size& m, const Size& n, const Size& min, const Size& cut, Base* a, Base* u, Base* s, Base* vt);

        template<>
        void run<float>(const Size& m, const Size& n, const Size& min, const Size& cut, float* a, float* u, float* s, float* vt) {
          lapack_int ns;
          auto superb = new lapack_int[12*min];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_sgesvdx(LAPACK_ROW_MAJOR, 'V', 'V', 'I', m, n, a, n, 0, 0, 1, cut, &ns, s, u, cut, vt, n, superb);
          assert(res==0);
          assert(ns==lapack_int(cut));
          delete[] superb;
        } // run<float>

        template<>
        void run<double>(const Size& m, const Size& n, const Size& min, const Size& cut, double* a, double* u, double* s, double* vt) {
          lapack_int ns;
          auto superb = new lapack_int[12*min];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_dgesvdx(LAPACK_ROW_MAJOR, 'V', 'V', 'I', m, n, a, n, 0, 0, 1, cut, &ns, s, u, cut, vt, n, superb);
          assert(res==0);
          assert(ns==lapack_int(cut));
          delete[] superb;
        } // run<double>

        template<>
        void run<std::complex<float>>(const Size& m, const Size& n, const Size& min, const Size& cut, std::complex<float>* a, std::complex<float>* u, std::complex<float>* s, std::complex<float>* vt) {
          lapack_int ns;
          auto superb = new lapack_int[12*min];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_cgesvdx(LAPACK_ROW_MAJOR, 'V', 'V', 'I', m, n, a, n, 0, 0, 1, cut, &ns, s, u, cut, vt, n, superb);
          assert(res==0);
          assert(ns==lapack_int(cut));
          delete[] superb;
        } // run<std::complex<float>>

        template<>
        void run<std::complex<double>>(const Size& m, const Size& n, const Size& min, const Size& cut, std::complex<double>* a, std::complex<double>* u, std::complex<double>* s, std::complex<double>* vt) {
          lapack_int ns;
          auto superb = new lapack_int[12*min];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_zgesvdx(LAPACK_ROW_MAJOR, 'V', 'V', 'I', m, n, a, n, 0, 0, 1, cut, &ns, s, u, cut, vt, n, superb);
          assert(res==0);
          assert(ns==lapack_int(cut));
          delete[] superb;
        } // run<std::complex<double>>
#endif // TAT_USE_GESVDX
      } // namespace data::CPU::svd

      namespace qr {
        template<class Base>
        void geqrf(Base* A, Base* tau, const Size& m, const Size& n);

        template<>
        void geqrf<float>(float* A, float* tau, const Size& m, const Size& n) {
#ifdef TAT_USE_GEQP3
          auto jpvt = new lapack_int[n];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_sgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt, tau);
          delete[] jpvt;
#endif // TAT_USE_GEQP3
#ifdef TAT_USE_GEQRF
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
#endif // TAT_USE_GEQRF
          assert(res==0);
        } // geqrf<float>

        template<>
        void geqrf<double>(double* A, double* tau, const Size& m, const Size& n) {
#ifdef TAT_USE_GEQP3
          auto jpvt = new lapack_int[n];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt, tau);
          delete[] jpvt;
#endif // TAT_USE_GEQP3
#ifdef TAT_USE_GEQRF
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
#endif // TAT_USE_GEQRF
          assert(res==0);
        } // geqrf<double>

        template<>
        void geqrf<std::complex<float>>(std::complex<float>* A, std::complex<float>* tau, const Size& m, const Size& n) {
#ifdef TAT_USE_GEQP3
          auto jpvt = new lapack_int[n];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_cgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt, tau);
          delete[] jpvt;
#endif // TAT_USE_GEQP3
#ifdef TAT_USE_GEQRF
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
#endif // TAT_USE_GEQRF
          assert(res==0);
        } // geqrf<std::complex<float>>

        template<>
        void geqrf<std::complex<double>>(std::complex<double>* A, std::complex<double>* tau, const Size& m, const Size& n) {
#ifdef TAT_USE_GEQP3
          auto jpvt = new lapack_int[n];
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_zgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt, tau);
          delete[] jpvt;
#endif // TAT_USE_GEQP3
#ifdef TAT_USE_GEQRF
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
#endif // TAT_USE_GEQRF
          assert(res==0);
        } // geqrf<std::complex<double>>

        template<class Base>
        void orgqr(Base* A, const Base* tau, const Size& m, const Size& min);

        template<>
        void orgqr<float>(float* A, const float* tau, const Size& m, const Size& min) {
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, min, min, A, min, tau);
          assert(res==0);
        } // orgqr<float>

        template<>
        void orgqr<double>(double* A, const double* tau, const Size& m, const Size& min) {
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, min, min, A, min, tau);
          assert(res==0);
        } // orgqr<double>

        template<>
        void orgqr<std::complex<float>>(std::complex<float>* A, const std::complex<float>* tau, const Size& m, const Size& min) {
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_cungqr(LAPACK_ROW_MAJOR, m, min, min, A, min, tau);
          assert(res==0);
        } // orgqr<std::complex<float>>

        template<>
        void orgqr<std::complex<double>>(std::complex<double>* A, const std::complex<double>* tau, const Size& m, const Size& min) {
#ifndef NDEBUG
          auto res =
#endif // NDEBUG
            LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, min, min, A, min, tau);
          assert(res==0);
        } // orgqr<std::complex<double>>

        template<class Base>
        void run(Base* Q, Base* R, const Size& m, const Size& n, const Size& min_mn) {
          auto tau = new Base[min_mn];
          geqrf(R, tau, m, n);
          // copy to Q and delete unused R
          if (min_mn==n) {
            std::memcpy(Q, R, m*n*sizeof(Base));
          } else {
            // Q is m*m
            auto q = Q;
            auto r = R;
            for (Size i=0; i<m; i++) {
              std::memcpy(q, r, m*sizeof(Base));
              q += m;
              r += n;
            } // for i
          } // if
          orgqr(Q, tau, m, min_mn);
          auto r = R;
          for (Size i=1; i<min_mn; i++) {
            r += n;
            std::memset(r, 0, i*sizeof(Base));
          } // for i
          delete[] tau;
        } // run
      } // namespace data::CPU::qr

      namespace norm {
        template<class Base>
        void vAbs(const Size& size, const Base* a, RealBase<Base>* y);

        template<>
        void vAbs<float>(const Size& size, const float* a, float* y) {
          vsAbs(size, a, y);
        } // vAbs<float>

        template<>
        void vAbs<double>(const Size& size, const double* a, double* y) {
          vdAbs(size, a, y);
        } // vAbs<double>

        template<>
        void vAbs<std::complex<float>>(const Size& size, const std::complex<float>* a, float* y) {
          vcAbs(size, a, y);
        } // vAbs<std::complex<float>>

        template<>
        void vAbs<std::complex<double>>(const Size& size, const std::complex<double>* a, double* y) {
          vzAbs(size, a, y);
        } // vAbs<std::complex<double>>

        template<class Base>
        void vPowx(const Size& size, const Base* a, const Base& n, Base* y);

        template<>
        void vPowx<float>(const Size& size, const float* a, const float& n, float* y) {
          vsPowx(size, a, n, y);
        } // vPowx<float>

        template<>
        void vPowx<double>(const Size& size, const double* a, const double& n, double* y) {
          vdPowx(size, a, n, y);
        } // vPowx<double>

        template<class Base>
        void vSqr(const Size& size, const Base* a, Base* y);

        template<>
        void vSqr<float>(const Size& size, const float* a, float* y) {
          vsSqr(size, a, y);
        } // vSqr<float>

        template<>
        void vSqr<double>(const Size& size, const double* a, double* y) {
          vdSqr(size, a, y);
        } // vSqr<double>

        template<class Base>
        Base asum(const Size& size, const Base* a);

        template<>
        float asum<float>(const Size& size, const float* a) {
          return cblas_sasum(size, a, 1);
        } // asum<float>

        template<>
        double asum<double>(const Size& size, const double* a) {
          return cblas_dasum(size, a, 1);
        } // asum<double>

        template<class Base>
        CBLAS_INDEX iamax(const Size& size, const Base* a);

        template<>
        CBLAS_INDEX iamax<float>(const Size& size, const float* a) {
          return cblas_isamax(size, a, 1);
        } // iamax<float>

        template<>
        CBLAS_INDEX iamax<double>(const Size& size, const double* a) {
          return cblas_idamax(size, a, 1);
        } // iamax<double>

        template<>
        CBLAS_INDEX iamax<std::complex<float>>(const Size& size, const std::complex<float>* a) {
          return cblas_icamax(size, a, 1);
        } // iamax<std::complex<float>>

        template<>
        CBLAS_INDEX iamax<std::complex<double>>(const Size& size, const std::complex<double>* a) {
          return cblas_izamax(size, a, 1);
        } // iamax<std::complex<double>>

        template<class Base, int n>
        Base run(const Size& size, const Base* data) {
          if (n==-1) {
            auto i = iamax<Base>(size, data);
            return std::abs(data[i]);
          }
          auto tmp = new RealBase<Base>[size];
          vAbs<Base>(size, data, tmp);
          if (n==2) {
            vSqr<RealBase<Base>>(size, tmp, tmp);
          } else {
            vPowx<RealBase<Base>>(size, tmp, Base(n), tmp);
          }
          auto res = asum<RealBase<Base>>(size, tmp);
          delete[] tmp;
          return res;
        } // run
      } // namespace data::CPU::norm

      template<class Base>
      class Data {
       public:
        Data() : size(0), base() {}

        Size size;
        std::unique_ptr<Base[]> base;

        ~Data() = default;
        Data(Data<Base>&& other) = default;
        Data<Base>& operator=(Data<Base>&& other) = default;
        Data(Size _size) : size(_size) {
          base = std::unique_ptr<Base[]>(new Base[size]);
        }
        Data(const Data<Base>& other) {
          new (this) Data(other.size);
          std::memcpy(get(), other.get(), size*sizeof(Base));
#ifndef TAT_NOT_WARN_COPY
          std::clog << "Copying Data..." << std::endl;
#endif // TAT_NOT_WARN_COPY
        }
        Data<Base>& operator=(const Data<Base>& other) {
          new (this) Data(other);
          return *this;
        }
        Data(const Base& num) {
          new (this) Data(Size(1));
          *base.get() = num;
        }

        const Base* get() const {
          return base.get();
        } // get
        Base* get() {
          return base.get();
        } // get

        Data<Base>& set_test() {
          for (Size i=0; i<size; i++) {
            base[i] = Base(i);
          } // for i
          return *this;
        } // set_test
        Data<Base>& set_zero() {
          for (Size i=0; i<size; i++) {
            base[i] = Base(0);
          } // for i
          return *this;
        } // set_zero
        Data<Base>& set_random(const std::function<Base()>& random) {
          for (Size i=0; i<size; i++) {
            base[i] = random();
          } // for i
          return *this;
        } // set_random
        Data<Base>& set_constant(Base num) {
          for (Size i=0; i<size; i++) {
            base[i] = num;
          } // for i
          return *this;
        } // set_constant

        template<int n>
        Data<Base> norm() const {
          Data<Base> res(Size(1));
          *res.get() = norm::run<Base, n>(size, get());
          return std::move(res);
        } // norm

        template<class Base2, ENABLE_IF(std::is_scalar<Base2>)>
        Data<Base2> to() const {
          Data<Base2> res(size);
          for (Size i=0; i<size; i++) {
            res.base[i] = Base2(base[i]);
          } // for i
          return std::move(res);
        } // to

        Data<Base> transpose(const std::vector<Size>& dims,
                             const std::vector<Rank>& plan) const {
          Data<Base> res(size);
          assert(dims.size()==plan.size());
          transpose::run(plan, dims, get(), res.get());
          return std::move(res);
        } // transpose

        static Data<Base> contract(const Data<Base>& data1,
                                   const Data<Base>& data2,
                                   const std::vector<Size>& dims1,
                                   const std::vector<Size>& dims2,
                                   const std::vector<Rank>& plan1,
                                   const std::vector<Rank>& plan2,
                                   const Size& m, const Size& k, const Size& n) {
          assert(m*k==data1.size);
          assert(k*n==data2.size);
          Data<Base> a = data1.transpose(dims1, plan1);
          Data<Base> b = data2.transpose(dims2, plan2);
          // wasted transpose
          Data<Base> res(m*n);
          contract::run<Base>(res.get(), a.get(), b.get(), m, n, k);
          return std::move(res);
        } // contract

        Data<Base> contract(const Data<Base>& data2,
                            const std::vector<Size>& dims1,
                            const std::vector<Size>& dims2,
                            const std::vector<Rank>& plan1,
                            const std::vector<Rank>& plan2,
                            const Size& m, const Size& k, const Size& n) const {
          return std::move(Data<Base>::contract(*this, data2, dims1, dims2, plan1, plan2, m, k, n));
        } // contract

        Data<Base> multiple(const Data<Base>& other, const Size& a, const Size& b, const Size& c) const {
          Data<Base> res(size);
          assert(b==other.size);
          assert(a*b*c==size);
          multiple::run<Base>(res.get(), get(), other.get(), a, b, c);
          return std::move(res);
        } // multiple

        friend class svd_res;
        class svd_res {
         public:
          Data<Base> U;
          Data<Base> S;
          Data<Base> V;
        }; // class svd_res

        svd_res svd(const std::vector<Size>& dims,
                    const std::vector<Rank>& plan,
                    const Size& u_size,
                    const Size& v_size,
                    const Size& min_mn,
                    const Size& cut) const {
          assert(size%u_size==0);
          Size cut_dim = (cut<min_mn)?cut:min_mn;
          // -1 > any integer
          Data<Base> tmp = transpose(dims, plan);
          // used in svd, gesvd will destroy it
          svd_res res;
#ifdef TAT_USE_GESVDX
          res.U = Data<Base>(u_size*cut_dim);
          res.S = Data<Base>(min_mn);
          res.S.size = cut_dim;
          res.V = Data<Base>(cut_dim*v_size);
          svd::run(u_size, v_size, min_mn, cut_dim, tmp.get(), res.U.get(), res.S.get(), res.V.get());
#endif // TAT_USE_GESVDX
#if (defined TAT_USE_GESVD) || (defined TAT_USE_GESDD)
          res.U = Data<Base>(u_size*min_mn);
          auto tmpS = Data<RealBase<Base>>(min_mn);
          res.V = Data<Base>(min_mn*v_size);
          svd::run(u_size, v_size, min_mn, tmp.get(), res.U.get(), tmpS.get(), res.V.get());
          if (cut_dim!=min_mn) {
            res.U = svd::cut(res.U, u_size, min_mn, u_size, cut_dim);
            res.V = svd::cut(res.V, min_mn, v_size, cut_dim, v_size);
          }
          res.S = svd::cutS<Base>(tmpS, min_mn, cut_dim);
#endif // TAT_USE_GESVD TAT_USE_GESDD
          return std::move(res);
        } // svd

        friend class qr_res;
        class qr_res {
         public:
          Data<Base> Q;
          Data<Base> R;
        }; // class qr_res

        qr_res qr(const std::vector<Size>& dims,
                  const std::vector<Rank>& plan,
                  const Size& q_size,
                  const Size& r_size,
                  const Size& min_mn) const {
          assert(size==q_size*r_size);
          qr_res res;
          res.Q = Data<Base>(q_size*min_mn);
          res.R = transpose(dims, plan);
          // R is q_size*r_size, should be min_mn*r_size
          qr::run(res.Q.get(), res.R.get(), q_size, r_size, min_mn);
          res.R.size = min_mn*r_size;
          return std::move(res);
        } // qr
      }; // class Data

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

      inline namespace io {
        template<class Base>
        std::ostream& operator<<(std::ostream& out, const Data<Base>& value) {
          out << "{\"" << rang::fgB::green << "size\": " << value.size << "" << rang::fg::reset << ", " << rang::fg::yellow << "\"base\": [";
          for (Size i=0; i<value.size-1; i++) {
            out << value.base[i] << ", ";
          } // for i
          if (value.size!=0) {
            out << value.base[value.size-1];
          } // if
          out << "]" << rang::fg::reset << "}";
          return out;
        } // operator<<

        template<class Base>
        std::ofstream& operator<<(std::ofstream& out, const Data<Base>& value) {
          out.write((char*)&value.size, sizeof(Size));
          out.write((char*)value.get(), value.size*sizeof(Base));
          return out;
        } // operator<<

        template<class Base>
        std::ifstream& operator>>(std::ifstream& in, Data<Base>& value) {
          in.read((char*)&value.size, sizeof(Size));
          value.base = std::unique_ptr<Base[]>(new Base[value.size]);
          in.read((char*)value.get(), value.size*sizeof(Base));
          return in;
        } // operator<<
      } // namespace data::CPU::io
#endif // TAT_USE_CPU
    } // namespace CPU
  } // namespace data
} // namespace TAT

#endif // TAT_Data_HPP_
