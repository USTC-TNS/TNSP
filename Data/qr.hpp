/* TAT/Data/qr.hpp
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

#ifndef TAT_Data_Qr_HPP_
#define TAT_Data_Qr_HPP_

#include "../Data.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
    namespace CPU {
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

      template<class Base>
      typename Data<Base>::qr_res Data<Base>::qr(const std::vector<Size>& dims,
          const std::vector<Rank>& plan,
          const Size& q_size,
          const Size& r_size,
          const Size& min_mn) const {
        assert(size==q_size*r_size);
        qr_res res;
        res.Q = Data<Base>(q_size*min_mn);
        res.R = transpose(dims, plan);
        // R is q_size*r_size, should be min_mn*r_size
        // so if q_size > r_size, R will occupy some unused memory
        qr::run(res.Q.get(), res.R.get(), q_size, r_size, min_mn);
        res.R.size = min_mn*r_size;
        return std::move(res);
      } // qr
    } // namespace data::CPU
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

#endif // TAT_Data_Qr_HPP_
