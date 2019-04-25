/* TAT/Data/contract.hpp
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

#ifndef TAT_Data_Contract_HPP_
#define TAT_Data_Contract_HPP_

#include "../Data.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
    namespace CPU {
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

      template<class Base>
      Data<Base> Data<Base>::contract(const Data<Base>& data1,
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
    } // namespace data::CPU
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

#endif // TAT_Data_Contract_HPP_
