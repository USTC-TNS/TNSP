/** TAT/Data/CPU/norm.hpp
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

#ifndef TAT_Data_CPU_Norm_HPP_
#define TAT_Data_CPU_Norm_HPP_

#include "../CPU.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
    namespace CPU {
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
      template<int n>
      Data<Base> Data<Base>::norm() const {
        Data<Base> res(Size(1));
        *res.get() = norm::run<Base, n>(size, get());
        return std::move(res);
      } // norm
    } // namespace data::CPU
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

#endif // TAT_Data_CPU_Norm_HPP_
