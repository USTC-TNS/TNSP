/** TAT/svd.hpp
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

#ifndef TAT_Svd_HPP_
#define TAT_Svd_HPP_

#include "../TAT.hpp"

namespace TAT {
  namespace data {
#ifdef TAT_USE_CPU
      namespace svd {
#if (defined TAT_USE_GESVD) || (defined TAT_USE_GESDD)
        template<class Base>
        void run(const Size& m, const Size& n, const Size& min, Base* a, Base* u, scalar_tools::RealBase<Base>* s, Base* vt);

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
        void run<std::complex<float>>(const Size& m, const Size& n, const Size& min, std::complex<float>* a, std::complex<float>* u, float* s, std::complex<float>* vt) {
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
        void run<std::complex<double>>(const Size& m, const Size& n, const Size& min, std::complex<double>* a, std::complex<double>* u, double* s, std::complex<double>* vt) {
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
          } // if continue then copy as a whole or as many part
          return std::move(res);
        } // cut

        template<class Base>
        Data<Base> cutS(const Data<scalar_tools::RealBase<Base>>& other,
                        const Size& n1,
                        const Size& n2) {
          (void)n1; // avoid warning of unused when NDEBUG
          Data<Base> res(n2);
          assert(n2<=n1);
          Base* dst = res.get();
          const scalar_tools::RealBase<Base>* src = other.get();
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

      template<class Base>
      typename Data<Base>::svd_res Data<Base>::svd(const std::vector<Size>& dims,
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
        auto tmpS = Data<scalar_tools::RealBase<Base>>(min_mn);
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
#endif // TAT_USE_CPU
  } // namespace data
} // namespace TAT

namespace TAT {
  namespace block {
    namespace svd {
      void plan(Size& u_size, const Rank& u_rank, const std::vector<Size>& dims) {
        for (Rank i=0; i<u_rank; i++) {
          u_size *= dims[i];
        } // for i
      } // plan
    } // namespace block::svd

    template<class Base>
    typename Block<Base>::svd_res Block<Base>::svd(const std::vector<Rank>& plan, const Rank& u_rank, const Size& cut) const {
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
  } // namespace block
} // namespace TAT

namespace TAT {
  namespace node {
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
    } // namespace node::svd

    template<class Base>
    typename Node<Base>::svd_res Node<Base>::svd(const std::vector<Legs>& input_u_legs, const Legs& new_u_legs, const Legs& new_v_legs, const Rank& cut) const {
      std::vector<Legs> u_legs = internal::in_and_in(legs, input_u_legs);
      svd_res res;
      std::vector<Legs> tmp_legs;
      std::vector<Rank> plan;
      Rank u_rank;
      svd::plan(res.U.legs, res.V.legs, tmp_legs, u_rank, legs, u_legs, new_u_legs, new_v_legs);
      transpose::plan(plan, tmp_legs, legs);
      auto tensor_res = tensor.svd(plan, u_rank, cut);
      res.S.legs = {new_u_legs};// new_u_legs or new_v_legs
      res.U.tensor = std::move(tensor_res.U);
      res.S.tensor = std::move(tensor_res.S);
      res.V.tensor = std::move(tensor_res.V);
      return std::move(res);
    } // svd
  } // namespace node
} // namespace TAT

#endif // TAT_Svd_HPP_
