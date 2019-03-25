/* TAT
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

/* TODO LIST
 * - transpose may be optimized
 * - Truncated SVD
 * - dgegqr magma
 * - cuda，sw
 * - use it, peps, kitaev, hubbard
 */

/* back for old cuda code
namespace shuffle
{
  void shuffle(PlainData    data_new,
               PlainData    data_old,
               const Dims&  dims_new,
               const Dims&  dims_old,
               const Order& plan)
  {
    //Stream !!!
    const Rank& size = plan.size();
    std::vector<int> int_plan(size, 0);//(plan.begin(), plan.end());
    std::vector<int> int_dims(size, 0);//(dims_old.begin(), dims_old.end());
    for(Rank i=0;i<size;i++)
      {
        int_plan[i] = size - plan[size-i-1] -1;
        int_dims[i] = dims_old[size-i-1];
        //std::cout << plan[i] << "\t" << int_plan[i] << "\t" << dims_old[i] << "\t" << int_dims[i] << "\n";
      }
    //std::cout << "\n\n\n";
    cuttHandle handle;
    internal::cuda::Stream* stream = internal::cuda::get_stream();
    cuttPlan(&handle, size, int_dims.data(), int_plan.data(), sizeof(Base), stream->stream);
    cuttExecute(handle, data_old, data_new);
    cudaStreamSynchronize(stream->stream);
    internal::cuda::delete_stream(stream);
  }
}

namespace contract
{
  template<>
  void gemm<double>(double* data,
                    double* data1,
                    double* data2,
                    Size    a,
                    Size    b,
                    Size    c)
  {
    double alpha = 1;
    double beta  = 0;
    internal::cuda::Stream* stream = internal::cuda::get_stream();
    cublasDgemm(stream->handle, CUBLAS_OP_N, CUBLAS_OP_N, c, a, b, &alpha, data2, c, data1, b, &beta, data, c);
    cudaStreamSynchronize(stream->stream);
    internal::cuda::delete_stream(stream);
  }
}
 */

#ifndef TAT_HPP_

#define TAT_VERSION "0.0.1"

#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>

#define PASS std::cerr << "calling a passing function at " << __FILE__ << ":" << __LINE__ << " in " << __PRETTY_FUNCTION__ << std::endl, exit(233)
#define ENABLE_IF(...) class = typename std::enable_if<__VA_ARGS__::value>::type

#ifdef TAT_USE_CPU
extern "C"
{
#include <mkl.h>
} // extern "C"
#include <hptt.h>

// SVD
#if (defined TAT_USE_GESDD && defined TAT_USE_GESVD) || (defined TAT_USE_GESVD && defined TAT_USE_GESVDX) || (defined TAT_USE_GESVDX && defined TAT_USE_GESDD)
#error only one of GESDD, GESVD and GESVDX could be in use
#endif
#if (!defined TAT_USE_GESDD && !defined TAT_USE_GESVD && !defined TAT_USE_GESVDX)
#warning must use one of GESDD, GESVD and GESVDX, default use GESVD now
#define TAT_USE_GESVD
#endif

// QR
#if (defined TAT_USE_GEQRF && defined TAT_USE_GEQP3)
#error only one of GEQRF and GEQP3 could be in use
#endif
#if (!defined TAT_USE_GEQRF && !defined TAT_USE_GEQP3)
#warning must use one of GEQRF and GEQP3, default use GEQRF now
#define TAT_USE_GEQRF
#endif

#ifdef TAT_USE_GEQP3
#warning GEQP3 is current unusable
#endif

#endif // TAT_USE_CPU

namespace TAT {

  enum class Device : unsigned char {CPU, CUDA, DCU, SW};

  namespace legs {
    enum class Legs : unsigned char {
#define CreateLeg(x) Left##x, Right##x, Up##x, Down##x, Phy##x
      CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4),
      CreateLeg(5), CreateLeg(6), CreateLeg(7), CreateLeg(8), CreateLeg(9)
#undef CreateLeg
    }; // enum class Legs

    inline namespace io {}
    namespace io {
#define IncEnum(p) {Legs::p, #p}
#define IncGroup(x) IncEnum(Left##x), IncEnum(Right##x), IncEnum(Up##x), IncEnum(Down##x), IncEnum(Phy##x)
      static const std::map<Legs, std::string> legs_str = {
        IncGroup(), IncGroup(1), IncGroup(2), IncGroup(3), IncGroup(4),
        IncGroup(5), IncGroup(6), IncGroup(7), IncGroup(8), IncGroup(9)
      };
#undef IncGroup
#undef IncEnum

      std::ostream& operator<<(std::ostream& out, const Legs& value) {
        return out << legs_str.at(value);
      } // operator<<
    } // namespace io
  } // namespace legs
  using legs::Legs;

  inline namespace legs_name {}
  namespace legs_name {
#define TAT_DefineLeg(x) static const TAT::Legs x = TAT::Legs::x
#define TAT_DefineLegs(n) TAT_DefineLeg(Left##n); TAT_DefineLeg(Right##n); TAT_DefineLeg(Up##n); TAT_DefineLeg(Down##n); TAT_DefineLeg(Phy##n)
#define TAT_Legs \
  TAT_DefineLegs(); TAT_DefineLegs(1); TAT_DefineLegs(2); TAT_DefineLegs(3); TAT_DefineLegs(4); \
  TAT_DefineLegs(5); TAT_DefineLegs(6); TAT_DefineLegs(7); TAT_DefineLegs(8); TAT_DefineLegs(9)

    TAT_Legs;
#undef TAT_Legs
#undef TAT_DefineLegs
#undef TAT_DefineLeg
  } // namespace legs_name

  using Size = std::size_t;
  using Rank = unsigned int;

  namespace data {
    template<Device device, class Base, ENABLE_IF(std::is_scalar<Base>)>
    class Data;
  } // namespace data
  using data::Data;

  namespace node {
    template<Device device, class Base>
    class Node;
  } // namespace node
  using node::Node;

  namespace tensor {
    template<Device device=Device::CPU, class Base=double>
    class Tensor;
  } // namespace tensor
  using tensor::Tensor;

  namespace data {
#ifdef TAT_USE_CPU
    static const Device device = Device::CPU;

    namespace transpose {}

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
                    1, data1, k, data2, n,
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
    } // namespace data::contract

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
    } // namespace data::multiple

    namespace svd {
#if (defined TAT_USE_GESVD) || (defined TAT_USE_GESDD)
      template<class Base>
      void run(const Size& m, const Size& n, const Size& min, Base* a, Base* u, Base* s, Base* vt);

      template<>
      void run<float>(const Size& m, const Size& n, const Size& min, float* a, float* u, float* s, float* vt) {
#ifdef TAT_USE_GESDD
        auto res = LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S', m, n, a, n, s, u, min, vt, n);
#endif // TAT_USE_GESDD
#ifdef TAT_USE_GESVD
        auto superb = new float[min-1];
        auto res = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, a, n, s, u, min, vt, n, superb);
        delete[] superb;
#endif // TAT_USE_GESVD
        assert(res==0);
      } // run<float>

      template<>
      void run<double>(const Size& m, const Size& n, const Size& min, double* a, double* u, double* s, double* vt) {
#ifdef TAT_USE_GESDD
        auto res = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', m, n, a, n, s, u, min, vt, n);
#endif // TAT_USE_GESDD
#ifdef TAT_USE_GESVD
        auto superb = new double[min-1];
        auto res = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, a, n, s, u, min, vt, n, superb);
        delete[] superb;
#endif // TAT_USE_GESVD
        assert(res==0);
      } // run<double>

      template<class Base>
      Data<device, Base> cut(const Data<device, Base>& other,
                             const Size& m1,
                             const Size& n1,
                             const Size& m2,
                             const Size& n2) {
        Data<device, Base> res(m2*n2);
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
        return res;
      } // cut
#endif // TAT_USE_GESVD TAT_USE_GESDD

#ifdef TAT_USE_GESVDX
      template<class Base>
      void run(const Size& m, const Size& n, const Size& min, const Size& cut, Base* a, Base* u, Base* s, Base* vt);

      template<>
      void run<float>(const Size& m, const Size& n, const Size& min, const Size& cut, float* a, float* u, float* s, float* vt) {
        lapack_int ns;
        auto superb = new lapack_int[12*min];
        auto res = LAPACKE_sgesvdx(LAPACK_ROW_MAJOR, 'V', 'V', 'I', m, n, a, n, 0, 0, 1, cut, &ns, s, u, cut, vt, n, superb);
        assert(res==0);
        assert(ns==lapack_int(cut));
        delete[] superb;
      } // runx<float>

      template<>
      void run<double>(const Size& m, const Size& n, const Size& min, const Size& cut, double* a, double* u, double* s, double* vt) {
        lapack_int ns;
        auto superb = new lapack_int[12*min];
        auto res = LAPACKE_dgesvdx(LAPACK_ROW_MAJOR, 'V', 'V', 'I', m, n, a, n, 0, 0, 1, cut, &ns, s, u, cut, vt, n, superb);
        assert(res==0);
        assert(ns==lapack_int(cut));
        delete[] superb;
      } // runx<double>
#endif // TAT_USE_GESVDX
    } // namespace data::svd

    namespace qr {
      template<class Base>
      void geqrf(Base* A, Base* tau, const Size& m, const Size& n);

      template<>
      void geqrf<float>(float* A, float* tau, const Size& m, const Size& n) {
#ifdef TAT_USE_GEQP3
        auto jpvt = new lapack_int[n];
        auto res = LAPACKE_sgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt, tau);
        delete[] jpvt;
#endif // TAT_USE_GEQP3
#ifdef TAT_USE_GEQRF
        auto res = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
#endif // TAT_USE_GEQRF
        assert(res==0);
      } // geqrf<float>

      template<>
      void geqrf<double>(double* A, double* tau, const Size& m, const Size& n) {
#ifdef TAT_USE_GEQP3
        auto jpvt = new lapack_int[n];
        auto res = LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt, tau);
        delete[] jpvt;
#endif // TAT_USE_GEQP3
#ifdef TAT_USE_GEQRF
        auto res = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
#endif // TAT_USE_GEQRF
        assert(res==0);
      } // geqrf<double>

      template<class Base>
      void orgqr(Base* A, const Base* tau, const Size& m, const Size& min);

      template<>
      void orgqr<float>(float* A, const float* tau, const Size& m, const Size& min) {
        auto res = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, min, min, A, min, tau);
        assert(res==0);
      } // orgqr<float>

      template<>
      void orgqr<double>(double* A, const double* tau, const Size& m, const Size& min) {
        auto res = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, min, min, A, min, tau);
        assert(res==0);
      } // orgqr<double>

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
    } // namespace data::qr

    namespace norm {
      template<class Base>
      void vAbs(const Size& size, const Base* a, Base* y);

      template<>
      void vAbs<float>(const Size& size, const float* a, float* y) {
        vsAbs(size, a, y);
      } // vAbs<float>

      template<>
      void vAbs<double>(const Size& size, const double* a, double* y) {
        vdAbs(size, a, y);
      } // vAbs<double>

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

      template<class Base, int n>
      Base run(const Size& size, const Base* data) {
        if (n==-1) {
          auto i = iamax<Base>(size, data);
          return abs(data[i]);
        }
        auto tmp = new Base[size];
        if (n==2) {
          vSqr<Base>(size, data, tmp);
        } else if (n%2==0) {
          vPowx<Base>(size, data, Base(n), tmp);
        } else {
          vAbs<Base>(size, data, tmp);
          vPowx<Base>(size, tmp, Base(n), tmp);
        }
        auto res = asum<Base>(size, tmp);
        delete[] tmp;
        return res;
      } // run
    } // namespace data::norm

    template<class Base>
    class Data<device, Base> {
     public:
      Data() : size(0), base() {}

      Size size;
      std::unique_ptr<Base[]> base;

      ~Data() = default;
      Data(Data<device, Base>&& other) = default;
      Data<device, Base>& operator=(Data<device, Base>&& other) = default;
      Data(Size _size) : size(_size) {
        base = std::unique_ptr<Base[]>(new Base[size]);
      }
      Data(const Data<device, Base>& other) {
        new (this) Data(other.size);
        std::memcpy(get(), other.get(), size*sizeof(Base));
      }
      Data<device, Base>& operator=(const Data<device, Base>& other) {
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

      void set_test() {
        for (Size i=0; i<size; i++) {
          base[i] = Base(i);
        } // for i
      } // set_test
      void set_zero() {
        for (Size i=0; i<size; i++) {
          base[i] = Base(0);
        } // for i
      } // set_zero
      void set_random(Base(*random)()) {
        for (Size i=0; i<size; i++) {
          base[i] = random();
        } // for i
      } // set_random

      template<int n>
      Data<device, Base> norm() const {
        Data<device, Base> res(Size(1));
        *res.get() = norm::run<Base, n>(size, get());
        return res;
      } // norm

      template<class Base2, ENABLE_IF(std::is_scalar<Base2>)>
      Data<device, Base2> to() const {
        Data<device, Base2> res(size);
        for (Size i=0; i<size; i++) {
          res.base[i] = Base2(base[i]);
        } // for i
        return res;
      } // to

      Data<device, Base> transpose(const std::vector<Size>& dims,
                                   const std::vector<Rank>& plan) const {
        Data<device, Base> res(size);
        assert(dims.size()==plan.size());
        std::vector<int> int_plan(plan.begin(), plan.end());
        std::vector<int> int_dims(dims.begin(), dims.end());
        hptt::create_plan(int_plan.data(), int_plan.size(),
                          1, get(), int_dims.data(), NULL,
                          0, res.get(), NULL,
                          hptt::ESTIMATE, 1, NULL, 1)->execute();
        return res;
      } // transpose

      static Data<device, Base> contract(const Data<device, Base>& data1,
                                         const Data<device, Base>& data2,
                                         const std::vector<Size>& dims1,
                                         const std::vector<Size>& dims2,
                                         const std::vector<Rank>& plan1,
                                         const std::vector<Rank>& plan2,
                                         const Size& m, const Size& k, const Size& n) {
        assert(m*k==data1.size);
        assert(k*n==data2.size);
        Data<device, Base> a = data1.transpose(dims1, plan1);
        Data<device, Base> b = data2.transpose(dims2, plan2);
        // wasted transpose
        Data<device, Base> res(m*n);
        contract::run<Base>(res.get(), a.get(), b.get(), m, n, k);
        return res;
      } // contract

      Data<device, Base> multiple(const Data<device, Base>& other, const Size& a, const Size& b, const Size& c) const {
        Data<device, Base> res(size);
        assert(b==other.size);
        assert(a*b*c==size);
        multiple::run<Base>(res.get(), get(), other.get(), a, b, c);
        return res;
      } // multiple

      friend class svd_res;
      class svd_res {
       public:
        Data<device, Base> U;
        Data<device, Base> S;
        Data<device, Base> V;
      }; // class svd_res

      svd_res svd(const std::vector<Size>& dims,
                  const std::vector<Rank>& plan,
                  const Size& u_size,
                  const Size& cut) const {
        assert(size%u_size==0);
        Size v_size = size/u_size;
        Size min_mn = (u_size<v_size)?u_size:v_size;
        Size cut_dim = (cut<min_mn)?cut:min_mn;
        // -1 > any integer
        Data<device, Base> tmp = transpose(dims, plan);
        // used in svd, dgesvd will destroy it
        svd_res res;
#ifdef TAT_USE_GESVDX
        res.U = Data<device, Base>(u_size*cut_dim);
        res.S = Data<device, Base>(min_mn);
        res.S.size = cut_dim;
        res.V = Data<device, Base>(cut_dim*v_size);
        svd::run(u_size, v_size, min_mn, cut_dim, tmp.get(), res.U.get(), res.S.get(), res.V.get());
#endif // TAT_USE_GESVDX
#if (defined TAT_USE_GESVD) || (defined TAT_USE_GESDD)
        res.U = Data<device, Base>(u_size*min_mn);
        res.S = Data<device, Base>(min_mn);
        res.V = Data<device, Base>(min_mn*v_size);
        svd::run(u_size, v_size, min_mn, tmp.get(), res.U.get(), res.S.get(), res.V.get());
        if (cut_dim!=min_mn) {
          res.U = svd::cut(res.U, u_size, min_mn, u_size, cut_dim);
          res.S = svd::cut(res.S, min_mn, 1, cut_dim, 1);
          res.V = svd::cut(res.V, min_mn, v_size, cut_dim, v_size);
        }
#endif // TAT_USE_GESVD TAT_USE_GESDD
        return res;
      } // svd

      friend class qr_res;
      class qr_res {
       public:
        Data<device, Base> Q;
        Data<device, Base> R;
      }; // class qr_res

      qr_res qr(const std::vector<Size>& dims,
                const std::vector<Rank>& plan,
                const Size& q_size,
                const Size& r_size) const {
        assert(size==q_size*r_size);
        Size min_mn = (q_size<r_size)?q_size:r_size;
        qr_res res;
        res.Q = Data<device, Base>(q_size*min_mn);
        res.R = transpose(dims, plan);
        // R is q_size*r_size, should be min_mn*r_size
        qr::run(res.Q.get(), res.R.get(), q_size, r_size, min_mn);
        res.R.size = min_mn*r_size;
        return res;
      } // qr
    }; // class Data

    inline namespace scalar {}
    namespace scalar {
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

      template<class Base>
      void LinearFrac(const Data<device, Base>& src, Data<device, Base>& dst,
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

      template<class Base>
      void Add(const Data<device, Base>& a, const Data<device, Base>& b, Data<device, Base>& y) {
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

      template<class Base>
      void Sub(const Data<device, Base>& a, const Data<device, Base>& b, Data<device, Base>& y) {
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

      template<class Base>
      void Mul(const Data<device, Base>& a, const Data<device, Base>& b, Data<device, Base>& y) {
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

      template<class Base>
      void Div(const Data<device, Base>& a, const Data<device, Base>& b, Data<device, Base>& y) {
        assert(a.size==b.size);
        vDiv<Base>(a.size, a.get(), b.get(), y.get());
      } // Div

      template<class Base>
      Data<device, Base>& operator*=(Data<device, Base>& a, const Data<device, Base>& b) {
        if (b.size==1) {
          LinearFrac<Base>(a, a, *b.get(), 0, 0, 1);
        } else {
          Mul<Base>(a, b, a);
        } // if
        return a;
      } // operator*=

      template<class Base>
      Data<device, Base> operator*(const Data<device, Base>& a, const Data<device, Base>& b) {
        if (a.size==1) {
          Data<device, Base> res(b.size);
          LinearFrac<Base>(b, res, *a.get(), 0, 0, 1);
          return res;
        } // if
        if (b.size==1) {
          Data<device, Base> res(a.size);
          LinearFrac<Base>(a, res, *b.get(), 0, 0, 1);
          return res;
        } // if
        Data<device, Base> res(a.size);
        Mul<Base>(a, b, res);
        return res;
      } // operator*

      template<class Base>
      Data<device, Base>& operator/=(Data<device, Base>& a, const Data<device, Base>& b) {
        if (b.size==1) {
          LinearFrac<Base>(a, a, 1, 0, 0, *b.get());
        } else {
          Div<Base>(a, b, a);
        } // if
        return a;
      } // operator/=

      template<class Base>
      Data<device, Base> operator/(const Data<device, Base>& a, const Data<device, Base>& b) {
        if (a.size==1) {
          Data<device, Base> res(b.size);
          LinearFrac<Base>(b, res, 0, *a.get(), 1, 0);
          return res;
        } // if
        if (b.size==1) {
          Data<device, Base> res(a.size);
          LinearFrac<Base>(a, res, 1, 0, 0, *b.get());
          return res;
        } // if
        Data<device, Base> res(a.size);
        Div<Base>(a, b, res);
        return res;
      } // operator/

      template<class Base>
      Data<device, Base> operator+(const Data<device, Base>& a) {
        return Data<device, Base>(a);
      } // operator+

      template<class Base>
      Data<device, Base>& operator+=(Data<device, Base>& a, const Data<device, Base>& b) {
        if (b.size==1) {
          LinearFrac<Base>(a, a, 1, *b.get(), 0, 1);
        } else {
          Add<Base>(a, b, a);
        } // if
        return a;
      } // operator+=

      template<class Base>
      Data<device, Base> operator+(const Data<device, Base>& a, const Data<device, Base>& b) {
        if (a.size==1) {
          Data<device, Base> res(b.size);
          LinearFrac<Base>(b, res, 1, *a.get(), 0, 1);
          return res;
        } // if
        if (b.size==1) {
          Data<device, Base> res(a.size);
          LinearFrac<Base>(a, res, 1, *b.get(), 0, 1);
          return res;
        } // if
        Data<device, Base> res(a.size);
        Add<Base>(a, b, res);
        return res;
      } // operator+

      template<class Base>
      Data<device, Base> operator-(const Data<device, Base>& a) {
        Data<device, Base> res(a.size);
        LinearFrac<Base>(a, res, -1, 0, 0, 1);
        return res;
      } // operator-

      template<class Base>
      Data<device, Base>& operator-=(Data<device, Base>& a, const Data<device, Base>& b) {
        if (b.size==1) {
          LinearFrac<Base>(a, a, 1, -*b.get(), 0, 1);
        } else {
          Sub<Base>(a, b, a);
        } // if
        return a;
      } // operator-=

      template<class Base>
      Data<device, Base> operator-(const Data<device, Base>& a, const Data<device, Base>& b) {
        if (a.size==1) {
          Data<device, Base> res(b.size);
          LinearFrac<Base>(b, res, -1, *a.get(), 0, 1);
          return res;
        } // if
        if (b.size==1) {
          Data<device, Base> res(a.size);
          LinearFrac<Base>(a, res, 1, -*b.get(), 0, 1);
          return res;
        } // if
        Data<device, Base> res(a.size);
        Sub<Base>(a, b, res);
        return res;
      } // operator-
    } // namespace data::scalar

    inline namespace io {}
    namespace io {
      template<Device device, class Base>
      std::ostream& operator<<(std::ostream& out, const Data<device, Base>& value) {
        for (Size i=0; i<value.size-1; i++) {
          out << value.base[i] << " ";
        } // for i
        if (value.size!=0) {
          out << value.base[value.size-1];
        } // if
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ofstream& operator<<(std::ofstream& out, const Data<device, Base>& value) {
        out.write((char*)&value.size, sizeof(Size));
        out.write((char*)value.get(), value.size*sizeof(Base));
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ifstream& operator>>(std::ifstream& in, Data<device, Base>& value) {
        in.read((char*)&value.size, sizeof(Size));
        value.base = std::unique_ptr<Base[]>(new Base[value.size]);
        in.read((char*)value.get(), value.size*sizeof(Base));
        return in;
      } // operator<<
    } // namespace data::io
#endif // TAT_USE_CPU
  } // namespace data

  namespace node {
    namespace transpose {
      void plan(std::vector<Size>& new_dims, const std::vector<Size>& dims, const std::vector<Rank>& plan) {
        for (const auto& i : plan) {
          new_dims.push_back(dims[i]);
        } // for i
      } // plan
    } // namespace node::transpose

    namespace contract {
      void plan(std::vector<Size>& dims, Size& m, Size& k, Size& n,
                const::std::vector<Size>& dims1,
                const::std::vector<Size>& dims2,
                const std::vector<Rank>& plan1,
                const std::vector<Rank>& plan2,
                const Rank& contract_num) {
        Rank i, tmp=dims1.size()-contract_num, rank2=dims2.size();
        for (i=0; i<tmp; i++) {
          const Size& t = dims1[plan1[i]];
          m *= t;
          dims.push_back(t);
        } // for i
        for (i=0; i<contract_num; i++) {
          k *= dims1[plan1[i+tmp]];
          assert(dims1[plan1[i+tmp]]==dims2[plan2[i]]);
        } // for i
        for (; i<rank2; i++) {
          const Size& t = dims2[plan2[i]];
          n *= t;
          dims.push_back(t);
        } // for i
      } // plan
    } // namespace node::contract

    namespace multiple {
      void plan(Size& a, Size& b, Size& c, const std::vector<Size>& dims, const Rank& index) {
        Rank i=0, rank=dims.size();
        for (; i<index; i++) {
          a *= dims[i];
        } // for i
        b = dims[i];
        i++;
        for (; i<rank; i++) {
          c *= dims[i];
        } // for
      } // plan
    } // namespace node::multiple

    namespace svd {
      void plan(Size& u_size, const Rank& u_rank, const std::vector<Size>& dims) {
        for (Rank i=0; i<u_rank; i++) {
          u_size *= dims[i];
        } // for i
      } // plan
    } // namespace node::svd

    namespace qr {}

    template<Device device, class Base>
    class Node {
     public:
      Node() : dims({}), data() {}

      std::vector<Size> dims;
      Data<device, Base> data;

      ~Node() = default;
      Node(Node<device, Base>&& other) = default;
      Node(const Node<device, Base>& other) = default;
      Node<device, Base>& operator=(Node<device, Base>&& other) = default;
      Node<device, Base>& operator=(const Node<device, Base>& other) = default;
      static Size get_size(const std::vector<Size>& _dims) {
        Size res = 1;
        for (const auto& i : _dims) {
          res *= i;
        } // for i
        return res;
      } // get_size
      template<class T=std::vector<Size>>
      Node(T&& _dims) : data(get_size(_dims)) {
        dims = std::forward<T>(_dims);
      }
      Node(const Base& num) : dims({}), data(num) {}

      const Size& size() const {
        return data.size;
      } // size
      const Base* get() const {
        return data.get();
      } // get
      Base* get() {
        return data.get();
      } // get

      void set_test() {
        data.set_test();
      } // set_test
      void set_zero() {
        data.set_zero();
      } // set_zero
      void set_random(Base(*random)()) {
        data.set_random(random);
      } // set_random

      template<int n>
      Node<device, Base> norm() const {
        Node<device, Base> res({});
        res.data = data.template norm<n>();
        return res;
      } // norm

      template<class Base2, ENABLE_IF(std::is_scalar<Base2>)>
      Node<device, Base2> to() const {
        Node<device, Base2> res;
        res.dims = dims;
        res.data = data.template to<Base2>();
        return res;
      } // to

      Node<device, Base> transpose(const std::vector<Rank>& plan) const {
        Node<device, Base> res;
        transpose::plan(res.dims, dims, plan);
        assert(plan.size()==dims.size());
        assert(get_size(res.dims)==data.size);
        res.data = data.transpose(dims, plan);
        return res;
      } // transpose

      static Node<device, Base> contract(const Node<device, Base>& node1,
                                         const Node<device, Base>& node2,
                                         const std::vector<Rank>& plan1,
                                         const std::vector<Rank>& plan2,
                                         const Rank& contract_num) {
        Node<device, Base> res;
        Size m=1, k=1, n=1;
        contract::plan(res.dims, m, k, n, node1.dims, node2.dims, plan1, plan2, contract_num);
        res.data = Data<device, Base>::contract(node1.data, node2.data, node1.dims, node2.dims, plan1, plan2, m, k, n);
        return res;
      } // contract

      Node<device, Base> multiple(const Node<device, Base>& other, const Rank& index) const {
        Node<device, Base> res;
        res.dims = dims;
        Size a=1, b=1, c=1;
        multiple::plan(a, b, c, dims, index);
        assert(b==other.dims[0]);
        res.data = data.multiple(other.data, a, b, c);
        return res;
      } // multiple

      friend class svd_res;
      class svd_res {
       public:
        Node<device, Base> U;
        Node<device, Base> S;
        Node<device, Base> V;
      }; // class svd_res

      svd_res svd(const std::vector<Rank>& plan, const Rank& u_rank, const Size& cut) const {
        svd_res res;
        Size u_size=1;
        std::vector<Size> tmp_dims;
        transpose::plan(tmp_dims, dims, plan);
        svd::plan(u_size, u_rank, tmp_dims);
        auto data_res = data.svd(dims, plan, u_size, cut);
        auto mid = tmp_dims.begin()+u_rank;
        res.U.dims.insert(res.U.dims.end(), tmp_dims.begin(), mid);
        res.U.dims.push_back(data_res.S.size);
        res.S.dims.push_back(data_res.S.size);
        res.V.dims.push_back(data_res.S.size);
        res.V.dims.insert(res.V.dims.end(), mid, tmp_dims.end());
        res.U.data = std::move(data_res.U);
        res.S.data = std::move(data_res.S);
        res.V.data = std::move(data_res.V);
        return res;
      } // svd

      friend class qr_res;
      class qr_res {
       public:
        Node<device, Base> Q;
        Node<device, Base> R;
      }; // class qr_res

      qr_res qr(const std::vector<Rank>& plan, const Rank& q_rank) const {
        qr_res res;
        Size q_size=1;
        std::vector<Size> tmp_dims;
        transpose::plan(tmp_dims, dims, plan);
        svd::plan(q_size, q_rank, tmp_dims);
        auto mid = tmp_dims.begin()+q_rank;
        Size r_size=data.size/q_size;
        Size min_size = (q_size<r_size)?q_size:r_size;
        auto data_res = data.qr(dims, plan, q_size, r_size);
        res.Q.dims.insert(res.Q.dims.end(), tmp_dims.begin(), mid);
        res.Q.dims.push_back(min_size);
        res.R.dims.push_back(min_size);
        res.R.dims.insert(res.R.dims.end(), mid, tmp_dims.end());
        res.Q.data = std::move(data_res.Q);
        res.R.data = std::move(data_res.R);
        return res;
      } // qr
    }; // class Node

    inline namespace scalar {}
    namespace scalar {
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

      template<Device device, class Base>
      Node<device, Base>& operator*=(Node<device, Base>& a, const Node<device, Base>& b) {
        if (b.size()!=1) {
          assert(a.dims==b.dims);
        } // if
        a.data *= b.data;
        return a;
      } // operator*=

      template<Device device, class Base>
      Node<device, Base> operator*(const Node<device, Base>& a, const Node<device, Base>& b) {
        Node<device, Base> res;
        if (b.size()==1) {
          res.dims = a.dims;
        } else if (a.size()==1) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data * b.data;
        return res;
      } // operator*

      template<Device device, class Base>
      Node<device, Base>& operator/=(Node<device, Base>& a, const Node<device, Base>& b) {
        if (b.size()!=1) {
          assert(a.dims==b.dims);
        } // if
        a.data /= b.data;
        return a;
      } // operator/=

      template<Device device, class Base>
      Node<device, Base> operator/(const Node<device, Base>& a, const Node<device, Base>& b) {
        Node<device, Base> res;
        if (b.size()==1) {
          res.dims = a.dims;
        } else if (a.size()==1) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data / b.data;
        return res;
      } // operator/

      template<Device device, class Base>
      Node<device, Base> operator+(const Node<device, Base>& a) {
        Node<device, Base> res;
        res.dims = a.dims;
        res.data = + a.data;
        return res;
      } // operator+

      template<Device device, class Base>
      Node<device, Base>& operator+=(Node<device, Base>& a, const Node<device, Base>& b) {
        if (b.size()!=1) {
          assert(a.dims==b.dims);
        } // if
        a.data += b.data;
        return a;
      } // operator+=

      template<Device device, class Base>
      Node<device, Base> operator+(const Node<device, Base>& a, const Node<device, Base>& b) {
        Node<device, Base> res;
        if (b.size()==1) {
          res.dims = a.dims;
        } else if (a.size()==1) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data + b.data;
        return res;
      } // operator+

      template<Device device, class Base>
      Node<device, Base> operator-(const Node<device, Base>& a) {
        Node<device, Base> res;
        res.dims = a.dims;
        res.data = - a.data;
        return res;
      } // operator-

      template<Device device, class Base>
      Node<device, Base>& operator-=(Node<device, Base>& a, const Node<device, Base>& b) {
        if (b.size()!=1) {
          assert(a.dims==b.dims);
        } // if
        a.data -= b.data;
        return a;
      } // operator-=

      template<Device device, class Base>
      Node<device, Base> operator-(const Node<device, Base>& a, const Node<device, Base>& b) {
        Node<device, Base> res;
        if (b.size()==1) {
          res.dims = a.dims;
        } else if (a.size()==1) {
          res.dims = b.dims;
        } else {
          res.dims = a.dims;
          assert(a.dims==b.dims);
        } // if
        res.data = a.data - b.data;
        return res;
      } // operator-
    } // namespace node::scalar

    inline namespace io {}
    namespace io {
      std::ostream& operator<<(std::ostream& out, const std::vector<Size>& value) {
        Rank size=value.size();
        for (Rank i=0; i<size-1; i++) {
          out << value[i] << " ";
        } // for i
        if (size!=0) {
          out << value[size-1];
        } // if
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ostream& operator<<(std::ostream& out, const Node<device, Base>& value) {
        return out << "[dims(" << value.dims << ") data(" << value.data << ")]";
      } // operator<<

      template<Device device, class Base>
      std::ofstream& operator<<(std::ofstream& out, const Node<device, Base>& value) {
        Rank rank = value.dims.size();
        out.write((char*)&rank, sizeof(Rank));
        out.write((char*)value.dims.data(), rank*sizeof(Size));
        out << value.data;
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ifstream& operator>>(std::ifstream& in, Node<device, Base>& value) {
        Rank rank;
        in.read((char*)&rank, sizeof(Rank));
        value.dims.resize(rank);
        in.read((char*)value.dims.data(), rank*sizeof(Size));
        in >> value.data;
        return in;
      } // operator<<
    } // namespace node::io
  } // namespace node

  namespace tensor {
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
      void plan(std::vector<Legs>& legs,
                std::vector<Legs>& new_legs1,
                std::vector<Legs>& new_legs2,
                const std::vector<Legs>& total_legs1,
                const std::vector<Legs>& total_legs2,
                const std::vector<Legs>& legs1,
                const std::vector<Legs>& legs2,
                const std::map<Legs, Legs>& map1,
                const std::map<Legs, Legs>& map2) {
        for (const auto& i : total_legs1) {
          auto pos = std::find(legs1.begin(), legs1.end(), i);
          if (pos == legs1.end()) {
            new_legs1.push_back(i);
            try {
              legs.push_back(map1.at(i));
            } catch (const std::out_of_range& e) {
              legs.push_back(i);
            } // try
          } // if
        } // for
        new_legs1.insert(new_legs1.end(), legs1.begin(), legs1.end());

        new_legs2.insert(new_legs2.end(), legs2.begin(), legs2.end());
        for (const auto& i : total_legs2) {
          auto pos = std::find(legs2.begin(), legs2.end(), i);
          if (pos == legs2.end()) {
            new_legs2.push_back(i);
            try {
              legs.push_back(map2.at(i));
            } catch (const std::out_of_range& e) {
              legs.push_back(i);
            } // try
          } // if
        } // for
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
          auto pos = std::find(u_legs.begin(), u_legs.end(), i);
          if (pos==u_legs.end()) { // to V
            V_legs.push_back(i);
          } else { // to U
            U_legs.push_back(i);
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
      template<class T1=std::vector<Size>, class T2=std::vector<Legs>>
      Tensor(T1&& _dims, T2&& _legs) : legs(std::forward<T2>(_legs)), node(std::forward<T1>(_dims)) {
        assert(legs.size()==node.dims.size());
        assert(std::set<Legs>(legs.begin(), legs.end()).size()==legs.size());
      }
      Tensor(const Base& num) : legs({}), node(num) {}

      const Size& size() const {
        return node.size();
      } // size
      const Base* get() const {
        return node.get();
      } // get
      Base* get() {
        return node.get();
      } // get

      void set_test() {
        node.set_test();
      } // set_test
      void set_zero() {
        node.set_zero();
      } // set_zero
      void set_random(Base(*random)()) {
        node.set_random(random);
      } // set_random

      void legs_rename(const std::map<Legs, Legs>& dict) {
        for (auto& i : legs) {
          auto where = dict.find(i);
          if (where!=dict.end()) {
            i = where->second;
          }
        }
      } // legs_rename

      template<int n>
      Tensor<device, Base> norm() const {
        Tensor<device, Base> res({}, {});
        res.node = node.template norm<n>();
        return res;
      } // norm

      template<class Base2, ENABLE_IF(std::is_scalar<Base2>)>
      Tensor<device, Base2> to() const {
        Tensor<device, Base2> res;
        res.legs = legs;
        res.node = node.template to<Base2>();
        return res;
      } // to

      template<class T=std::vector<Legs>>
      Tensor<device, Base> transpose(T&& new_legs) const {
        Tensor<device, Base> res;
        res.legs = new_legs;
        std::vector<Rank> plan;
        transpose::plan(plan, res.legs, legs);
        assert(new_legs.size()==legs.size());
        assert(plan.size()==legs.size());
        res.node = node.transpose(plan);
        return res;
      } // transpose

      static Tensor<device, Base> contract(const Tensor<device, Base>& tensor1,
                                           const Tensor<device, Base>& tensor2,
                                           const std::vector<Legs> legs1,
                                           const std::vector<Legs> legs2,
                                           const std::map<Legs, Legs>& map1 = {},
                                           const std::map<Legs, Legs>& map2 = {}) {
        Tensor<device, Base> res;
        std::vector<Legs> new_legs1, new_legs2;
        std::vector<Rank> plan1, plan2;
        Rank contract_num = legs1.size();
        assert(legs1.size()==legs2.size());
        contract::plan(res.legs, new_legs1, new_legs2, tensor1.legs, tensor2.legs, legs1, legs2, map1, map2);
        transpose::plan(plan1, new_legs1, tensor1.legs);
        transpose::plan(plan2, new_legs2, tensor2.legs);
        assert(new_legs1.size()==tensor1.legs.size());
        assert(plan1.size()==tensor1.legs.size());
        assert(new_legs2.size()==tensor2.legs.size());
        assert(plan2.size()==tensor2.legs.size());
        res.node = Node<device, Base>::contract(tensor1.node, tensor2.node, plan1, plan2, contract_num);
        return res;
      } // contract

      Tensor<device, Base> multiple(const Tensor<device, Base>& other, const Legs& position) const {
        Tensor<device, Base> res;
        assert(other.legs.size()==1);
        res.legs = legs;
        auto pos = std::find(legs.begin(), legs.end(), position);
        Rank index = std::distance(legs.begin(), pos);
        res.node = node.multiple(other.node, index);
        return res;
      } // multiple

      friend class svd_res;
      class svd_res {
       public:
        Tensor<device, Base> U;
        Tensor<device, Base> S;
        Tensor<device, Base> V;
      }; // class svd_res

      svd_res svd(const std::vector<Legs>& u_legs, const Legs& new_u_legs, const Legs& new_v_legs, const Rank& cut=-1) const {
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
        return res;
      } // svd

      friend class qr_res;
      class qr_res {
       public:
        Tensor<device, Base> Q;
        Tensor<device, Base> R;
      }; // class qr_res

      qr_res qr(const std::vector<Legs>& q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const {
        qr_res res;
        std::vector<Legs> tmp_legs;
        std::vector<Rank> plan;
        Rank q_rank;
        svd::plan(res.Q.legs, res.R.legs, tmp_legs, q_rank, legs, q_legs, new_q_legs, new_r_legs);
        transpose::plan(plan, tmp_legs, legs);
        auto node_res = node.qr(plan, q_rank);
        res.Q.node = std::move(node_res.Q);
        res.R.node = std::move(node_res.R);
        return res;
      } // qr
    }; // class Tensor

    inline namespace scalar {}
    namespace scalar {
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
        if (b.size()!=1) {
          assert(a.legs==b.legs);
        }
        a.node *= b.node;
        return a;
      } // operator*=

      template<Device device, class Base>
      Tensor<device, Base> operator*(const Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        Tensor<device, Base> res;
        if (b.size()==1) {
          res.legs = a.legs;
        } else if (a.size()==1) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.node = a.node * b.node;
        return res;
      } // operator*

      template<Device device, class Base>
      Tensor<device, Base>& operator/=(Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        if (b.size()!=1) {
          assert(a.legs==b.legs);
        } // if
        a.node /= b.node;
        return a;
      } // operator/=

      template<Device device, class Base>
      Tensor<device, Base> operator/(const Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        Tensor<device, Base> res;
        if (b.size()==1) {
          res.legs = a.legs;
        } else if (a.size()==1) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.node = a.node / b.node;
        return res;
      } // operator/

      template<Device device, class Base>
      Tensor<device, Base> operator+(const Tensor<device, Base>& a) {
        Tensor<device, Base> res;
        res.legs = a.legs;
        res.node = + a.node;
        return res;
      } // operator+

      template<Device device, class Base>
      Tensor<device, Base>& operator+=(Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        if (b.size()!=1) {
          assert(a.legs==b.legs);
        } // if
        a.node += b.node;
        return a;
      } // operator+=

      template<Device device, class Base>
      Tensor<device, Base> operator+(const Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        Tensor<device, Base> res;
        if (b.size()==1) {
          res.legs = a.legs;
        } else if (a.size()==1) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.node = a.node + b.node;
        return res;
      } // operator+

      template<Device device, class Base>
      Tensor<device, Base> operator-(const Tensor<device, Base>& a) {
        Tensor<device, Base> res;
        res.legs = a.legs;
        res.node = - a.node;
        return res;
      } // operator-

      template<Device device, class Base>
      Tensor<device, Base>& operator-=(Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        if (b.size()!=1) {
          assert(a.legs==b.legs);
        } // if
        a.node -= b.node;
        return a;
      } // operator-=

      template<Device device, class Base>
      Tensor<device, Base> operator-(const Tensor<device, Base>& a, const Tensor<device, Base>& b) {
        Tensor<device, Base> res;
        if (b.size()==1) {
          res.legs = a.legs;
        } else if (a.size()==1) {
          res.legs = b.legs;
        } else {
          res.legs = a.legs;
          assert(a.legs==b.legs);
        } // if
        res.node = a.node - b.node;
        return res;
      } // operator-

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base>& operator*=(Tensor<device, Base>& a, const B& b) {
        return a*=Tensor<device, Base>(b);
      } // operator*=

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base> operator*(const Tensor<device, Base>& a, const B& b) {
        return a*Tensor<device, Base>(b);
      } // operator*

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base> operator*(const B& b, const Tensor<device, Base>& a) {
        return Tensor<device, Base>(b)*a;
      } // operator*

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base>& operator/=(Tensor<device, Base>& a, const B& b) {
        return a/=Tensor<device, Base>(b);
      } // operator/=

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base> operator/(const Tensor<device, Base>& a, const B& b) {
        return a/Tensor<device, Base>(b);
      } // operator/

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base> operator/(const B& b, const Tensor<device, Base>& a) {
        return Tensor<device, Base>(b)/a;
      } // operator/

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base>& operator+=(Tensor<device, Base>& a, const B& b) {
        return a+=Tensor<device, Base>(b);
      } // operator+

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base> operator+(const Tensor<device, Base>& a, const B& b) {
        return a+Tensor<device, Base>(b);
      } // operator+

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base> operator+(const B& b, const Tensor<device, Base>& a) {
        return Tensor<device, Base>(b)+a;
      } // operator+

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base>& operator-=(Tensor<device, Base>& a, const B& b) {
        return a-=Tensor<device, Base>(b);
      } // operator-=

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base> operator-(const Tensor<device, Base>& a, const B& b) {
        return a-Tensor<device, Base>(b);
      } // operator-

      template<Device device, class Base, class B, ENABLE_IF(std::is_scalar<B>)>
      Tensor<device, Base> operator-(const B& b, const Tensor<device, Base>& a) {
        return Tensor<device, Base>(b)-a;
      } // operator-
    } // namespace tensor::scalar

    inline namespace io {}
    namespace io {
      std::ostream& operator<<(std::ostream& out, const std::vector<Legs>& value) {
        Rank size=value.size();
        for (Rank i=0; i<size-1; i++) {
          out << value[i] << " ";
        } // for i
        if (size!=0) {
          out << value[size-1];
        } // if
        return out;
      } // operator<<

      template<Device device, class Base>
      std::ostream& operator<<(std::ostream& out, const Tensor<device, Base>& value) {
        return out << "[rank(" << value.legs.size() << ") legs(" << value.legs << ") node(" << value.node << ")]";
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

#ifdef TAT_TEST_MAIN
using namespace TAT;
int main() {
  std::ios_base::sync_with_stdio(false);
  std::cout << "scalar\n";
  {
    // scalar
    {
      Tensor<> t1({2, 3}, {Up, Down});
      t1.set_zero();
      std::cout << t1 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      t1.set_test();
      std::cout << t1 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      t1.set_test();
      t1 += 1.2;
      std::cout << t1 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      t1.set_test();
      t1 -= 1.2;
      std::cout << t1 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      t1.set_test();
      t1 *= 1.2;
      std::cout << t1 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      t1.set_test();
      t1 /= 1.2;
      std::cout << t1 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      Tensor<> t2({2, 3}, {Up, Down});
      t1.set_test();
      t2.set_test();
      t1 += t2;
      std::cout << t1*2.3 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      Tensor<> t2({2, 3}, {Up, Down});
      t1.set_zero();
      t2.set_test();
      t1 -= t2;
      std::cout << 1-t1/3.4 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      Tensor<> t2({2, 3}, {Up, Down});
      t1.set_test();
      t2.set_test();
      std::cout << 1+3/(t1+1)+t2 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      Tensor<> t2({2, 3}, {Up, Down});
      t1.set_test();
      t2.set_test();
      std::cout << +(t1-1.2)-t2 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      t1.set_test();
      std::cout << 3+1.2/(t1*1.2) << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      t1.set_test();
      std::cout << -(2.4*(t1/1.2)) << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      Tensor<> t2({2, 3}, {Up, Down});
      t1.set_test();
      t2.set_test();
      std::cout << t1/t2 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      Tensor<> t2({2, 3}, {Up, Down});
      t1.set_test();
      t2.set_test();
      t2+=1;
      t1/=t2;
      std::cout << t1 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      Tensor<> t2({2, 3}, {Up, Down});
      t1.set_test();
      t2.set_test();
      std::cout << t1* t2 << std::endl;
    }
    {
      Tensor<> t1({2, 3}, {Up, Down});
      Tensor<> t2({2, 3}, {Up, Down});
      t1.set_test();
      t2.set_test();
      t1 += 1;
      t1 *= t2;
      std::cout << t1 << std::endl;
    }
    {
      //Tensor<> t1({2},{});
    }
    {
      //Tensor<> t1({2,3},{Down,Down});
    }
  } // scalar
  std::cout << "transpose\n";
  {
    // transpose
    {
      Tensor<> t1({2, 3}, {Left, Right});
      t1.set_test();
      auto t2 = t1.transpose({Right, Left});
      std::cout << t1 << std::endl << t2 << std::endl;
    }
    {
      Tensor<> t1({2, 3, 4, 5}, {Down, Up, Left, Right});
      t1.set_test();
      auto t2 = t1.transpose({Left, Down, Right, Up});
      std::cout << t1 << std::endl << t2 << std::endl;
    }
    {
      //Tensor<> t1({2,3},{Left,Right});
      //auto t2 = t1.transpose({Right,Down});
    }
    {
      //Tensor<> t1({2,3},{Left,Right});
      //auto t2 = t1.transpose({Right,Left,Left});
    }
    {
      //Tensor<> t1({2,3},{Left,Right});
      //auto t2 = t1.transpose({Right,Right});
    }
  } // transpose
  std::cout << "to\n";
  {
    // to
    {
      Tensor<> t1({2, 3}, {Left, Right});
      t1.set_test();
      Tensor<Device::CPU, int> t2 = t1.to<int>();
      std::cout << t1 << std::endl << t2 << std::endl;
    }
  } // to
  std::cout << "contract\n";
  {
    // contract
    {
      Tensor<> t1({2, 3}, {Down, Up});
      Tensor<> t2({2, 3}, {Down, Up});
      t1.set_test();
      t2.set_test();
      std::cout << t1 << std::endl << t2 << std::endl << Tensor<>::contract(t1, t2, {Up}, {Up}, {}, {{Down, Down1}}) << std::endl;
    }
    {
      Tensor<> t1({2, 3, 4, 5, 6}, {Down, Up, Left, Right, Phy});
      Tensor<> t2({5, 3, 7}, {Down, Up, Left});
      t1.set_test();
      t2.set_test();
      std::cout << t1 << std::endl << t2 << std::endl << Tensor<>::contract(t1, t2, {Up, Right}, {Up, Down}, {}, {{Left, Left3}}) << std::endl;
    }
    {
      //Tensor<> t1({2,3}, {Down, Up});
      //Tensor<> t2({2,3}, {Down, Up});
      //Tensor<>::contract(t1, t2, {Up}, {Left}, {}, {{Down, Down1}});
    }
    {
      //Tensor<> t1({2,3}, {Down, Up});
      //Tensor<> t2({2,3}, {Down, Up});
      //Tensor<>::contract(t1, t2, {Up}, {Down}, {}, {{Up, Down1}});
    }
    {
      //Tensor<> t1({2,3}, {Down, Up});
      //Tensor<> t2({2,3}, {Down, Up});
      //Tensor<>::contract(t1, t2, {Up,Down}, {Up, Up}, {}, {{Up, Down1}});
    }
  } // contract
  std::cout << "multiple\n";
  {
    // multiple
    {
      Tensor<> t1({3, 4}, {Down, Up});
      Tensor<> t2({4}, {Down});
      t1.set_test();
      t2.set_test();
      auto t3 = t1.multiple(t2, Up);
      std::cout << t1 << std::endl << t2 << std::endl << t3 << std::endl;
    }
    {
      Tensor<> t1({2, 3, 4}, {Right, Down, Up});
      Tensor<> t2({3}, {Down});
      t1.set_test();
      t2.set_test();
      auto t3 = t1.multiple(t2, Down);
      std::cout << t1 << std::endl << t2 << std::endl << t3 << std::endl;
    }
    {
      //Tensor<> t1({2,3,4}, {Right,Down, Up});
      //Tensor<> t2({3}, {Down});
      //t1.set_test();
      //t2.set_test();
      //auto t3 = t1.multiple(t2, Up);
      //std::cout << t1 << std::endl << t2 << std::endl << t3 << std::endl;
    }
  } // multiple
  std::cout << "svd\n";
  {
    // svd
    {
      Tensor<> t1({4, 6}, {Left, Right});
      t1.set_test();
      auto res = t1.svd({Left}, Right, Down, 4);
      std::cout << res.U << std::endl << res.S << std::endl << res.V << std::endl;
    }
    {
      Tensor<> t1({2, 2, 3, 2}, {Left, Right, Up, Down});
      t1.set_test();
      auto res = t1.svd({Left, Right}, Right1, Down1);
      std::cout << res.U << std::endl << res.S << std::endl << res.V << std::endl;
    }
    {
      Tensor<> t1({2, 2, 3, 2}, {Left, Right, Up, Down});
      t1.set_test();
      auto res = t1.svd({Left, Down}, Right1, Down1, -1);
      std::cout << res.U << std::endl << res.S << std::endl << res.V << std::endl;
    }
    {
      Tensor<> t1({2, 2, 3, 2}, {Left, Right, Up, Down});
      t1.set_test();
      auto res = t1.svd({Left, Down}, Right1, Down1, 3);
      std::cout << res.U << std::endl << res.S << std::endl << res.V << std::endl;
      std::ofstream f2;
      f2.open("test_io2.out");
      f2 << res.V;
      f2.close();
    }
  } // svd
  std::cout << "io\n";
  {
    // io
    {
      Tensor<> t1({2, 2, 3, 2}, {Left, Right, Up, Down});
      t1.set_test();
      std::cout << t1 << std::endl;
      std::ofstream f1;
      f1.open("test_io.out");
      f1 << t1;
      f1.close();
      Tensor<> t2;
      std::ifstream f2;
      f2.open("test_io.out");
      f2 >> t2;
      f2.close();
      std::cout << t2 << std::endl;
    }
  } // io
  std::cout << "qr\n";
  {
    // qr
    {
      Tensor<> t1({4, 6}, {Left, Right});
      t1.set_test();
      auto res = t1.qr({Left}, Right, Down);
      std::cout << res.Q << std::endl << res.R << std::endl;
    }
    {
      Tensor<> t1({4, 6}, {Left, Right});
      t1.set_test();
      auto res = t1.qr({Right}, Up, Down);
      std::cout << res.Q << std::endl << res.R << std::endl;
    }
  } // qr
  return 0;
} // main
#endif // TAT_TEST_MAIN

#endif // TAT_HPP_
