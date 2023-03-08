/**
 * Copyright (C) 2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#ifndef TAT_TETRAUX_SCALAPACK_FUNCTIONS
#define TAT_TETRAUX_SCALAPACK_FUNCTIONS

#include <complex>
#include <type_traits>
#include <vector>

template<typename T>
constexpr bool is_real = std::is_scalar_v<T>;
template<typename T>
struct is_complex_helper : std::bool_constant<false> {};
template<typename T>
struct is_complex_helper<std::complex<T>> : std::bool_constant<true> {};
template<typename T>
constexpr bool is_complex = is_complex_helper<T>::value;
template<typename T>
struct real_scalar_helper : std::conditional<is_real<T>, T, void> {};
template<typename T>
struct real_scalar_helper<std::complex<T>> : std::conditional<is_real<T>, T, void> {};
template<typename T>
using real_scalar = typename real_scalar_helper<T>::type;

extern "C" {
   void blacs_pinfo_(int* mypnum, int* nprocs);
   void blacs_get_(const int* icontxt, const int* what, int* val);
   void blacs_gridinit_(const int* icontxt, const char* layout, const int* nprow, const int* npcol);
   void blacs_gridinfo_(const int* icontxt, const int* nprow, const int* npcol, int* myprow, int* mypcol);
   void blacs_gridexit_(const int* icontxt);

   void pssyevd_(
         const char* jobz,
         const char* uplo,
         const int* n,
         float* a,
         const int* ia,
         const int* ja,
         const void* desca,
         float* w,
         float* z,
         const int* iz,
         const int* jz,
         const void* descz,
         float* work,
         const int* lwork,
         int* iwork,
         const int* liwork,
         int* info);
   void pdsyevd_(
         const char* jobz,
         const char* uplo,
         const int* n,
         double* a,
         const int* ia,
         const int* ja,
         const void* desca,
         double* w,
         double* z,
         const int* iz,
         const int* jz,
         const void* descz,
         double* work,
         const int* lwork,
         int* iwork,
         const int* liwork,
         int* info);
   void pcheevd_(
         const char* jobz,
         const char* uplo,
         const int* n,
         std::complex<float>* a,
         const int* ia,
         const int* ja,
         const void* desca,
         float* w,
         std::complex<float>* z,
         const int* iz,
         const int* jz,
         const void* descz,
         std::complex<float>* work,
         const int* lwork,
         float* rwork,
         const int* lrwork,
         int* iwork,
         const int* liwork,
         int* info);
   void pzheevd_(
         const char* jobz,
         const char* uplo,
         const int* n,
         std::complex<double>* a,
         const int* ia,
         const int* ja,
         const void* desca,
         double* w,
         std::complex<double>* z,
         const int* iz,
         const int* jz,
         const void* descz,
         std::complex<double>* work,
         const int* lwork,
         double* rwork,
         const int* lrwork,
         int* iwork,
         const int* liwork,
         int* info);

   void psgemm_(
         const char* transa,
         const char* transb,
         const int* m,
         const int* n,
         const int* k,
         const float* alpha,
         const float* a,
         const int* ia,
         const int* ja,
         const void* desca,
         const float* b,
         const int* ib,
         const int* jb,
         const void* descb,
         const float* beta,
         float* c,
         const int* ic,
         const int* jc,
         const void* descc);
   void pdgemm_(
         const char* transa,
         const char* transb,
         const int* m,
         const int* n,
         const int* k,
         const double* alpha,
         const double* a,
         const int* ia,
         const int* ja,
         const void* desca,
         const double* b,
         const int* ib,
         const int* jb,
         const void* descb,
         const double* beta,
         double* c,
         const int* ic,
         const int* jc,
         const void* descc);
   void pcgemm_(
         const char* transa,
         const char* transb,
         const int* m,
         const int* n,
         const int* k,
         const std::complex<float>* alpha,
         const std::complex<float>* a,
         const int* ia,
         const int* ja,
         const void* desca,
         const std::complex<float>* b,
         const int* ib,
         const int* jb,
         const void* descb,
         const std::complex<float>* beta,
         std::complex<float>* c,
         const int* ic,
         const int* jc,
         const void* descc);
   void pzgemm_(
         const char* transa,
         const char* transb,
         const int* m,
         const int* n,
         const int* k,
         const std::complex<double>* alpha,
         const std::complex<double>* a,
         const int* ia,
         const int* ja,
         const void* desca,
         const std::complex<double>* b,
         const int* ib,
         const int* jb,
         const void* descb,
         const std::complex<double>* beta,
         std::complex<double>* c,
         const int* ic,
         const int* jc,
         const void* descc);

   void psgemv_(
         const char* trans,
         const int* m,
         const int* n,
         const float* alpha,
         const float* a,
         const int* ia,
         const int* ja,
         const void* desca,
         const float* x,
         const int* ix,
         const int* jx,
         const void* descx,
         const int* incx,
         const float* beta,
         float* y,
         const int* iy,
         const int* jy,
         const void* descy,
         const int* incy);
   void pdgemv_(
         const char* trans,
         const int* m,
         const int* n,
         const double* alpha,
         const double* a,
         const int* ia,
         const int* ja,
         const void* desca,
         const double* x,
         const int* ix,
         const int* jx,
         const void* descx,
         const int* incx,
         const double* beta,
         double* y,
         const int* iy,
         const int* jy,
         const void* descy,
         const int* incy);
   void pcgemv_(
         const char* trans,
         const int* m,
         const int* n,
         const std::complex<float>* alpha,
         const std::complex<float>* a,
         const int* ia,
         const int* ja,
         const void* desca,
         const std::complex<float>* x,
         const int* ix,
         const int* jx,
         const void* descx,
         const int* incx,
         const std::complex<float>* beta,
         std::complex<float>* y,
         const int* iy,
         const int* jy,
         const void* descy,
         const int* incy);
   void pzgemv_(
         const char* trans,
         const int* m,
         const int* n,
         const std::complex<double>* alpha,
         const std::complex<double>* a,
         const int* ia,
         const int* ja,
         const void* desca,
         const std::complex<double>* x,
         const int* ix,
         const int* jx,
         const void* descx,
         const int* incx,
         const std::complex<double>* beta,
         std::complex<double>* y,
         const int* iy,
         const int* jy,
         const void* descy,
         const int* incy);

   void sgemv_(
         const char* trans,
         const int* m,
         const int* n,
         const float* alpha,
         const float* a,
         const int* lda,
         const float* x,
         const int* incx,
         const float* beta,
         float* y,
         const int* incy);
   void dgemv_(
         const char* trans,
         const int* m,
         const int* n,
         const double* alpha,
         const double* a,
         const int* lda,
         const double* x,
         const int* incx,
         const double* beta,
         double* y,
         const int* incy);
   void cgemv_(
         const char* trans,
         const int* m,
         const int* n,
         const std::complex<float>* alpha,
         const std::complex<float>* a,
         const int* lda,
         const std::complex<float>* x,
         const int* incx,
         const std::complex<float>* beta,
         std::complex<float>* y,
         const int* incy);
   void zgemv_(
         const char* trans,
         const int* m,
         const int* n,
         const std::complex<double>* alpha,
         const std::complex<double>* a,
         const int* lda,
         const std::complex<double>* x,
         const int* incx,
         const std::complex<double>* beta,
         std::complex<double>* y,
         const int* incy);

   int isamax_(const int* n, const float* x, const int* incx);
   int idamax_(const int* n, const double* x, const int* incx);
   int icamax_(const int* n, const std::complex<float>* x, const int* incx);
   int izamax_(const int* n, const std::complex<double>* x, const int* incx);

   int numroc_(const int* n, const int* nb, const int* iproc, const int* srcproc, const int* nprocs);

   void psgemr2d_(
         const int* m,
         const int* n,
         const float* a,
         const int* ia,
         const int* ja,
         const void* desca,
         float* b,
         const int* ib,
         const int* jb,
         const void* descb,
         const int* ictxt);
   void pdgemr2d_(
         const int* m,
         const int* n,
         const double* a,
         const int* ia,
         const int* ja,
         const void* desca,
         double* b,
         const int* ib,
         const int* jb,
         const void* descb,
         const int* ictxt);
   void pcgemr2d_(
         const int* m,
         const int* n,
         const std::complex<float>* a,
         const int* ia,
         const int* ja,
         const void* desca,
         std::complex<float>* b,
         const int* ib,
         const int* jb,
         const void* descb,
         const int* ictxt);
   void pzgemr2d_(
         const int* m,
         const int* n,
         const std::complex<double>* a,
         const int* ia,
         const int* ja,
         const void* desca,
         std::complex<double>* b,
         const int* ib,
         const int* jb,
         const void* descb,
         const int* ictxt);

   void sgebs2d_(const int* icontxt, const char* scope, const char* top, const int* m, const int* n, const float* a, const int* lda);
   void dgebs2d_(const int* icontxt, const char* scope, const char* top, const int* m, const int* n, const double* a, const int* lda);
   void cgebs2d_(const int* icontxt, const char* scope, const char* top, const int* m, const int* n, const std::complex<float>* a, const int* lda);
   void zgebs2d_(const int* icontxt, const char* scope, const char* top, const int* m, const int* n, const std::complex<double>* a, const int* lda);

   void sgebr2d_(
         const int* icontxt,
         const char* scope,
         const char* top,
         const int* m,
         const int* n,
         float* a,
         const int* lda,
         const int* rsrc,
         const int* csrc);
   void dgebr2d_(
         const int* icontxt,
         const char* scope,
         const char* top,
         const int* m,
         const int* n,
         double* a,
         const int* lda,
         const int* rsrc,
         const int* csrc);
   void cgebr2d_(
         const int* icontxt,
         const char* scope,
         const char* top,
         const int* m,
         const int* n,
         std::complex<float>* a,
         const int* lda,
         const int* rsrc,
         const int* csrc);
   void zgebr2d_(
         const int* icontxt,
         const char* scope,
         const char* top,
         const int* m,
         const int* n,
         std::complex<double>* a,
         const int* lda,
         const int* rsrc,
         const int* csrc);
}

template<typename ScalarType>
constexpr void (*pgemm)(
      const char* transa,
      const char* transb,
      const int* m,
      const int* n,
      const int* k,
      const ScalarType* alpha,
      const ScalarType* a,
      const int* ia,
      const int* ja,
      const void* desca,
      const ScalarType* b,
      const int* ib,
      const int* jb,
      const void* descb,
      const ScalarType* beta,
      ScalarType* c,
      const int* ic,
      const int* jc,
      const void* descc) = nullptr;

template<>
inline auto pgemm<float> = psgemm_;
template<>
inline auto pgemm<double> = pdgemm_;
template<>
inline auto pgemm<std::complex<float>> = pcgemm_;
template<>
inline auto pgemm<std::complex<double>> = pzgemm_;

template<typename ScalarType>
constexpr void (*pgemv)(
      const char* trans,
      const int* m,
      const int* n,
      const ScalarType* alpha,
      const ScalarType* a,
      const int* ia,
      const int* ja,
      const void* desca,
      const ScalarType* x,
      const int* ix,
      const int* jx,
      const void* descx,
      const int* incx,
      const ScalarType* beta,
      ScalarType* y,
      const int* iy,
      const int* jy,
      const void* descy,
      const int* incy);

template<>
inline auto pgemv<float> = psgemv_;
template<>
inline auto pgemv<double> = pdgemv_;
template<>
inline auto pgemv<std::complex<float>> = pcgemv_;
template<>
inline auto pgemv<std::complex<double>> = pzgemv_;

template<typename ScalarType>
constexpr void (*gemv)(
      const char* trans,
      const int* m,
      const int* n,
      const ScalarType* alpha,
      const ScalarType* a,
      const int* lda,
      const ScalarType* x,
      const int* incx,
      const ScalarType* beta,
      ScalarType* y,
      const int* incy);

template<>
inline auto gemv<float> = sgemv_;
template<>
inline auto gemv<double> = dgemv_;
template<>
inline auto gemv<std::complex<float>> = cgemv_;
template<>
inline auto gemv<std::complex<double>> = zgemv_;

template<typename ScalarType>
constexpr int (*iamax)(const int* n, const ScalarType* x, const int* incx);

template<>
inline auto iamax<float> = isamax_;
template<>
inline auto iamax<double> = idamax_;
template<>
inline auto iamax<std::complex<float>> = icamax_;
template<>
inline auto iamax<std::complex<double>> = izamax_;

template<typename ScalarType>
constexpr void (*pgemr2d)(
      const int* m,
      const int* n,
      const ScalarType* a,
      const int* ia,
      const int* ja,
      const void* desca,
      ScalarType* b,
      const int* ib,
      const int* jb,
      const void* descb,
      const int* ictxt);

template<>
inline auto pgemr2d<float> = psgemr2d_;
template<>
inline auto pgemr2d<double> = pdgemr2d_;
template<>
inline auto pgemr2d<std::complex<float>> = pcgemr2d_;
template<>
inline auto pgemr2d<std::complex<double>> = pzgemr2d_;

template<typename ScalarType>
constexpr void (*gebs2d)(const int* icontxt, const char* scope, const char* top, const int* m, const int* n, const ScalarType* a, const int* lda);

template<>
inline auto gebs2d<float> = sgebs2d_;
template<>
inline auto gebs2d<double> = dgebs2d_;
template<>
inline auto gebs2d<std::complex<float>> = cgebs2d_;
template<>
inline auto gebs2d<std::complex<double>> = zgebs2d_;

template<typename ScalarType>
constexpr void (*gebr2d)(
      const int* icontxt,
      const char* scope,
      const char* top,
      const int* m,
      const int* n,
      ScalarType* a,
      const int* lda,
      const int* rsrc,
      const int* csrc);

template<>
inline auto gebr2d<float> = sgebr2d_;
template<>
inline auto gebr2d<double> = dgebr2d_;
template<>
inline auto gebr2d<std::complex<float>> = cgebr2d_;
template<>
inline auto gebr2d<std::complex<double>> = zgebr2d_;

template<typename ScalarType>
int psyevd(
      const char* jobz,
      const char* uplo,
      const int* n,
      ScalarType* a,
      const int* ia,
      const int* ja,
      const void* desca,
      real_scalar<ScalarType>* w,
      ScalarType* z,
      const int* iz,
      const int* jz,
      const void* descz) {
   const int neg_one = -1;
   int info;

   ScalarType f_lwork;
   int lwork;
   real_scalar<ScalarType> f_lrwork;
   int lrwork;
   int liwork;

   if constexpr (std::is_same_v<ScalarType, float>) {
      pssyevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, &f_lwork, &neg_one, &liwork, &neg_one, &info);
      if (info != 0) {
         return info;
      }
      lwork = f_lwork;
      auto work = std::vector<ScalarType>(lwork);
      auto iwork = std::vector<int>(liwork);
      pssyevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work.data(), &lwork, iwork.data(), &liwork, &info);
      if (info != 0) {
         return info;
      }
   } else if constexpr (std::is_same_v<ScalarType, double>) {
      pdsyevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, &f_lwork, &neg_one, &liwork, &neg_one, &info);
      if (info != 0) {
         return info;
      }
      lwork = f_lwork;
      auto work = std::vector<ScalarType>(lwork);
      auto iwork = std::vector<int>(liwork);
      pdsyevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work.data(), &lwork, iwork.data(), &liwork, &info);
      if (info != 0) {
         return info;
      }
   } else if constexpr (std::is_same_v<ScalarType, std::complex<float>>) {
      pcheevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, &f_lwork, &neg_one, &f_lrwork, &neg_one, &liwork, &neg_one, &info);
      if (info != 0) {
         return info;
      }
      lwork = f_lwork.real();
      lrwork = f_lrwork;
      auto work = std::vector<ScalarType>(lwork);
      auto rwork = std::vector<real_scalar<ScalarType>>(lrwork);
      auto iwork = std::vector<int>(liwork);
      pcheevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
      if (info != 0) {
         return info;
      }
   } else if constexpr (std::is_same_v<ScalarType, std::complex<double>>) {
      pzheevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, &f_lwork, &neg_one, &f_lrwork, &neg_one, &liwork, &neg_one, &info);
      if (info != 0) {
         return info;
      }
      lwork = f_lwork.real();
      lrwork = f_lrwork;
      auto work = std::vector<ScalarType>(lwork);
      auto rwork = std::vector<real_scalar<ScalarType>>(lrwork);
      auto iwork = std::vector<int>(liwork);
      pzheevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
      if (info != 0) {
         return info;
      }
   } else {
      return -1;
   }
   return 0;
}
#endif
