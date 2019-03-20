#ifndef TENSOR_CPU_HPP_
#define TENSOR_CPU_HPP_

#include "meta.hpp"

extern "C"
{
#include <cblas.h>
}
#include <hptt.h>

namespace Node
{
  namespace internal
  {
    namespace memory
    {
      void deleter::operator()(Base* ptr) const
      {
        delete[] ptr;
      }

      std::unique_ptr<Base[], deleter> newer(Size size)
      {
        return std::unique_ptr<Base[], deleter>(new Base[size]);
      }

      void memCopy(void* dst, const void* src, Size size)
      {
        std::memcpy(dst, src, size);
      }

      void memSend(void*dst, const void* src, Size size)
      {
        std::memcpy(dst, src, size);
      }

      void memRecv(void* dst, const void* src, Size size)
      {
        std::memcpy(dst, src, size);
      }
    }

    namespace shuffle
    {
      void shuffle(PlainData    data_new,
                   PlainData    data_old,
                   const Dims&  dims_new,
                   const Dims&  dims_old,
                   const Order& plan)
      {
        std::vector<int> int_plan(plan.begin(), plan.end());
        std::vector<int> int_dims(dims_old.begin(), dims_old.end());
        hptt::create_plan(int_plan.data(), int_plan.size(),
                          1, data_old, int_dims.data(), NULL,
                          0, data_new, NULL,
                          hptt::ESTIMATE, 1, NULL, 1)->execute();
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
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    a, c, b,
                    1, data1, b, data2, c,
                    0, data, c);
      }

      template<>
      void gemm<float>(float* data,
                       float* data1,
                       float* data2,
                       Size  a,
                       Size  b,
                       Size  c)
      {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    a, c, b,
                    1, data1, b, data2, c,
                    0, data, c);
      }
    }
  }
}

#endif
