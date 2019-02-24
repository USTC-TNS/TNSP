#ifndef TENSOR_CUDA_HPP_
#define TENSOR_CUDA_HPP_

#include "meta.hpp"

namespace Node
{
  namespace internal
  {
    namespace stream
    {
      static std::vector<std::pair<cudaStream_t*, int>> stream_pool;
    }
  }

  template<>
  class Stream<Device::CUDA>
  {
  public:
    void wait() const {}
    Stream() {}
    ~Stream() {}
    Stream& operator=(Stream<Device::CUDA>& stream) {return *this;}
  };

  namespace internal
  {
    namespace memory
    {
      // CUDA
      template<>
      inline void* malloc<Device::CUDA>(Size size)
      {
        void* res;
        cudaMalloc(&res, size);
        return res;
      }

      template<>
      inline void free<Device::CUDA>(void* ptr)
      {
        cudaFree(ptr);
      }

      template<>
      inline void memCopy<Device::CUDA>(void* dst, const void* src, Size size)
      {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
      }

      template<>
      void memCopyAsync<Device::CUDA>(void* dst, const void* src, Size size, Stream<Device::CUDA>& stream)
      {
        PASS;
      }

      template<>
      void memSend<Device::CUDA>(void*dst, const void* src, Size size)
      {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
      }

      template<>
      void memSendAsync<Device::CUDA>(void* dst, const void* src, Size size, Stream<Device::CUDA>& stream)
      {
        PASS;
      }

      template<>
      void memRecv<Device::CUDA>(void* dst, const void* src, Size size)
      {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
      }

      template<>
      void memRecvAsync<Device::CUDA>(void* dst, const void* src, Size size, Stream<Device::CUDA>& stream)
      {
        PASS;
      }
    }

    namespace shuffle
    {
      template<>
        void shuffle<Device::CUDA>(Data                                   data_new,
                                   Data                                   data_old,
                                   const Dims&                            dims_new,
                                   const Dims&                            dims_old,
                                   const Order&                           plan,
                                   Stream<Device::CUDA>& stream)
      {
        PASS;
      }
    }

    namespace contract
    {
      template<>
      void gemm<Device::CUDA, double>(double*                                data,
                                      double*                                data1,
                                      double*                                data2,
                                      Size                                   a,
                                      Size                                   b,
                                      Size                                   c,
                                      Stream<Device::CUDA>& stream)
      {
        PASS;
      }
    }
  }
}

#endif
