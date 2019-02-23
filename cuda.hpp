#ifndef TENSOR_CUDA_HPP_
#define TENSOR_CUDA_HPP_

#include "meta.hpp"

namespace Node
{
  namespace internal::stream
  {
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

  namespace internal::memory
  {
    // CUDA
    template<>
    inline void* malloc<Device::CUDA>(Size size)
    {
      PASS;//return std::malloc(size);
    }

    template<>
    inline void free<Device::CUDA>(void* ptr)
    {
      PASS;//std::free(ptr);
    }

    template<>
    inline void memCopy<Device::CUDA>(void* dst, const void* src, Size size)
    {
      PASS;//std::memcpy(dst, src, size);
    }

    template<>
    void memCopyAsync<Device::CUDA>(void* dst, const void* src, Size size, Stream<Device::CUDA>& stream)
    {
      PASS;
    }

    template<>
    void memSend<Device::CUDA>(void*dst, const void* src, Size size)
    {
      PASS;
    }

    template<>
    void memSendAsync<Device::CUDA>(void* dst, const void* src, Size size, Stream<Device::CUDA>& stream)
    {
      PASS;
    }

    template<>
    void memRecv<Device::CUDA>(void* dst, const void* src, Size size)
    {
      PASS;
    }

    template<>
    void memRecvAsync<Device::CUDA>(void* dst, const void* src, Size size, Stream<Device::CUDA>& stream)
    {
      PASS;
    }
  }

  namespace internal::shuffle
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

  namespace internal::contract
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

#endif
