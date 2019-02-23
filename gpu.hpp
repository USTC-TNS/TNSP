#ifndef TENSOR_GPU_HPP_
#define TENSOR_GPU_HPP_

#include "meta.hpp"

namespace Node
{
  namespace internal::stream
  {
    template<>
    class Stream<Device::GPU>
    {
    public:
      void wait() const {}
      Stream() {}
      ~Stream() {}
      Stream& operator=(Stream<Device::GPU>& stream) {return *this;}
    };
  }

  namespace internal::memory
  {
    // GPU
    template<>
    inline void* malloc<Device::GPU>(Size size)
    {
      PASS;//return std::malloc(size);
    }

    template<>
    inline void free<Device::GPU>(void* ptr)
    {
      PASS;//std::free(ptr);
    }

    template<>
    inline void memCopy<Device::GPU>(void* dst, const void* src, Size size)
    {
      PASS;//std::memcpy(dst, src, size);
    }

    template<>
    void memCopyAsync<Device::GPU>(void* dst, const void* src, Size size, Stream<Device::CPU>& stream)
    {
      PASS;
    }

    template<>
    void memSend<Device::GPU>(void*dst, const void* src, Size size)
    {
      PASS;
    }

    template<>
    void memSendAsync<Device::GPU>(void* dst, const void* src, Size size, Stream<Device::CPU>& stream)
    {
      PASS;
    }

    template<>
    void memRecv<Device::GPU>(void* dst, const void* src, Size size)
    {
      PASS;
    }

    template<>
    void memRecvAsync<Device::GPU>(void* dst, const void* src, Size size, Stream<Device::CPU>& stream)
    {
      PASS;
    }
  }

  namespace internal::shuffle
  {
    template<>
    void shuffle<Device::GPU>(Data                                   data_new,
                              Data                                   data_old,
                              const Dims&                            dims_new,
                              const Dims&                            dims_old,
                              const Order&                           plan,
                              Stream<Device::GPU>& stream)
    {
      PASS;
    }
  }

  namespace internal::contract
  {
    template<>
    void gemm<Device::GPU, double>(double*                                data,
                                   double*                                data1,
                                   double*                                data2,
                                   Size                                   a,
                                   Size                                   b,
                                   Size                                   c,
                                   Stream<Device::GPU>& stream)
    {
      PASS;
    }
  }
}

#endif
