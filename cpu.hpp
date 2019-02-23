#ifndef TENSOR_CPU_HPP_
#define TENSOR_CPU_HPP_

#include "meta.hpp"

namespace Node
{
  namespace internal::stream
  {
    namespace internal
    {
      template<>
      class stream_aux<Device::CPU>
      {
      public:
        using stream = int;
      };
    }
  }

  namespace internal::memory
  {
    // CPU
    template<>
    inline void* malloc<Device::CPU>(Size size)
    {
      return std::malloc(size);
    }

    template<>
    inline void free<Device::CPU>(void* ptr)
    {
      std::free(ptr);
    }

    template<>
    inline void memCopy<Device::CPU>(void* dst, const void* src, Size size)
    {
      std::memcpy(dst, src, size);
    }

    template<>
    void memCopyAsync<Device::CPU>(void* dst, const void* src, Size size, internal::stream::Stream<Device::CPU>& stream)
    {
      PASS;
    }

    template<>
    void memSend<Device::CPU>(void*dst, const void* src, Size size)
    {
      PASS;
    }

    template<>
    void memSendAsync<Device::CPU>(void* dst, const void* src, Size size, internal::stream::Stream<Device::CPU>& stream)
    {
      PASS;
    }

    template<>
    void memRecv<Device::CPU>(void* dst, const void* src, Size size)
    {
      PASS;
    }

    template<>
    void memRecvAsync<Device::CPU>(void* dst, const void* src, Size size, internal::stream::Stream<Device::CPU>& stream)
    {
      PASS;
    }
  }

  namespace internal::shuffle
  {
    template<>
    void shuffle<Device::CPU>(Data                                   data_new,
                              Data                                   data_old,
                              const Dims&                            dims,
                              const Order&                           plan,
                              internal::stream::Stream<Device::CPU>& stream)
    {
      PASS;
    }
  }

  namespace internal::contract
  {
    template<>
    void gemm<Device::CPU, double>(double*                                data,
                                   double*                                data1,
                                   double*                                data2,
                                   Size                                   a,
                                   Size                                   b,
                                   Size                                   c,
                                   internal::stream::Stream<Device::CPU>& stream)
    {
      PASS;
    }
  }
}

#endif
