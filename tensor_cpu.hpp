#ifndef TENSOR_CPU_HPP_
#define TENSOR_CPU_HPP_

#include "meta.hpp"

namespace Node
{

  namespace internal::memory
  {
    // CPU
    template<>
    inline void* malloc<Device::CPU>(std::size_t size)
    {
      return std::malloc(size);
    }

    template<>
    inline void free<Device::CPU>(void* ptr)
    {
      std::free(ptr);
    }

    template<>
    inline void memcpy<Device::CPU>(void* dst, const void* src, std::size_t size)
    {
      std::memcpy(dst, src, size);
    }
  }

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

  namespace internal::shuffle
  {
    template<>
    void shuffle<Device::CPU>(
      Data                                   data_new,
      Data                                   data_old,
      Size                                   rank,
      const std::vector<Size>&               dims,
      const std::vector<Size>&               plan,
      internal::stream::Stream<Device::CPU>& stream)
    {
      PASS;
    }
  }

  namespace internal::contract
  {
    void dgemm()
    {
      PASS;
    }
  }
}

#endif