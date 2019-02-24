#ifndef TENSOR_CUDA_HPP_
#define TENSOR_CUDA_HPP_

#include "meta.hpp"

namespace Node
{
  namespace internal
  {
    namespace stream
    {
      class stream_pair
      {
      public:
        cudaStream_t stream;
        unsigned int count;
        stream_pair()
        {
          cudaStreamCreate(&stream);
          count = 0;
        }
      };

      using Stream = stream_pair*;

      static std::vector<Stream> stream_pool;

      Stream get_stream()
      {
        for(auto& i : stream_pool)
          {
            if(i->count==0)
              {
                i->count++;
                return i;
              }
          }
        auto ptr = new stream_pair;
        ptr->count++;
        stream_pool.push_back(ptr);
        return ptr;
      }

      void delete_stream(Stream stream)
      {
        stream->count--;
      }
    }
  }

  template<>
  class Stream<Device::CUDA>
  {
  public:
    internal::stream::Stream stream;
    void wait() const {}
    Stream()
    {
      stream = internal::stream::get_stream();
    }
    ~Stream()
    {
      internal::stream::delete_stream(stream);
    }
    Stream& operator=(Stream<Device::CUDA>& other)
    {
      internal::stream::delete_stream(stream);
      stream = other.stream;
      stream->count++;
      return *this;
    }
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
