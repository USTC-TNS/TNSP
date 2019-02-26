#ifndef TENSOR_CUDA_HPP_
#define TENSOR_CUDA_HPP_

#include "meta.hpp"
#include <cutt.h>

namespace Node
{
  /*
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

  class Stream
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
    Stream& operator=(Stream& other)
    {
      internal::stream::delete_stream(stream);
      stream = other.stream;
      stream->count++;
      return *this;
    }
  };
    */

  namespace internal
  {
    namespace memory
    {
      void deleter::operator()(Base* ptr) const
      {
        cudaFree(ptr);
      }

      std::unique_ptr<Base[], deleter> newer(Size size)
      {
        void* res;
        cudaMalloc(&res, size*sizeof(Base));
        return std::unique_ptr<Base[], deleter>((Base*)res);
      }

      void memCopy(void* dst, const void* src, Size size)
      {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
      }

      void memSend(void*dst, const void* src, Size size)
      {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
      }

      void memRecv(void* dst, const void* src, Size size)
      {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
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
        //Stream !!!
        const Size& size = plan.size();
        std::vector<int> int_plan(size, 0);//(plan.begin(), plan.end());
        std::vector<int> int_dims(size, 0);//(dims_old.begin(), dims_old.end());
        for(Size i=0;i<size;i++)
          {
            int_plan[i] = size - plan[size-i-1] -1;
            int_dims[i] = dims_old[size-i-1];
            //std::cout << plan[i] << "\t" << int_plan[i] << "\t" << dims_old[i] << "\t" << int_dims[i] << "\n";
          }
        //std::cout << "\n\n\n";
        cuttHandle handle;
        cuttPlan(&handle, size, int_dims.data(), int_plan.data(), sizeof(double), 0);
        cuttExecute(handle, data_old, data_new);
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
        PASS;
      }
    }
  }
}

#endif
