#ifndef TENSOR_CUDA_HPP_
#define TENSOR_CUDA_HPP_

#include "meta.hpp"
#include <cublas_v2.h>
#include <cutt.h>

namespace Node
{
  namespace internal
  {
    namespace cuda
    {
      class Stream
      {
      public:
        cudaStream_t stream;
        cublasHandle_t handle;
        unsigned int count;
        Stream()
        {
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
          cublasCreate(&handle);
          cublasSetStream(handle, stream);
          count = 0;
        }
        ~Stream()
        {
          cublasDestroy(handle);
          cudaStreamDestroy(stream);
        }
      };

      static std::vector<std:: unique_ptr<Stream>> stream_pool;

      Stream* get_stream()
      {
        for(auto& i : stream_pool)
          {
            if(i->count==0)
              {
                i->count++;
                return i.get();
              }
          }
        auto res = new Stream;
        res->count++;
        stream_pool.push_back(std::unique_ptr<Stream>(res));
        return res;
      }

      void delete_stream(Stream* stream)
      {
        stream->count--;
      }
    }
  }

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

      template<>
      void gemm<float>(float* data,
                       float* data1,
                       float* data2,
                       Size    a,
                       Size    b,
                       Size    c)
      {
        float alpha = 1;
        float beta  = 0;
        internal::cuda::Stream* stream = internal::cuda::get_stream();
        cublasSgemm(stream->handle, CUBLAS_OP_N, CUBLAS_OP_N, c, a, b, &alpha, data2, c, data1, b, &beta, data, c);
        cudaStreamSynchronize(stream->stream);
        internal::cuda::delete_stream(stream);
      }
    }
  }
}

#endif
