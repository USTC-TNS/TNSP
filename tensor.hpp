#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "meta.hpp"
#ifdef USE_CPU
#include "cpu.hpp"
#endif
#ifdef USE_CUDA
#include "cuda.hpp"
#endif

namespace Node
{
  class Tensor
  {
    using HostData = std::unique_ptr<Base[]>;
    using DeviceData = std::unique_ptr<Base[], internal::memory::deleter>;
    using Data = std::shared_future<DeviceData>;

    Rank rank;
    Dims dims;
    Legs legs;
    Data data;
    Size size;

  public:
    friend std::ostream& operator<<(std::ostream& out, const Tensor& value)
    {
      Rank i;
      out << "Tensor_" << value.rank << "[";
      if(value.rank!=0)
        {
          for(i=0;i<value.rank-1;i++)
            {
              out << "(" << value.dims[i] << "|" << value.legs[i] << "), ";
            }
          out << "(" << value.dims[i] << "|" << value.legs[i] << ")]";
        }
      else
        {
          out << "]";
        }
      return out;
    }

    Tensor(const Size& _rank, const Dims& _dims, const Legs& _legs)
      : rank(_rank), dims(_dims), legs(_legs), data()
    {
      size = 1;
      for(Size i=0;i<rank;i++)
        {
          size *= dims[i];
        }
    }

    void set_test_data()
    {
      data = std::async
        ([size(size)]{
           PlainData tmp = new Base[size];
           for(Size i=0;i<size;i++)
             {
               tmp[i] = i;
             }
           DeviceData data = internal::memory::newer(size);
           internal::memory::memSend(data.get(), tmp, size*sizeof(Base));
           delete[] tmp;
           return data;
         });
    }

    HostData get() const
    {
      HostData res = HostData(new Base[size]);
      internal::memory::memRecv(res.get(), data.get().get(), size*sizeof(Base));
      return res;
    }

    static Tensor shuffle(const Tensor& tensor, const Legs& new_legs)
    {
      Order plan;
      Dims dims;
      internal::shuffle::make_plan(plan, new_legs, tensor.legs);
      internal::shuffle::get_dims(dims, tensor.dims, plan);
      Tensor res = Tensor(tensor.rank, dims, new_legs);
      res.data = std::async
        ([=, src(tensor.data), old_dims(tensor.dims), plan(plan), dims(dims), size(res.size)]{
           DeviceData data = internal::memory::newer(size);
           internal::shuffle::shuffle(data.get(), src.get().get(), dims, old_dims, plan);
           return data;
         });
      return res;
    }
  };

}

// std::shared_future<std::unique_ptr<const Tensor>> new_tensor =
// shuffle(tensor, leg)

// a = f(b)
// print await a
// async f(){await b ... return ...}

#endif
