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

    class TensorData
    {
    public:
      Tensor* tensor;
      TensorData(Tensor* _tensor) : tensor(_tensor) {}

      friend std::ostream& operator<<(std::ostream& out, const TensorData& value)
      {
        Tensor& tensor = *value.tensor;
        Size i;
        auto data = tensor.get();
        for(i=0;i<tensor.size-1;i++)
          {
            out << data[i] << ", ";
          }
        out << data[i];
        return out;
      }
    };

  public:
    TensorData content = this;

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
        (std::launch::async,
         [src(tensor.data), old_dims(tensor.dims), plan(plan), dims(dims), size(res.size)]{
           DeviceData data = internal::memory::newer(size);
           internal::shuffle::shuffle(data.get(), src.get().get(), dims, old_dims, plan);
           return data;
         });
      return res;
    }

    static Tensor contract(const Tensor& tensor1,
                           const Tensor& tensor2,
                           const Legs& leg1,
                           const Legs& leg2,
                           const std::map<Leg, Leg>& map1 = {},
                           const std::map<Leg, Leg>& map2 = {})
    {
      Size a, b, c; // a*b , b*c -> a*c
      Legs tmp_leg1, tmp_leg2;
      Rank rank;
      Dims dims;
      Legs legs;
      Size size;
      internal::contract::set_dim_and_leg(rank, dims, legs, size, tmp_leg1, tmp_leg2, a, b, c,
                                          tensor1.rank, tensor1.dims, tensor1.legs, leg1, map1,
                                          tensor2.rank, tensor2.dims, tensor2.legs, leg2, map2);

      Tensor res = Tensor(rank, dims, legs);
      res.data = std::async
        (std::launch::async,
         [=, size(res.size)]{
           DeviceData data = internal::memory::newer(size);
           Tensor tmp_tensor1 = shuffle(tensor1, tmp_leg1);
           Tensor tmp_tensor2 = shuffle(tensor2, tmp_leg2);
           internal::contract::gemm<Base>(data.get(), tmp_tensor1.data.get().get(), tmp_tensor2.data.get().get(), a, b, c);
           return data;
         });
      return res;
    }

    void svd_to(Tensor<device>& U,
                Tensor<device>& S,
                Tensor<device>& V,
                const Legs&     leg,
                Rank            cut=0)
    {
      PASS;

    }

    void qr_to()
    {
      PASS;

    }

    void multiple_from()
    {
      PASS;

    }

    void norm()
    {
      PASS;

    }

    void max() // abs and max
    {
      PASS;

    }

    // single dimension permute ?
    // scalar ?
  };
}

// std::shared_future<std::unique_ptr<const Tensor>> new_tensor =
// shuffle(tensor, leg)

// a = f(b)
// print await a
// async f(){await b ... return ...}

#endif
