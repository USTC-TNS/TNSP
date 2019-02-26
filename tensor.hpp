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
  public:
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
        auto host_data = tensor.get();
        for(i=0;i<tensor.size-1;i++)
          {
            out << host_data[i] << ", ";
          }
        out << host_data[i];
        return out;
      }
    };

    const TensorData content = this;

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

    void update_size()
    {
      size = 1;
      for(Size i=0;i<rank;i++)
        {
          size *= dims[i];
        }
    }

    Tensor(const Size& _rank, const Dims& _dims, const Legs& _legs)
      : rank(_rank), dims(_dims), legs(_legs), data() {update_size();}

    Tensor(const Size& _rank, Dims&& _dims, const Legs& _legs)
      : rank(_rank), dims(std::move(_dims)), legs(_legs), data() {update_size();}

    Tensor(const Size& _rank, const Dims& _dims, Legs&& _legs)
      : rank(_rank), dims(_dims), legs(std::move(_legs)), data() {update_size();}

    Tensor(const Size& _rank, Dims&& _dims, Legs&& _legs)
      : rank(_rank), dims(std::move(_dims)), legs(std::move(_legs)), data() {update_size();}

    void set_test_data()
    {
      data = std::async
        (std::launch::async,
         [size(size)]{
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

    void set_zero_data()
    {
      data = std::async
        (std::launch::async,
         [size(size)]{
           PlainData tmp = new Base[size];
           for(Size i=0;i<size;i++)
             {
               tmp[i] = 0;
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

    // Tensor本身copy起来问题不大, 会顺带copy data futuree, 但是不推荐, 因为有小的数据copy
    // Tensor.data可以随便copy, 反正是shared_future, 也用它来维护存储时间
    // Tensor.data.get() 不能动, 他是unique_ptr
    // Tensor.data.get().get() 作为指针传递给下面

    static Tensor shuffle(const Tensor& tensor, const Legs& new_legs)
    {
      Order plan;
      Dims dims;
      internal::shuffle::make_plan(plan, new_legs, tensor.legs);
      internal::shuffle::get_dims(dims, tensor.dims, plan);
      Tensor res = Tensor(tensor.rank, dims, new_legs);
      res.data = std::async
        (std::launch::async,
         [src(tensor.data),
          dims(std::move(dims)),
          old_dims(tensor.dims),
          plan(std::move(plan)),
          size(res.size)]{
           src.wait();
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
      Order plan1;
      Dims dims1;
      internal::shuffle::make_plan(plan1, tmp_leg1, tensor1.legs);
      internal::shuffle::get_dims(dims1, tensor1.dims, plan1);
      Order plan2;
      Dims dims2;
      internal::shuffle::make_plan(plan2, tmp_leg2, tensor2.legs);
      internal::shuffle::get_dims(dims2, tensor2.dims, plan2);

      Tensor res = Tensor(rank, std::move(dims), std::move(legs));
      res.data = std::async
        (std::launch::async,
         [size(res.size),
          size1(tensor1.size),
          size2(tensor2.size),
          src1(tensor1.data),
          src2(tensor2.data),
          dims1(std::move(dims1)),
          dims2(std::move(dims2)),
          old_dims1(tensor1.dims),
          old_dims2(tensor2.dims),
          plan1(std::move(plan1)),
          plan2(std::move(plan2)),
          a, b, c]{
           DeviceData data1;
           DeviceData data2;
           auto f1 = std::async
           (std::launch::async,
            [&]{
            src1.wait();
            data1 = internal::memory::newer(size1);
            internal::shuffle::shuffle(data1.get(), src1.get().get(), dims1, old_dims1, plan1);
            });
           auto f2 = std::async
               (std::launch::async,
                [&]{
                src2.wait();
                data2 = internal::memory::newer(size2);
                internal::shuffle::shuffle(data2.get(), src2.get().get(), dims2, old_dims2, plan2);
                });
           f1.wait();
           f2.wait();
           DeviceData data = internal::memory::newer(size);
           internal::contract::gemm<Base>(data.get(), data1.get(), data2.get(), a, b, c);
           return data;
         });
      return res;
    }

    static void svd(Tensor& U,
                    Tensor& S,
                    Tensor& V,
                    const Legs&     leg,
                    Rank            cut=0)
    {
      PASS;

    }

    static void qr()
    {
      PASS;

    }

    static void multiple()
    {
      PASS;

    }

    static void norm()
    {
      PASS;

    }

    static void max() // abs and max
    {
      PASS;

    }

    // single dimension permute ?
    // scalar ?
  };
}

#endif
