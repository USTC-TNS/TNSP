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
  template<Device device>
  class Tensor
  {
    using Dptr = std::unique_ptr<Base[], internal::memory::deleter<device>>;

    Rank rank;
    Dims dims;
    Legs legs;
    Dptr data;
    Size size;

  public:
    Tensor(Size _rank, Dims _dims, Legs _legs)
      : rank(_rank), dims(_dims), legs(_legs)
    {
      size = 1;
      for(Size i=0;i<rank;i++)
        {
          size *= dims[i];
        }
      update_size();
      data = newer(size);
    }

    std::unique_ptr<Data> get(Data dst) const
    {
      memSend(data.get(), src, size*sizeof(Base));
    }
  };

  template<Device device>
  using NodeSync = std::unique_ptr<const Tensor<device>>;

  template<Device device>
  using Node = std::shared_future<std::unique_ptr<const Tensor<device>>>;

  template<Device device>
  Node<device> shuffle(Node<device> tensor, Legs new_legs)
  {
    Dims dims
    Order plan;
    internal::shuffle::make_plan(plan, new_legs, legs);
    internal::shuffle::get_dims(tensor.dims, dims, plan);

    std::shared_ptr<Tensor<device>> res = new Tensor<device> (tensor.rank, dims, newlegs);

         wait();
         //tensor.wait(); // 免得写了数据又被别人覆盖
         tensor.free_data();
         tensor.new_data();
         internal::shuffle::shuffle<device>(tensor.data, data, tensor.dims, dims, plan);
       });
  }
}

// std::shared_future<std::unique_ptr<const Tensor>> new_tensor =
// shuffle(tensor, leg)


#endif
