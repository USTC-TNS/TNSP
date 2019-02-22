#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#define PASS

#include "meta.hpp"
#include "tensor_cpu.hpp"

namespace Node
{
  template<Device _device>
  class Tensor
  {
    public:
      Size rank;
      Dims dims;
      Legs legs;
      Data data;
      Size size;

      static const Device device = _device;
      using Stream = internal::stream::Stream<device>;

    private:
      inline void* malloc(std::size_t size) const
      {
        return internal::memory::malloc<device>(size);
      }

      inline Data new_data(std::size_t size) const
      {
        return Data(malloc(sizeof(Base)*size));
      }

      inline void free(void* ptr) const
      {
        internal::memory::free<device>(ptr);
      }

      inline void memcpy(void* dst, const void* src, std::size_t size) const
      {
        internal::memory::memcpy<device>(dst, src, size);
      }

      inline void memcpyAsync(void* dst, const void* src, std::size_t size) const
      {
        PASS;
      }

      inline void free_all() const
      {
        if(data) free(data);
      }

      inline void init()
      {
        rank = 0;
        dims = {};
        legs = {};
        data = nullptr;
        size = 1;
      }

      inline void copy_from(const Tensor<device>& tensor)
      {
        rank = tensor.rank;
        dims = tensor.dims;
        legs = tensor.legs;
        size = tensor.size;
        data = new_data(size);
        memcpy(data, tensor.data, sizeof(Base)*size);
      }

      inline void move_from(Tensor<device>&& tensor)
      {
        rank = tensor.rank;
        dims = std::move(tensor.dims);
        legs = std::move(tensor.legs);
        data = tensor.data;
        size = tensor.size;
        tensor.data = nullptr;
      }

      inline void update_size()
      {
        size = 1;
        for(Size i=0;i<rank;i++)
        {
          size *= dims[i];
        }
      }

    public:
      inline void clean()
      {
        free_all();
        init();
      }

      Tensor()
      {
        init();
      }

      Tensor(Size _rank, Dims _dims, Legs _legs)
       : rank(_rank), dims(_dims), legs(_legs)
      {
        update_size();
        data = new_data(size);
      }

      Tensor(const Tensor<device>& tensor)
      {
        copy_from(tensor);
      }

      Tensor(Tensor<device>&& tensor)
      {
        move_from(tensor);
      }

      Tensor<device>& operator= (const Tensor<device>& tensor)
      {
        copy_from(tensor);
        return *this;
      }

      Tensor<device>& operator= (Tensor<device>&& tensor)
      {
        move_from(tensor);
        return *this;
      }

      ~Tensor()
      {
        free_all();
      }

      inline void shuffle_to(
        Tensor<device>& tensor,
        const Legs& new_legs,
        Stream& stream) const
      {
        tensor.clean();
        tensor.rank = rank;
        tensor.size = size;
        tensor.data = new_data(size);

        std::vector<Size> plan;
        internal::shuffle::get_plan(rank, plan, legs, new_legs);
        internal::shuffle::shuffle<device>(tensor.data, data, rank, dims, plan, stream);

        for(Size i=0;i<rank;i++)
        {
          tensor.dims.push_back(dims[plan[i]]);
        }
        tensor.legs = new_legs;
      }

      inline void shuffle_from(
        Tensor<device>& tensor,
        const Legs& new_legs,
        Stream& stream)
      {
        tensor.shuffle_to(*this, new_legs, stream);
      }

      inline const Tensor<device>& rename_leg(const std::map<Leg, Leg>& dict)
      {
        for(auto& i : legs)
        {
          auto where = dict.find(i);
          if(where!=dict.end())
          {
            i = where->second;
          }
        }
        return *this;
      }

      void contract_from(
        const Tensor<device>& tensor1,
        const Tensor<device>& tensor2,
        const std::vector<Leg>& leg1,
        const std::vector<Leg>& leg2
        )
      {
        clean();
        PASS;
      }

      void svd_to()
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
  };
}


#endif
