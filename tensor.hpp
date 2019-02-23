#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#define PASS

#include "meta.hpp"
#include "cpu.hpp"

namespace Node
{
  template<Device _device>
  class Tensor
  {
  public:
    Rank rank;
    Dims dims;
    Legs legs;
    Data data;
    Size size;

    static const Device device = _device;
    using Stream = internal::stream::Stream<device>;

  private:
    inline Data new_data(Size size) const
    {
      return Data(internal::memory::malloc<device>(sizeof(Base)*size));
    }

    inline void delete_data(Data ptr) const
    {
      internal::memory::free<device>(ptr);
    }

    inline void copy_data(Data dst, Data src, Size size) const
    {
      internal::memory::memCopy<device>(dst, src, size);
    }

    inline void copy_data_async(Data dst, Data src, Size size, Stream& stream) const
    {
      internal::memory::memCopyAsync<device>(dst, src, size, stream);
    }

    inline void send_data(Data dst, Data src, Size size) const
    {
      internal::memory::memSend<device>(dst, src, size);
    }

    inline void send_data_async(Data dst, Data src, Size size, Stream& stream) const
    {
      internal::memory::memSendAsync<device>(dst, src, size, stream);
    }

    inline void recv_data(Data dst, Data src, Size size) const
    {
      internal::memory::memRecv<device>(dst, src, size);
    }

    inline void recv_data_async(Data dst, Data src, Size size, Stream& stream) const
    {
      internal::memory::memRecvAsync<device>(dst, src, size, stream);
    }

    inline void free_all() const
    {
      if(data) delete_data(data);
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

    inline void clean()
    {
      free_all();
      init();
    }

  public:
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

    inline Tensor<device>& rename_leg(const std::map<Leg, Leg>& dict)
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

    inline void shuffle_to(Tensor<device>& tensor,
                           const Legs& new_legs,
                           Stream& stream) const
    {
      tensor.clean();
      tensor.rank = rank;
      tensor.size = size;
      tensor.data = new_data(size);

      Order plan;
      internal::shuffle::make_plan(plan, new_legs, legs);
      internal::shuffle::shuffle<device>(tensor.data, data, dims, plan, stream);
      internal::shuffle::get_dims(tensor.dims, dims, plan);
      tensor.legs = new_legs;
    }

    inline void shuffle_from(Tensor<device>& tensor,
                             const Legs& new_legs,
                             Stream& stream)
    {
      tensor.shuffle_to(*this, new_legs, stream);
    }

    void contract_from(const Tensor<device>& tensor1,
                       const Tensor<device>& tensor2,
                       const Legs& leg1,
                       const Legs& leg2,
                       Stream& stream)
    {
      clean();
      Size contractRank = leg1.size();
      Size a, b, c; // a*b , b*c -> a*c
      std::vector<Size> plan1, plan2;
      internal::contract::get_shuffle_plan(plan1, a, b, leg1, tensor1.legs, tensor1.dims);
      //shuffle first !!!
      //internal::contract::contract<device>();
      //internal::contract::new_dim(dims, tensor1.dims, dim1, tensor2.dims, dim2);
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

    void max() // abs and max
    {
      PASS;
    }
  };
}


#endif
