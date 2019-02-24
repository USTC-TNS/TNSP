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
  class always_valid : public std::future<void>
  {
    void wait() {}
  };

  template<Device device>
  class Tensor
  {
  public:
    Rank rank;
    Dims dims;
    Legs legs;
    Data data;
    Size size;

    const TensorData<device> content = this;
    std::future<void> future;

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

    inline void send_data(Data dst, Data src, Size size) const
    {
      internal::memory::memSend<device>(dst, src, size);
    }

    inline void recv_data(Data dst, Data src, Size size) const
    {
      internal::memory::memRecv<device>(dst, src, size);
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
      copy_data(data, tensor.data, sizeof(Base)*size);
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

    void send_data(Data src)
    {
      send_data(data, src, size*sizeof(Base));
    }

    void recv_data(Data dst) const
    {
      recv_data(dst, data, size*sizeof(Base));
    }

    void set_test_data()
    {
      if(device==Device::CPU)
        {
          for(Size i=0;i<size;i++)
            {
              data[i] = i;
            }
        }
      else
        {
          Base* tmp = new Base[size];
          for(Size i=0;i<size;i++)
            {
              tmp[i] = i;
            }
          send_data(tmp);
          delete[] tmp;
        }
    }

    void set_test_zero()
    {
      if(device==Device::CPU)
        {
          for(Size i=0;i<size;i++)
            {
              data[i] = i;
            }
        }
      else
        {
          Base* tmp = new Base[size];
          for(Size i=0;i<size;i++)
            {
              tmp[i] = i;
            }
          send_data(data, tmp, size*sizeof(Base));
          delete[] tmp;
        }
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

    void shuffle_to(Tensor<device>& tensor,
                    const Legs& new_legs) const
    {
      tensor.clean();
      tensor.rank = rank;
      tensor.size = size;
      tensor.data = new_data(size);

      tensor.legs = new_legs;
      Order plan;
      internal::shuffle::make_plan(plan, new_legs, legs);
      internal::shuffle::get_dims(tensor.dims, dims, plan);
      internal::shuffle::shuffle<device>(tensor.data, data, tensor.dims, dims, plan);
    }

    inline void shuffle_from(const Tensor<device>& tensor,
                             const Legs& new_legs)
    {
      tensor.shuffle_to(*this, new_legs);
    }

    void contract_from(const Tensor<device>& tensor1,
                       const Tensor<device>& tensor2,
                       const Legs& leg1,
                       const Legs& leg2,
                       const std::map<Leg, Leg> map1 = {},
                       const std::map<Leg, Leg> map2 = {})
    {
      clean();
      Size a, b, c; // a*b , b*c -> a*c
      Legs tmp_leg1, tmp_leg2;
      internal::contract::set_dim_and_leg(rank, dims, legs, size, tmp_leg1, tmp_leg2, a, b, c,
                                          tensor1.rank, tensor1.dims, tensor1.legs, leg1, map1,
                                          tensor2.rank, tensor2.dims, tensor2.legs, leg2, map2);
      Tensor<device> tmp_tensor1, tmp_tensor2;
      data = new_data(size);
      tmp_tensor1.shuffle_from(tensor1, tmp_leg1);
      tmp_tensor2.shuffle_from(tensor2, tmp_leg2);
      internal::contract::gemm<device>(data, tmp_tensor1.data, tmp_tensor2.data, a, b, c);
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

  template<Device device>
  inline std::ostream& operator<<(std::ostream& out, const Tensor<device>& value)
  {
    Rank i;
    out << "Tensor_" << value.rank << "[";
    for(i=0;i<value.rank-1;i++)
      {
        out << "(" << value.dims[i] << "|" << value.legs[i] << "), ";
      }
    out << "(" << value.dims[i] << "|" << value.legs[i] << ")]";
    return out;
  }

  template<Device device>
  class TensorData
  {
  public:
    Tensor<device>* tensor;
    TensorData(Tensor<device>* _tensor) : tensor(_tensor) {}
  };

  template<Device device>
  inline std::ostream& operator<<(std::ostream& out, const TensorData<device>& value)
  {
    Size i;
    const auto& tensor = *value.tensor;
    Base* data = new Base[tensor.size];
    tensor.recv_data(data);
    for(i=0;i<tensor.size-1;i++)
      {
        out << data[i] << ", ";
      }
    out << data[i];
    delete[] data;
    return out;
  }

  template<>
  inline std::ostream& operator<<<Device::CPU>(std::ostream& out, const TensorData<Device::CPU>& value)
  {
    Size i;
    const auto& tensor = *value.tensor;
    for(i=0;i<tensor.size-1;i++)
      {
        out << tensor.data[i] << ", ";
      }
    out << tensor.data[i];
    return out;
  }
}


#endif
