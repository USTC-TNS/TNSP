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
  public:
    Rank rank;
    Dims dims;
    Legs legs;
    Data data;
    Size size;

    const TensorData<device> content = this;
    std::shared_future<void> future = std::async(std::launch::async, []{});

    using T = Tensor<device>;

  public: // future相关
    inline void wait () const
    {
      future.wait();
    }

  private: // device 5个内存的wrapper
    inline static Data __new_data(Size size)
    {
      return Data(internal::memory::malloc<device>(sizeof(Base)*size));
    }

    inline static void __delete_data(Data ptr)
    {
      internal::memory::free<device>(ptr);
    }

    inline static void __copy_data(Data dst, Data src, Size size)
    {
      internal::memory::memCopy<device>(dst, src, size);
    }

    inline static void __send_data(Data dst, Data src, Size size)
    {
      internal::memory::memSend<device>(dst, src, size);
    }

    inline static void __recv_data(Data dst, Data src, Size size)
    {
      internal::memory::memRecv<device>(dst, src, size);
    }

  private: //Data 的6个函数, 其中前4个private
    inline void new_data()
    {
      data = __new_data(sizeof(Base)*size);
    }

    inline void delete_data()
    {
      __delete_data(data);
    }

    inline void copy_data_to(Data dst)
    {
      __copy_data(dst, data, sizeof(Base)*size);
    }

    inline void copy_data_from(Data src)
    {
      __copy_data(data, src, sizeof(Base)*size);
    }

  public:
    inline void send_data(Data dst)
    {
      __send_data(dst, data, sizeof(Base)*size);
    }

    inline void recv_data(Data src)
    {
      __recv_data(data, src, size);
    }

  private: // 用于组成构造函数析构函数的辅助函数
    inline void free_data()
    {
      if(data) delete_data();
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
      new_data();
      future = std::async(std::launch::async, [&]{copy_data_from(tensor.data);});
    }

    inline void move_from(Tensor<device>&& tensor)
    {
      rank = tensor.rank;
      dims = std::move(tensor.dims);
      legs = std::move(tensor.legs);
      size = tensor.size;
      data = tensor.data;
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
      free_data();
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
      new_data();
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
      free_data();
    }

    // 这个层次之下,只有阻塞函数,这个层次之上,都再await输入节点后立即返回
  public: // 测试用的数据设置
    void set_test_data()
    {
      if(device==Device::CPU)
        {
          future = std::async
            (std::launch::async,
             [&]{
               for(Size i=0;i<size;i++)
                 {
                   data[i] = i;
                 }
             });
        }
      else
        {
          future = std::async
            (std::launch::async,
             [&]{
               Base* tmp = new Base[size];
               for(Size i=0;i<size;i++)
                 {
                   tmp[i] = i;
                 }
               wait();
               send_data(tmp);
               delete[] tmp;
             });
        }
    }

    void set_test_zero()
    {
      if(device==Device::CPU)
        {
          future = std::async
            (std::launch::async,
             [&]{
               for(Size i=0;i<size;i++)
                 {
                   data[i] = 0;
                 }
             });
        }
      else
        {
          future = std::async
            (std::launch::async,
             [&]{
               Base* tmp = new Base[size];
               for(Size i=0;i<size;i++)
                 {
                   tmp[i] = 0;
                 }
               wait();
               send_data(tmp);
               delete[] tmp;
             });
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

  public:
    inline void shuffle_to(Tensor<device>& tensor,
                           const Legs& new_legs) const
    {
      tensor.rank = rank;
      tensor.size = size;
      tensor.legs = new_legs;

      Order plan;
      internal::shuffle::make_plan(plan, new_legs, legs);
      internal::shuffle::get_dims(tensor.dims, dims, plan);

      tensor.wait();
      tensor.future = std::async
        (std::launch::async,
        [&, new_legs, plan]{
          wait();
          //tensor.wait(); // 免得写了数据又被别人覆盖
          tensor.free_data();
          tensor.new_data();
          internal::shuffle::shuffle<device>(tensor.data, data, tensor.dims, dims, plan);
        });
    }

    inline void shuffle_from(const Tensor<device>& tensor,
                             const Legs& new_legs)
    {
      tensor.shuffle_to(*this, new_legs);
    }

  public:
    void contract_from(const Tensor<device>& tensor1,
                       const Tensor<device>& tensor2,
                       const Legs& leg1,
                       const Legs& leg2,
                       const std::map<Leg, Leg>& map1 = {},
                       const std::map<Leg, Leg>& map2 = {})
    {
      Size a, b, c; // a*b , b*c -> a*c
      Legs tmp_leg1, tmp_leg2;
      internal::contract::set_dim_and_leg(rank, dims, legs, size, tmp_leg1, tmp_leg2, a, b, c,
                                          tensor1.rank, tensor1.dims, tensor1.legs, leg1, map1,
                                          tensor2.rank, tensor2.dims, tensor2.legs, leg2, map2);

      wait();
      future = std::async
        (std::launch::async,
         [&, leg1, leg2, map1, map2, a, b, c, tmp_leg1, tmp_leg2]{
           free_data();
           new_data();
           Tensor<device> tmp_tensor1, tmp_tensor2;
           tensor1.wait();
           tensor2.wait();
           tmp_tensor1.shuffle_from(tensor1, tmp_leg1);
           tmp_tensor2.shuffle_from(tensor2, tmp_leg2);
           tmp_tensor1.wait();
           tmp_tensor2.wait();
           internal::contract::gemm<device>(data, tmp_tensor1.data, tmp_tensor2.data, a, b, c);
         });
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
    tensor.wait();
    Base* data = new Base[tensor.size];
    tensor.RecvData(data);
    if((tensor.size!=0)&&(tensor.data!=nullptr))
      {
        for(i=0;i<tensor.size-1;i++)
          {
            out << data[i] << ", ";
          }
        out << data[i];
      }
    delete[] data;
    return out;
  }

  template<>
  inline std::ostream& operator<<<Device::CPU>(std::ostream& out, const TensorData<Device::CPU>& value)
  {
    Size i;
    const auto& tensor = *value.tensor;
    tensor.wait();
    if((tensor.size!=0)&&(tensor.data!=nullptr))
      {
        for(i=0;i<tensor.size-1;i++)
          {
            out << tensor.data[i] << ", ";
          }
        out << tensor.data[i];
      }
    return out;
  }
}


#endif
