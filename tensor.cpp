#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>

#define PASS

namespace Node
{
  // 约定, 几乎都用引用, 除了data
  using Size = std::size_t;
  using Base = double;
  using Data = Base*;
  enum class Leg
  {
    #define CreateLeg(x) Left##x, Right##x, Up##x, Down##x, Phy##x
    CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4)
    #undef CreateLeg
  };
  using Dims = std::vector<Size>;
  using Legs = std::vector<Leg>;

  namespace internal::leg
  {
    #define IncEnum(p) {Leg::p, #p}
    #define IncGroup(x) IncEnum(Left##x), IncEnum(Right##x), IncEnum(Up##x), IncEnum(Down##x), IncEnum(Phy##x)
    static const std::map<Leg, std::string> leg_str = {IncGroup(), IncGroup(1), IncGroup(2), IncGroup(3), IncGroup(4)};
    // 如果const的话会报[]没有mark as const的错误
    #undef IncGroup
    #undef IncEnum
  }

  // ostream重载，使用一个static map来完成
  inline std::ostream& operator<<(std::ostream& out, const Leg& value)
  {
    try
    {
      return out << internal::leg::leg_str.at(value);
    }
    catch(const std::out_of_range& e)
    {
      return out;
    }
  }


  enum class Device
  {
    CUDA, SW, AMD, CPU
  };

  namespace internal::memory
  {
    // run in host, malloc in device
    // 输入一个size, 返回malloc的指针
    template<Device device>
    void* malloc(std::size_t);

    // free device上的东西
    template<Device device>
    void free(void*);

    template<Device device>
    void memcpy(void*, const void*, std::size_t);

    // CPU
    template<>
    inline void* malloc<Device::CPU>(std::size_t size)
    {
      return std::malloc(size);
    }

    template<>
    inline void free<Device::CPU>(void* ptr)
    {
      std::free(ptr);
    }

    template<>
    inline void memcpy<Device::CPU>(void* dst, const void* src, std::size_t size)
    {
      std::memcpy(dst, src, size);
    }
  }

  namespace internal::stream
  {
    namespace internal
    {
      template<Device device>
      class stream_aux;

      template<>
      class stream_aux<Device::SW>
      {
        public:
          using stream = int;
      };

      template<>
      class stream_aux<Device::CPU>
      {
        public:
          using stream = int;
      };
    }

    template<Device device>
    using Stream = typename internal::stream_aux<device>::stream;
  }

  namespace internal::shuffle
  {
    inline void get_plan(Size rank, std::vector<Size>& plan, const std::vector<Leg>& legs_old, const std::vector<Leg>& legs_new)
    {
      for(Size i=0;i<rank;i++)
      {
        for(Size j=0;j<rank;j++)
        {
          if(legs_new[i]==legs_old[j])
          {
            plan.push_back(j);
            break;
          }
        }
      }
    }

    template<Device device>
    void shuffle(
      Data                              data_new,
      Data                              data_old,
      Size                              rank,
      const std::vector<Size>&          dims,
      const std::vector<Size>&          plan,
      internal::stream::Stream<device>& stream)
    {
      PASS;
    }
  }

  namespace internal::contract
  {
    void dgemm()
    {
      PASS;
    }
  }

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
        const std::vector<Leg>& new_legs,
        internal::stream::Stream<device>& stream) const
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
        const std::vector<Leg>& new_legs,
        internal::stream::Stream<device>& stream)
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
        const std::vector<Leg>& leg2)
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


int main()
{
  int stream = 0;
  auto s = {2ul,3ul,4ul,5ul};
  auto l = {Node::Leg::Down, Node::Leg::Up, Node::Leg::Left, Node::Leg::Right};
  auto m = {Node::Leg::Right, Node::Leg::Left, Node::Leg::Down, Node::Leg::Up};
  Node::Tensor<Node::Device::CPU> t(4, s, l), r;
  t.shuffle_to(r, m, stream);
  for(auto i : r.dims)
  {
    std::cout << i << " ";
  }
  std::cout << "\n";
  t.rename_leg({{Node::Leg::Down,Node::Leg::Down1},{Node::Leg::Left,Node::Leg::Right},
    {Node::Leg::Right,Node::Leg::Left},{Node::Leg::Up,Node::Leg::Up1}});
  for(auto i : t.legs)
  {
    std::cout << i << " ";
  }
  std::cout << "\n";
  Node::Tensor<Node::Device::CPU> ok = t;
  ok = r;
}