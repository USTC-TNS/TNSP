#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>

namespace Node
{
  using Size = std::size_t;
  using Base = double;
  enum class Leg
  {
    #define CreateLeg(x) Left##x, Right##x, Up##x, Down##x, Phy##x
    CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4)
    #undef CreateLeg
  };

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
    template<Device device>
    void* malloc(std::size_t);

    template<Device device>
    void free(void*);

    template<Device device>
    void memcpy(void*, const void*, std::size_t);

    template<>
    void* malloc<Device::CPU>(std::size_t size)
    {
      return std::malloc(size);
    }

    template<>
    void free<Device::CPU>(void* ptr)
    {
      std::free(ptr);
    }

    template<>
    void memcpy<Device::CPU>(void* dst, const void* src, std::size_t size)
    {
      std::memcpy(dst, src, size);
    }
  }

  namespace internal::stream
  {
    namespace internal
    {
      template<Device device>
      class stream_aux
      {
      };

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
    void get_plan(Size rank, std::vector<Size>& plan, const std::vector<Leg>& legs_old, const std::vector<Leg>& legs_new)
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
      Base* data_new,
      Base* data_old,
      Size rank,
      const std::vector<Size>& dims,
      const std::vector<Size>& plan,
      internal::stream::Stream<device>& stream)
    {

    }
  }

  template<Device _device>
  class Tensor
  {
    public:
      Size              rank;
      std::vector<Size> dims;
      std::vector<Leg>  legs;
      Base*             data;
      Size              size;

      static const Device device = _device;

    private:
      void* malloc(std::size_t size) const
      {
        return internal::memory::malloc<device>(size);
      }

      Base* new_data(std::size_t size) const
      {
        return (Base*)malloc(sizeof(Base)*size);
      }

      void free(void* ptr) const
      {
        internal::memory::free<device>(ptr);
      }

      void memcpy(void* dst, const void* src, std::size_t size) const
      {
        internal::memory::memcpy<device>(dst, src, size);
      }

      void memcpyAsync(void* dst, const void* src, std::size_t size) const;

      void free_all() const
      {
        if(data) free(data);
      }

      void init()
      {
        rank = 0;
        data = nullptr;
        size = 1;
      }

    public:
      void clean()
      {
        free_all();
        init();
      }

      Tensor()
      {
        init();
      }

      Tensor(Size _rank, std::vector<Size> _dims, std::vector<Leg> _legs)
       : rank(_rank), size(1), dims(_dims), legs(_legs)
      {
        for(Size i=0;i<rank;i++)
        {
          size *= dims[i];
        }
        data = new_data(size);
      }

      ~Tensor()
      {
        free_all();
      }

      void shuffle_to(Tensor<device>& tensor, const std::vector<Leg>& new_legs, internal::stream::Stream<device>& stream) const
      {
        tensor.clean();
        tensor.rank = rank;
        tensor.data = new_data(size);
        tensor.size = size;

        std::vector<Size> plan;
        internal::shuffle::get_plan(rank, plan, legs, new_legs);
        internal::shuffle::shuffle<device>(tensor.data, data, rank, dims, plan, stream);

        for(Size i=0;i<rank;i++)
        {
          tensor.dims.push_back(dims[plan[i]]);
        }
        tensor.legs = new_legs;
      }

      void shuffle_from(Tensor<device>& tensor, const std::vector<Leg>& new_legs, internal::stream::Stream<device>& stream)
      {
        tensor.shuffle_to(*this, new_legs, stream);
      }

      const Tensor<device>& rename_leg(std::map<Leg, Leg> dict)
      {
        for(auto& i : legs)
        {
          auto where = dict.find(i);
          if(where!=dict.end())
          {
            i = where->second;
          }
        }
        return *this
      }

      void contract_from(
        const Tensor<device>& tensor1,
        const Tensor<device>& tensor2,
        const std::vector<Leg>& leg1,
        const std::vector<Leg>& leg2)
      {
        clean();
      }

      void svd_to();

      void qr_to();

      void multiple_from();
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
}