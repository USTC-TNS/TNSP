#ifndef META_HPP_
#define META_HPP_

#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include "eigen-git-mirror/Eigen/Dense"
#include "eigen-git-mirror/unsupported/Eigen/CXX11/Tensor"

#define PASS

namespace Node
{
  using Base = double;
  enum class Leg
    {
#define CreateLeg(x) Left##x, Right##x, Up##x, Down##x, Phy##x
     CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4)
#undef CreateLeg
    };

  using Rank  = unsigned int;
  using Size  = std::size_t;
  using Data  = Base*;
  using Dims  = std::vector<Size>;
  using Legs  = std::vector<Leg>;
  using Order = std::vector<Rank>;

  namespace internal::leg
  {
#define IncEnum(p) {Leg::p, #p}
#define IncGroup(x) IncEnum(Left##x), IncEnum(Right##x), IncEnum(Up##x), IncEnum(Down##x), IncEnum(Phy##x)
    static const std::map<Leg, std::string> leg_str = {IncGroup(), IncGroup(1), IncGroup(2), IncGroup(3), IncGroup(4)};
#undef IncGroup
#undef IncEnum
  }

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

  namespace internal::stream
  {
    namespace internal
    {
      template<Device device>
      class stream_aux;
    }

    template<Device device>
    using Stream = typename internal::stream_aux<device>::stream;
    // wait输入的stream handle，并运行自己，返回等待自己的handle
  }

  namespace internal::memory
  {
    template<Device device>
    void* malloc(Size);

    template<Device device>
    void free(void*);

    template<Device device>
    void memCopy(void*, const void*, Size);

    template<Device device>
    void memCopyAsync(void*, const void*, Size, internal::stream::Stream<device>& stream);

    template<Device device>
    void memSend(void*, const void*, Size);

    template<Device device>
    void memSendAsync(void*, const void*, Size, internal::stream::Stream<device>& stream);

    template<Device device>
    void memRecv(void*, const void*, Size);

    template<Device device>
    void memRecvAsync(void*, const void*, Size, internal::stream::Stream<device>& stream);
  }

  namespace internal::shuffle
  {
    inline void make_plan(Order& plan, const Legs& new_legs, const Legs& legs)
    {
      const Size& rank = legs.size();
      for(Size i=0;i<rank;i++)
        {
          for(Size j=0;j<rank;j++)
            {
              if(new_legs[i]==legs[j])
                {
                  plan.push_back(j);
                  break;
                }
            }
        }
    }

    inline void get_dims(Dims& new_dims, const Dims& dims, const Dims& plan)
    {
      const Size& rank = dims.size();
      for(Size i=0;i<rank;i++)
        {
          new_dims.push_back(dims[plan[i]]);
        }
    }

    template<Device device>
    void shuffle(Data                              data_new,
                 Data                              data_old,
                 const Size&                       dims,
                 const Order&                      plan,
                 internal::stream::Stream<device>& stream);
  }

  namespace internal::contract
  {
    void get_shuffle_plan(std::vector<Size>);

    void gemm();

  }

  namespace internal::svd
  {
  }

  namespace internal::qr
  {
  }

  namespace internal::multiple
  {
  }

  namespace internal::norm
  {
  }
}

#endif
