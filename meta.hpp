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
  // 约定, 几乎都用引用, 除了data用指针
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

  namespace internal::stream
  {
    namespace internal
    {
      template<Device device>
      class stream_aux;
    }

    template<Device device>
    using Stream = typename internal::stream_aux<device>::stream;
  }

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

    template<Device device>
    void memcpyAsync(void*, const void*, std::size_t, internal::stream::Stream<device> stream);

    template<Device device>
    void memSend(void*, const void*, std::size_t);

    template<Device device>
    void memSendAsync(void*, const void*, std::size_t, internal::stream::Stream<device> stream);

    template<Device device>
    void memRecv(void*, const void*, std::size_t);

    template<Device device>
    void memRecvAsync(void*, const void*, std::size_t, internal::stream::Stream<device> stream);
  }

  namespace internal::shuffle
  {
    inline void get_plan(Size rank, std::vector<Size>& plan, const Legs& legs_old, const Legs& legs_new)
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
      internal::stream::Stream<device>& stream);
  }

  namespace internal::contract
  {
    void get_plan();

    void dgemm();
  }
}

#endif