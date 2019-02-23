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
    void memCopyAsync(void*, const void*, Size, internal::stream::Stream<device>&);

    template<Device device>
    void memSend(void*, const void*, Size);

    template<Device device>
    void memSendAsync(void*, const void*, Size, internal::stream::Stream<device>&);

    template<Device device>
    void memRecv(void*, const void*, Size);

    template<Device device>
    void memRecvAsync(void*, const void*, Size, internal::stream::Stream<device>&);
  }

  namespace internal::shuffle
  {
    inline void make_plan(Order& plan, const Legs& new_legs, const Legs& legs)
    {
      const Rank& rank = legs.size();
      for(Rank i=0;i<rank;i++)
        {
          for(Rank j=0;j<rank;j++)
            {
              if(new_legs[i]==legs[j])
                {
                  plan.push_back(j);
                  break;
                }
            }
        }
    }

    inline void get_dims(Dims& new_dims, const Dims& dims, const Order& plan)
    {
      const Rank& rank = dims.size();
      for(Rank i=0;i<rank;i++)
        {
          new_dims.push_back(dims[plan[i]]);
        }
    }

    template<Device device>
    void shuffle(Data                              data_new,
                 Data                              data_old,
                 const Dims&                       dims,
                 const Order&                      plan,
                 internal::stream::Stream<device>& stream);
  }

  namespace internal::contract
  {
    inline void set_dim_and_leg(Rank&                     rank,
                                Dims&                     dims,
                                Legs&                     legs,
                                Size&                     size,
                                Legs&                     new_legs1,
                                Legs&                     new_legs2,
                                Size&                     a,
                                Size&                     b,
                                Size&                     c,
                                const Rank&               rank1,
                                const Dims&               dims1,
                                const Legs&               legs1,
                                const Legs&               plan1,
                                const std::map<Leg, Leg>& map1,
                                const Rank&               rank2,
                                const Dims&               dims2,
                                const Legs&               legs2,
                                const Legs&               plan2,
                                const std::map<Leg, Leg>& map2)
    {
      const Rank& contractNum = plan1.size();
      rank = rank1 + rank2 - 2*contractNum;
      new_legs1 = Legs {rank1};
      new_legs2 = Legs {rank2};
      a = 1;
      b = 1;
      c = 1;
      Rank j, k;
      j = 0;
      k = rank1 - contractNum;
      for(Rank i=0;i<rank1;i++)
        {
          if(std::find(plan1.begin(), plan1.end(), legs1[i]) == plan1.end())
            {
              a            *= dims1[i];
              size         *= dims1[i];
              new_legs1[j]  = legs1[i];
              dims.push_back(dims1[i]);
              try
                {
                  legs.push_back(map1.at(legs1[i]));
                }
              catch(const std::out_of_range& e)
                {
                  legs.push_back(legs1[i]);
                }
              j++;
            }
          else
            {
              b            *= dims1[i];
              new_legs1[k]  = legs1[i];
              k++;
            }
        }
      j = 0;
      k = rank2 - contractNum;
      for(Rank i=0;i<rank2;i++)
        {
          if(std::find(plan2.begin(), plan2.end(), legs2[i]) == plan2.end())
            {
              c            *= dims2[i];
              size         *= dims2[i];
              new_legs2[j]  = legs2[i];
              dims.push_back(dims2[i]);
              try
                {
                  legs.push_back(map2.at(legs2[i]));
                }
              catch(const std::out_of_range& e)
                {
                  legs.push_back(legs2[i]);
                }
              j++;
            }
          else
            {
              b            *= dims2[i];
              new_legs2[k]  = legs2[i];
              k++;
            }
        }
    }

    template<Device device, class Type=Base>
    void gemm(Type*                             data,
              Type*                             data1,
              Type*                             data2,
              Size                              a,
              Size                              b,
              Size                              c,
              internal::stream::Stream<device>& stream);
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
