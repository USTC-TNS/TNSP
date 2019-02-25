#ifndef META_HPP_
#define META_HPP_

#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <future>
#include <memory>
#include <cstdlib>
#include <cstring>

#ifdef __NVCC__
#define USE_CUDA
#else
#define USE_CPU
#endif

#define PASS std::cerr << "calling a passing function at " << __FILE__ << ":" << __LINE__ << " in " << __PRETTY_FUNCTION__ <<std::endl;

namespace Node
{
#ifndef Type
#define Type double
#endif
  using Base = Type;
#undef Type
  enum class Leg
    {
#define CreateLeg(x) Left##x, Right##x, Up##x, Down##x, Phy##x
     CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4)
#undef CreateLeg
    };

  using Rank  = unsigned int;
  using Size  = std::size_t;
  using Dims  = std::vector<Size>;
  using Legs  = std::vector<Leg>;
  using Order = std::vector<Rank>;
  using PlainData  = Base*;

  namespace internal
  {
    namespace leg
    {
#define IncEnum(p) {Leg::p, #p}
#define IncGroup(x) IncEnum(Left##x), IncEnum(Right##x), IncEnum(Up##x), IncEnum(Down##x), IncEnum(Phy##x)
      static const std::map<Leg, std::string> leg_str = {IncGroup(), IncGroup(1), IncGroup(2), IncGroup(3), IncGroup(4)};
#undef IncGroup
#undef IncEnum
    }
  }

  // 不知道为什么这里用引用会错误, 好像只是gdb的事
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

  namespace internal
  {
    namespace memory
    {
      class deleter
      {
      public:
        inline void operator()(Base*) const;
      };

      std::unique_ptr<Base[], deleter> newer(Size);// size是元素个数,需要乘上sizeof(Base)才是需要malloc的大小

      void memCopy(void*, const void*, Size);

      void memSend(void*, const void*, Size);

      void memRecv(void*, const void*, Size);
    }

    namespace shuffle
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

      void shuffle(PlainData    data_new,
                   PlainData    data_old,
                   const Dims&  dims_new,
                   const Dims&  dims_old,
                   const Order& plan);
    }

    namespace contract
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
            auto pos = std::find(plan1.begin(), plan1.end(), legs1[i]);
            if(pos == plan1.end())
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
                auto index          = std::distance(plan1.begin(), pos);
                new_legs1[index+k]  = legs1[i];
                b                  *= dims1[i];
              }
          }
        j = contractNum;
        for(Rank i=0;i<rank2;i++)
          {
            auto pos = std::find(plan2.begin(), plan2.end(), legs2[i]);
            if(pos == plan2.end())
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
                auto index       = std::distance(plan2.begin(), pos);
                new_legs2[index] = legs2[i];
              }
          }
      }

      template<class Type>
      void gemm(Type* data,
                Type* data1,
                Type* data2,
                Size  a,
                Size  b,
                Size  c);
    }
  }

  class TensorData;

  class Tensor;
}

#define DefineLeg(x) static const Node::Leg x = Node::Leg::x
#define DefineLegs(n) DefineLeg(Left##n); DefineLeg(Right##n); DefineLeg(Up##n); DefineLeg(Down##n); DefineLeg(Phy##n)
DefineLegs(); DefineLegs(1); DefineLegs(2); DefineLegs(3); DefineLegs(4);
#undef DefineLegs
#undef DefineLeg

#endif
