/* TAT/TAT.hpp
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef TAT_HPP_
#define TAT_HPP_

#ifndef TAT_VERSION
#define TAT_VERSION "unknown"
#endif // TAT_VERSION

#include <cassert>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>
#include <functional>

#define PASS std::cerr << "calling a passing function at " << __FILE__ << ":" << __LINE__ << " in " << __PRETTY_FUNCTION__ << std::endl, exit(233)
#define ENABLE_IF(...) class = typename std::enable_if<__VA_ARGS__::value>::type

#if (!defined TAT_USE_CPU && !defined TAT_USE_CUDA && !defined TAT_USE_DCU && !defined TAT_USE_SW)
#warning use CPU by default
#define TAT_USE_CPU
#endif

#ifdef TAT_USE_CPU
extern "C"
{
#include <mkl.h>
} // extern "C"
#include <hptt.h>
#include <rang.hpp>

// SVD
#if (defined TAT_USE_GESDD && defined TAT_USE_GESVD) || (defined TAT_USE_GESVD && defined TAT_USE_GESVDX) || (defined TAT_USE_GESVDX && defined TAT_USE_GESDD)
#error only one of GESDD, GESVD and GESVDX could be in use
#endif
#if (!defined TAT_USE_GESDD && !defined TAT_USE_GESVD && !defined TAT_USE_GESVDX)
#warning must use one of GESDD, GESVD and GESVDX, default use GESVD now
#define TAT_USE_GESVD
#endif

// QR
#if (defined TAT_USE_GEQRF && defined TAT_USE_GEQP3)
#error only one of GEQRF and GEQP3 could be in use
#endif
#if (!defined TAT_USE_GEQRF && !defined TAT_USE_GEQP3)
#warning must use one of GEQRF and GEQP3, default use GEQRF now
#define TAT_USE_GEQRF
#endif

#ifdef TAT_USE_GEQP3
#error GEQP3 is current unusable
#endif

#endif // TAT_USE_CPU

namespace TAT {
  enum class Device : unsigned char {CPU, CUDA, DCU, SW};

  namespace legs {
    enum class Legs : unsigned char {
#define CreateLeg(x) \
        Left##x, Right##x, Up##x, Down##x, Phy##x, \
        LeftUp##x, LeftDown##x, RightUp##x, RightDown##x
      CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4),
      CreateLeg(5), CreateLeg(6), CreateLeg(7), CreateLeg(8), CreateLeg(9)
#undef CreateLeg
    }; // enum class Legs

    inline namespace io {
#define IncEnum(p) {Legs::p, #p}
#define IncGroup(x) \
        IncEnum(Left##x), IncEnum(Right##x), IncEnum(Up##x), IncEnum(Down##x), IncEnum(Phy##x), \
        IncEnum(LeftUp##x), IncEnum(LeftDown##x), IncEnum(RightUp##x), IncEnum(RightDown##x)
      static const std::map<Legs, std::string> legs_str = {
        IncGroup(), IncGroup(1), IncGroup(2), IncGroup(3), IncGroup(4),
        IncGroup(5), IncGroup(6), IncGroup(7), IncGroup(8), IncGroup(9)
      };
#undef IncGroup
#undef IncEnum

      std::ostream& operator<<(std::ostream& out, const Legs& value) {
        return out << legs_str.at(value);
      } // operator<<
    } // namespace io

    inline namespace scalar {
#define IncEnum(p, q) {Legs::p, Legs::q}
#define IncGroup(x) \
        IncEnum(Left##x, Right##x), IncEnum(Right##x, Left##x), IncEnum(Up##x, Down##x), IncEnum(Down##x, Up##x), IncEnum(Phy##x, Phy##x), \
        IncEnum(LeftUp##x, RightDown##x), IncEnum(LeftDown##x, RightUp##x), IncEnum(RightUp##x, LeftDown##x), IncEnum(RightDown##x, LeftUp##x)
      static const std::map<Legs, Legs> minus_legs = {
        IncGroup(), IncGroup(1), IncGroup(2), IncGroup(3), IncGroup(4),
        IncGroup(5), IncGroup(6), IncGroup(7), IncGroup(8), IncGroup(9)
      };
#undef IncGroup
#undef IncEnum

      Legs operator-(const Legs& value) {
        return minus_legs.at(value);
      } // operator-

#define IncEnum(p, q) {Legs::p, Legs::q}
#define IncGroup(x, y) \
        IncEnum(Left##x, Left##y), IncEnum(Right##x, Right##y), IncEnum(Up##x, Up##y), IncEnum(Down##x, Down##y), IncEnum(Phy##x, Phy##y), \
        IncEnum(LeftUp##x, LeftUp##y), IncEnum(LeftDown##x, LeftDown##y), IncEnum(RightUp##x, RightUp##y), IncEnum(RightDown##x, RightDown##y)
      static const std::map<Legs, Legs> plus_legs = {
        IncGroup(, 1), IncGroup(1, 2), IncGroup(2, 3), IncGroup(3, 4), IncGroup(4, 5),
        IncGroup(5, 6), IncGroup(6, 7), IncGroup(7, 8), IncGroup(8, 9), IncGroup(9,)
      };
#undef IncGroup
#undef IncEnum

      Legs operator+(const Legs& value) {
        return plus_legs.at(value);
      } // operator+
    } // namespace scalar;
  } // namespace legs
  using legs::Legs;

  namespace legs_name {
#define TAT_DefineLeg(x) static const TAT::Legs x = TAT::Legs::x
#define TAT_DefineLegs(n) \
      TAT_DefineLeg(Left##n); TAT_DefineLeg(Right##n); TAT_DefineLeg(Up##n); TAT_DefineLeg(Down##n); TAT_DefineLeg(Phy##n); \
      TAT_DefineLeg(LeftUp##n); TAT_DefineLeg(LeftDown##n); TAT_DefineLeg(RightUp##n); TAT_DefineLeg(RightDown##n)
#define TAT_Legs \
      TAT_DefineLegs(); TAT_DefineLegs(1); TAT_DefineLegs(2); TAT_DefineLegs(3); TAT_DefineLegs(4); \
      TAT_DefineLegs(5); TAT_DefineLegs(6); TAT_DefineLegs(7); TAT_DefineLegs(8); TAT_DefineLegs(9)

    TAT_Legs;
#undef TAT_Legs
#undef TAT_DefineLegs
#undef TAT_DefineLeg
  } // namespace legs_name

  using Size = std::size_t;
  using Rank = unsigned int;

  namespace data {
    template<Device device, class Base>
    class Magic;
#define DefineData(x) \
      namespace x { \
        template<class Base> \
        class Data; \
      } \
      template<class Base> \
      class Magic<Device::x, Base>{ \
       public: \
        using type=x::Data<Base>; \
      }

    DefineData(CPU);
    DefineData(CUDA);
    DefineData(DCU);
    DefineData(SW);
#undef DefineData

    template<Device device, class Base, ENABLE_IF(std::is_scalar<Base>)>
    using Data = typename Magic<device, Base>::type;
  } // namespace data
  using data::Data;

  namespace node {
    template<Device device, class Base>
    class Node;
  } // namespace node
  using node::Node;

  namespace tensor {
    template<Device device=Device::CPU, class Base=double>
    class Tensor;
  } // namespace tensor
  using tensor::Tensor;

  namespace site {
    template<Device device=Device::CPU, class Base=double>
    class Site;
  } // namespace site
  using site::Site;

  namespace lattice {
    template<int dimension>
    class Dimension;

    template<class Tags=Dimension<2>, Device device=Device::CPU, class Base=double>
    class Lattice;
  } // namespace lattice
  using site::Site;
} // namespace TAT

#include "Data.hpp"
#include "Node.hpp"
#include "Tensor.hpp"
#include "Site.hpp"
#include "Lattice.hpp"

#endif // TAT_HPP_
