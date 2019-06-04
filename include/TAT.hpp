/** TAT.hpp
 * @file
 * @author  Hao Zhang <zh970204@mail.ustc.edu.cn>
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
#include <complex>

/**
 * 类似python中的pass,可以编译,但会在运行时报错.
 */
#define PASS std::cerr << "calling a passing function at " << __FILE__ << ":" << __LINE__ << " in " << __PRETTY_FUNCTION__ << std::endl, exit(233)

/** 
 * enable_if, 应用于一些type trait.
 */
#define ENABLE_IF(...) class = typename std::enable_if<__VA_ARGS__::value>::type

#if (!defined TAT_USE_CPU && !defined TAT_USE_CUDA && !defined TAT_USE_DCU && !defined TAT_USE_SW)
#if !defined TAT_DEFAULT
#warning use CPU by default
#endif
#define TAT_USE_CPU
#endif

#ifdef TAT_EXTREME
#warning EXTREME compile may cost much of compile time
#endif // TAT_EXTREME

#ifdef TAT_USE_CPU
extern "C"
{
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
} // extern "C"
#include <hptt.h>
#ifdef TAT_EXTREME
#include <../src/hptt.cpp>
#include <../src/plan.cpp>
#include <../src/transpose.cpp>
#include <../src/utils.cpp>
#endif // TAT_EXTREME
#include <rang.hpp>

// SVD
#if (defined TAT_USE_GESDD && defined TAT_USE_GESVD) || (defined TAT_USE_GESVD && defined TAT_USE_GESVDX) || (defined TAT_USE_GESVDX && defined TAT_USE_GESDD)
#error only one of GESDD, GESVD and GESVDX could be in use
#endif
#if (!defined TAT_USE_GESDD && !defined TAT_USE_GESVD && !defined TAT_USE_GESVDX)
#if !defined TAT_DEFAULT
#warning must use one of GESDD, GESVD and GESVDX, default use GESVD now
#endif
#define TAT_USE_GESVD
#endif

// QR
#if (defined TAT_USE_GEQRF && defined TAT_USE_GEQP3)
#error only one of GEQRF and GEQP3 could be in use
#endif
#if (!defined TAT_USE_GEQRF && !defined TAT_USE_GEQP3)
#if !defined TAT_DEFAULT
#warning must use one of GEQRF and GEQP3, default use GEQRF now
#endif
#define TAT_USE_GEQRF
#endif

#ifdef TAT_USE_GEQP3
#error GEQP3 is current unusable
#endif

#endif // TAT_USE_CPU

namespace TAT {
  namespace scalar_tools {
    /**
     * is_scalar provide the member `value` to tell whether Base is scalar type.
     * 
     * If Base is a scalar type, provides the member constant value equal true, and for any other type, value is false.
     * It is almost the same class as `std::is_scalar`, but value also equal true for `std::complex<Base>` where `std::is_scalar<Base>::value` is true,
     * 
     * the example is
     * ~~~{.cpp}
     * assert(is_scalar<double>::value);
     * assert(is_scalar<std::complex<float>>::value);
     * ~~~
     * 
     * @tparam Base the type to check whether is scalar.
     */
    template<class Base>
    class is_scalar {
    public:
      static constexpr bool value = std::is_scalar<Base>::value; /*!< true if std::is_scalar say Base is scalar */
    };
    /**
     * Specialization for std::complex<Base>, which should be true if std::is_scalar<Base> is true
     */
    template<class Base>
    class is_scalar<std::complex<Base>> {
    public:
      static constexpr bool value = std::is_scalar<Base>::value; /*!< true if std::is_scalar say Base in std::complex<Base> is scalar */
    };

    /**
     * _real_base provide member type unpack the real type from a complex type.
     * 
     * type if T if Base=std::complex<T> and else is Base.
     * 
     * for example, _real_base<std::complex<double>>::type is double.
     * 
     * @tparam Base the type to unpack
     */
    template<class Base>
    class _real_base {
    public:
      using type=Base; /*!< type is Base by default */
    };
    /**
     * Specialization for std::complex<Base> where type=Base.
     */
    template<class Base>
    class _real_base<std::complex<Base>> {
    public:
      using type=Base; /*< type is real type if Base is create by std::complex */
    };
    /**
     * RealBase return the real type from a complex type, or itself if Base is not created by std::complex.
     * 
     * @tparam Base the type to unpack
     */
    template<class Base>
    using RealBase = typename _real_base<Base>::type;
  } // namespace scalar_tools

  enum class Device : unsigned char {CPU, CUDA, DCU, SW};

  namespace legs {
    class Legs {
     public:
      unsigned char id;
      bool operator==(const Legs& other) const {
        return id==other.id;
      }
      bool operator!=(const Legs& other) const {
        return id!=other.id;
      }
      bool operator<(const Legs& other) const {
        return id<other.id;
      }
      Legs() = default;
      Legs(const std::string& name) {
        try {
          id = name2id.at(name);
        } catch (const std::out_of_range& e) {
          id = total++;
          name2id[name]=id;
          id2name[id]=name;
        } // exsit name
      }
      static unsigned char total;
      static std::map<std::string, unsigned char> name2id;
      static std::map<unsigned char, std::string> id2name;

      friend std::ostream& operator<<(std::ostream& out, const Legs& value) {
        try {
          return out << id2name.at(value.id);
        } catch (const std::out_of_range& e) {
          return out << "UserDefinedLeg" << value.id;
        } // get name
      } // operator<<
    }; // class Legs

    unsigned char Legs::total = 0;
    std::map<std::string, unsigned char> Legs::name2id = {};
    std::map<unsigned char, std::string> Legs::id2name = {};
  } // namespace legs
  using legs::Legs;

  namespace legs_name {
#define TAT_DefineLeg(x) static const TAT::Legs x(#x)
#define TAT_DefineLegs(n) \
      TAT_DefineLeg(Phy##n); TAT_DefineLeg(Left##n); TAT_DefineLeg(Right##n); TAT_DefineLeg(Up##n); TAT_DefineLeg(Down##n); \
      TAT_DefineLeg(LeftUp##n); TAT_DefineLeg(LeftDown##n); TAT_DefineLeg(RightUp##n); TAT_DefineLeg(RightDown##n)
#define TAT_Legs \
      TAT_DefineLegs(); TAT_DefineLegs(1); TAT_DefineLegs(2); TAT_DefineLegs(3); TAT_DefineLegs(4); \
      TAT_DefineLegs(5); TAT_DefineLegs(6); TAT_DefineLegs(7); TAT_DefineLegs(8); TAT_DefineLegs(9)
    TAT_Legs;
#undef TAT_Legs
#undef TAT_DefineLegs
#define TAT_DefineLegs(n) \
      TAT_DefineLeg(Tmp##n##0); TAT_DefineLeg(Tmp##n##1); TAT_DefineLeg(Tmp##n##2); TAT_DefineLeg(Tmp##n##3); TAT_DefineLeg(Tmp##n##4); \
      TAT_DefineLeg(Tmp##n##5); TAT_DefineLeg(Tmp##n##6); TAT_DefineLeg(Tmp##n##7); TAT_DefineLeg(Tmp##n##8); TAT_DefineLeg(Tmp##n##9)
#define TAT_Legs \
      TAT_DefineLegs(); TAT_DefineLegs(1); TAT_DefineLegs(2); TAT_DefineLegs(3); TAT_DefineLegs(4); \
      TAT_DefineLegs(5); TAT_DefineLegs(6); TAT_DefineLegs(7); TAT_DefineLegs(8); TAT_DefineLegs(9)
    TAT_Legs;
#undef TAT_Legs
#undef TAT_DefineLegs
#undef TAT_DefineLeg
    // total = 190 currently
  } // namespace legs_name

  using Size = std::size_t;
  using Rank = unsigned int;

  namespace data {
    template<class Base=double>
    class Data;
  } // namespace data
  using data::Data;

  namespace block {
    template<class Base=double>
    class Block;
  } // namespace block
  using block::Block;

  namespace tensor {
    template<class Base=double>
    class Tensor;
  } // namespace tensor
  //using tensor::Tensor;
  template<class Base=double>
  using Tensor=Block<Base>;

  namespace node {
    template<class Base=double>
    class Node;
  } // namespace node
  using node::Node;

  namespace lazy {
    class BaseLazy : public std::enable_shared_from_this<BaseLazy> {
     public:
      virtual void reset(bool reset_itself=true)=0;
    };

    template<class Base=double>
    class Lazy;
  } // namespace lazy
  using lazy::Lazy;

  // deprecated
  namespace site {
    template<class Base=double>
    class Site;
  } // namespace site
  using site::Site;

  namespace lattice {
    template<int dimension>
    class Dimension;

    template<class Tags=Dimension<2>, class Base=double>
    class Lattice;
  } // namespace lattice
  using site::Site;
} // namespace TAT

#include "TAT/Data.hpp"
#include "TAT/Block.hpp"
// Tensor
#include "TAT/Node.hpp"
#include "TAT/Lazy.hpp"

#include "TAT/Site.hpp"
#include "TAT/Lattice.hpp"

#endif // TAT_HPP_
