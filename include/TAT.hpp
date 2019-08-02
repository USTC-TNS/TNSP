/**
 * Copyright (C) 2019  Hao Zhang <zh970204@mail.ustc.edu.cn>
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

/**
 * @file
 */

#ifndef TAT_HPP_
#define TAT_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef TAT_VERSION
/**
 * TAT_VERSION is usually generated from git tag
 */
#      define TAT_VERSION "unknown"
#endif // TAT_VERSION

#if (!defined TAT_USE_CPU && !defined TAT_USE_CUDA && !defined TAT_USE_DCU && !defined TAT_USE_SW)
#      if !defined TAT_DEFAULT
#            warning use CPU by default
#      endif
#      define TAT_USE_CPU
#endif

#ifdef TAT_EXTREME
#      warning EXTREME compile may cost much of compile time
#endif // TAT_EXTREME

#ifdef TAT_USE_CPU
#      ifdef TAT_USE_MKL
extern "C" {
#            define MKL_Complex8 std::complex<float>
#            define MKL_Complex16 std::complex<double>
#            include <mkl.h>
} // extern "C"
#            ifdef TAT_USE_VML
#                  warning use intel mkl vml
#            endif // TAT_USE_VML
#      else
extern "C" {
#            define lapack_complex_float std::complex<float>
#            define lapack_complex_double std::complex<double>
#            include <cblas.h>
#            include <lapacke.h>
}
#            ifdef TAT_USE_VML
#                  error vml set on but mkl set off
#            endif // TAT_USE_VML
#      endif // TAT_USE_MKL
#      include <hptt.h>
#      ifdef TAT_EXTREME
#            include <../src/hptt.cpp>
#            include <../src/plan.cpp>
#            include <../src/transpose.cpp>
#            include <../src/utils.cpp>
#      endif // TAT_EXTREME
#      include <rang.hpp>

// SVD
#      if (defined TAT_USE_GESDD && defined TAT_USE_GESVD) || (defined TAT_USE_GESVD && defined TAT_USE_GESVDX) || \
            (defined TAT_USE_GESVDX && defined TAT_USE_GESDD)
#            error only one of GESDD, GESVD and GESVDX could be in use
#      endif
#      if (!defined TAT_USE_GESDD && !defined TAT_USE_GESVD && !defined TAT_USE_GESVDX)
#            if !defined TAT_DEFAULT
#                  warning must use one of GESDD, GESVD and GESVDX, default use GESVD now
#            endif
#            define TAT_USE_GESVD
#      endif

// QR
#      if (defined TAT_USE_GEQRF && defined TAT_USE_GEQP3)
#            error only one of GEQRF and GEQP3 could be in use
#      endif
#      if (!defined TAT_USE_GEQRF && !defined TAT_USE_GEQP3)
#            if !defined TAT_DEFAULT
#                  warning must use one of GEQRF and GEQP3, default use GEQRF now
#            endif
#            define TAT_USE_GEQRF
#      endif

#      ifdef TAT_USE_GEQP3
#            error GEQP3 is current unusable
#      endif
#endif // TAT_USE_CPU

/**
 * TAT is A Tensor library
 */
namespace TAT {
      /**
       * used for dimensions value
       */
      using Size = std::size_t;
      /**
       * used for dimensions index
       */
      using Rank = unsigned int;

      //
      //       SSS    CCC     AA    L        AA    RRRR          TTTTT   OOO    OOO   L       SSS
      //      S   S  C   C   A  A   L       A  A   R   R           T    O   O  O   O  L      S   S
      //      S      C      A    A  L      A    A  R   R           T    O   O  O   O  L      S
      //      S      C      A    A  L      A    A  R   R           T    O   O  O   O  L      S
      //       SSS   C      A    A  L      A    A  RRRR            T    O   O  O   O  L       SSS
      //          S  C      AAAAAA  L      AAAAAA  RR              T    O   O  O   O  L          S
      //          S  C      A    A  L      A    A  R R             T    O   O  O   O  L          S
      //      S   S  C   C  A    A  L      A    A  R  R            T    O   O  O   O  L      S   S
      //       SSS    CCC   A    A  LLLLL  A    A  R   R  _____    T     OOO    OOO   LLLLL   SSS
      //
      /**
       * define is_scalar_v and real_base_t for complex operators.
       */
      namespace scalar_tools {
            template<class T>
            class is_scalar {
               public:
                  static constexpr bool value = std::is_scalar<T>::value;
            };

            template<class T>
            class is_scalar<std::complex<T>> {
               public:
                  static constexpr bool value = std::is_scalar<T>::value;
            };

            /**
             * check if T is a scalar type
             */
            template<class T>
            static const bool is_scalar_v = is_scalar<T>::value;

            template<class T>
            class real_base {
               public:
                  using type = T;
            };

            template<class T>
            class real_base<std::complex<T>> {
               public:
                  using type = T;
            };

            /**
             * return correspond real type if T is complex type, return itself else
             */
            template<class T>
            using real_base_t = typename real_base<T>::type;
      } // namespace scalar_tools
      using scalar_tools::is_scalar_v;
      using scalar_tools::real_base_t;

      //
      //      L      EEEEE   GGGG    SSS
      //      L      E      G    G  S   S
      //      L      E      G       S
      //      L      E      G       S
      //      L      EEEE   G        SSS
      //      L      E      G  GGG      S
      //      L      E      G    G      S
      //      L      E      G   GG  S   S
      //      LLLLL  EEEEE   GGG G   SSS
      //
      /**
       * define class Legs.
       */
      namespace legs {
            /**
             * class Legs is to identify a leg with its id.
             *
             * Legs(string) create new legs with next id,
             * or return the legs created with same name.
             * Legs(IdType) to specify its id directly,
             * this method will NOT maintain id and name map.
             */
            class Legs {
               public:
                  using IdType = int;
                  IdType id = -1;
                  Legs() = default;
                  explicit Legs(IdType id) : id{id} {}
                  explicit Legs(const std::string& name) {
                        try {
                              id = name2id.at(name);
                        } catch (const std::out_of_range& e) {
                              id = total++;
                              name2id[name] = id;
                              id2name[id] = name;
                        } // exsit name
                  }
                  static IdType total;
                  static std::map<std::string, IdType> name2id;
                  static std::map<IdType, std::string> id2name;
            }; // class Legs

            Legs::IdType Legs::total = 0;
            std::map<std::string, Legs::IdType> Legs::name2id = {};
            std::map<Legs::IdType, std::string> Legs::id2name = {};

            bool operator==(const Legs& a, const Legs& b) {
                  return a.id == b.id;
            }
            bool operator!=(const Legs& a, const Legs& b) {
                  return a.id != b.id;
            }
            bool operator<(const Legs& a, const Legs& b) {
                  return a.id < b.id;
            }

            std::ostream& operator<<(std::ostream& out, const Legs& value) {
                  try {
                        return out << Legs::id2name.at(value.id);
                  } catch (const std::out_of_range& e) {
                        return out << "UserDefinedLeg" << value.id;
                  }
            }
      } // namespace legs
      using legs::Legs;

      //
      //      L      EEEEE   GGGG    SSS          N    N    AA    M     M  EEEEE
      //      L      E      G    G  S   S         N    N   A  A   MM   MM  E
      //      L      E      G       S             NN   N  A    A  M M M M  E
      //      L      E      G       S             N N  N  A    A  M  M  M  E
      //      L      EEEE   G        SSS          N  N N  A    A  M     M  EEEE
      //      L      E      G  GGG      S         N   NN  AAAAAA  M     M  E
      //      L      E      G    G      S         N    N  A    A  M     M  E
      //      L      E      G   GG  S   S         N    N  A    A  M     M  E
      //      LLLLL  EEEEE   GGG G   SSS   _____  N    N  A    A  M     M  EEEEE
      //
      /**
       * namespace legs_name containt 190 predefined legs.
       *
       * 190 predefined legs, including Tmp0~99,
       * and (Phy, 8 direction) * 0~9 (90 totaly).
       */
      namespace legs_name {
#define TAT_DefineLeg(x) static const TAT::Legs x(#x)
#define TAT_DefineLegs(n)         \
      TAT_DefineLeg(Phy##n);      \
      TAT_DefineLeg(Left##n);     \
      TAT_DefineLeg(Right##n);    \
      TAT_DefineLeg(Up##n);       \
      TAT_DefineLeg(Down##n);     \
      TAT_DefineLeg(LeftUp##n);   \
      TAT_DefineLeg(LeftDown##n); \
      TAT_DefineLeg(RightUp##n);  \
      TAT_DefineLeg(RightDown##n)
#define TAT_Legs         \
      TAT_DefineLegs();  \
      TAT_DefineLegs(1); \
      TAT_DefineLegs(2); \
      TAT_DefineLegs(3); \
      TAT_DefineLegs(4); \
      TAT_DefineLegs(5); \
      TAT_DefineLegs(6); \
      TAT_DefineLegs(7); \
      TAT_DefineLegs(8); \
      TAT_DefineLegs(9)
            TAT_Legs;
#undef TAT_Legs
#undef TAT_DefineLegs
#define TAT_DefineLegs(n)       \
      TAT_DefineLeg(Leg##n##0); \
      TAT_DefineLeg(Leg##n##1); \
      TAT_DefineLeg(Leg##n##2); \
      TAT_DefineLeg(Leg##n##3); \
      TAT_DefineLeg(Leg##n##4); \
      TAT_DefineLeg(Leg##n##5); \
      TAT_DefineLeg(Leg##n##6); \
      TAT_DefineLeg(Leg##n##7); \
      TAT_DefineLeg(Leg##n##8); \
      TAT_DefineLeg(Leg##n##9)
#define TAT_Legs         \
      TAT_DefineLegs();  \
      TAT_DefineLegs(1); \
      TAT_DefineLegs(2); \
      TAT_DefineLegs(3); \
      TAT_DefineLegs(4); \
      TAT_DefineLegs(5); \
      TAT_DefineLegs(6); \
      TAT_DefineLegs(7); \
      TAT_DefineLegs(8); \
      TAT_DefineLegs(9)
            TAT_Legs;
#undef TAT_Legs
#undef TAT_DefineLegs
#undef TAT_DefineLeg
      } // namespace legs_name

      //
      //      DDDDD     AA    TTTTT    AA
      //       D   D   A  A     T     A  A
      //       D   D  A    A    T    A    A
      //       D   D  A    A    T    A    A
      //       D   D  A    A    T    A    A
      //       D   D  AAAAAA    T    AAAAAA
      //       D   D  A    A    T    A    A
      //       D   D  A    A    T    A    A
      //      DDDDD   A    A    T    A    A
      //
      /**
       * define class Data.
       */
      namespace data {
            /**
             * Data is wrap of container, with several tensor operator,
             * but with no infomation about even tensor size.
             */
            template<class Base = double>
            class Data {
               public:
                  using type = Base;
                  std::vector<Base> base = {};
                  Size size = 0;

                  // constructors
                  explicit Data(Size size) : base(size), size(size) {}
                  explicit Data(Base num) : base(1), size(1) {
                        base[0] = num;
                  }

                  // default contructors, write then only to warn when copy.
                  ~Data() = default;
                  Data() = default;
                  Data(Data<Base>&& other) = default;
                  Data<Base>& operator=(Data<Base>&& other) = default;
                  Data(const Data<Base>& other) : base(other.base), size(other.size) {
#ifndef TAT_NOT_WARN_COPY
                        std::clog << "Copying Data..." << std::endl;
#endif // TAT_NOT_WARN_COPY
                  }
                  Data<Base>& operator=(const Data<Base>& other) {
                        base = other.base;
                        size = other.size;
#ifndef TAT_NOT_WARN_COPY
                        std::clog << "Copying Data..." << std::endl;
#endif // TAT_NOT_WARN_COPY
                  }

                  /**
                   * set data content with a function as argument, which return data one by one.
                   */
                  template<class Generator>
                  Data<Base>& set(Generator&& setter) & {
                        std::generate(base.begin(), base.end(), std::forward<Generator>(setter));
                        return *this;
                  }
                  template<class Generator>
                  Data<Base>&& set(Generator&& setter) && {
                        return std::move(set(std::forward<Generator>(setter)));
                  }

                  // operators

                  /**
                   * convert value type to another.
                   */
                  template<class Base2>
                  Data<Base2> to() const {
                        auto res = Data<Base2>{size};
                        for (Size i = 0; i < size; i++) {
                              res.base[i] = static_cast<Base2>(base[i]);
                        }
                        return res;
                  }

                  /**
                   * get n-norm of this tensor, if n=-1, it means max norm.
                   */
                  template<int n>
                  Data<Base> norm() const;

                  Data<Base> transpose(const std::vector<Size>& dims, const std::vector<Rank>& plan) const;

                  class svd_res {
                     public:
                        Data<Base> U;
                        Data<real_base_t<Base>> S;
                        Data<Base> V;
                  }; // class svd_res

                  svd_res
                  svd(const std::vector<Size>& dims,
                      const std::vector<Rank>& plan,
                      const Size& u_size,
                      const Size& v_size,
                      const Size& min_mn,
                      const Size& cut) const;

                  class qr_res {
                     public:
                        Data<Base> Q;
                        Data<Base> R;
                  }; // class qr_res

                  qr_res
                  qr(const std::vector<Size>& dims,
                     const std::vector<Rank>& plan,
                     const Size& q_size,
                     const Size& r_size,
                     const Size& min_mn) const;

                  static Data<Base> contract(
                        const Data<Base>& data1,
                        const Data<Base>& data2,
                        const std::vector<Size>& dims1,
                        const std::vector<Size>& dims2,
                        const std::vector<Rank>& plan1,
                        const std::vector<Rank>& plan2,
                        const Size& m,
                        const Size& k,
                        const Size& n);

                  Data<Base> multiple(const Data<Base>& other, const Size& a, const Size& b, const Size& c) const;

                  Base at(Size pos) const {
                        return base[pos];
                  }
            };
      } // namespace data
      using data::Data;

      //
      //      BBBBB   L       OOO    CCC   K    K
      //       B   B  L      O   O  C   C  K   K
      //       B   B  L      O   O  C      K  K
      //       B   B  L      O   O  C      K K
      //       BBBB   L      O   O  C      KK
      //       B   B  L      O   O  C      K K
      //       B   B  L      O   O  C      K  K
      //       B   B  L      O   O  C   C  K   K
      //      BBBBB   LLLLL   OOO    CCC   K    K
      //
      /**
       * define class Block.
       */
      namespace block {
            /**
             * Block record the dimension size of a tensor based on class Data.
             */
            template<class Base = double>
            class Block {
               public:
                  using type = Base;
                  Data<Base> data = {};
                  std::vector<Size> dims = {};

                  // constructors
                  template<class S = std::vector<Size>>
                  explicit Block(S&& dims) :
                        data(std::accumulate(dims.begin(), dims.end(), Size(1), std::multiplies<Size>())),
                        dims(std::forward<S>(dims)) {}
                  explicit Block(Base num) : data(num), dims({}) {}

                  // default contructors, I don't know why it should be written explicitly
                  ~Block() = default;
                  Block() = default;
                  Block(Block<Base>&& other) = default;
                  Block<Base>& operator=(Block<Base>&& other) = default;
                  Block(const Block<Base>& other) = default;
                  Block<Base>& operator=(const Block<Base>& other) = default;

                  // data member
                  const Size& size() const {
                        return data.size;
                  }

                  /**
                   * set block content with a function as argument, which return data one by one.
                   */
                  template<class Generator>
                  Block<Base>& set(Generator&& setter) & {
                        data.set(std::forward<Generator>(setter));
                        return *this;
                  }
                  template<class Generator>
                  Block<Base>&& set(Generator&& setter) && {
                        return std::move(set(std::forward<Generator>(setter)));
                  }

                  // operators

                  /**
                   * convert value type to another
                   */
                  template<class Base2>
                  Block<Base2> to() const {
                        auto res = Block<Base2>{};
                        res.data = data.template to<Base2>();
                        res.dims = dims;
                        return res;
                  }

                  /**
                   * get n-norm of this block, if n=-1, it means max norm.
                   */
                  template<int n>
                  Block<Base> norm() const;

                  Block<Base> transpose(const std::vector<Rank>& plan) const;

                  class svd_res {
                     public:
                        Block<Base> U;
                        Block<real_base_t<Base>> S;
                        Block<Base> V;
                  }; // class svd_res

                  svd_res svd(const std::vector<Rank>& plan, const Rank& u_rank, const Size& cut) const;

                  class qr_res {
                     public:
                        Block<Base> Q;
                        Block<Base> R;
                  }; // class qr_res

                  qr_res qr(const std::vector<Rank>& plan, const Rank& q_rank) const;

                  static Block<Base> contract(
                        const Block<Base>& block1,
                        const Block<Base>& block2,
                        const std::vector<Rank>& plan1,
                        const std::vector<Rank>& plan2,
                        const Rank& contract_num);

                  Block<Base> multiple(const Block<Base>& other, const Rank& index) const;

                  Base at(const std::vector<Size>& pos) const {
                        Size res = 0;
                        for (Rank i = 0; i < dims.size(); i++) {
                              res = res * dims[i] + pos[i];
                        }
                        return data.at(res);
                  }
            };
      } // namespace block
      using block::Block;

      /**
       * define class Tensor.
       *
       * but not implemented,
       * currently use dense tensor,
       * namely block directly.
       */
      // namespace tensor {
      //       template<class=double>
      //       class Tensor;
      // } // namespace tensor
      // using tensor::Tensor;
      namespace tensor = block;
      template<class Base = double>
      using Tensor = Block<Base>;

      //
      //      V   V  EEEEE   CCC   TTTTT   OOO   RRRR          L      EEEEE   GGGG    SSS
      //      V   V  E      C   C    T    O   O  R   R         L      E      G    G  S   S
      //      V   V  E      C        T    O   O  R   R         L      E      G       S
      //       V V   E      C        T    O   O  R   R         L      E      G       S
      //       V V   EEEE   C        T    O   O  RRRR          L      EEEE   G        SSS
      //       V V   E      C        T    O   O  RR            L      E      G  GGG      S
      //        V    E      C        T    O   O  R R           L      E      G    G      S
      //        V    E      C   C    T    O   O  R  R          L      E      G   GG  S   S
      //        V    EEEEE   CCC     T     OOO   R   R  _____  LLLLL  EEEEE   GGG G   SSS
      //
      namespace vector_legs {
            template<class T>
            bool no_duplicated(const std::vector<T>& legs) {
                  return legs.size() == std::set<T>(legs.begin(), legs.end()).size();
            }

            template<class T>
            void replace(std::vector<T>& legs, const std::vector<std::tuple<T, T>>& dict) {
                  // it is allowed if i not in legs
                  for (const auto& [i, j] : dict) {
                        std::replace(legs.begin(), legs.end(), i, j);
                  }
            }

            template<class T>
            std::vector<T> filter_out_not_in(const std::vector<T>& new_legs, const std::vector<T>& legs) {
                  std::vector<T> res;
                  std::copy_if(new_legs.begin(), new_legs.end(), std::back_inserter(res), [&](const Legs& i) {
                        return std::find(legs.begin(), legs.end(), i) != legs.end();
                  });
                  return res;
            }

            template<class T>
            std::vector<T> filter_out_in(const std::vector<T>& new_legs, const std::vector<T>& legs) {
                  std::vector<T> res;
                  std::copy_if(new_legs.begin(), new_legs.end(), std::back_inserter(res), [&](const Legs& i) {
                        return std::find(legs.begin(), legs.end(), i) == legs.end();
                  });
                  return res;
            }

            template<class T>
            auto find_iter(const std::vector<T>& vec, const T& i) {
                  return std::find(vec.begin(), vec.end(), i);
            }

            template<class T>
            void append(std::vector<T>& a, const std::vector<T>& b) {
                  a.insert(a.end(), b.begin(), b.end());
            } // append
      } // namespace vector_legs

      //
      //      N    N   OOO   DDDDD   EEEEE
      //      N    N  O   O   D   D  E
      //      NN   N  O   O   D   D  E
      //      N N  N  O   O   D   D  E
      //      N  N N  O   O   D   D  EEEE
      //      N   NN  O   O   D   D  E
      //      N    N  O   O   D   D  E
      //      N    N  O   O   D   D  E
      //      N    N   OOO   DDDDD   EEEEE
      //
      /**
       * define class Node.
       */
      namespace node {
            /**
             * besides dimenion size, class Node also maintain Legs name of a tensor.
             */
            template<class Base = double>
            class Node {
               public:
                  using type = Base;
                  Tensor<Base> tensor = {};
                  std::vector<Legs> legs = {};

                  // constructors
                  // this will rewrite after implement tensor
                  template<class T1 = std::vector<Legs>, class T2 = std::vector<Size>>
                  explicit Node(T1&& _legs, T2&& _dims) :
                        tensor(std::forward<T2>(_dims)), legs(std::forward<T1>(_legs)) {
                        // expect length of legs and dims is same
                        assert(legs.size() == tensor.dims.size());
                        // expect no same element in legs
                        assert(vector_legs::no_duplicated(legs));
                  }
                  explicit Node(Base num) : tensor(num), legs({}) {}

                  // default contructors, I don't know why it should be written explicitly
                  ~Node() = default;
                  Node() = default;
                  Node(Node<Base>&& other) = default;
                  Node<Base>& operator=(Node<Base>&& other) = default;
                  Node(const Node<Base>& other) = default;
                  Node<Base>& operator=(const Node<Base>& other) = default;

                  // tensor member
                  const Size& size() const {
                        return tensor.size();
                  }

                  const std::vector<Size>& dims() const {
                        return tensor.dims();
                  }

                  /**
                   * set node content with a function as argument, which return data one by one.
                   */
                  template<class Generator>
                  Node<Base>& set(Generator&& setter) & {
                        tensor.set(std::forward<Generator>(setter));
                        return *this;
                  }
                  template<class Generator>
                  Node<Base>&& set(Generator&& setter) && {
                        return std::move(set(std::forward<Generator>(setter)));
                  }

                  // operators
                  /**
                   * rename the legs name of node inplace
                   */
                  Node<Base>& legs_rename(const std::vector<std::tuple<Legs, Legs>>& dict) & {
                        vector_legs::replace(legs, dict);
                        return *this;
                  }
                  Node<Base>&& legs_rename(const std::vector<std::tuple<Legs, Legs>>& dict) && {
                        return std::move(legs_rename(dict));
                  }

                  /**
                   * convert value type to another
                   */
                  template<class Base2>
                  Node<Base2> to() const {
                        auto res = Node<Base2>{};
                        res.tensor = tensor.template to<Base2>();
                        res.legs = legs;
                        return res;
                  }

                  /**
                   * get n-norm of this node, if n=-1, it means max norm.
                   */
                  template<int n>
                  Node<Base> norm() const;

                  Node<Base> transpose(const std::vector<Legs>& new_legs) const;

                  class svd_res {
                     public:
                        Node<Base> U;
                        Node<real_base_t<Base>> S;
                        Node<Base> V;
                  };

                  svd_res
                  svd(const std::vector<Legs>& input_u_legs,
                      const Legs& new_u_legs,
                      const Legs& new_v_legs,
                      const Rank& cut = -1) const;

                  class qr_res {
                     public:
                        Node<Base> Q;
                        Node<Base> R;
                  }; // class qr_res

                  qr_res
                  qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const;

                  static Node<Base> contract(
                        const Node<Base>& node1,
                        const Node<Base>& node2,
                        const std::vector<Legs>& legs1,
                        const std::vector<Legs>& legs2,
                        const std::vector<std::tuple<Legs, Legs>>& map1 = {},
                        const std::vector<std::tuple<Legs, Legs>>& map2 = {});

                  Node<Base> multiple(const Node<Base>& other, const Legs& position) const;

                  Base at(const std::map<Legs, Size>& dict) const {
                        std::vector<Size> res;
                        std::transform(legs.begin(), legs.end(), std::back_inserter(res), [&dict](const Legs& l) {
                              return dict.at(l);
                        });
                        return tensor.at(res);
                  }
            };
      } // namespace node
      using node::Node;

      //
      //      III   OOO
      //       I   O   O
      //       I   O   O
      //       I   O   O
      //       I   O   O
      //       I   O   O
      //       I   O   O
      //       I   O   O
      //      III   OOO
      //
      namespace data {
            template<class Base>
            std::ostream& operator<<(std::ostream& out, const Data<Base>& value) {
                  out << "{\"" << rang::fgB::green << "size\": " << value.size << rang::fg::reset << ", "
                      << rang::fg::yellow << "\"base\": [";
                  if (value.size != 0) {
                        for (Size i = 0; i < value.size - 1; i++) {
                              out << value.base[i] << ", ";
                        } // for i
                        out << value.base[value.size - 1];
                  } // if
                  out << "]" << rang::fg::reset << "}";
                  return out;
            } // operator<<

            template<class Base>
            std::ofstream& operator<<(std::ofstream& out, const Data<Base>& value) {
                  out.write(reinterpret_cast<const char*>(&value.size), sizeof(Size));
                  out.write(reinterpret_cast<const char*>(value.base.data()), value.size * sizeof(Base));
                  return out;
            } // operator<<

            template<class Base>
            std::ifstream& operator>>(std::ifstream& in, Data<Base>& value) {
                  in.read(reinterpret_cast<char*>(&value.size), sizeof(Size));
                  value.base = std::vector<Base>(value.size);
                  in.read(reinterpret_cast<char*>(value.base.data()), value.size * sizeof(Base));
                  return in;
            } // operator<<
      } // namespace data
      namespace block {
            std::ostream& operator<<(std::ostream& out, const std::vector<Size>& value) {
                  Rank size = value.size();
                  out << "[";
                  for (Rank i = 0; i < size; i++) {
                        out << value[i];
                        if (i != size - 1) {
                              out << ", ";
                        } // if not last
                  } // for i
                  out << "]";
                  return out;
            } // operator<<

            template<class Base>
            std::ostream& operator<<(std::ostream& out, const Block<Base>& value) {
                  return out << "{" << rang::fg::magenta << "\"dims\": " << value.dims << rang::fg::reset
                             << ", \"data\": " << value.data << "}";
            } // operator<<

            template<class Base>
            std::ofstream& operator<<(std::ofstream& out, const Block<Base>& value) {
                  Rank rank = value.dims.size();
                  out.write(reinterpret_cast<const char*>(&rank), sizeof(Rank));
                  out.write(reinterpret_cast<const char*>(value.dims.data()), rank * sizeof(Size));
                  out << value.data;
                  return out;
            } // operator<<

            template<class Base>
            std::ifstream& operator>>(std::ifstream& in, Block<Base>& value) {
                  Rank rank;
                  in.read(reinterpret_cast<char*>(&rank), sizeof(Rank));
                  value.dims = std::vector<Size>(rank);
                  in.read(reinterpret_cast<char*>(value.dims.data()), rank * sizeof(Size));
                  in >> value.data;
                  return in;
            } // operator<<
      } // namespace block
      namespace node {
            std::ostream& operator<<(std::ostream& out, const std::vector<Legs>& value) {
                  Rank size = value.size();
                  out << "[";
                  for (Rank i = 0; i < size; i++) {
                        out << "\"" << value[i] << "\"";
                        if (i != size - 1) {
                              out << ", ";
                        } // if not last
                  } // for i
                  out << "]";
                  return out;
            } // operator<<

            template<class Base>
            std::ostream& operator<<(std::ostream& out, const Node<Base>& value) {
                  return out << "{" << rang::fgB::yellow << "\"rank\": " << value.legs.size() << rang::fg::reset << ", "
                             << rang::fgB::blue << "\"legs\": " << value.legs << rang::fg::reset
                             << ", \"tensor\": " << value.tensor << "}";
            } // operator<<

            template<class Base>
            std::ofstream& operator<<(std::ofstream& out, const Node<Base>& value) {
                  Rank rank = value.legs.size();
                  out.write(reinterpret_cast<const char*>(&rank), sizeof(Rank));
                  out.write(reinterpret_cast<const char*>(value.legs.data()), rank * sizeof(Legs));
                  out << value.tensor;
                  return out;
            } // operator<<

            template<class Base>
            std::ifstream& operator>>(std::ifstream& in, Node<Base>& value) {
                  Rank rank;
                  in.read(reinterpret_cast<char*>(&rank), sizeof(Rank));
                  value.legs = std::vector<Legs>(rank);
                  in.read(reinterpret_cast<char*>(value.legs.data()), rank * sizeof(Legs));
                  in >> value.tensor;
                  return in;
            } // operator<<
      } // namespace node
      //
      //      N    N   OOO   RRRR   M     M
      //      N    N  O   O  R   R  MM   MM
      //      NN   N  O   O  R   R  M M M M
      //      N N  N  O   O  R   R  M  M  M
      //      N  N N  O   O  RRRR   M     M
      //      N   NN  O   O  RR     M     M
      //      N    N  O   O  R R    M     M
      //      N    N  O   O  R  R   M     M
      //      N    N   OOO   R   R  M     M
      //
      namespace data {

            namespace norm {
#ifdef TAT_USE_VML
                  template<class Base>
                  void vAbs(const Size& size, const Base* a, real_base_t<Base>* y);

                  template<>
                  void vAbs<float>(const Size& size, const float* a, float* y) {
                        vsAbs(size, a, y);
                  } // vAbs<float>

                  template<>
                  void vAbs<double>(const Size& size, const double* a, double* y) {
                        vdAbs(size, a, y);
                  } // vAbs<double>

                  template<>
                  void vAbs<std::complex<float>>(const Size& size, const std::complex<float>* a, float* y) {
                        vcAbs(size, a, y);
                  } // vAbs<std::complex<float>>

                  template<>
                  void vAbs<std::complex<double>>(const Size& size, const std::complex<double>* a, double* y) {
                        vzAbs(size, a, y);
                  } // vAbs<std::complex<double>>

                  template<class Base>
                  void vPowx(const Size& size, const Base* a, const Base& n, Base* y);

                  template<>
                  void vPowx<float>(const Size& size, const float* a, const float& n, float* y) {
                        vsPowx(size, a, n, y);
                  } // vPowx<float>

                  template<>
                  void vPowx<double>(const Size& size, const double* a, const double& n, double* y) {
                        vdPowx(size, a, n, y);
                  } // vPowx<double>

                  template<class Base>
                  void vSqr(const Size& size, const Base* a, Base* y);

                  template<>
                  void vSqr<float>(const Size& size, const float* a, float* y) {
                        vsSqr(size, a, y);
                  } // vSqr<float>

                  template<>
                  void vSqr<double>(const Size& size, const double* a, double* y) {
                        vdSqr(size, a, y);
                  } // vSqr<double>

                  template<class Base>
                  Base asum(const Size& size, const Base* a);

                  template<>
                  float asum<float>(const Size& size, const float* a) {
                        return cblas_sasum(size, a, 1);
                  } // asum<float>

                  template<>
                  double asum<double>(const Size& size, const double* a) {
                        return cblas_dasum(size, a, 1);
                  } // asum<double>

                  template<class Base>
                  CBLAS_INDEX iamax(const Size& size, const Base* a);

                  template<>
                  CBLAS_INDEX iamax<float>(const Size& size, const float* a) {
                        return cblas_isamax(size, a, 1);
                  } // iamax<float>

                  template<>
                  CBLAS_INDEX iamax<double>(const Size& size, const double* a) {
                        return cblas_idamax(size, a, 1);
                  } // iamax<double>

                  template<>
                  CBLAS_INDEX iamax<std::complex<float>>(const Size& size, const std::complex<float>* a) {
                        return cblas_icamax(size, a, 1);
                  } // iamax<std::complex<float>>

                  template<>
                  CBLAS_INDEX iamax<std::complex<double>>(const Size& size, const std::complex<double>* a) {
                        return cblas_izamax(size, a, 1);
                  } // iamax<std::complex<double>>

                  template<class Base, int n>
                  Base run(const Size& size, const Base* data) {
                        if constexpr (n == -2) {
                              auto i = iamax<Base>(size, data);
                              return std::abs(data[i]);
                        }
                        auto tmp = std::vector<real_base_t<Base>>(size);
                        if constexpr ((std::is_same_v<Base, real_base_t<Base>>)&&(n == 2)) {
                              vSqr<real_base_t<Base>>(size, data, tmp.data());
                        } else {
                              vAbs<Base>(size, data, tmp.data());
                              if constexpr (n == 2) {
                                    vSqr<real_base_t<Base>>(size, tmp.data(), tmp.data());
                              } else {
                                    vPowx<real_base_t<Base>>(size, tmp.data(), real_base_t<Base>(n), tmp.data());
                              }
                        }
                        auto res = asum<real_base_t<Base>>(size, tmp.data());
                        return std::pow(res, 1. / n);
                  } // run
#else
                  template<class Base, int n>
                  Base run(const Size& size, const Base* data) {
                        if constexpr (n == -1) {
                              if (size == 0) {
                                    return 0;
                              }
                              auto max = std::abs(data[0]);
                              for (Size i = 1; i < size; i++) {
                                    auto tmp = std::abs(data[i]);
                                    if (tmp > max) {
                                          max = tmp;
                                    }
                              }
                              return max;
                        }
                        real_base_t<Base> sum = 0;
                        for (Size i = 0; i < size; i++) {
                              if constexpr (n == 2) {
                                    real_base_t<Base> tmp;
                                    if constexpr (std::is_same_v<Base, real_base_t<Base>>) {
                                          // real
                                          tmp = data[i];
                                    } else {
                                          // complex
                                          tmp = std::abs(data[i]);
                                    }
                                    sum += tmp * tmp;
                              } else {
                                    sum += std::pow(std::abs(data[i]), n);
                              }
                        }
                        return std::pow(sum, 1. / n);
                  } // run
#endif // TAT_USE_VML
            } // namespace norm

            template<class Base>
            template<int n>
            Data<Base> Data<Base>::norm() const {
                  return Data<Base>(norm::run<Base, n>(size, base.data()));
            } // norm
      } // namespace data
      namespace block {
            template<class Base>
            template<int n>
            Block<Base> Block<Base>::norm() const {
                  auto res = Block<Base>{};
                  res.data = data.template norm<n>();
                  return res;
            } // norm
      } // namespace block
      namespace node {
            template<class Base>
            template<int n>
            Node<Base> Node<Base>::norm() const {
                  auto res = Node<Base>{};
                  res.tensor = tensor.template norm<n>();
                  return res;
            } // norm
      } // namespace node
      //
      //       SSS    CCC     AA    L        AA    RRRR          M     M  K    K  L
      //      S   S  C   C   A  A   L       A  A   R   R         MM   MM  K   K   L
      //      S      C      A    A  L      A    A  R   R         M M M M  K  K    L
      //      S      C      A    A  L      A    A  R   R         M  M  M  K K     L
      //       SSS   C      A    A  L      A    A  RRRR          M     M  KK      L
      //          S  C      AAAAAA  L      AAAAAA  RR            M     M  K K     L
      //          S  C      A    A  L      A    A  R R           M     M  K  K    L
      //      S   S  C   C  A    A  L      A    A  R  R          M     M  K   K   L
      //       SSS    CCC   A    A  LLLLL  A    A  R   R  _____  M     M  K    K  LLLLL
      //
      namespace data {
#ifdef TAT_USE_VML
            namespace scalar {
                  template<class Base>
                  void vLinearFrac(
                        const Size& n,
                        const Base* a,
                        const Base* b,
                        const Base& sa,
                        const Base& oa,
                        const Base& sb,
                        const Base& ob,
                        Base* y);
                  // y = (a*sa + oa)/(b*sb + ob)

                  template<>
                  void vLinearFrac<float>(
                        const Size& n,
                        const float* a,
                        const float* b,
                        const float& sa,
                        const float& oa,
                        const float& sb,
                        const float& ob,
                        float* y) {
                        vsLinearFrac(n, a, b, sa, oa, sb, ob, y);
                  } // vLinearFrac

                  template<>
                  void vLinearFrac<double>(
                        const Size& n,
                        const double* a,
                        const double* b,
                        const double& sa,
                        const double& oa,
                        const double& sb,
                        const double& ob,
                        double* y) {
                        vdLinearFrac(n, a, b, sa, oa, sb, ob, y);
                  } // vLinearFrac

                  template<>
                  void vLinearFrac<std::complex<float>>(
                        const Size& n,
                        const std::complex<float>* a,
                        const std::complex<float>* b,
                        const std::complex<float>& sa,
                        const std::complex<float>& oa,
                        const std::complex<float>& sb,
                        const std::complex<float>& ob,
                        std::complex<float>* y) {
                        // vcLinearFrac(n, a, b, sa, oa, sb, ob, y);
                        for (Size i = 0; i < n; i++) {
                              y[i] = (a[i] * sa + oa) / (b[i] * sb + ob);
                        } // for
                  } // vLinearFrac

                  template<>
                  void vLinearFrac<std::complex<double>>(
                        const Size& n,
                        const std::complex<double>* a,
                        const std::complex<double>* b,
                        const std::complex<double>& sa,
                        const std::complex<double>& oa,
                        const std::complex<double>& sb,
                        const std::complex<double>& ob,
                        std::complex<double>* y) {
                        // vzLinearFrac(n, a, b, sa, oa, sb, ob, y);
                        for (Size i = 0; i < n; i++) {
                              y[i] = (a[i] * sa + oa) / (b[i] * sb + ob);
                        } // for
                  } // vLinearFrac

                  template<class Base>
                  void LinearFrac(
                        const Data<Base>& src,
                        Data<Base>& dst,
                        const Base& sa,
                        const Base& oa,
                        const Base& sb,
                        const Base& ob) {
                        assert(src.size == dst.size);
                        vLinearFrac<Base>(src.size, src.base.data(), src.base.data(), sa, oa, sb, ob, dst.base.data());
                  } // LinearFrac

                  template<class Base>
                  void vAdd(const Size& n, const Base* a, const Base* b, Base* y);

                  template<>
                  void vAdd<float>(const Size& n, const float* a, const float* b, float* y) {
                        vsAdd(n, a, b, y);
                  } // vAdd

                  template<>
                  void vAdd<double>(const Size& n, const double* a, const double* b, double* y) {
                        vdAdd(n, a, b, y);
                  } // vAdd

                  template<>
                  void vAdd<std::complex<float>>(
                        const Size& n,
                        const std::complex<float>* a,
                        const std::complex<float>* b,
                        std::complex<float>* y) {
                        vcAdd(n, a, b, y);
                  } // vAdd

                  template<>
                  void vAdd<std::complex<double>>(
                        const Size& n,
                        const std::complex<double>* a,
                        const std::complex<double>* b,
                        std::complex<double>* y) {
                        vzAdd(n, a, b, y);
                  } // vAdd

                  template<class Base>
                  void Add(const Data<Base>& a, const Data<Base>& b, Data<Base>& y) {
                        assert(a.size == b.size);
                        vAdd<Base>(a.size, a.base.data(), b.base.data(), y.base.data());
                  } // Add

                  template<class Base>
                  void vSub(const Size& n, const Base* a, const Base* b, Base* y);

                  template<>
                  void vSub<float>(const Size& n, const float* a, const float* b, float* y) {
                        vsSub(n, a, b, y);
                  } // vSub

                  template<>
                  void vSub<double>(const Size& n, const double* a, const double* b, double* y) {
                        vdSub(n, a, b, y);
                  } // vSub

                  template<>
                  void vSub<std::complex<float>>(
                        const Size& n,
                        const std::complex<float>* a,
                        const std::complex<float>* b,
                        std::complex<float>* y) {
                        vcSub(n, a, b, y);
                  } // vSub

                  template<>
                  void vSub<std::complex<double>>(
                        const Size& n,
                        const std::complex<double>* a,
                        const std::complex<double>* b,
                        std::complex<double>* y) {
                        vzSub(n, a, b, y);
                  } // vSub

                  template<class Base>
                  void Sub(const Data<Base>& a, const Data<Base>& b, Data<Base>& y) {
                        assert(a.size == b.size);
                        vSub<Base>(a.size, a.base.data(), b.base.data(), y.base.data());
                  } // Sub

                  template<class Base>
                  void vMul(const Size& n, const Base* a, const Base* b, Base* y);

                  template<>
                  void vMul<float>(const Size& n, const float* a, const float* b, float* y) {
                        vsMul(n, a, b, y);
                  } // vMul

                  template<>
                  void vMul<double>(const Size& n, const double* a, const double* b, double* y) {
                        vdMul(n, a, b, y);
                  } // vMul

                  template<>
                  void vMul<std::complex<float>>(
                        const Size& n,
                        const std::complex<float>* a,
                        const std::complex<float>* b,
                        std::complex<float>* y) {
                        vcMul(n, a, b, y);
                  } // vMul

                  template<>
                  void vMul<std::complex<double>>(
                        const Size& n,
                        const std::complex<double>* a,
                        const std::complex<double>* b,
                        std::complex<double>* y) {
                        vzMul(n, a, b, y);
                  } // vMul

                  template<class Base>
                  void Mul(const Data<Base>& a, const Data<Base>& b, Data<Base>& y) {
                        assert(a.size == b.size);
                        vMul<Base>(a.size, a.base.data(), b.base.data(), y.base.data());
                  } // Mul

                  template<class Base>
                  void vDiv(const Size& n, const Base* a, const Base* b, Base* y);

                  template<>
                  void vDiv<float>(const Size& n, const float* a, const float* b, float* y) {
                        vsDiv(n, a, b, y);
                  } // vDiv

                  template<>
                  void vDiv<double>(const Size& n, const double* a, const double* b, double* y) {
                        vdDiv(n, a, b, y);
                  } // vDiv

                  template<>
                  void vDiv<std::complex<float>>(
                        const Size& n,
                        const std::complex<float>* a,
                        const std::complex<float>* b,
                        std::complex<float>* y) {
                        vcDiv(n, a, b, y);
                  } // vDiv

                  template<>
                  void vDiv<std::complex<double>>(
                        const Size& n,
                        const std::complex<double>* a,
                        const std::complex<double>* b,
                        std::complex<double>* y) {
                        vzDiv(n, a, b, y);
                  } // vDiv

                  template<class Base>
                  void Div(const Data<Base>& a, const Data<Base>& b, Data<Base>& y) {
                        assert(a.size == b.size);
                        vDiv<Base>(a.size, a.base.data(), b.base.data(), y.base.data());
                  } // Div
            } // namespace scalar
#endif // TAT_USE_VML
      } // namespace data

      //
      //       SSS    CCC     AA    L        AA    RRRR          DDDDD     AA    TTTTT    AA
      //      S   S  C   C   A  A   L       A  A   R   R          D   D   A  A     T     A  A
      //      S      C      A    A  L      A    A  R   R          D   D  A    A    T    A    A
      //      S      C      A    A  L      A    A  R   R          D   D  A    A    T    A    A
      //       SSS   C      A    A  L      A    A  RRRR           D   D  A    A    T    A    A
      //          S  C      AAAAAA  L      AAAAAA  RR             D   D  AAAAAA    T    AAAAAA
      //          S  C      A    A  L      A    A  R R            D   D  A    A    T    A    A
      //      S   S  C   C  A    A  L      A    A  R  R           D   D  A    A    T    A    A
      //       SSS    CCC   A    A  LLLLL  A    A  R   R  _____  DDDDD   A    A    T    A    A
      //
      namespace data {
            template<class Base>
            Data<Base>& operator*=(Data<Base>& a, const Data<Base>& b) {
                  if (b.size == 1) {
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(a, a, b.base[0], 0, 0, 1);
#else
                        for (Size i = 0; i < a.size; i++) {
                              a.base[i] *= b.base[0];
                        }
#endif
                  } else {
#ifdef TAT_USE_VML
                        scalar::Mul<Base>(a, b, a);
#else
                        for (Size i = 0; i < a.size; i++) {
                              a.base[i] *= b.base[i];
                        }
#endif
                  } // if
                  return a;
            } // operator*=

            template<class Base>
            Data<Base> operator*(const Data<Base>& a, const Data<Base>& b) {
                  if (a.size == 1) {
                        auto res = Data<Base>(b.size);
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(b, res, a.base[0], 0, 0, 1);
#else
                        for (Size i = 0; i < b.size; i++) {
                              res.base[i] = a.base[0] * b.base[i];
                        }
#endif
                        return res;
                  } // if
                  if (b.size == 1) {
                        auto res = Data<Base>(a.size);
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(a, res, b.base[0], 0, 0, 1);
#else
                        for (Size i = 0; i < a.size; i++) {
                              res.base[i] = a.base[i] * b.base[0];
                        }
#endif
                        return res;
                  } // if
                  auto res = Data<Base>(a.size);
#ifdef TAT_USE_VML
                  scalar::Mul<Base>(a, b, res);
#else
                  for (Size i = 0; i < a.size; i++) {
                        res.base[i] = a.base[i] * b.base[i];
                  }
#endif
                  return res;
            } // operator*

            template<class Base>
            Data<Base>& operator/=(Data<Base>& a, const Data<Base>& b) {
                  if (b.size == 1) {
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(a, a, 1, 0, 0, *b.base.data());
#else
                        for (Size i = 0; i < a.size; i++) {
                              a.base[i] /= b.base[0];
                        }
#endif
                  } else {
#ifdef TAT_USE_VML
                        scalar::Div<Base>(a, b, a);
#else
                        for (Size i = 0; i < a.size; i++) {
                              a.base[i] /= b.base[i];
                        }
#endif
                  } // if
                  return a;
            } // operator/=

            template<class Base>
            Data<Base> operator/(const Data<Base>& a, const Data<Base>& b) {
                  if (a.size == 1) {
                        auto res = Data<Base>(b.size);
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(b, res, 0, *a.base.data(), 1, 0);
#else
                        for (Size i = 0; i < b.size; i++) {
                              res.base[i] = a.base[0] / b.base[i];
                        }
#endif
                        return res;
                  } // if
                  if (b.size == 1) {
                        Data<Base> res(a.size);
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(a, res, 1, 0, 0, *b.base.data());
#else
                        for (Size i = 0; i < a.size; i++) {
                              res.base[i] = a.base[i] / b.base[0];
                        }
#endif
                        return res;
                  } // if
                  Data<Base> res(a.size);
#ifdef TAT_USE_VML
                  scalar::Div<Base>(a, b, res);
#else
                  for (Size i = 0; i < a.size; i++) {
                        res.base[i] = a.base[i] / b.base[i];
                  }
#endif
                  return res;
            } // operator/

            template<class Base>
            const Data<Base>& operator+(const Data<Base>& a) {
                  return a;
            } // operator+

            template<class Base>
            Data<Base> operator+(Data<Base>&& a) {
                  return Data<Base>(std::move(a));
            } // operator+

            template<class Base>
            Data<Base>& operator+=(Data<Base>& a, const Data<Base>& b) {
                  if (b.size == 1) {
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(a, a, 1, *b.base.data(), 0, 1);
#else
                        for (Size i = 0; i < a.size; i++) {
                              a.base[i] += b.base[0];
                        }
#endif
                  } else {
#ifdef TAT_USE_VML
                        scalar::Add<Base>(a, b, a);
#else
                        for (Size i = 0; i < a.size; i++) {
                              a.base[i] += b.base[i];
                        }
#endif
                  } // if
                  return a;
            } // operator+=

            template<class Base>
            Data<Base> operator+(const Data<Base>& a, const Data<Base>& b) {
                  if (a.size == 1) {
                        auto res = Data<Base>(b.size);
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(b, res, 1, *a.base.data(), 0, 1);
#else
                        for (Size i = 0; i < b.size; i++) {
                              res.base[i] = a.base[0] + b.base[i];
                        }
#endif
                        return res;
                  } // if
                  if (b.size == 1) {
                        Data<Base> res(a.size);
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(a, res, 1, *b.base.data(), 0, 1);
#else
                        for (Size i = 0; i < a.size; i++) {
                              res.base[i] = a.base[i] + b.base[0];
                        }
#endif
                        return res;
                  } // if
                  Data<Base> res(a.size);
#ifdef TAT_USE_VML
                  scalar::Add<Base>(a, b, res);
#else
                  for (Size i = 0; i < a.size; i++) {
                        res.base[i] = a.base[i] + b.base[i];
                  }
#endif
                  return res;
            } // operator+

            template<class Base>
            Data<Base> operator-(const Data<Base>& a) {
                  auto res = Data<Base>(a.size);
#ifdef TAT_USE_VML
                  scalar::LinearFrac<Base>(a, res, -1, 0, 0, 1);
#else
                  for (Size i = 0; i < a.size; i++) {
                        res.base[i] = -a.base[i];
                  }
#endif
                  return res;
            } // operator-

            template<class Base>
            Data<Base>& operator-=(Data<Base>& a, const Data<Base>& b) {
                  if (b.size == 1) {
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(a, a, 1, -*b.base.data(), 0, 1);
#else
                        for (Size i = 0; i < a.size; i++) {
                              a.base[i] -= b.base[0];
                        }
#endif
                  } else {
#ifdef TAT_USE_VML
                        scalar::Sub<Base>(a, b, a);
#else
                        for (Size i = 0; i < a.size; i++) {
                              a.base[i] -= b.base[i];
                        }
#endif
                  } // if
                  return a;
            } // operator-=

            template<class Base>
            Data<Base> operator-(const Data<Base>& a, const Data<Base>& b) {
                  if (a.size == 1) {
                        auto res = Data<Base>(b.size);
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(b, res, -1, *a.base.data(), 0, 1);
#else
                        for (Size i = 0; i < b.size; i++) {
                              res.base[i] = a.base[0] - b.base[i];
                        }
#endif
                        return res;
                  } // if
                  if (b.size == 1) {
                        auto res = Data<Base>(a.size);
#ifdef TAT_USE_VML
                        scalar::LinearFrac<Base>(a, res, 1, -*b.base.data(), 0, 1);
#else
                        for (Size i = 0; i < a.size; i++) {
                              res.base[i] = a.base[i] - b.base[0];
                        }
#endif
                        return res;
                  } // if
                  auto res = Data<Base>(a.size);
#ifdef TAT_USE_VML
                  scalar::Sub<Base>(a, b, res);
#else
                  for (Size i = 0; i < a.size; i++) {
                        res.base[i] = a.base[i] - b.base[i];
                  }
#endif
                  return res;
            } // operator-
      } // namespace data
      //
      //       SSS    CCC     AA    L        AA    RRRR          BBBBB   L       OOO    CCC   K    K
      //      S   S  C   C   A  A   L       A  A   R   R          B   B  L      O   O  C   C  K   K
      //      S      C      A    A  L      A    A  R   R          B   B  L      O   O  C      K  K
      //      S      C      A    A  L      A    A  R   R          B   B  L      O   O  C      K K
      //       SSS   C      A    A  L      A    A  RRRR           BBBB   L      O   O  C      KK
      //          S  C      AAAAAA  L      AAAAAA  RR             B   B  L      O   O  C      K K
      //          S  C      A    A  L      A    A  R R            B   B  L      O   O  C      K  K
      //      S   S  C   C  A    A  L      A    A  R  R           B   B  L      O   O  C   C  K   K
      //       SSS    CCC   A    A  LLLLL  A    A  R   R  _____  BBBBB   LLLLL   OOO    CCC   K    K
      //
      namespace block {
            bool operator==(const std::vector<Size>& a, const std::vector<Size>& b) {
                  if (a.size() != b.size()) {
                        return false;
                  } // if size
                  Rank size = a.size();
                  for (Rank i = 0; i < size; i++) {
                        if (a[i] != b[i]) {
                              return false;
                        } // if
                  } // for i
                  return true;
            } // operator==

#define DEF_OP(OP)                                            \
      template<class Base>                                    \
      Block<Base>& OP(Block<Base>& a, const Block<Base>& b) { \
            assert(b.dims.size() == 0 || a.dims == b.dims);   \
            data::OP(a.data, b.data);                         \
            return a;                                         \
      }

            DEF_OP(operator*=)
            DEF_OP(operator/=)
            DEF_OP(operator+=)
            DEF_OP(operator-=)
#undef DEF_OP

#define DEF_OP(OP)                                                 \
      template<class Base>                                         \
      Block<Base> OP(const Block<Base>& a, const Block<Base>& b) { \
            Block<Base> res;                                       \
            if (b.dims.size() == 0) {                              \
                  res.dims = a.dims;                               \
            } else if (a.dims.size() == 0) {                       \
                  res.dims = b.dims;                               \
            } else {                                               \
                  res.dims = a.dims;                               \
                  assert(a.dims == b.dims);                        \
            }                                                      \
            res.data = data::OP(a.data, b.data);                   \
            return res;                                            \
      }

            DEF_OP(operator*)
            DEF_OP(operator/)
            DEF_OP(operator+)
            DEF_OP(operator-)
#undef DEF_OP

            template<class Base>
            const Block<Base>& operator+(const Block<Base>& a) {
                  return a;
            } // operator+

            template<class Base>
            Block<Base> operator+(Block<Base>&& a) {
                  Block<Base> res;
                  res.dims = std::move(a.dims);
                  res.data = +std::move(a.data);
                  return res;
            } // operator+

            template<class Base>
            Block<Base> operator-(const Block<Base>& a) {
                  Block<Base> res;
                  res.dims = a.dims;
                  res.data = -a.data;
                  return res;
            } // operator-
      } // namespace block
      //
      //       SSS    CCC     AA    L        AA    RRRR          N    N   OOO   DDDDD   EEEEE
      //      S   S  C   C   A  A   L       A  A   R   R         N    N  O   O   D   D  E
      //      S      C      A    A  L      A    A  R   R         NN   N  O   O   D   D  E
      //      S      C      A    A  L      A    A  R   R         N N  N  O   O   D   D  E
      //       SSS   C      A    A  L      A    A  RRRR          N  N N  O   O   D   D  EEEE
      //          S  C      AAAAAA  L      AAAAAA  RR            N   NN  O   O   D   D  E
      //          S  C      A    A  L      A    A  R R           N    N  O   O   D   D  E
      //      S   S  C   C  A    A  L      A    A  R  R          N    N  O   O   D   D  E
      //       SSS    CCC   A    A  LLLLL  A    A  R   R  _____  N    N   OOO   DDDDD   EEEEE
      //
      namespace node {
            bool operator==(const std::vector<Legs>& a, const std::vector<Legs>& b) {
                  if (a.size() != b.size()) {
                        return false;
                  } // if size
                  Rank size = a.size();
                  for (Rank i = 0; i < size; i++) {
                        if (a[i] != b[i]) {
                              return false;
                        } // if i
                  } // for
                  return true;
            } // operator==

#define DEF_OP(OP)                                          \
      template<class Base>                                  \
      Node<Base>& OP(Node<Base>& a, const Node<Base>& b) {  \
            assert(b.legs.size() == 0 || a.legs == b.legs); \
            tensor::OP(a.tensor, b.tensor);                 \
            return a;                                       \
      }                                                     \
      template<class Base, class B>                         \
      Node<Base>& OP(Node<Base>& a, const B& b) {           \
            return OP(a, Node<Base>(b));                    \
      }

            DEF_OP(operator*=)
            DEF_OP(operator/=)
            DEF_OP(operator+=)
            DEF_OP(operator-=)
#undef DEF_OP

#define DEF_OP(OP)                                              \
      template<class Base>                                      \
      Node<Base> OP(const Node<Base>& a, const Node<Base>& b) { \
            Node<Base> res;                                     \
            if (b.legs.size() == 0) {                           \
                  res.legs = a.legs;                            \
            } else if (a.legs.size() == 0) {                    \
                  res.legs = b.legs;                            \
            } else {                                            \
                  res.legs = a.legs;                            \
                  assert(a.legs == b.legs);                     \
            }                                                   \
            res.tensor = tensor::OP(a.tensor, b.tensor);        \
            return res;                                         \
      }                                                         \
      template<class Base, class B>                             \
      Node<Base> OP(const Node<Base>& a, const B& b) {          \
            return OP(a, Node<Base>(b));                        \
      }                                                         \
      template<class Base, class B>                             \
      Node<Base> OP(const B& b, const Node<Base>& a) {          \
            return OP(Node<Base>(b), a);                        \
      }

            DEF_OP(operator*)
            DEF_OP(operator/)
            DEF_OP(operator+)
            DEF_OP(operator-)
#undef DEF_OP

            template<class Base>
            const Node<Base>& operator+(const Node<Base>& a) {
                  return a;
            } // operator+

            template<class Base>
            Node<Base> operator+(Node<Base>&& a) {
                  Node<Base> res;
                  res.legs = std::move(a.legs);
                  res.tensor = +std::move(a.tensor);
                  return res;
            } // operator+

            template<class Base>
            Node<Base> operator-(const Node<Base>& a) {
                  Node<Base> res;
                  res.legs = a.legs;
                  res.tensor = -a.tensor;
                  return res;
            } // operator-
      } // namespace node
      //
      //      TTTTT  RRRR     AA    N    N   SSS   PPPP    OOO    SSS   EEEEE
      //        T    R   R   A  A   N    N  S   S  P   P  O   O  S   S  E
      //        T    R   R  A    A  NN   N  S      P   P  O   O  S      E
      //        T    R   R  A    A  N N  N  S      P   P  O   O  S      E
      //        T    RRRR   A    A  N  N N   SSS   PPPP   O   O   SSS   EEEE
      //        T    RR     AAAAAA  N   NN      S  P      O   O      S  E
      //        T    R R    A    A  N    N      S  P      O   O      S  E
      //        T    R  R   A    A  N    N  S   S  P      O   O  S   S  E
      //        T    R   R  A    A  N    N   SSS   P       OOO    SSS   EEEEE
      //
      namespace data {
            namespace transpose {
                  template<class Base>
                  void run(const std::vector<Rank>& plan, const std::vector<Size>& dims, const Base* src, Base* dst) {
                        // currently use hptt only
                        std::vector<int> int_plan(plan.begin(), plan.end());
                        std::vector<int> int_dims(dims.begin(), dims.end());
                        hptt::create_plan(
                              int_plan.data(),
                              int_plan.size(),
                              1,
                              src,
                              int_dims.data(),
                              NULL,
                              0,
                              dst,
                              NULL,
                              hptt::ESTIMATE,
                              1,
                              NULL,
                              1)
                              ->execute();
                  } // run
            } // namespace transpose

            template<class Base>
            Data<Base> Data<Base>::transpose(const std::vector<Size>& dims, const std::vector<Rank>& plan) const {
                  assert(dims.size() == plan.size());
                  Data<Base> res(size);
                  transpose::run(plan, dims, base.data(), res.base.data());
                  return res;
            } // transpose
      } // namespace data

      namespace block {
            namespace transpose {
                  std::vector<Size> get_new_dims(const std::vector<Size>& dims, const std::vector<Rank>& plan) {
                        std::vector<Size> new_dims;
                        for (const auto& i : plan) {
                              new_dims.push_back(dims[i]);
                        } // for i
                        return new_dims;
                  } // plan
            } // namespace transpose

            template<class Base>
            Block<Base> Block<Base>::transpose(const std::vector<Rank>& plan) const {
                  Block<Base> res;
                  res.dims = transpose::get_new_dims(dims, plan);
                  res.data = data.transpose(dims, plan);
                  return res;
            } // transpose
      } // namespace block

      namespace node {
            namespace transpose {
                  std::vector<Rank> generate_plan(const std::vector<Legs>& new_legs, const std::vector<Legs>& legs) {
                        std::vector<Rank> res;
                        for (const auto& i : new_legs) {
                              res.push_back(std::distance(legs.begin(), vector_legs::find_iter(legs, i)));
                        } // for i
                        return res;
                  } // plan
            } // namespace transpose

            template<class Base>
            Node<Base> Node<Base>::transpose(const std::vector<Legs>& new_legs) const {
                  Node<Base> res;
                  res.legs = vector_legs::filter_out_not_in(new_legs, legs);
                  assert(vector_legs::no_duplicated(res.legs));
                  assert(res.legs.size() == legs.size());
                  std::vector<Rank> plan = transpose::generate_plan(res.legs, legs);
                  assert(plan.size() == legs.size());
                  assert(res.legs.size() == legs.size());
                  res.tensor = tensor.transpose(plan);
                  return res;
            } // transpose
      } // namespace node
      //
      //       SSS   V   V  DDDDD
      //      S   S  V   V   D   D
      //      S      V   V   D   D
      //      S       V V    D   D
      //       SSS    V V    D   D
      //          S   V V    D   D
      //          S    V     D   D
      //      S   S    V     D   D
      //       SSS     V    DDDDD
      //
      namespace data {
            namespace svd {
#if (defined TAT_USE_GESVD) || (defined TAT_USE_GESDD)
                  template<class Base>
                  void
                  run(const Size& m,
                      const Size& n,
                      const Size& min,
                      Base* a,
                      Base* u,
                      scalar_tools::real_base_t<Base>* s,
                      Base* vt);

                  template<>
                  void
                  run<float>(const Size& m, const Size& n, const Size& min, float* a, float* u, float* s, float* vt) {
#      ifdef TAT_USE_GESDD
#            ifndef NDEBUG
                        auto res =
#            endif // NDEBUG
                              LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S', m, n, a, n, s, u, min, vt, n);
#      endif // TAT_USE_GESDD
#      ifdef TAT_USE_GESVD
                        auto superb = std::vector<float>(min - 1);
#            ifndef NDEBUG
                        auto res =
#            endif // NDEBUG
                              LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, a, n, s, u, min, vt, n, superb.data());
#      endif // TAT_USE_GESVD
                        assert(res == 0);
                  } // run<float>

                  template<>
                  void run<double>(
                        const Size& m,
                        const Size& n,
                        const Size& min,
                        double* a,
                        double* u,
                        double* s,
                        double* vt) {
#      ifdef TAT_USE_GESDD
#            ifndef NDEBUG
                        auto res =
#            endif // NDEBUG
                              LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', m, n, a, n, s, u, min, vt, n);
#      endif // TAT_USE_GESDD
#      ifdef TAT_USE_GESVD
                        auto superb = std::vector<double>(min - 1);
#            ifndef NDEBUG
                        auto res =
#            endif // NDEBUG
                              LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, a, n, s, u, min, vt, n, superb.data());
#      endif // TAT_USE_GESVD
                        assert(res == 0);
                  } // run<double>

                  template<>
                  void run<std::complex<float>>(
                        const Size& m,
                        const Size& n,
                        const Size& min,
                        std::complex<float>* a,
                        std::complex<float>* u,
                        float* s,
                        std::complex<float>* vt) {
#      ifdef TAT_USE_GESDD
#            ifndef NDEBUG
                        auto res =
#            endif // NDEBUG
                              LAPACKE_cgesdd(LAPACK_ROW_MAJOR, 'S', m, n, a, n, s, u, min, vt, n);
#      endif // TAT_USE_GESDD
#      ifdef TAT_USE_GESVD
                        auto superb = std::vector<float>(min - 1);
#            ifndef NDEBUG
                        auto res =
#            endif // NDEBUG
                              LAPACKE_cgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, a, n, s, u, min, vt, n, superb.data());
#      endif // TAT_USE_GESVD
                        assert(res == 0);
                  } // run<std::complex<float>>

                  template<>
                  void run<std::complex<double>>(
                        const Size& m,
                        const Size& n,
                        const Size& min,
                        std::complex<double>* a,
                        std::complex<double>* u,
                        double* s,
                        std::complex<double>* vt) {
#      ifdef TAT_USE_GESDD
#            ifndef NDEBUG
                        auto res =
#            endif // NDEBUG
                              LAPACKE_zgesdd(LAPACK_ROW_MAJOR, 'S', m, n, a, n, s, u, min, vt, n);
#      endif // TAT_USE_GESDD
#      ifdef TAT_USE_GESVD
                        auto superb = std::vector<double>(min - 1);
#            ifndef NDEBUG
                        auto res =
#            endif // NDEBUG
                              LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, a, n, s, u, min, vt, n, superb.data());
#      endif // TAT_USE_GESVD
                        assert(res == 0);
                  } // run<std::complex<double>>

                  template<class Base>
                  Data<Base> cut(Data<Base>&& other, const Size& m1, const Size& n1, const Size& m2, const Size& n2) {
                        (void)m1; // avoid warning of unused when NDEBUG
                        assert(n2 <= n1);
                        assert(m2 <= m1);
                        if (n2 == n1) {
                              other.size = m2 * n2;
                              return std::move(other);
                        }
                        Data<Base> res(m2 * n2);
                        Base* dst = res.base.data();
                        const Base* src = other.base.data();
                        Size size = n2 * sizeof(Base);
                        for (Size i = 0; i < m2; i++) {
                              std::memcpy(dst, src, size);
                              dst += n2;
                              src += n1;
                        } // for i
                        return res;
                  } // cut
#endif // TAT_USE_GESVD TAT_USE_GESDD

#ifdef TAT_USE_GESVDX
                  template<class Base>
                  void
                  run(const Size& m,
                      const Size& n,
                      const Size& min,
                      const Size& cut,
                      Base* a,
                      Base* u,
                      real_base_t<Base>* s,
                      Base* vt);

                  template<>
                  void run<float>(
                        const Size& m,
                        const Size& n,
                        const Size& min,
                        const Size& cut,
                        float* a,
                        float* u,
                        float* s,
                        float* vt) {
                        lapack_int ns;
                        auto superb = std::vector<lapack_int>(12 * min);
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_sgesvdx(
                                    LAPACK_ROW_MAJOR,
                                    'V',
                                    'V',
                                    'I',
                                    m,
                                    n,
                                    a,
                                    n,
                                    0,
                                    0,
                                    1,
                                    cut,
                                    &ns,
                                    s,
                                    u,
                                    cut,
                                    vt,
                                    n,
                                    superb.data());
                        assert(res == 0);
                        assert(ns == lapack_int(cut));
                  } // run<float>

                  template<>
                  void run<double>(
                        const Size& m,
                        const Size& n,
                        const Size& min,
                        const Size& cut,
                        double* a,
                        double* u,
                        double* s,
                        double* vt) {
                        lapack_int ns;
                        auto superb = std::vector<lapack_int>(12 * min);
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_dgesvdx(
                                    LAPACK_ROW_MAJOR,
                                    'V',
                                    'V',
                                    'I',
                                    m,
                                    n,
                                    a,
                                    n,
                                    0,
                                    0,
                                    1,
                                    cut,
                                    &ns,
                                    s,
                                    u,
                                    cut,
                                    vt,
                                    n,
                                    superb.data());
                        assert(res == 0);
                        assert(ns == lapack_int(cut));
                  } // run<double>

                  template<>
                  void run<std::complex<float>>(
                        const Size& m,
                        const Size& n,
                        const Size& min,
                        const Size& cut,
                        std::complex<float>* a,
                        std::complex<float>* u,
                        float* s,
                        std::complex<float>* vt) {
                        lapack_int ns;
                        auto superb = std::vector<lapack_int>(12 * min);
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_cgesvdx(
                                    LAPACK_ROW_MAJOR,
                                    'V',
                                    'V',
                                    'I',
                                    m,
                                    n,
                                    a,
                                    n,
                                    0,
                                    0,
                                    1,
                                    cut,
                                    &ns,
                                    s,
                                    u,
                                    cut,
                                    vt,
                                    n,
                                    superb.data());
                        assert(res == 0);
                        assert(ns == lapack_int(cut));
                  } // run<std::complex<float>>

                  template<>
                  void run<std::complex<double>>(
                        const Size& m,
                        const Size& n,
                        const Size& min,
                        const Size& cut,
                        std::complex<double>* a,
                        std::complex<double>* u,
                        double* s,
                        std::complex<double>* vt) {
                        lapack_int ns;
                        auto superb = std::vector<lapack_int>(12 * min);
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_zgesvdx(
                                    LAPACK_ROW_MAJOR,
                                    'V',
                                    'V',
                                    'I',
                                    m,
                                    n,
                                    a,
                                    n,
                                    0,
                                    0,
                                    1,
                                    cut,
                                    &ns,
                                    s,
                                    u,
                                    cut,
                                    vt,
                                    n,
                                    superb.data());
                        assert(res == 0);
                        assert(ns == lapack_int(cut));
                  } // run<std::complex<double>>
#endif // TAT_USE_GESVDX
            } // namespace svd

            template<class Base>
            typename Data<Base>::svd_res Data<Base>::svd(
                  const std::vector<Size>& dims,
                  const std::vector<Rank>& plan,
                  const Size& u_size,
                  const Size& v_size,
                  const Size& min_mn,
                  const Size& cut) const {
                  assert(size % u_size == 0);
                  Size cut_dim = (cut < min_mn) ? cut : min_mn;
                  // -1 > any integer
                  Data<Base> tmp = transpose(dims, plan);
                  // used in svd, gesvd will destroy it
                  svd_res res;
#ifdef TAT_USE_GESVDX
                  res.U = Data<Base>(u_size * cut_dim);
                  res.S = Data<real_base_t<Base>>(min_mn);
                  res.S.size = cut_dim;
                  res.V = Data<Base>(cut_dim * v_size);
                  svd::run(u_size, v_size, min_mn, cut_dim, tmp.get(), res.U.get(), res.S.get(), res.V.get());
#endif // TAT_USE_GESVDX
#if (defined TAT_USE_GESVD) || (defined TAT_USE_GESDD)
                  res.U = Data<Base>(u_size * min_mn);
                  res.S = Data<real_base_t<Base>>(min_mn);
                  res.V = Data<Base>(min_mn * v_size);
                  svd::run(
                        u_size,
                        v_size,
                        min_mn,
                        tmp.base.data(),
                        res.U.base.data(),
                        res.S.base.data(),
                        res.V.base.data());
                  if (cut_dim != min_mn) {
                        res.U = svd::cut(std::move(res.U), u_size, min_mn, u_size, cut_dim);
                        res.V = svd::cut(std::move(res.V), min_mn, v_size, cut_dim, v_size);
                        res.S = svd::cut(std::move(res.S), min_mn, 1, cut_dim, 1);
                  }
#endif // TAT_USE_GESVD TAT_USE_GESDD
                  return res;
            } // svd
      } // namespace data
      namespace block {
            namespace svd {
                  Size get_u_size(const Rank& u_rank, const std::vector<Size>& dims) {
                        Size u_size = 1;
                        for (Rank i = 0; i < u_rank; i++) {
                              u_size *= dims[i];
                        } // for i
                        return u_size;
                  } // plan
            } // namespace svd

            template<class Base>
            typename Block<Base>::svd_res
            Block<Base>::svd(const std::vector<Rank>& plan, const Rank& u_rank, const Size& cut) const {
                  svd_res res;
                  std::vector<Size> tmp_dims = transpose::get_new_dims(dims, plan);
                  Size u_size = svd::get_u_size(u_rank, tmp_dims);
                  Size v_size = size() / u_size;
                  Size min_mn = (u_size < v_size) ? u_size : v_size;
                  auto data_res = data.svd(dims, plan, u_size, v_size, min_mn, cut);
                  auto mid = tmp_dims.begin() + u_rank;
                  res.U.dims.insert(res.U.dims.end(), tmp_dims.begin(), mid);
                  res.U.dims.push_back(data_res.S.size);
                  res.S.dims.push_back(data_res.S.size);
                  res.V.dims.push_back(data_res.S.size);
                  res.V.dims.insert(res.V.dims.end(), mid, tmp_dims.end());
                  res.U.data = std::move(data_res.U);
                  res.S.data = std::move(data_res.S);
                  res.V.data = std::move(data_res.V);
                  return res;
            } // svd
      } // namespace block
      namespace node {
            // 因为生成U, V的legs时，可以顺便获得转置plan, 故传给block的参数为直接的plan
            namespace svd {
                  auto
                  plan(const std::vector<Legs>& total_legs,
                       const std::vector<Legs>& u_legs,
                       const Legs& new_u_legs,
                       const Legs& new_v_legs) {
                        std::vector<Legs> U_legs;
                        std::vector<Legs> V_legs;
                        std::vector<Legs> tmp_legs;
                        Rank u_rank = u_legs.size();
                        V_legs.push_back(new_v_legs);
                        for (const auto& i : total_legs) {
                              if (vector_legs::find_iter(u_legs, i) != u_legs.end()) {
                                    U_legs.push_back(i);
                              } else {
                                    V_legs.push_back(i);
                              } // if
                        } // for
                        // std::copy_if(
                        //       total_legs.begin(), total_legs.end(), std::back_inserter(U_legs), [&](const Legs& i) {
                        //       return vector_legs::find_iter(u_legs, i)!=u_legs.end();
                        // });
                        // std::copy_if(
                        //       total_legs.begin(), total_legs.end(), std::back_inserter(V_legs), [&](const Legs& i) {
                        //       return vector_legs::find_iter(u_legs, i)==u_legs.end();
                        // });
                        U_legs.push_back(new_u_legs);
                        tmp_legs.insert(tmp_legs.end(), U_legs.begin(), U_legs.end() - 1);
                        tmp_legs.insert(tmp_legs.end(), V_legs.begin() + 1, V_legs.end());
                        return std::tuple{U_legs, V_legs, tmp_legs, u_rank};
                  } // plan
            } // namespace svd

            template<class Base>
            typename Node<Base>::svd_res Node<Base>::svd(
                  const std::vector<Legs>& input_u_legs,
                  const Legs& new_u_legs,
                  const Legs& new_v_legs,
                  const Rank& cut) const {
                  std::vector<Legs> u_legs = vector_legs::filter_out_not_in(legs, input_u_legs);
                  svd_res res;
                  auto [U_legs, V_legs, tmp_legs, u_rank] = svd::plan(legs, u_legs, new_u_legs, new_v_legs);
                  res.U.legs = std::move(U_legs);
                  res.V.legs = std::move(V_legs);
                  std::vector<Rank> plan = transpose::generate_plan(tmp_legs, legs);
                  auto tensor_res = tensor.svd(plan, u_rank, cut);
                  res.S.legs = {new_u_legs}; // new_u_legs or new_v_legs
                  res.U.tensor = std::move(tensor_res.U);
                  res.S.tensor = std::move(tensor_res.S);
                  res.V.tensor = std::move(tensor_res.V);
                  return res;
            } // svd
      } // namespace node
      //
      //       QQQ    RRRR
      //      Q   Q   R   R
      //      Q   Q   R   R
      //      Q   Q   R   R
      //      Q   Q   RRRR
      //      Q   Q   RR
      //      Q Q Q   R R
      //      Q  QQ   R  R
      //       QQQQ   R   R
      //           Q
      //
      namespace data {
            namespace qr {
                  template<class Base>
                  void geqrf(Base* A, Base* tau, const Size& m, const Size& n);

                  template<>
                  void geqrf<float>(float* A, float* tau, const Size& m, const Size& n) {
#ifdef TAT_USE_GEQP3
                        auto jpvt = std::vector<lapack_int>(n);
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_sgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt.data(), tau);
#endif // TAT_USE_GEQP3
#ifdef TAT_USE_GEQRF
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
#endif // TAT_USE_GEQRF
                        assert(res == 0);
                  } // geqrf<float>

                  template<>
                  void geqrf<double>(double* A, double* tau, const Size& m, const Size& n) {
#ifdef TAT_USE_GEQP3
                        auto jpvt = std::vector<lapack_int>(n);
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt.data(), tau);
#endif // TAT_USE_GEQP3
#ifdef TAT_USE_GEQRF
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
#endif // TAT_USE_GEQRF
                        assert(res == 0);
                  } // geqrf<double>

                  template<>
                  void geqrf<std::complex<float>>(
                        std::complex<float>* A,
                        std::complex<float>* tau,
                        const Size& m,
                        const Size& n) {
#ifdef TAT_USE_GEQP3
                        auto jpvt = std::vector<lapack_int>(n);
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_cgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt.data(), tau);
#endif // TAT_USE_GEQP3
#ifdef TAT_USE_GEQRF
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
#endif // TAT_USE_GEQRF
                        assert(res == 0);
                  } // geqrf<std::complex<float>>

                  template<>
                  void geqrf<std::complex<double>>(
                        std::complex<double>* A,
                        std::complex<double>* tau,
                        const Size& m,
                        const Size& n) {
#ifdef TAT_USE_GEQP3
                        auto jpvt = std::vector<lapack_int>(n);
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_zgeqp3(LAPACK_ROW_MAJOR, m, n, A, n, jpvt.data(), tau);
#endif // TAT_USE_GEQP3
#ifdef TAT_USE_GEQRF
#      ifndef NDEBUG
                        auto res =
#      endif // NDEBUG
                              LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, A, n, tau);
#endif // TAT_USE_GEQRF
                        assert(res == 0);
                  } // geqrf<std::complex<double>>

                  template<class Base>
                  void orgqr(Base* A, const Base* tau, const Size& m, const Size& min);

                  template<>
                  void orgqr<float>(float* A, const float* tau, const Size& m, const Size& min) {
#ifndef NDEBUG
                        auto res =
#endif // NDEBUG
                              LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, min, min, A, min, tau);
                        assert(res == 0);
                  } // orgqr<float>

                  template<>
                  void orgqr<double>(double* A, const double* tau, const Size& m, const Size& min) {
#ifndef NDEBUG
                        auto res =
#endif // NDEBUG
                              LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, min, min, A, min, tau);
                        assert(res == 0);
                  } // orgqr<double>

                  template<>
                  void orgqr<std::complex<float>>(
                        std::complex<float>* A,
                        const std::complex<float>* tau,
                        const Size& m,
                        const Size& min) {
#ifndef NDEBUG
                        auto res =
#endif // NDEBUG
                              LAPACKE_cungqr(LAPACK_ROW_MAJOR, m, min, min, A, min, tau);
                        assert(res == 0);
                  } // orgqr<std::complex<float>>

                  template<>
                  void orgqr<std::complex<double>>(
                        std::complex<double>* A,
                        const std::complex<double>* tau,
                        const Size& m,
                        const Size& min) {
#ifndef NDEBUG
                        auto res =
#endif // NDEBUG
                              LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, min, min, A, min, tau);
                        assert(res == 0);
                  } // orgqr<std::complex<double>>

                  template<class Base>
                  void run(Base* Q, Base* R, const Size& m, const Size& n, const Size& min_mn) {
                        auto tau = std::vector<Base>(min_mn);
                        geqrf(R, tau.data(), m, n);
                        // copy to Q and delete unused R
                        if (min_mn == n) {
                              std::memcpy(Q, R, m * n * sizeof(Base));
                        } else {
                              // Q is m*m
                              auto q = Q;
                              auto r = R;
                              for (Size i = 0; i < m; i++) {
                                    std::memcpy(q, r, m * sizeof(Base));
                                    q += m;
                                    r += n;
                              } // for i
                        } // if
                        orgqr(Q, tau.data(), m, min_mn);
                        auto r = R;
                        for (Size i = 1; i < min_mn; i++) {
                              r += n;
                              std::memset((void*)r, 0, i * sizeof(Base)); // avoid complex memset warning
                        } // for i
                  } // run
            } // namespace qr

            template<class Base>
            typename Data<Base>::qr_res Data<Base>::qr(
                  const std::vector<Size>& dims,
                  const std::vector<Rank>& plan,
                  const Size& q_size,
                  const Size& r_size,
                  const Size& min_mn) const {
                  assert(size == q_size * r_size);
                  qr_res res;
                  res.Q = Data<Base>(q_size * min_mn);
                  res.R = transpose(dims, plan);
                  // R is q_size*r_size, should be min_mn*r_size
                  // so if q_size > r_size, R will occupy some unused memory
                  qr::run(res.Q.base.data(), res.R.base.data(), q_size, r_size, min_mn);
                  res.R.size = min_mn * r_size;
                  return res;
            } // qr
      } // namespace data
      namespace block {
            template<class Base>
            typename Block<Base>::qr_res Block<Base>::qr(const std::vector<Rank>& plan, const Rank& q_rank) const {
                  qr_res res;
                  std::vector<Size> tmp_dims = transpose::get_new_dims(dims, plan);
                  Size q_size = svd::get_u_size(q_rank, tmp_dims);
                  auto mid = tmp_dims.begin() + q_rank;
                  Size r_size = data.size / q_size;
                  Size min_size = (q_size < r_size) ? q_size : r_size;
                  auto data_res = data.qr(dims, plan, q_size, r_size, min_size);
                  res.Q.dims.insert(res.Q.dims.end(), tmp_dims.begin(), mid);
                  res.Q.dims.push_back(min_size);
                  res.R.dims.push_back(min_size);
                  res.R.dims.insert(res.R.dims.end(), mid, tmp_dims.end());
                  res.Q.data = std::move(data_res.Q);
                  res.R.data = std::move(data_res.R);
                  return res;
            } // qr
      } // namespace block
      namespace node {
            template<class Base>
            typename Node<Base>::qr_res
            Node<Base>::qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs)
                  const {
                  std::vector<Legs> q_legs = vector_legs::filter_out_not_in(legs, input_q_legs);
                  qr_res res;
                  auto [Q_legs, R_legs, tmp_legs, q_rank] = svd::plan(legs, q_legs, new_q_legs, new_r_legs);
                  res.Q.legs = std::move(Q_legs);
                  res.R.legs = std::move(R_legs);
                  std::vector<Rank> plan = transpose::generate_plan(tmp_legs, legs);
                  auto tensor_res = tensor.qr(plan, q_rank);
                  res.Q.tensor = std::move(tensor_res.Q);
                  res.R.tensor = std::move(tensor_res.R);
                  return res;
            } // qr
      } // namespace node
      //
      //       CCC    OOO   N    N  TTTTT  RRRR     AA     CCC   TTTTT
      //      C   C  O   O  N    N    T    R   R   A  A   C   C    T
      //      C      O   O  NN   N    T    R   R  A    A  C        T
      //      C      O   O  N N  N    T    R   R  A    A  C        T
      //      C      O   O  N  N N    T    RRRR   A    A  C        T
      //      C      O   O  N   NN    T    RR     AAAAAA  C        T
      //      C      O   O  N    N    T    R R    A    A  C        T
      //      C   C  O   O  N    N    T    R  R   A    A  C   C    T
      //       CCC    OOO   N    N    T    R   R  A    A   CCC     T
      //
      namespace data {
            namespace contract {
                  template<class Base>
                  void
                  run(Base* data, const Base* data1, const Base* data2, const Size& m, const Size& n, const Size& k);

                  template<>
                  void run<float>(
                        float* data,
                        const float* data1,
                        const float* data2,
                        const Size& m,
                        const Size& n,
                        const Size& k) {
                        cblas_sgemm(
                              CblasRowMajor,
                              CblasNoTrans,
                              CblasNoTrans,
                              m,
                              n,
                              k,
                              1,
                              const_cast<float*>(data1),
                              k,
                              const_cast<float*>(data2),
                              n,
                              0,
                              data,
                              n);
                  } // run<float>

                  template<>
                  void run<double>(
                        double* data,
                        const double* data1,
                        const double* data2,
                        const Size& m,
                        const Size& n,
                        const Size& k) {
                        cblas_dgemm(
                              CblasRowMajor,
                              CblasNoTrans,
                              CblasNoTrans,
                              m,
                              n,
                              k,
                              1,
                              const_cast<double*>(data1),
                              k,
                              const_cast<double*>(data2),
                              n,
                              0,
                              data,
                              n);
                  } // run<double>

                  template<>
                  void run<std::complex<float>>(
                        std::complex<float>* data,
                        const std::complex<float>* data1,
                        const std::complex<float>* data2,
                        const Size& m,
                        const Size& n,
                        const Size& k) {
                        std::complex<float> alpha = 1;
                        std::complex<float> beta = 0;
                        cblas_cgemm(
                              CblasRowMajor,
                              CblasNoTrans,
                              CblasNoTrans,
                              m,
                              n,
                              k,
                              &alpha,
                              const_cast<std::complex<float>*>(data1),
                              k,
                              const_cast<std::complex<float>*>(data2),
                              n,
                              &beta,
                              data,
                              n);
                  } // run<std::complex<float>>

                  template<>
                  void run<std::complex<double>>(
                        std::complex<double>* data,
                        const std::complex<double>* data1,
                        const std::complex<double>* data2,
                        const Size& m,
                        const Size& n,
                        const Size& k) {
                        std::complex<double> alpha = 1;
                        std::complex<double> beta = 0;
                        cblas_zgemm(
                              CblasRowMajor,
                              CblasNoTrans,
                              CblasNoTrans,
                              m,
                              n,
                              k,
                              &alpha,
                              const_cast<std::complex<double>*>(data1),
                              k,
                              const_cast<std::complex<double>*>(data2),
                              n,
                              &beta,
                              data,
                              n);
                  } // run<std::complex<double>>
            } // namespace contract

            template<class Base>
            Data<Base> Data<Base>::contract(
                  const Data<Base>& data1,
                  const Data<Base>& data2,
                  const std::vector<Size>& dims1,
                  const std::vector<Size>& dims2,
                  const std::vector<Rank>& plan1,
                  const std::vector<Rank>& plan2,
                  const Size& m,
                  const Size& k,
                  const Size& n) {
                  assert(m * k == data1.size);
                  assert(k * n == data2.size);
                  Data<Base> a = data1.transpose(dims1, plan1);
                  Data<Base> b = data2.transpose(dims2, plan2);
                  // wasted transpose
                  Data<Base> res(m * n);
                  contract::run<Base>(res.base.data(), a.base.data(), b.base.data(), m, n, k);
                  return res;
            } // contract
      } // namespace data
      namespace block {
            namespace contract {
                  auto
                  plan(const std::vector<Size>& dims1,
                       const std::vector<Size>& dims2,
                       const std::vector<Rank>& plan1,
                       const std::vector<Rank>& plan2,
                       const Rank& contract_num) {
                        std::vector<Size> dims;
                        Size m = 1;
                        Size k = 1;
                        Size n = 1;
                        Rank i;
                        Rank tmp = dims1.size() - contract_num;
                        Rank rank2 = dims2.size();
                        for (i = 0; i < tmp; i++) {
                              const Size& t = dims1[plan1[i]];
                              m *= t;
                              dims.push_back(t);
                        } // for i
                        for (i = 0; i < contract_num; i++) {
                              k *= dims1[plan1[i + tmp]];
                              assert(dims1[plan1[i + tmp]] == dims2[plan2[i]]);
                        } // for i
                        for (; i < rank2; i++) {
                              const Size& t = dims2[plan2[i]];
                              n *= t;
                              dims.push_back(t);
                        } // for i
                        return std::tuple{dims, m, k, n};
                  } // plan
            } // namespace contract

            template<class Base>
            Block<Base> Block<Base>::contract(
                  const Block<Base>& block1,
                  const Block<Base>& block2,
                  const std::vector<Rank>& plan1,
                  const std::vector<Rank>& plan2,
                  const Rank& contract_num) {
                  Block<Base> res;
                  auto [dims, m, k, n] = contract::plan(block1.dims, block2.dims, plan1, plan2, contract_num);
                  res.dims = std::move(dims);
                  res.data =
                        Data<Base>::contract(block1.data, block2.data, block1.dims, block2.dims, plan1, plan2, m, k, n);
                  return res;
            } // contract
      } // namespace block
      namespace node {
            namespace contract {
                  auto
                  plan(const std::vector<Legs>& total_legs1,
                       const std::vector<Legs>& total_legs2,
                       const std::vector<Legs>& legs1,
                       const std::vector<Legs>& legs2,
                       const std::vector<std::tuple<Legs, Legs>>& map1,
                       const std::vector<std::tuple<Legs, Legs>>& map2) {
                        std::vector<Legs> legs;
                        std::vector<Legs> new_legs1;
                        std::vector<Legs> new_legs2;

                        auto filt_legs1 = vector_legs::filter_out_in(total_legs1, legs1);
                        vector_legs::append(new_legs1, filt_legs1);
                        vector_legs::replace(filt_legs1, map1);
                        vector_legs::append(legs, filt_legs1);

                        auto tmp_legs1 = vector_legs::filter_out_not_in(legs1, total_legs1);
                        vector_legs::append(new_legs1, tmp_legs1);

                        auto tmp_legs2 = vector_legs::filter_out_not_in(legs2, total_legs2);
                        vector_legs::append(new_legs2, tmp_legs2);

                        auto filt_legs2 = vector_legs::filter_out_in(total_legs2, legs2);
                        vector_legs::append(new_legs2, filt_legs2);
                        vector_legs::replace(filt_legs2, map2);
                        vector_legs::append(legs, filt_legs2);

                        assert(tmp_legs1.size() == tmp_legs2.size());
                        Rank contract_num = tmp_legs1.size();
                        return std::tuple{contract_num, legs, new_legs1, new_legs2};
                  } // plan
            } // namespace contract

            template<class Base>
            Node<Base> Node<Base>::contract(
                  const Node<Base>& node1,
                  const Node<Base>& node2,
                  const std::vector<Legs>& legs1,
                  const std::vector<Legs>& legs2,
                  const std::vector<std::tuple<Legs, Legs>>& map1,
                  const std::vector<std::tuple<Legs, Legs>>& map2) {
                  Node<Base> res;
                  assert(legs1.size() == legs2.size());
                  auto [contract_num, res_legs, new_legs1, new_legs2] =
                        contract::plan(node1.legs, node2.legs, legs1, legs2, map1, map2);
                  res.legs = std::move(res_legs);
                  auto plan1 = transpose::generate_plan(new_legs1, node1.legs);
                  auto plan2 = transpose::generate_plan(new_legs2, node2.legs);
                  assert(new_legs1.size() == node1.legs.size());
                  assert(plan1.size() == node1.legs.size());
                  assert(new_legs2.size() == node2.legs.size());
                  assert(plan2.size() == node2.legs.size());
                  res.tensor = Tensor<Base>::contract(node1.tensor, node2.tensor, plan1, plan2, contract_num);
                  return res;
            } // contract
      } // namespace node
      //
      //      M     M  U   U  L      TTTTT  III  PPPP   L      EEEEE
      //      MM   MM  U   U  L        T     I   P   P  L      E
      //      M M M M  U   U  L        T     I   P   P  L      E
      //      M  M  M  U   U  L        T     I   P   P  L      E
      //      M     M  U   U  L        T     I   PPPP   L      EEEE
      //      M     M  U   U  L        T     I   P      L      E
      //      M     M  U   U  L        T     I   P      L      E
      //      M     M  U   U  L        T     I   P      L      E
      //      M     M   UUU   LLLLL    T    III  P      LLLLL  EEEEE
      //
      namespace data {
            namespace multiple {
                  template<class Base>
                  void
                  run(Base* res_data,
                      const Base* src_data,
                      const Base* other_data,
                      const Size& a,
                      const Size& b,
                      const Size& c) {
                        for (Size i = 0; i < a; i++) {
                              for (Size j = 0; j < b; j++) {
                                    Base v = other_data[j];
                                    for (Size k = 0; k < c; k++) {
                                          *(res_data++) = *(src_data++) * v;
                                    } // for k
                              } // for j
                        } // for i
                  } // run
            } // namespace multiple

            template<class Base>
            Data<Base>
            Data<Base>::multiple(const Data<Base>& other, const Size& a, const Size& b, const Size& c) const {
                  Data<Base> res(size);
                  assert(b == other.size);
                  assert(a * b * c == size);
                  multiple::run<Base>(res.base.data(), base.data(), other.base.data(), a, b, c);
                  return res;
            } // multiple
      } // namespace data
      namespace block {
            namespace multiple {
                  auto plan(const std::vector<Size>& dims, const Rank& index) {
                        Size a = 1, b = 1, c = 1;
                        Rank i = 0, rank = dims.size();
                        for (; i < index; i++) {
                              a *= dims[i];
                        } // for i
                        b = dims[i];
                        i++;
                        for (; i < rank; i++) {
                              c *= dims[i];
                        } // for
                        return std::tuple{a, b, c};
                  } // plan
            } // namespace multiple

            template<class Base>
            Block<Base> Block<Base>::multiple(const Block<Base>& other, const Rank& index) const {
                  Block<Base> res;
                  res.dims = dims;
                  auto [a, b, c] = multiple::plan(dims, index);
                  assert(other.dims.size() == 1);
                  assert(b == other.dims[0]);
                  res.data = data.multiple(other.data, a, b, c);
                  return res;
            } // multiple
      } // namespace block
      namespace node {
            template<class Base>
            Node<Base> Node<Base>::multiple(const Node<Base>& other, const Legs& position) const {
                  Node<Base> res;
                  assert(other.legs.size() == 1);
                  res.legs = legs;
                  auto pos = vector_legs::find_iter(legs, position);
                  if (pos == legs.end()) {
                        return *this;
                  } // if not multiple
                  Rank index = std::distance(legs.begin(), pos);
                  res.tensor = tensor.multiple(other.tensor, index);
                  return res;
            } // multiple
      } // namespace node

      //
      //      L        AA    ZZZZZ  Y   Y          CCC    OOO   RRRR   EEEEE
      //      L       A  A       Z  Y   Y         C   C  O   O  R   R  E
      //      L      A    A     Z    Y Y          C      O   O  R   R  E
      //      L      A    A     Z    Y Y          C      O   O  R   R  E
      //      L      A    A    Z      Y           C      O   O  RRRR   EEEE
      //      L      AAAAAA   Z       Y           C      O   O  RR     E
      //      L      A    A   Z       Y           C      O   O  R R    E
      //      L      A    A  Z        Y           C   C  O   O  R  R   E
      //      LLLLL  A    A  ZZZZZ    Y    _____   CCC    OOO   R   R  EEEEE
      //
      namespace lazy {
            // class LazyBase : public std::enable_shared_from_this<LazyBase> {
            class LazyCoreBase {
               public:
                  bool value_flag = false; // own valid value
                  bool func_flag = false; // initialized by function

                  std::set<LazyCoreBase*> downstream;
                  std::set<LazyCoreBase*> upstream; // this should be always valid because of func

                  void reset(bool reset_itself = true) {
                        if (value_flag) {
                              if (reset_itself) {
                                    value_flag = false;
                              }
                              for (const auto& ds : downstream) {
                                    ds->reset();
                              }
                        }
                  }

                  std::set<LazyCoreBase*> dump_upstream() {
                        for (const auto& us : upstream) {
                              us->downstream.erase(this); // if there is duplicated, it still work
                        }
                        return std::move(upstream);
                  }
            };

            template<class T>
            class LazyCore : public LazyCoreBase {
               public:
                  T value;
                  std::function<T()> func;

                  ~LazyCore() {
                        dump_upstream(); // this must be in derived class destructor
                  }

                  T& calc() {
                        if (!value_flag) {
                              value = func();
                              value_flag = true;
                        }
                        return value;
                  } // not mask as const because in qr, svd, it need move its member, namely change it

                  T pop() {
                        calc();
                        T res = std::move(value);
                        reset();
                        return res;
                  }

                  // set to a func_flaged lazy is dangerous
                  template<class... Args>
                  void set_value(Args&&... args) {
                        reset();
                        dump_upstream();
                        func_flag = false;
                        value = T(std::forward<Args>(args)...);
                        value_flag = true;
                  }

                  template<class Func, class... Args>
                  void set_func(Func&& f, std::shared_ptr<LazyCore<Args>>... args) {
                        reset();
                        dump_upstream();
                        (..., args->downstream.insert(this));
                        (..., upstream.insert(args.get()));
                        func = [=, f(std::move(f))]() { return f(args->calc()...); };
                        func_flag = true;
                  }

                  // update on a lazy with downstream is dangerous
                  template<class Func, class... Args>
                  void update(Func&& modify, std::shared_ptr<LazyCore<Args>>... args) {
                        (..., args->downstream.insert(this));
                        (..., upstream.insert(args.get()));
                        reset();
                        if (func_flag) {
                              func = [=, f(std::move(func)), m(std::move(modify))]() {
                                    auto current = f();
                                    m(current, args->calc()...);
                                    return current;
                              };
                        } else {
                              if constexpr (sizeof...(Args) == 0) {
                                    modify(value);
                                    value_flag = true;
                              } else {
                                    func = [=, current(std::move(func)), m(std::move(modify))]() {
                                          auto res = current; // copy happended here;
                                          m(res, args->calc()...);
                                          return res;
                                    };
                                    func_flag = true;
                              }
                        }
                  }

                  void replace(std::shared_ptr<LazyCore<T>> src) {
                        set_func([](const T& res) { return res; }, src);
                  } // copy happen
            };
      } // namespace lazy

      //
      //      L        AA    ZZZZZ  Y   Y
      //      L       A  A       Z  Y   Y
      //      L      A    A     Z    Y Y
      //      L      A    A     Z    Y Y
      //      L      A    A    Z      Y
      //      L      AAAAAA   Z       Y
      //      L      A    A   Z       Y
      //      L      A    A  Z        Y
      //      LLLLL  A    A  ZZZZZ    Y
      //
      namespace lazy {
            class LazyBase {};

            template<class T>
            class Lazy : LazyBase {
               public:
                  std::shared_ptr<LazyCore<T>> core;

                  /**
                   * init invalid state
                   */
                  Lazy() : core(std::make_shared<LazyCore<T>>()) {}

                  /**
                   * set value, similar to constructor
                   */
                  template<class... Args>
                  Lazy<T> set_value(Args&&... args) {
                        core->set_value(std::forward<Args>(args)...);
                        return *this;
                  }

                  /**
                   * set with a function, like make_lazy
                   */
                  template<class Func, class... Args>
                  Lazy<T> set_func(Func&& f, Lazy<Args>... args) {
                        core->set_func(std::move(f), args.core...);
                        return *this;
                  }

                  /**
                   * inplace op
                   */
                  template<class Func, class... Args>
                  Lazy<T> update(Func&& f, Lazy<Args>... args) {
                        core->update(std::move(f), args.core...);
                        return *this;
                  }

                  Lazy<T> replace(Lazy<T> src) {
                        core->replace(src.core);
                        return *this;
                  }

                  /**
                   * get its value
                   */
                  const T& value() const {
                        return core->calc();
                  }

                  T pop() const {
                        return core->pop();
                  }

                  explicit Lazy(std::shared_ptr<LazyCore<T>> c) : core(c) {}
                  template<class C, class = std::enable_if_t<std::is_base_of_v<Lazy<T>, C>>>
                  explicit Lazy(C c) : core(c.core) {}
                  explicit Lazy(T c) : core(std::make_shared<LazyCore<T>>()) {
                        set_value(c);
                  }
            };

            template<class T>
            Lazy(T)->Lazy<T>;

            template<class T>
            Lazy(std::shared_ptr<LazyCore<T>>)->Lazy<T>;

            template<class T>
            Lazy(Lazy<T>)->Lazy<T>;
      } // namespace lazy
      using lazy::Lazy;

      //
      //      L        AA    ZZZZZ  Y   Y          SSS    CCC     AA    L        AA    RRRR
      //      L       A  A       Z  Y   Y         S   S  C   C   A  A   L       A  A   R   R
      //      L      A    A     Z    Y Y          S      C      A    A  L      A    A  R   R
      //      L      A    A     Z    Y Y          S      C      A    A  L      A    A  R   R
      //      L      A    A    Z      Y            SSS   C      A    A  L      A    A  RRRR
      //      L      AAAAAA   Z       Y               S  C      AAAAAA  L      AAAAAA  RR
      //      L      A    A   Z       Y               S  C      A    A  L      A    A  R R
      //      L      A    A  Z        Y           S   S  C   C  A    A  L      A    A  R  R
      //      LLLLL  A    A  ZZZZZ    Y    _____   SSS    CCC   A    A  LLLLL  A    A  R   R
      //
      namespace lazy {
            template<class T>
            std::ostream& operator<<(std::ostream& out, const Lazy<T>& value) {
                  return out << value.value();
            }

            template<class T>
            auto operator+(const Lazy<T>& a) {
                  auto res = Lazy<T>();
                  res.set_func([](const T& a) { return +a; }, a);
                  return res;
            }

            template<class T>
            auto operator-(const Lazy<T>& a) {
                  auto res = Lazy<T>();
                  res.set_func([](const T& a) { return -a; }, a);
                  return res;
            }

#define DEF_OP(OP, EVAL)                                                                    \
      template<class A, class B>                                                            \
      auto OP(Lazy<A>& a, const Lazy<B>& b) {                                               \
            a.update([](A& a, const B& b) { EVAL; }, b);                                    \
            return a;                                                                       \
      }                                                                                     \
      template<class T, class B, class = std::enable_if_t<!std::is_base_of_v<LazyBase, B>>> \
      auto OP(Lazy<T>& a, const B& b) {                                                     \
            a.update([=](T& a) { EVAL; });                                                  \
            return a;                                                                       \
      }

            DEF_OP(operator*=, a *= b)
            DEF_OP(operator/=, a /= b)
            DEF_OP(operator+=, a += b)
            DEF_OP(operator-=, a -= b)
#undef DEF_OP

#define DEF_OP(OP, EVAL)                                                                    \
      template<class A, class B>                                                            \
      auto OP(const Lazy<A>& a, const Lazy<B>& b) {                                         \
            auto func = [](const A& a, const B& b) { return EVAL; };                        \
            auto res = Lazy<decltype(func(std::declval<A>(), std::declval<B>()))>();        \
            res.set_func(std::move(func), a, b);                                            \
            return res;                                                                     \
      }                                                                                     \
      template<class T, class B, class = std::enable_if_t<!std::is_base_of_v<LazyBase, B>>> \
      auto OP(const Lazy<T>& a, const B& b) {                                               \
            auto func = [=](const T& a) { return EVAL; };                                   \
            auto res = Lazy<decltype(func(std::declval<T>()))>();                           \
            res.set_func(std::move(func), a);                                               \
            return res;                                                                     \
      }                                                                                     \
      template<class T, class B, class = std::enable_if_t<!std::is_base_of_v<LazyBase, B>>> \
      auto OP(const B& a, const Lazy<T>& b) {                                               \
            auto func = [=](const T& b) { return EVAL; };                                   \
            auto res = Lazy<decltype(func(std::declval<T>()))>();                           \
            res.set_func(std::move(func), b);                                               \
            return res;                                                                     \
      }

            DEF_OP(operator*, a* b)
            DEF_OP(operator/, a / b)
            DEF_OP(operator+, a + b)
            DEF_OP(operator-, a - b)
#undef DEF_OP
      } // namespace lazy

      //
      //      L        AA    ZZZZZ  Y   Y         N    N   OOO   DDDDD   EEEEE
      //      L       A  A       Z  Y   Y         N    N  O   O   D   D  E
      //      L      A    A     Z    Y Y          NN   N  O   O   D   D  E
      //      L      A    A     Z    Y Y          N N  N  O   O   D   D  E
      //      L      A    A    Z      Y           N  N N  O   O   D   D  EEEE
      //      L      AAAAAA   Z       Y           N   NN  O   O   D   D  E
      //      L      A    A   Z       Y           N    N  O   O   D   D  E
      //      L      A    A  Z        Y           N    N  O   O   D   D  E
      //      LLLLL  A    A  ZZZZZ    Y    _____  N    N   OOO   DDDDD   EEEEE
      //
      namespace lazy_node {
            template<class Base = double>
            class LazyNode : public Lazy<Node<Base>> {
               public:
                  using Lazy<Node<Base>>::set_value;
                  using Lazy<Node<Base>>::set_func;
                  using Lazy<Node<Base>>::update;
                  using Lazy<Node<Base>>::replace;
                  using Lazy<Node<Base>>::value;
                  using Lazy<Node<Base>>::pop;

                  LazyNode() : Lazy<Node<Base>>() {}
                  template<class T1 = std::vector<Legs>, class T2 = std::vector<Size>>
                  explicit LazyNode(T1&& legs, T2&& dims) : Lazy<Node<Base>>() {
                        set_value(std::forward<T1>(legs), std::forward<T2>(dims));
                  }
                  template<class Arg0, class... Args>
                  explicit LazyNode(Arg0&& arg0, Args&&... args) {
                        set_value(std::forward<Arg0>(arg0), std::forward<Args>(args)...);
                  }
                  // explicit LazyNode(std::shared_ptr<LazyCore<Node<Base>>> src) : Lazy<Node<Base>>(src) {}
                  explicit LazyNode(Lazy<Node<Base>> src) : Lazy<Node<Base>>(src) {}

                  // set op is valid only when lazy contain node
                  template<class Generator>
                  LazyNode<Base>& set(Generator&& setter) & {
                        update([setter(std::forward<Generator>(setter))](Node<Base>& node) { node.set(setter); });
                        return *this;
                  }

                  template<class Generator>
                  LazyNode<Base>&& set(Generator&& setter) && {
                        return std::move(set(std::forward<Generator>(setter)));
                  }

                  LazyNode<Base>& legs_rename(const std::vector<std::tuple<Legs, Legs>>& dict) & {
                        update([=](Node<Base>& node) { node.legs_rename(dict); });
                        return *this;
                  }

                  LazyNode<Base>&& legs_rename(const std::vector<std::tuple<Legs, Legs>>& dict) && {
                        return std::move(legs_rename(dict));
                  }

                  template<class Base2>
                  LazyNode<Base2> to() const {
                        if constexpr (std::is_same_v<Base, Base2>) {
                              return *this;
                        } else {
                              auto res = LazyNode<Base2>();
                              res.set_func([](const Node<Base>& node) { return node.template to<Base2>(); }, *this);
                              return res;
                        }
                  }

                  template<int n>
                  LazyNode<Base> norm() const {
                        auto res = LazyNode<Base>();
                        res.set_func([](const Node<Base>& node) { return node.template norm<n>(); }, *this);
                        return res;
                  }

                  LazyNode<Base> transpose(const std::vector<Legs>& new_legs) const {
                        auto res = LazyNode<Base>();
                        res.set_func([=](const Node<Base>& node) { return node.transpose(new_legs); }, *this);
                        return res;
                  }

                  class svd_res {
                     public:
                        LazyNode<Base> U;
                        LazyNode<real_base_t<Base>> S;
                        LazyNode<Base> V;
                  };

                  svd_res
                  svd(const std::vector<Legs>& input_u_legs,
                      const Legs& new_u_legs,
                      const Legs& new_v_legs,
                      const Rank& cut = -1) const {
                        auto tmp = Lazy<typename Node<Base>::svd_res>();
                        tmp.set_func(
                              [=](const Node<Base>& node) {
                                    return node.svd(input_u_legs, new_u_legs, new_v_legs, cut);
                              },
                              *this);
                        auto res = svd_res();
                        res.U.set_func([](typename Node<Base>::svd_res& r) { return std::move(r.U); }, tmp);
                        res.S.set_func([](typename Node<Base>::svd_res& r) { return std::move(r.S); }, tmp);
                        res.V.set_func([](typename Node<Base>::svd_res& r) { return std::move(r.V); }, tmp);
                        return res;
                  }

                  class qr_res {
                     public:
                        LazyNode<Base> Q;
                        LazyNode<Base> R;
                  }; // class qr_res

                  qr_res
                  qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const {
                        auto tmp = Lazy<typename Node<Base>::qr_res>();
                        tmp.set_func(
                              [=](const Node<Base>& node) { return node.qr(input_q_legs, new_q_legs, new_r_legs); },
                              *this);
                        auto res = qr_res();
                        res.Q.set_func([](typename Node<Base>::qr_res& r) { return std::move(r.Q); }, tmp);
                        res.R.set_func([](typename Node<Base>::qr_res& r) { return std::move(r.R); }, tmp);
                        return res;
                  }

                  qr_res
                  rq(const std::vector<Legs>& input_r_legs, const Legs& new_r_legs, const Legs& new_q_legs) const {
                        auto tmp = Lazy<typename Node<Base>::qr_res>();
                        tmp.set_func(
                              [=](const Node<Base>& node) {
                                    auto input_q_legs = vector_legs::filter_out_in(node.legs, input_r_legs);
                                    return node.qr(input_q_legs, new_q_legs, new_r_legs);
                              },
                              *this);
                        auto res = qr_res();
                        res.Q.set_func([](typename Node<Base>::qr_res& r) { return std::move(r.Q); }, tmp);
                        res.R.set_func([](typename Node<Base>::qr_res& r) { return std::move(r.R); }, tmp);
                        return res;
                  }

                  static LazyNode<Base> contract(
                        const LazyNode<Base>& node1,
                        const LazyNode<Base>& node2,
                        const std::vector<Legs>& legs1,
                        const std::vector<Legs>& legs2,
                        const std::vector<std::tuple<Legs, Legs>>& map1 = {},
                        const std::vector<std::tuple<Legs, Legs>>& map2 = {}) {
                        auto res = LazyNode<Base>();
                        res.set_func(
                              [=](const Node<Base>& node1, const Node<Base>& node2) {
                                    // using Lazy<Node<Base>>::update_inplace;
                                    return Node<Base>::contract(node1, node2, legs1, legs2, map1, map2);
                              },
                              node1,
                              node2);
                        return res;
                  }

                  LazyNode<Base> multiple(const LazyNode<Base>& other, const Legs& position) const {
                        auto res = LazyNode<Base>();
                        res.set_func(
                              [=](const Node<Base>& self, const Node<Base>& other) {
                                    return self.multiple(other, position);
                              },
                              *this,
                              other);
                        return res;
                  }
            };

            template<class T>
            std::ostream& operator<<(std::ostream& out, const LazyNode<T>& value) {
                  return out << value.value();
            }
      } // namespace lazy_node
      using lazy_node::LazyNode;

      //
      //      L      N    N          SSS    CCC     AA    L        AA    RRRR
      //      L      N    N         S   S  C   C   A  A   L       A  A   R   R
      //      L      NN   N         S      C      A    A  L      A    A  R   R
      //      L      N N  N         S      C      A    A  L      A    A  R   R
      //      L      N  N N          SSS   C      A    A  L      A    A  RRRR
      //      L      N   NN             S  C      AAAAAA  L      AAAAAA  RR
      //      L      N    N             S  C      A    A  L      A    A  R R
      //      L      N    N         S   S  C   C  A    A  L      A    A  R  R
      //      LLLLL  N    N  _____   SSS    CCC   A    A  LLLLL  A    A  R   R
      //
      namespace lazy_node {
            template<class T>
            LazyNode<T> operator+(const LazyNode<T>& a) {
                  return lazy::operator+(a);
            }

            template<class T>
            LazyNode<T> operator-(const LazyNode<T>& a) {
                  return lazy::operator-(a);
            }

#define DEF_OP(OP)                                                                                \
      template<class A, class B>                                                                  \
      LazyNode<A> OP(LazyNode<A>& a, const LazyNode<B>& b) {                                      \
            auto res = lazy::OP(TAT::Lazy<Node<A>>(a), TAT::Lazy<Node<A>>(b.template to<A>()));   \
            return LazyNode<A>(res);                                                              \
      }                                                                                           \
      template<class T, class B, class = std::enable_if_t<!std::is_base_of_v<lazy::LazyBase, B>>> \
      LazyNode<T>& OP(LazyNode<T>& a, const B& b) {                                               \
            auto res = lazy::OP(TAT::Lazy<Node<T>>(a), b);                                        \
            return LazyNode<T>(res);                                                              \
      }

            DEF_OP(operator*=)
            DEF_OP(operator/=)
            DEF_OP(operator+=)
            DEF_OP(operator-=)
#undef DEF_OP

#define DEF_OP(OP)                                                                                                     \
      template<class A, class B>                                                                                       \
      LazyNode<std::common_type_t<A, B>> OP(LazyNode<A> a, LazyNode<B> b) {                                            \
            using common = std::common_type_t<A, B>;                                                                   \
            auto res = lazy::OP(                                                                                       \
                  TAT::Lazy<Node<common>>(a.template to<common>()), TAT::Lazy<Node<common>>(b.template to<common>())); \
            return LazyNode<common>(res);                                                                              \
      }                                                                                                                \
      template<class T, class B, class = std::enable_if_t<!std::is_base_of_v<lazy::LazyBase, B>>>                      \
      LazyNode<T> OP(LazyNode<T> a, const B& b) {                                                                      \
            return LazyNode<T>(lazy::OP(TAT::Lazy<Node<T>>(a), b));                                                    \
      }                                                                                                                \
      template<class T, class B, class = std::enable_if_t<!std::is_base_of_v<lazy::LazyBase, B>>>                      \
      LazyNode<T> OP(const B& a, LazyNode<T> b) {                                                                      \
            return LazyNode<T>(lazy::OP(a, TAT::Lazy<Node<T>>(b)));                                                    \
      }

            DEF_OP(operator*)
            DEF_OP(operator/)
            DEF_OP(operator+)
            DEF_OP(operator-)
#undef DEF_OP
      } // namespace lazy_node

      /**
       * define class Lattice.
       */
      namespace lattice {
            template<int dimension>
            class Dimension;

            /**
             * Lattice use smart pointer and lambda lazy to maintain a lattice.
             */
            template<class Tags = Dimension<2>, class = double>
            class Lattice;
      } // namespace lattice
      using lattice::Lattice;
} // namespace TAT

#endif // TAT_HPP_
