/**
 * Copyright (C) 2019  Hao Zhang <zh970205@mail.ustc.edu.cn>
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
 * @file TAT.hpp
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
 * TAT库的版本号, 通常由CMakeLists.txt设置, 虽然库本身不使用, 但在一些程序中会使用这个宏
 */
#      define TAT_VERSION "unknown"
#endif // TAT_VERSION

#if (!defined TAT_USE_CPU && !defined TAT_USE_CUDA && !defined TAT_USE_DCU && !defined TAT_USE_SW)
#      if !defined TAT_DEFAULT
#            warning use CPU by default
#      endif
/**
 * 默认只使用CPU, 其实现在并没实现在诸如GPU等device上运行
 */
#      define TAT_USE_CPU
#endif

#ifdef TAT_EXTREME
#      warning EXTREME compile may cost much of compile time
#endif // TAT_EXTREME

#ifdef TAT_USE_CPU
#      ifdef TAT_USE_MKL
/**
 * MKL的头文件中, 需要使用宏来确定复数类型
 */
#            define MKL_Complex8 std::complex<float>
/**
 * MKL的头文件中, 需要使用宏来确定复数类型
 */
#            define MKL_Complex16 std::complex<double>
#            include <mkl.h>
#      else
/**
 * lapack的头文件中, 需要使用宏来确定复数类型
 */
#            define lapack_complex_float std::complex<float>
/**
 * lapack的头文件中, 需要使用宏来确定复数类型
 */
#            define lapack_complex_double std::complex<double>
#            include <cblas.h>
#            include <lapacke.h>
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
/**
 * 默认使用GESVD来计算SVD
 */
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
/**
 * 默认使用GEQRF来计算QR
 */
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
       * 用来表示张量的某个指标的维度, 张量的总大小也是用它, 因为总大小是维度的乘积
       */
      using Size = std::size_t;

      /**
       * 用来表示张量的阶, 指标的索引用的也是它
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
       * 标量类型的一些工具, 定义is_scalar_v和real_base_t, 主要是为了处理复数
       */
      namespace scalar_tools {
            /**
             * 判断一个类型是否是标量,对于大多数情况, 和std::is_scalar效果相同, 但是额外处理了复数情况
             *
             * \tparam T 被判断是否是标量的类型, 如果T是标量类型, 那么为true, 否者为false
             */
            template<class T>
            struct is_scalar : std::is_scalar<T> {};

            /**
             * 但是如果输入是std::complex<T>的话, 那么按照T本身是否是标量来判断
             */
            template<class T>
            struct is_scalar<std::complex<T>> : std::is_scalar<T> {};

            template<class T>
            static constexpr bool is_scalar_v = is_scalar<T>::value;

            /**
             * c++20支持的type_identity
             */
            template<class T>
            struct type_identity {
                  using type = T;
            };

            /**
             * 判断复类型对应的实类型, 在SVD当中的S对角阵中会使用到
             *
             * \tparam T 被判断对应实类型的类型, 如果不是复类型, 那么给出自己
             */
            template<class T>
            struct real_base : type_identity<T> {};

            /**
             * 如果输入是一个std::complex<T>, 那么type设置为T
             */
            template<class T>
            struct real_base<std::complex<T>> : type_identity<T> {};

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
       * 定义struct Legs
       */
      namespace legs {
            using IdType = int;
            /**
             * 记录已使用的id号的全局变量, 每次新建一个id时通过读取它来判断下一个id是啥, 同时将这个全局变量增加一
             */
            IdType total = 0;
            /**
             * Legs的name到id的map, 之所以不放在类内, 是因为放在外面的话gdb中的调试信息要好看一些
             */
            std::map<std::string, IdType> name2id = {};
            /**
             * Legs的id到name的map, 之所以不放在类内, 是因为放在外面的话gdb中的调试信息要好看一些
             */
            std::map<IdType, std::string> id2name = {};

            /**
             * 使用id来唯一确定一个张量指标, 并通过map拥有一个字符串名字, 这样在做比较时比较快, 存储时也比较节省空间
             */
            struct Legs {
                  /**
                   * Legs仅保存自己的id, 不保存其他诸如name的东西
                   */
                  IdType id = -1;
                  Legs() = default;
                  /**
                   * 返回直接给定id的Legs, 但是不保证此id真的有对应的name
                   *
                   * \param id 给定的id
                   */
                  explicit Legs(IdType id) : id{id} {}
                  /**
                   * 返回给定name对应id的Legs, 如果该string没有对应id, 那么创建一个新的id用来对应此name
                   *
                   * \param name 给定的name
                   */
                  explicit Legs(const std::string& name) {
                        try {
                              id = name2id.at(name);
                        } catch (const std::out_of_range& e) {
                              id = total++;
                              name2id[name] = id;
                              id2name[id] = name;
                        }
                  }
            }; // struct Legs

            /**
             * 实现legs的==, 因为之后Legs需要作为map的key
             */
            bool operator==(const Legs& a, const Legs& b) {
                  return a.id == b.id;
            }
            /**
             * 实现legs的!=, 因为之后Legs需要作为map的key
             */
            bool operator!=(const Legs& a, const Legs& b) {
                  return a.id != b.id;
            }
            /**
             * 实现legs的<, 因为之后Legs需要作为map的key
             */
            bool operator<(const Legs& a, const Legs& b) {
                  return a.id < b.id;
            }

            /**
             * 提供Legs的流输出, 有对应的name则输出name, 没有的话使用UserDefinedLeg{id}
             */
            std::ostream& operator<<(std::ostream& out, const Legs& value) {
                  try {
                        return out << id2name.at(value.id);
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
       * 这个命名空间中包含着若干默认name的Legs, 包括(Phy, 8 个方向)*(,1,2,3,4), 一共有45个
       */
      namespace legs_name {
/**
 * 定义一个name为#x的Legs
 */
#define TAT_DefineLeg(x) const TAT::Legs x(#x)
/**
 * 给定标号n, 定义9个Legs, 分别是Phy和8个方向
 */
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
/**
 * 定义legs_name中的全部45个Legs
 */
#define TAT_Legs         \
      TAT_DefineLegs();  \
      TAT_DefineLegs(1); \
      TAT_DefineLegs(2); \
      TAT_DefineLegs(3); \
      TAT_DefineLegs(4);
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
       * 定义struct Data
       */
      namespace data {
            /**
             * 每次新建一个Data都初始化内存太浪费时间了, 所以创建一个allocator, 他在条件允许的情况下不做初始化
             *
             * \tparam T 所allocate的类型
             */
            template<typename T>
            struct no_init_allocator : std::allocator<T> {
                  /**
                   * allocator需要有rebind函数
                   */
                  template<typename U>
                  struct rebind {
                        using other = no_init_allocator<U>;
                  };

                  /**
                   * 一个智能的construct, 对于标量类型的默认初始化, 直接不初始化
                   */
                  template<class U, class... Args>
                  void construct(U* p, Args&&... args) {
                        (void)p;
                        if constexpr (!((sizeof...(args) == 0) && (is_scalar_v<U>))) {
                              new (p) T(args...);
                        }
                  }
            };

            /**
             * 对一段数据的封装, 并提供一些张量操作, 他不保存和张量维度相关的信息
             *
             * \tparam Base 张量的基类型
             */
            template<class Base = double>
            struct Data {
                  using type = Base;
                  /**
                   * 存着具体数据的vector, 通过no_init_allocator, 使得对于标量类型并不会做初始化
                   */
                  std::vector<Base, no_init_allocator<Base>> base;
                  /**
                   * 这个vector的有效大小, 在一些特殊情况下, 他与数据vector的size不完全一样
                   */
                  Size size = 0;

                  // constructors
                  /**
                   * 生成给定大小的Data, 内容并不会初始化
                   *
                   * \param size Data的数据大小
                   */
                  explicit Data(Size size) : base(size), size(size) {}
                  /**
                   * 生成大小为1的Data, 直接初始化其中的数据, 用来将标量类型转为对应的无指标张量
                   *
                   * \param num 填入无指标的零阶张量中的唯一分量的值
                   */
                  explicit Data(Base num) : base(1), size(1) {
                        base[0] = num;
                  }

                  // default contructors, write then only to warn when copy.
                  Data() = default;
                  Data(Data<Base>&& other) = default;
                  Data<Base>& operator=(Data<Base>&& other) = default;
                  /**
                   * 默认复制构造函数, 只是为了在复制时给出warning
                   */
                  Data(const Data<Base>& other) : base(other.base), size(other.size) {
#ifndef TAT_NOT_WARN_COPY
                        std::clog << "Copying Data..." << std::endl;
#endif // TAT_NOT_WARN_COPY
                  }
                  /**
                   * 默认复制赋值函数, 只是为了在复制时给出warning
                   */
                  Data<Base>& operator=(const Data<Base>& other) {
                        base = other.base;
                        size = other.size;
#ifndef TAT_NOT_WARN_COPY
                        std::clog << "Copying Data..." << std::endl;
#endif // TAT_NOT_WARN_COPY
                  }

                  /**
                   * 给定generator, 设置此Data
                   *
                   * \param setter 用来设置Data值的generator, 其将一次一次被调用并赋值给Data中的数据
                   */
                  template<class Generator>
                  Data<Base>& set(Generator&& setter) & {
                        std::generate(base.begin(), base.end(), std::forward<Generator>(setter));
                        return *this;
                  }
                  /**
                   * set的右值版本
                   *
                   * \see set(Generator&& setter) &
                   */
                  template<class Generator>
                  Data<Base>&& set(Generator&& setter) && {
                        return std::move(set(std::forward<Generator>(setter)));
                  }

                  // operators

                  /**
                   * 将本Data转换成另一个类型的Data
                   *
                   * \tparam Base2 目标Data的基类型
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
                   * 获得Data看作向量后的的n模, 如果n=-1, 那么意味着最大模
                   *
                   * \tparam n 模的种类
                   */
                  template<int n>
                  Data<Base> norm() const;

                  /**
                   * 张量转置
                   *
                   * \param dims 张量的维度信息, 需保证dims的乘积等于Data的size
                   * \param plan 张量转置的方式, 如{0, 2, 1}表示对三阶张量的后两维进行转置
                   */
                  Data<Base> transpose(const std::vector<Size>& dims, const std::vector<Rank>& plan) const;

                  /**
                   * SVD的结果, 包括U, S, V, 注意S的类型可能不一样, 如果自己是复数张量的话
                   */
                  struct svd_res {
                        Data<Base> U;
                        Data<real_base_t<Base>> S;
                        Data<Base> V;
                  }; // struct svd_res

                  /**
                   * 对Data做SVD
                   *
                   * \param dims 张量的维度信息
                   * \param plan 转置方式, 先将Data转置, 之后再做SVD
                   * \param u_size, v_size 转置之后, 为了进行SVD, 将张量看作矩阵时, u边和v边的维度大小
                   * \param min_mn u_size和v_size当中的较小值
                   * \param cut S作为对角矩阵, 可以截断掉较小的奇异值, cut是截断后的长度, 如果cut=-1, 则代表不截断
                   */
                  svd_res
                  svd(const std::vector<Size>& dims,
                      const std::vector<Rank>& plan,
                      const Size& u_size,
                      const Size& v_size,
                      const Size& min_mn,
                      const Size& cut) const;

                  /**
                   * QR的结果, 包括Q, R
                   */
                  struct qr_res {
                        Data<Base> Q;
                        Data<Base> R;
                  }; // struct qr_res

                  /**
                   * 对Data做QR
                   *
                   * \param dims 张量的维度信息
                   * \param plan 转置方式, 先将Data转置, 之后再做QR
                   * \param q_size, r_size 转置之后, 进行QR时, 需将张量看作矩阵, q边和r边的的大小是q_size, r_size
                   * \param min_mn q_size和r_size当中的较小值
                   */
                  qr_res
                  qr(const std::vector<Size>& dims,
                     const std::vector<Rank>& plan,
                     const Size& q_size,
                     const Size& r_size,
                     const Size& min_mn) const;

                  /**
                   * 张量的缩并
                   *
                   * \param data1, data2 两个张量的Data
                   * \param dims1, dims2 两个张量的维度信息
                   * \param plan1, plan2 两个张量的转置方式, 缩并需先转置再做乘积
                   * \param m, k, n 两个张量转置后, 看作矩阵乘即可, 两个矩阵的维度便是m, k, n
                   */
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

                  /**
                   * 张量在一个指标上乘上对角阵
                   *
                   * \param other 存储对角阵的张量Data, 其阶数应该是1
                   * \param a, b, c 相当于data[a,b,c]*other[b], 注意不求和, b为相乘的维度, a, c分别是前面和后面的总维度
                   */
                  template<class Base2>
                  Data<Base> multiple(const Data<Base2>& other, const Size& a, const Size& b, const Size& c) const;

                  /**
                   * 给出某个位置上的数据, 即对其中的数据取索引
                   *
                   * \param pos 索引位置, 就是数据数组的索引
                   */
                  const Base& at(Size pos) const {
                        return base[pos];
                  }

                  /**
                   * at的可变版本
                   *
                   * \see at(Size pos) const
                   */
                  Base& at(Size pos) {
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
       * 定义struct Block.
       */
      namespace block {
            /**
             * Block相比于Data还记录了每个指标的维度信息, 与常见的张量库中的张量是同一个层次的抽象
             *
             * \tparam Base 张量的基类型
             */
            template<class Base = double>
            struct Block {
                  using type = Base;
                  /**
                   * 存着一个没有维度结构的Data
                   */
                  Data<Base> data;
                  /**
                   * 记录着每个指标的维度信息, 有了维度信息, 便是一个正常的张量或者张量分块了
                   */
                  std::vector<Size> dims;

                  // constructors
                  /**
                   * 构造出对应维度的Block, 其中Data的size由维度求积得到
                   *
                   * \param dims Block的维度信息
                   */
                  template<class S = std::vector<Size>, class = std::enable_if_t<!std::is_same_v<S, Block<Base>>>>
                  explicit Block(S&& dims) :
                        data(std::accumulate(dims.begin(), dims.end(), Size(1), std::multiplies<Size>())),
                        dims(std::forward<S>(dims)) {}
                  /**
                   * 生成大小为1的Block, 直接初始化其中的数据, 用来将标量类型转为对应的无指标张量
                   *
                   * \param num 填入无指标的零阶张量中的唯一分量的值
                   */
                  explicit Block(Base num) : data(num), dims() {}

                  // default contructor
                  Block() = default;

                  /**
                   * 获得Data的size
                   */
                  const Size& size() const {
                        return data.size;
                  }

                  /**
                   * 给定generator, 设置此Block
                   *
                   * \param setter 用来设置Block值的generator, 其将一次一次被调用并赋值给Block中的数据
                   */
                  template<class Generator>
                  Block<Base>& set(Generator&& setter) & {
                        data.set(std::forward<Generator>(setter));
                        return *this;
                  }
                  /**
                   * set的右值版本
                   *
                   * \see set(Generator&& setter) &
                   */
                  template<class Generator>
                  Block<Base>&& set(Generator&& setter) && {
                        return std::move(set(std::forward<Generator>(setter)));
                  }

                  // operators

                  /**
                   * 将本Block转换成另一个类型的Block
                   *
                   * \tparam Base2 目标Block的基类型
                   */
                  template<class Base2>
                  Block<Base2> to() const {
                        auto res = Block<Base2>{};
                        res.data = data.template to<Base2>();
                        res.dims = dims;
                        return res;
                  }

                  /**
                   * 获得Block看作向量后的的n模, 如果n=-1, 那么意味着最大模
                   *
                   * \tparam n 模的种类
                   */
                  template<int n>
                  Block<Base> norm() const;

                  /**
                   * 张量转置
                   *
                   * \param plan 张量转置的方式, 如{0, 2, 1}表示对三阶张量的后两维进行转置
                   */
                  Block<Base> transpose(const std::vector<Rank>& plan) const;

                  /**
                   * SVD的结果, 包括U, S, V, 注意S的类型可能不一样, 如果自己是复数张量的话
                   */
                  struct svd_res {
                        Block<Base> U;
                        Block<real_base_t<Base>> S;
                        Block<Base> V;
                  }; // struct svd_res

                  /**
                   * 对Block做SVD
                   *
                   * \param plan 转置方式, 先将Block转置, 之后再做SVD
                   * \param u_rank 转置之后, 为了进行SVD, 将张量看作矩阵时, u边包含的原指标个数
                   * \param cut S作为对角矩阵, 可以截断掉较小的奇异值, cut是截断后的长度, cut=-1代表不截断
                   */
                  svd_res svd(const std::vector<Rank>& plan, const Rank& u_rank, const Size& cut) const;

                  /**
                   * QR的结果, 包括Q, R
                   */
                  struct qr_res {
                        Block<Base> Q;
                        Block<Base> R;
                  }; // struct qr_res

                  /**
                   * 对Block做QR
                   *
                   * \param plan 转置方式, 先将Block转置, 之后再做QR
                   * \param q_rank 转置之后, 为了进行QR, 将张量看作矩阵时, q边包含的原指标个数
                   */
                  qr_res qr(const std::vector<Rank>& plan, const Rank& q_rank) const;

                  /**
                   * 张量的缩并
                   *
                   * \param block1, block2 两个张量的Block
                   * \param plan1, plan2 两个张量的转置方式, 缩并需先转置再做乘积
                   * \param contract_num 两个张量转置后, 看作矩阵乘时, 矩阵乘中所要乘掉的指标数目
                   */
                  static Block<Base> contract(
                        const Block<Base>& block1,
                        const Block<Base>& block2,
                        const std::vector<Rank>& plan1,
                        const std::vector<Rank>& plan2,
                        const Rank& contract_num);

                  /**
                   * 张量在一个指标上乘上对角阵
                   *
                   * \param other 存储对角阵的张量Data, 其阶数应该是1
                   * \param index 所需乘上去的指标的索引
                   */
                  template<class Base2>
                  Block<Base> multiple(const Block<Base2>& other, const Rank& index) const;

                  /**
                   * 通过张量的索引获得在数据中的位置, 在at中使用
                   *
                   * \param index 张量的索引, 比如一个三阶张量, index={1, 2, 3}则代表着取Block[1, 2, 3]
                   * \see at(const std::vector<Size>& index) const
                   */
                  Size get_position(const std::vector<Size>& index) const {
                        Size res = 0;
                        for (Rank i = 0; i < dims.size(); i++) {
                              res = res * dims[i] + index[i];
                        }
                        return res;
                  }

                  /**
                   * 给出某个张量索引上的数据
                   *
                   * \param index 张量的索引, 比如一个三阶张量, index={1, 2, 3}则代表着取Block[1, 2, 3]
                   * \see get_position(const std::vector<Size>& index) const
                   */
                  const Base& at(const std::vector<Size>& index) const {
                        return data.at(get_position(index));
                  }

                  /**
                   * at的可变版本
                   *
                   * \see at(const std::vector<Size>& index) const
                   */
                  Base& at(const std::vector<Size>& index) {
                        return data.at(get_position(index));
                  }
            };
      } // namespace block
      using block::Block;

      /**
       * 定义struct Tensor
       *
       * 尚未实现, 现在直接通过alias使用Block
       * 暂时不支持分块张量, 使用Block作为Tensor
       */
      namespace tensor = block;
      namespace block {
            template<class Base = double>
            using Tensor = Block<Base>;
      }
      using tensor::Tensor;

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
       * 定义struct Node
       */
      namespace node {
            /**
             * 除了Tensor中的维度信息, struct Node还会额外保存指标的id, 用来在复杂的操作中方便地确定指标而不至于混乱
             *
             * \tparam Base 张量的基类型
             */
            template<class Base = double>
            struct Node {
                  using type = Base;
                  /**
                   * 存一个Tensor保存维度信息和本身的内容, 为了让legs_rename不改变原node, 使用shared_ptr
                   */
                  std::shared_ptr<Tensor<Base>> tensor_ptr;

                  /**
                   * 记录Legs的vector用来识别指标
                   */
                  std::vector<Legs> legs;

                  /**
                   * 获得Tensor的值, 因为tensor为了legs_rename方便使用了shared_ptr
                   */
                  const Tensor<Base>& tensor() const {
                        return *tensor_ptr;
                  }
                  /**
                   * tensor的可变版本
                   *
                   * \see tensor() const
                   */
                  Tensor<Base>& tensor() {
                        return *tensor_ptr;
                  }

                  // constructors
                  /**
                   * 通过指标name和维度信息构造一个Node, 这在分块张量实现后需要重写, 比较好的方式是传入一个config类
                   *
                   * \param legs 指标的name信息
                   * \param dims Block的维度信息
                   */
                  template<class T1 = std::vector<Legs>, class T2 = std::vector<Size>>
                  explicit Node(T1&& _legs, T2&& _dims) :
                        tensor_ptr(std::make_shared<Tensor<Base>>(std::forward<T2>(_dims))),
                        legs(std::forward<T1>(_legs)) {
                        // expect length of legs and dims is same
                        assert(legs.size() == tensor().dims.size());
                        // expect no same element in legs
                        assert(legs.size() == std::set<TAT::Legs>(legs.begin(), legs.end()).size());
                  }
                  /**
                   * 生成大小为1的Node, 直接初始化其中的数据, 用来将标量类型转为对应的无指标张量
                   *
                   * \param num 填入无指标的零阶张量中的唯一分量的值
                   */
                  explicit Node(Base num) : tensor_ptr(std::make_shared<Tensor<Base>>(num)), legs() {}

                  // default contructor
                  Node() : tensor_ptr(std::make_shared<Tensor<Base>>()), legs(){};

                  /**
                   * 获得Data的size
                   */
                  const Size& size() const {
                        return tensor().size();
                  }

                  /**
                   * 获得Block的dims
                   */
                  const std::vector<Size>& dims() const {
                        return tensor().dims();
                  }

                  /**
                   * 给定generator, 设置此Node, 这是唯一的inplace操作
                   *
                   * \param setter 用来设置Node值的generator, 其将一次一次被调用并赋值给Node中的数据
                   */
                  template<class Generator>
                  Node<Base>& set(Generator&& setter) & {
                        tensor().set(std::forward<Generator>(setter));
                        return *this;
                  }
                  /**
                   * set的右值版本
                   *
                   * \see set(Generator&& setter) &
                   */
                  template<class Generator>
                  Node<Base>&& set(Generator&& setter) && {
                        return std::move(set(std::forward<Generator>(setter)));
                  }

                  // operators
                  /**
                   * 更改指标的name即legs
                   *
                   * \param dict 如何替换指标name的字典
                   */
                  Node<Base> legs_rename(const std::map<Legs, Legs>& dict) const {
                        auto res = Node<Base>();
                        res.tensor_ptr = tensor_ptr;
                        std::transform(legs.begin(), legs.end(), std::back_inserter(res.legs), [&](Legs leg) {
                              try {
                                    return dict.at(leg);
                              } catch (const std::out_of_range& e) {
                                    return leg;
                              }
                        });
                        return res;
                  }

                  /**
                   * 将本Node转换成另一个类型的Node
                   *
                   * \tparam Base2 目标Node的基类型
                   */
                  template<class Base2>
                  Node<Base2> to() const {
                        auto res = Node<Base2>{};
                        res.tensor() = tensor().template to<Base2>();
                        res.legs = legs;
                        return res;
                  }

                  /**
                   * 获得Node看作向量后的的n模, 如果n=-1, 那么意味着最大模
                   *
                   * \tparam n 模的种类
                   */
                  template<int n>
                  Node<Base> norm() const;

                  /**
                   * 张量转置
                   *
                   * \paran 转置后的目标legs
                   */
                  Node<Base> transpose(const std::vector<Legs>& new_legs) const;

                  /**
                   * SVD的结果, 包括U, S, V, 注意S的类型可能不一样, 如果自己是复数张量的话
                   */
                  struct svd_res {
                        Node<Base> U;
                        Node<real_base_t<Base>> S;
                        Node<Base> V;
                  }; // struct svd_res

                  /**
                   * 对Node做SVD
                   *
                   * \param input_u_legs SVD后成为U边所包含的指标name
                   * \param new_u_legs, new_v_legs SVD出来的U矩阵和V矩阵的两个新指标
                   * \param cut S作为对角矩阵, 可以截断掉较小的奇异值, cut是截断后的长度, cut=-1代表不截断
                   */
                  svd_res
                  svd(const std::vector<Legs>& input_u_legs,
                      const Legs& new_u_legs,
                      const Legs& new_v_legs,
                      const Rank& cut = -1) const;

                  /**
                   * QR的结果, 包括Q, R
                   */
                  struct qr_res {
                        Node<Base> Q;
                        Node<Base> R;
                  }; // struct qr_res

                  /**
                   * 对Node做QR
                   *
                   * \param input_q_legs QR后成为Q边所包含的指标name
                   * \param new_q_legs, new_r_legs QR出来的Q矩阵和R矩阵的两个新指标
                   */
                  qr_res
                  qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const;

                  /**
                   * 对Node做QR
                   *
                   * \param input_r_legs QR后成为R边所包含的指标name
                   * \param new_r_legs, new_q_legs QR出来的R矩阵和Q矩阵的两个新指标
                   */
                  qr_res
                  rq(const std::vector<Legs>& input_r_legs, const Legs& new_r_legs, const Legs& new_q_legs) const {
                        std::vector<TAT::Legs> input_q_legs;
                        std::copy_if(legs.begin(), legs.end(), std::back_inserter(input_q_legs), [&](const Legs& i) {
                              return std::find(input_r_legs.begin(), input_r_legs.end(), i) == input_r_legs.end();
                        });
                        return qr(input_q_legs, new_q_legs, new_r_legs);
                  }

                  /**
                   * 张量的缩并
                   *
                   * \param node1, node2 两个张量的Node
                   * \param legs1, legs2 两个张量所要缩并掉的指标name即legs
                   */
                  static Node<Base> contract(
                        const Node<Base>& node1,
                        const Node<Base>& node2,
                        const std::vector<Legs>& legs1,
                        const std::vector<Legs>& legs2);

                  /**
                   * 相当于在一个指标上乘上对角阵, Node层次给出相乘的指标的name
                   */
                  template<class Base2>
                  Node<Base> multiple(const Node<Base2>& other, const Legs& position) const;

                  /**
                   * 通过指标字典获得索引, 在at中使用
                   */
                  std::vector<Size> get_index(const std::map<Legs, Size>& dict) const {
                        std::vector<Size> res;
                        std::transform(legs.begin(), legs.end(), std::back_inserter(res), [&dict](const Legs& l) {
                              return dict.at(l);
                        });
                        return res;
                  }

                  /**
                   * 根据指标字典找到Node上的某个数据
                   */
                  const Base& at(const std::map<Legs, Size>& dict) const {
                        return tensor().at(get_index(dict));
                  }

                  /**
                   * at的可变版本
                   */
                  Base& at(const std::map<Legs, Size>& dict) {
                        return tensor().at(get_index(dict));
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
            /**
             * Data的流输出, 如果是cout, cerr, clog之一则输出文本, 否则输出二进制数据
             */
            template<class Base>
            std::ostream& operator<<(std::ostream& out, const Data<Base>& value) {
                  if (&out == &std::cout || &out == &std::cerr || &out == &std::clog) {
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
                  } else {
                        out.write(reinterpret_cast<const char*>(&value.size), sizeof(Size));
                        out.write(reinterpret_cast<const char*>(value.base.data()), value.size * sizeof(Base));
                        return out;
                  }
            } // operator<<

            /**
             * Data的流输入, 只支持二进制数据
             */
            template<class Base>
            std::istream& operator>>(std::istream& in, Data<Base>& value) {
                  in.read(reinterpret_cast<char*>(&value.size), sizeof(Size));
                  value.base = std::vector<Base, no_init_allocator<Base>>(value.size);
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

            /**
             * Block的流输出, 如果是cout, cerr, clog之一则输出文本, 否则输出二进制数据
             */
            template<class Base>
            std::ostream& operator<<(std::ostream& out, const Block<Base>& value) {
                  if (&out == &std::cout || &out == &std::cerr || &out == &std::clog) {
                        return out << "{" << rang::fg::magenta << "\"dims\": " << value.dims << rang::fg::reset
                                   << ", \"data\": " << value.data << "}";
                  } else {
                        Rank rank = value.dims.size();
                        out.write(reinterpret_cast<const char*>(&rank), sizeof(Rank));
                        out.write(reinterpret_cast<const char*>(value.dims.data()), rank * sizeof(Size));
                        out << value.data;
                        return out;
                  }
            } // operator<<

            /**
             * Block的流输入, 只支持二进制数据
             */
            template<class Base>
            std::istream& operator>>(std::istream& in, Block<Base>& value) {
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

            /**
             * Node的流输出, 如果是cout, cerr, clog之一则输出文本, 否则输出二进制数据
             */
            template<class Base>
            std::ostream& operator<<(std::ostream& out, const Node<Base>& value) {
                  if (&out == &std::cout || &out == &std::cerr || &out == &std::clog) {
                        return out << "{" << rang::fgB::yellow << "\"rank\": " << value.legs.size() << rang::fg::reset
                                   << ", " << rang::fgB::blue << "\"legs\": " << value.legs << rang::fg::reset
                                   << ", \"tensor\": " << value.tensor() << "}";

                  } else {
                        Rank rank = value.legs.size();
                        out.write(reinterpret_cast<const char*>(&rank), sizeof(Rank));
                        out.write(reinterpret_cast<const char*>(value.legs.data()), rank * sizeof(Legs));
                        out << value.tensor();
                        return out;
                  }
            } // operator<<

            /**
             * Node的流输入, 只支持二进制数据
             */
            template<class Base>
            std::istream& operator>>(std::istream& in, Node<Base>& value) {
                  Rank rank;
                  in.read(reinterpret_cast<char*>(&rank), sizeof(Rank));
                  value.legs = std::vector<Legs>(rank);
                  in.read(reinterpret_cast<char*>(value.legs.data()), rank * sizeof(Legs));
                  in >> value.tensor();
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
                  res.tensor() = tensor().template norm<n>();
                  return res;
            } // norm
      } // namespace node
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
#define DEF_OP(OP, EVAL1, EVAL2)                            \
      template<class Base, class Base2>                     \
      Data<Base>& OP(Data<Base>& a, const Data<Base2>& b) { \
            if (b.size == 1) {                              \
                  for (Size i = 0; i < a.size; i++) {       \
                        EVAL1;                              \
                  }                                         \
            } else {                                        \
                  for (Size i = 0; i < a.size; i++) {       \
                        EVAL2;                              \
                  }                                         \
            }                                               \
            return a;                                       \
      }

            DEF_OP(operator*=, a.base[i] *= b.base[0], a.base[i] *= b.base[i])
            DEF_OP(operator/=, a.base[i] /= b.base[0], a.base[i] /= b.base[i])
            DEF_OP(operator+=, a.base[i] += b.base[0], a.base[i] += b.base[i])
            DEF_OP(operator-=, a.base[i] -= b.base[0], a.base[i] -= b.base[i])
#undef DEF_OP

#define DEF_OP(OP, EVAL1, EVAL2, EVAL3)                     \
      template<class Base1, class Base2>                    \
      auto OP(const Data<Base1>& a, const Data<Base2>& b) { \
            using Base = std::common_type_t<Base1, Base2>;  \
            if (a.size == 1) {                              \
                  auto res = Data<Base>(b.size);            \
                  for (Size i = 0; i < b.size; i++) {       \
                        res.base[i] = EVAL1;                \
                  }                                         \
                  return res;                               \
            }                                               \
            if (b.size == 1) {                              \
                  auto res = Data<Base>(a.size);            \
                  for (Size i = 0; i < a.size; i++) {       \
                        res.base[i] = EVAL2;                \
                  }                                         \
                  return res;                               \
            }                                               \
            auto res = Data<Base>(a.size);                  \
            for (Size i = 0; i < a.size; i++) {             \
                  res.base[i] = EVAL3;                      \
            }                                               \
            return res;                                     \
      }

            DEF_OP(operator*, a.base[0]* b.base[i], a.base[i]* b.base[0], a.base[i]* b.base[i])
            DEF_OP(operator/, a.base[0] / b.base[i], a.base[i] / b.base[0], a.base[i] / b.base[i])
            DEF_OP(operator+, a.base[0] + b.base[i], a.base[i] + b.base[0], a.base[i] + b.base[i])
            DEF_OP(operator-, a.base[0] - b.base[i], a.base[i] - b.base[0], a.base[i] - b.base[i])
#undef DEF_OP

            template<class Base>
            const Data<Base>& operator+(const Data<Base>& a) {
                  return a;
            } // operator+

            template<class Base>
            Data<Base> operator+(Data<Base>&& a) {
                  return Data<Base>(std::move(a));
            } // operator+

            template<class Base>
            Data<Base> operator-(const Data<Base>& a) {
                  auto res = Data<Base>(a.size);
                  for (Size i = 0; i < a.size; i++) {
                        res.base[i] = -a.base[i];
                  }
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

#define DEF_OP(OP)                                             \
      template<class Base, class Base2>                        \
      Block<Base>& OP(Block<Base>& a, const Block<Base2>& b) { \
            assert(b.dims.size() == 0 || a.dims == b.dims);    \
            data::OP(a.data, b.data);                          \
            return a;                                          \
      }

            DEF_OP(operator*=)
            DEF_OP(operator/=)
            DEF_OP(operator+=)
            DEF_OP(operator-=)
#undef DEF_OP

#define DEF_OP(OP)                                            \
      template<class Base1, class Base2>                      \
      auto OP(const Block<Base1>& a, const Block<Base2>& b) { \
            using Base = std::common_type_t<Base1, Base2>;    \
            Block<Base> res;                                  \
            if (b.dims.size() == 0) {                         \
                  res.dims = a.dims;                          \
            } else if (a.dims.size() == 0) {                  \
                  res.dims = b.dims;                          \
            } else {                                          \
                  res.dims = a.dims;                          \
                  assert(a.dims == b.dims);                   \
            }                                                 \
            res.data = data::OP(a.data, b.data);              \
            return res;                                       \
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
      template<class Base, class Base2>                     \
      Node<Base>& OP(Node<Base>& a, const Node<Base2>& b) { \
            assert(b.legs.size() == 0 || a.legs == b.legs); \
            tensor::OP(a.tensor(), b.tensor());             \
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

#define DEF_OP(OP)                                             \
      template<class Base1, class Base2>                       \
      auto OP(const Node<Base1>& a, const Node<Base2>& b) {    \
            using Base = std::common_type_t<Base1, Base2>;     \
            Node<Base> res;                                    \
            if (b.legs.size() == 0) {                          \
                  res.legs = a.legs;                           \
            } else if (a.legs.size() == 0) {                   \
                  res.legs = b.legs;                           \
            } else {                                           \
                  res.legs = a.legs;                           \
                  assert(a.legs == b.legs);                    \
            }                                                  \
            res.tensor() = tensor::OP(a.tensor(), b.tensor()); \
            return res;                                        \
      }                                                        \
      template<class Base, class B>                            \
      Node<Base> OP(const Node<Base>& a, const B& b) {         \
            return OP(a, Node<Base>(b));                       \
      }                                                        \
      template<class Base, class B>                            \
      Node<Base> OP(const B& b, const Node<Base>& a) {         \
            return OP(Node<Base>(b), a);                       \
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
                  res.tensor() = +std::move(a.tensor());
                  return res;
            } // operator+

            template<class Base>
            Node<Base> operator-(const Node<Base>& a) {
                  Node<Base> res;
                  res.legs = a.legs;
                  res.tensor() = -a.tensor();
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
                        if (dims.size() == 0) {
                              *dst = *src;
                        } else {
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
                        }
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
                              res.push_back(std::distance(legs.begin(), std::find(legs.begin(), legs.end(), i)));
                        } // for i
                        return res;
                  } // plan
            } // namespace transpose

            template<class Base>
            Node<Base> Node<Base>::transpose(const std::vector<Legs>& new_legs) const {
                  Node<Base> res;
                  std::copy_if(new_legs.begin(), new_legs.end(), std::back_inserter(res.legs), [&](const Legs& i) {
                        return std::find(legs.begin(), legs.end(), i) != legs.end();
                  });
                  // remove legs not exist, but all existing legs should not ignore
                  assert(res.legs.size() == std::set<TAT::Legs>(res.legs.begin(), res.legs.end()).size());
                  assert(res.legs.size() == legs.size());
                  std::vector<Rank> plan = transpose::generate_plan(res.legs, legs);
                  assert(plan.size() == legs.size());
                  assert(res.legs.size() == legs.size());
                  res.tensor() = tensor().transpose(plan);
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
                        Rank u_rank = 0; // u_legs.size();
                        V_legs.push_back(new_v_legs);
                        for (const auto& i : total_legs) {
                              if (std::find(u_legs.begin(), u_legs.end(), i) != u_legs.end()) {
                                    U_legs.push_back(i);
                                    u_rank++;
                              } else {
                                    V_legs.push_back(i);
                              } // if
                        } // for
                        U_legs.push_back(new_u_legs);
                        // std::copy_if(
                        //       total_legs.begin(), total_legs.end(), std::back_inserter(U_legs), [&](const Legs& i) {
                        //       return vector_legs::find_iter(u_legs, i)!=u_legs.end();
                        // });
                        // std::copy_if(
                        //       total_legs.begin(), total_legs.end(), std::back_inserter(V_legs), [&](const Legs& i) {
                        //       return vector_legs::find_iter(u_legs, i)==u_legs.end();
                        // });
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
                  svd_res res;
                  auto [U_legs, V_legs, tmp_legs, u_rank] = svd::plan(legs, input_u_legs, new_u_legs, new_v_legs);
                  res.U.legs = std::move(U_legs);
                  res.V.legs = std::move(V_legs);
                  std::vector<Rank> plan = transpose::generate_plan(tmp_legs, legs);
                  auto tensor_res = tensor().svd(plan, u_rank, cut);
                  res.S.legs = {new_u_legs}; // new_u_legs or new_v_legs
                  res.U.tensor() = std::move(tensor_res.U);
                  res.S.tensor() = std::move(tensor_res.S);
                  res.V.tensor() = std::move(tensor_res.V);
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
                  qr_res res;
                  auto [Q_legs, R_legs, tmp_legs, q_rank] = svd::plan(legs, input_q_legs, new_q_legs, new_r_legs);
                  res.Q.legs = std::move(Q_legs);
                  res.R.legs = std::move(R_legs);
                  std::vector<Rank> plan = transpose::generate_plan(tmp_legs, legs);
                  auto tensor_res = tensor().qr(plan, q_rank);
                  res.Q.tensor() = std::move(tensor_res.Q);
                  res.R.tensor() = std::move(tensor_res.R);
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
                              // 为了一些不完善的cblas.h的兼容性
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
                       const std::vector<Legs>& legs2) {
                        assert(legs1.size() == legs2.size());

                        std::vector<Legs> legs;
                        std::vector<Legs> new_legs1;
                        std::vector<Legs> new_legs2;
                        Rank contract_num = 0;

                        // find the legs which need contract
                        std::vector<Legs> legs1_contract;
                        std::vector<Legs> legs2_contract;
                        for (Size i = 0; i < legs1.size(); i++) {
                              if (std::find(total_legs1.begin(), total_legs1.end(), legs1[i]) != total_legs1.end() &&
                                  std::find(total_legs2.begin(), total_legs2.end(), legs2[i]) != total_legs2.end()) {
                                    contract_num++;
                                    legs1_contract.push_back(legs1[i]);
                                    legs2_contract.push_back(legs2[i]);
                              }
                        }
                        // new_legs1 append if not in contract
                        int left_number = 0;
                        for (const auto& i : total_legs1) {
                              if (std::find(legs1_contract.begin(), legs1_contract.end(), i) == legs1_contract.end()) {
                                    new_legs1.push_back(i);
                                    legs.push_back(i);
                                    left_number++;
                              }
                        }
                        // new_legs1 append contract
                        new_legs1.insert(new_legs1.end(), legs1_contract.begin(), legs1_contract.end());
                        // new_legs2 append contract
                        new_legs2.insert(new_legs2.end(), legs2_contract.begin(), legs2_contract.end());
                        // new_legs2 append if not in contract
                        for (const auto& i : total_legs2) {
                              if (std::find(legs2_contract.begin(), legs2_contract.end(), i) == legs2_contract.end()) {
                                    new_legs2.push_back(i);
                                    legs.push_back(i);
                              }
                        }
                        // legs append 2 total_legs if not in contract with map
                        /*
                        for (const auto& [i, j] : map1) {
                              std::replace(legs.begin(), legs.begin() + left_number, i, j);
                        }
                        for (const auto& [i, j] : map2) {
                              std::replace(legs.begin() + left_number, legs.end(), i, j);
                        }
                        */

                        return std::tuple{contract_num, legs, new_legs1, new_legs2};
                  } // plan
            } // namespace contract

            template<class Base>
            Node<Base> Node<Base>::contract(
                  const Node<Base>& node1,
                  const Node<Base>& node2,
                  const std::vector<Legs>& legs1,
                  const std::vector<Legs>& legs2) {
                  Node<Base> res;
                  auto [contract_num, res_legs, new_legs1, new_legs2] =
                        contract::plan(node1.legs, node2.legs, legs1, legs2);
                  res.legs = std::move(res_legs);
                  auto plan1 = transpose::generate_plan(new_legs1, node1.legs);
                  auto plan2 = transpose::generate_plan(new_legs2, node2.legs);
                  assert(new_legs1.size() == node1.legs.size());
                  assert(plan1.size() == node1.legs.size());
                  assert(new_legs2.size() == node2.legs.size());
                  assert(plan2.size() == node2.legs.size());
                  assert(res.legs.size() == std::set<TAT::Legs>(res.legs.begin(), res.legs.end()).size());
                  res.tensor() = Tensor<Base>::contract(node1.tensor(), node2.tensor(), plan1, plan2, contract_num);
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
                  template<class Base, class Base2>
                  void
                  run(Base* res_data,
                      const Base* src_data,
                      const Base2* other_data,
                      const Size& a,
                      const Size& b,
                      const Size& c) {
                        for (Size i = 0; i < a; i++) {
                              for (Size j = 0; j < b; j++) {
                                    Base2 v = other_data[j];
                                    for (Size k = 0; k < c; k++) {
                                          *(res_data++) = *(src_data++) * v;
                                    } // for k
                              } // for j
                        } // for i
                  } // run
            } // namespace multiple

            template<class Base>
            template<class Base2>
            Data<Base>
            Data<Base>::multiple(const Data<Base2>& other, const Size& a, const Size& b, const Size& c) const {
                  Data<Base> res(size);
                  assert(b == other.size);
                  assert(a * b * c == size);
                  multiple::run<Base, Base2>(res.base.data(), base.data(), other.base.data(), a, b, c);
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
            template<class Base2>
            Block<Base> Block<Base>::multiple(const Block<Base2>& other, const Rank& index) const {
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
            template<class Base2>
            Node<Base> Node<Base>::multiple(const Node<Base2>& other, const Legs& position) const {
                  Node<Base> res;
                  assert(other.legs.size() == 1);
                  res.legs = legs;
                  auto pos = std::find(legs.begin(), legs.end(), position);
                  if (pos == legs.end()) {
                        return *this;
                  } // if not multiple
                  Rank index = std::distance(legs.begin(), pos);
                  res.tensor() = tensor().multiple(other.tensor(), index);
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
      /**
       * lazy框架
       */
      namespace lazy {
#ifndef NDEBUG
            /**
             * 用来记录lazy core在哪里创建的全局变量
             */
            std::vector<std::string> position;
#endif
            /**
             * 在作用域中新建一个name_scope变量可将此作用域内所有创建的lazy core的position变量都append一个字符串
             */
            struct name_scope {
                  /**
                   * 根据str构造一个name_scope
                   *
                   * \param str 构造的name_scope的名字
                   * \note 使用方式是 \code{.cpp}
                   * {
                   *    lazy::name_scope name_scope("some_name");
                   *    // other operator
                   * }
                   * \endcode
                   */
                  template<class Str = std::string>
                  name_scope(Str&& str) {
                        (void)str;
#ifndef NDEBUG
                        position.push_back(std::forward<Str>(str));
#endif
                  }

                  ~name_scope() {
#ifndef NDEBUG
                        assert(position.size() != 0);
                        position.pop_back();
#endif
                  }
            };

            /**
             * LazyCore的基类, 用来存储无类型信息的上下游信息
             */
            struct LazyCoreBase {
                  /**
                   * 重置或不重置本lazy core, 并重置所有下游的lazy core
                   *
                   * \param reset_itself 是否重置自身的flag
                   */
                  virtual void reset(bool reset_itself = true) = 0;
                  /**
                   * 析构lazy core, 只会在没有下游时调用, 调用时会将自己从上游的下游列表中去除
                   */
                  virtual ~LazyCoreBase() = default;

                  /**
                   * 引用了本lazy core的其他lazy core列表, 并没有所有权, 资源由子类中的func所拷贝的shared_ptr管理
                   */
                  std::set<LazyCoreBase*> downstream;
                  /**
                   * 本lazy core所引用的其他lazy core列表, 并没有所有权, 资源由子类中的func所拷贝的shared_ptr管理
                   */
                  std::set<LazyCoreBase*> upstream;

#ifndef NDEBUG
                  /**
                   * lazy core创建时的位置信息, debug用
                   */
                  std::vector<std::string> local_position;
#endif
            };

            /**
             * 给std::unique_ptr用的deleter, 会根据可变的own_flag来判断是否真的delete指针
             *
             * \tparam T 被delete的类型
             */
            template<class T>
            struct maybe_deleter {
                  /**
                   * 记录是否拥有所有权的flag, 如果为true的在调用operator()时真的会delte指针, 反之不会
                   */
                  bool own_flag = false;
                  /**
                   * std::unique_ptr的deleter会通过调用operator()来delete指针
                   *
                   * \param ptr 或将delete的指针
                   */
                  void operator()(const T* ptr) const {
                        if (own_flag) {
                              delete ptr;
                        }
                  }
                  /**
                   * 设定记录所有权的own_flag
                   *
                   * \param f 设定的值
                   * \see own_flag
                   */
                  void set_own(bool f) {
                        own_flag = f;
                  }
            };

            /**
             * LazyCore是一个类型Lazy化后的类型, 但是一会以shared_ptr<LazyCore<T>>的形式存在, 所以需要Lazy类型的包装
             *
             * \tparam T 被lazy化的类型
             */
            template<class T>
            struct LazyCore : LazyCoreBase {
                  using type = T;

                  void reset(bool reset_itself = true) override {
                        if (value_ptr) {
                              if (reset_itself) {
                                    value_ptr.reset();
                              }
                              for (const auto& ds : downstream) {
                                    ds->reset();
                              }
                        }
                  }

                  ~LazyCore() override {
                        dump_upstream();
                  }

                  /**
                   * 设置本lazy core所存储的值, 包括设置指针和设置指针所有权
                   */
                  void record_value(const T* ptr, bool own_flag) {
                        value_ptr.reset(ptr);
                        value_ptr.get_deleter().set_own(own_flag);
                  }

                  /**
                   * 一个所有权不确定的指针, 通过maybe_deleter实现
                   *
                   * \see maybe_deleter
                   */
                  std::unique_ptr<const T, maybe_deleter<T>> value_ptr;
                  /**
                   * 用来计算出value的函数, func没有返回值, 调用func时应自己将值通过record_value赋值给value
                   *
                   * \see record_value
                   */
                  std::function<void()> func;

                  /**
                   * 将自己从自己上游的下游列表中清除
                   *
                   * \note 此操作必须在子类中调用, 否则会出现先调用两个上下游对象的子类析构函数,
                   * 再调用父类析构函数的情况, 使得下游的dump_upstream还没调用的时候, 上有对象便开始析构,
                   * 无法保证下游先调用dump_upstream
                   */
                  void dump_upstream() {
                        for (const auto& us : upstream) {
                              us->downstream.erase(this); // if there is duplicated, it still work
                        }
                        upstream.clear();
                  }

                  /**
                   * 获得value的值, 如果不存在, 则通过func计算出
                   */
                  const T& value() {
                        if (!value_ptr) {
                              try {
                                    func();
                              } catch (std::exception& e) {
#ifndef NDEBUG
                                    std::clog << " [LAZY ERROR] in" << std::endl;
                                    for (const auto& i : local_position) {
                                          std::clog << "  " << i << std::endl;
                                    }
#else
                                    std::clog << " [LAZY ERROR compile debug mode to show position] " << std::endl;
#endif
                                    throw;
                              }
                        }
                        return *value_ptr;
                  }

                  /**
                   * 获得value的值, 并移动走, 如果本身没有所有权, 可能会有危险
                   *
                   * \note 如果本身是通过直接设置value的方式创建的, 那么pop后就难以找到原来的值了
                   */
                  T pop() {
                        value();
                        T res = std::move(*const_cast<T*>(value_ptr.get()));
                        reset();
                        return res;
                  }

                  /**
                   * 给定指针, 设置value, 不拥有所有权
                   *
                   * \param ptr 用来设置value的指针
                   */
                  void set_point_value(const T* ptr) {
                        reset();
                        dump_upstream();
                        func = std::function<void()>();
                        record_value(ptr, false);
                  }

                  /**
                   * 给定值, 设置value, 拥有所有权
                   *
                   * \param args 创建value的参数, 一般就是可移动的T类型本身
                   */
                  template<class... Args>
                  void set_value(Args&&... args) {
                        reset();
                        dump_upstream();
                        func = std::function<void()>();
                        record_value(new T(std::forward<Args>(args)...), true);
                  }

                  /**
                   * 将func设定为调用f函数, 但是f的参数为其他几个lazy core
                   *
                   * \param f 用来设置value的函数, 参数为其他lazy core calc出来的值, 返回本lazy
                   * core将调用record_value的参数, 包括指针value_point和所有权标志own_flag
                   * \param args 可calc出f的参数的lazy core
                   * \see maybe_res
                   */
                  template<class Func, class... Args>
                  void set_maybe_point_func(Func&& f, std::shared_ptr<LazyCore<Args>>... args) {
                        reset();
                        dump_upstream();
                        (..., args->downstream.insert(this));
                        (..., upstream.insert(args.get()));
                        func = [args..., this, f = std::move(f)]() {
                              auto res = f(args->value()...);
                              record_value(res.value_point, res.own_flag);
                        };
#ifndef NDEBUG
                        local_position.assign(position.begin(), position.end());
#endif
                  }

                  /**
                   * 被set_maybe_point_func使用的中间参数
                   *
                   * \see set_maybe_point_func
                   */
                  struct maybe_res {
                        const T* value_point;
                        bool own_flag;
                  }; // struct maybe_res

                  /**
                   * 与set_maybe_point_func类似, 但是一定不拥有所有权
                   *
                   * \param f 返回的时类型为T*的值, 用来设置value
                   * \see set_maybe_point_func
                   */
                  template<class Func, class... Args>
                  void set_point_func(Func&& f, std::shared_ptr<LazyCore<Args>>... args) {
                        set_maybe_point_func(
                              [f = std::move(f)](const Args&... args_v) {
                                    return maybe_res{f(args_v...), false};
                              },
                              args...);
                  }

                  /**
                   * 与set_maybe_point_func类似, 但是一定拥有所有权
                   *
                   * \param f 返回的时类型为T的值, 用来设置value
                   * \see set_maybe_point_func
                   */
                  template<class Func, class... Args>
                  void set_func(Func&& f, std::shared_ptr<LazyCore<Args>>... args) {
                        set_maybe_point_func(
                              [f = std::move(f)](const Args&... args_v) {
                                    return maybe_res{new T(f(args_v...)), true};
                              },
                              args...);
                  }

                  /**
                   * inplace的更新一个lazy core
                   *
                   * \param modify 将调用modify来更新此lazy core, modify会将本lazy core的左值作为第一个参数1
                   * \param args modify的其他lazy core参数
                   * \note 对一个有下游的lazy core调用update可能会危险, 因为更新后可能已经不是被下游func所期待的值了
                   * \note 无论本身有没有func, 拥有所有权的话时没问题的, 但是如果没有所有权, 会破坏他人的值
                   * \note 如果本身没有func, 且args中含有暂时尚未calc的lazy core, 会因立即计算而crash, 如果不这么设计,
                   * 不可避免T的复制, 这是不希望发生的, 当然也可以通过立即calc来实现, 但是那样的话, 违背了lazy的初衷,
                   * 如果可以立即calc, 在这里尽量不用lazy
                   */
                  template<class Func, class... Args>
                  void update(Func&& modify, std::shared_ptr<LazyCore<Args>>... args) {
                        (..., args->downstream.insert(this));
                        (..., upstream.insert(args.get()));
                        if (func) {
                              reset();
                              func = [args..., this, func = std::move(func), modify = std::move(modify)]() {
                                    func();
                                    modify(*const_cast<T*>(value_ptr.get()), args->value()...);
                              };
#ifndef NDEBUG
                              local_position.assign(position.begin(), position.end());
#endif
                        } else {
                              reset(false);
                              modify(*const_cast<T*>(value_ptr.get()), *(args->value_ptr)...);
                        }
                  }
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
            struct LazyBase {};

            /**
             * 将lazy core的指针封装为单独的类, 为了使用方便, 比如运算符重载
             *
             * \tparam T 被lazy化的类型
             */
            template<class T>
            struct Lazy : LazyBase {
                  using type = T;

                  /**
                   * 唯一的member, std::shared_ptr<LazyCore<T>>, 本身这个类就是对他的封装
                   */
                  std::shared_ptr<LazyCore<T>> core;

                  /**
                   * 无论如何, 这个指针不会为空的, 新建一个空的lazy core
                   */
                  Lazy() : core(std::make_shared<LazyCore<T>>()) {}

                  /**
                   * \see LazyCore<T>::reset
                   *
                   * \note 在set point value后变更value后可以调用reset来更新图
                   */
                  Lazy<T> reset(bool reset_itself=false) {
                        core->reset(reset_itself);
                        return *this;
                  }
                  /**
                   * \see LazyCore<T>::set_point_value
                   */
                  Lazy<T> set_point_value(const T* ptr) {
                        core->set_point_value(ptr);
                        return *this;
                  }

                  /**
                   * \see LazyCore<T>::set_value
                   */
                  template<class... Args>
                  Lazy<T> set_value(Args&&... args) {
                        core->set_value(std::forward<Args>(args)...);
                        return *this;
                  }

                  /**
                   * \see LazyCore<T>::set_maybe_point_func
                   */
                  template<class Func, class... Args>
                  Lazy<T> set_maybe_point_func(Func&& f, Lazy<Args>... args) {
                        core->set_maybe_point_func(std::move(f), args.core...);
                        return *this;
                  }

                  /**
                   * \see LazyCore<T>::set_point_func
                   */
                  template<class Func, class... Args>
                  Lazy<T> set_point_func(Func&& f, Lazy<Args>... args) {
                        core->set_point_func(std::move(f), args.core...);
                        return *this;
                  }

                  /**
                   * \see LazyCore<T>::set_func
                   */
                  template<class Func, class... Args>
                  Lazy<T> set_func(Func&& f, Lazy<Args>... args) {
                        core->set_func(std::move(f), args.core...);
                        return *this;
                  }

                  /**
                   * \see LazyCore<T>::update
                   */
                  template<class Func, class... Args>
                  Lazy<T> update(Func&& f, Lazy<Args>... args) {
                        core->update(std::move(f), args.core...);
                        return *this;
                  }

                  /**
                   * \see LazyCore<T>::value
                   */
                  const T& value() const {
                        return core->value();
                  }

                  /**
                   * \see LazyCore<T>::pop
                   */
                  T pop() const {
                        return core->pop();
                  }

                  /**
                   * 从Lazy<T>的唯一member构建Lazy
                   */
                  explicit Lazy(std::shared_ptr<LazyCore<T>> c) : core(c) {}
                  /**
                   * 从Lazy<T>或者Lazy<T>的子类构建Lazy
                   */
                  template<class C, class = std::enable_if_t<std::is_base_of_v<Lazy<T>, C>>>
                  explicit Lazy(C c) : core(c.core) {}
                  /**
                   * 从value构建Lazy
                   */
                  explicit Lazy(const T& c) : core(std::make_shared<LazyCore<T>>()) {
                        set_value(c);
                  }
                  /**
                   * 从可移动的value构建Lazy
                   */
                  explicit Lazy(T&& c) : core(std::make_shared<LazyCore<T>>()) {
                        set_value(std::move(c));
                  }
                  /**
                   * 从value的指针构建Lazy
                   */
                  explicit Lazy(const T* c) : core(std::make_shared<LazyCore<T>>()) {
                        set_point_value(c);
                  }
            };

            template<class T, class = std::enable_if_t<!std::is_base_of_v<LazyBase, T>>>
            Lazy(T)->Lazy<T>;

            template<class T, class = std::enable_if_t<!std::is_base_of_v<LazyBase, T>>>
            Lazy(T*)->Lazy<T>;

            template<class T>
            Lazy(std::shared_ptr<LazyCore<T>>)->Lazy<T>;

            template<class T>
            Lazy(Lazy<T>)->Lazy<T>;

            template<class T, class = std::enable_if_t<std::is_base_of_v<LazyBase, T>>>
            Lazy(T)->Lazy<typename T::type>;
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
            /**
             * 对张量的lazy化
             *
             * \tparam N 张量的类型, 最简单的情况是N=Node, 但是也可以是高阶的lazy, 比如
             * \code{.cpp}
             * template<class Base>
             * using N = LazyNode<Node, Base>;
             *
             * using ResultType = LazyNode<N, double>;
             * \endcode
             * \tparam Base 被lazy化的张量的基类
             * \see node::Node
             * \see lazy::Lazy
             */
            template<template<class> class N = Node, class Base = double>
            struct LazyNode : Lazy<N<Base>> {
                  using Lazy<N<Base>>::set_point_value;
                  using Lazy<N<Base>>::set_value;
                  using Lazy<N<Base>>::set_maybe_point_func;
                  using Lazy<N<Base>>::set_point_func;
                  using Lazy<N<Base>>::set_func;
                  using Lazy<N<Base>>::update;
                  using Lazy<N<Base>>::value;
                  using Lazy<N<Base>>::pop;

                  LazyNode() : Lazy<N<Base>>() {}
                  template<class T1 = std::vector<Legs>, class T2 = std::vector<Size>>
                  explicit LazyNode(T1&& legs, T2&& dims) : Lazy<N<Base>>() {
                        set_value(std::forward<T1>(legs), std::forward<T2>(dims));
                  }
                  template<class Arg0, class... Args>
                  explicit LazyNode(Arg0&& arg0, Args&&... args) {
                        set_value(std::forward<Arg0>(arg0), std::forward<Args>(args)...);
                  }
                  /**
                   * \note 需要没有const限定, 不然会被决议到上面那个函数上
                   */
                  explicit LazyNode(N<Base>* ptr) {
                        set_point_value(ptr);
                  }
                  explicit LazyNode(Lazy<N<Base>> src) : Lazy<N<Base>>(src) {}

                  template<class Generator>
                  LazyNode<N, Base>& set(Generator&& setter) & {
                        lazy::name_scope name_scope("set");
                        update([setter(std::forward<Generator>(setter))](N<Base>& node) { node.set(setter); });
                        return *this;
                  }

                  template<class Generator>
                  LazyNode<N, Base>&& set(Generator&& setter) && {
                        return std::move(set(std::forward<Generator>(setter)));
                  }

                  LazyNode<N, Base> legs_rename(const std::map<Legs, Legs>& dict) const {
                        lazy::name_scope name_scope("legs_rename");
                        auto res = LazyNode<N, Base>();
                        res.set_func([=](const N<Base>& node) { return node.legs_rename(dict); }, *this);
                        return res;
                  }

                  template<class Base2>
                  LazyNode<N, Base2> to() const {
                        lazy::name_scope name_scope("to");
                        if constexpr (std::is_same_v<Base, Base2>) {
                              return *this;
                        } else {
                              auto res = LazyNode<N, Base2>();
                              res.set_func([](const N<Base>& node) { return node.template to<Base2>(); }, *this);
                              return res;
                        }
                  }

                  template<int n>
                  LazyNode<N, Base> norm() const {
                        lazy::name_scope name_scope("norm");
                        auto res = LazyNode<N, Base>();
                        res.set_func([](const N<Base>& node) { return node.template norm<n>(); }, *this);
                        return res;
                  }

                  LazyNode<N, Base> transpose(const std::vector<Legs>& new_legs) const {
                        lazy::name_scope name_scope("transpose");
                        auto res = LazyNode<N, Base>();
                        res.set_func([=](const N<Base>& node) { return node.transpose(new_legs); }, *this);
                        return res;
                  }

                  struct svd_res {
                        LazyNode<N, Base> U;
                        LazyNode<N, real_base_t<Base>> S;
                        LazyNode<N, Base> V;
                  }; // struct svd_res

                  svd_res
                  svd(const std::vector<Legs>& input_u_legs,
                      const Legs& new_u_legs,
                      const Legs& new_v_legs,
                      const Rank& cut = -1) const {
                        lazy::name_scope name_scope("svd");
                        auto tmp = Lazy<typename N<Base>::svd_res>();
                        tmp.set_func(
                              [=](const N<Base>& node) { return node.svd(input_u_legs, new_u_legs, new_v_legs, cut); },
                              *this);
                        auto res = svd_res();
                        res.U.set_point_func([](const typename N<Base>::svd_res& r) { return &r.U; }, tmp);
                        res.S.set_point_func([](const typename N<Base>::svd_res& r) { return &r.S; }, tmp);
                        res.V.set_point_func([](const typename N<Base>::svd_res& r) { return &r.V; }, tmp);
                        return res;
                  }

                  struct qr_res {
                        LazyNode<N, Base> Q;
                        LazyNode<N, Base> R;
                  }; // struct qr_res

                  qr_res
                  qr(const std::vector<Legs>& input_q_legs, const Legs& new_q_legs, const Legs& new_r_legs) const {
                        lazy::name_scope name_scope("qr");
                        auto tmp = Lazy<typename N<Base>::qr_res>();
                        tmp.set_func(
                              [=](const N<Base>& node) { return node.qr(input_q_legs, new_q_legs, new_r_legs); },
                              *this);
                        auto res = qr_res();
                        res.Q.set_point_func([](const typename N<Base>::qr_res& r) { return &r.Q; }, tmp);
                        res.R.set_point_func([](const typename N<Base>::qr_res& r) { return &r.R; }, tmp);
                        return res;
                  }

                  qr_res
                  rq(const std::vector<Legs>& input_r_legs, const Legs& new_r_legs, const Legs& new_q_legs) const {
                        lazy::name_scope name_scope("rq");
                        auto tmp = Lazy<typename N<Base>::qr_res>();
                        tmp.set_func(
                              [=](const N<Base>& node) { return node.rq(input_r_legs, new_r_legs, new_q_legs); },
                              *this);
                        auto res = qr_res();
                        res.Q.set_point_func([](const typename N<Base>::qr_res& r) { return &r.Q; }, tmp);
                        res.R.set_point_func([](const typename N<Base>::qr_res& r) { return &r.R; }, tmp);
                        return res;
                  }

                  static LazyNode<N, Base> contract(
                        const LazyNode<N, Base>& node1,
                        const LazyNode<N, Base>& node2,
                        const std::vector<Legs>& legs1,
                        const std::vector<Legs>& legs2) {
                        lazy::name_scope name_scope("contract");
                        auto res = LazyNode<N, Base>();
                        res.set_func(
                              [=](const N<Base>& node1, const N<Base>& node2) {
                                    return N<Base>::contract(node1, node2, legs1, legs2);
                              },
                              node1,
                              node2);
                        return res;
                  }

                  LazyNode<N, Base> partial_contract(
                        const N<Base>& node,
                        const std::vector<Legs>& legs1,
                        const std::vector<Legs>& legs2) {
                        lazy::name_scope name_scope("partial_contract");
                        auto res = LazyNode<N, Base>();
                        res.set_func(
                              [=, &node](const N<Base>& self) { return N<Base>::contract(self, node, legs1, legs2); },
                              *this);
                        return res;
                  }

                  LazyNode<N, Base> multiple(const LazyNode<N, Base>& other, const Legs& position) const {
                        lazy::name_scope name_scope("multiple");
                        auto res = LazyNode<N, Base>();
                        res.set_maybe_point_func(
                              [=](const N<Base>& self, const N<Base>& other) {
                                    struct maybe_res {
                                          const N<Base>* value_point;
                                          bool own_flag;
                                    };
                                    if (std::find(self.legs.begin(), self.legs.end(), position) == self.legs.end()) {
                                          return maybe_res{&self, false};
                                    } else {
                                          return maybe_res{new N<Base>(self.multiple(other, position)), true};
                                    }
                              },
                              *this,
                              other);
                        return res;
                  }
            };

            template<template<class> class T, template<class> class U>
            struct is_same_template : std::false_type {};

            template<template<class> class T>
            struct is_same_template<T, T> : std::true_type {};

            template<template<class> class N, class Base, class = std::enable_if_t<!is_same_template<N, Lazy>::value>>
            LazyNode(N<Base>)->LazyNode<N, Base>;

            template<template<class> class N, class Base>
            LazyNode(Lazy<N<Base>>)->LazyNode<N, Base>;

            template<template<class> class N, class Base>
            LazyNode(LazyNode<N, Base>)->LazyNode<N, Base>;

            template<template<class> class N, class T>
            std::ostream& operator<<(std::ostream& out, const LazyNode<N, T>& value) {
                  return out << value.value();
            }
      } // namespace lazy_node
      using lazy_node::LazyNode;

      namespace lazy_node {
            template<template<class> class N, class T>
            LazyNode<N, T> operator+(const LazyNode<N, T>& a) {
                  return lazy::operator+(a);
            }

            template<template<class> class N, class T>
            LazyNode<N, T> operator-(const LazyNode<N, T>& a) {
                  return lazy::operator-(a);
            }

#define DEF_OP(OP)                                                           \
      template<template<class> class N, class A, class B>                    \
      LazyNode<N, A>& OP(LazyNode<N, A>& a, const LazyNode<N, B>& b) {       \
            auto tmp = TAT::Lazy(a);                                         \
            lazy::OP(tmp, TAT::Lazy(b));                                     \
            return a;                                                        \
      }                                                                      \
      template<                                                              \
            template<class>                                                  \
            class N,                                                         \
            class T,                                                         \
            class B,                                                         \
            class = std::enable_if_t<!std::is_base_of_v<lazy::LazyBase, B>>> \
      LazyNode<N, T>& OP(LazyNode<N, T>& a, const B& b) {                    \
            auto tmp = TAT::Lazy(a);                                         \
            lazy::OP(tmp, b);                                                \
            return a;                                                        \
      }

            DEF_OP(operator*=)
            DEF_OP(operator/=)
            DEF_OP(operator+=)
            DEF_OP(operator-=)
#undef DEF_OP

#define DEF_OP(OP)                                                           \
      template<template<class> class N, class A, class B>                    \
      auto OP(const LazyNode<N, A>& a, const LazyNode<N, B>& b) {            \
            auto res = lazy::OP(TAT::Lazy(a), TAT::Lazy(b));                 \
            return LazyNode<N, std::common_type_t<A, B>>(res);               \
      }                                                                      \
      template<                                                              \
            template<class>                                                  \
            class N,                                                         \
            class T,                                                         \
            class B,                                                         \
            class = std::enable_if_t<!std::is_base_of_v<lazy::LazyBase, B>>> \
      auto OP(const LazyNode<N, T>& a, const B& b) {                         \
            return LazyNode<N, T>(lazy::OP(TAT::Lazy(a), b));                \
      }                                                                      \
      template<                                                              \
            template<class>                                                  \
            class N,                                                         \
            class T,                                                         \
            class B,                                                         \
            class = std::enable_if_t<!std::is_base_of_v<lazy::LazyBase, B>>> \
      auto OP(const B& a, const LazyNode<N, T>& b) {                         \
            return LazyNode<N, T>(lazy::OP(a, TAT::Lazy(b)));                \
      }

            DEF_OP(operator*)
            DEF_OP(operator/)
            DEF_OP(operator+)
            DEF_OP(operator-)
#undef DEF_OP
      } // namespace lazy_node
} // namespace TAT

#endif // TAT_HPP_
