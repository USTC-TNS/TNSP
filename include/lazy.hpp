/**
 * \file lazy.hpp
 *
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

#ifndef TAT_LAZY_HPP_
#define TAT_LAZY_HPP_

#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <type_traits>

/**
 * 一套lazy框架, 避免重复计算与复杂的临时变量复用的处理
 *
 * 实现lazy框架的三个层次
 *
 * 1. lazy_core_base 管理依赖流
 * 2. lazy_core 值和函数的容器, 因为他需要以shared_ptr形式存在, 所以单独一层
 * 3. Lazy 设置函数和值的接口
 *
 * 设计思路如下:
 * Lazy是一个Functor, 将类型T映射为Lazy<T>, 需要的操作是如下
 *
 * - value : lazy T -> T
 * - Lazilize : T -> lazy T
 * - Lazilize : (a -> b) -> (lazy a -> lazy b)
 *
 * 适配c++, 最后一个变成了 Lazilize(Func func, Args... args).
 * 其中Args也可以不是lazy, 不然每次Lazylize太麻烦了,
 * 如果是lazy的话需要通过一个exec封装标记本映射中需要解开lazy,
 * 虽然这一点可以通过判断是否是Lazy类型来决定, 但是涉及到高阶lazy的时候会变得不可控.
 * 当然args甚至可能是std::cref, 用来避免复制.
 * 最后两个运算都在构造函数中实现.
 *
 * lazy的graph是需要变化的, 为了方便的变化, 与其不停set, 不如多个replace操作, 及重载operator=.
 * 另外, 本框架不允许inplace操作.
 */
namespace lazy {
      /**
       * 获取类型T的名称, 调试时使用
       *
       * 例如
       * \code{.cpp}
       * std::cout << type_name<const int&>() << "\n";
       * \endcode
       * 输出为\code const int & \endcode
       *
       * \tparam T 将要返回名称字符串的类型
       * \returns T类型的名称字符串
       * \note 构造string_view的时候用的偏移量是编译器相关的, 换一个编译器不能使用, 现在是clang版本的

       */
      template<class T>
      constexpr std::string_view type_name() {
            std::string_view p = __PRETTY_FUNCTION__;
            return std::string_view(p.data() + 40, p.size() - 40 - 1);
      }

      /**
       * 用于处理所有权事先不确定的智能指针, 类似std::unique_ptr, 但是可以标记不拥有所有权, 使得析构时不delete管理的指针
       */
      template<class T>
      struct maybe_ptr {
            using element_type = T;
            using pointer = T*;
            using self = maybe_ptr<T>;

            /**
             * 所管理的指针, 所有权是否拥有通过flag标记
             */
            pointer ptr;
            /**
             * 指针指向的对象所有权信息, true表示拥有所有权, false表示不拥有
             */
            bool flag;

            /**
             * 与free一起构成完备正交的两个操作, 设置此maybe_ptr的值, 包括指针和所有权, 另外有一个不正交的clear操作,
             * 相当于set(nullptr)
             *
             * \param p 新的maybe_ptr所管理的指针
             * \param f 所管理的指针的所有权, 如果p=nullptr, 所有权将被强制设置为false
             * \see free()
             * \see clear()
             */
            void set(pointer p = nullptr, bool f = true) {
                  ptr = p;
                  flag = p ? f : false;
            }
            /**
             * 和set一起构成完备正交的两个操作, 释放内存, 但不更新自己的值, 一般后面会跟随clear或者set
             * \see set()
             */
            void free() {
                  if (flag) {
                        delete ptr;
                  }
            }
            /**
             * 与set不正交, 相当于set(nullptr)单独写出来是为了少一个判断而已
             */
            void clear() {
                  ptr = nullptr;
                  flag = false;
            }

            /**
             * 新建一个maybe_ptr, 直接调用set设置maybe_ptr的两个member
             *
             * \param p 新的maybe_ptr所管理的指针
             * \param f 所管理的指针的所有权, 如果p=nullptr, 所有权将被强制设置为false
             */
            maybe_ptr(pointer p = nullptr, bool f = true) {
                  set(p, f);
            }

            /**
             * 直接free即可, 虽然处于非法状态, 但不需要再clear, 因为是析构函数
             */
            ~maybe_ptr() {
                  free();
            }

            /**
             * 移动一个maybe_ptr, set自己后需要将原来的maybe_ptr clear
             */
            maybe_ptr(maybe_ptr<T>&& other) {
                  set(other.ptr, other.flag);
                  other.clear();
            }

            /**
             * 移动赋值一个maybe_ptr, 需要先reset自己, 然后clear原maybe_ptr保证其合法状态
             */
            self& operator=(maybe_ptr<T>&& other) {
                  free();
                  set(other.ptr, other.flag);
                  other.clear();
                  return *this;
            }

            /**
             * 获取被管理的指针
             */
            pointer get() {
                  return ptr;
            }
            /**
             * 判断是否含有被管理的指针, 与所有权无关, 判断所有权的话使用maybe_ptr.flag
             */
            operator bool() {
                  return ptr;
            }

            /**
             * 获得指针所指的对象引用
             */
            element_type& operator*() {
                  return *ptr;
            }
            /**
             * 实现operator->使之和指针更像
             */
            pointer operator->() {
                  return ptr;
            }
      };

      /**
       * lazy框架的第一层, 处理流的依赖信息, 但是reset因为牵扯到指针类型, 所以放在了下一层中实现,
       * 在这里以虚函数形式存在
       */
      struct lazy_core_base {
            /**
             * 使自己的value清空, 并传递给下游
             *
             * \param reset_itself 是否清空自己, 手动更改自身的值后应当设置会false并调用reset
             * \see Lazy<T>::fresh()
             */
            virtual void reset(bool reset_itself = true) = 0;
            /**
             * 子类的析构函数应该调用dump_upstream
             *
             * \see dump_upstream()
             */
            virtual ~lazy_core_base() = default;

            /**
             * 依赖下游集合, 列表中的lazy依赖于本lazy
             */
            std::set<lazy_core_base*> downstream;
            /**
             * 依赖上游集合, 本lazy依赖于列表中的lazy
             */
            std::set<lazy_core_base*> upstream;

            /**
             * 与自己的上游断开关系, 将自己在上游的下游set中清除, 在重置或者析构自己前需要调用
             *
             * \param clearit 断开链接后也清空自己的upstream集合, 在Lazy移动中因为需要知道原Lazy的上游,
             * 所以原Lazy断开上游后不应清空自己upstream, 随后会被移动
             * \note 此操作必须在子类析构函数中调用, 否则会出现先调用两个上下游对象的子类析构函数,
             * 再调用父类析构函数的情况, 使得下游的dump_upstream还没调用的时候, 上有对象便开始析构,
             * 无法保证下游先调用dump_upstream
             * \see Lazy::operator=(self&& other)
             */
            void dump_upstream(bool clearit = true) {
                  // 将自己从每个上游的下游列表中去除
                  for (const auto& us : upstream) {
                        // 如果this在downstream中重复依赖, 仍然不会出问题, 因为是集合
                        us->downstream.erase(this);
                  }
                  // 如果标记了clearit(默认为true), 则clear掉upstream
                  if (clearit) {
                        upstream.clear();
                  }
            }
      };

      /**
       * lazy框架的第二层, 在上一层基础上加上了类型信息, 作为缓存和函数的容器, 并未提供多少方法, 除了一个set_value
       */
      template<class T>
      struct lazy_core : lazy_core_base {
            using element_type = T;

            /**
             * 存储缓存的指针, 所有权未知, 故使用maybe_ptr
             *
             * \note 本类中不会改变element_type的值, 且可能出现需要将value指向一个常量的情况, 所以为const
             * \see maybe_ptr
             */
            maybe_ptr<const element_type> ptr;
            /**
             * 当无缓存时, 将调用此函数设置本对象的ptr
             *
             * \note 因为func可能在不同Lazy之间移动,
             * 所以调用的时候需要传入一个被设置的Lazy的指针或者lazy_core的指针
             */
            std::function<void(lazy_core<element_type>*)> func;

            void reset(bool reset_itself = true) override {
                  // 如果没有ptr的话, 说明此lazy尚未计算, 直接跳过即可
                  if (ptr) {
                        // 判断是否要reset自己, 如果需要执行free和clear两个操作, clear是为了下次判断bool(ptr),
                        if (reset_itself) {
                              ptr.free();
                              ptr.clear();
                        }
                        // 递归reset所有下游
                        for (const auto& ds : downstream) {
                              ds->reset();
                        }
                  }
            }

            ~lazy_core() override {
                  dump_upstream();
            }

            /**
             * 设置此lazy_core的值
             *
             * \tparam Arg 设置值用的参数的类型, 可以是maybe_ptr<T>, T&, T
             * \param arg 设置值用到参数
             * \note 如果是T的话, 那么ptr拥有所有权, 如果是T&则不拥有所有权, 如果是maybe_ptr,
             * 所有权由自身表示
             */
            template<class Arg>
            void set_value(Arg&& arg) {
                  if constexpr (std::is_same_v<Arg, maybe_ptr<const T>> || std::is_same_v<Arg, maybe_ptr<T>>) {
                        // Arg = maybe_ptr<T>
                        ptr.set(arg.ptr, arg.flag);
                        arg.clear();
                  } else if constexpr (std::is_same_v<Arg, const T&> || std::is_same_v<Arg, T&>) {
                        // Arg = T&
                        ptr.set(&arg, false);
                        // std::clog << "init by lvalue\n";
                  } else {
                        // Arg = T
                        static_assert(std::is_same_v<Arg, const T> || std::is_same_v<Arg, T>);
                        ptr.set(new T(std::move(arg)), true);
                        // std::clog << "init by rvalue\n";
                  }
            }
      };

      // 申明
      struct lazy_base;
      template<class T>
      struct Lazy;
      template<class T>
      struct Exec;
      template<class T>
      struct Hold;

      template<class T>
      struct remove_cvref {
            using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
      };
      /**
       * 返回类型T本身, 但是去除cv限定和引用
       */
      template<class T>
      using remove_cvref_t = typename remove_cvref<T>::type;

      template<class T>
      struct remove_cvref_or_get_first : remove_cvref<T> {};
      template<class T>
      struct remove_cvref_or_get_first<maybe_ptr<T>> : remove_cvref<T> {};
      /**
       * 删除cv和ref, 但是如果是maybe_ptr<T>的形式, 则返回删除cv和ref后的T
       *
       * \see lazy_core<T>::set_value(Arg&& arg)
       */
      template<class T>
      using remove_cvref_or_get_first_t = typename remove_cvref_or_get_first<T>::type;

      /**
       * 判断是否是lazy, 通过父类是否是lazy_base实现
       */
      template<class T>
      constexpr bool is_lazy_v = std::is_base_of_v<lazy_base, T>;

      template<class T>
      struct is_exec : std::false_type {};
      template<class U>
      struct is_exec<Exec<U>> : std::true_type {};
      /**
       * 判断是否是exec handle的包装
       */
      template<class T>
      constexpr bool is_exec_v = is_exec<T>::value;

      template<class T>
      struct is_hold : std::false_type {};
      template<class U>
      struct is_hold<Hold<U>> : std::true_type {};
      /**
       * 判断是否是hold handle的包装
       */
      template<class T>
      constexpr bool is_hold_v = is_hold<T>::value;

      template<class T>
      struct is_reference_wrapper : std::false_type {};
      template<class U>
      struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};
      /**
       * 判断类型是否是std::cref的包装, 其实std::ref也可以, 不过本框架中未使用
       */
      template<class T>
      constexpr bool is_reference_wrapper_v = is_reference_wrapper<T>::value;

      template<class T, class = void>
      struct is_callable : std::false_type {};
      template<class T>
      struct is_callable<T, std::void_t<decltype(std::function(std::declval<T>()))>> : std::true_type {};
      /**
       * 判断类型是否可以被调用, 如函数或者拥有operator()的类型, 注意成员函数并不被判断为可调用
       *
       * \note 通过是否可以转化为std::function来实现的
       */
      template<class T>
      constexpr bool is_callable_v = is_callable<T>::value;

      template<class T>
      struct is_function : std::bool_constant<is_callable<T>::value || std::is_member_function_pointer<T>::value> {};
      /**
       * 判断类型是否是函数类型, 返回is_callable_v和is_member_function_pointer_v的或
       *
       * \see is_callable_v
       */
      template<class T>
      constexpr bool is_function_v = is_function<T>::value;

      /**
       * Lazy的父类, 用来判断是否是Lazy, 本身为空类型
       */
      struct lazy_base {};

      /**
       * exec handle, 本身只是对Lazy做了一个包装, 没有任何其他信息,
       * 通过自己本身的类型信息告诉Lazy在计算的时候将自己解封
       *
       * \see exec(T arg)
       */
      template<class T>
      struct Exec {
            using element_type = T;
            Lazy<T> lazy;
            Exec(const Lazy<T>& arg) : lazy(arg) {}
      };
      template<class T>
      Exec(const Lazy<T>&)->Exec<T>;

      /**
       * hold handle, 用来在高阶lazy的四则运算中区分阶数用
       *
       * \see exec(T arg)
       */
      template<class T>
      struct Hold {
            using element_type = T;
            Lazy<T> lazy;
            Hold(const Lazy<T>& arg) : lazy(arg) {}
      };
      template<class T>
      Hold(const Lazy<T>&)->Hold<T>;

      /**
       * 将一个lazy转化为exec handle用来在func中进行计算
       *
       * \note 但如果输入时一个Hold(Lazy<T>)则返回Lazy<T>自身, 标记此参数不要进行计算, 在高阶lazy的四则运算中使用
       * \see Hold
       * \see Exec
       * \see hold(T arg)
       */
      template<class T>
      decltype(auto) exec(const T& arg) {
            if constexpr (is_lazy_v<T>) {
                  return Exec(arg);
            } else if constexpr (is_hold_v<T>) {
                  return arg.lazy;
            } else {
                  return arg;
            }
      }

      /**
       * hold一个lazy防止他在func中被计算
       *
       * \note 如果输入是Exec对象, 则去除外层的Exec包装, 虽然现在这个机制并没被使用
       * \see exec(T arg)
       */
      template<class T>
      decltype(auto) hold(const T& arg) {
            if constexpr (is_lazy_v<T>) {
                  return Hold(arg);
            } else if constexpr (is_exec_v<T>) {
                  return arg.lazy;
            } else {
                  return arg;
            }
      }

      /**
       * 在lazy调用的func中解包一个参数
       *
       * \tparam Arg 可以是cref<T>, exec<T>, 或者普通变量T
       * \param arg 被解包的参数
       *
       * \note 如果是cref<T>则获取引用, 如果是exec<T>则计算他, 其他情况看作普通变量返回自身
       */
      template<class Arg>
      decltype(auto) unwrap_lazy(const Arg& arg) {
            // std::clog << type_name<Arg>() << "\n";
            if constexpr (is_reference_wrapper_v<Arg>) {
                  return arg.get();
            } else if constexpr (is_exec_v<Arg>) {
                  return arg.lazy.value();
            } else {
                  return arg;
            }
      }

      /**
       * 建立上下游依赖关系, 但是上游类型不确定, 只有是exec类型时真的建立依赖
       *
       * \tparam T 上游目标的类型, 如果是exec<T>则链接上下游, 否则不做任何事
       * \tparam Base 下游lazy的element_type
       * \param up 上游目标, 可能是exec也可能不是, 不是exec的话则不做任何事
       * \param down 下游lazy
       */
      template<class T, class Base>
      void try_link_stream(const T& up, const Lazy<Base>& down) {
            if constexpr (is_exec_v<T>) {
                  up.lazy.core->downstream.insert(down.core.get());
                  down.core->upstream.insert(up.lazy.core.get());
            }
      }

      /**
       * lazy框架的第三层也是最上层, 对lazy_core包一层shared_ptr, 同时提供设置和使用func, ptr的接口
       */
      template<class T>
      struct Lazy : lazy_base {
            using element_type = T;
            using self = Lazy<element_type>;

            /**
             * 因为生存权不确定, 所以需要使用shared_ptr管理lazy
             */
            std::shared_ptr<lazy_core<element_type>> core;

            /**
             * 默认构造函数会新建一个shared_ptr<lazy_core>, 这个shared_ptr任何时候都不会变
             */
            Lazy() : core(std::make_shared<lazy_core<element_type>>()) {}
            // Lazy(const self&) = default;
            // Lazy(self&&) = default;

            /**
             * 自己到子类的转换
             */
            template<class Arg, class = std::enable_if_t<std::is_base_of_v<self, Arg> && !std::is_same_v<self, Arg>>>
            operator Arg() {
                  // return Arg(*this) 会调用自己
                  auto res = Arg();
                  res.core = core;
                  return res;
            }

            /**
             * 子类到自己的转换, 和两种类型的lazilize
             *
             * \note 构造的途径有
             *
             * - 给出一个自己的子类, 直接设定core
             * - 如果第一个参数是函数, 那么设置好上下游依赖, 并通过func方式设置Lazy
             * - 如果第一个参数不是函数, 那么直接构造出此Lazy的值, 但如果只有一个参数且是element_type的左值引用,
             * 那么value设置为指针
             */
            template<
                  class firstArg,
                  class... Args,
                  class = std::enable_if_t<!(sizeof...(Args) == 0 && std::is_base_of_v<self, firstArg>)>>
            Lazy(firstArg&& first_arg, Args&&... args) : Lazy() {
                  if constexpr (sizeof...(Args) == 0 && std::is_base_of_v<self, remove_cvref_t<firstArg>>) {
                        core = first_arg.core;
                  } else if constexpr (is_function_v<firstArg>) {
                        // 是函数调用方式的lazy创建方式, Args可能被std::cref包装, 可能是值, 也可能是lazy
                        // 是不是成员函数都可以用invoke
                        // 目前函数本身不会被wrap
                        // try_link_stream<firstArg>(first_arg, *this);
                        (..., try_link_stream(args, *this));
                        core->func = [=](lazy_core<element_type>* c) -> void {
                              // core的位置可能被移动, 所以作为参数而不是捕获
                              c->set_value(std::invoke(first_arg, unwrap_lazy(args)...));
                              // c->set_value(first_arg(unwrap_lazy(args)...));
                        };
                  } else {
                        // 参数可能是一个, 也可能是多个, 但是如果是单个本类型左值参数时, 应调用引用方式的set_value
                        if constexpr (sizeof...(Args) == 0 && std::is_same_v<element_type, remove_cvref_t<firstArg>>) {
                              // 只有一个参数, 可能是左值, 可能是右值, 如果是左值
                              // 实际上单个本类型右值也会走这个if, 但是无所谓
                              core->set_value(std::forward<firstArg>(first_arg));
                        } else {
                              // 不止一个参数, 或者一个参数不是本类型左值时, 使用构造函数
                              core->set_value(
                                    element_type(std::forward<firstArg>(first_arg), std::forward<Args>(args)...));
                        }
                  }
            }

            /**
             * 获得value的值, 如果不存在, 则通过func计算出
             */
            element_type& value() const {
                  if (!core->ptr) {
                        core->func(core.get());
                  }
                  return *const_cast<element_type*>(core->ptr.get());
            }

            /**
             * 获得value的值, 并移动走, 如果本身没有所有权, 可能会有危险
             *
             * \note 如果本身是通过直接设置value的方式创建的, 那么pop后就难以找到原来的值了,
             * 如果是设置指针的方式创建的, 会破坏其他人的数据, 所以此函数应该在通过func方式创建的Lazy中调用,
             * pop中间产物后, 上游reset无法传递到下游, 所以调用pop的lazy需要没有下游
             */
            element_type pop() const {
                  // assert(core->func);
                  // assert(core->downstream.size() == 0);
                  element_type res = std::move(value());
                  core->reset();
                  return res;
            }

            /**
             * 获得value的指针, 如果不存在, 则通过func计算出
             */
            element_type* get() const {
                  return &value();
            }

            /**
             * reset下游依赖, 在手动改变value后应调用一次
             *
             * \note 这个函数不应该在通过func方式创建的Lazy中调用, 因为上游reset后会导致自己值被重置
             */
            void fresh() const {
                  // assert(!core->func);
                  core->reset(false);
            }

            /**
             * 移动赋值运算, 直接替换包括ptr和func的所有信息, 同时更新上下游依赖信息
             *
             * \note other应该无下游, 否则other的原下游会找不到上游
             */
            self& operator==(self&& other) {
                  // other需无下游, 否则应用复制赋值
                  // assert(other.core->downstream.size() == 0);
                  // 需要在dump_upstream之后set func
                  core->reset();
                  core->dump_upstream();
                  other.core->dump_upstream(false);
                  core->upstream = std::move(other.core->upstream);
                  for (auto& i : core->upstream) {
                        i->downstream.insert(core.get());
                  }
                  core->func = std::move(other.core->func);
                  core->ptr = std::move(other.core->ptr);
                  return *this;
            }
            /**
             * 复制赋值运算, 将自己替换为value指向other的value的lazy
             */
            self& operator==(const self& other) {
                  auto link = self();
                  try_link_stream(exec(other), link);
                  link.core->func = [=](lazy_core<element_type>* c) -> void { c->set_value(other.value()); };
                  return *this == std::move(link);
                  // operator>>(self([](element_type& v) -> element_type& { return v; }, exec(other)));
                  // return *this;
            }

            friend std::ostream& operator<<(std::ostream& out, const self& self) {
                  return out << self.value();
            }
      };

      template<class T>
      Lazy(T &&)->Lazy<remove_cvref_t<T>>;
      template<class F, class... Args, class = std::enable_if_t<is_function_v<F>>>
      Lazy(F&&, Args&&...)
            ->Lazy<remove_cvref_or_get_first_t<
                  std::invoke_result_t<F, std::invoke_result_t<decltype(unwrap_lazy<Args>), Args>...>>>;
      template<class Arg, class = std::enable_if_t<std::is_base_of_v<lazy_base, Arg>>>
      Lazy(Arg)->Lazy<typename Arg::element_type>;

      // 警告, 不同阶lazy混合时的exec可能会出问题, 需要使用hold
#define DEF_OP(OP, EVAL)                                                                               \
      template<class A, class B>                                                                       \
      auto OP(const Lazy<A>& a, const Lazy<B>& b) {                                                    \
            auto func = [](const A& a, const B& b) { return EVAL; };                                   \
            return Lazy<decltype(func(std::declval<A>(), std::declval<B>()))>(func, exec(a), exec(b)); \
      }                                                                                                \
      template<class T, class B, class = std::enable_if_t<!is_lazy_v<B>>>                              \
      auto OP(const Lazy<T>& a, const B& b) {                                                          \
            auto func = [](const T& a, const B& b) { return EVAL; };                                   \
            return Lazy<decltype(func(std::declval<T>(), std::declval<B>()))>(func, exec(a), b);       \
      }                                                                                                \
      template<class T, class A, class = std::enable_if_t<!is_lazy_v<A>>>                              \
      auto OP(const A& a, const Lazy<T>& b) {                                                          \
            auto func = [](const A& a, const T& b) { return EVAL; };                                   \
            return Lazy<decltype(func(std::declval<A>(), std::declval<T>()))>(func, a, exec(b));       \
      }

      DEF_OP(operator*, a* b)
      DEF_OP(operator/, a / b)
      DEF_OP(operator+, a + b)
      DEF_OP(operator-, a - b)
#undef DEF_OP

      template<class T, int n>
      struct high_lazy_helper {
            using type = Lazy<typename high_lazy_helper<T, n - 1>::type>;
      };
      template<class T>
      struct high_lazy_helper<T, 0> {
            using type = T;
      };
      /**
       * 高阶lazy
       *
       * \tparam T lazy的基础类型
       * \tparam n 高阶lazy的阶数, 如果n=0, 那么是T本身, 如果n=1, 那么为Lazy<T>
       */
      template<class T, int n>
      using HighLazy = typename high_lazy_helper<T, n>::type;
} // namespace lazy

#endif // TAT_LAZY_HPP_