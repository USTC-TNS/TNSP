/**
 * \file lazy.hpp
 *
 * Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#pragma once
#ifndef LAZY_HPP
#define LAZY_HPP

#ifndef __cplusplus
#error only work for c++
#endif

#ifdef _MSVC_LANG
#if _MSVC_LANG < 201703L
#error require c++17 or later
#endif
#else
#if __cplusplus < 201703L
#error require c++17 or later
#endif
#endif

#include <any>
#include <functional>
#include <list>
#include <memory>
#include <tuple>
#include <type_traits>

namespace lazy {
   /**
    * \defgroup Lazy
    * @{
    */
   struct lazy_base : std::enable_shared_from_this<lazy_base> {
    private:
      /**
       * 重置当前节点
       */
      virtual void release() = 0;

    protected:
      /**
       * 重置当前节点和下游
       */
      void unset() {
         release();
         for (auto iter = downstream.begin(); iter != downstream.end();) {
            if (auto ptr = iter->lock(); ptr) {
               ptr->unset();
               ++iter;
            } else {
               iter = downstream.erase(iter);
            }
         }
      }

    public:
      /**
       * 下游节点的列表
       */
      std::list<std::weak_ptr<lazy_base>> downstream;

      virtual ~lazy_base() {}
   };

   struct data_lazy_base : lazy_base {
    public:
      /**
       * 从一个any中load出数据
       */
      virtual void load(std::any) = 0;
      /**
       * dump出一个any数据
       */
      virtual std::any dump() = 0;
   };

   // path的type可以有cv ref
   template<typename Type>
   struct path : lazy_base {
    private:
      std::function<Type()> function;

      void release() override {}

    public:
      decltype(auto) get() {
         return function();
      }

      path(std::function<Type()>&& f) : function(std::move(f)) {}
      path() = delete;
      path(const path&) = delete;
      path(path&&) = delete;
      path& operator=(const path<Type>&) = delete;
      path& operator=(path<Type>&&) = delete;
   };

   template<typename Type>
   struct root : data_lazy_base {
    private:
      std::shared_ptr<const Type> value;

      void release() override {
         value.reset();
      }

    public:
      const Type& get() {
         return *value;
      }

      void load(std::any v) override {
         value = std::any_cast<std::shared_ptr<const Type>>(v);
      }

      std::any dump() override {
         return std::any(value);
      }

      using lazy_base::unset;

      void set(Type&& v) {
         unset();
         value.reset(new Type(std::move(v)));
      }
      void set(const Type& v) {
         unset();
         value.reset(new Type(v));
      }

      root() = default;
      root(const root&) = delete;
      root(root&&) = delete;
      root& operator=(const root<Type>&) = delete;
      root& operator=(root<Type>&&) = delete;
   };

   template<typename Type>
   struct node : data_lazy_base {
    private:
      std::function<Type()> function;
      using RealType = std::remove_cv_t<std::remove_reference_t<Type>>;
      std::shared_ptr<const RealType> value;

      void release() override {
         value.reset();
      }

      void set(RealType&& v) {
         unset();
         value.reset(new RealType(std::move(v)));
      }
      void set(const RealType& v) {
         unset();
         value.reset(new RealType(v));
      }

    public:
      const auto& get() {
         if (!bool(value)) {
            set(function());
         }
         return *value;
      }

      node(std::function<Type()>&& f) : function(std::move(f)) {}
      node() = delete;
      node(const node&) = delete;
      node(node&&) = delete;
      node& operator=(const node<Type>&) = delete;
      node& operator=(node<Type>&&) = delete;

      void load(std::any v) override {
         value = std::any_cast<std::shared_ptr<const RealType>>(v);
      }

      std::any dump() override {
         return std::any(value);
      }
   };
   // graph

   using Snapshot = std::list<std::tuple<std::weak_ptr<data_lazy_base>, std::any>>;
   /**
    * Graph用于restore各个node中的数据
    */
   struct Graph {
      // std::any will store std::shared_ptr<T> for any T
      std::list<std::weak_ptr<data_lazy_base>> nodes;

      Snapshot dump() {
         auto result = Snapshot();
         for (auto iter = nodes.begin(); iter != nodes.end();) {
            if (auto ptr = iter->lock(); ptr) {
               result.emplace_back(*iter, ptr->dump());
               ++iter;
            } else {
               iter = nodes.erase(iter);
            }
         }
         return result;
      }
      void load(Snapshot& snapshot) {
         for (auto iter = snapshot.begin(); iter != snapshot.end();) {
            auto& [weak, value] = *iter;
            if (auto ptr = weak.lock(); ptr) {
               ptr->load(value);
               ++iter;
            } else {
               iter = snapshot.erase(iter);
            }
         }
      }

      template<typename T>
      void add(T value) {
         nodes.push_back(std::dynamic_pointer_cast<data_lazy_base>(value));
      }
   };

   inline Graph default_graph;
   inline Graph* active_graph = &default_graph;
   inline Graph& current_graph() {
      return *active_graph;
   }
   inline void use_graph(Graph& graph = default_graph) {
      active_graph = &graph;
   }

   // helper function

   template<typename Function, typename... Args>
   auto function_wrapper(Function&& function, Args&... args) {
      return std::function([=] {
         return function(args->get()...);
      });
   }

   template<typename Type>
   auto Root() {
      using RealType = std::remove_cv_t<std::remove_reference_t<Type>>;
      auto result = std::make_shared<root<RealType>>();
      current_graph().add(result);
      return result;
   }

   template<typename Type>
   auto Root(Type&& v) {
      using RealType = std::remove_cv_t<std::remove_reference_t<Type>>;
      auto result = std::make_shared<root<RealType>>();
      result->set(std::forward<Type>(v));
      current_graph().add(result);
      return result;
   }

   template<typename Function, typename... Args>
   auto Node(Function&& function, Args&... args) {
      // 应该返回一个值而非引用, 否则无法放入shared_ptr中
      auto f = function_wrapper(function, args...);
      using Type = typename decltype(f)::result_type;
      auto result = std::make_shared<node<Type>>(std::move(f));
      (args->downstream.push_back(result->shared_from_this()), ...);
      current_graph().add(result);
      return result;
   }

   template<typename Function, typename... Args>
   auto Path(Function&& function, Args&... args) {
      // 返回啥都行
      auto f = function_wrapper(function, args...);
      using Type = typename decltype(f)::result_type;
      auto result = std::make_shared<path<Type>>(std::move(f));
      (args->downstream.push_back(result->shared_from_this()), ...);
      return result;
   }
   /**@}*/
} // namespace lazy

#endif
