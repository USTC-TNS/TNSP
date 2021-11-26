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

#if __cplusplus < 201703L
#error require c++17 or later
#endif

#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>

/**
 * Lazy framework
 */
namespace lazy {
   /**
    * Detail of lazy framwork
    */
   namespace detail {
      /**
       * Check if a type is reference wrapper
       */
      template<class>
      constexpr bool is_reference_wrapper_v = false;
      template<class U>
      constexpr bool is_reference_wrapper_v<std::reference_wrapper<U>> = true;

      template<typename T>
      struct node;

      /**
       * Check if a type is a node type
       */
      template<class>
      constexpr bool is_node_v = false;
      template<class U>
      constexpr bool is_node_v<node<U>> = true;

      /**
       * Unwrap object to get value
       *
       * \tparam Maybe one of const X&, std::reference_wrapper<const X> and node<X>
       */
      template<typename Arg>
      const auto& unwrap_arg(const Arg& arg) {
         if constexpr (is_node_v<Arg>) {
            // call arg() to calculate/select value from node<X>
            return arg();
         } else if constexpr (is_reference_wrapper_v<Arg>) {
            return arg.get();
         } else {
            return arg;
         }
      }

      /**
       * Apply func to tuple of arguments which may be wrapped
       *
       * \param func a function accept several parameters
       * \param args arguments which will be used to apply into function, it is bundled by tuple
       */
      template<typename Func, typename Args, std::size_t... Is>
      auto unwrap_apply(const Func& func, const Args& args, std::index_sequence<Is...>) {
         return func(unwrap_arg(std::get<Is>(args))...);
      }

      /**
       * Lazy core used to store dependence relations between lazy node
       */
      struct general_base : std::enable_shared_from_this<general_base> {
         /**
          * A list storing all downstream of this lazy node
          *
          * Any another depends on this lazy node should not change the current node,
          * So it should be mutable.
          */
         mutable std::list<std::weak_ptr<general_base>> downstream;

         /**
          * reset value of current node and all its downstreams
          *
          * for example, used before change root lazy node value
          */
         bool reset() const {
            if (unset()) {
               for (auto i = downstream.begin(); i != downstream.end();) {
                  if (auto ptr = i->lock(); ptr) {
                     ptr->reset();
                     ++i;
                  } else {
                     i = downstream.erase(i);
                  }
               }
               return true;
            } else {
               return false;
            }
         }

         /**
          * reset the current node only, called by reset
          *
          * \see reset
          * \return If unset really happened return true, if this node has already be unset before, nothing happend and return false
          */
         virtual bool unset() const = 0;
      };

      /**
       * Map type used to help copying lazy node graph
       *
       * The key is old node pointer and the value is new node pointer
       */
      using map_t = std::unordered_map<std::shared_ptr<general_base>, std::shared_ptr<general_base>>;

      /**
       * lazy node of specific type
       *
       * \tparam the type which the lazy node can get
       */
      template<typename T>
      struct specific_type : general_base {
         // A lazy node can be reset with certain value
         /**
          * Reset a lazy node with const lvalue, it will copy the value
          */
         virtual bool reset(const T&) const = 0;
         /**
          * Reset a lazy node with rvalue, it will move the value
          */
         virtual bool reset(T&&) const = 0;
         /**
          * Reset a lazy node with a reference, it will get the reference directly
          */
         virtual bool reset(std::reference_wrapper<const T> u) const = 0;

         /**
          * Get value of this lazy node
          *
          * It will always return a const reference
          */
         virtual const T& get() const = 0;

         /**
          * Copy the current lazy node
          *
          * \param map A map from old lazy node to new lazy node, it will update its upstream with it automatically
          * \return A wrapped node
          * \see node
          */
         virtual node<T> copy(const map_t& map) const = 0;
      };

      /**
       * A root node in lazy graph, its value must be set manually
       */
      template<typename T>
      struct given_value final : specific_type<T> {
         /**
          * Storing the given value, it may be empty or just value or reference to outside.
          */
         mutable std::variant<std::monostate, T, std::reference_wrapper<const T>> value;
         bool unset() const override {
            if (std::holds_alternative<std::monostate>(value)) {
               return false;
            } else {
               value = std::monostate();
               return true;
            }
         }
         bool reset(const T& u) const override {
            bool result = general_base::reset();
            value = u;
            return result;
         }
         bool reset(T&& u) const override {
            bool result = general_base::reset();
            value = std::move(u);
            return result;
         }
         bool reset(std::reference_wrapper<const T> u) const override {
            bool result = general_base::reset();
            value = u;
            return result;
         }
         const T& get() const override {
            if (auto v = std::get_if<T>(&value)) {
               return *v;
            } else if (auto v = std::get_if<std::reference_wrapper<const T>>(&value)) {
               return v->get();
            } else {
               throw std::runtime_error("try to get value from empty Root");
            }
         }
         node<T> copy(const map_t& map) const override;
      };

      /**
       * Base of lazy node set by specific function
       *
       * \see specific_function
       */
      template<typename T>
      struct specific_function_base : specific_type<T> {
         /**
          * Cahcing the calculated value, it may be empty or just value.
          */
         mutable std::optional<T> value;
         bool unset() const override {
            if (value.has_value()) {
               value.reset();
               return true;
            } else {
               return false;
            }
         }
         bool reset(const T& u) const override {
            throw std::runtime_error("try to set value of functional node");
         }
         bool reset(T&& u) const override {
            throw std::runtime_error("try to set value of functional node");
         }
         bool reset(std::reference_wrapper<const T> u) const override {
            throw std::runtime_error("try to set value of functional node");
         }
      };

      /**
       * lazy node set by specific function
       */
      template<typename T, typename Func, typename... Args>
      struct specific_function final : specific_function_base<T> {
         using specific_function_base<T>::value;
         /**
          * The function to get the value of this node
          *
          * It maybe std::reference_wrapper of function, but it can also be called directly
          */
         Func func;
         /**
          * A tuple storing arguments of the function
          *
          * It contains value, other node or std::reference_wrapper
          */
         std::tuple<Args...> args;
         template<typename Func0, typename... Args0>
         specific_function(Func0&& func0, Args0&&... args0) : func(std::forward<Func0>(func0)), args(std::forward<Args0>(args0)...) {}
         const T& get() const override {
            if (!value.has_value()) {
               value = unwrap_apply(func, args, std::index_sequence_for<Args...>());
            }
            return value.value();
         }
         node<T> copy(const map_t& map) const override;
      };

      /**
       * Wrapped lazy node
       *
       * With this, we can call node.reset(), node() instead of node->reset(), node->get()
       */
      template<typename T>
      struct node {
         using pointer_t = std::shared_ptr<specific_type<T>>;
         pointer_t pointer;
         node(pointer_t&& p) : pointer(std::move(p)) {}
         bool reset() const {
            return pointer->reset();
         };
         bool reset(const T& u) const {
            return pointer->reset(u);
         };
         bool reset(T&& u) const {
            return pointer->reset(std::move(u));
         };
         bool reset(std::reference_wrapper<const T> u) const {
            return pointer->reset(u);
         };
         const T& operator()() const {
            return pointer->get();
         }
      };

      /**
       * Add single downstream for the upstream node
       *
       * If the upstream is not a node, do nothing
       *
       * \param down it is shared_ptr of specific_function lazy node
       * \param up it is a wrapped node or just a normal object
       * \return if the upstream is really a node, return true, otherwise return false
       */
      template<typename Down, typename Up>
      bool try_add_downstream(const Down& down, const Up& up) {
         if constexpr (is_node_v<Up>) {
            up.pointer->downstream.push_back(down);
            return true;
         } else {
            return false;
         }
      }

      /**
       * Add downtreams for all upstreams of this current node
       *
       * \param t is a shared_ptr of specific_function lazy node
       */
      template<typename T, std::size_t... Is>
      void add_as_downstream(const T& t, std::index_sequence<Is...>) {
         (try_add_downstream(t, std::get<Is>(t->args)), ...);
      }

      /**
       * Create a lazy node which value will be calculated by a function and several value or lazy node
       *
       * \param func the function
       * \param args arguments to be put into function to calculate the value of result lazy node.
       * it could be value, reference_wrapper or another node
       */
      template<typename Func, typename... Args>
      auto Node(Func&& func, Args&&... args) {
         using T = std::invoke_result_t<Func, decltype(unwrap_arg(std::declval<Args>()))...>;
         // Remove reference here, all data will be stored as value in node.
         // If it is needed to store reference, use referernce_wrapper instead.
         auto result = std::make_shared<specific_function<T, std::remove_reference_t<Func>, std::remove_reference_t<Args>...>>(
               std::forward<Func>(func),
               std::forward<Args>(args)...);
         add_as_downstream(result, std::index_sequence_for<Args...>());
         return node<T>(std::move(result));
      }
      /**
       * Create a empty root lazy node
       */
      template<typename T>
      auto Root() {
         auto result = std::make_shared<given_value<T>>();
         return node<T>(std::move(result));
      }
      /**
       * Create a root lazy node by given const lvalue
       */
      template<typename T>
      auto Root(const T& arg) {
         auto result = std::make_shared<given_value<T>>();
         result->reset(arg);
         return node<T>(std::move(result));
      }
      /**
       * Create a root lazy node by a given rvalue
       */
      template<typename T>
      auto Root(T&& arg) {
         auto result = std::make_shared<given_value<T>>();
         result->reset(std::move(arg));
         return node<T>(std::move(result));
      }
      /**
       * Create a root lazy node by a given reference
       */
      template<typename T>
      auto Root(std::reference_wrapper<const T> arg) {
         auto result = std::make_shared<given_value<T>>();
         result->reset(arg);
         return node<T>(std::move(result));
      }

      /**
       * Auxiliary object to copy lazy node graph
       */
      struct Copy {
         /**
          * A map record relation between old node and new node
          */
         map_t map;
         /**
          * Copy a node
          *
          * It will replace upstream of the result node automatically, and record this copy in the map
          */
         template<typename T>
         node<T> operator()(const node<T>& n0) {
            node<T> result = n0.pointer->copy(map);
            map[n0.pointer] = result.pointer;
            return result;
         }
      };

      template<typename T>
      node<T> given_value<T>::copy(const map_t& map) const {
         auto result = std::make_shared<given_value<T>>();
         if (auto v = std::get_if<T>(&value)) {
            result->reset(*v); // copy value
         } else if (auto v = std::get_if<std::reference_wrapper<const T>>(&value)) {
            result->reset(*v); // copy reference_wrapper
         }
         return node<T>(std::move(result));
      }

      /**
       * Map old wrapped node to new wrapped node, if the parameter is not a node, return reference itself
       *
       * If it is not a node, a copy will happened later when constructing lazy node, so no copy needed here.
       */
      template<typename T>
      decltype(auto) try_map(const map_t& map, const T& arg) {
         if constexpr (is_node_v<T>) {
            // T is node<U>
            return T(std::dynamic_pointer_cast<typename T::pointer_t::element_type>(map.at(arg.pointer))); // value
         } else {
            return (arg); // reference
         }
      }

      /**
       * Helper function to copy specific_function lazy node
       */
      template<typename T, typename Func, typename Args, std::size_t... Is>
      auto copy_specific_function_impl(const Func& func, const Args& args, const map_t& map, std::index_sequence<Is...>) {
         return std::make_shared<T>(func, try_map(map, std::get<Is>(args))...);
      }

      template<typename T, typename Func, typename... Args>
      node<T> specific_function<T, Func, Args...>::copy(const map_t& map) const {
         auto result = copy_specific_function_impl<specific_function<T, Func, Args...>>(func, args, map, std::index_sequence_for<Args...>());
         add_as_downstream(result, std::index_sequence_for<Args...>());
         result->value = value;
         return node<T>(std::move(result));
      }
   } // namespace detail

   using detail::Copy;
   using detail::Node;
   using detail::Root;
} // namespace lazy

#endif
