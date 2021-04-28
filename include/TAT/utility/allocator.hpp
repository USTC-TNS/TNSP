/**
 * \file allocator.hpp
 *
 * Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_ALLOCATOR_HPP
#define TAT_ALLOCATOR_HPP

#include <memory>
#include <memory_resource>

namespace TAT {
   namespace detail {
      inline std::pmr::memory_resource* default_resource = std::pmr::new_delete_resource();
      inline std::pmr::memory_resource* get_default_resource() {
         return default_resource;
      }
      inline std::pmr::memory_resource* set_default_resource(std::pmr::memory_resource* input) {
         auto result = default_resource;
         default_resource = input;
         return result;
      }

      // Similar to std::pmr::polymorphic_allocator
      // But default source(get_default_resource) is thread unsafe
      template<typename T>
      struct polymorphic_allocator : std::pmr::polymorphic_allocator<T> {
         using std::pmr::polymorphic_allocator<T>::polymorphic_allocator;
         polymorphic_allocator() noexcept : polymorphic_allocator(get_default_resource()) {}

         template<typename U>
         struct rebind {
            using other = polymorphic_allocator<U>;
         };

         polymorphic_allocator<T> select_on_container_copy_construction() const {
            return polymorphic_allocator<T>();
         }
      };

      template<typename T1, typename T2>
      bool operator==(const polymorphic_allocator<T1>& lhs, const polymorphic_allocator<T2>& rhs) noexcept {
         return *lhs.resource() == *rhs.resource();
      }
      template<typename T1, typename T2>
      bool operator!=(const polymorphic_allocator<T1>& lhs, const polymorphic_allocator<T2>& rhs) noexcept {
         return !(lhs == rhs);
      }

      template<typename T>
      struct no_initialize_polymorphic_allocator : polymorphic_allocator<T> {
         using polymorphic_allocator<T>::polymorphic_allocator;

         template<typename U>
         struct rebind {
            using other = no_initialize_polymorphic_allocator<U>;
         };

         no_initialize_polymorphic_allocator<T> select_on_container_copy_construction() const {
            return no_initialize_polymorphic_allocator<T>();
         }

         template<typename U, typename... Args>
         void construct([[maybe_unused]] U* p, Args&&... args) {
            if constexpr (!((sizeof...(args) == 0) && (std::is_trivially_destructible_v<T>))) {
               new (p) U(std::forward<Args>(args)...);
            }
         }
      };

      template<typename T1, typename T2>
      bool operator==(const no_initialize_polymorphic_allocator<T1>& lhs, const no_initialize_polymorphic_allocator<T2>& rhs) noexcept {
         return *lhs.resource() == *rhs.resource();
      }
      template<typename T1, typename T2>
      bool operator!=(const no_initialize_polymorphic_allocator<T1>& lhs, const no_initialize_polymorphic_allocator<T2>& rhs) noexcept {
         return !(lhs == rhs);
      }
   } // namespace detail

   constexpr std::size_t default_buffer_size = 1 << 20;

   // never use stack, always use heap
   struct scope_resource {
      std::byte* buffer;
      std::pmr::monotonic_buffer_resource resource;
      std::pmr::memory_resource* upstream;
      scope_resource(std::size_t size = default_buffer_size) :
            buffer(new std::byte[size]),
            resource(buffer, size * sizeof(std::byte)),
            upstream(set_default_resource(&resource)) {}
      ~scope_resource() {
         set_default_resource(upstream);
         delete[] buffer;
      }
   };

   namespace detail {
      /**
       * Allocator without initialize the element if no parameter given
       *
       * Inherit from std::allocator
       */
      template<typename T>
      struct no_initialize_allocator : std::allocator<T> {
         template<typename U>
         struct rebind {
            using other = no_initialize_allocator<U>;
         };

         template<typename U, typename... Args>
         void construct([[maybe_unused]] U* p, Args&&... args) {
            if constexpr (!((sizeof...(args) == 0) && (std::is_trivially_destructible_v<T>))) {
               new (p) U(std::forward<Args>(args)...);
            }
         }
      };
   } // namespace detail
} // namespace TAT

#include <list>
#include <map>
#include <set>
#include <vector>

namespace TAT {
   namespace no_initialize {
      template<typename T>
      using vector = std::vector<T, detail::no_initialize_allocator<T>>;

      using string = std::basic_string<char, std::char_traits<char>, detail::no_initialize_allocator<char>>;
      using istringstream = std::basic_istringstream<char, std::char_traits<char>, detail::no_initialize_allocator<char>>;

      namespace pmr {
         /**
          * No initialize version of pmr vector used in tensor content
          */
         template<typename T>
         using vector = std::vector<T, detail::no_initialize_polymorphic_allocator<T>>;
      } // namespace pmr

   } // namespace no_initialize

   namespace pmr {
      // The only difference betwen the below and std::pmr is default resource getter is thread unsafe
      template<typename T>
      using vector = std::vector<T, detail::polymorphic_allocator<T>>;

      template<typename T>
      using list = std::list<T, detail::polymorphic_allocator<T>>;

      template<typename Key, typename T, typename Compare = std::less<Key>>
      using map = std::map<Key, T, Compare, detail::polymorphic_allocator<std::pair<const Key, T>>>;

      template<class Key, class Compare = std::less<Key>>
      using set = std::set<Key, Compare, detail::polymorphic_allocator<Key>>;
   } // namespace pmr
} // namespace TAT
#endif
