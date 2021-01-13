/**
 * \file pmr_resource.hpp
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
#ifndef TAT_PMR_RESOURCE_HPP
#define TAT_PMR_RESOURCE_HPP

// TODO 虽然pmr在c++17中, 但是gcc 7并不支持pmr, 所以增加一个使用boost的选项
#ifdef TAT_USE_BOOST_PMR
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>
#else
#include <memory_resource>
#endif

#include <map>
#include <set>
#include <vector>

namespace TAT {
   namespace pmr {
      namespace pmr_source =
#ifdef TAT_USE_BOOST_PMR
            boost::container::pmr
#else
            std::pmr
#endif
            ;

      using pmr_source::memory_resource;
      using pmr_source::monotonic_buffer_resource;

      inline memory_resource* default_resource = pmr_source::new_delete_resource();
      inline memory_resource* get_default_resource() {
         return default_resource;
      }
      inline memory_resource* set_default_resource(memory_resource* input) {
         auto result = default_resource;
         default_resource = input;
         return result;
      }

      // 和std::pmr::polymorphic_allocator几乎一模一样
      // 和std::pmr::polymorphic_allocator的初始化时的默认resource不同, 使用的是自己的thread unsafe版本
      template<typename T>
      struct polymorphic_allocator {
         memory_resource* _M_resource;

         using value_type = T;
         polymorphic_allocator() noexcept : polymorphic_allocator(get_default_resource()) {}
         polymorphic_allocator(const polymorphic_allocator& other) = default;
         template<typename U>
         polymorphic_allocator(const polymorphic_allocator<U>& other) noexcept : polymorphic_allocator(other.resource()) {}
         polymorphic_allocator(memory_resource* r) : _M_resource(r) {}

         polymorphic_allocator<T>& operator=(const polymorphic_allocator<T>&) = delete;

         T* allocate(std::size_t n) {
            return static_cast<T*>(resource()->allocate(n * sizeof(T), alignof(T)));
         }

         void deallocate(T* p, std::size_t n) {
            resource()->deallocate(p, n * sizeof(T), alignof(T));
         }

         template<class U, class... Args>
         void construct(U* p, Args&&... args) {
            new (p) U(std::forward<Args>(args)...);
         }

         template<class U>
         void destroy(U* p) {
            p->~U();
         }

         polymorphic_allocator<T> select_on_container_copy_construction() const {
            return polymorphic_allocator<T>();
         }

         memory_resource* resource() const {
            return _M_resource;
         }
      };

      template<class T1, class T2>
      bool operator==(const polymorphic_allocator<T1>& lhs, const polymorphic_allocator<T2>& rhs) noexcept {
         return *lhs.resource() == *rhs.resource();
      }

      template<typename T>
      using vector = std::vector<T, polymorphic_allocator<T>>;

      template<typename Key, typename T, typename Compare = std::less<Key>>
      using map = std::map<Key, T, Compare, polymorphic_allocator<std::pair<const Key, T>>>;

      template<class Key, class Compare = std::less<Key>>
      using set = std::set<Key, Compare, polymorphic_allocator<Key>>;
   } // namespace pmr

   // on windows stack size is 1MB(1<<20), and on linux, stack size is 8M(1<<23)
   constexpr std::size_t default_buffer_size = 1 << 15;

   template<std::size_t buffer_size = default_buffer_size>
   struct scope_resource {
      std::byte buffer[buffer_size];
      pmr::monotonic_buffer_resource resource;
      pmr::memory_resource* upstream;
      scope_resource() : resource(buffer, sizeof(buffer)), upstream(pmr::set_default_resource(&resource)) {}
      ~scope_resource() {
         pmr::set_default_resource(upstream);
      }
   };

#if 0
   struct scope_resource_adapter {
      pmr::monotonic_buffer_resource resource;
      pmr::memory_resource* upstream;
      scope_resource_adapter(void* buffer, std::size_t size) : resource(buffer, size), upstream(pmr::set_default_resource(&resource)) {}
      ~scope_resource_adapter() {
         pmr::set_default_resource(upstream);
      }
   };
#endif
} // namespace TAT
#endif
