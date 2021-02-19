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

// 虽然pmr在c++17中, 但是gcc 7并不支持pmr, 所以自己实现了个
// #include <memory_resource>

#include <cstddef>
#include <forward_list>
#include <list>
#include <map>
#include <new>
#include <set>
#include <vector>

namespace TAT {
   namespace pmr {
      struct memory_resource {
         virtual ~memory_resource() {}

         void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
            return do_allocate(bytes, alignment);
         }

         void deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
            do_deallocate(p, bytes, alignment);
         }

         bool is_equal(const memory_resource& other) const noexcept {
            return do_is_equal(other);
         }

         virtual void* do_allocate(std::size_t bytes, std::size_t alignment) = 0;
         virtual void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) = 0;
         virtual bool do_is_equal(const memory_resource& other) const noexcept = 0;
      };
      inline bool operator==(const memory_resource& a, const memory_resource& b) noexcept {
         return &a == &b || a.is_equal(b);
      }
      inline bool operator!=(const memory_resource& a, const memory_resource& b) noexcept {
         return !(a == b);
      }

      inline memory_resource* get_default_resource();
      inline memory_resource* set_default_resource(memory_resource* input);

      struct new_delete_resource : memory_resource {
         void* do_allocate(std::size_t bytes, std::size_t) override {
            return new std::byte[bytes];
         }
         virtual void do_deallocate(void* p, std::size_t, std::size_t) override {
            delete[]((std::byte*)p);
         }
         virtual bool do_is_equal(const memory_resource& other) const noexcept override {
            return this == &other;
         }
      };

      struct buffer_t {
         void* buffer;
         std::size_t size;
      };

      // 抄boost的
      struct monotonic_buffer_resource : memory_resource {
         std::forward_list<buffer_t> m_buffer_list;
         memory_resource* m_upstream;
         void* m_current_buffer;
         std::size_t m_current_buffer_size;
         std::size_t m_next_buffer_size;
         void* const m_initial_buffer;
         std::size_t const m_initial_buffer_size;

         static const std::size_t default_next_buffer_size = 32u * sizeof(void*);

         monotonic_buffer_resource(void* buffer, std::size_t buffer_size) : monotonic_buffer_resource(buffer, buffer_size, get_default_resource()) {}
         monotonic_buffer_resource(void* buffer, std::size_t buffer_size, memory_resource* upstream) :
               m_buffer_list(),
               m_upstream(upstream),
               m_current_buffer(buffer),
               m_current_buffer_size(buffer_size),
               m_next_buffer_size(default_next_buffer_size),
               m_initial_buffer(buffer),
               m_initial_buffer_size(buffer_size) {
            increase_next_size_to(buffer_size);
         }
         monotonic_buffer_resource(const monotonic_buffer_resource&) = delete;

         void increase_next_size() {
            m_next_buffer_size = (std::size_t(-1) / 2 < m_next_buffer_size) ? std::size_t(-1) : m_next_buffer_size * 2;
         }
         void increase_next_size_to(std::size_t minimum_size) {
            while (m_next_buffer_size < minimum_size) {
               increase_next_size();
            }
         }

         memory_resource* upstream_resource() const {
            return m_upstream;
         }

         ~monotonic_buffer_resource() override {
            release();
         }

         void release() {
            for (auto [buffer, size] : m_buffer_list) {
               m_upstream->deallocate(buffer, size, alignof(std::max_align_t));
            }
            m_buffer_list.clear();
            m_current_buffer = m_initial_buffer;
            m_current_buffer_size = m_initial_buffer_size;
            m_next_buffer_size = default_next_buffer_size;
         }

         std::size_t remaining_storage(std::size_t alignment, std::size_t& aligner) {
            const std::size_t up_alignment_minus1 = alignment - 1u;
            const std::size_t up_alignment_mask = ~up_alignment_minus1;
            const std::size_t up_addr = std::size_t(m_current_buffer);
            const std::size_t up_aligned_addr = (up_addr + up_alignment_minus1) & up_alignment_mask;
            aligner = std::size_t(up_aligned_addr - up_addr);
            return m_current_buffer_size <= aligner ? 0u : m_current_buffer_size - aligner;
         }

         void* allocate_from_current(std::size_t aligner, std::size_t bytes) {
            std::byte* result = (std::byte*)m_current_buffer + aligner;
            m_current_buffer = result + bytes;
            m_current_buffer_size -= aligner + bytes;
            return result;
         }

         void* do_allocate(std::size_t bytes, std::size_t alignment) override {
            std::size_t aligner = 0u;
            if (remaining_storage(alignment, aligner) < bytes) {
               aligner = 0u;
               increase_next_size_to(bytes);
               m_current_buffer = m_upstream->allocate(m_next_buffer_size, alignof(std::max_align_t));
               m_current_buffer_size = m_next_buffer_size;
               m_buffer_list.push_front({m_current_buffer, m_current_buffer_size});
               increase_next_size();
            }
            return allocate_from_current(aligner, bytes);
         }

         virtual void do_deallocate(void*, std::size_t, std::size_t) override {}

         virtual bool do_is_equal(const memory_resource& other) const noexcept override {
            return this == &other;
         }
      };

      inline new_delete_resource new_delete_resource_object;
      inline memory_resource* default_resource = &new_delete_resource_object;
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
      template<typename Derived, typename T, bool try_not_initialize>
      struct polymorphic_allocator_base {
         memory_resource* m_resource;

         using value_type = T;
         polymorphic_allocator_base() noexcept : polymorphic_allocator_base(get_default_resource()) {}
         polymorphic_allocator_base(const polymorphic_allocator_base& other) = default;
         template<typename DerivedAllocator, typename U>
         polymorphic_allocator_base(const polymorphic_allocator_base<DerivedAllocator, U, try_not_initialize>& other) noexcept :
               polymorphic_allocator_base(other.resource()) {}
         polymorphic_allocator_base(memory_resource* r) : m_resource(r) {}

         polymorphic_allocator_base<Derived, T, try_not_initialize>&
         operator=(const polymorphic_allocator_base<Derived, T, try_not_initialize>&) = delete;

         T* allocate(std::size_t n) {
            return static_cast<T*>(resource()->allocate(n * sizeof(T), alignof(T)));
         }

         void deallocate(T* p, std::size_t n) {
            resource()->deallocate(p, n * sizeof(T), alignof(T));
         }

         template<class U, class... Args>
         void construct([[maybe_unused]] U* p, Args&&... args) {
            if constexpr (!(try_not_initialize && (sizeof...(args) == 0) && (std::is_trivially_destructible_v<T>))) {
               new (p) U(std::forward<Args>(args)...);
            }
         }

         template<class U>
         void destroy(U* p) {
            p->~U();
         }

         Derived select_on_container_copy_construction() const {
            return Derived();
         }

         memory_resource* resource() const {
            return m_resource;
         }
      };

      template<typename T>
      struct polymorphic_allocator : polymorphic_allocator_base<polymorphic_allocator<T>, T, false> {
         using polymorphic_allocator_base<polymorphic_allocator<T>, T, false>::polymorphic_allocator_base;
      };
      template<typename T>
      struct polymorphic_allocator_without_initialize : polymorphic_allocator_base<polymorphic_allocator_without_initialize<T>, T, true> {
         using polymorphic_allocator_base<polymorphic_allocator_without_initialize<T>, T, true>::polymorphic_allocator_base;
      };

      template<class T1, class T2>
      bool operator==(const polymorphic_allocator<T1>& lhs, const polymorphic_allocator<T2>& rhs) noexcept {
         return *lhs.resource() == *rhs.resource();
      }
      template<class T1, class T2>
      bool operator!=(const polymorphic_allocator<T1>& lhs, const polymorphic_allocator<T2>& rhs) noexcept {
         return !(lhs == rhs);
      }
      template<class T1, class T2>
      bool operator==(const polymorphic_allocator_without_initialize<T1>& lhs, const polymorphic_allocator_without_initialize<T2>& rhs) noexcept {
         return *lhs.resource() == *rhs.resource();
      }
      template<class T1, class T2>
      bool operator!=(const polymorphic_allocator_without_initialize<T1>& lhs, const polymorphic_allocator_without_initialize<T2>& rhs) noexcept {
         return !(lhs == rhs);
      }

      template<typename T>
      using content_vector = std::vector<T, polymorphic_allocator_without_initialize<T>>;

      template<typename T>
      using vector = std::vector<T, polymorphic_allocator<T>>;

      template<typename T>
      using list = std::list<T, polymorphic_allocator<T>>;

      template<typename Key, typename T, typename Compare = std::less<Key>>
      using map = std::map<Key, T, Compare, polymorphic_allocator<std::pair<const Key, T>>>;

      template<class Key, class Compare = std::less<Key>>
      using set = std::set<Key, Compare, polymorphic_allocator<Key>>;
   } // namespace pmr

   // on windows stack size is 1MB(1<<20), and on linux, stack size is 8M(1<<23)
   // 这个buffer应当仅仅用于零碎的变量
   // 对于tensor中的数据, 在一些背景下很容易有几百兆以上的大小
   constexpr std::size_t default_buffer_size = 1 << 15;

   template<std::size_t, bool dynamic = false>
   struct scope_resource {
      std::byte* buffer;
      pmr::monotonic_buffer_resource resource;
      pmr::memory_resource* upstream;
      scope_resource(std::size_t size) :
            buffer(new std::byte[size]), resource(buffer, size * sizeof(std::byte)), upstream(pmr::set_default_resource(&resource)) {}
      ~scope_resource() {
         pmr::set_default_resource(upstream);
         delete[] buffer;
      }
   };
   template<std::size_t buffer_size>
   struct scope_resource<buffer_size, false> {
      std::byte buffer[buffer_size];
      pmr::monotonic_buffer_resource resource;
      pmr::memory_resource* upstream;
      scope_resource() : resource(buffer, sizeof(buffer)), upstream(pmr::set_default_resource(&resource)) {}
      ~scope_resource() {
         pmr::set_default_resource(upstream);
      }
   };
   scope_resource(std::size_t)->scope_resource<0, true>;
} // namespace TAT
#endif
