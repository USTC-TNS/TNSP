/**
 * \file allocator.hpp
 *
 * Copyright (C) 2021-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

// gcc 7 and llvm does not support memory_resource so implement by myself

#include <cstddef>
#include <forward_list>
#include <memory>

namespace TAT {
    namespace detail {
        struct memory_resource {
            virtual ~memory_resource() { }

            void* allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
                return do_allocate(bytes, alignment);
            }

            void deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
                do_deallocate(p, bytes, alignment);
            }

            bool is_equal(const memory_resource& other) const {
                return do_is_equal(other);
            }

            virtual void* do_allocate(std::size_t bytes, std::size_t alignment) = 0;
            virtual void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) = 0;
            virtual bool do_is_equal(const memory_resource& other) const = 0;
        };
        inline bool operator==(const memory_resource& a, const memory_resource& b) {
            return &a == &b || a.is_equal(b);
        }
        inline bool operator!=(const memory_resource& a, const memory_resource& b) {
            return !(a == b);
        }

        // This is different with std::pmr, it is thread unsafe
        inline memory_resource* get_default_resource();
        inline memory_resource* set_default_resource(memory_resource* input);

        struct new_delete_resource : memory_resource {
            void* do_allocate(std::size_t bytes, std::size_t) override {
                return new std::byte[bytes];
            }
            virtual void do_deallocate(void* p, std::size_t, std::size_t) override {
                delete[] ((std::byte*)p);
            }
            virtual bool do_is_equal(const memory_resource& other) const override {
                return this == &other;
            }
        };

        struct buffer_t {
            void* buffer;
            std::size_t size;
        };

        // copy from boost
        struct monotonic_buffer_resource : memory_resource {
            std::forward_list<buffer_t> m_buffer_list;
            memory_resource* m_upstream;
            void* m_current_buffer;
            std::size_t m_current_buffer_size;
            std::size_t m_next_buffer_size;
            void* const m_initial_buffer;
            std::size_t const m_initial_buffer_size;

            static const std::size_t default_next_buffer_size = 32u * sizeof(void*);

            monotonic_buffer_resource(void* buffer, std::size_t buffer_size) :
                monotonic_buffer_resource(buffer, buffer_size, get_default_resource()) { }
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

            virtual void do_deallocate(void*, std::size_t, std::size_t) override { }

            virtual bool do_is_equal(const memory_resource& other) const override {
                return this == &other;
            }
        };

        inline new_delete_resource new_delete_resource_object;
        inline memory_resource* default_resource = &new_delete_resource_object;
        inline memory_resource* get_default_resource() {
            return default_resource;
        }
        inline memory_resource* set_default_resource(memory_resource* input) {
            auto output = default_resource;
            default_resource = input;
            return output;
        }

        // Similar to std::pmr::polymorphic_allocator
        // But default source(get_default_resource) is thread unsafe
        template<typename T>
        struct polymorphic_allocator {
            memory_resource* m_resource;

            using value_type = T;
            polymorphic_allocator() : polymorphic_allocator(get_default_resource()) { }
            polymorphic_allocator(const polymorphic_allocator& other) = default;
            template<typename U>
            polymorphic_allocator(const polymorphic_allocator<U>& other) : polymorphic_allocator(other.resource()) { }
            polymorphic_allocator(memory_resource* r) : m_resource(r) { }

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
                return m_resource;
            }
        };

        template<typename T1, typename T2>
        bool operator==(const polymorphic_allocator<T1>& lhs, const polymorphic_allocator<T2>& rhs) {
            return *lhs.resource() == *rhs.resource();
        }
        template<typename T1, typename T2>
        bool operator!=(const polymorphic_allocator<T1>& lhs, const polymorphic_allocator<T2>& rhs) {
            return !(lhs == rhs);
        }

        template<typename T>
        struct no_initialize_polymorphic_allocator : polymorphic_allocator<T> {
            using polymorphic_allocator<T>::polymorphic_allocator;

            no_initialize_polymorphic_allocator(const no_initialize_polymorphic_allocator& other) = default;
            template<typename U>
            no_initialize_polymorphic_allocator(const no_initialize_polymorphic_allocator<U>& other) :
                no_initialize_polymorphic_allocator(other.resource()) { }

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
        bool operator==(const no_initialize_polymorphic_allocator<T1>& lhs, const no_initialize_polymorphic_allocator<T2>& rhs) {
            return *lhs.resource() == *rhs.resource();
        }
        template<typename T1, typename T2>
        bool operator!=(const no_initialize_polymorphic_allocator<T1>& lhs, const no_initialize_polymorphic_allocator<T2>& rhs) {
            return !(lhs == rhs);
        }

        /**
         * Allocator without initialize the element if no parameter given
         *
         * Inherit from std::allocator
         */
        template<typename T>
        struct no_initialize_allocator : std::allocator<T> {
            using std::allocator<T>::allocator;

            no_initialize_allocator(const no_initialize_allocator& other) = default;
            template<typename U>
            no_initialize_allocator(const no_initialize_allocator<U>& other) : no_initialize_allocator() { }

            no_initialize_allocator<T> select_on_container_copy_construction() const {
                return no_initialize_allocator<T>();
            }

            // It is useless, but base class has it so derived class must have it.
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

    constexpr std::size_t default_buffer_size = 1 << 20;

    // never use stack, always use heap
    struct scope_resource {
        std::byte* buffer;
        detail::monotonic_buffer_resource resource;
        detail::memory_resource* upstream;
        scope_resource(std::size_t size = default_buffer_size) :
            buffer(new std::byte[size]),
            resource(buffer, size * sizeof(std::byte)),
            upstream(set_default_resource(&resource)) { }
        ~scope_resource() {
            set_default_resource(upstream);
            delete[] buffer;
        }
    };
} // namespace TAT

#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace TAT {
    namespace no_initialize {
        template<typename T>
        using vector = std::vector<T, detail::no_initialize_allocator<T>>;

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

        template<typename Key, typename T, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
        using unordered_map = std::unordered_map<Key, T, Hash, KeyEqual, detail::polymorphic_allocator<std::pair<const Key, T>>>;

        template<class Key, class Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
        using unordered_set = std::unordered_set<Key, Hash, KeyEqual, detail::polymorphic_allocator<Key>>;
    } // namespace pmr
} // namespace TAT
#endif
