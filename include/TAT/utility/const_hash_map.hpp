/**
 * \file const_hash_map.hpp
 *
 * Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#include <type_traits>
#ifndef TAT_CONST_HASH_MAP_HPP
#define TAT_CONST_HASH_MAP_HPP

#include <functional>

namespace TAT {
   namespace detail {
      template<typename Key, typename T, typename Hash = std::hash<Key>, template<typename> class Allocator = std::allocator>
      struct const_hash_map {
       public:
         using value_type = std::pair<Key, T>;
         using real_value_type = std::pair<std::size_t, value_type>;

       private:
         using data_type = std::vector<real_value_type, Allocator<real_value_type>>;
         data_type data;

         template<typename Value>
         static std::size_t get_hash(const Value& value) {
            if constexpr (std::is_same_v<Value, std::size_t>) {
               return value;
            } else {
               return value.first;
            }
         }
       public:
         struct const_iterator {
          private:
            using real_iterator_type = typename data_type::const_iterator;
            real_iterator_type real_iterator;
          public:
            const_iterator(real_iterator_type i) noexcept : real_iterator(i) {}
            const_iterator& operator++() {
               ++real_iterator;
               return *this;
            }
            const_iterator& operator--() {
               --real_iterator;
               return *this;
            }
            const value_type& operator*() const {
               return real_iterator->second;
            }
            const value_type* operator->() const {
               return &(real_iterator->second);
            }
            bool operator==(const const_iterator& other) const noexcept {
               return real_iterator == other.real_iterator;
            }
            bool operator!=(const const_iterator& other) const noexcept {
               return real_iterator != other.real_iterator;
            }
         };

         void reserve(std::size_t size) {
            data.reserve(size);
         }

         template<typename K, typename V>
         void add(K&& key, V&& value) {
            std::size_t hash = Hash{}(key);
            data.push_back({hash, {std::forward<K>(key), std::forward<V>(value)}});
         }

         void sort() {
            std::sort(data.begin(), data.end(), [&](const auto& a, const auto& b) {
               return get_hash(a) < get_hash(b);
            });
         }

         template<typename K>
         const_iterator find(K&& key) const {
            std::size_t hash = Hash{}(key);
            auto hash_found = std::lower_bound(data.begin(), data.end(), hash, [](const auto& a, const auto& b) {
               return get_hash(a) < get_hash(b);
            });
            while (true) {
               if (hash_found == data.end()) {
                  return end();
               }
               if (hash_found->first != hash) {
                  return end();
               }
               if (hash_found->second.first == key) {
                  return const_iterator(hash_found);
               }
               ++hash_found;
            }
         }

         const_iterator cbegin() const {
            return data.cbegin();
         }
         const_iterator cend() const {
            return data.cend();
         }
         const_iterator begin() const {
            return data.begin();
         }
         const_iterator end() const {
            return data.end();
         }
      };
   } // namespace detail
} // namespace TAT
#endif
