/**
 * \file hash_for_list.hpp
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
#ifndef TAT_HASH_FOR_LIST_HPP
#define TAT_HASH_FOR_LIST_HPP

#include <cstddef>
#include <functional>

namespace TAT {
   namespace detail {
      inline std::size_t& hash_absorb(std::size_t& seed, std::size_t value) {
         // copy from boost
         return seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }

      struct hash_for_list {
         template<typename List>
         std::size_t operator()(const List& list) const {
            std::hash<typename List::value_type> hash_for_item;
            std::size_t seed = list.size();
            for (const auto& item : list) {
               hash_absorb(seed, hash_for_item(item));
            }
            return seed;
         }
      };
   } // namespace detail
} // namespace TAT

#endif
