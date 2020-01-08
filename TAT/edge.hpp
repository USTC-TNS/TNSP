/**
 * \file edge.hpp
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
#pragma once
#ifndef TAT_EDGE_HPP
#define TAT_EDGE_HPP

#include "misc.hpp"

namespace TAT {
   template<class T, class F1, class F2, class F3, class F4>
   void loop_edge(const T& edges, F1&& rank0, F2&& check, F3&& append, F4&& update) {
      const Rank rank = edges.size();
      if (!rank) {
         rank0();
         return;
      }
      auto pos = vector<typename std::remove_pointer_t<typename T::value_type>::const_iterator>();
      for (const auto& i : edges) {
         auto ptr = std_begin(i);
         if (ptr == std_end(i)) {
            return;
         }
         pos.push_back(ptr);
      }
      auto min_ptr = 0;
      while (true) {
         if (check(pos)) {
            update(pos, min_ptr);
            min_ptr = rank;
            append(pos);
         }
         auto ptr = rank - 1;
         ++pos[ptr];
         while (pos[ptr] == std_end(edges[ptr])) {
            if (ptr == 0) {
               return;
            }
            pos[ptr] = std_begin(edges[ptr]);
            --ptr;
            ++pos[ptr];
         }
         min_ptr = min_ptr < ptr ? min_ptr : ptr;
      }
   }

   template<class Symmetry>
   struct EdgePosition {
      Symmetry sym;
      Size position;

      EdgePosition(const Size p) : sym(Symmetry()), position(p) {}
      EdgePosition(Symmetry s, const Size p) : sym(s), position(p) {}
   };

   template<class Symmetry>
   [[nodiscard]] map<Symmetry, Size>
   get_merged_edge(const vector<const map<Symmetry, Size>*>& edges_to_merge) {
      auto res_edge = map<Symmetry, Size>();

      auto sym = vector<Symmetry>(edges_to_merge.size());
      auto dim = vector<Size>(edges_to_merge.size());

      using PosType = vector<typename std::map<Symmetry, Size>::const_iterator>;

      auto update_sym_and_dim = [&sym, &dim](const PosType& pos, const Rank start) {
         for (auto i = start; i < pos.size(); i++) {
            const auto& ptr = pos[i];
            if (i == 0) {
               sym[i] = ptr->first;
               dim[i] = ptr->second;
            } else {
               sym[i] = ptr->first + sym[i - 1];
               dim[i] = ptr->second * dim[i - 1];
               // do not check dim=0, because in constructor, i didn't check
            }
         }
      };

      loop_edge(
            edges_to_merge,
            [&res_edge]() { res_edge[Symmetry()] = 1; },
            []([[maybe_unused]] const PosType& pos) { return true; },
            [&res_edge, &sym, &dim]([[maybe_unused]] const PosType& pos) {
               res_edge[sym[pos.size() - 1]] += dim[pos.size() - 1];
            },
            update_sym_and_dim);

      return res_edge;
   }
} // namespace TAT
#endif
