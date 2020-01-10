/**
 * \file core.hpp
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
#ifndef TAT_CORE_HPP
#define TAT_CORE_HPP

#include "edge.hpp"
#include "name.hpp"

namespace TAT {
   template<class ScalarType, class Symmetry>
   struct Block {
      vector<Symmetry> symmetries;
      vector<ScalarType> raw_data;

      template<
            class T = vector<Symmetry>,
            class = std::enable_if_t<std::is_convertible_v<T, vector<Symmetry>>>>
      Block(const vector<map<Symmetry, Size>>& e, T&& s) : symmetries(std::forward<T>(s)) {
         Size size = 1;
         for (Rank i = 0; i < e.size(); i++) {
            size *= e[i].at(symmetries[i]);
         }
         raw_data.resize(size);
      }
   };

   template<class Symmetry>
   auto initialize_block_symmetries_with_check(const vector<map<Symmetry, Size>>& edges) {
      auto res = vector<vector<Symmetry>>();
      auto vec = vector<Symmetry>(edges.size());
      using PosType = vector<typename std::map<Symmetry, Size>::const_iterator>;
      loop_edge(
            edges,
            [&res]() { res.push_back({}); },
            []([[maybe_unused]] const PosType& pos) {
               auto sum = Symmetry();
               for (const auto& i : pos) {
                  sum += i->first;
               }
               return sum == Symmetry();
            },
            [&res, &vec]([[maybe_unused]] const PosType& pos) { res.push_back(vec); },
            [&vec](const PosType& pos, const Rank ptr) {
               for (auto i = ptr; i < pos.size(); i++) {
                  vec[i] = pos[i]->first;
               }
            });
      return res;
   }

   template<class ScalarType, class Symmetry>
   struct Core {
      vector<map<Symmetry, Size>> edges;
      vector<Block<ScalarType, Symmetry>> blocks;

      template<
            class T = vector<map<Symmetry, Size>>,
            class = std::enable_if_t<std::is_convertible_v<T, vector<map<Symmetry, Size>>>>>
      Core(T&& e) : edges(std::forward<T>(e)) {
         auto symmetries_list = initialize_block_symmetries_with_check(edges);
         for (auto& i : symmetries_list) {
            blocks.push_back(Block<ScalarType, Symmetry>(edges, std::move(i)));
         }
      }

      Nums find_block(const vector<Symmetry>& symmetries) {
         const auto number = blocks.size();
         for (auto i = 0; i < number; i++) {
            if (symmetries == blocks[i].symmetries) {
               return i;
            }
         }
         TAT_WARNING("Block Not Found");
         return number;
      }
   };

   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto get_pos_for_at(
         const std::map<Name, Symmetry>& position,
         const std::map<Name, Rank>& name_to_index,
         const Core<ScalarType, Symmetry>& core) {
      auto rank = Rank(core.edges.size());
      vector<Symmetry> block_symmetries(rank);
      for (const auto& [name, sym] : position) {
         auto index = name_to_index.at(name);
         block_symmetries[index] = sym;
      }
      for (Nums i = 0; i < core.blocks.size(); i++) {
         if (block_symmetries == core.blocks[i].symmetries) {
            return i;
         }
      }
      TAT_WARNING("Cannot Find Correct Block When Get Item");
      return Nums(0);
   }

   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto get_pos_for_at(
         const std::map<Name, Size>& position,
         const std::map<Name, Rank>& name_to_index,
         const Core<ScalarType, Symmetry>& core) {
      auto rank = Rank(core.edges.size());
      vector<Size> scalar_position(rank);
      vector<Size> dimensions(rank);
      for (const auto& [name, res] : position) {
         auto index = name_to_index.at(name);
         scalar_position[index] = res;
         dimensions[index] = core.edges[index].at(Symmetry());
      }
      Size offset = 0;
      for (Rank j = 0; j < rank; j++) {
         offset *= dimensions[j];
         offset += scalar_position[j];
      }
      return offset;
   }

   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto get_pos_for_at(
         const std::map<Name, std::tuple<Symmetry, Size>>& position,
         const std::map<Name, Rank>& name_to_index,
         const Core<ScalarType, Symmetry>& core) {
      auto rank = Rank(core.edges.size());
      vector<Symmetry> block_symmetries(rank);
      vector<Size> scalar_position(rank);
      vector<Size> dimensions(rank);
      for (const auto& [name, res] : position) {
         auto index = name_to_index.at(name);
         block_symmetries[index] = std::get<0>(res);
         scalar_position[index] = std::get<1>(res);
         dimensions[index] = core.edges[index].at(std::get<0>(res));
      }
      Size offset = 0;
      for (Rank j = 0; j < rank; j++) {
         offset *= dimensions[j];
         offset += scalar_position[j];
      }
      for (Nums i = 0; i < core.blocks.size(); i++) {
         if (block_symmetries == core.blocks[i].symmetries) {
            return std::make_tuple(i, offset);
         }
      }
      TAT_WARNING("Cannot Find Correct Block When Get Item");
      return std::make_tuple(Nums(0), Size(0));
   }
} // namespace TAT
#endif
