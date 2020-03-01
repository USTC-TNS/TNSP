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
   /**
    * \brief 记录了张量的核心数据的类型, 核心数据指的是除了角标名称之外的信息, 包括边的形状,
    * 以及张量内本身的数据
    * \tparam ScalarType 张量内本身的数据的标量类型
    * \tparam Symmetry 张量所拥有的对称性
    */
   template<class ScalarType, class Symmetry>
   struct Core {
      /**
       * \brief 张量的形状, 是边的形状的列表, 列表长度为张量的秩, 每个边是一个对称性值到子边长度的映射表
       * \see Edge
       */
      vector<Edge<Symmetry>> edges = {};
      /**
       * \brief 张量内本身的数据, 是对称性列表到数据列表的映射表, 数据列表就是张量内本身的数据,
       * 而对称性列表表示此子块各个子边在各自的边上所对应的对称性值
       */
      std::map<vector<Symmetry>, vector<ScalarType>> blocks = {};

      /**
       * \brief 根据边的形状构造张量, 然后根据对称性条件自动构造张量的分块
       * \param edges_init 边的形状的列表
       * \param auto_reverse 对于费米张量是否自动对含有负对称值的边整个取反
       * 原则上构造时费米对称性值应该全正或全负, 如果不是这样, 结果会难以理解
       */
      template<
            class T = vector<Edge<Symmetry>>,
            class = std::enable_if_t<std::is_convertible_v<T, vector<Edge<Symmetry>>>>>
      Core(T&& edges_init, [[maybe_unused]] const bool auto_reverse = false) :
            edges(std::forward<T>(edges_init)) {
         // 自动翻转边
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            if (auto_reverse) {
               for (auto& i : edges) {
                  i.possible_reverse();
               }
            }
         }
         // 生成数据
         auto symmetries_list = initialize_block_symmetries_with_check(edges);
         for (const auto& [i, j] : symmetries_list) {
            blocks[i] = vector<ScalarType>(j);
         }
         // 删除不在block中用到的sym
         auto edge_mark = vector<std::map<Symmetry, bool>>();
         for (const auto& e : edges) {
            auto m = std::map<Symmetry, bool>();
            for (const auto& [s, _] : e.map) {
               m[s] = true;
            }
            edge_mark.push_back(std::move(m));
         }
         for (const auto& [s, b] : blocks) {
            for (auto i = 0; i < edges.size(); i++) {
               edge_mark[i].at(s[i]) = false;
            }
         }
         for (auto i = 0; i < edges.size(); i++) {
            for (const auto& [s, f] : edge_mark[i]) {
               if (f) {
                  edges[i].map.erase(s);
               }
            }
         }
      }

      Core() = default;
      Core(const Core&) = default;
      Core(Core&&) = default;
      Core& operator=(const Core&) = default;
      Core& operator=(Core&&) = default;
      ~Core() = default;
   };

   /**
    * \brief 寻找有对称性张量中的某个子块
    */
   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto get_block_for_get_item(
         const std::map<Name, Symmetry>& position,
         const std::map<Name, Rank>& name_to_index,
         const Core<ScalarType, Symmetry>& core) {
      auto symmetries = vector<Symmetry>(core.edges.size());
      for (const auto& [name, sym] : position) {
         symmetries[name_to_index.at(name)] = sym;
      }
      return symmetries;
   }

   /**
    * \brief 寻找无对称性张量中每个元素
    */
   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto get_offset_for_get_item(
         const std::map<Name, Size>& position,
         const std::map<Name, Rank>& name_to_index,
         const Core<ScalarType, Symmetry>& core) {
      const auto rank = Rank(core.edges.size());
      auto scalar_position = vector<Size>(rank);
      auto dimensions = vector<Size>(rank);
      for (const auto& [name, res] : position) {
         auto index = name_to_index.at(name);
         scalar_position[index] = res;
         dimensions[index] = core.edges[index].map.begin()->second;
      }
      Size offset = 0;
      for (Rank j = 0; j < rank; j++) {
         offset *= dimensions[j];
         offset += scalar_position[j];
      }
      return offset;
   }

   /**
    * \brief 寻找有对称性张量中的某个子块的某个元素
    */
   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto get_block_and_offset_for_get_item(
         const std::map<Name, std::tuple<Symmetry, Size>>& position,
         const std::map<Name, Rank>& name_to_index,
         const Core<ScalarType, Symmetry>& core) {
      const auto rank = Rank(core.edges.size());
      auto symmetries = vector<Symmetry>(rank);
      auto scalar_position = vector<Size>(rank);
      auto dimensions = vector<Size>(rank);
      for (const auto& [name, _] : position) {
         const auto& [sym, res] = _;
         auto index = name_to_index.at(name);
         symmetries[index] = sym;
         scalar_position[index] = res;
         dimensions[index] = core.edges[index].map.at(sym);
      }
      Size offset = 0;
      for (Rank j = 0; j < rank; j++) {
         offset *= dimensions[j];
         offset += scalar_position[j];
      }
      return std::make_tuple(symmetries, offset);
   }
} // namespace TAT
#endif
