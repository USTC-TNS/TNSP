/**
 * \file core.hpp
 *
 * Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <numeric>
#include <vector>

#include "edge.hpp"

namespace TAT {
   /**
    * \defgroup Tensor
    * @{
    */

   /**
    * 记录了张量的核心数据的类型, 核心数据指的是除了角标名称之外的信息, 包括边的形状, 以及张量内本身的数据
    * \tparam ScalarType 张量内本身的数据的标量类型
    * \tparam Symmetry 张量所拥有的对称性
    * \note Core的存在是为了让边的名称的重命名节省时间
    */
   template<typename ScalarType, typename Symmetry, template<typename> class Allocator = std::allocator>
   struct Core {
      using edge_type = Edge<Symmetry, Allocator>;
      using symmetry_vector = std::vector<Symmetry, Allocator<Symmetry>>;
      using edge_vector = std::vector<edge_type, Allocator<edge_type>>;
      using content_vector = pmr::content_vector<ScalarType>;
      /**
       * 张量的形状, 是边的形状的列表, 列表长度为张量的秩, 每个边是一个对称性值到子边长度的映射表
       * \see Edge
       */
      edge_vector edges = {};

      using normal_map =
            std::map<symmetry_vector, content_vector, std::less<symmetry_vector>, Allocator<std::pair<const symmetry_vector, content_vector>>>;
      using fake_block_map = fake_map<symmetry_vector, content_vector>;
#ifdef TAT_USE_SIMPLE_NOSYMMETRY
      using block_map = std::conditional_t<std::is_same_v<Symmetry, NoSymmetry>, fake_block_map, normal_map>;
#else
      using block_map = normal_map;
#endif

      std::vector<ScalarType, Allocator<ScalarType>> storage;
      pmr::monotonic_buffer_resource* resource;

      /**
       * 张量内本身的数据, 是对称性列表到数据列表的映射表, 数据列表就是张量内本身的数据,
       * 而对称性列表表示此子块各个子边在各自的边上所对应的对称性值
       */
      block_map blocks = {};

      /**
       * 根据边的形状构造张量, 然后根据对称性条件自动构造张量的分块
       * \param initial_edge 边的形状的列表
       * \param auto_reverse 对于费米张量是否自动对含有负对称值的边整个取反
       * \note 使用auto_reverse时, 原则上构造时费米对称性值应该全正或全负, 如果不是这样, 结果会难以理解
       * \note 将会自动删除不出现于数据中的对称性
       */
      template<typename VectorEdge = pmr::vector<Edge<Symmetry>>>
      Core(const VectorEdge& initial_edge, [[maybe_unused]] const bool auto_reverse = false) : edges(initial_edge.begin(), initial_edge.end()) {
         // 自动翻转边
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            if (auto_reverse) {
               for (auto& edge : edges) {
                  edge.possible_reverse();
               }
            }
         }
         // 生成数据
         const auto symmetries_list = initialize_block_symmetries_with_check(edges);
         Size total_size = 0;
         for (const auto& [symmetries, size] : symmetries_list) {
            total_size += size;
         }
         storage.resize(total_size);
         resource = new pmr::monotonic_buffer_resource(storage.data(), total_size * sizeof(ScalarType));
         for (const auto& [symmetries, size] : symmetries_list) {
            blocks.emplace(symmetry_vector(symmetries.begin(), symmetries.end()), content_vector(size, resource));
         }
         // 删除不在block中用到的symmetry
         const Rank rank = edges.size();
         auto pmr_guard = scope_resource<1 << 10>();
         auto edge_mark = pmr::vector<pmr::map<Symmetry, bool>>();
         edge_mark.reserve(rank);
         for (const auto& edge : edges) {
            auto& this_mark = edge_mark.emplace_back();
            for (const auto& [symmetry, _] : edge.map) {
               this_mark[symmetry] = true;
            }
         }
         for (const auto& [symmetries, _] : blocks) {
            for (Rank i = 0; i < rank; i++) {
               edge_mark[i].at(symmetries[i]) = false;
            }
         }
         for (Rank i = 0; i < rank; i++) {
            for (const auto& [symmetry, flag] : edge_mark[i]) {
               if (flag) {
                  edges[i].map.erase(symmetry);
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
   /**@}*/
} // namespace TAT
#endif
