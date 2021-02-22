/**
 * \file edge.hpp
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
#ifndef TAT_EDGE_HPP
#define TAT_EDGE_HPP

#include <map>
#include <set>

#include "basic_type.hpp"
#include "pmr_resource.hpp"
#include "symmetry.hpp"

namespace TAT {
   /** \defgroup Edge
    * @{
    */

   TAT_CHECK_MEMBER(map)

   template<typename Symmetry, bool is_pointer = false>
   struct edge_map_t {
      using symmetry_t = Symmetry;
      using map_t = pmr::map<Symmetry, Size>;
      std::conditional_t<is_pointer, const map_t&, map_t> map;
      // TODO bool conjugated;

      edge_map_t() = default;

      edge_map_t(const edge_map_t&) = default;
      edge_map_t(const edge_map_t& other, const map_t::allocator_type& alloc) : map(other, alloc) {}
      edge_map_t(const map_t& map) : map(map) {}
      edge_map_t(const map_t& map, const map_t::allocator_type& alloc) : map(map, alloc) {}

      edge_map_t(edge_map_t&&) noexcept = default;
      edge_map_t(edge_map_t&& other, const map_t::allocator_type& alloc) : map(std::move(other), alloc) {}
      edge_map_t(map_t&& map) : map(std::move(map)) {}
      edge_map_t(map_t&& map, const map_t allocator_type& alloc) : map(std::move(map), alloc) {}

      edge_map_t& operator=(const edge_map_t&) = default;
      edge_map_t& operator=(edge_map_t&&) noexcept = default;
      ~edge_map_t() = default;

      template<typename InputIt>
      edge_map_t(InputIt first, InputIt last) : map(first, last) {}
      template<typename InputIt>
      edge_map_t(InputIt first, InputIt last, const map_t::allocator_type& alloc) : map(first, last, alloc) {}

      template<typename OtherEdge, std::enable_if_t<has_map_v<OtherEdge>, int> = 0>
      edge_map_t(const OtherEdge& edge) : map(edge.map) {}
      template<typename OtherEdge, std::enable_if_t<has_map_v<OtherEdge>, int> = 0>
      edge_map_t(const OtherEdge& edge, const map_t::allocator_type& alloc) : map(edge.map, alloc) {}
      template<typename MapSymmetrySize, std::enable_if_t<is_map_of_v<MapSymmetrySize, Symmetry, Size>, int> = 0>
      edge_map_t(const MapSymmetrySize& map) : map(map.begin(), map.end()) {}
      template<typename MapSymmetrySize, std::enable_if_t<is_map_of_v<MapSymmetrySize, Symmetry, Size>, int> = 0>
      edge_map_t(const MapSymmetrySize& map, const map_t::allocator_type& alloc) : map(map.begin(), map.end(), alloc) {}

      edge_map_t(const std::initializer_list<std::pair<const Symmetry, Size>>& map) : map(map) {}
      edge_map_t(const std::initializer_list<std::pair<const Symmetry, Size>>& map, map_t& alloc) : map(map, alloc) {}

      /**
       * 由一些对称性的集合构造, 意味着每一个对称性对应的维度都为1
       */
      template<typename SetSymmetry, std::enable_if_t<is_set_of_v<SetSymmetry, Symmetry>, int> = 0>
      edge_map_t(const SetSymmetry& symmetries) {
         for (const auto& symmetry : symmetries) {
            map[symmetry] = 1;
         }
      }
      edge_map_t(const std::initializer_list<Symmetry>& symmetries) {
         for (const auto& symmetry : symmetries) {
            map[symmetry] = 1;
         }
      }

      /**
       * 构造一个平凡的边, 仅含一个对称性
       */
      edge_map_t(const Size dimension) : map({{Symmetry(), dimension}}) {}
   };

   struct edge_nosymmetry_conjugated_t {
      static constexpr bool conjugated = false;
   };
   struct edge_symmetry_conjugated_t {
      bool conjugated;
      edge_symmetry_conjugated_t() : conjugated(0) {}
      edge_symmetry_conjugated_t(bool conjugated) : conjugated(conjugated) {}
   };

   template<bool is_nosymmetry>
   using edge_conjugated_t = std::conditional_t<is_nosymmetry, edge_nosymmetry_conjugated_t, edge_symmetry_conjugated_t>;

   struct edge_bose_arrow_t {
      static constexpr Arrow arrow = 0;
   };
   struct edge_fermi_arrow_t {
      Arrow arrow;
      edge_fermi_arrow_t() : arrow(0) {}
      edge_fermi_arrow_t(Arrow arrow) : arrow(arrow) {}
   };
   template<bool is_fermi>
   using edge_arrow_t = std::conditional_t<is_fermi, edge_fermi_arrow_t, edge_bose_arrow_t>;

   /**
    * 张量的边的形状的类型, 是一个Symmetry到Size的映射表, 如果是费米对称性, 还会含有一个箭头方向
    * \tparam Symmetry 张量所拥有的对称性
    * \tparam is_pointer map是否为引用而非真是存储着数据的伪边
    */
   template<typename Symmetry, bool is_pointer = false>
   struct Edge : edge_map_t<Symmetry, is_pointer>, edge_conjugated_t<Symmetry::length == 0>, edge_arrow_t<Symmetry::is_fermi_symmetry> {
      using base_map_t = edge_map_t<Symmetry, is_pointer>;
      using base_conjugated_t = edge_conjugated_t<Symmetry::length == 0>;
      using base_arrow_t = edge_arrow_t<Symmetry::is_fermi_symmetry>;

      using base_arrow_t::arrow;
      using base_conjugated_t::conjugated;
      using base_map_t::map;

      Edge() = default;
      Edge(const Edge&) = default;
      Edge(Edge&&) noexcept = default;
      Edge& operator=(const Edge&) = default;
      Edge& operator=(Edge&&) noexcept = default;
      ~Edge() = default;

      template<typename... Args>
      Edge(Args&&... args) : base_map_t(std::forward<Args>(args)...) {}
      template<typename... Args>
      Edge(const std::initializer_list<std::pair<const Symmetry, Size>>& map, Args&&... args) : base_map_t(map, std::forward<Args>(args)...) {}
      template<typename... Args>
      Edge(const std::initializer_list<Symmetry>& symmetries, Args&&... args) : base_map_t(symmetries, std::forward<Args>(args)...) {}

      template<typename... Args>
      Edge(Arrow arrow, Args&&... args) : base_map_t(std::forward<Args>(args)...), base_arrow_t(arrow) {}
      template<typename... Args>
      Edge(Arrow arrow, const std::initializer_list<std::pair<const Symmetry, Size>>& map, Args&&... args) :
            base_map_t(map, std::forward<Args>(args)...), base_arrow_t(arrow) {}
      template<typename... Args>
      Edge(Arrow arrow, const std::initializer_list<Symmetry>& symmetries, Args&&... args) :
            base_map_t(symmetries, std::forward<Args>(args)...), base_arrow(arrow) {}

      /**
       * 由费米子数自动构造箭头方向, 虽然这个不一定需要一致, 仅仅在只含有一个Fermi对称性时有效
       */
      void possible_reverse() {
         for (const auto& [symmetry, size] : map) {
            if (symmetry.get_first_parity() < 0) {
               arrow ^= true;
               return;
            }
         }
      }

      /**
       * 检查箭头是否有效, 即含有非零的费米子数
       */
      [[nodiscard]] bool arrow_valid() const {
         for (const auto& [symmetry, size] : map) {
            if (symmetry.get_first_parity() != 0) {
               return true;
            }
         }
         return false;
      }
   };
   template<typename Symmetry, bool is_pointer = false>
   bool operator==(const Edge<Symmetry, is_pointer>& edge_1, const Edge<Symmetry, is_pointer>& edge_2) {
      return edge_1.arrow == edge_2.arrow && std::equal(edge_1.map.begin(), edge_1.map.end(), edge_2.map.begin(), edge_2.map.end());
   }

   /**
    * 中间处理中常用到的数据类型, 类似Edge但是其中对称性值到子边长的映射表为指针
    * \see Edge
    */
   template<typename Symmetry>
   using EdgePointer = Edge<Symmetry, true>;

   /**
    * 对一个边的形状列表进行枚举分块, 并做一些其他操作
    * \tparam T 应是vector<Edge>或者vector<EdgePointer>的iterator
    * \param edges 即将要枚举的边列表的开头指针
    * \param rank 即将要枚举的边列表的大小
    * \param rank0 如果边列表为空，则调用rank0后返回
    * \param dims0 如果边列表中存在零维的边，则调用dims0后返回
    * \param operate 对枚举的每一个情况做操作
    * \note operate输入两个参数, 一个是每个边所在的位置列表, 一个是需要更新信息的位置开头, 并返回操作后需要更新的位置开头
    * \see initialize_block_symmetries_with_check, get_merged_edge
    */
   template<typename Edge, typename Func1, typename Func2, typename Func3>
   void loop_edge(const Edge* edges, const Rank rank, Func1&& rank0, Func2&& dims0, Func3&& operate) {
      if (rank == 0) {
         rank0();
         return;
      }
      using Symmetry = typename Edge::symmetry_t;
      auto symmetry_iterator_list = pmr::vector<typename Edge::map_t::const_iterator>();
      symmetry_iterator_list.reserve(rank);
      for (auto i = 0; i != rank; ++i) {
         const auto& map = edges[i].map;
         if (edges[i].map.empty()) {
            dims0();
            return;
         }
         // 如果没有conjugated，指向当前位置，如果conjugated，指向当前位置的下一个位置
         symmetry_iterator_list.push_back(edges[i].conjugated ? std::prev(map.end()) : map.begin());
      }
      Rank minimum_changed = 0;
      while (true) {
         minimum_changed = operate(symmetry_iterator_list, minimum_changed);
         auto edge_position = rank - 1;

         while (edges[edge_position].conjugated ? symmetry_iterator_list[edge_position]-- == edges[edge_position].map.begin() :
                                                  ++symmetry_iterator_list[edge_position] == edges[edge_position].map.end()) {
            if (edge_position == 0) {
               return;
            }
            symmetry_iterator_list[edge_position] =
                  edges[edge_position].conjugated ? std::prev(edge[edge_position].map.end()) : edges[edge_position].map.begin();
            --edge_position;
         }
         minimum_changed = minimum_changed < edge_position ? minimum_changed : edge_position;
      }
   }

   /// \private
   template<typename Edges>
   [[nodiscard]] auto initialize_block_symmetries_with_check(const Edges& edges) {
      using Symmetry = typename Edges::value_type::symmetry_t;
      Rank rank = edges.size();
      auto result = pmr::vector<std::tuple<pmr::vector<Symmetry>, Size>>();
      auto symmetries = pmr::vector<Symmetry>(rank);
      auto sizes = pmr::vector<Size>(rank);
      loop_edge(
            edges.data(),
            rank,
            [&result]() {
               result.push_back({pmr::vector<Symmetry>{}, 1});
            },
            []() {},
            [&](const auto& symmetry_iterator_list, Rank minimum_changed) {
               auto symmetry_summary = Symmetry();
               for (const auto& symmetry_iterator : symmetry_iterator_list) {
                  symmetry_summary += symmetry_iterator->first;
               }
               if (symmetry_summary == Symmetry()) {
                  for (auto i = minimum_changed; i < rank; i++) {
                     symmetries[i] = symmetry_iterator_list[i]->first;
                     sizes[i] = symmetry_iterator_list[i]->second * (i ? sizes[i - 1] : 1);
                  }
                  result.push_back({symmetries, sizes.back()});
                  return rank;
               }
               return minimum_changed;
            });
      return result;
   }

   /**
    * 判断一个类型是否为Edge类型, 这里不认为map为引用的Edge类型为Edge
    * \tparam T 如果T是Edge类型, 则value为true
    * \see is_edge_v
    */
   template<typename T>
   struct is_edge : std::bool_constant<false> {};
   /// \private
   template<typename T>
   struct is_edge<Edge<T>> : std::bool_constant<true> {};
   template<typename T>
   constexpr bool is_edge_v = is_edge<T>::value;

   /// \private
   template<typename VectorSymmetry, typename VectorArrow>
   [[nodiscard]] auto get_edge_from_edge_symmetry_and_arrow(const VectorSymmetry& edge_symmetry, const VectorArrow& edge_arrow, Rank rank) {
      using Symmetry = typename VectorSymmetry::value_type;
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         return pmr::vector<Edge<Symmetry>>(rank, {1});
      } else {
         auto result = pmr::vector<Edge<Symmetry>>();
         result.reserve(rank);
         for (auto i = 0; i < rank; i++) {
            if constexpr (is_fermi_symmetry_v<Symmetry>) {
               result.push_back({edge_arrow[i], {{edge_symmetry[i], 1}}});
            } else {
               result.push_back({{{edge_symmetry[i], 1}}});
            }
         }
         return result;
      }
   }
   /**@}*/
} // namespace TAT
#endif
