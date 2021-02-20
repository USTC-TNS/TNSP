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
   /**
    * \defgroup Miscellaneous
    * @{
    */
   /**
    * 只有一个元素的假map
    *
    * 无对称性的系统为一个NoSymmetry到Size的map, 显然只有一个元素, 使用一个假map来节省一层指针, 在无对称性的block中也用到了这个类
    */
   template<typename Key, typename Value, typename Compare = void, typename Allocator = void>
   struct fake_map {
      using iterator = fake_map*;
      using const_iterator = const fake_map*;
      using key_type = Key;
      using mapped_type = Value;

      Key first;
      Value second;

      fake_map() : first(), second() {}
      template<typename Iterator>
      fake_map(Iterator iter) : first(iter->first), second(iter->second) {}
      template<typename Iterator>
      fake_map(Iterator iter, Iterator) : fake_map(iter) {}

      fake_map(const std::initializer_list<std::pair<const Key, Value>>& map) : fake_map(map.begin()) {}
      template<typename Map, typename = std::enable_if_t<is_map_of_v<Map, Key, Size>>>
      fake_map(const Map& map) : fake_map(map.begin()) {}

      [[nodiscard]] Value& at(const Key&) {
         return second;
      }
      [[nodiscard]] const Value& at(const Key&) const {
         return second;
      }
      Value& operator[](const Key&) {
         return second;
      }
      [[nodiscard]] iterator begin() {
         return this;
      }
      [[nodiscard]] const_iterator begin() const {
         return this;
      }
      [[nodiscard]] iterator end() {
         return this + 1;
      }
      [[nodiscard]] const_iterator end() const {
         return this + 1;
      }
      [[nodiscard]] iterator find(const Key&) {
         return this;
      }
      [[nodiscard]] const_iterator find(const Key&) const {
         return this;
      }
      iterator erase(const Key&) {
         return this;
      }
      std::pair<iterator, bool> insert(const std::tuple<Key, Value>& pair) {
         second = std::get<1>(pair);
         return {this, true};
      }
      template<typename T>
      std::pair<iterator, bool> emplace(const Key&, T&& arg) {
         second.~Value();
         new (&second) Value(std::forward<T>(arg));
         return {this, true};
      }
      Size size() const {
         return 1;
      }
      void clear() {}
   };
   template<typename Key, typename Value>
   bool operator==(const fake_map<Key, Value>& map_1, const fake_map<Key, Value>& map_2) {
      return map_1.second == map_2.second;
   }

#ifdef TAT_USE_SIMPLE_NOSYMMETRY
   constexpr bool use_simple_nosymmetry = true;
#else
   constexpr bool use_simple_nosymmetry = false;
#endif

   /**@}*/
   /** \defgroup Edge
    * @{
    */

   TAT_CHECK_MEMBER(map)

   template<typename Symmetry, template<typename> class Allocator = std::allocator, bool is_pointer = false>
   struct edge_map_t {
      using symmetry_t = Symmetry;
      using real_map_t = std::map<Symmetry, Size, std::less<Symmetry>, Allocator<std::pair<const Symmetry, Size>>>;
#ifdef TAT_USE_SIMPLE_NOSYMMETRY
      using map_t = std::conditional_t<std::is_same_v<Symmetry, NoSymmetry>, fake_map<Symmetry, Size>, real_map_t>;
#else
      using map_t = real_map_t;
#endif
      std::conditional_t<is_pointer, const map_t&, map_t> map;
      // TODO bool conjuugated;

      edge_map_t() = default;
      edge_map_t(const edge_map_t&) = default;
      edge_map_t(edge_map_t&&) noexcept = default;
      edge_map_t& operator=(const edge_map_t&) = default;
      edge_map_t& operator=(edge_map_t&&) noexcept = default;
      ~edge_map_t() = default;

      template<typename OtherEdge, std::enable_if_t<has_map_v<OtherEdge>, int> = 0>
      edge_map_t(const OtherEdge& edge) : edge_map_t(edge.map) {}
      template<typename MapSymmetrySize, std::enable_if_t<is_map_of_v<MapSymmetrySize, Symmetry, Size>, int> = 0>
      edge_map_t(const MapSymmetrySize& map) : map(map.begin(), map.end()) {}

      /**
       * 由对称性到维度的映射表直接构造
       */
      edge_map_t(map_t&& map) : map(std::move(map)) {}
      edge_map_t(const map_t& map) : map(map) {}
      edge_map_t(const std::initializer_list<std::pair<const Symmetry, Size>>& map) : map(map) {}

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
   template<typename Symmetry, template<typename> class Allocator = std::allocator, bool is_pointer = false>
   struct Edge : edge_map_t<Symmetry, Allocator, is_pointer>, edge_arrow_t<Symmetry::is_fermi_symmetry> {
      using base_map_t = edge_map_t<Symmetry, Allocator, is_pointer>;
      using base_arrow_t = edge_arrow_t<Symmetry::is_fermi_symmetry>;

      using base_arrow_t::arrow;
      using base_map_t::map;

      Edge() = default;
      Edge(const Edge&) = default;
      Edge(Edge&&) noexcept = default;
      Edge& operator=(const Edge&) = default;
      Edge& operator=(Edge&&) noexcept = default;
      ~Edge() = default;

      template<typename T>
      Edge(T&& arg) : base_map_t(std::forward<T>(arg)) {}
      Edge(const std::initializer_list<std::pair<const Symmetry, Size>>& map) : base_map_t(map) {}
      Edge(const std::initializer_list<Symmetry>& symmetries) : base_map_t(symmetries) {}

      template<typename T>
      Edge(Arrow arrow, T&& arg) : base_map_t(std::forward<T>(arg)), base_arrow_t(arrow) {}
      Edge(Arrow arrow, const std::initializer_list<std::pair<const Symmetry, Size>>& map) : base_map_t(map), base_arrow_t(arrow) {}
      Edge(Arrow arrow, const std::initializer_list<Symmetry>& symmetries) : base_map_t(symmetries), base_arrow_t(arrow) {}

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
   template<typename Symmetry, template<typename> class Allocator1, template<typename> class Allocator2, bool is_pointer = false>
   bool operator==(const Edge<Symmetry, Allocator1, is_pointer>& edge_1, const Edge<Symmetry, Allocator2, is_pointer>& edge_2) {
      return edge_1.arrow == edge_2.arrow && std::equal(edge_1.map.begin(), edge_1.map.end(), edge_2.map.begin(), edge_2.map.end());
   }

   /**
    * 中间处理中常用到的数据类型, 类似Edge但是其中对称性值到子边长的映射表为指针
    * \see Edge
    */
   template<typename Symmetry, template<typename> class Allocator = std::allocator>
   using EdgePointer = Edge<Symmetry, Allocator, true>;

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
   template<typename T, typename F1, typename F2, typename F3>
   void loop_edge(const T* edges, const Rank rank, F1&& rank0, F2&& dims0, F3&& operate) {
      if (rank == 0) {
         rank0();
         return;
      }
      using Symmetry = typename T::symmetry_t;
      auto symmetry_iterator_list = pmr::vector<typename T::map_t::const_iterator>();
      symmetry_iterator_list.reserve(rank);
      for (auto i = 0; i != rank; ++i) {
         const auto& map = edges[i].map;
         auto symmetry_iterator = map.begin();
         if (symmetry_iterator == map.end()) {
            dims0();
            return;
         }
         symmetry_iterator_list.push_back(symmetry_iterator);
      }
      Rank minimum_changed = 0;
      while (true) {
         minimum_changed = operate(symmetry_iterator_list, minimum_changed);
         auto edge_position = rank - 1;
         ++symmetry_iterator_list[edge_position];
         while (symmetry_iterator_list[edge_position] == edges[edge_position].map.end()) {
            if (edge_position == 0) {
               return;
            }
            symmetry_iterator_list[edge_position] = edges[edge_position].map.begin();
            --edge_position;
            ++symmetry_iterator_list[edge_position];
         }
         minimum_changed = minimum_changed < edge_position ? minimum_changed : edge_position;
      }
   }

   /// \private
   template<typename T>
   [[nodiscard]] auto initialize_block_symmetries_with_check(const T& edges) {
      using Symmetry = typename T::value_type::symmetry_t;
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
   template<typename T, template<typename> class Allocator>
   struct is_edge<Edge<T, Allocator>> : std::bool_constant<true> {};
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
