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
   /**
    * \see Edge
    */
   template<class Symmetry>
   struct BoseEdge {
      using symmetry_type = Symmetry;

      std::map<Symmetry, Size> map = {};

      BoseEdge() = default;
      BoseEdge(const BoseEdge&) = default;
      BoseEdge(BoseEdge&&) = default;
      BoseEdge& operator=(const BoseEdge&) = default;
      BoseEdge& operator=(BoseEdge&&) = default;
      ~BoseEdge() = default;

      template<class T = std::map<Symmetry, Size>, class = std::enable_if_t<std::is_convertible_v<T, std::map<Symmetry, Size>>>>
      BoseEdge(T&& map) : map(std::forward<T>(map)) {}
      BoseEdge(std::initializer_list<std::pair<const Symmetry, Size>> map) : map(map) {}

      BoseEdge(const std::vector<Symmetry>& symmetries) {
         for (const auto& symmetry : symmetries) {
            map[symmetry] = 1;
         }
      }
      BoseEdge(const std::initializer_list<Symmetry>& symmetries) : BoseEdge(std::vector<Symmetry>(symmetries)) {}
      BoseEdge(const Size dimension) : map({{Symmetry(), dimension}}) {}
   };
   template<class Symmetry>
   bool operator==(const BoseEdge<Symmetry>& edge_1, const BoseEdge<Symmetry>& edge_2) {
      return edge_1.map == edge_2.map;
   }

   /**
    * \see Edge
    */
   template<class Symmetry>
   struct FermiEdge {
      using symmetry_type = Symmetry;

      /**
       * \brief 费米箭头方向
       * \note 当map中只含fermi=0的对称性值时, arrow无法定义,
       * 这在possible_reverse中得到体现
       * \see arrow_valid
       */
      Arrow arrow = false;
      std::map<Symmetry, Size> map = {};

      FermiEdge() = default;
      FermiEdge(const FermiEdge&) = default;
      FermiEdge(FermiEdge&&) = default;
      FermiEdge& operator=(const FermiEdge&) = default;
      FermiEdge& operator=(FermiEdge&&) = default;
      ~FermiEdge() = default;

      template<class T = std::map<Symmetry, Size>, class = std::enable_if_t<std::is_convertible_v<T, std::map<Symmetry, Size>>>>
      FermiEdge(T&& map) : map(std::forward<T>(map)) {}
      FermiEdge(std::initializer_list<std::pair<const Symmetry, Size>> map) : map(map) {}

      FermiEdge(const std::vector<Symmetry>& symmetries) {
         for (const auto& symmetry : symmetries) {
            map[symmetry] = 1;
         }
      }
      FermiEdge(const std::initializer_list<Symmetry>& symmetries) : FermiEdge(std::vector<Symmetry>(symmetries)) {}
      FermiEdge(const Size dimension) : map({{Symmetry(), dimension}}) {}

      template<class T = std::map<Symmetry, Size>, class = std::enable_if_t<std::is_convertible_v<T, std::map<Symmetry, Size>>>>
      FermiEdge(const Arrow arrow, T&& map) : arrow(arrow), map(std::forward<T>(map)) {}
      FermiEdge(const Arrow arrow, std::initializer_list<std::pair<const Symmetry, Size>> map) : arrow(arrow), map(map) {}

      void possible_reverse() {
         for (const auto& [symmetry, size] : map) {
            if (symmetry.fermi < 0) {
               arrow ^= true;
               return;
            }
         }
      }

      [[nodiscard]] bool arrow_valid() const {
         for (const auto& [symmetry, size] : map) {
            if (symmetry.fermi != 0) {
               return true;
            }
         }
         return false;
      }
   };
   template<class Symmetry>
   bool operator==(const FermiEdge<Symmetry>& edge_1, const FermiEdge<Symmetry>& edge_2) {
      return edge_1.map == edge_2.map && edge_1.arrow == edge_2.arrow;
   }

   template<class Symmetry>
   using EdgeBase = std::conditional_t<is_fermi_symmetry_v<Symmetry>, FermiEdge<Symmetry>, BoseEdge<Symmetry>>;
   /**
    * \brief 张量的边的形状的类型, 是一个Symmetry到Size的映射表, 如果是费米对称性, 还会含有一个箭头方向
    * \tparam Symmetry 张量所拥有的对称性
    * \see BoseEdge, FermiEdge
    */
   template<class Symmetry, class = std::enable_if_t<is_symmetry_v<Symmetry>>>
   struct Edge : public EdgeBase<Symmetry> {
      using EdgeBase<Symmetry>::EdgeBase;
   };

   template<class Symmetry>
   struct PtrBoseEdge {
      using symmetry_type = Symmetry;

      const std::map<Symmetry, Size>* map;

      PtrBoseEdge() = default;
      PtrBoseEdge(const PtrBoseEdge&) = default;
      PtrBoseEdge(PtrBoseEdge&&) = default;
      PtrBoseEdge& operator=(const PtrBoseEdge&) = default;
      PtrBoseEdge& operator=(PtrBoseEdge&&) = default;
      ~PtrBoseEdge() = default;

      PtrBoseEdge(const std::map<Symmetry, Size>* m) : map(m) {}
   };
   template<class Symmetry>
   struct PtrFermiEdge {
      using symmetry_type = Symmetry;

      Arrow arrow = false;
      const std::map<Symmetry, Size>* map = nullptr;

      PtrFermiEdge() = default;
      PtrFermiEdge(const PtrFermiEdge&) = default;
      PtrFermiEdge(PtrFermiEdge&&) = default;
      PtrFermiEdge& operator=(const PtrFermiEdge&) = default;
      PtrFermiEdge& operator=(PtrFermiEdge&&) = default;
      ~PtrFermiEdge() = default;

      PtrFermiEdge(const Arrow arrow, const std::map<Symmetry, Size>* map) : arrow(arrow), map(map) {}

      [[nodiscard]] bool arrow_valid() const {
         for (const auto& [symmetry, size] : *map) {
            if (symmetry.fermi != 0) {
               return true;
            }
         }
         return false;
      }
   };
   template<class Symmetry>
   using PtrEdgeBase = std::conditional_t<is_fermi_symmetry_v<Symmetry>, PtrFermiEdge<Symmetry>, PtrBoseEdge<Symmetry>>;
   /**
    * \brief 中间处理中常用到的数据类型, 类似Edge但是其中对称性值到子边长的映射表为指针
    * \see Edge
    */
   template<class Symmetry>
   struct PtrEdge : PtrEdgeBase<Symmetry> {
      using PtrEdgeBase<Symmetry>::PtrEdgeBase;
   };

   /**
    * \brief PtrEdge的辅助函数, 用来提取其中的map
    * \note 使用方式是remove_pointer(edge.map)
    */
   template<class T>
   const auto& remove_pointer(const T& v) {
      if constexpr (std::is_pointer_v<T>) {
         return *v;
      } else {
         return v;
      }
   }

   /**
    * \brief 对一个边的形状列表进行枚举分块, 并做一些其他操作
    * \tparam T 应是vector<Edge>或者vector<PtrEdge>的iterator
    * \param edges 即将要枚举的边列表的开头指针
    * \param rank 即将要枚举的边列表的大小
    * \param rank0 如果边列表为空，则调用rank0后返回
    * \param dims0 如果边列表中存在零维的边，则调用dims0后返回
    * \param operate 对枚举的每一个情况做操作
    * \note operate输入两个参数, 一个是每个边所在的位置列表, 一个是需要更新信息的位置开头, 并返回操作后需要更新的位置开头
    * \see initialize_block_symmetries_with_check, get_merged_edge
    */
   template<class T, class F1, class F2, class F3>
   void loop_edge(const T* edges, const Rank rank, F1&& rank0, F2&& dims0, F3&& operate) {
      if (rank == 0) {
         rank0();
         return;
      }
      using Symmetry = typename T::symmetry_type;
      using MapIteratorList = std::vector<typename std::map<Symmetry, Size>::const_iterator>;
      auto symmetry_iterator_list = MapIteratorList();
      for (auto i = 0; i != rank; ++i) {
         const auto& map = remove_pointer(edges[i].map);
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
         while (symmetry_iterator_list[edge_position] == remove_pointer(edges[edge_position].map).end()) {
            if (edge_position == 0) {
               return;
            }
            symmetry_iterator_list[edge_position] = remove_pointer(edges[edge_position].map).begin();
            --edge_position;
            ++symmetry_iterator_list[edge_position];
         }
         minimum_changed = minimum_changed < edge_position ? minimum_changed : edge_position;
      }
   }

   /**
    * \brief 根据边的形状的列表, 得到所有满足对称性条件的张量分块
    * \return 分块信息, 为一个vector, 元素为两个类型的tuple, 分别是子块的各个子边对称性值和子块的总大小
    * \tparam T 为vector<Edge>或者vector<PtrEdge>
    * \see loop_edge
    */
   template<class T>
   [[nodiscard]] auto initialize_block_symmetries_with_check(const T& edges) {
      using Symmetry = typename T::value_type::symmetry_type;
      using MapIteratorList = std::vector<typename std::map<Symmetry, Size>::const_iterator>;
      auto result = std::vector<std::tuple<std::vector<Symmetry>, Size>>();
      auto symmetries = std::vector<Symmetry>(edges.size());
      auto sizes = std::vector<Size>(edges.size());
      Rank rank = edges.size();
      loop_edge(
            edges.data(),
            rank,
            [&result]() {
               result.push_back({std::vector<Symmetry>{}, 1});
            },
            []() {},
            [&](const MapIteratorList& symmetry_iterator_list, Rank minimum_changed) {
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
} // namespace TAT
#endif
