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

      template<
            class T = std::map<Symmetry, Size>,
            class = std::enable_if_t<std::is_convertible_v<T, std::map<Symmetry, Size>>>>
      BoseEdge(T&& t) : map(std::forward<T>(t)) {}
      BoseEdge(std::initializer_list<std::pair<const Symmetry, Size>> t) : map(t) {}

      BoseEdge(const Size s) : map({{Symmetry(), s}}) {}
   };
   template<class Symmetry>
   bool operator==(const BoseEdge<Symmetry>& e1, const BoseEdge<Symmetry>& e2) {
      return e1.map == e2.map;
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

      template<
            class T = std::map<Symmetry, Size>,
            class = std::enable_if_t<std::is_convertible_v<T, std::map<Symmetry, Size>>>>
      FermiEdge(T&& t) : map(std::forward<T>(t)) {}
      FermiEdge(std::initializer_list<std::pair<const Symmetry, Size>> t) : map(t) {}

      template<
            class T = std::map<Symmetry, Size>,
            class = std::enable_if_t<std::is_convertible_v<T, std::map<Symmetry, Size>>>>
      FermiEdge(const Arrow arrow, T&& boson) : arrow(arrow), map(std::forward<T>(boson)) {}

      void possible_reverse() {
         for (const auto& [i, j] : map) {
            if (i.fermi < 0) {
               arrow ^= true;
               return;
            }
         }
      }

      [[nodiscard]] bool arrow_valid() const {
         for (const auto& [i, j] : map) {
            if (i.fermi != 0) {
               return true;
            }
         }
         return false;
      }
   };
   template<class Symmetry>
   bool operator==(const FermiEdge<Symmetry>& e1, const FermiEdge<Symmetry>& e2) {
      return e1.map == e2.map && e1.arrow == e2.arrow;
   }

   template<class Symmetry>
   using EdgeBase =
         std::conditional_t<is_fermi_symmetry_v<Symmetry>, FermiEdge<Symmetry>, BoseEdge<Symmetry>>;
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
   std::ostream& operator<<(std::ostream& out, const Edge<Symmetry>& edge);
   template<class Symmetry>
   std::ostream& operator<=(std::ostream& out, const Edge<Symmetry>& edge);
   template<class Symmetry>
   std::istream& operator>=(std::istream& in, Edge<Symmetry>& edge);

   template<class Symmetry>
   struct PtrBoseEdge {
      using symmetry_type = Symmetry;

      const std::map<Symmetry, Size>* map;

      PtrBoseEdge(const std::map<Symmetry, Size>* m) : map(m) {}
   };
   template<class Symmetry>
   struct PtrFermiEdge {
      using symmetry_type = Symmetry;

      Arrow arrow;
      const std::map<Symmetry, Size>* map;

      PtrFermiEdge(Arrow a, const std::map<Symmetry, Size>* m) : arrow(a), map(m) {}

      [[nodiscard]] bool arrow_valid() const {
         for (const auto& [i, j] : *map) {
            if (i.fermi != 0) {
               return true;
            }
         }
         return false;
      }
   };
   template<class Symmetry>
   using PtrEdgeBase = std::conditional_t<
         is_fermi_symmetry_v<Symmetry>,
         PtrFermiEdge<Symmetry>,
         PtrBoseEdge<Symmetry>>;
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
    * \tparam T 应是vector<Edge>或者vector<PtrEdge>
    * \param edges 即将要枚举的边列表
    * \param rank0 如果边列表为空，则调用rank0后返回
    * \param check 对于边列表划分的每个分块, 使用check进行检查, check的参数是Edge中map的iterator的列表
    * \param append 如果check检查成功, 则运行append, append的参数与check相同
    * \param update 每次append前将根据append的输入参数列表中, 变化了的元, 进行最小化的更新,
    * 第一个参数与append相同, 第二个参数min_ptr表示仅需要更新从[min_ptr,rank)的元素
    * \see std_begin, initialize_block_symmetries_with_check, get_merged_edge
    */
   template<class T, class F1, class F2, class F3, class F4>
   void loop_edge(const T& edges, F1&& rank0, F2&& check, F3&& append, F4&& update) {
      const Rank rank = edges.size();
      if (!rank) {
         rank0();
         return;
      }
      using Symmetry = typename T::value_type::symmetry_type;
      using PosType = vector<typename std::map<Symmetry, Size>::const_iterator>;
      auto pos = PosType();
      for (const auto& i : edges) {
         const auto& map = remove_pointer(i.map);
         auto ptr = map.begin();
         if (ptr == map.end()) {
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
         while (pos[ptr] == remove_pointer(edges[ptr].map).end()) {
            if (ptr == 0) {
               return;
            }
            pos[ptr] = remove_pointer(edges[ptr].map).begin();
            --ptr;
            ++pos[ptr];
         }
         min_ptr = min_ptr < ptr ? min_ptr : ptr;
      }
   }

   /**
    * \brief 根据边的形状的列表, 得到所有满足对称性条件的张量分块
    * \return 分块信息, 为一个vector, 元素为两个类型的tuple, 分别是子块的各个子边对称性值和子块的总大小
    * \tparam T 为vector<Edge>或者vector<PtrEdge>
    * \see loop_edge
    */
   template<class T>
   auto initialize_block_symmetries_with_check(const T& edges) {
      using Symmetry = typename T::value_type::symmetry_type;
      using PosType = vector<typename std::map<Symmetry, Size>::const_iterator>;
      auto res = vector<std::tuple<vector<Symmetry>, Size>>();
      auto vec = vector<Symmetry>(edges.size());
      auto size = vector<Size>(edges.size());
      loop_edge(
            edges,
            [&res]() {
               res.push_back({vector<Symmetry>{}, 1});
            },
            []([[maybe_unused]] const PosType& pos) {
               auto sum = Symmetry();
               for (const auto& i : pos) {
                  sum += i->first;
               }
               return sum == Symmetry();
            },
            [&res, &vec, &size]([[maybe_unused]] const PosType& pos) {
               res.push_back({vec, size.back()});
            },
            [&vec, &size](const PosType& pos, const Rank ptr) {
               for (auto i = ptr; i < pos.size(); i++) {
                  vec[i] = pos[i]->first;
                  size[i] = pos[i]->second * (i ? size[i - 1] : 1);
               }
            });
      return res;
   }

   /**
    * \brief 获取一些已知形状的边合并之后的形状
    * \param edges_to_merge 已知的边的形状列表
    * \note 需要调用者保证输入费米箭头方向相同
    * \note 且并不设定结果的费米箭头方向
    */
   template<class T>
   [[nodiscard]] auto get_merged_edge(const T& edges_to_merge) {
      using Symmetry = typename T::value_type::symmetry_type;
      using PosType = vector<typename std::map<Symmetry, Size>::const_iterator>;

      auto res_edge = Edge<Symmetry>();

      auto sym = vector<Symmetry>(edges_to_merge.size());
      auto dim = vector<Size>(edges_to_merge.size());

      loop_edge(
            edges_to_merge,
            [&res_edge]() { res_edge.map[Symmetry()] = 1; },
            []([[maybe_unused]] const PosType& pos) { return true; },
            [&res_edge, &sym, &dim]([[maybe_unused]] const PosType& pos) {
               res_edge.map[sym[pos.size() - 1]] += dim[pos.size() - 1];
            },
            [&sym, &dim](const PosType& pos, const Rank start) {
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
            });

      return res_edge;
   }
} // namespace TAT
#endif
