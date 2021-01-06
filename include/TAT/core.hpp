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
    * \defgroup Miscellaneous
    * @{
    */
   /**
    * 用于不初始化的`vector`的`allocator`, 仅用于张量数据的存储
    *
    * \see vector
    */
   template<typename T>
   struct allocator_without_initialize : std::allocator<T> {
      template<typename U>
      struct rebind {
         using other = allocator_without_initialize<U>;
      };

      /**
       * 初始化函数, 如果没有参数, 且类型T可以被平凡的析构, 则不做任何初始化操作, 否则进行正常的就地初始化
       * \tparam Args 初始化的参数类型
       * \param pointer 被初始化的值的地址
       * \param arguments 初始化的参数
       * \note c++20废弃了std::allocator<T>的construct, 但是c++20的行为是检测allocator是否有construct, 有则调用没有则自己construct, 所以没关系
       */
      template<typename... Args>
      void construct([[maybe_unused]] T* pointer, Args&&... arguments) {
         if constexpr (!((sizeof...(arguments) == 0) && (std::is_trivially_destructible_v<T>))) {
            new (pointer) T(std::forward<Args>(arguments)...);
         }
      }

      /**
       * 对齐地分配内存, 对齐方式`lcm(1024, alignof(T))`
       */
      T* allocate(std::size_t n) {
         constexpr auto align = std::align_val_t(std::lcm(1024, alignof(T)));
         return (T*)operator new(n * sizeof(T), align);
      }

      /**
       * 释放对齐分配的内存
       */
      void deallocate(T* p, std::size_t n) {
         constexpr auto align = std::align_val_t(std::lcm(1024, alignof(T)));
         operator delete(p, align);
      }

      allocator_without_initialize() = default;
      template<typename U>
      explicit allocator_without_initialize(allocator_without_initialize<U>) {}
   };

   /**
    * 尽可能不做初始化的vector容器
    * \see allocator_without_initialize
    * \note 为了其他部分与stl兼容性, 仅在张量的数据处使用
    */
   template<typename T>
   struct vector : std::vector<T, allocator_without_initialize<T>> {
      using std::vector<T, allocator_without_initialize<T>>::vector;

      vector(const std::vector<T>& other) : vector(other.begin(), other.end()) {}
      operator std::vector<T>() const {
         using Base = std::vector<T, allocator_without_initialize<T>>;
         return std::vector<T>(Base::cbegin(), Base::cend());
      }
   };

   /**@}*/
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
   template<typename ScalarType, typename Symmetry>
   struct Core {
      /**
       * 张量的形状, 是边的形状的列表, 列表长度为张量的秩, 每个边是一个对称性值到子边长度的映射表
       * \see Edge
       */
      std::vector<Edge<Symmetry>> edges = {};

      using normal_map = std::map<std::vector<Symmetry>, vector<ScalarType>>;
      using fake_block_map = fake_map<std::vector<Symmetry>, vector<ScalarType>>;
#ifdef TAT_USE_SIMPLE_NOSYMMETRY
      using block_map = std::conditional_t<std::is_same_v<Symmetry, NoSymmetry>, fake_block_map, normal_map>;
#else
      using block_map = normal_map;
#endif
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
         auto pmr_guard = scope_resource<1 << 10>();
         // 自动翻转边
         if constexpr (is_fermi_symmetry_v<Symmetry>) {
            if (auto_reverse) {
               for (auto& edge : edges) {
                  edge.possible_reverse();
               }
            }
         }
         // 生成数据
         auto symmetries_list = initialize_block_symmetries_with_check(edges);
         for (auto& [symmetries, size] : symmetries_list) {
            blocks[{symmetries.begin(), symmetries.end()}] = vector<ScalarType>(size);
         }
         // 删除不在block中用到的symmetry
         const Rank rank = edges.size();
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
