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

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "symmetry.hpp"

namespace TAT {
   template<typename Symmetry, bool _is_pointer = false>
   struct edge_segment_t {
      static_assert(is_symmetry<Symmetry>);

      static constexpr bool is_pointer = _is_pointer;
      static constexpr bool is_not_pointer = !is_pointer;

      using symmetry_t = Symmetry;
      using segment_t = std::vector<std::pair<Symmetry, Size>>;
      std::conditional_t<is_pointer, const segment_t&, segment_t> segment;
      using symlist_t = std::vector<Symmetry>;

      edge_segment_t() = default;
      edge_segment_t(const edge_segment_t& edge) = default;
      edge_segment_t(edge_segment_t&& edge) = default;
      edge_segment_t& operator=(const edge_segment_t&) = default;
      edge_segment_t& operator=(edge_segment_t&&) noexcept = default;
      ~edge_segment_t() = default;

      void check_valid_symmetry() const {
         Nums seg_num = segment.size();
         for (auto i = 0; i < seg_num; i++) {
            for (auto j = i + 1; j < seg_num; j++) {
               if (segment[i].first == segment[j].first) {
                  detail::error("Duplicated Symmetry in Edge Segment");
               }
            }
         }
      }

      // only valid if is_not_pointer
      edge_segment_t(segment_t&& s) : segment(std::move(s)) {
         static_assert(is_not_pointer);
         if constexpr (debug_mode) {
            check_valid_symmetry();
         }
      }
      // valid for both
      edge_segment_t(const segment_t& s) : segment(s) {
         static_assert(true);
         if constexpr (debug_mode) {
            check_valid_symmetry();
         }
      }

      /**
       * construct the edge with list of symmetry, each size of them are 1
       */
      edge_segment_t(const symlist_t& symmetries) {
         static_assert(is_not_pointer);
         segment.reserve(symmetries.size());
         for (const auto& symmetry : symmetries) {
            segment.push_back({symmetry, 1});
         }
         if constexpr (debug_mode) {
            check_valid_symmetry();
         }
      }

      /**
       * construct a trivial edge, only contain a single symmetry
       */
      edge_segment_t(const Size dimension, const Symmetry symmetry = Symmetry()) : segment({{symmetry, dimension}}) {
         static_assert(is_not_pointer);
         if constexpr (debug_mode) {
            check_valid_symmetry();
         }
      }

      using point_t = std::pair<symmetry_t, Size>;
      using index_t = Size;
      using position_t = Nums;

      point_t get_point_from_index(index_t index) const {
         for (const auto& [sym, dim] : segment) {
            if (index < dim) {
               return {sym, index};
            } else {
               index -= dim;
            }
         }
         detail::error("Index is more than edge total dimension");
      }

      Size get_index_from_point(const point_t& pair) const {
         Size result = pair.second;
         for (const auto& [sym, dim] : segment) {
            if (sym == pair.first) {
               return result;
            }
            result += dim;
         }
         detail::error("The symmetry not found in this edge");
      }

      auto find_by_symmetry(const symmetry_t& symmetry) const {
         return std::find_if(segment.begin(), segment.end(), [symmetry](const auto& x) {
            return symmetry == x.first;
         });
      }

      Size get_dimension_from_symmetry(const symmetry_t& symmetry) const {
         auto where = find_by_symmetry(symmetry);
         if constexpr (debug_mode) {
            if (where == segment.end()) {
               detail::error("No such symmetry in this edge");
            }
         }
         return where->second;
      }

      Size get_dimension_from_position(const position_t& position) const {
         if constexpr (debug_mode) {
            if (position >= segment.size()) {
               detail::error("Position is large than segment number");
            }
         }
         return segment[position].second;
      }

      symmetry_t get_symmetry_from_position(const position_t& position) const {
         if constexpr (debug_mode) {
            if (position >= segment.size()) {
               detail::error("Position is large than segment number");
            }
         }
         return segment[position].first;
      }

      position_t get_position_from_symmetry(const symmetry_t& symmetry) const {
         auto where = find_by_symmetry(symmetry);
         if constexpr (debug_mode) {
            if (where == segment.end()) {
               detail::error("No such symmetry in this edge");
            }
         }
         return std::distance(segment.begin(), where);
      }
      // position  : 0 1 2 3 4
      // symmetry  : x x x x x
      // dimension : d d d d d

      // point     : (sym, xxx)
      // index     : xxx

      void check_valid_reorder_symmetry(const std::vector<Symmetry>& symmetry_order) {
         if (symmetry_order.size() != segment.size()) {
            detail::error("Invalid new order when reorder symmetry of edge segment");
         }
         for (auto i = symmetry_order.begin(); i != symmetry_order.end(); ++i) {
            for (auto j = std::next(i); j != symmetry_order.end(); ++j) {
               if (*i == *j) {
                  detail::error("Duplicated symmetry in new symmetry list");
               }
            }
         }
      }

      void reorder_symmetry(const std::vector<Symmetry>& symmetry_order) {
         static_assert(is_not_pointer);
         if constexpr (debug_mode) {
            check_valid_reorder_symmetry(symmetry_order);
         }
         auto new_segment = segment_t();
         new_segment.reserve(segment.size());
         for (const auto& symmetry : symmetry_order) {
            new_segment.emplace_back(symmetry, get_dimension_from_symmetry(symmetry));
         }
         segment = std::move(new_segment);
      }

      edge_segment_t<symmetry_t> reordered_symmetry(const std::vector<Symmetry>& symmetry_order) const {
         if constexpr (debug_mode) {
            check_valid_reorder_symmetry(symmetry_order);
         }
         auto new_segment = segment_t();
         new_segment.reserve(segment.size());
         for (const auto& symmetry : symmetry_order) {
            new_segment.emplace_back(symmetry, get_dimension_from_symmetry(symmetry));
         }
         return edge_segment_t<symmetry_t>(std::move(new_segment));
      }

      std::vector<Symmetry> get_symmetry_order() const {
         std::vector<Symmetry> result;
         result.reserve(segment.size());
         for (const auto& [sym, dim] : segment) {
            result.push_back(sym);
         }
         return result;
      }

      void conjugate_edge() {
         static_assert(is_not_pointer);
         for (auto& [sym, dim] : segment) {
            sym = -sym;
         }
      }

      edge_segment_t<symmetry_t> conjugated_edge() const {
         segment_t result;
         result.reserve(segment.size());
         for (const auto& [sym, dim] : segment) {
            result.emplace_back(-sym, dim);
         }
         return edge_segment_t<symmetry_t>(std::move(result));
      }

      Size total_dimension() const {
         Size total_dim = 0;
         for (const auto& [sym, dim] : segment) {
            total_dim += dim;
         }
         return total_dim;
      }
   };

   struct edge_bose_arrow_t {
      static constexpr Arrow arrow = false;
      edge_bose_arrow_t() {}
      edge_bose_arrow_t(Arrow) {}
   };

   // there are background EPR pair for each edge, for fermi edge, it is needed to record the order of this EPR pair, which is so called fermi arrow
   struct edge_fermi_arrow_t {
      Arrow arrow;
      edge_fermi_arrow_t() : arrow(false) {}
      edge_fermi_arrow_t(Arrow arrow) : arrow(arrow) {}
   };
   template<typename Symmetry>
   using edge_arrow_t = std::conditional_t<Symmetry::is_fermi_symmetry, edge_fermi_arrow_t, edge_bose_arrow_t>;

   /**
    * The shape of tensor edge, is a list of pair of symmetry and size, which construct a structure like line segment.
    * If it is fermi edge, an arrow is also included
    *
    * \tparam Symmetry The symmetry of the tensor
    * \tparam is_pointer whether it is just point to the data or the edge containing the real structure.
    */
   template<typename Symmetry, bool is_pointer = false>
   struct Edge : edge_segment_t<Symmetry, is_pointer>, edge_arrow_t<Symmetry> {
      static_assert(is_symmetry<Symmetry>);

      using base_arrow_t = edge_arrow_t<Symmetry>;
      using base_segment_t = edge_segment_t<Symmetry, is_pointer>;

      using base_arrow_t::arrow;
      using base_segment_t::segment;

      Edge() = default;
      Edge(const Edge&) = default;
      Edge(Edge&&) noexcept = default;
      Edge& operator=(const Edge&) = default;
      Edge& operator=(Edge&&) noexcept = default;
      ~Edge() = default;

      template<typename Arg, typename = std::enable_if_t<!std::is_same_v<remove_cvref_t<Arg>, Edge<Symmetry, is_pointer>>>>
      Edge(Arg&& arg, Arrow arrow = false) : base_segment_t(std::forward<Arg>(arg)), base_arrow_t(arrow) {}
      Edge(std::initializer_list<std::pair<Symmetry, Size>> segment, Arrow arrow = false) :
            base_segment_t(typename base_segment_t::segment_t(segment)),
            base_arrow_t(arrow) {}
      Edge(std::initializer_list<Symmetry> symmetries, Arrow arrow = false) :
            base_segment_t(typename base_segment_t::symlist_t(symmetries)),
            base_arrow_t(arrow) {}

      void conjugate_edge() {
         if constexpr (Symmetry::is_fermi_symmetry) {
            arrow = !arrow;
         }
         base_segment_t::conjugate_edge();
      }
      Edge<Symmetry> conjugated_edge() const {
         return Edge<Symmetry>(std::move(base_segment_t::conjugated_edge()), !arrow);
      }
      Edge<Symmetry> reordered_symmetry() const {
         return Edge<Symmetry>(std::move(base_segment_t::reordered_symmetry()), arrow);
      }
   };

   template<typename Symmetry, bool is_pointer>
   bool operator==(const Edge<Symmetry, is_pointer>& edge_1, const Edge<Symmetry, is_pointer>& edge_2) {
      return edge_1.arrow == edge_2.arrow && std::equal(edge_1.segment.begin(), edge_1.segment.end(), edge_2.segment.begin(), edge_2.segment.end());
   }
   template<typename Symmetry, bool is_pointer>
   bool operator!=(const Edge<Symmetry, is_pointer>& edge_1, const Edge<Symmetry, is_pointer>& edge_2) {
      return edge_1.arrow != edge_2.arrow || !std::equal(edge_1.segment.begin(), edge_1.segment.end(), edge_2.segment.begin(), edge_2.segment.end());
   }

   /**
    * An edge but only containing a pointer to other edge's segment data
    * \see Edge
    */
   template<typename Symmetry>
   using EdgePointer = Edge<Symmetry, true>;

   namespace detail {
      template<typename T>
      struct is_edge_helper : std::false_type {};

      template<typename T>
      struct is_edge_helper<Edge<T, false>> : std::true_type {};

      template<typename T>
      struct is_edge_pointer_helper : std::false_type {};

      template<typename T>
      struct is_edge_pointer_helper<Edge<T, true>> : std::true_type {};
   } // namespace detail

   template<typename T>
   constexpr bool is_edge = detail::is_edge_helper<T>::value;

   template<typename T>
   constexpr bool is_edge_pointer = detail::is_edge_pointer_helper<T>::value;

   template<typename T>
   constexpr bool is_general_edge = is_edge<T> || is_edge_pointer<T>;

   namespace detail {
      template<typename Edge>
      using LoopEdgeIter = typename Edge::segment_t::const_iterator;
      template<template<typename> class Allocator, typename Edge>
      using LoopEdgeIterList = std::vector<LoopEdgeIter<Edge>, Allocator<LoopEdgeIter<Edge>>>;
   } // namespace detail
   /**
    * Loop over each block generated by list of edge
    *
    * \tparam Allocator allocator of iterator vector
    * \param edges edges list
    * \param rank0 if edges list is empty, call rank0
    * \param dims0 if edges list contains emtpy edge, call dims0
    * \param operate call operator for each combination of different symmetries in edges list
    * \note operate has two arguments, first is vector of iterators from each edge segment,
    * another is a record for point needed to be updated because it is changed by loop_edge
    * \see initialize_block_symmetries_with_check, get_merged_edge
    */
   template<
         template<typename> class Allocator = std::allocator,
         typename Edge,
         typename Rank0,
         typename Dims0,
         typename Op,
         typename = std::enable_if_t<
               is_general_edge<Edge> && std::is_invocable_v<Rank0> && std::is_invocable_v<Dims0> &&
               std::is_invocable_r_v<Rank, Op, detail::LoopEdgeIterList<Allocator, Edge>, Rank>>>
   void loop_edge(const Edge* edges, const Rank rank, Rank0&& rank0, Dims0&& dims0, Op&& operate) {
      if (rank == 0) {
         rank0();
         return;
      }
      auto symmetry_iterator_list = detail::LoopEdgeIterList<Allocator, Edge>();
      symmetry_iterator_list.reserve(rank);
      for (auto i = 0; i != rank; ++i) {
         const auto& segment = edges[i].segment;
         if (segment.empty()) {
            dims0();
            return;
         }
         symmetry_iterator_list.push_back(segment.begin());
      }
      Rank minimum_changed = 0; // included, need update
      // minimum tell operate, what is changed since last call
      while (true) {
         minimum_changed = operate(symmetry_iterator_list, minimum_changed);
         auto edge_position = rank - 1;

         while (++symmetry_iterator_list[edge_position] == edges[edge_position].segment.end()) {
            if (edge_position == 0) {
               return;
            }
            symmetry_iterator_list[edge_position] = edges[edge_position].segment.begin();
            --edge_position;
         }
         minimum_changed = minimum_changed < edge_position ? minimum_changed : edge_position;
      }
   }

   template<template<typename> class Allocator = std::allocator, typename Edge, typename = std::enable_if_t<is_general_edge<Edge>>>
   [[nodiscard]] auto initialize_block_symmetries_with_check(const Edge* edges, const Rank rank) {
      // symmetries list and its size
      using Symmetry = typename Edge::symmetry_t;
      using ResultItem = std::pair<std::vector<Symmetry, Allocator<Symmetry>>, Size>;
      auto result = std::vector<ResultItem, Allocator<ResultItem>>(); // following the normal order of blocks
      auto symmetries = std::vector<Symmetry, Allocator<Symmetry>>(rank);
      auto sizes = std::vector<Size, Allocator<Size>>(rank);
      loop_edge<Allocator>(
            edges,
            rank,
            [&] {
               result.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{1});
            },
            [] {},
            [&](const auto& symmetry_iterator_list, Rank minimum_changed) {
               auto symmetry_summary = Symmetry();
               for (const auto& symmetry_iterator : symmetry_iterator_list) {
                  symmetry_summary += symmetry_iterator->first;
               }
               if (symmetry_summary == Symmetry()) {
                  // Symmetry iterator list is changed from minimum_changed since last call of this function
                  for (auto i = minimum_changed; i < rank; i++) {
                     symmetries[i] = symmetry_iterator_list[i]->first;
                     sizes[i] = symmetry_iterator_list[i]->second * (i ? sizes[i - 1] : 1);
                  }
                  result.emplace_back(std::piecewise_construct, std::tuple{symmetries.begin(), symmetries.end()}, std::tuple{sizes.back()});
                  return rank;
               }
               return minimum_changed;
            });
      return result;
   }
} // namespace TAT
#endif
