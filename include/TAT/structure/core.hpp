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

#include "../utility/allocator.hpp"
#include "edge.hpp"

namespace TAT {
   template<typename Symmetry>
   struct core_edges_t {
      static_assert(is_symmetry<Symmetry>);

      using symmetry_t = Symmetry;
      using edge_t = Edge<symmetry_t>;
      using edge_vector_t = std::vector<edge_t>;

      /**
       * The shape of tensor, is edge list, which length is tensor rank, each edge is a list of pair of symmetry and size
       * \see Edge
       */
      edge_vector_t edges = {};

      core_edges_t(std::vector<Edge<Symmetry>> initial_edge) : edges(std::move(initial_edge)) {}

      core_edges_t() = default;
      core_edges_t(const core_edges_t& other) = default;
      core_edges_t(core_edges_t&& other) = default;
      core_edges_t& operator=(const core_edges_t&) = default;
      core_edges_t& operator=(core_edges_t&&) = default;
      ~core_edges_t() = default;
   };

   template<typename ScalarType, typename Symmetry>
   struct core_blocks_t {
      static_assert(is_scalar<ScalarType> && is_symmetry<Symmetry>);

      using symmetry_vector_t = std::vector<Symmetry>;
      using content_vector_t = no_initialize::pmr::vector<ScalarType>;

      using block_map_t = std::vector<std::pair<symmetry_vector_t, content_vector_t>>;

      no_initialize::vector<ScalarType> storage;
      detail::pmr::monotonic_buffer_resource resource;

      /**
       * tensor data itself, is a map from symmetries list to data, every term is a block of tensor
       */
      block_map_t blocks;

      core_blocks_t(std::vector<std::pair<std::vector<Symmetry>, Size>>&& symmetries_list) :
            storage(std::accumulate(
                  symmetries_list.begin(),
                  symmetries_list.end(),
                  0,
                  [&](const Size total_size, const auto& p) {
                     return total_size + p.second;
                  })),
            resource(storage.data(), storage.size() * sizeof(ScalarType)),
            blocks() {
         std::sort(symmetries_list.begin(), symmetries_list.end(), [&](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
         });
         // symmetries_list : vector<Symmetry> -> Size
         for (auto&& [symmetries, size] : symmetries_list) {
            // symmetries list is rvalue, it can be moved
            blocks.push_back({std::move(symmetries), content_vector_t(size, &resource)});
         }
      }

      core_blocks_t(const core_blocks_t& other) : storage(other.storage), resource(storage.data(), storage.size() * sizeof(ScalarType)), blocks() {
         for (const auto& [symmetries, block] : other.blocks) {
            blocks.push_back({symmetries, content_vector_t(block.size(), &resource)});
         }
      }
      core_blocks_t(core_blocks_t&& other) :
            storage(std::move(other.storage)),
            resource(storage.data(), storage.size() * sizeof(ScalarType)),
            blocks() {
         for (auto&& [symmetries, block] : other.blocks) {
            blocks.push_back({std::move(symmetries), content_vector_t(block.size(), &resource)});
         }
      }

      core_blocks_t() = delete;
      core_blocks_t& operator=(const core_blocks_t&) = delete;
      core_blocks_t& operator=(core_blocks_t&&) = delete;
   };

   /**
    * Contains nearly all tensor data except edge name, include edge shape and tensor content
    *
    * \tparam ScalarType scalar type of tensor content
    * \tparam Symmetry the symmetry owned by tensor
    * \note Core used to erase data copy when rename edge name of tensor, which is a common operation
    */
   template<typename ScalarType, typename Symmetry>
   struct Core : core_edges_t<Symmetry>, core_blocks_t<ScalarType, Symmetry> {
      static_assert(is_scalar<ScalarType> && is_symmetry<Symmetry>);

      using base_edges = core_edges_t<Symmetry>;
      using base_blocks = core_blocks_t<ScalarType, Symmetry>;

      using base_blocks::blocks;
      using base_blocks::storage;
      using base_edges::edges;

      // this is the only constructor, from constructor of tensor
      Core(std::vector<Edge<Symmetry>> initial_edge) :
            base_edges(std::move(initial_edge)),
            base_blocks(initialize_block_symmetries_with_check(edges.data(), edges.size())) {
         // delete symmetry not used in block from edge data
         if constexpr (Symmetry::length != 0) {
            const Rank rank = edges.size();
            auto edge_mark = std::vector<std::vector<bool>>(rank);
            for (Rank i = 0; i < rank; i++) {
               edge_mark[i] = std::vector<bool>(edges[i].segment.size(), false);
            }
            for (const auto& [symmetries, _] : blocks) {
               for (Rank i = 0; i < rank; i++) {
                  auto symmetry_position = edges[i].get_position_from_symmetry(symmetries[i]);
                  edge_mark[i][symmetry_position] = true;
               }
            }
            for (Rank i = 0; i < rank; i++) {
               auto& edge = edges[i];
               const auto& this_mark = edge_mark[i];
               const Nums number = edge.segment.size();
               Nums k = 0;
               for (Nums j = 0; j < number; j++) {
                  if (this_mark[j]) {
                     edge.segment[k++] = std::move(edge.segment[j]);
                  }
               }
               edge.segment.resize(k);
            }
         }
      }

      Core() = delete;
      Core(const Core& other) = default;
      Core(Core&& other) = default;
      Core& operator=(const Core&) = delete;
      Core& operator=(Core&&) = delete;
   };
} // namespace TAT
#endif
