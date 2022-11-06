/**
 * \file core.hpp
 *
 * Copyright (C) 2019-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include "../utility/multidimension_span.hpp"
#include "edge.hpp"

namespace TAT {

   /**
    * Contains nearly all tensor data except edge name, include edge shape and tensor content
    *
    * \tparam ScalarType scalar type of tensor content
    * \tparam Symmetry the symmetry owned by tensor
    * \note Core used to erase data copy when rename edge name of tensor, which is a common operation
    */
   template<typename ScalarType, typename Symmetry>
   class Core {
      static_assert(is_scalar<ScalarType> && is_symmetry<Symmetry>);

    public:
      using scalar_t = ScalarType;
      using symmetry_t = Symmetry;
      using edge_t = Edge<symmetry_t>;

    private:
      const std::vector<edge_t> m_edges;
      mdspan<std::optional<mdspan<scalar_t>>> m_blocks;

      std::vector<std::optional<mdspan<scalar_t>>> m_pool;
      std::vector<scalar_t> m_storage;

    public:
      const std::vector<scalar_t>& storage() const {
         return m_storage;
      }
      std::vector<scalar_t>& storage() {
         return m_storage;
      }
      const std::vector<edge_t>& edges() const {
         return m_edges;
      }
      const edge_t& edges(Rank i) const {
         return m_edges[i];
      }
      const mdspan<std::optional<mdspan<scalar_t>>>& blocks() const {
         return m_blocks;
      }
      mdspan<std::optional<mdspan<scalar_t>>>& blocks() {
         return m_blocks;
      }
      template<typename T, typename A>
      const mdspan<scalar_t>& blocks(const std::vector<T, A>& arg) const {
         if constexpr (std::is_same_v<T, symmetry_t>) {
            const auto& symmetries = arg;
            std::vector<Size, typename std::allocator_traits<A>::template rebind_alloc<Size>> positions;
            positions.reserve(blocks().rank());
            for (auto i = 0; i < blocks().rank(); i++) {
               positions.push_back(edges(i).position_by_symmetry(symmetries[i]));
            }
            return blocks(positions);
         } else {
            const auto& positions = arg;
            return m_blocks.at(positions).value();
         }
      }
      // non const version for all parameter
      template<typename Arg>
      auto& blocks(const Arg& arg) {
         return const_cast<mdspan<scalar_t>&>(const_cast<const Core<scalar_t, symmetry_t>*>(this)->blocks(arg));
      }
      template<typename T, typename A>
      const scalar_t& at(const std::vector<T, A>& arg) const {
         std::vector<Size, typename std::allocator_traits<A>::template rebind_alloc<Size>> positions, offsets;
         positions.reserve(blocks().rank());
         offsets.reserve(blocks().rank());
         for (auto i = 0; i < blocks().rank(); i++) {
            auto [position, offset] = overloaded{
                  [](const edge_t& edge, const typename edge_t::coord_t& coord) {
                     return coord;
                  },
                  [](const edge_t& edge, const typename edge_t::index_t& index) {
                     return edge.coord_by_index(index);
                  },
                  [](const edge_t& edge, const typename edge_t::point_t& point) {
                     return edge.coord_by_point(point);
                  }}(edges(i), arg[i]);
            positions.push_back(position);
            offsets.push_back(offset);
         }
         return blocks(positions).at(offsets);
      }
      // non const version for all parameter
      template<typename Arg>
      auto& at(const Arg& arg) {
         return const_cast<scalar_t&>(const_cast<const Core<scalar_t, symmetry_t>*>(this)->at(arg));
      }

    private:
      static std::vector<Size> blocks_dimensions(const std::vector<edge_t>& edges) {
         std::vector<Size> result;
         result.reserve(edges.size());
         for (const auto& edge : edges) {
            result.emplace_back(edge.segments_size());
         }
         return result;
      }

      std::vector<Size> single_block_dimensions(const std::vector<Size>& positions) const {
         std::vector<Size> result;
         result.reserve(blocks().rank());
         for (auto i = 0; i < blocks().rank(); i++) {
            result.push_back(edges(i).segments(positions[i]).second);
         }
         return result;
      }

      void refresh_storage_pointer_in_pool() {
         auto storage_pointer = m_storage.data();

#if 1
         for (const auto& index : _order) {
            auto& block = blocks(index);
            block.set_data(storage_pointer);
            storage_pointer += block.size();
         }
         return;
#endif
         for (auto& block : m_pool) {
            if (block.has_value()) {
               block.value().set_data(storage_pointer);
               storage_pointer += block.value().size();
            }
         }
      }

    public:
// TODO use new order storage
#if 1
      std::vector<std::vector<symmetry_t>> _order;
#endif

      // this is the only constructor, from constructor of tensor
      Core(std::vector<edge_t> input_edges) :
            m_edges(std::move(input_edges)),
            m_blocks(nullptr, blocks_dimensions(m_edges)),
            m_pool(m_blocks.size()) {
         m_blocks.set_data(m_pool.data());

         if (blocks().size() == 0) {
            return;
         }

         auto total_symmetry_pool = std::vector<symmetry_t>(blocks().size());
         auto total_symmetry = mdspan<symmetry_t>(total_symmetry_pool.data(), blocks().dimensions());
         for (auto i = 0; i < blocks().rank(); ++i) {
            Size self_size = edges(i).segments_size();
            Size in_size = blocks().leadings(i);
            Size out_size = blocks().size() / (self_size * in_size);
            for (Size x = 0; x < out_size; x++) {
               auto offset_for_x = x;
               for (Size y = 0; y < self_size; y++) {
                  symmetry_t here = edges(i).segments(y).first;
                  auto offset_for_y = offset_for_x * self_size + y;
                  for (Size z = 0; z < in_size; z++) {
                     auto offset_for_z = offset_for_y * in_size + z;
                     total_symmetry_pool[offset_for_z] += here;
                  }
               }
            }
         }

         Size storage_size = 0;
         for (auto it = blocks().begin(); it.valid; ++it) {
            if (total_symmetry_pool[it.offset] == symmetry_t()) {
               it->emplace(nullptr, single_block_dimensions(it.indices));
#if 1
               std::vector<symmetry_t> result;
               result.reserve(blocks().rank());
               for (auto i = 0; i < blocks().rank(); i++) {
                  result.push_back(edges(i).segments(it.indices[i]).first);
               }
               _order.push_back(result);
#endif
               storage_size += it->value().size();
            }
         }
#if 1
         std::sort(_order.begin(), _order.end());
#endif

         m_storage.resize(storage_size);
         refresh_storage_pointer_in_pool();
      }

      Core() = delete;
      Core(const Core& other) : m_edges(other.m_edges), m_blocks(other.m_blocks), m_pool(other.m_pool), m_storage(other.m_storage) {
         m_blocks.set_data(m_pool.data());
#if 1
         _order = other._order;
#endif
         refresh_storage_pointer_in_pool();
      };
      Core(Core&& other) = default;
      Core& operator=(const Core&) = delete;
      Core& operator=(Core&&) = delete;
   };
} // namespace TAT
#endif
