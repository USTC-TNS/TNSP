/**
 * \file get_item_and_clear_symmetry.hpp
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
#ifndef TAT_GET_ITEM_AND_CLEAR_SYMMETRY_HPP
#define TAT_GET_ITEM_AND_CLEAR_SYMMETRY_HPP

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "transpose.hpp"

namespace TAT {
   template<typename ScalarType, typename Symmetry, typename Name>
   template<typename PositionType>
   const ScalarType& Tensor<ScalarType, Symmetry, Name>::get_item(const PositionType& position) const& {
      constexpr bool is_vector_not_map =
            std::is_same_v<PositionType, std::vector<Size>> || std::is_same_v<PositionType, std::vector<std::pair<Symmetry, Size>>>;
      constexpr bool is_index_not_point = std::is_same_v<PositionType, std::vector<Size>> || std::is_same_v<PositionType, std::map<Name, Size>>;
      auto pmr_guard = scope_resource(default_buffer_size);
      auto rank = get_rank();
      auto symmetries = pmr::vector<Symmetry>();
      auto scalar_position = pmr::vector<Size>();
      auto dimensions = pmr::vector<Size>();
      symmetries.reserve(rank);
      scalar_position.reserve(rank);
      dimensions.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         const auto& point_or_index = [&]() {
            if constexpr (is_vector_not_map) {
               return position[i];
            } else {
               auto found = position.find(names[i]);
               if constexpr (debug_mode) {
                  if (found == position.end()) {
                     detail::error("Name not found in position map when finding block and offset");
                  }
               }
               return found->second;
            }
         }();
         const auto& [symmetry, index] = [&]() {
            if constexpr (is_index_not_point) {
               return edges(i).get_point_from_index(point_or_index);
            } else {
               return point_or_index;
            }
         }();
         symmetries.push_back(symmetry);
         scalar_position.push_back(index);
         dimensions.push_back(edges(i).get_dimension_from_symmetry(symmetry));
      }
      Size offset = 0;
      for (Rank j = 0; j < rank; j++) {
         offset *= dimensions[j];
         offset += scalar_position[j];
      }
      return blocks(symmetries)[offset];
   }

   inline auto get_leading(const pmr::vector<Size>& dim) {
      Rank rank = dim.size();
      pmr::vector<Size> res(rank, 0);
      for (auto i = rank; i-- > 0;) {
         if (i == rank - 1) {
            res[i] = 1;
         } else {
            res[i] = res[i + 1] * dim[i + 1];
         }
      }
      return res;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, NoSymmetry, Name> Tensor<ScalarType, Symmetry, Name>::clear_symmetry() const {
      auto pmr_guard = scope_resource(default_buffer_size);
      if constexpr (Symmetry::is_fermi_symmetry) {
         detail::warning("Clearing a fermi tensor's symmetry");
      }
      Rank rank = get_rank();
      std::vector<Edge<NoSymmetry>> result_edge;
      pmr::vector<Size> result_dim;
      pmr::vector<Rank> plan;
      result_edge.reserve(rank);
      result_dim.reserve(rank);
      plan.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         Size dim = edges(i).total_dimension();
         result_edge.push_back(dim);
         result_dim.push_back(dim);
         plan.push_back(i);
      }
      auto result = Tensor<ScalarType, NoSymmetry, Name>(names, std::move(result_edge)).zero();
      for (const auto& [symmetry_list, block] : core->blocks) {
         pmr::vector<Size> block_dimension;
         std::map<Name, Size> block_index;
         block_dimension.reserve(rank);
         for (auto i = 0; i < rank; i++) {
            // TODO opt it, 但是由于这个函数一般只用于测试正确性，所以优先级不高
            // 两个可以一起做，而且下面的block index的使用方式不需要Name
            block_dimension.push_back(edges(i).get_dimension_from_symmetry(symmetry_list[i]));
            block_index[names[i]] = edges(i).get_index_from_point({symmetry_list[i], 0});
         }
         // source dimension is block dimension
         // source leading is block dimension
         const ScalarType* data_source = block.data();
         // destination dimension is block dimension
         // destination leading is result edge
         ScalarType* data_destination = &result.at(block_index);
         simple_transpose<ScalarType, false>(
               data_source,
               data_destination,
               plan,
               plan,
               block_dimension,
               block_dimension,
               get_leading(block_dimension),
               get_leading(result_dim),
               rank);
      }
      return result;
   }
} // namespace TAT
#endif
