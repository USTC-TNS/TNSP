/**
 * \file get_item.hpp
 *
 * Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_GET_ITEM_HPP
#define TAT_GET_ITEM_HPP

#include "tensor.hpp"

namespace TAT {
   /**
    * \brief 寻找有对称性张量中的某个子块
    */
   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto
   get_block_for_get_item(const std::map<Name, Symmetry>& position, const std::vector<Name>& names, const Core<ScalarType, Symmetry>& core) {
      auto symmetries = std::vector<Symmetry>();
      symmetries.reserve(names.size());
      for (const auto& name : names) {
         symmetries.push_back(position.at(name));
      }
      return symmetries;
   }

   /**
    * \brief 寻找无对称性张量中每个元素
    */
   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto
   get_offset_for_get_item(const std::map<Name, Size>& position, const std::vector<Name>& names, const Core<ScalarType, Symmetry>& core) {
      const auto rank = Rank(names.size());
      auto scalar_position = std::vector<Size>();
      auto dimensions = std::vector<Size>();
      scalar_position.reserve(rank);
      dimensions.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         scalar_position.push_back(position.at(names[i]));
         dimensions.push_back(core.edges[i].map.begin()->second);
      }
      Size offset = 0;
      for (Rank j = 0; j < rank; j++) {
         offset *= dimensions[j];
         offset += scalar_position[j];
      }
      return offset;
   }

   /**
    * \brief 寻找有对称性张量中的某个子块的某个元素
    */
   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto get_block_and_offset_for_get_item(
         const std::map<Name, std::tuple<Symmetry, Size>>& position,
         const std::vector<Name>& names,
         const Core<ScalarType, Symmetry>& core) {
      const auto rank = Rank(core.edges.size());
      auto symmetries = std::vector<Symmetry>();
      auto scalar_position = std::vector<Size>();
      auto dimensions = std::vector<Size>();
      symmetries.reserve(rank);
      scalar_position.reserve(rank);
      dimensions.reserve(rank);
      for (auto i = 0; i < rank; i++) {
         const auto& name = names[i];
         const auto& [symmetry, index] = position.at(name);
         symmetries.push_back(symmetry);
         scalar_position.push_back(index);
         dimensions.push_back(core.edges[i].map.at(symmetry));
      }
      Size offset = 0;
      for (Rank j = 0; j < rank; j++) {
         offset *= dimensions[j];
         offset += scalar_position[j];
      }
      return std::make_tuple(symmetries, offset);
   }

   template<class ScalarType, class Symmetry>
   const auto& Tensor<ScalarType, Symmetry>::block(const std::map<Name, Symmetry>& position) const& {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         return core->blocks.begin()->second;
      }
      auto symmetry = get_block_for_get_item(position, names, *core);
      return core->blocks.at(symmetry);
   }

   template<class ScalarType, class Symmetry>
   auto& Tensor<ScalarType, Symmetry>::block(const std::map<Name, Symmetry>& position) & {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         return core->blocks.begin()->second;
      }
      auto symmetry = get_block_for_get_item(position, names, *core);
      return core->blocks.at(symmetry);
   }

   template<class ScalarType, class Symmetry>
   ScalarType Tensor<ScalarType, Symmetry>::at(const std::map<Name, EdgeInfoForGetItem>& position) const& {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         auto offset = get_offset_for_get_item(position, names, *core);
         return core->blocks.begin()->second[offset];
      } else {
         auto [symmetry, offset] = get_block_and_offset_for_get_item(position, names, *core);
         return core->blocks.at(symmetry)[offset];
      }
   }

   template<class ScalarType, class Symmetry>
   ScalarType& Tensor<ScalarType, Symmetry>::at(const std::map<Name, EdgeInfoForGetItem>& position) & {
      if constexpr (std::is_same_v<Symmetry, NoSymmetry>) {
         auto offset = get_offset_for_get_item(position, names, *core);
         return core->blocks.begin()->second[offset];
      } else {
         auto [symmetry, offset] = get_block_and_offset_for_get_item(position, names, *core);
         return core->blocks.at(symmetry)[offset];
      }
   }
} // namespace TAT
#endif
