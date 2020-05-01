/**
 * \file get_item.hpp
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
#ifndef TAT_GET_ITEM_HPP
#define TAT_GET_ITEM_HPP

#include "tensor.hpp"

namespace TAT {
   /**
    * \brief 寻找有对称性张量中的某个子块
    */
   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto get_block_for_get_item(
         const ::std::map<Name, Symmetry>& position,
         const ::std::map<Name, Rank>& name_to_index,
         const Core<ScalarType, Symmetry>& core) {
      auto symmetries = ::std::vector<Symmetry>(core.edges.size());
      for (const auto& [name, symmetry] : position) {
         symmetries[name_to_index.at(name)] = symmetry;
      }
      return symmetries;
   }

   /**
    * \brief 寻找无对称性张量中每个元素
    */
   template<class ScalarType, class Symmetry>
   [[nodiscard]] auto get_offset_for_get_item(
         const ::std::map<Name, Size>& position,
         const ::std::map<Name, Rank>& name_to_index,
         const Core<ScalarType, Symmetry>& core) {
      const auto rank = Rank(core.edges.size());
      auto scalar_position = ::std::vector<Size>(rank);
      auto dimensions = ::std::vector<Size>(rank);
      for (const auto& [name, position] : position) {
         auto index = name_to_index.at(name);
         scalar_position[index] = position;
         dimensions[index] = core.edges[index].map.begin()->second;
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
         const ::std::map<Name, ::std::tuple<Symmetry, Size>>& position,
         const ::std::map<Name, Rank>& name_to_index,
         const Core<ScalarType, Symmetry>& core) {
      const auto rank = Rank(core.edges.size());
      auto symmetries = ::std::vector<Symmetry>(rank);
      auto scalar_position = ::std::vector<Size>(rank);
      auto dimensions = ::std::vector<Size>(rank);
      for (const auto& [name, _] : position) {
         const auto& [symmetry, position] = _;
         auto index = name_to_index.at(name);
         symmetries[index] = symmetry;
         scalar_position[index] = position;
         dimensions[index] = core.edges[index].map.at(symmetry);
      }
      Size offset = 0;
      for (Rank j = 0; j < rank; j++) {
         offset *= dimensions[j];
         offset += scalar_position[j];
      }
      return ::std::make_tuple(symmetries, offset);
   }

   template<class ScalarType, class Symmetry>
   const auto& Tensor<ScalarType, Symmetry>::block(const ::std::map<Name, Symmetry>& position) const& {
      // using has_symmetry = ::std::enable_if_t<!::std::is_same_v<Symmetry, NoSymmetry>>;
      auto symmetry = get_block_for_get_item(position, name_to_index, *core);
      return core->blocks.at(symmetry);
   }

   template<class ScalarType, class Symmetry>
   auto& Tensor<ScalarType, Symmetry>::block(const ::std::map<Name, Symmetry>& position) & {
      // using has_symmetry = ::std::enable_if_t<!::std::is_same_v<Symmetry, NoSymmetry>>;
      auto symmetry = get_block_for_get_item(position, name_to_index, *core);
      return core->blocks.at(symmetry);
   }

   template<class ScalarType, class Symmetry>
   ScalarType Tensor<ScalarType, Symmetry>::at(const ::std::map<Name, EdgeInfoForGetItem>& position) const& {
      if constexpr (::std::is_same_v<Symmetry, NoSymmetry>) {
         auto offset = get_offset_for_get_item(position, name_to_index, *core);
         return core->blocks.begin()->second[offset];
      } else {
         auto [symmetry, offset] = get_block_and_offset_for_get_item(position, name_to_index, *core);
         return core->blocks.at(symmetry)[offset];
      }
   }

   template<class ScalarType, class Symmetry>
   ScalarType& Tensor<ScalarType, Symmetry>::at(const ::std::map<Name, EdgeInfoForGetItem>& position) & {
      if constexpr (::std::is_same_v<Symmetry, NoSymmetry>) {
         auto offset = get_offset_for_get_item(position, name_to_index, *core);
         return core->blocks.begin()->second[offset];
      } else {
         auto [symmetry, offset] = get_block_and_offset_for_get_item(position, name_to_index, *core);
         return core->blocks.at(symmetry)[offset];
      }
   }
} // namespace TAT
#endif
