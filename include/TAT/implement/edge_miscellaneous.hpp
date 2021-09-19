/**
 * \file edge_miscellaneous.hpp
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
#ifndef TAT_EDGE_MISCELLANEOUS_HPP
#define TAT_EDGE_MISCELLANEOUS_HPP

#include "../structure/tensor.hpp"

namespace TAT {
   template<typename ScalarType, typename Symmetry, typename Name>
   template<typename ResultName, typename>
   auto Tensor<ScalarType, Symmetry, Name>::edge_rename(const std::map<Name, ResultName>& dictionary) const {
      if constexpr (debug_mode) {
         for (const auto& [name, new_name] : dictionary) {
            auto found = std::find(names.begin(), names.end(), name);
            if (found == names.end()) {
               detail::warning("Name missing in edge_rename");
            }
         }
      }
      auto result = Tensor<ScalarType, Symmetry, ResultName>{};
      result.core = core;
      result.names.reserve(get_rank());
      std::transform(names.begin(), names.end(), std::back_inserter(result.names), [&dictionary](const Name& name) {
         if (auto position = dictionary.find(name); position == dictionary.end()) {
            if constexpr (std::is_same_v<ResultName, Name>) {
               return name;
            } else {
               detail::error("New names not found in edge_rename which change type of name");
            }
         } else {
            return position->second;
         }
      });
      return result;
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::merge_edge(
         const std::map<Name, std::vector<Name>>& merge,
         bool apply_parity,
         const std::set<Name>&& parity_exclude_name_merge,
         const std::set<Name>& parity_exclude_name_reverse) const {
      auto pmr_guard = scope_resource(default_buffer_size);
      if constexpr (debug_mode) {
         // check if the edge not exist in merge map
         for (const auto& [new_edge, old_edges] : merge) {
            for (const auto& old_edge : old_edges) {
               auto found = find_rank_from_name(old_edge);
               if (found == names.end()) {
                  detail ::error("No such edge in merge map");
               }
            }
         }
      }
      std::vector<Name> target_name;
      target_name.reserve(get_rank());
      for (auto iterator = names.rbegin(); iterator != names.rend(); ++iterator) {
         // 找到且最后 -> 添加新的
         // 找到不最后 -> 不做事
         // 没找到 -> 添加
         auto found_in_merge = false;
         for (const auto& [name_after_merge, names_before_merge] : merge) {
            if (auto position_in_group = std::find(names_before_merge.begin(), names_before_merge.end(), *iterator);
                position_in_group != names_before_merge.end()) {
               if (*iterator == names_before_merge.back()) {
                  target_name.push_back(name_after_merge);
               }
               found_in_merge = true;
               break;
            }
         }
         if (!found_in_merge) {
            target_name.push_back(*iterator);
         }
      }
      // 插入空merge的edge
      for (const auto& [name_after_merge, names_before_merge] : merge) {
         if (names_before_merge.empty()) {
            target_name.push_back(name_after_merge);
         }
      }
      // 翻转target_name
      std::reverse(target_name.begin(), target_name.end());
      return edge_operator_implement(
            empty_list<std::pair<Name, empty_list<std::pair<Name, edge_segment_t<Symmetry>>>>>(),
            empty_list<Name>(),
            merge,
            std::move(target_name),
            apply_parity,
            empty_list<Name>(),
            empty_list<Name>(),
            std::forward<decltype(parity_exclude_name_reverse)>(parity_exclude_name_reverse),
            std::forward<decltype(parity_exclude_name_merge)>(parity_exclude_name_merge),
            empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());
   }

   template<typename ScalarType, typename Symmetry, typename Name>
   Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::split_edge(
         const std::map<Name, std::vector<std::pair<Name, edge_segment_t<Symmetry>>>>& split,
         bool apply_parity,
         const std::set<Name>& parity_exclude_name_split) const {
      auto pmr_guard = scope_resource(default_buffer_size);
      if constexpr (debug_mode) {
         for (const auto& [old_edge, new_edges] : split) {
            auto found = find_rank_from_name(old_edge);
            if (found == names.end()) {
               detail ::error("No such edge in split map");
            }
         }
      }
      // 生成target_name
      std::vector<Name> target_name;
      target_name.reserve(get_rank()); // 不够, 但是可以减少new的次数
      for (const auto& n : names) {
         if (auto found = split.find(n); found != split.end()) {
            for (const auto& edge_after_split : found->second) {
               target_name.push_back(std::get<0>(edge_after_split));
            }
         } else {
            target_name.push_back(n);
         }
      }
      return edge_operator_implement(
            split,
            empty_list<Name>(),
            empty_list<std::pair<Name, empty_list<Name>>>(),
            std::move(target_name),
            apply_parity,
            std::forward<decltype(parity_exclude_name_split)>(parity_exclude_name_split),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<Name>(),
            empty_list<std::pair<Name, empty_list<std::pair<Symmetry, Size>>>>());
   }
} // namespace TAT
#endif
