/**
 * \file split_and_merge.hpp
 *
 * Copyright (C) 2019-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_SPLIT_AND_MERGE_HPP
#define TAT_SPLIT_AND_MERGE_HPP

#include "../structure/tensor.hpp"

namespace TAT {
    template<typename ScalarType, typename Symmetry, typename Name>
    Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::split_edge(
        const std::unordered_map<Name, std::vector<std::pair<Name, edge_segments_t<Symmetry>>>>& split,
        bool apply_parity,
        const std::unordered_set<Name>& parity_exclude_names_split
    ) const {
        auto pmr_guard = scope_resource(default_buffer_size);
        if constexpr (debug_mode) {
            for (const auto& [old_name, new_names_edges] : split) {
                if (auto found = find_by_name(old_name); found == names().end()) {
                    detail ::error("No such edge in split map");
                }
            }
        }
        // generate target_name
        std::vector<Name> target_name;
        target_name.reserve(rank()); // not enough but it is ok to reduce realloc time
        for (const auto& name : names()) {
            if (auto found = split.find(name); found != split.end()) {
                for (const auto& [new_name, segments] : found->second) {
                    target_name.push_back(new_name);
                }
            } else {
                target_name.push_back(name);
            }
        }
        return edge_operator_implement(split, {}, {}, std::move(target_name), apply_parity, parity_exclude_names_split, {}, {}, {}, {});
    }

    template<typename ScalarType, typename Symmetry, typename Name>
    Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::merge_edge(
        const std::unordered_map<Name, std::vector<Name>>& merge,
        bool apply_parity,
        const std::unordered_set<Name>& parity_exclude_names_merge,
        const std::unordered_set<Name>& parity_exclude_names_reverse
    ) const {
        auto pmr_guard = scope_resource(default_buffer_size);
        if constexpr (debug_mode) {
            // check if the edge not exist in merge map
            for (const auto& [new_name, old_names] : merge) {
                for (const auto& old_name : old_names) {
                    if (auto found = find_by_name(old_name); found == names().end()) {
                        detail ::error("No such edge in merge map");
                    }
                }
            }
        }
        std::vector<Name> target_name;
        target_name.reserve(rank());
        for (auto index = rank(); index-- > 0;) {
            const auto& name = names(index);
            // find and it is last -> add new merge list
            // find but not last -> do nothing
            // not found -> add this single edge
            auto found_in_merge = false;
            for (const auto& [name_after_merge, names_before_merge] : merge) {
                if (auto position_in_group = std::find(names_before_merge.begin(), names_before_merge.end(), name);
                    position_in_group != names_before_merge.end()) {
                    if (name == names_before_merge.back()) {
                        target_name.push_back(name_after_merge);
                    }
                    found_in_merge = true;
                    break;
                }
            }
            if (!found_in_merge) {
                target_name.push_back(name);
            }
        }
        // and empty merge edge
        for (const auto& [name_after_merge, names_before_merge] : merge) {
            if (names_before_merge.empty()) {
                target_name.push_back(name_after_merge);
            }
        }
        // reverse target name
        std::reverse(target_name.begin(), target_name.end());
        return edge_operator_implement(
            {},
            {},
            merge,
            std::move(target_name),
            apply_parity,
            {},
            {},
            parity_exclude_names_reverse,
            parity_exclude_names_merge,
            {}
        );
    }
} // namespace TAT
#endif
