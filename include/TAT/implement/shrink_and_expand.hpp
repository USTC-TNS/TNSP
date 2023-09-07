/**
 * \file shrink_and_expand.hpp
 *
 * Copyright (C) 2020-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_SHRINK_AND_EXPAND_HPP
#define TAT_SHRINK_AND_EXPAND_HPP

#include "../structure/tensor.hpp"
#include "../utility/timer.hpp"

namespace TAT {
    inline timer expand_guard("expand");

    template<typename ScalarType, typename Symmetry, typename Name>
    Tensor<ScalarType, Symmetry, Name> Tensor<ScalarType, Symmetry, Name>::expand(
        const std::unordered_map<Name, std::pair<Size, Edge<Symmetry>>>& configure,
        const Name& old_name
    ) const {
        auto pmr_guard = scope_resource(default_buffer_size);
        auto timer_guard = expand_guard();
        constexpr bool is_no_symmetry = Symmetry::length == 0;
        constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
        if constexpr (is_fermi) {
            detail::warning("expand edge of fermi tensor is dangerous, please contract helper tensor manually");
        }

        // generator helper tensor names and edges
        auto new_names = std::vector<Name>();
        auto new_edges = std::vector<Edge<Symmetry>>();
        auto reserve_size = configure.size() + 1;
        new_names.reserve(reserve_size);
        new_edges.reserve(reserve_size);

        auto points = pmr::unordered_map<Name, std::pair<Symmetry, Size>>(unordered_parameter * configure.size());
        auto total_symmetry = Symmetry();
        auto contract_pairs = std::unordered_set<std::pair<Name, Name>>(unordered_parameter * configure.size());

        for (const auto& [name, information] : configure) {
            const auto& [index, edge] = information;
            const auto& point = edge.point_by_index(index);
            auto symmetry = point.first;

            new_names.push_back(name);
            new_edges.push_back(edge);
            points[name] = point;
            total_symmetry += symmetry;
        }
        if (old_name != InternalName<Name>::No_Old_Name) {
            const auto& old_edge = edges(old_name);
            if constexpr (debug_mode) {
                if (old_edge.total_dimension() != 1) {
                    detail::error("Cannot Expand a Edge which dimension is not one");
                }
            }

            new_names.push_back(InternalName<Name>::No_Old_Name);
            new_edges.push_back({{{-total_symmetry, 1}}, !old_edge.arrow()});
            points[InternalName<Name>::No_Old_Name] = std::pair<Symmetry, Size>{-total_symmetry, 0};
            contract_pairs.insert({old_name, InternalName<Name>::No_Old_Name});
        } else {
            if constexpr (debug_mode) {
                if constexpr (!is_no_symmetry) {
                    if (total_symmetry != Symmetry()) {
                        detail::error("Cannot Expand to such Edges whose total Symmetry is not zero");
                    }
                }
            }
        }
        auto helper = Tensor<ScalarType, Symmetry, Name>(std::move(new_names), std::move(new_edges)).zero();
        helper.at(points) = 1;
        return contract(helper, contract_pairs);
    }

    inline timer shrink_guard("shrink");

    template<typename ScalarType, typename Symmetry, typename Name>
    Tensor<ScalarType, Symmetry, Name>
    Tensor<ScalarType, Symmetry, Name>::shrink(const std::unordered_map<Name, Size>& configure, const Name& new_name, Arrow arrow) const {
        auto pmr_guard = scope_resource(default_buffer_size);
        auto timer_guard = shrink_guard();
        constexpr bool is_no_symmetry = Symmetry::length == 0;
        constexpr bool is_fermi = Symmetry::is_fermi_symmetry;
        if constexpr (is_fermi) {
            detail::warning("shrink edge of fermi tensor is dangerous, please contract helper tensor manually");
        }

        auto new_names = std::vector<Name>();
        auto new_edges = std::vector<Edge<Symmetry>>();
        auto reserve_size = configure.size() + 1;
        new_names.reserve(reserve_size);
        new_edges.reserve(reserve_size);

        auto points = pmr::unordered_map<Name, std::pair<Symmetry, Size>>(unordered_parameter * configure.size());
        auto total_symmetry = Symmetry();
        auto contract_pairs = std::unordered_set<std::pair<Name, Name>>(unordered_parameter * configure.size());

        for (auto i = 0; i < rank(); i++) {
            const auto& name = names(i);
            if (auto found = configure.find(name); found != configure.end()) {
                // shrinking
                const auto& edge = edges(i);
                const auto& index = found->second;
                auto point = edge.point_by_index(index);
                auto symmetry = point.first;

                new_names.push_back(name);
                new_edges.push_back(edge.conjugated());
                points[name] = point;
                total_symmetry += symmetry;
                contract_pairs.insert({name, name});
            }
        }
        if (new_name != InternalName<Name>::No_New_Name) {
            new_names.push_back(new_name);
            new_edges.push_back({{{total_symmetry, 1}}, arrow});
            points[new_name] = std::pair<Symmetry, Size>{total_symmetry, 0};
        } else {
            if constexpr (debug_mode) {
                if constexpr (!is_no_symmetry) {
                    if (total_symmetry != Symmetry()) {
                        detail::error("Need to Create a New Edge but Name not set in Slice");
                    }
                }
            }
        }
        auto helper = Tensor<ScalarType, Symmetry, Name>(std::move(new_names), std::move(new_edges)).zero();
        helper.at(points) = 1;
        return contract(helper, contract_pairs);
    }
} // namespace TAT
#endif
