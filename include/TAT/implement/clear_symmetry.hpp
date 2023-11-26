/**
 * \file clear_symmetry.hpp
 *
 * Copyright (C) 2019-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
#ifndef TAT_CLEAR_SYMMETRY_HPP
#define TAT_CLEAR_SYMMETRY_HPP

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/multidimension_span.hpp"

namespace TAT {
    template<typename ScalarType, typename Symmetry, typename Name>
    Tensor<ScalarType, NoSymmetry, Name> Tensor<ScalarType, Symmetry, Name>::clear_bose_symmetry() const {
        auto pmr_guard = scope_resource(default_buffer_size);
        if constexpr (Symmetry::is_fermi_symmetry) {
            detail::warning("Clearing a fermi tensor's symmetry, it is dangerous if you do not take care of edge order");
        }
        std::vector<Edge<NoSymmetry>> result_edges;
        result_edges.reserve(rank());
        for (auto i = 0; i < rank(); i++) {
            result_edges.push_back(edges(i).total_dimension());
        }
        // Generate no symmetry tensor edge total dimension and create it first
        auto result = Tensor<ScalarType, NoSymmetry, Name>(names(), std::move(result_edges)).zero_();
        auto& result_block = result.blocks().begin()->value();
        // copy every block into no symmetry tensor
        // find the dimension of the block, the result leading is same to total dimension
        // and find the offset of destination
        // it is easy to get the offset of source then call transpose
        for (auto it = blocks().begin(); it.valid; ++it) {
            if (it->has_value()) {
                pmr::vector<Size> result_indices;
                pmr::vector<Size> result_dimensions;
                result_indices.reserve(rank());
                result_dimensions.reserve(rank());
                for (auto i = 0; i < rank(); i++) {
                    auto& [source_symmetry, source_dimension] = edges(i).segments(it.indices[i]);
                    result_indices.push_back(edges(i).index_by_point({source_symmetry, 0}));
                    result_dimensions.push_back(source_dimension);
                }
                mdspan<ScalarType, pmr::vector<Size>> temporary_span(
                    &result_block.at(result_indices),
                    std::move(result_dimensions),
                    {result_block.leadings().begin(), result_block.leadings().end()}
                );
                mdspan_transform(it->value(), temporary_span, [](const auto& x) { return x; });
            }
        }
        return result;
    }

    template<typename ScalarType, typename Symmetry, typename Name>
    Tensor<ScalarType, ParitySymmetry, Name> Tensor<ScalarType, Symmetry, Name>::clear_fermi_symmetry() const {
        auto pmr_guard = scope_resource(default_buffer_size);
        if constexpr (!Symmetry::is_fermi_symmetry) {
            detail::error("It is invalid to call clear fermi symmetry on a bose symmetry tensor");
        }
        std::vector<Edge<ParitySymmetry>> result_edges;
        result_edges.reserve(rank());
        for (auto i = 0; i < rank(); i++) {
            std::array<Size, 2> dimensions = {0, 0};
            for (const auto& [symmetry, dimension] : edges(i).segments()) {
                dimensions[symmetry.parity()] += dimension;
            }
            std::vector<std::pair<ParitySymmetry, Size>> segments;
            segments.reserve(2);
            if (const auto dimension = dimensions[false]; dimension) {
                segments.push_back({false, dimension});
            }
            if (const auto dimension = dimensions[true]; dimension) {
                segments.push_back({true, dimension});
            }
            result_edges.emplace_back(std::move(segments), edges(i).arrow());
        }
        auto result = Tensor<ScalarType, ParitySymmetry, Name>(names(), std::move(result_edges)).zero_();

        // copy every block into parity symmetry tensor
        // find the dimension of the block, the result leading is same to total dimension
        // and find the offset of destination
        // it is easy to get the offset of source then call transpose
        for (auto it = blocks().begin(); it.valid; ++it) {
            if (it->has_value()) {
                // Find the result block and generate the temporary_span
                pmr::vector<ParitySymmetry> block_indices;
                block_indices.reserve(rank());
                pmr::vector<Size> result_dimensions;
                result_dimensions.reserve(rank());
                pmr::vector<Size> result_indices;
                result_indices.reserve(rank());
                for (auto i = 0; i < rank(); i++) {
                    auto& [source_symmetry, source_dimension] = edges(i).segments(it.indices[i]);
                    block_indices.push_back(source_symmetry.parity());
                    result_dimensions.push_back(source_dimension);

                    Size index = 0;
                    for (auto [symmetry, dimension] : edges(i).segments()) {
                        if (symmetry == source_symmetry) {
                            break;
                        }
                        if (symmetry.parity() == source_symmetry.parity()) {
                            index += dimension;
                        }
                    }
                    result_indices.push_back(index);
                }
                auto& result_block = result.blocks(block_indices);

                mdspan<ScalarType, pmr::vector<Size>> temporary_span(
                    &result_block.at(result_indices),
                    std::move(result_dimensions),
                    {result_block.leadings().begin(), result_block.leadings().end()}
                );
                mdspan_transform(it->value(), temporary_span, [](const auto& x) { return x; });
            }
        }
        return result;
    }
} // namespace TAT
#endif
