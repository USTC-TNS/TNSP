/**
 * \file identity.hpp
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
#ifndef TAT_IDENTITY_HPP
#define TAT_IDENTITY_HPP

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/timer.hpp"

namespace TAT {
    template<typename ScalarType, typename Symmetry, typename Name>
    Tensor<ScalarType, Symmetry, Name>& Tensor<ScalarType, Symmetry, Name>::identity_(const std::unordered_set<std::pair<Name, Name>>& pairs) & {
        // the order of fermi arrow should be (false true) before set to delta
        auto pmr_guard = scope_resource(default_buffer_size);
        acquire_data_ownership("Set the the shared tensor to identity, copy happened here");
        auto half_rank = rank() / 2;

        auto ordered_index_pairs = pmr::vector<std::pair<Rank, Rank>>();
        ordered_index_pairs.reserve(half_rank);

        Rank destination_index = 0;
        pmr::vector<Rank> transpose_plan_source_to_destination;
        if constexpr (Symmetry::is_fermi_symmetry) {
            transpose_plan_source_to_destination.resize(rank());
        }

        auto valid_indices = pmr::vector<bool>(rank(), true);
        for (Rank i = 0; i < rank(); i++) {
            if (valid_indices[i]) {
                const auto& name_to_find = names(i);
                const Name* name_correspond = nullptr;
                for (const auto& [name_1, name_2] : pairs) {
                    if (name_1 == name_to_find) {
                        name_correspond = &name_2;
                        break;
                    }
                    if (name_2 == name_to_find) {
                        name_correspond = &name_1;
                        break;
                    }
                }
                auto index_correspond = rank_by_name(*name_correspond);
                valid_indices[index_correspond] = false;
                ordered_index_pairs.push_back({i, index_correspond});
                if constexpr (Symmetry::is_fermi_symmetry) {
                    if (edges(i).arrow() == false) {
                        // i index_corresponding
                        transpose_plan_source_to_destination[i] = destination_index++;
                        transpose_plan_source_to_destination[index_correspond] = destination_index++;
                    } else {
                        // index_corresponding i
                        transpose_plan_source_to_destination[index_correspond] = destination_index++;
                        transpose_plan_source_to_destination[i] = destination_index++;
                    }
                }
            }
        }

        zero_();

        for (auto it = blocks().begin(); it.valid; ++it) {
            if (!it->has_value()) {
                continue;
            }
            auto& block = it->value();

            auto symmetries = pmr::vector<Symmetry>();
            symmetries.reserve(rank());
            for (auto i = 0; i < rank(); i++) {
                symmetries.emplace_back(edges(i).segments(it.indices[i]).first);
            }

            bool not_diagonal = false;
            for (const auto& [i0, i1] : ordered_index_pairs) {
                not_diagonal = symmetries[i0] + symmetries[i1] != Symmetry();
                if (not_diagonal) {
                    break;
                }
            }
            if (not_diagonal) {
                continue;
            }

            auto pair_dimensions = pmr::vector<Size>();
            auto pair_leadings = pmr::vector<Size>();
            pair_dimensions.reserve(half_rank);
            pair_leadings.reserve(half_rank);
            for (const auto& [i0, i1] : ordered_index_pairs) {
                pair_dimensions.push_back(block.dimensions(i0));
                pair_leadings.push_back(block.leadings(i0) + block.leadings(i1));
                // ordered_pair_index order is from leading large to leading small so pair_leading is descreasing
            }

            bool parity = false;
            if constexpr (Symmetry::is_fermi_symmetry) {
                for (auto i = 0; i < rank(); i++) {
                    for (auto j = i + 1; j < rank(); j++) {
                        if (transpose_plan_source_to_destination[i] > transpose_plan_source_to_destination[j]) {
                            parity ^= symmetries[i].parity() && symmetries[j].parity();
                        }
                    }
                }
            }

            auto span_one = mdspan<ScalarType, pmr::vector<Size>>(block.data(), std::move(pair_dimensions), std::move(pair_leadings));
            if (parity) {
                for (auto i = span_one.begin(); i.valid; ++i) {
                    *i = -1;
                }
            } else {
                for (auto i = span_one.begin(); i.valid; ++i) {
                    *i = +1;
                }
            }
        }

        return *this;
    }
} // namespace TAT
#endif
