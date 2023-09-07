/**
 * \file exponential.hpp
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
#ifndef TAT_EXPONENTIAL_HPP
#define TAT_EXPONENTIAL_HPP

#include <algorithm>
#include <cmath>

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/timer.hpp"
#include "contract.hpp"

extern "C" {
    int sgesv_(const int* n, const int* nrhs, float* A, const int* lda, int* ipiv, float* B, const int* ldb, int* info);
    int dgesv_(const int* n, const int* nrhs, double* A, const int* lda, int* ipiv, double* B, const int* ldb, int* info);
    int cgesv_(const int* n, const int* nrhs, std::complex<float>* A, const int* lda, int* ipiv, std::complex<float>* B, const int* ldb, int* info);
    int zgesv_(const int* n, const int* nrhs, std::complex<double>* A, const int* lda, int* ipiv, std::complex<double>* B, const int* ldb, int* info);
}

namespace TAT {
    namespace detail {
        template<typename ScalarType>
        constexpr int (*gesv)(const int* n, const int* nrhs, ScalarType* A, const int* lda, int* ipiv, ScalarType* B, const int* ldb, int* info) =
            nullptr;
        template<>
        inline auto gesv<float> = sgesv_;
        template<>
        inline auto gesv<double> = dgesv_;
        template<>
        inline auto gesv<std::complex<float>> = cgesv_;
        template<>
        inline auto gesv<std::complex<double>> = zgesv_;

        template<typename ScalarType>
        void linear_solve(int n, ScalarType* A, int nrhs, ScalarType* B, ScalarType* X) {
            // AX=B
            // A: n*n
            // B: n*nrhs
            no_initialize::pmr::vector<ScalarType> AT(n * n);
            matrix_transpose<pmr::vector<Size>>(n, n, A, AT.data());
            no_initialize::pmr::vector<ScalarType> BT(n * nrhs);
            matrix_transpose<pmr::vector<Size>>(n, nrhs, B, BT.data());
            no_initialize::pmr::vector<int> ipiv(n);
            int result;
            gesv<ScalarType>(&n, &nrhs, AT.data(), &n, ipiv.data(), BT.data(), &n, &result);
            if (result != 0) {
                detail::what_if_lapack_error("error in GESV");
            }
            matrix_transpose<pmr::vector<Size>>(nrhs, n, BT.data(), X);
        }

        template<typename ScalarType>
        auto max_of_abs(const ScalarType* data, Size n) {
            real_scalar<ScalarType> result = 0;
            for (Size i = 0; i < n * n; i++) {
                auto here = std::abs(data[i]);
                result = result < here ? here : result;
            }
            return result;
        }

        template<typename ScalarType>
        void initialize_identity_matrix(ScalarType* data, Size n) {
            for (Size i = 0; i < n - 1; i++) {
                *(data++) = 1;
                for (Size j = 0; j < n; j++) {
                    *(data++) = 0;
                }
            }
            *data = 1;
        }

        template<typename ScalarType>
        void matrix_exponential(Size n, ScalarType* A, ScalarType* F, int q) {
            int int_n = n;
            // j = max(0, 1+floor(log2(|A|_inf)))
            auto j = std::max(0, 1 + int(std::log2(max_of_abs(A, n))));
            // A = A/2^j
            ScalarType parameter = ScalarType(1) / ScalarType(1 << j);
            for (Size i = 0; i < n * n; i++) {
                A[i] *= parameter;
            }
            // D=I, N=I, X=I, c=1
            no_initialize::pmr::vector<ScalarType> D(n * n);
            initialize_identity_matrix(D.data(), n);
            no_initialize::pmr::vector<ScalarType> N(n * n);
            initialize_identity_matrix(N.data(), n);
            no_initialize::pmr::vector<ScalarType> X1(n * n);
            initialize_identity_matrix(X1.data(), n);
            no_initialize::pmr::vector<ScalarType> X2(n * n);
            ScalarType c = 1;
            // for k=1:q
            const ScalarType alpha = 1;
            const ScalarType beta = 0;
            for (auto k = 1; k <= q; k++) {
                // c = (c*(q-k+1))/((2*q-k+1)*k)
                c = (c * ScalarType(q - k + 1)) / ScalarType((2 * q - k + 1) * k);
                // X = A@X, N=N+c*X, D=D+(-1)^k*c*X
                auto& X_old = k % 2 == 1 ? X1 : X2;
                auto& X_new = k % 2 == 0 ? X1 : X2;
                // new = A @ old
                // new.T = old.T @ A.T
                detail::gemm<ScalarType>("N", "N", &int_n, &int_n, &int_n, &alpha, X_old.data(), &int_n, A, &int_n, &beta, X_new.data(), &int_n);
                ScalarType d = k % 2 == 0 ? c : -c;
                for (Size i = 0; i < n * n; i++) {
                    auto x = X_new[i];
                    N[i] += c * x;
                    D[i] += d * x;
                }
            }
            // solve D@F=N for F
            no_initialize::pmr::vector<ScalarType> F1(n * n);
            no_initialize::pmr::vector<ScalarType> F2(n * n);
            auto* R = j == 0 ? F : F1.data();
            // D@R=N
            linear_solve<ScalarType>(n, D.data(), n, N.data(), R);
            // for k=1:j
            for (auto k = 1; k <= j; k++) {
                // F = F@F
                const auto* F_old = k % 2 == 1 ? F1.data() : F2.data();
                auto* F_new = k == j ? F : k % 2 == 0 ? F1.data() : F2.data();
                // new = old * old
                // new.T = old.T * old.T
                detail::gemm<ScalarType>("N", "N", &int_n, &int_n, &int_n, &alpha, F_old, &int_n, F_old, &int_n, &beta, F_new, &int_n);
            }
        }
    } // namespace detail

    inline timer exponential_guard("exponential");

    template<typename ScalarType, typename Symmetry, typename Name>
    Tensor<ScalarType, Symmetry, Name>
    Tensor<ScalarType, Symmetry, Name>::exponential(const std::unordered_set<std::pair<Name, Name>>& pairs, int step) const {
        auto pmr_guard = scope_resource(default_buffer_size);
        auto timer_guard = exponential_guard();

        Rank half_rank = rank() / 2;
        // reverse -> merge -> exp -> split -> reverse

        // split map and merge map
        auto merge_map = pmr::unordered_map<Name, pmr::vector<Name>>(unordered_parameter * 2);
        auto& merge_1 = merge_map[InternalName<Name>::Exp_1];
        merge_1.reserve(half_rank);
        auto& merge_2 = merge_map[InternalName<Name>::Exp_2];
        merge_2.reserve(half_rank);
        auto split_map_result = pmr::unordered_map<Name, pmr::vector<std::pair<Name, edge_segments_t<Symmetry>>>>(unordered_parameter * 2);
        auto& split_1 = split_map_result[InternalName<Name>::Exp_1];
        split_1.reserve(half_rank);
        auto& split_2 = split_map_result[InternalName<Name>::Exp_2];
        split_2.reserve(half_rank);

        // split result name
        auto result_names = std::vector<Name>();
        result_names.reserve(rank());

        // apply merge/split flag
        // apply reverse flag
        // reverse set
        auto apply_reverse_parity_names = pmr::unordered_set<Name>(unordered_parameter * rank());
        auto reverse_names = pmr::unordered_set<Name>(unordered_parameter * rank());
        // merged edge arrow is (false true)

        auto valid_indices = pmr::vector<bool>(rank(), true);
        for (Rank i = rank(); i-- > 0;) {
            if (valid_indices[i]) {
                const auto& name_to_found = names(i);
                const Name* name_correspond = nullptr;
                bool this_former = false;
                for (const auto& [name_1, name_2] : pairs) {
                    if (name_1 == name_to_found) {
                        name_correspond = &name_2;
                        this_former = true;
                        break;
                    }
                    if (name_2 == name_to_found) {
                        name_correspond = &name_1;
                        this_former = false;
                        break;
                    }
                }
                auto index_correspond = rank_by_name(*name_correspond);
                valid_indices[index_correspond] = false;

                const auto& name_1 = this_former ? name_to_found : *name_correspond;
                const auto& index_1 = this_former ? i : index_correspond;
                const auto& name_2 = this_former ? *name_correspond : name_to_found;
                const auto& index_2 = this_former ? index_correspond : i;

                merge_1.push_back(name_1);
                merge_2.push_back(name_2);
                split_1.push_back({name_1, {edges(index_1).segments()}});
                split_2.push_back({name_2, {edges(index_2).segments()}});
                if constexpr (debug_mode) {
                    if (edges(index_1).conjugated() != edges(index_2)) {
                        detail::error("Incompatible edges in exponential");
                    }
                }
                if constexpr (Symmetry::is_fermi_symmetry) {
                    // reverse flag and reverse set
                    if (edges(index_1).arrow() != false) {
                        reverse_names.insert(name_1);
                        reverse_names.insert(name_2);
                        apply_reverse_parity_names.insert(name_2); // only apply to edge 2
                    }
                }
            }
        }
        std::reverse(merge_1.begin(), merge_1.end());
        std::reverse(merge_2.begin(), merge_2.end());
        std::reverse(split_1.begin(), split_1.end());
        std::reverse(split_2.begin(), split_2.end());

        for (const auto& name : merge_1) {
            result_names.push_back(name);
        }
        for (const auto& name : merge_2) {
            result_names.push_back(name);
        }

        auto tensor_merged = edge_operator_implement(
            {},
            reverse_names,
            merge_map,
            std::vector<Name>{InternalName<Name>::Exp_1, InternalName<Name>::Exp_2},
            false,
            {},
            apply_reverse_parity_names,
            {},
            pmr::unordered_set<Name>{InternalName<Name>::Exp_2},
            {}
        );
        auto result = tensor_merged.same_shape();

        for (auto i = 0; i < tensor_merged.blocks().size(); i++) {
            if (!tensor_merged.blocks().data()[i].has_value()) {
                continue;
            }
            auto& data_source = tensor_merged.blocks().data()[i].value();
            auto& data_destination = result.blocks().data()[i].value();
            auto n = data_destination.dimensions(0);
            detail::matrix_exponential(n, data_source.data(), data_destination.data(), step);
        }
        return result.edge_operator_implement(
            split_map_result,
            reverse_names,
            {},
            std::move(result_names),
            false,
            pmr::unordered_set<Name>{InternalName<Name>::Exp_2},
            apply_reverse_parity_names,
            {},
            {},
            {}
        );
    }
} // namespace TAT
#endif
