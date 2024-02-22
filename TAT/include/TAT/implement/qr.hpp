/**
 * \file qr.hpp
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
#ifndef TAT_QR_HPP
#define TAT_QR_HPP

#include "../structure/tensor.hpp"
#include "../utility/allocator.hpp"
#include "../utility/timer.hpp"

extern "C" {
    int sgeqrf_(const int* m, const int* n, float* A, const int* lda, float* tau, float* work, const int* lwork, int* info);
    int dgeqrf_(const int* m, const int* n, double* A, const int* lda, double* tau, double* work, const int* lwork, int* info);
    int cgeqrf_(
        const int* m,
        const int* n,
        std::complex<float>* A,
        const int* lda,
        std::complex<float>* tau,
        std::complex<float>* work,
        const int* lwork,
        int* info
    );
    int zgeqrf_(
        const int* m,
        const int* n,
        std::complex<double>* A,
        const int* lda,
        std::complex<double>* tau,
        std::complex<double>* work,
        const int* lwork,
        int* info
    );
    int sgelqf_(const int* m, const int* n, float* A, const int* lda, float* tau, float* work, const int* lwork, int* info);
    int dgelqf_(const int* m, const int* n, double* A, const int* lda, double* tau, double* work, const int* lwork, int* info);
    int cgelqf_(
        const int* m,
        const int* n,
        std::complex<float>* A,
        const int* lda,
        std::complex<float>* tau,
        std::complex<float>* work,
        const int* lwork,
        int* info
    );
    int zgelqf_(
        const int* m,
        const int* n,
        std::complex<double>* A,
        const int* lda,
        std::complex<double>* tau,
        std::complex<double>* work,
        const int* lwork,
        int* info
    );
    int sorgqr_(const int* m, const int* n, const int* k, float* A, const int* lda, const float* tau, float* work, const int* lwork, int* info);
    int dorgqr_(const int* m, const int* n, const int* k, double* A, const int* lda, const double* tau, double* work, const int* lwork, int* info);
    int cungqr_(
        const int* m,
        const int* n,
        const int* k,
        std::complex<float>* A,
        const int* lda,
        const std::complex<float>* tau,
        std::complex<float>* work,
        const int* lwork,
        int* info
    );
    int zungqr_(
        const int* m,
        const int* n,
        const int* k,
        std::complex<double>* A,
        const int* lda,
        const std::complex<double>* tau,
        std::complex<double>* work,
        const int* lwork,
        int* info
    );
    int sorglq_(const int* m, const int* n, const int* k, float* A, const int* lda, const float* tau, float* work, const int* lwork, int* info);
    int dorglq_(const int* m, const int* n, const int* k, double* A, const int* lda, const double* tau, double* work, const int* lwork, int* info);
    int cunglq_(
        const int* m,
        const int* n,
        const int* k,
        std::complex<float>* A,
        const int* lda,
        const std::complex<float>* tau,
        std::complex<float>* work,
        const int* lwork,
        int* info
    );
    int zunglq_(
        const int* m,
        const int* n,
        const int* k,
        std::complex<double>* A,
        const int* lda,
        const std::complex<double>* tau,
        std::complex<double>* work,
        const int* lwork,
        int* info
    );
}

namespace TAT {

    inline timer qr_kernel_guard("qr_kernel");

    namespace detail {
        template<typename ScalarType>
        constexpr auto geqrf = nullptr;
        template<>
        inline constexpr auto geqrf<float> = sgeqrf_;
        template<>
        inline constexpr auto geqrf<double> = dgeqrf_;
        template<>
        inline constexpr auto geqrf<std::complex<float>> = cgeqrf_;
        template<>
        inline constexpr auto geqrf<std::complex<double>> = zgeqrf_;
        template<typename ScalarType>
        constexpr auto gelqf = nullptr;
        template<>
        inline constexpr auto gelqf<float> = sgelqf_;
        template<>
        inline constexpr auto gelqf<double> = dgelqf_;
        template<>
        inline constexpr auto gelqf<std::complex<float>> = cgelqf_;
        template<>
        inline constexpr auto gelqf<std::complex<double>> = zgelqf_;
        template<typename ScalarType>
        constexpr auto orgqr = nullptr;
        template<>
        inline constexpr auto orgqr<float> = sorgqr_;
        template<>
        inline constexpr auto orgqr<double> = dorgqr_;
        template<>
        inline constexpr auto orgqr<std::complex<float>> = cungqr_;
        template<>
        inline constexpr auto orgqr<std::complex<double>> = zungqr_;
        template<typename ScalarType>
        constexpr auto orglq = nullptr;
        template<>
        inline constexpr auto orglq<float> = sorglq_;
        template<>
        inline constexpr auto orglq<double> = dorglq_;
        template<>
        inline constexpr auto orglq<std::complex<float>> = cunglq_;
        template<>
        inline constexpr auto orglq<std::complex<double>> = zunglq_;

        template<typename ScalarType>
        int to_int(const ScalarType& value) {
            if constexpr (is_complex<ScalarType>) {
                return int(value.real());
            } else {
                return int(value);
            }
        }

        template<typename ScalarType>
        void calculate_qr_kernel(
            const int& m,
            const int& n,
            const int& min,
            ScalarType* __restrict data,
            ScalarType* __restrict data_1,
            ScalarType* __restrict data_2,
            bool use_qr_not_lq
        ) {
            // m*n c matrix
            // n*m fortran matrix
            if (use_qr_not_lq) {
                // c qr -> fortran lq
                // LQ
                //
                // here X means matrix content, and Q means the data to generate Q matrix
                //
                // XX   X        XQ
                // XX   XX XX    XX
                // XX = XX XX -> XX
                //
                // XXX   X  XXX    XQQ
                // XXX = XX XXX -> XXQ
                int result;
                auto tau = no_initialize::pmr::vector<ScalarType>(min);
                const int lwork_query = -1;
                ScalarType float_lwork;
                gelqf<ScalarType>(&n, &m, data, &n, tau.data(), &float_lwork, &lwork_query, &result);
                if (result != 0) {
                    detail::what_if_lapack_error("Error in LQ");
                }
                const int lwork = to_int(float_lwork);
                auto work = no_initialize::pmr::vector<ScalarType>(lwork);
                gelqf<ScalarType>(&n, &m, data, &n, tau.data(), work.data(), &lwork, &result);
                if (result != 0) {
                    detail::what_if_lapack_error("Error in LQ");
                }
                // Q matrix
                // data n*m
                // data_1 min*m
                for (auto i = 0; i < m; i++) {
                    // it does not copy entire matrix for the first situation
                    // but it still copy useless lower triangular part
                    std::copy(data + i * n, data + i * n + min, data_1 + i * min);
                }
                orglq<ScalarType>(&min, &m, &min, data_1, &min, tau.data(), work.data(), &lwork, &result);
                // WRONG -> orglq<ScalarType>(&min, &min, &min, data_1, &min, tau.data(), work.data(), &lwork, &result);
                if (result != 0) {
                    detail::what_if_lapack_error("Error in LQ");
                }
                // L matrix
                for (auto i = 0; i < min; i++) {
                    std::fill(data_2 + i * n, data_2 + i * n + i, 0);
                    std::copy(data + i * n + i, data + i * n + n, data_2 + i * n + i);
                }
            } else {
                // c lq -> fortran qr
                // QR
                //
                // XX   XX       XX
                // XX   XX XX    QX
                // XX = XX  X -> QQ
                //
                // XXX   XX XXX    XXX
                // XXX = XX  XX -> QXX
                int result;
                auto tau = no_initialize::pmr::vector<ScalarType>(min);
                const int lwork_query = -1;
                ScalarType float_lwork;
                geqrf<ScalarType>(&n, &m, data, &n, tau.data(), &float_lwork, &lwork_query, &result);
                if (result != 0) {
                    detail::what_if_lapack_error("Error in QR");
                }
                const int lwork = to_int(float_lwork);
                auto work = no_initialize::pmr::vector<ScalarType>(lwork);
                geqrf<ScalarType>(&n, &m, data, &n, tau.data(), work.data(), &lwork, &result);
                if (result != 0) {
                    detail::what_if_lapack_error("Error in QR");
                }
                // Q matrix
                // it does copy the entire matrix for the both situation, it is different to the c qr branch
                std::copy(data, data + n * min, data_2); // this copy useless upper triangular part
                // fortran
                // data n*m
                // data_2 n*min
                orgqr<ScalarType>(&n, &min, &min, data_2, &n, tau.data(), work.data(), &lwork, &result);
                // WRONG -> orgqr<ScalarType>(&min, &min, &min, data_2, &n, tau.data(), work.data(), &lwork, &result);
                // same size of lwork
                if (result != 0) {
                    detail::what_if_lapack_error("Error in QR");
                }
                // R matrix
                for (auto i = 0; i < min; i++) {
                    std::copy(data + n * i, data + n * i + i + 1, data_1 + min * i);
                    std::fill(data_1 + min * i + i + 1, data_1 + min * i + min, 0);
                }
                std::copy(data + n * min, data + n * m, data_1 + min * min);
                // for the first situation, min == m, this copy do nothing
            }
        }

        template<typename ScalarType>
        void calculate_qr(
            const int& m,
            const int& n,
            const int& min,
            ScalarType* __restrict data,
            ScalarType* __restrict data_1,
            ScalarType* __restrict data_2,
            bool use_qr_not_lq
        ) {
            auto kernel_guard = qr_kernel_guard();
            // sometimes, transpose before qr/lq is faster, there is a simular operation in svd
            // by testing, m > n is better
            if (m > n) {
                auto new_data = no_initialize::pmr::vector<ScalarType>(n * m);
                auto old_data_1 = no_initialize::pmr::vector<ScalarType>(n * min);
                auto old_data_2 = no_initialize::pmr::vector<ScalarType>(min * m);
                matrix_transpose<pmr::vector<Size>>(m, n, data, new_data.data());
                calculate_qr_kernel(n, m, min, new_data.data(), old_data_1.data(), old_data_2.data(), !use_qr_not_lq);
                matrix_transpose<pmr::vector<Size>>(n, min, old_data_1.data(), data_2);
                matrix_transpose<pmr::vector<Size>>(min, m, old_data_2.data(), data_1);
            } else {
                calculate_qr_kernel(m, n, min, data, data_1, data_2, use_qr_not_lq);
            }
        }
    } // namespace detail

    inline timer qr_guard("qr");

    template<typename ScalarType, typename Symmetry, typename Name>
    typename Tensor<ScalarType, Symmetry, Name>::qr_result Tensor<ScalarType, Symmetry, Name>::qr(
        char free_names_direction,
        const std::unordered_set<Name>& free_names,
        const Name& common_name_q,
        const Name& common_name_r
    ) const {
        auto pmr_guard = scope_resource(default_buffer_size);
        auto timer_guard = qr_guard();

        constexpr bool is_fermi = Symmetry::is_fermi_symmetry;

        if constexpr (debug_mode) {
            // check free_name_set is valid
            for (const auto& name : free_names) {
                if (auto found = find_by_name(name); found == names().end()) {
                    detail::error("Missing name in qr");
                }
            }
        }

        // determine do LQ or QR
        bool use_r_name;
        if (free_names_direction == 'r' || free_names_direction == 'R') {
            use_r_name = true;
        } else if (free_names_direction == 'q' || free_names_direction == 'Q') {
            use_r_name = false;
        } else {
            detail::error("Invalid direction in QR");
        };
        bool use_qr_not_lq = names().empty() || ((free_names.find(names().back()) != free_names.end()) == use_r_name);
        // use r, last in r -> qr
        // use q, last in q -> last not in r -> lq

        // merge plan
        auto free_name_list_1 = pmr::vector<Name>(); // part of merge map
        auto free_name_list_2 = pmr::vector<Name>(); // part of merge map
        auto reversed_names_input = pmr::unordered_set<Name>(unordered_parameter * rank());
        // result name is trivial

        // split plan
        auto reversed_names_1 = pmr::unordered_set<Name>(unordered_parameter * rank());
        auto reversed_names_2 = pmr::unordered_set<Name>(unordered_parameter * rank());
        auto result_names_1 = std::vector<Name>();
        auto result_names_2 = std::vector<Name>();
        auto free_names_and_edges_1 = pmr::vector<std::pair<Name, edge_segments_t<Symmetry, true>>>(); // part of split map
        auto free_names_and_edges_2 = pmr::vector<std::pair<Name, edge_segments_t<Symmetry, true>>>(); // part of split map

        free_name_list_1.reserve(rank());
        free_name_list_2.reserve(rank());
        result_names_1.reserve(rank() + 1);
        result_names_2.reserve(rank() + 1);
        free_names_and_edges_1.reserve(rank());
        free_names_and_edges_2.reserve(rank());

        result_names_2.push_back(use_qr_not_lq ? common_name_r : common_name_q);
        for (Rank i = 0; i < rank(); i++) {
            const auto& n = names(i);
            // set.find() != set.end() => n in the set
            // (!=) == use_r_name => n in the r name
            // (!=) == use_r_name == use_qr_not_lq => in the second name
            if ((free_names.find(n) != free_names.end()) == use_r_name == use_qr_not_lq) {
                // tensor_2 side
                free_name_list_2.push_back(n);
                result_names_2.push_back(n);
                free_names_and_edges_2.push_back({n, {edges(i).segments()}});
                if constexpr (is_fermi) {
                    if (edges(i).arrow()) {
                        reversed_names_2.insert(n);
                        reversed_names_input.insert(n);
                    }
                }
            } else {
                // tensor_1 side
                free_name_list_1.push_back(n);
                result_names_1.push_back(n);
                free_names_and_edges_1.push_back({n, {edges(i).segments()}});
                if constexpr (is_fermi) {
                    if (edges(i).arrow()) {
                        reversed_names_1.insert(n);
                        reversed_names_input.insert(n);
                    }
                }
            }
        }
        result_names_1.push_back(use_qr_not_lq ? common_name_q : common_name_r);

        auto tensor_merged = edge_operator_implement(
            {},
            reversed_names_input,
            pmr::map<Name, pmr::vector<Name>>{
                {InternalName<Name>::QR_1, std::move(free_name_list_1)},
                {InternalName<Name>::QR_2, std::move(free_name_list_2)}},
            std::vector<Name>{InternalName<Name>::QR_1, InternalName<Name>::QR_2},
            false,
            {},
            {},
            {},
            {},
            {}
        );

        const auto& edge_0_input = tensor_merged.edges(0);
        const auto& edge_1_input = tensor_merged.edges(1);
        // prepare result tensor
        auto common_edge_segments_1 = std::vector<std::pair<Symmetry, Size>>();
        auto common_edge_segments_2 = std::vector<std::pair<Symmetry, Size>>();
        common_edge_segments_1.reserve(edge_0_input.segments_size());
        common_edge_segments_2.reserve(edge_0_input.segments_size());
        // arrow all false currently, arrow check is processed at the last
        for (const auto& [symmetry_0, dimension_0] : edge_0_input.segments()) {
            auto symmetry_1 = -symmetry_0;
            if (auto found = edge_1_input.find_by_symmetry(symmetry_1); found != edge_1_input.segments().end()) {
                // This block does exist
                auto m = dimension_0;
                auto n = found->second;
                auto k = m > n ? n : m;
                common_edge_segments_1.emplace_back(symmetry_1, k);
                common_edge_segments_2.emplace_back(symmetry_0, k);
            }
        }
        auto tensor_1 = Tensor<ScalarType, Symmetry, Name>{
            {InternalName<Name>::QR_1, use_qr_not_lq ? common_name_q : common_name_r},
            {std::move(tensor_merged.edges(0)), std::move(common_edge_segments_1)}};
        auto tensor_2 = Tensor<ScalarType, Symmetry, Name>{
            {use_qr_not_lq ? common_name_r : common_name_q, InternalName<Name>::QR_2},
            {std::move(common_edge_segments_2), std::move(tensor_merged.edges(1))}};
        // const auto& edge_common_1 = tensor_1.edges(1);
        const auto& edge_common_2 = tensor_2.edges(0);

        // call lapack
        for (const auto& [symmetry, _] : edge_0_input.segments()) {
            Size position_0_input = edge_0_input.find_by_symmetry(symmetry) - edge_0_input.segments().begin();
            Size position_1_input = edge_1_input.find_by_symmetry(-symmetry) - edge_1_input.segments().begin();
            if (position_1_input == edge_1_input.segments_size()) {
                continue;
            }
            Size position_common = edge_common_2.find_by_symmetry(symmetry) - edge_common_2.segments().begin();
            const int m = edge_0_input.segments(position_0_input).second;
            const int n = edge_1_input.segments(position_1_input).second;
            const int k = edge_common_2.segments(position_common).second;

            auto* data_1 = tensor_1.blocks(pmr::vector<Size>{position_0_input, position_common}).data();
            auto* data_2 = tensor_2.blocks(pmr::vector<Size>{position_common, position_1_input}).data();
            auto* data = tensor_merged.blocks(pmr::vector<Size>{position_0_input, position_1_input}).data();

            if (m * n != 0) {
                detail::calculate_qr<ScalarType>(m, n, k, data, data_1, data_2, use_qr_not_lq);
            }
        }

        // this is simular to svd
        //
        // no matter whether it is qr or lq
        // tensor 1 common edge should be true
        // tensor 2 common edge should be false
        //
        // beside reverse the input edge without sign
        // what need to do is: tensor_1 reverse without apply sign
        //
        // what will be done in the following code:
        // QR:
        // tensor_1 reverse without apply sign
        // LQ:
        // tensor_2 reverse with apply sign

        // tensor_1 not_apply_reverse -> tensor_2 apply_reverse operation is:
        // tensor_1 not_apply_reverse and tensor_2 apply_reverse, which does not change the network
        // so up till now, it maintains that tensor Q always have a true edge
        if constexpr (is_fermi) {
            (use_qr_not_lq ? reversed_names_1 : reversed_names_2).insert(common_name_q);
        }
        auto new_tensor_1 = tensor_1.edge_operator_implement(
            pmr::map<Name, pmr::vector<std::pair<Name, edge_segments_t<Symmetry, true>>>>{
                {InternalName<Name>::QR_1, std::move(free_names_and_edges_1)}},
            reversed_names_1,
            {},
            std::move(result_names_1),
            false,
            {},
            {},
            {},
            {},
            {}
        );
        auto new_tensor_2 = tensor_2.edge_operator_implement(
            pmr::map<Name, pmr::vector<std::pair<Name, edge_segments_t<Symmetry, true>>>>{
                {InternalName<Name>::QR_2, std::move(free_names_and_edges_2)}},
            reversed_names_2,
            {},
            std::move(result_names_2),
            false,
            {},
            use_qr_not_lq ? pmr::set<Name>{} : pmr::set<Name>{common_name_q},
            {},
            {},
            {}
        );
        return {std::move(use_qr_not_lq ? new_tensor_1 : new_tensor_2), std::move(use_qr_not_lq ? new_tensor_2 : new_tensor_1)};
    }
} // namespace TAT
#endif
