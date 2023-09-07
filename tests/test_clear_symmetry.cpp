#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_clear_symmetry, bose_mode) {
    auto a = TAT::Tensor<float, TAT::U1Symmetry>({"i", "j"}, {{{0, 2}, {1, 3}, {2, 5}}, {{0, 5}, {-1, 4}, {-2, 2}}}).range();
    auto b = a.clear_symmetry();
    for (auto sym = 0; sym < 3; sym++) {
        auto dim_i = a.edges(0).dimension_by_symmetry(+sym);
        auto dim_j = a.edges(1).dimension_by_symmetry(-sym);
        for (auto i = 0; i < dim_i; i++) {
            for (auto j = 0; j < dim_j; j++) {
                auto index_i = a.edges(0).index_by_point({+sym, i});
                auto index_j = a.edges(1).index_by_point({-sym, j});
                ASSERT_EQ(a.at(std::vector{index_i, index_j}), b.at(std::vector{index_i, index_j}));
            }
        }
    }
}

TEST(test_clear_symmetry, fermi_mode) {
    auto a = TAT::Tensor<float, TAT::FermiSymmetry>({"i", "j"}, {{{0, 2}, {1, 3}, {2, 5}}, {{0, 5}, {-1, 4}, {-2, 2}}}).range();
    auto b = a.clear_symmetry();
    for (auto sym = 0; sym < 3; sym++) {
        auto dim_i = a.edges(0).dimension_by_symmetry(+sym);
        auto dim_j = a.edges(1).dimension_by_symmetry(-sym);
        for (auto i = 0; i < dim_i; i++) {
            for (auto j = 0; j < dim_j; j++) {
                bool p_sym = bool(sym % 2);
                auto p_i = i;
                for (const auto& [s, d] : a.edges(0).segments()) {
                    if (s == TAT::FermiSymmetry(+sym)) {
                        break;
                    }
                    if (s.parity() == p_sym) {
                        p_i += d;
                    }
                }
                auto p_j = j;
                for (const auto& [s, d] : a.edges(1).segments()) {
                    if (s == TAT::FermiSymmetry(-sym)) {
                        break;
                    }
                    if (s.parity() == p_sym) {
                        p_j += d;
                    }
                }
                auto index_i = a.edges(0).index_by_point({+sym, i});
                auto index_j = a.edges(1).index_by_point({-sym, j});
                ASSERT_EQ(a.at({{+sym, i}, {-sym, j}}), b.at({{p_sym, p_i}, {p_sym, p_j}}));
            }
        }
    }
}
