#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_transpose, no_symmetry_basic_0) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {2, 3}}.range_();
    auto b = a.transpose({"Right", "Left"});
    ASSERT_THAT(a.storage(), ElementsAre(0, 1, 2, 3, 4, 5));
    ASSERT_THAT(b.storage(), ElementsAre(0, 3, 1, 4, 2, 5));
}

TEST(test_transpose, no_symmetry_basic_1) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"i", "j", "k"}, {2, 3, 4}}.range_();
    for (auto result_edge : std::vector<
             std::vector<std::string>>{{"i", "j", "k"}, {"i", "k", "j"}, {"j", "k", "i"}, {"j", "i", "k"}, {"k", "i", "j"}, {"k", "j", "i"}}) {
        auto b = a.transpose(result_edge);
        for (auto i = 0; i < 2; i++) {
            for (auto j = 0; j < 3; j++) {
                for (auto k = 0; k < 4; k++) {
                    ASSERT_EQ(a.const_at({{"i", i}, {"j", j}, {"k", k}}), b.const_at({{"i", i}, {"j", j}, {"k", k}}));
                }
            }
        }
    }
}

TEST(test_transpose, no_symmetry_high_dimension) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"i", "j", "k", "l", "m", "n"}, {2, 2, 2, 2, 2, 2}}.range_();
    auto b = a.transpose({"l", "j", "i", "n", "k", "m"});
    for (auto i = 0; i < 2; i++) {
        for (auto j = 0; j < 2; j++) {
            for (auto k = 0; k < 2; k++) {
                for (auto l = 0; l < 2; l++) {
                    for (auto m = 0; m < 2; m++) {
                        for (auto n = 0; n < 2; n++) {
                            ASSERT_EQ(
                                a.at({{"i", i}, {"j", j}, {"k", k}, {"l", l}, {"m", m}, {"n", n}}),
                                b.at({{"i", i}, {"j", j}, {"k", k}, {"l", l}, {"m", m}, {"n", n}})
                            );
                        }
                    }
                }
            }
        }
    }
}

TEST(test_transpose, z2_symmetry_high_dimension) {
    auto edge = TAT::Edge<TAT::BoseZ2Symmetry>({{false, 2}, {true, 2}});
    auto a = TAT::Tensor<double, TAT::BoseZ2Symmetry>{{"i", "j", "k", "l", "m", "n"}, {edge, edge, edge, edge, edge, edge}}.range_();
    auto b = a.transpose({"l", "j", "i", "n", "k", "m"});
    for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < 4; j++) {
            for (auto k = 0; k < 4; k++) {
                for (auto l = 0; l < 4; l++) {
                    for (auto m = 0; m < 4; m++) {
                        for (auto n = 0; n < 4; n++) {
                            bool p_i = bool(i & 2);
                            bool p_j = bool(j & 2);
                            bool p_k = bool(k & 2);
                            bool p_l = bool(l & 2);
                            bool p_m = bool(m & 2);
                            bool p_n = bool(n & 2);
                            if (p_i ^ p_j ^ p_k ^ p_l ^ p_m ^ p_n) {
                                continue;
                            }
                            ASSERT_EQ(
                                a.at({{"i", i}, {"j", j}, {"k", k}, {"l", l}, {"m", m}, {"n", n}}),
                                b.at({{"i", i}, {"j", j}, {"k", k}, {"l", l}, {"m", m}, {"n", n}})
                            );
                        }
                    }
                }
            }
        }
    }
}

TEST(test_transpose, parity_symmetry_high_dimension) {
    auto edge = TAT::Edge<TAT::FermiZ2Symmetry>({{false, 2}, {true, 2}});
    auto a = TAT::Tensor<double, TAT::FermiZ2Symmetry>{{"i", "j", "k", "l", "m", "n"}, {edge, edge, edge, edge, edge, edge}}.range_();
    auto b = a.transpose({"l", "j", "i", "n", "k", "m"});
    for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < 4; j++) {
            for (auto k = 0; k < 4; k++) {
                for (auto l = 0; l < 4; l++) {
                    for (auto m = 0; m < 4; m++) {
                        for (auto n = 0; n < 4; n++) {
                            bool p_i = bool(i & 2);
                            bool p_j = bool(j & 2);
                            bool p_k = bool(k & 2);
                            bool p_l = bool(l & 2);
                            bool p_m = bool(m & 2);
                            bool p_n = bool(n & 2);
                            if (p_i ^ p_j ^ p_k ^ p_l ^ p_m ^ p_n) {
                                continue;
                            }
                            // ijklmn
                            // l(ijk)mn
                            // lj(i)kmn
                            // ljin(km)
                            bool parity = (p_l & (p_i ^ p_j ^ p_k)) ^ (p_j & p_i) ^ (p_n & (p_k ^ p_m));
                            if (parity) {
                                ASSERT_EQ(
                                    -a.at({{"i", i}, {"j", j}, {"k", k}, {"l", l}, {"m", m}, {"n", n}}),
                                    b.at({{"i", i}, {"j", j}, {"k", k}, {"l", l}, {"m", m}, {"n", n}})
                                );
                            } else {
                                ASSERT_EQ(
                                    a.at({{"i", i}, {"j", j}, {"k", k}, {"l", l}, {"m", m}, {"n", n}}),
                                    b.at({{"i", i}, {"j", j}, {"k", k}, {"l", l}, {"m", m}, {"n", n}})
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
