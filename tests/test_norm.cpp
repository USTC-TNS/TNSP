#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_norm, basic_usage) {
    auto t = (TAT::Tensor<double, TAT::U1Symmetry>{
        {"Left", "Right", "Up"},
        {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
                  .range(7)
                  .to<std::complex<double>>());
    //  0  0  0 -> 1 2 3 -> 6
    //  0 +1 -1 -> 1 3 2 -> 6
    //  0 -1 +1 -> 1 1 1 -> 1
    // +1  0 -1 -> 2 2 2 -> 8
    // -1  0 +1 -> 3 2 1 -> 6
    // +1 -1  0 -> 2 1 3 -> 6
    // -1 +1  0 -> 3 3 3 -> 27
    // total: 60
    // from 7 to 66
    ASSERT_EQ(t.storage().size(), 60);
    ASSERT_FLOAT_EQ(t.norm<-1>(), 66);
    ASSERT_FLOAT_EQ(t.norm<0>(), 60);
    ASSERT_FLOAT_EQ(t.norm<1>(), (7 + 66) * 60 / 2);
    ASSERT_FLOAT_EQ(t.norm<2>(), std::pow((66 * (66 + 1) * (2 * 66 + 1) - 6 * (6 + 1) * (2 * 6 + 1)) / 6., 1 / 2.));
}

TEST(test_norm, rank3) {
    auto t = (TAT::Tensor<double, TAT::U1Symmetry>{
        {"Left", "Right", "Up"},
        {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
                  .range(7)
                  .to<std::complex<double>>());
    double expect = 0;
    constexpr int k = 3;
    for (int i = 7; i < 67; i++) {
        expect += std::pow(i, k);
    }
    expect = std::pow(expect, 1. / k);
    ASSERT_FLOAT_EQ(t.norm<k>(), expect);
}

TEST(test_norm, rank4) {
    auto t = (TAT::Tensor<double, TAT::U1Symmetry>{
        {"Left", "Right", "Up"},
        {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
                  .range(7));
    double expect = 0;
    constexpr int k = 4;
    for (int i = 7; i < 67; i++) {
        expect += std::pow(i, k);
    }
    expect = std::pow(expect, 1. / k);
    ASSERT_FLOAT_EQ(t.norm<k>(), expect);
}
