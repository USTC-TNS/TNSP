#define TAT_ERROR_BITS 2
#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_ownership, basic_usage_0) {
    auto a = TAT::Tensor<float, TAT::NoSymmetry>({"i"}, {233}).range();
    auto b = a;
    auto& cb = a.const_blocks();
    EXPECT_ANY_THROW({ auto& mb = a.blocks(); });
}

TEST(test_ownership, basic_usage_1) {
    const auto a = TAT::Tensor<float, TAT::NoSymmetry>({"i"}, {233}).range();
    auto b = a;
    auto& cb = a.const_blocks();
    auto& mb = a.blocks();
}
