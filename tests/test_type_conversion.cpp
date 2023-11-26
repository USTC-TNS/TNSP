#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_type_conversion, dummy) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range_(2);
    auto b = a.to<double>();
    ASSERT_EQ(a.storage().data(), b.storage().data());
}

TEST(test_type_conversion, inside_float) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range_(2);
    auto b = a.to<float>();
    ASSERT_THAT(a.storage(), ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
    ASSERT_THAT(b.storage(), ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
}

TEST(test_type_conversion, inside_complex) {
    auto a = TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range_(2);
    auto b = a.to<std::complex<float>>();
    ASSERT_THAT(a.storage(), ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
    ASSERT_THAT(b.storage(), ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
}

TEST(test_type_conversion, complex_to_float) {
    auto a = TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range_(2);
    auto b = a.to<float>();
    ASSERT_THAT(a.storage(), ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
    ASSERT_THAT(b.storage(), ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
}

TEST(test_type_conversion, float_to_complex) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range_(2);
    auto b = a.to<std::complex<float>>();
    ASSERT_THAT(a.storage(), ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
    ASSERT_THAT(b.storage(), ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
}
