#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_binary_io, no_symmetry) {
    std::stringstream ss;
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right", "Up"}, {2, 3, 4}}.range_();
    ss < a;
    auto b = TAT::Tensor<double, TAT::NoSymmetry>();
    ss > b;
    ASSERT_FLOAT_EQ((a - b).norm<-1>(), 0);
}

TEST(test_binary_io, u1_symmetry) {
    std::stringstream ss;
    auto a = (TAT::Tensor<double, TAT::U1Symmetry>{
        {"Left", "Right", "Up"},
        {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
                  .range_(2));
    ss < a;
    auto b = TAT::Tensor<double, TAT::U1Symmetry>();
    ss > b;
    ASSERT_FLOAT_EQ((a - b).norm<-1>(), 0);
}

TEST(test_binary_io, no_symmetry_complex_number) {
    std::stringstream ss;
    auto a = TAT::Tensor<std::complex<int>, TAT::NoSymmetry>{{"Up", "Left", "Right"}, {1, 2, 3}}.set_([]() {
        static int i = 0;
        static int arr[6] = {0x12345, 0x23456, 0x34567, 0x45678, 0x56789, 0x6789a};
        return arr[i++];
    });
    ss < a;
    auto b = TAT::Tensor<std::complex<int>, TAT::NoSymmetry>();
    ss > b;
    ASSERT_FLOAT_EQ((a - b).norm<-1>(), 0);
}

TEST(test_binary_io, u1_symmetry_complex_number) {
    std::stringstream ss;
    auto a =
        TAT::Tensor<std::complex<double>, TAT::U1Symmetry>{
            {"Left", "Right", "Up"},
            {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
            .range_(2);
    ss < a;
    auto b = TAT::Tensor<std::complex<double>, TAT::U1Symmetry>();
    ss > b;
    ASSERT_FLOAT_EQ((a - b).norm<-1>(), 0);
}

TEST(test_binary_io, fermi_symmetry) {
    std::stringstream ss;
#define f_edge(...) \
    { {__VA_ARGS__}, false }
#define t_edge(...) \
    { {__VA_ARGS__}, true }
    auto a = (TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>{
        {"Left", "Right", "Up"},
        {t_edge({-2, 3}, {0, 1}, {-1, 2}), f_edge({0, 2}, {1, 3}), f_edge({0, 3}, {1, 1})}}
                  .range_(2));
    ss < a;
    auto b = TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>();
    ss > b;
    ASSERT_FLOAT_EQ((a - b).norm<-1>(), 0);
}

TEST(test_binary_io, dump_and_load) {
    std::stringstream ss;
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right", "Up"}, {2, 3, 4}}.range_();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>();
    b.load(a.dump());
    ASSERT_FLOAT_EQ((a - b).norm<-1>(), 0);
}

TEST(test_binary_io, rvalue_of_stream) {
    using namespace TAT;

    int v = 2333;
    std::string vs(100, ' ');
    *reinterpret_cast<int*>(vs.data()) = v;
    std::istringstream in(vs);
    int i;
    in > i;
    ASSERT_EQ(i, v);
    std::string out_str = static_cast<std::ostringstream&&>(std::ostringstream() < i).str();
    ASSERT_EQ(*reinterpret_cast<int*>(out_str.data()), v);
}
