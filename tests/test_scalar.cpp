#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <valarray>

using namespace testing;

template<typename A, typename B>
auto difference(const A& a, const B& b) {
    double error = 0;
    for (auto i = 0; i < a.size(); i++) {
        auto diff = std::abs(a[i] - b[i]);
        error = error < diff ? diff : error;
    }
    return error;
}

TEST(test_scalar, tensor_and_number) {
    auto a =
        TAT::Tensor<std::complex<double>, TAT::Z2Symmetry>{{"Left", "Right", "Phy"}, {{{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}}}.range_(
        ); // 0..31
    std::valarray<std::complex<double>> s_a = std::valarray<std::complex<double>>(32);
    for (auto i = 0; i < 32; i++) {
        s_a[i] = i;
    }
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
    auto b = a + 1.0;
    std::valarray<std::complex<double>> s_b = s_a + std::complex<double>(1.0);
    ASSERT_LE(difference(TAT::ensure_cpu(b.storage()), s_b), 1e-8);
    auto c = 1.0 + a;
    std::valarray<std::complex<double>> s_c = std::complex<double>(1.0) + s_a;
    ASSERT_LE(difference(TAT::ensure_cpu(c.storage()), s_c), 1e-8);
    auto d = a - 1.0;
    std::valarray<std::complex<double>> s_d = s_a - std::complex<double>(1.0);
    ASSERT_LE(difference(TAT::ensure_cpu(d.storage()), s_d), 1e-8);
    auto e = 1.0 - a;
    std::valarray<std::complex<double>> s_e = std::complex<double>(1.0) - s_a;
    ASSERT_LE(difference(TAT::ensure_cpu(e.storage()), s_e), 1e-8);
    auto f = a * 1.5;
    std::valarray<std::complex<double>> s_f = s_a * std::complex<double>(1.5);
    ASSERT_LE(difference(TAT::ensure_cpu(f.storage()), s_f), 1e-8);
    auto g = 1.5 * a;
    std::valarray<std::complex<double>> s_g = std::complex<double>(1.5) * s_a;
    ASSERT_LE(difference(TAT::ensure_cpu(g.storage()), s_g), 1e-8);
    auto h = a / 1.5;
    std::valarray<std::complex<double>> s_h = s_a / std::complex<double>(1.5);
    ASSERT_LE(difference(TAT::ensure_cpu(h.storage()), s_h), 1e-8);
    auto i = 1.5 / (a + 1);
    std::valarray<std::complex<double>> s_i = std::complex<double>(1.5) / (s_a + std::complex<double>(1));
    ASSERT_LE(difference(TAT::ensure_cpu(i.storage()), s_i), 1e-8);
}

TEST(test_scalar, tensor_and_tensor) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range_();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range_(0, 0.1);
    std::valarray<double> s_a = std::valarray<double>(12);
    std::valarray<double> s_b = std::valarray<double>(12);
    s_a[0] = s_b[0] = 0;
    for (auto i = 1; i < 12; i++) {
        s_a[i] = s_a[i - 1] + 1;
        s_b[i] = s_b[i - 1] + 0.1;
    }
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
    ASSERT_LE(difference(TAT::ensure_cpu(b.storage()), s_b), 1e-8);
    auto c = a + b;
    std::valarray<double> s_c = s_a + s_b;
    ASSERT_LE(difference(TAT::ensure_cpu(c.storage()), s_c), 1e-8);
    auto d = a - b;
    std::valarray<double> s_d = s_a - s_b;
    ASSERT_LE(difference(TAT::ensure_cpu(d.storage()), s_d), 1e-8);
    auto e = a * b;
    std::valarray<double> s_e = s_a * s_b;
    ASSERT_LE(difference(TAT::ensure_cpu(e.storage()), s_e), 1e-8);
    auto f = a / (b + 1);
    std::valarray<double> s_f = s_a / (s_b + double(1));
    ASSERT_LE(difference(TAT::ensure_cpu(f.storage()), s_f), 1e-8);
}

TEST(test_scalar, tensor_and_number_inplace) {
    auto a =
        TAT::Tensor<std::complex<double>, TAT::Z2Symmetry>{{"Left", "Right", "Phy"}, {{{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}}}.range_(
        ); // 0..31
    std::valarray<std::complex<double>> s_a = std::valarray<std::complex<double>>(32);
    for (auto i = 0; i < 32; i++) {
        s_a[i] = i;
    }
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
    a += 1.5;
    s_a += 1.5;
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
    a *= 0.9;
    s_a *= 0.9;
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
    a -= 0.1;
    s_a -= 0.1;
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
    a /= 2.0;
    s_a /= 2.0;
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
}

TEST(test_scalar, tensor_and_tensor_inplace) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range_();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range_(0, 0.1);
    std::valarray<double> s_a = std::valarray<double>(12);
    std::valarray<double> s_b = std::valarray<double>(12);
    s_a[0] = s_b[0] = 0;
    for (auto i = 1; i < 12; i++) {
        s_a[i] = s_a[i - 1] + 1;
        s_b[i] = s_b[i - 1] + 0.1;
    }
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
    ASSERT_LE(difference(TAT::ensure_cpu(b.storage()), s_b), 1e-8);
    a += b;
    s_a += s_b;
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
    a *= b;
    s_a *= s_b;
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
    a -= b;
    s_a -= s_b;
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
    a /= b + 1;
    s_a /= s_b + double(1);
    ASSERT_LE(difference(TAT::ensure_cpu(a.storage()), s_a), 1e-8);
}
