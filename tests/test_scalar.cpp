#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <valarray>

using namespace testing;

TEST(test_scalar, tensor_and_number) {
   auto a = TAT::Tensor<std::complex<double>, TAT::Z2Symmetry>{{"Left", "Right", "Phy"}, {{{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}}}
                  .range(); // 0..31
   std::valarray<std::complex<double>> s_a = std::valarray<std::complex<double>>(32);
   for (auto i = 0; i < 32; i++) {
      s_a[i] = i;
   }
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
   auto b = a + 1.0;
   std::valarray<std::complex<double>> s_b = s_a + 1.0;
   ASSERT_THAT(b.storage(), ElementsAreArray(std::begin(s_b), std::end(s_b)));
   auto c = 1.0 + a;
   std::valarray<std::complex<double>> s_c = 1.0 + s_a;
   ASSERT_THAT(c.storage(), ElementsAreArray(std::begin(s_c), std::end(s_c)));
   auto d = a - 1.0;
   std::valarray<std::complex<double>> s_d = s_a - 1.0;
   ASSERT_THAT(d.storage(), ElementsAreArray(std::begin(s_d), std::end(s_d)));
   auto e = 1.0 - a;
   std::valarray<std::complex<double>> s_e = 1.0 - s_a;
   ASSERT_THAT(e.storage(), ElementsAreArray(std::begin(s_e), std::end(s_e)));
   auto f = a * 1.5;
   std::valarray<std::complex<double>> s_f = s_a * 1.5;
   ASSERT_THAT(f.storage(), ElementsAreArray(std::begin(s_f), std::end(s_f)));
   auto g = 1.5 * a;
   std::valarray<std::complex<double>> s_g = 1.5 * s_a;
   ASSERT_THAT(g.storage(), ElementsAreArray(std::begin(s_g), std::end(s_g)));
   auto h = a / 1.5;
   std::valarray<std::complex<double>> s_h = s_a / 1.5;
   ASSERT_THAT(h.storage(), ElementsAreArray(std::begin(s_h), std::end(s_h)));
   auto i = 1.5 / (a + 1);
   std::valarray<std::complex<double>> s_i = 1.5 / (s_a + 1);
   ASSERT_THAT(i.storage(), ElementsAreArray(std::begin(s_i), std::end(s_i)));
}

TEST(test_scalar, tensor_and_tensor) {
   auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range();
   auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range(0, 0.1);
   std::valarray<double> s_a = std::valarray<double>(12);
   std::valarray<double> s_b = std::valarray<double>(12);
   s_a[0] = s_b[0] = 0;
   for (auto i = 1; i < 12; i++) {
      s_a[i] = s_a[i - 1] + 1;
      s_b[i] = s_b[i - 1] + 0.1;
   }
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
   ASSERT_THAT(b.storage(), ElementsAreArray(std::begin(s_b), std::end(s_b)));
   auto c = a + b;
   std::valarray<double> s_c = s_a + s_b;
   ASSERT_THAT(c.storage(), ElementsAreArray(std::begin(s_c), std::end(s_c)));
   auto d = a - b;
   std::valarray<double> s_d = s_a - s_b;
   ASSERT_THAT(d.storage(), ElementsAreArray(std::begin(s_d), std::end(s_d)));
   auto e = a * b;
   std::valarray<double> s_e = s_a * s_b;
   ASSERT_THAT(e.storage(), ElementsAreArray(std::begin(s_e), std::end(s_e)));
   auto f = a / (b + 1);
   std::valarray<double> s_f = s_a / (s_b + 1);
   ASSERT_THAT(f.storage(), ElementsAreArray(std::begin(s_f), std::end(s_f)));
}

TEST(test_scalar, tensor_and_number_inplace) {
   auto a = TAT::Tensor<std::complex<double>, TAT::Z2Symmetry>{{"Left", "Right", "Phy"}, {{{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}}}
                  .range(); // 0..31
   std::valarray<std::complex<double>> s_a = std::valarray<std::complex<double>>(32);
   for (auto i = 0; i < 32; i++) {
      s_a[i] = i;
   }
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
   a += 1.5;
   s_a += 1.5;
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
   a *= 0.9;
   s_a *= 0.9;
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
   a -= 0.1;
   s_a -= 0.1;
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
   a /= 2.0;
   s_a /= 2.0;
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
}

TEST(test_scalar, tensor_and_tensor_inplace) {
   auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range();
   auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range(0, 0.1);
   std::valarray<double> s_a = std::valarray<double>(12);
   std::valarray<double> s_b = std::valarray<double>(12);
   s_a[0] = s_b[0] = 0;
   for (auto i = 1; i < 12; i++) {
      s_a[i] = s_a[i - 1] + 1;
      s_b[i] = s_b[i - 1] + 0.1;
   }
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
   ASSERT_THAT(b.storage(), ElementsAreArray(std::begin(s_b), std::end(s_b)));
   a += b;
   s_a += s_b;
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
   a *= b;
   s_a *= s_b;
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
   a -= b;
   s_a -= s_b;
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
   a /= b + 1;
   s_a /= s_b + 1;
   ASSERT_THAT(a.storage(), ElementsAreArray(std::begin(s_a), std::end(s_a)));
}
