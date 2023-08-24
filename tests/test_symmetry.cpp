#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_symmetry, basic_usage) {
   auto x = TAT::Symmetry<>();
   ASSERT_EQ(TAT::Symmetry<int>(0), TAT::Symmetry<TAT::bose<int>>());
   ASSERT_EQ(TAT::Symmetry<int>(2), TAT::Symmetry<int>(1) + TAT::Symmetry<int>(1));
   ASSERT_EQ(TAT::Symmetry<TAT::fermi<int>>(2), TAT::Symmetry<TAT::fermi<int>>(1) + TAT::Symmetry<TAT::fermi<int>>(1));
   ASSERT_EQ(TAT::Symmetry<TAT::bose<bool>>(false), TAT::Symmetry<TAT::bose<bool>>(true) + TAT::Symmetry<TAT::bose<bool>>(true));
   ASSERT_EQ(
         (TAT::Symmetry<TAT::bose<bool>, TAT::fermi<int>>(false, 2)),
         (TAT::Symmetry<TAT::bose<bool>, TAT::fermi<int>>(true, 1) + TAT::Symmetry<TAT::bose<bool>, TAT::fermi<int>>(true, 1)));
   auto y = TAT::Symmetry<int, bool>(3, false);
   y += TAT::Symmetry<int, bool>(1, false);
   ASSERT_EQ(y, (TAT::Symmetry<int, bool>(4, false)));
   y -= TAT::Symmetry<int, bool>(2, true);
   ASSERT_EQ(y, (TAT::Symmetry<int, bool>(2, true)));
   y = -y;
   ASSERT_EQ(y, (TAT::Symmetry<int, bool>(-2, true)));
   y = TAT::Symmetry<int, bool>(3, false) - y;
   ASSERT_EQ(y, (TAT::Symmetry<int, bool>(5, true)));
}

TEST(test_symmetry, hash_and_order) {
   auto x = std::map<TAT::Symmetry<bool, int>, int>();
   x[{6, false}] = 7;
   x[{2, true}] = 9;
   ASSERT_EQ((x[{6, false}]), 7);
   ASSERT_EQ((x[{2, true}]), 9);
   auto y = std::unordered_map<TAT::Symmetry<bool, int>, int>();
   y[{6, false}] = 7;
   y[{2, true}] = 9;
   ASSERT_EQ((y[{6, false}]), 7);
   ASSERT_EQ((y[{2, true}]), 9);
}

TEST(test_symmetry, parity) {
   ASSERT_EQ((TAT::Symmetry<int, int>(1, 2).parity()), false);
   ASSERT_EQ((TAT::Symmetry<TAT::fermi<int>, int>(1, 2).parity()), true);
   ASSERT_EQ((TAT::Symmetry<int, TAT::fermi<int>>(1, 2).parity()), false);
   ASSERT_EQ((TAT::Symmetry<bool, int>(true, 2).parity()), false);
   ASSERT_EQ((TAT::Symmetry<TAT::fermi<bool>, int>(true, 2).parity()), true);
   ASSERT_EQ((TAT::Symmetry<bool, TAT::fermi<int>>(true, 2).parity()), false);
}
