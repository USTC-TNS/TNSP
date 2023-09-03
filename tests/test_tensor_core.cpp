#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_tensor_core, create_core_no_symmetry) {
   auto a = TAT::Core<float, TAT::NoSymmetry>({3, 4});
   auto e0_a = TAT::Edge<TAT::NoSymmetry>(3);
   auto e1_a = TAT::Edge<TAT::NoSymmetry>(4);
   ASSERT_EQ(a.storage().size(), 3 * 4);
   ASSERT_EQ(a.edges(0), e0_a);
   ASSERT_EQ(a.edges(1), e1_a);
   ASSERT_THAT(a.edges(), ElementsAre(e0_a, e1_a));
   ASSERT_EQ(&a.blocks(std::vector<TAT::NoSymmetry>{{}, {}}), &a.blocks(std::vector<int>{0, 0}));
   ASSERT_EQ(&a.at(std::vector<TAT::Size>{2, 3}), &a.at(std::vector<std::pair<TAT::Nums, TAT::Size>>{{0, 2}, {0, 3}}));
   ASSERT_EQ(&a.at(std::vector<TAT::Size>{1, 0}), &a.at(std::vector<std::pair<TAT::NoSymmetry, TAT::Size>>{{{}, 1}, {{}, 0}}));
}

TEST(test_tensor_core, create_core_u1_symmetry) {
   const auto b = TAT::Core<float, TAT::U1Symmetry>({{{0, 2}, {1, 3}}, {{0, 4}, {-1, 5}}});
   auto e0_b = TAT::Edge<TAT::U1Symmetry>({{0, 2}, {1, 3}});
   auto e1_b = TAT::Edge<TAT::U1Symmetry>({{0, 4}, {-1, 5}});
   ASSERT_EQ(b.storage().size(), 2 * 4 + 3 * 5);
   ASSERT_EQ(b.edges(0), e0_b);
   ASSERT_EQ(b.edges(1), e1_b);
   ASSERT_THAT(b.edges(), ElementsAre(e0_b, e1_b));
   ASSERT_EQ(&b.blocks(std::vector<TAT::U1Symmetry>{0, 0}), &b.blocks(std::vector<int>{0, 0}));
   ASSERT_EQ(&b.blocks(std::vector<TAT::U1Symmetry>{1, -1}), &b.blocks(std::vector<int>{1, 1}));
   ASSERT_EQ(&b.at(std::vector<TAT::Size>{1, 3}), &b.at(std::vector<std::pair<TAT::Nums, TAT::Size>>{{0, 1}, {0, 3}}));
   ASSERT_EQ(&b.at(std::vector<TAT::Size>{3, 6}), &b.at(std::vector<std::pair<TAT::U1Symmetry, TAT::Size>>{{1, 1}, {-1, 2}}));
}

TEST(test_tensor_core, create_core_u1_symmetry_empty_blocks) {
   auto c = TAT::Core<float, TAT::U1Symmetry>({{}, {}});
   auto e0_c = TAT::Edge<TAT::U1Symmetry>();
   auto e1_c = TAT::Edge<TAT::U1Symmetry>();
   ASSERT_EQ(c.storage().size(), 0);
   ASSERT_EQ(c.edges(0), e0_c);
   ASSERT_EQ(c.edges(1), e1_c);
   ASSERT_THAT(c.edges(), ElementsAre(e0_c, e1_c));
}

TEST(test_tensor_core, copy_core) {
   auto a = TAT::Core<float, TAT::NoSymmetry>({3, 4});
   for (TAT::Size i = 0; i < 3; i++) {
      for (TAT::Size j = 0; j < 3; j++) {
         a.at(std::vector<TAT::Size>{i, j}) = i * 10 + j;
      }
   }
   auto b = a;
   for (TAT::Size i = 0; i < 3; i++) {
      for (TAT::Size j = 0; j < 3; j++) {
         auto& element_a = a.at(std::vector<TAT::Size>{i, j});
         auto& element_b = b.at(std::vector<TAT::Size>{i, j});
         ASSERT_EQ(element_a, element_b);
         ASSERT_NE(&element_a, &element_b);
      }
   }
}
