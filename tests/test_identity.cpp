#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

int arrange_pairs_indices[15][6] = {
      {0, 1, 2, 3, 4, 5},
      {0, 1, 2, 4, 5, 3},
      {0, 1, 2, 5, 3, 4},

      {0, 2, 1, 3, 4, 5},
      {0, 2, 1, 4, 5, 3},
      {0, 2, 1, 5, 3, 4},

      {0, 3, 1, 2, 4, 5},
      {0, 3, 1, 4, 5, 2},
      {0, 3, 1, 5, 2, 4},

      {0, 4, 1, 2, 3, 5},
      {0, 4, 1, 3, 5, 2},
      {0, 4, 1, 5, 2, 3},

      {0, 5, 1, 2, 3, 4},
      {0, 5, 1, 3, 4, 2},
      {0, 5, 1, 4, 2, 3},
};

bool order_lists[8][3]{
      {0, 0, 0},
      {0, 0, 1},
      {0, 1, 0},
      {0, 1, 1},
      {1, 0, 0},
      {1, 0, 1},
      {1, 1, 0},
      {1, 1, 1},
};

TEST(test_identity, no_symmetry_0) {
   auto a = TAT::Tensor<float, TAT::NoSymmetry>({"i", "j"}, {4, 4}).identity({{"i", "j"}});
   ASSERT_FLOAT_EQ((a - a.contract(a, {{"i", "j"}})).norm<-1>(), 0);
   ASSERT_FLOAT_EQ((a - a.contract(a, {{"j", "i"}})).norm<-1>(), 0);
}

TEST(test_identity, no_symmetry_1) {
   auto a = TAT::Tensor<float, TAT::NoSymmetry>({"i", "j"}, {4, 4}).identity({{"j", "i"}});
   ASSERT_FLOAT_EQ((a - a.contract(a, {{"i", "j"}})).norm<-1>(), 0);
   ASSERT_FLOAT_EQ((a - a.contract(a, {{"j", "i"}})).norm<-1>(), 0);
}

TEST(test_identity, no_symmetry_2) {
   int half_rank = 3;
   std::vector<std::string> names;
   for (auto pairs_index : arrange_pairs_indices) {
      auto a = TAT::Tensor<float, TAT::NoSymmetry>({"1", "2", "3", "4", "5", "6"}, {4, 4, 4, 4, 4, 4});
      auto pairs = std::unordered_set<std::pair<std::string, std::string>>();
      for (auto i = 0; i < half_rank; i++) {
         auto p0 = pairs_index[i * 2];
         auto p1 = pairs_index[i * 2 + 1];
         pairs.insert({a.names(p0), a.names(p1)});
      }
      a.identity(pairs);
      ASSERT_FLOAT_EQ((a - a.contract(a, pairs)).norm<-1>(), 0);
   }
}

TEST(test_identity, z2_symmetry_0) {
   int half_rank = 3;
   std::vector<std::string> names;
   for (auto pairs_index : arrange_pairs_indices) {
      auto edge = TAT::Edge<TAT::Z2Symmetry>({{false, 2}, {true, 2}});
      auto a = TAT::Tensor<float, TAT::Z2Symmetry>({"1", "2", "3", "4", "5", "6"}, {edge, edge, edge, edge, edge, edge});
      auto pairs = std::unordered_set<std::pair<std::string, std::string>>();
      for (auto i = 0; i < half_rank; i++) {
         auto p0 = pairs_index[i * 2];
         auto p1 = pairs_index[i * 2 + 1];
         pairs.insert({a.names(p0), a.names(p1)});
      }
      a.identity(pairs);
      ASSERT_FLOAT_EQ((a - a.contract(a, pairs)).norm<-1>(), 0);
   }
}

TEST(test_identity, u1_symmetry_0) {
   int half_rank = 3;
   std::vector<std::string> names;
   for (auto pairs_index : arrange_pairs_indices) {
      auto edge0 = TAT::Edge<TAT::U1Symmetry>({{-1, 1}, {0, 1}, {+1, 1}});
      auto edge1 = TAT::Edge<TAT::U1Symmetry>({{+1, 1}, {0, 1}, {-1, 1}});
      auto names = std::vector<std::string>{"1", "2", "3", "4", "5", "6"};
      auto edges = std::vector<TAT::Edge<TAT::U1Symmetry>>(6);
      auto pairs = std::unordered_set<std::pair<std::string, std::string>>();
      for (auto i = 0; i < half_rank; i++) {
         auto p0 = pairs_index[i * 2];
         auto p1 = pairs_index[i * 2 + 1];
         pairs.insert({names[p0], names[p1]});
         edges[p0] = edge0;
         edges[p1] = edge1;
      }
      auto a = TAT::Tensor<float, TAT::U1Symmetry>(names, edges);
      a.identity(pairs);
      ASSERT_FLOAT_EQ((a - a.contract(a, pairs)).norm<-1>(), 0);
   }
}

TEST(test_identity, fermi_symmetry_0) {
   int half_rank = 3;
   std::vector<std::string> names;
   for (auto order : order_lists) {
      for (auto pairs_index : arrange_pairs_indices) {
         auto edge0 = TAT::Edge<TAT::FermiSymmetry>({{-1, 1}, {0, 1}, {+1, 1}}, false);
         auto edge1 = TAT::Edge<TAT::FermiSymmetry>({{+1, 1}, {0, 1}, {-1, 1}}, true);
         auto names = std::vector<std::string>{"1", "2", "3", "4", "5", "6"};
         auto edges = std::vector<TAT::Edge<TAT::FermiSymmetry>>(6);
         auto pairs = std::unordered_set<std::pair<std::string, std::string>>();
         for (auto i = 0; i < half_rank; i++) {
            auto p0 = pairs_index[i * 2];
            auto p1 = pairs_index[i * 2 + 1];
            pairs.insert({names[p0], names[p1]});
            if (order[i]) {
               edges[p0] = edge0;
               edges[p1] = edge1;
            } else {
               edges[p0] = edge1;
               edges[p1] = edge0;
            }
         }
         auto a = TAT::Tensor<float, TAT::FermiSymmetry>(names, edges);
         a.identity(pairs);
         ASSERT_FLOAT_EQ((a - a.contract(a, pairs)).norm<-1>(), 0);
      }
   }
}
