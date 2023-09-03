#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_split_and_merge, no_symmetry_basic) {
   const auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {2, 3}}.range();

   const auto b = a.merge_edge({{"Merged", {"Left", "Right"}}});
   ASSERT_THAT(b.storage(), ElementsAreArray(a.storage()));
   const auto b_a = b.split_edge({{"Merged", {{"Left", 2}, {"Right", 3}}}});
   ASSERT_FLOAT_EQ((b_a - a).norm<-1>(), 0);

   auto c = a.merge_edge({{"Merged", {"Right", "Left"}}});
   ASSERT_THAT(c.storage(), ElementsAreArray(a.transpose({"Right", "Left"}).storage()));
   const auto c_a = c.split_edge({{"Merged", {{"Right", 3}, {"Left", 2}}}});
   ASSERT_FLOAT_EQ((c_a - a).norm<-1>(), 0);
}

TEST(test_split_and_merge, no_symmetry_high_dimension) {
   const auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"1", "2", "3", "4", "5", "6", "7", "8"}, {2, 2, 2, 2, 2, 2, 2, 2}}.range();
   for (auto i = 0; i < 8; i++) {
      for (auto j = i; j < 8; j++) {
         std::vector<std::string> names;
         std::vector<std::pair<std::string, TAT::edge_segments_t<TAT::NoSymmetry>>> plans;
         for (auto k = i; k < j; k++) {
            names.push_back(a.names(k));
            plans.push_back({a.names(k), 2});
         }
         const auto b = a.merge_edge({{"m", names}});
         ASSERT_THAT(b.storage(), ElementsAreArray(a.storage()));
         const auto c = b.split_edge({{"m", plans}});
         ASSERT_FLOAT_EQ((c - a).norm<-1>(), 0);
      }
   }
}

TEST(test_split_and_merge, u1_symmetry_basic) {
   const auto a = TAT::Tensor<double, TAT::U1Symmetry>{{"i", "j"}, {{-1, 0, +1}, {-1, 0, +1}}}.range();
   const auto d = TAT::Tensor<double, TAT::U1Symmetry>{{"m"}, {{{-2, 1}, {-1, 2}, {0, 3}, {+1, 2}, {+2, 1}}}}.range();

   const auto b = a.merge_edge({{"m", {"i", "j"}}});
   ASSERT_FLOAT_EQ((d - b).norm<-1>(), 0);
   const auto c = b.split_edge({{"m", {{"i", {{-1, 0, +1}}}, {"j", {{-1, 0, +1}}}}}});
   ASSERT_FLOAT_EQ((c - a).norm<-1>(), 0);
}

TEST(test_split_and_merge, u1_symmetry_high_dimension) {
   const auto edge = TAT::Edge<TAT::U1Symmetry>({{-1, 2}, {0, 2}, {+1, 2}});
   // 6^5 = 7776
   const auto a = TAT::Tensor<double, TAT::U1Symmetry>{{"1", "2", "3", "4", "5"}, {edge, edge, edge, edge, edge}}.range();
   for (auto i = 0; i < 5; i++) {
      for (auto j = i; j < 5; j++) {
         std::vector<std::string> names;
         std::vector<std::pair<std::string, TAT::edge_segments_t<TAT::U1Symmetry>>> plans;
         for (auto k = i; k < j; k++) {
            names.push_back(a.names(k));
            plans.push_back({a.names(k), edge});
         }
         const auto b = a.merge_edge({{"m", names}});
         const auto c = b.split_edge({{"m", plans}});
         ASSERT_FLOAT_EQ((c - a).norm<-1>(), 0);
      }
   }
}

TEST(test_split_and_merge, fermi_symmetry_high_dimension) {
   const auto edge = TAT::Edge<TAT::FermiSymmetry>({{-1, 2}, {0, 2}, {+1, 2}});
   const auto a = TAT::Tensor<double, TAT::FermiSymmetry>{{"1", "2", "3", "4", "5"}, {edge, edge, edge, edge, edge}}.range();
   for (auto i = 0; i < 5; i++) {
      for (auto j = i; j < 5; j++) {
         for (auto apply_parity = 0; apply_parity < 2; apply_parity++) {
            std::vector<std::string> names;
            std::vector<std::pair<std::string, TAT::edge_segments_t<TAT::FermiSymmetry>>> plans;
            for (auto k = i; k < j; k++) {
               names.push_back(a.names(k));
               plans.push_back({a.names(k), edge});
            }
            const auto b = a.merge_edge({{"m", names}}, bool(apply_parity));
            const auto c = b.split_edge({{"m", plans}}, bool(apply_parity));
            ASSERT_FLOAT_EQ((c - a).norm<-1>(), 0);
         }
      }
   }
}

TEST(test_split_and_merge, fermi_symmetry_high_dimension_compare_u1) {
   const auto edge_u1 = TAT::Edge<TAT::U1Symmetry>({{-1, 1}, {0, 1}, {+1, 1}});
   const auto a_u1 = TAT::Tensor<double, TAT::U1Symmetry>{{"1", "2", "3", "4", "5"}, {edge_u1, edge_u1, edge_u1, edge_u1, edge_u1}}.range(1);
   const auto edge_f = TAT::Edge<TAT::FermiSymmetry>({{-1, 1}, {0, 1}, {+1, 1}});
   const auto a_f = TAT::Tensor<double, TAT::FermiSymmetry>{{"1", "2", "3", "4", "5"}, {edge_f, edge_f, edge_f, edge_f, edge_f}}.range(1);
   for (auto i = 0; i < 5; i++) {
      for (auto j = i; j < 5; j++) {
         for (auto apply_parity = 0; apply_parity < 2; apply_parity++) {
            std::vector<std::string> names;
            std::vector<std::pair<std::string, TAT::edge_segments_t<TAT::U1Symmetry>>> plans_u1;
            std::vector<std::pair<std::string, TAT::edge_segments_t<TAT::FermiSymmetry>>> plans_f;
            for (auto k = i; k < j; k++) {
               names.push_back(a_u1.names(k));
               plans_u1.push_back({a_u1.names(k), edge_u1});
               plans_f.push_back({a_f.names(k), edge_f});
            }
            const auto b_u1 = a_u1.merge_edge({{"m", names}});
            const auto b_f = a_f.merge_edge({{"m", names}}, bool(apply_parity));
            if (apply_parity) {
               int s[5];
               for (s[0] = -1; s[0] < 2; s[0]++) {
                  for (s[1] = -1; s[1] < 2; s[1]++) {
                     for (s[2] = -1; s[2] < 2; s[2]++) {
                        for (s[3] = -1; s[3] < 2; s[3]++) {
                           for (s[4] = -1; s[4] < 2; s[4]++) {
                              if (s[0] + s[1] + s[2] + s[3] + s[4] != 0) {
                                 continue;
                              }
                              float item = a_u1.at(
                                    std::vector<std::pair<TAT::U1Symmetry, TAT::Size>>{{s[0], 0}, {s[1], 0}, {s[2], 0}, {s[3], 0}, {s[4], 0}});
                              ASSERT_THAT(b_u1.storage(), Contains(item));
                              bool p[5];
                              for (int x = 0; x < 5; x++) {
                                 p[x] = s[x] != 0;
                              }
                              int count = 0;
                              for (int x = i; x < j; x++) {
                                 if (p[x]) {
                                    count++;
                                 }
                              }
                              bool parity = bool(count & 2);
                              if (parity) {
                                 ASSERT_THAT(b_f.storage(), Contains(-item));
                              } else {
                                 ASSERT_THAT(b_f.storage(), Contains(+item));
                              }
                           }
                        }
                     }
                  }
               }
            } else {
               ASSERT_THAT(b_f.storage(), ElementsAreArray(b_u1.storage()));
            }
         }
      }
   }
}
