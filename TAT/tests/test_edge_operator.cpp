#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_edge_operator, no_symmetry_example_0) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {8, 8}}.range_();
    auto b = a.edge_rename({{"A", "C"}}
    ).edge_operator({{"C", {{"D", 4}, {"E", 2}}}, {"B", {{"F", 2}, {"G", 4}}}}, {"D", "F"}, {{"I", {"D", "F"}}, {"J", {"G", "E"}}}, {"J", "I"});
    auto b_s = a.edge_rename({{"A", "C"}})
                   .split_edge({{"C", {{"D", 4}, {"E", 2}}}, {"B", {{"F", 2}, {"G", 4}}}})
                   .merge_edge({{"I", {"D", "F"}}, {"J", {"G", "E"}}})
                   .transpose({"J", "I"});
    ASSERT_FLOAT_EQ((b - b_s).norm<-1>(), 0);
}

TEST(test_edge_operator, u1_symmetry_example_0) {
    auto a = (TAT::Tensor<double, TAT::U1Symmetry>{
        {"Left", "Right", "Up", "Down"},
        {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 4}, {1, 2}}, {{-1, 2}, {0, 3}, {1, 1}}, {{-1, 1}, {0, 3}, {1, 2}}}}
                  .range_());
    auto b = a.edge_rename({{"Right", "Right1"}}).split_edge({{"Down", {{"Down1", {{{0, 1}, {1, 2}}}}, {"Down2", {{{-1, 1}, {0, 1}}}}}}});
    auto c = b.transpose({"Down1", "Right1", "Up", "Left", "Down2"});
    auto d = c.merge_edge({{"Left", {"Left", "Down2"}}});
    auto total = a.edge_rename({{"Right", "Right1"}})
                     .edge_operator(
                         {{"Down", {{"Down1", {{{0, 1}, {1, 2}}}}, {"Down2", {{{-1, 1}, {0, 1}}}}}}},
                         {},
                         {{"Left", {"Left", "Down2"}}},
                         {"Down1", "Right1", "Up", "Left"}
                     );
    ASSERT_FLOAT_EQ((total - d).norm<-1>(), 0);
}

TEST(test_edge_operator, fermi_symmetry_example_0) {
    auto a = (TAT::Tensor<double, TAT::FermiSymmetry>{
        {"Left", "Right", "Up", "Down"},
        {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 4}, {1, 2}}, {{-1, 2}, {0, 3}, {1, 1}}, {{-1, 1}, {0, 3}, {1, 2}}}}
                  .range_());
    auto b = a.edge_rename({{"Right", "Right1"}}).split_edge({{"Down", {{"Down1", {{{0, 1}, {1, 2}}}}, {"Down2", {{{-1, 1}, {0, 1}}}}}}});
    auto r = b.reverse_edge({"Left"});
    auto c = r.transpose({"Down1", "Right1", "Up", "Left", "Down2"});
    auto d = c.merge_edge({{"Left", {"Left", "Down2"}}});
    auto total = a.edge_rename({{"Right", "Right1"}})
                     .edge_operator(
                         {{"Down", {{"Down1", {{{0, 1}, {1, 2}}}}, {"Down2", {{{-1, 1}, {0, 1}}}}}}},
                         {"Left"},
                         {{"Left", {"Left", "Down2"}}},
                         {"Down1", "Right1", "Up", "Left"}
                     );
    ASSERT_FLOAT_EQ((total - d).norm<-1>(), 0);
}
