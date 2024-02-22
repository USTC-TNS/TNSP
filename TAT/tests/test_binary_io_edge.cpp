#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_binary_io_edge, bose_symmetry) {
    std::stringstream ss;
    TAT::Edge<TAT::Symmetry<int>> e1({{1, 2}, {2, 3}, {3, 4}});
    ss < e1;
    TAT::Edge<TAT::Symmetry<int>> e2;
    ss > e2;
    ASSERT_EQ(e1, e2);
}

TEST(test_binary_io_edge, fermi_symmetry) {
    std::stringstream ss;
    TAT::Edge<TAT::Symmetry<TAT::fermi<int>>> e1({{1, 2}, {2, 3}, {3, 4}}, true);
    ss < e1;
    TAT::Edge<TAT::Symmetry<TAT::fermi<int>>> e2;
    ss > e2;
    ASSERT_EQ(e1, e2);
}
