#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_text_io_edge, no_symmetry) {
    std::stringstream ss;
    TAT::Edge<TAT::Symmetry<>> e1(233);
    ss << e1;
    TAT::Edge<TAT::Symmetry<>> e2;
    ss >> e2;
    ASSERT_EQ(e1, e2);
    ASSERT_STREQ(ss.str().c_str(), "233");
}

TEST(test_text_io_edge, bose_symmetry) {
    std::stringstream ss;
    TAT::Edge<TAT::Symmetry<int>> e1({{1, 2}, {2, 3}, {3, 4}});
    ss << e1;
    TAT::Edge<TAT::Symmetry<int>> e2;
    ss >> e2;
    ASSERT_EQ(e1, e2);
    ASSERT_STREQ(ss.str().c_str(), "{1:2,2:3,3:4}");
}

TEST(test_text_io_edge, bose_symmetry_empty) {
    std::stringstream ss;
    TAT::Edge<TAT::Symmetry<int>> e1;
    ss << e1;
    TAT::Edge<TAT::Symmetry<int>> e2;
    ss >> e2;
    ASSERT_EQ(e1, e2);
    ASSERT_STREQ(ss.str().c_str(), "{}");
}

TEST(test_text_io_edge, fermi_symmetry) {
    std::stringstream ss;
    TAT::Edge<TAT::Symmetry<TAT::fermi<int>>> e1({{1, 2}, {2, 3}, {3, 4}}, true);
    ss << e1;
    TAT::Edge<TAT::Symmetry<TAT::fermi<int>>> e2;
    ss >> e2;
    ASSERT_EQ(e1, e2);
    ASSERT_STREQ(ss.str().c_str(), "{arrow:1,segment:{1:2,2:3,3:4}}");
}
