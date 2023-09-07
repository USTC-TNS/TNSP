#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_edge, segments) {
    ASSERT_THAT(TAT::Edge<TAT::Symmetry<>>().segments(), ElementsAre());
    ASSERT_THAT(TAT::Edge<TAT::Symmetry<>>(2).segments(), ElementsAre(Pair(TAT::Symmetry<>(), 2)));
    ASSERT_THAT(
        TAT::Edge<TAT::Symmetry<int>>({1, 2, 3}).segments(),
        ElementsAre(Pair(TAT::Symmetry<int>(1), 1), Pair(TAT::Symmetry<int>(2), 1), Pair(TAT::Symmetry<int>(3), 1))
    );
    auto e = TAT::Edge<TAT::Symmetry<int>>({1, 2, 3});
    ASSERT_THAT(
        TAT::Edge<TAT::Symmetry<int>>(e.segments()).segments(),
        ElementsAre(Pair(TAT::Symmetry<int>(1), 1), Pair(TAT::Symmetry<int>(2), 1), Pair(TAT::Symmetry<int>(3), 1))
    );
    ASSERT_THAT(
        TAT::Edge<TAT::Symmetry<int>>({{1, 2}, {2, 2}, {3, 2}}).segments(),
        ElementsAre(Pair(TAT::Symmetry<int>(1), 2), Pair(TAT::Symmetry<int>(2), 2), Pair(TAT::Symmetry<int>(3), 2))
    );
    ASSERT_THAT(
        TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}}).segments(),
        ElementsAre(Pair(TAT::Symmetry<int>(1), 2), Pair(TAT::Symmetry<int>(2), 2), Pair(TAT::Symmetry<int>(3), 2))
    );
    ASSERT_THAT(
        TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}}, true).segments(),
        ElementsAre(Pair(TAT::Symmetry<int>(1), 2), Pair(TAT::Symmetry<int>(2), 2), Pair(TAT::Symmetry<int>(3), 2))
    );
}

TEST(test_edge, arrow_when_construct) {
    ASSERT_EQ(TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}}, false).arrow(), false);
    ASSERT_EQ(TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}}, true).arrow(), true);
    ASSERT_EQ(TAT::Edge<TAT::Symmetry<TAT::bose<int>>>({{1, 2}, {2, 2}, {3, 2}}, false).arrow(), false);
    ASSERT_EQ(TAT::Edge<TAT::Symmetry<TAT::bose<int>>>({{1, 2}, {2, 2}, {3, 2}}, true).arrow(), false);
}

TEST(test_edge, arrow_update_fermi) {
    auto e1 = TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}}, false);
    ASSERT_EQ(e1.arrow(), false);
    e1.reverse_arrow();
    ASSERT_EQ(e1.arrow(), true);
    e1.set_arrow(false);
    ASSERT_EQ(e1.arrow(), false);
}

TEST(test_edge, arrow_update_bose) {
    auto e1 = TAT::Edge<TAT::Symmetry<TAT::bose<int>>>({{1, 2}, {2, 2}, {3, 2}}, false);
    ASSERT_EQ(e1.arrow(), false);
    e1.reverse_arrow();
    ASSERT_EQ(e1.arrow(), false);
    e1.set_arrow(false);
    ASSERT_EQ(e1.arrow(), false);
}

TEST(test_edge, edge_compare_bose_arrow) {
    auto e1 = TAT::Edge<TAT::Symmetry<TAT::bose<int>>>({{1, 2}, {2, 2}, {3, 2}}, true);
    auto e2 = TAT::Edge<TAT::Symmetry<TAT::bose<int>>>({{1, 2}, {2, 2}, {3, 2}}, false);
    ASSERT_TRUE(e1 == e2);
}

TEST(test_edge, edge_compare_fermi_arrow) {
    auto e1 = TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}}, true);
    auto e2 = TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}}, false);
    ASSERT_TRUE(e1 != e2);
}

TEST(test_edge, edge_compare_seg_1) {
    auto e1 = TAT::Edge<TAT::Symmetry<TAT::bose<int>>>({{1, 2}, {2, 2}, {3, 1}});
    auto e2 = TAT::Edge<TAT::Symmetry<TAT::bose<int>>>({{1, 2}, {2, 2}, {3, 2}});
    ASSERT_TRUE(e1 != e2);
}

TEST(test_edge, edge_compare_seg_2) {
    auto e1 = TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}}, true);
    auto e2 = TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}}, true);
    ASSERT_TRUE(e1 == e2);
}

TEST(test_edge, compare_seg) {
    auto e1 = TAT::edge_segments_t<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}});
    auto e2 = TAT::edge_segments_t<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}});
    ASSERT_TRUE(e1 == e2);
    ASSERT_FALSE(e1 != e2);
}

TEST(test_edge, conjugate) {
    auto e1 = TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 2}}, true);
    auto e2 = TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{-1, 2}, {-2, 2}, {-3, 2}}, false);
    ASSERT_TRUE(e1 == e2.conjugated());
    e1.conjugate();
    ASSERT_TRUE(e1 == e2);
}

TEST(test_edge, segment_query) {
    auto e1 = TAT::Edge<TAT::Symmetry<TAT::fermi<int>>>({{1, 2}, {2, 2}, {3, 4}}, true);
    ASSERT_THAT(e1.segments(0), Pair(1, 2));

    ASSERT_THAT(e1.coord_by_point({2, 1}), Pair(1, 1));
    ASSERT_THAT(e1.point_by_coord({1, 1}), Pair(TAT::Symmetry<TAT::fermi<int>>(2), 1));
    ASSERT_THAT(e1.coord_by_index(3), Pair(1, 1));
    ASSERT_THAT(e1.index_by_coord({1, 1}), 3);
    ASSERT_THAT(e1.point_by_index(3), Pair(TAT::Symmetry<TAT::fermi<int>>(2), 1));
    ASSERT_THAT(e1.index_by_point({2, 1}), 3);

    ASSERT_THAT(e1.dimension_by_symmetry(1), 2);
    ASSERT_THAT(e1.dimension_by_symmetry(2), 2);
    ASSERT_THAT(e1.dimension_by_symmetry(3), 4);

    ASSERT_THAT(e1.total_dimension(), 8);
}
