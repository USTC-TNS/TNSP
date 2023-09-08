#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_contract, no_symmetry_example_0) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {2, 2}}.range();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"C", "D"}, {2, 2}}.range();

    auto c = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"A", "C"}});
    auto c_expect =
        TAT::Tensor<double, TAT::NoSymmetry>{{"B", "D"}, {2, 2}}.set([i = 0, v = std::vector<double>{4, 6, 6, 10}]() mutable { return v[i++]; });
    ASSERT_FLOAT_EQ((c - c_expect).norm<-1>(), 0);

    auto d = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"A", "D"}});
    auto d_expect =
        TAT::Tensor<double, TAT::NoSymmetry>{{"B", "C"}, {2, 2}}.set([i = 0, v = std::vector<double>{2, 6, 3, 11}]() mutable { return v[i++]; });
    ASSERT_FLOAT_EQ((d - d_expect).norm<-1>(), 0);

    auto e = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"B", "C"}});
    auto e_expect =
        TAT::Tensor<double, TAT::NoSymmetry>{{"A", "D"}, {2, 2}}.set([i = 0, v = std::vector<double>{2, 3, 6, 11}]() mutable { return v[i++]; });
    ASSERT_FLOAT_EQ((e - e_expect).norm<-1>(), 0);

    auto f = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"B", "D"}});
    auto f_expect =
        TAT::Tensor<double, TAT::NoSymmetry>{{"A", "C"}, {2, 2}}.set([i = 0, v = std::vector<double>{1, 3, 3, 13}]() mutable { return v[i++]; });
    ASSERT_FLOAT_EQ((f - f_expect).norm<-1>(), 0);
}

TEST(test_contract, no_symmetry_example_1) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>::contract(
        TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B", "C", "D"}, {1, 2, 3, 4}}.range(),
        TAT::Tensor<double, TAT::NoSymmetry>{{"E", "F", "G", "H"}, {3, 1, 2, 4}}.range(),
        {{"B", "G"}, {"D", "H"}}
    );
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "C", "E", "F"}, {1, 3, 3, 1}}.set(
        [i = 0, v = std::vector<double>{316, 796, 1276, 428, 1164, 1900, 540, 1532, 2524}]() mutable { return v[i++]; }
    );
    ASSERT_FLOAT_EQ((a - b).norm<-1>(), 0);
}

TEST(test_contract, u1_symmetry_example_0) {
    using Tensor = TAT::Tensor<double, TAT::U1Symmetry>;
    auto edge1 = TAT::Edge<TAT::U1Symmetry>{{-1, 2}, {0, 2}, {+1, 2}};
    auto edge2 = TAT::Edge<TAT::U1Symmetry>{{+1, 2}, {0, 2}, {-1, 2}};
    auto a = Tensor({"a", "b", "c", "d"}, {edge1, edge2, edge1, edge2}).range();
    auto b = Tensor({"e", "f", "g", "h"}, {edge1, edge2, edge1, edge2}).range();
    for (auto plan : std::vector<std::unordered_set<std::pair<std::string, std::string>>>{
             {{"a", "f"}, {"b", "e"}}, {{"a", "f"}, {"b", "g"}}, {{"a", "h"}, {"b", "e"}}, {{"a", "h"}, {"b", "g"}}, {{"a", "f"}, {"c", "h"}},
             {{"a", "h"}, {"c", "f"}}, {{"a", "f"}, {"d", "e"}}, {{"a", "f"}, {"d", "g"}}, {{"a", "h"}, {"d", "e"}}, {{"a", "h"}, {"d", "g"}},
             {{"c", "f"}, {"b", "e"}}, {{"c", "f"}, {"b", "g"}}, {{"c", "h"}, {"b", "e"}}, {{"c", "h"}, {"b", "g"}}, {{"b", "e"}, {"d", "g"}},
             {{"b", "g"}, {"d", "e"}}, {{"c", "f"}, {"d", "e"}}, {{"c", "f"}, {"d", "g"}}, {{"c", "h"}, {"d", "e"}}, {{"c", "h"}, {"d", "g"}},
         }) {
        auto c = a.contract(b, plan).clear_symmetry();
        auto d = a.clear_symmetry().contract(b.clear_symmetry(), plan);
        ASSERT_FLOAT_EQ((c - d).norm<-1>(), 0);
    }
}

TEST(test_contract, fermi_symmetry_example_0) {
    using FermiTensor = TAT::Tensor<double, TAT::FermiSymmetry>;
    auto fermi_edge1 = TAT::edge_segments_t<TAT::FermiSymmetry>({{-1, 2}, {0, 2}, {+1, 2}});
    auto fermi_edge2 = TAT::edge_segments_t<TAT::FermiSymmetry>({{+1, 2}, {0, 2}, {-1, 2}});
    auto fermi_a = FermiTensor({"a", "b", "c", "d"}, {{fermi_edge1, true}, fermi_edge2, {fermi_edge1, true}, {fermi_edge2, true}}).range();
    auto fermi_b = FermiTensor({"e", "f", "g", "h"}, {fermi_edge1, fermi_edge2, {fermi_edge1, true}, fermi_edge2}).range();
    auto fermi_c = fermi_a.contract(fermi_b, {{"d", "e"}, {"c", "f"}});
    auto fermi_d = fermi_b.contract(fermi_a, {{"e", "d"}, {"f", "c"}});
    ASSERT_FLOAT_EQ((fermi_c - fermi_d).norm<-1>(), 0);

    using U1Tensor = TAT::Tensor<double, TAT::U1Symmetry>;
    auto u1_edge1 = TAT::edge_segments_t<TAT::U1Symmetry>({{-1, 2}, {0, 2}, {+1, 2}});
    auto u1_edge2 = TAT::edge_segments_t<TAT::U1Symmetry>({{+1, 2}, {0, 2}, {-1, 2}});
    auto u1_a = U1Tensor({"a", "b", "c", "d"}, {u1_edge1, u1_edge2, u1_edge1, u1_edge2}).range();
    auto u1_b = U1Tensor({"e", "f", "g", "h"}, {u1_edge1, u1_edge2, u1_edge1, u1_edge2}).range();
    auto u1_c = u1_a.contract(u1_b, {{"d", "e"}, {"c", "f"}});

    ASSERT_THAT(fermi_a.storage(), ElementsAreArray(u1_a.storage()));
    ASSERT_THAT(fermi_b.storage(), ElementsAreArray(u1_b.storage()));
    ASSERT_THAT(fermi_c.storage(), ElementsAreArray(u1_c.storage()));
}

TEST(test_contract, contract_with_split_and_merge) {
    using Tensor = TAT::Tensor<double, TAT::FermiSymmetry>;
    auto edge1 = TAT::Edge<TAT::FermiSymmetry>({{-1, 2}, {0, 2}, {+1, 2}}, false);
    auto edge2 = TAT::Edge<TAT::FermiSymmetry>({{+1, 2}, {0, 2}, {-1, 2}}, true);
    auto a = Tensor({"a", "b", "c", "d"}, {edge1, edge2, edge1, edge2}).range();
    auto b = Tensor({"e", "f", "g", "h"}, {edge1, edge2, edge1, edge2}).range();
    auto c = a.contract(b, {{"a", "f"}, {"b", "g"}, {"c", "h"}});

    auto a_merged = a.merge_edge({{"m", {"b", "a"}}}, false);
    auto b_merged = b.merge_edge({{"m", {"g", "f"}}}, true);
    auto c_merged = a_merged.contract(b_merged, {{"m", "m"}, {"c", "h"}});

    ASSERT_FLOAT_EQ((c - c_merged).norm<-1>(), 0);
}

TEST(test_contract, contract_with_reverse_0) {
    auto a = TAT::Tensor<double, TAT::ParitySymmetry>({"i", "j"}, {{{{false, 2}, {true, 2}}, false}, {{{false, 2}, {true, 2}}, true}}).range();
    auto b = TAT::Tensor<double, TAT::ParitySymmetry>({"i", "j"}, {{{{false, 2}, {true, 2}}, false}, {{{false, 2}, {true, 2}}, true}})
                 .range()
                 .transpose({"j", "i"});
    auto c = a.contract(b, {{"j", "i"}});

    auto a_reversed = a.reverse_edge({"j"}, false);
    auto b_reversed = b.reverse_edge({"i"}, true);
    auto c_reversed = a_reversed.contract(b_reversed, {{"j", "i"}});

    ASSERT_FLOAT_EQ((c - c_reversed).norm<-1>(), 0);
}

TEST(test_contract, contract_with_reverse_1) {
    using Tensor = TAT::Tensor<double, TAT::FermiSymmetry>;
    auto edge1 = TAT::Edge<TAT::FermiSymmetry>({{-1, 2}, {0, 2}, {+1, 2}}, false);
    auto edge2 = TAT::Edge<TAT::FermiSymmetry>({{+1, 2}, {0, 2}, {-1, 2}}, true);
    auto a = Tensor({"a", "b", "c", "d"}, {edge1, edge2, edge1, edge2}).range();
    auto b = Tensor({"e", "f", "g", "h"}, {edge1, edge2, edge1, edge2}).range();
    auto c = a.contract(b, {{"a", "f"}, {"b", "g"}, {"c", "h"}});

    auto a_reversed = a.reverse_edge({"b", "a"}, false);
    auto b_reversed = b.reverse_edge({"g", "f"}, true);
    auto c_reversed = a_reversed.contract(b_reversed, {{"a", "f"}, {"b", "g"}, {"c", "h"}});

    ASSERT_FLOAT_EQ((c - c_reversed).norm<-1>(), 0);
}

TEST(test_contract, fuse) {
    TAT::Size fuse_d = 3;
    TAT::Size common_d = 4;
    auto a = TAT::Tensor<>({"A", "B", "C"}, {fuse_d, common_d, 5}).range();
    auto b = TAT::Tensor<>({"A", "B", "D"}, {fuse_d, common_d, 7}).range();
    auto c = TAT::Tensor<>::contract(a, b, {{"B", "B"}}, {"A"});
    for (TAT::Size i = 0; i < fuse_d; i++) {
        auto hat = TAT::Tensor<>({"A"}, {fuse_d}).zero();
        hat.storage()[i] = 1;
        auto a0 = a.contract(hat, {{"A", "A"}});
        auto b0 = b.contract(hat, {{"A", "A"}});
        auto c0 = c.contract(hat, {{"A", "A"}});
        ASSERT_FLOAT_EQ((a0.contract(b0, {{"B", "B"}}) - c0).norm<-1>(), 0);
    }
}

TEST(test_contract, corner_no_symmetry_0k) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {2, 0}}.range();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"C", "D"}, {0, 2}}.range();
    auto c = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"B", "C"}});
    ASSERT_NE(c.storage().size(), 0);
    ASSERT_FLOAT_EQ(c.norm<-1>(), 0);
}

TEST(test_contract, corner_z2_symmetry_0k) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{0, 2}}, {{0, 0}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{0, 0}}, {{0, 2}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"B", "C"}});
    ASSERT_NE(c.storage().size(), 0);
    ASSERT_FLOAT_EQ(c.norm<-1>(), 0);
}

TEST(test_contract, corner_z2_symmetry_not_match_missing_left) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{1, 2}}, {{0, 2}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{0, 2}}, {{0, 2}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"B", "C"}});
    ASSERT_FLOAT_EQ(c.norm<0>(), 0);
}

TEST(test_contract, corner_z2_symmetry_not_match_missing_right) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{0, 2}}, {{0, 2}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{0, 2}}, {{1, 2}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"B", "C"}});
    ASSERT_FLOAT_EQ(c.norm<0>(), 0);
}

TEST(test_contract, corner_z2_symmetry_not_match_missing_middle) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{0, 2}}, {{1, 2}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{1, 2}}, {{0, 2}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"B", "C"}});
    ASSERT_NE(c.storage().size(), 0);
    ASSERT_FLOAT_EQ(c.norm<-1>(), 0);
}
