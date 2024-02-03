#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

#define edge_seg(...) \
    { __VA_ARGS__ }

TEST(test_create_symmetry_tensor, basic_usage) {
    // 1 1 0 : 3*1*3
    // 1 0 1 : 3*2*2
    // 0 1 1 : 1*1*2
    // 0 0 0 : 1*2*3
    auto a =
        (TAT::Tensor<double, TAT::Z2Symmetry>{{"Left", "Right", "Up"}, {edge_seg({1, 3}, {0, 1}), edge_seg({1, 1}, {0, 2}), edge_seg({1, 2}, {0, 3})}}
             .range_());
    ASSERT_EQ(a.names(0), "Left");
    ASSERT_EQ(a.names(1), "Right");
    ASSERT_EQ(a.names(2), "Up");
    ASSERT_THAT(a.names(), ElementsAre("Left", "Right", "Up"));
    ASSERT_EQ(a.rank_by_name("Left"), 0);
    ASSERT_EQ(a.rank_by_name("Right"), 1);
    ASSERT_EQ(a.rank_by_name("Up"), 2);
    ASSERT_EQ(TAT::ensure_cpu(a.storage()).size(), 1 * 2 * 3 + 1 * 1 * 2 + 3 * 2 * 2 + 3 * 1 * 3);
    ASSERT_EQ(&a.edges("Left"), &a.edges(0));
    ASSERT_EQ(&a.edges("Right"), &a.edges(1));
    ASSERT_EQ(&a.edges("Up"), &a.edges(2));

    ASSERT_THAT(a.blocks(std::vector<int>{0, 0, 1}).dimensions(), ElementsAre(3, 1, 3));
    ASSERT_THAT(a.const_blocks(std::vector<int>{1, 1, 1}).dimensions(), ElementsAre(1, 2, 3));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, int>{{"Left", 0}, {"Right", 1}, {"Up", 0}}).dimensions(), ElementsAre(3, 2, 2));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, int>{{"Left", 1}, {"Right", 0}, {"Up", 0}}).dimensions(), ElementsAre(1, 1, 2));
    ASSERT_THAT(a.blocks(std::vector<TAT::Z2Symmetry>{1, 1, 0}).dimensions(), ElementsAre(3, 1, 3));
    ASSERT_THAT(a.const_blocks(std::vector<TAT::Z2Symmetry>{0, 0, 0}).dimensions(), ElementsAre(1, 2, 3));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, TAT::Z2Symmetry>{{"Left", 1}, {"Right", 0}, {"Up", 1}}).dimensions(), ElementsAre(3, 2, 2));
    ASSERT_THAT(
        a.const_blocks(std::unordered_map<std::string, TAT::Z2Symmetry>{{"Left", 0}, {"Right", 1}, {"Up", 1}}).dimensions(),
        ElementsAre(1, 1, 2)
    );

    ASSERT_FLOAT_EQ(TAT::ensure_cpu(a.at(std::vector<std::pair<TAT::Z2Symmetry, TAT::Size>>{{1, 1}, {1, 0}, {0, 2}})), 5);
    ASSERT_FLOAT_EQ(
        TAT::ensure_cpu(
            a.at(std::unordered_map<std::string, std::pair<TAT::Z2Symmetry, TAT::Size>>{{"Left", {1, 2}}, {"Right", {0, 0}}, {"Up", {1, 1}}})
        ),
        3 * 1 * 3 + 9
    );
    ASSERT_FLOAT_EQ(
        TAT::ensure_cpu(a.const_at(std::vector<std::pair<TAT::Z2Symmetry, TAT::Size>>{{0, 0}, {1, 0}, {1, 1}})),
        3 * 1 * 3 + 3 * 2 * 2 + 1
    );
    ASSERT_FLOAT_EQ(
        TAT::ensure_cpu(
            a.const_at(std::unordered_map<std::string, std::pair<TAT::Z2Symmetry, TAT::Size>>{{"Left", {0, 0}}, {"Right", {0, 1}}, {"Up", {0, 2}}})
        ),
        3 * 1 * 3 + 3 * 2 * 2 + 1 * 1 * 2 + 5
    );
}

TEST(test_create_symmetry_tensor, when_0rank) {
    auto a = TAT::Tensor<double, TAT::U1Symmetry>{{}, {}}.range_(2333);
    ASSERT_THAT(a.names(), ElementsAre());
    ASSERT_THAT(TAT::ensure_cpu(a.storage()), ElementsAre(2333));

    ASSERT_THAT(a.blocks(std::vector<int>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.const_blocks(std::vector<int>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, int>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, int>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.blocks(std::vector<TAT::U1Symmetry>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.const_blocks(std::vector<TAT::U1Symmetry>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, TAT::U1Symmetry>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, TAT::U1Symmetry>{}).dimensions(), ElementsAre());

    ASSERT_FLOAT_EQ(TAT::ensure_cpu(a.at(std::vector<TAT::Size>{})), 2333);
    ASSERT_FLOAT_EQ(TAT::ensure_cpu(a.at(std::unordered_map<std::string, TAT::Size>{})), 2333);
    ASSERT_FLOAT_EQ(TAT::ensure_cpu(a.const_at(std::vector<TAT::Size>{})), 2333);
    ASSERT_FLOAT_EQ(TAT::ensure_cpu(a.const_at(std::unordered_map<std::string, TAT::Size>{})), 2333);
}

TEST(test_create_symmetry_tensor, when_0size) {
    using sym_t = TAT::U1Symmetry;
    auto a =
        (TAT::Tensor<double, sym_t>({"Left", "Right", "Up"}, {edge_seg({0, 0}), edge_seg({-1, 1}, {0, 2}, {1, 3}), edge_seg({-1, 2}, {0, 3}, {1, 1})})
             .zero_());
    ASSERT_EQ(a.names(0), "Left");
    ASSERT_EQ(a.names(1), "Right");
    ASSERT_EQ(a.names(2), "Up");
    ASSERT_THAT(a.names(), ElementsAre("Left", "Right", "Up"));
    ASSERT_EQ(a.rank_by_name("Left"), 0);
    ASSERT_EQ(a.rank_by_name("Right"), 1);
    ASSERT_EQ(a.rank_by_name("Up"), 2);
    ASSERT_THAT(TAT::ensure_cpu(a.storage()), ElementsAre());
    ASSERT_EQ(&a.edges("Left"), &a.edges(0));
    ASSERT_EQ(&a.edges("Right"), &a.edges(1));
    ASSERT_EQ(&a.edges("Up"), &a.edges(2));

    ASSERT_THAT(a.blocks(std::vector<int>{0, 1, 1}).dimensions(), ElementsAre(0, 2, 3));
    ASSERT_THAT(a.const_blocks(std::vector<int>{0, 0, 2}).dimensions(), ElementsAre(0, 1, 1));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, int>{{"Left", 0}, {"Right", 2}, {"Up", 0}}).dimensions(), ElementsAre(0, 3, 2));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, int>{{"Left", 0}, {"Right", 2}, {"Up", 0}}).dimensions(), ElementsAre(0, 3, 2));
    ASSERT_THAT(a.blocks(std::vector<sym_t>{0, 0, 0}).dimensions(), ElementsAre(0, 2, 3));
    ASSERT_THAT(a.const_blocks(std::vector<sym_t>{0, -1, +1}).dimensions(), ElementsAre(0, 1, 1));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, sym_t>{{"Left", 0}, {"Right", +1}, {"Up", -1}}).dimensions(), ElementsAre(0, 3, 2));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, sym_t>{{"Left", 0}, {"Right", +1}, {"Up", -1}}).dimensions(), ElementsAre(0, 3, 2));
}

TEST(test_create_symmetry_tensor, when_0block) {
    using sym_t = TAT::U1Symmetry;
    auto a = (TAT::Tensor<double, sym_t>(
                  {"Left", "Right", "Up"},
                  {std::vector<sym_t>(), edge_seg({-1, 1}, {0, 2}, {1, 3}), edge_seg({-1, 2}, {0, 3}, {1, 1})}
    )
                  .zero_());
    ASSERT_EQ(a.names(0), "Left");
    ASSERT_EQ(a.names(1), "Right");
    ASSERT_EQ(a.names(2), "Up");
    ASSERT_THAT(a.names(), ElementsAre("Left", "Right", "Up"));
    ASSERT_EQ(a.rank_by_name("Left"), 0);
    ASSERT_EQ(a.rank_by_name("Right"), 1);
    ASSERT_EQ(a.rank_by_name("Up"), 2);
    ASSERT_THAT(TAT::ensure_cpu(a.storage()), ElementsAre());
    ASSERT_EQ(&a.edges("Left"), &a.edges(0));
    ASSERT_EQ(&a.edges("Right"), &a.edges(1));
    ASSERT_EQ(&a.edges("Up"), &a.edges(2));
}

TEST(test_create_symmetry_tensor, conversion_scalar) {
    auto a = TAT::Tensor<double, TAT::U1Symmetry>(2333, {"i", "j"}, {-2, +2});
    ASSERT_EQ(a.names(0), "i");
    ASSERT_EQ(a.names(1), "j");
    ASSERT_THAT(a.names(), ElementsAre("i", "j"));
    ASSERT_EQ(a.rank_by_name("i"), 0);
    ASSERT_EQ(a.rank_by_name("j"), 1);
    ASSERT_THAT(TAT::ensure_cpu(a.storage()), ElementsAre(2333));
    ASSERT_EQ(&a.edges("i"), &a.edges(0));
    ASSERT_EQ(&a.edges("j"), &a.edges(1));

    ASSERT_THAT(a.blocks(std::vector<int>{0, 0}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.const_blocks(std::vector<int>{0, 0}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, int>{{"i", 0}, {"j", 0}}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, int>{{"i", 0}, {"j", 0}}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.blocks(std::vector<TAT::U1Symmetry>{-2, +2}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.const_blocks(std::vector<TAT::U1Symmetry>{-2, +2}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, TAT::U1Symmetry>{{"i", -2}, {"j", +2}}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, TAT::U1Symmetry>{{"i", -2}, {"j", +2}}).dimensions(), ElementsAre(1, 1));

    ASSERT_FLOAT_EQ(TAT::ensure_cpu(a.at()), 2333);
    ASSERT_FLOAT_EQ(TAT::ensure_cpu(a.const_at()), 2333);
    ASSERT_FLOAT_EQ(double(a), 2333);
}

TEST(test_create_symmetry_tensor, conversion_scalar_empty) {
    auto a = TAT::Tensor<double, TAT::U1Symmetry>({"i"}, {{{+1, 2333}}}).range_(2333);
    ASSERT_FLOAT_EQ(double(a), 0);
}
