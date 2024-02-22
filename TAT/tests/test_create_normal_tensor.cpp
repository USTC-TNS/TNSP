#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_create_normal_tensor, basic_usage) {
    auto a = TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range_();
    ASSERT_EQ(a.names(0), "Left");
    ASSERT_EQ(a.names(1), "Right");
    ASSERT_THAT(a.names(), ElementsAre("Left", "Right"));
    ASSERT_EQ(a.rank_by_name("Left"), 0);
    ASSERT_EQ(a.rank_by_name("Right"), 1);
    ASSERT_THAT(a.storage(), ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11));
    ASSERT_EQ(&a.edges("Left"), &a.edges(0));
    ASSERT_EQ(&a.edges("Right"), &a.edges(1));

    ASSERT_THAT(a.blocks(std::vector<int>{0, 0}).dimensions(), ElementsAre(3, 4));
    ASSERT_THAT(a.const_blocks(std::vector<int>{0, 0}).dimensions(), ElementsAre(3, 4));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, int>{{"Left", 0}, {"Right", 0}}).dimensions(), ElementsAre(3, 4));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, int>{{"Left", 0}, {"Right", 0}}).dimensions(), ElementsAre(3, 4));
    ASSERT_THAT(a.blocks(std::vector<TAT::NoSymmetry>{{}, {}}).dimensions(), ElementsAre(3, 4));
    ASSERT_THAT(a.const_blocks(std::vector<TAT::NoSymmetry>{{}, {}}).dimensions(), ElementsAre(3, 4));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, TAT::NoSymmetry>{{"Left", {}}, {"Right", {}}}).dimensions(), ElementsAre(3, 4));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, TAT::NoSymmetry>{{"Left", {}}, {"Right", {}}}).dimensions(), ElementsAre(3, 4));

    ASSERT_FLOAT_EQ(a.at(std::vector<TAT::Size>{1, 2}).real(), 6);
    ASSERT_FLOAT_EQ(a.at(std::unordered_map<std::string, TAT::Size>{{"Right", 1}, {"Left", 2}}).real(), 9);
    ASSERT_FLOAT_EQ(a.const_at(std::vector<TAT::Size>{1, 2}).real(), 6);
    ASSERT_FLOAT_EQ(a.const_at(std::unordered_map<std::string, TAT::Size>{{"Right", 1}, {"Left", 2}}).real(), 9);
}

TEST(test_create_normal_tensor, when_0rank) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{}, {}}.range_();
    ASSERT_THAT(a.names(), ElementsAre());
    ASSERT_THAT(a.storage(), ElementsAre(0));

    ASSERT_THAT(a.blocks(std::vector<int>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.const_blocks(std::vector<int>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, int>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, int>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.blocks(std::vector<TAT::NoSymmetry>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.const_blocks(std::vector<TAT::NoSymmetry>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, TAT::NoSymmetry>{}).dimensions(), ElementsAre());
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, TAT::NoSymmetry>{}).dimensions(), ElementsAre());

    ASSERT_FLOAT_EQ(a.at(std::vector<TAT::Size>{}), 0);
    ASSERT_FLOAT_EQ(a.at(std::unordered_map<std::string, TAT::Size>{}), 0);
    ASSERT_FLOAT_EQ(a.const_at(std::vector<TAT::Size>{}), 0);
    ASSERT_FLOAT_EQ(a.const_at(std::unordered_map<std::string, TAT::Size>{}), 0);
}

TEST(test_create_normal_tensor, when_0size) {
    auto a = TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {0, 4}}.range_();
    ASSERT_EQ(a.names(0), "Left");
    ASSERT_EQ(a.names(1), "Right");
    ASSERT_THAT(a.names(), ElementsAre("Left", "Right"));
    ASSERT_EQ(a.rank_by_name("Left"), 0);
    ASSERT_EQ(a.rank_by_name("Right"), 1);
    ASSERT_THAT(a.storage(), ElementsAre());
    ASSERT_EQ(&a.edges("Left"), &a.edges(0));
    ASSERT_EQ(&a.edges("Right"), &a.edges(1));

    ASSERT_THAT(a.blocks(std::vector<int>{0, 0}).dimensions(), ElementsAre(0, 4));
    ASSERT_THAT(a.const_blocks(std::vector<int>{0, 0}).dimensions(), ElementsAre(0, 4));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, int>{{"Left", 0}, {"Right", 0}}).dimensions(), ElementsAre(0, 4));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, int>{{"Left", 0}, {"Right", 0}}).dimensions(), ElementsAre(0, 4));
    ASSERT_THAT(a.blocks(std::vector<TAT::NoSymmetry>{{}, {}}).dimensions(), ElementsAre(0, 4));
    ASSERT_THAT(a.const_blocks(std::vector<TAT::NoSymmetry>{{}, {}}).dimensions(), ElementsAre(0, 4));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, TAT::NoSymmetry>{{"Left", {}}, {"Right", {}}}).dimensions(), ElementsAre(0, 4));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, TAT::NoSymmetry>{{"Left", {}}, {"Right", {}}}).dimensions(), ElementsAre(0, 4));
}

TEST(test_create_normal_tensor, conversion_scalar) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>(2333, {"i", "j"});
    ASSERT_EQ(a.names(0), "i");
    ASSERT_EQ(a.names(1), "j");
    ASSERT_THAT(a.names(), ElementsAre("i", "j"));
    ASSERT_EQ(a.rank_by_name("i"), 0);
    ASSERT_EQ(a.rank_by_name("j"), 1);
    ASSERT_THAT(a.storage(), ElementsAre(2333));
    ASSERT_EQ(&a.edges("i"), &a.edges(0));
    ASSERT_EQ(&a.edges("j"), &a.edges(1));

    ASSERT_THAT(a.blocks(std::vector<int>{0, 0}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.const_blocks(std::vector<int>{0, 0}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, int>{{"i", 0}, {"j", 0}}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, int>{{"i", 0}, {"j", 0}}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.blocks(std::vector<TAT::NoSymmetry>{{}, {}}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.const_blocks(std::vector<TAT::NoSymmetry>{{}, {}}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.blocks(std::unordered_map<std::string, TAT::NoSymmetry>{{"i", {}}, {"j", {}}}).dimensions(), ElementsAre(1, 1));
    ASSERT_THAT(a.const_blocks(std::unordered_map<std::string, TAT::NoSymmetry>{{"i", {}}, {"j", {}}}).dimensions(), ElementsAre(1, 1));

    ASSERT_FLOAT_EQ(a.at(), 2333);
    ASSERT_FLOAT_EQ(a.const_at(), 2333);
    ASSERT_FLOAT_EQ(double(a), 2333);
}
