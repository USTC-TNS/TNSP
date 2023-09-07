#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_transform_map_and_set, transform) {
    auto t = TAT::Tensor<>({"i", "j"}, {2, 3}).range(); // 0, 1, 2, 3, 4, 5
    ASSERT_THAT(t.storage(), ElementsAre(0, 1, 2, 3, 4, 5));
    t.transform([](auto x) { return x + 1; });
    ASSERT_THAT(t.storage(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(test_transform_map_and_set, transform_rvalue) {
    auto t = TAT::Tensor<>({"i", "j"}, {2, 3}).range().transform([](auto x) { return x + 1; });
    ASSERT_THAT(t.storage(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(test_transform_map_and_set, zip_transform) {
    auto t = TAT::Tensor<>({"i", "j"}, {2, 3}).range(0); // 0, 1, 2, 3, 4, 5
    auto s = TAT::Tensor<>({"i", "j"}, {2, 3}).range(1); // 1, 2, 3, 4, 5, 6
    ASSERT_THAT(t.storage(), ElementsAre(0, 1, 2, 3, 4, 5));
    ASSERT_THAT(s.storage(), ElementsAre(1, 2, 3, 4, 5, 6));
    t.zip_transform(s, [](auto it, auto is) { return is - it; });
    ASSERT_THAT(t.storage(), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST(test_transform_map_and_set, zip_transform_rvalue) {
    auto s = TAT::Tensor<>({"i", "j"}, {2, 3}).range(1); // 1, 2, 3, 4, 5, 6
    ASSERT_THAT(s.storage(), ElementsAre(1, 2, 3, 4, 5, 6));
    auto t = TAT::Tensor<>({"i", "j"}, {2, 3}).range(0).zip_transform(s, [](auto it, auto is) { return is - it; });
    ASSERT_THAT(t.storage(), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST(test_transform_map_and_set, map) {
    const auto t = TAT::Tensor<>({"i", "j"}, {2, 3}).range(); // 0, 1, 2, 3, 4, 5
    ASSERT_THAT(t.storage(), ElementsAre(0, 1, 2, 3, 4, 5));
    const auto z = t.map([](auto x) { return x + 1; });
    ASSERT_THAT(z.storage(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(test_transform_map_and_set, zip_map) {
    const auto t = TAT::Tensor<>({"i", "j"}, {2, 3}).range(0); // 0, 1, 2, 3, 4, 5
    const auto s = TAT::Tensor<>({"i", "j"}, {2, 3}).range(1); // 1, 2, 3, 4, 5, 6
    ASSERT_THAT(t.storage(), ElementsAre(0, 1, 2, 3, 4, 5));
    ASSERT_THAT(s.storage(), ElementsAre(1, 2, 3, 4, 5, 6));
    const auto z = t.zip_map(s, [](auto it, auto is) { return is - it; });
    ASSERT_THAT(z.storage(), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST(test_transform_map_and_set, copy) {
    const auto t = TAT::Tensor<>({"i", "j"}, {2, 3}).range(0); // 0, 1, 2, 3, 4, 5
    const auto s = t.copy();
    ASSERT_THAT(t.storage(), ElementsAre(0, 1, 2, 3, 4, 5));
    ASSERT_THAT(s.storage(), ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(test_transform_map_and_set, set) {
    auto t = TAT::Tensor<>({"i", "j"}, {2, 3});
    t.set([i = 0]() mutable {
        double v[] = {6, 2, 8, 3, 7, 1};
        return v[i++];
    });
    ASSERT_THAT(t.storage(), ElementsAre(6, 2, 8, 3, 7, 1));
}

TEST(test_transform_map_and_set, set_rvalue) {
    const auto t = TAT::Tensor<>({"i", "j"}, {2, 3}).set([i = 0]() mutable {
        double v[] = {6, 2, 8, 3, 7, 1};
        return v[i++];
    });
    ASSERT_THAT(t.storage(), ElementsAre(6, 2, 8, 3, 7, 1));
}

TEST(test_transform_map_and_set, zero) {
    const auto t = TAT::Tensor<>({"i", "j"}, {2, 3}).zero();
    ASSERT_THAT(t.storage(), ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST(test_transform_map_and_set, range) {
    const auto t = TAT::Tensor<>({"i", "j"}, {2, 3}).range(3, 2);
    ASSERT_THAT(t.storage(), ElementsAre(3, 5, 7, 9, 11, 13));
}
