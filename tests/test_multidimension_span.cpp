#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

TEST(test_multidimension_span, create_object_and_basic_attribute) {
    auto data = std::vector<int>(24);
    auto span = TAT::mdspan<int>(data.data(), {2, 3, 4});
    ASSERT_EQ(span.rank(), 3);
    ASSERT_EQ(span.size(), 24);
    ASSERT_EQ(span.dimensions(0), 2);
    ASSERT_EQ(span.dimensions(1), 3);
    ASSERT_EQ(span.dimensions(2), 4);
    ASSERT_EQ(span.leadings(2), 1);
    ASSERT_EQ(span.leadings(1), 4);
    ASSERT_EQ(span.leadings(0), 12);
    ASSERT_EQ(span.dimensions()[0], 2);
    ASSERT_EQ(span.dimensions()[1], 3);
    ASSERT_EQ(span.dimensions()[2], 4);
    ASSERT_EQ(span.leadings()[2], 1);
    ASSERT_EQ(span.leadings()[1], 4);
    ASSERT_EQ(span.leadings()[0], 12);
    ASSERT_EQ(span.data(), data.data());
}

TEST(test_multidimension_span, set_data_later) {
    auto span = TAT::mdspan<int>(nullptr, {2, 3, 4});
    auto data = std::vector<int>(24);
    span.set_data(data.data());
    ASSERT_EQ(span.data(), data.data());
    const auto const_span = TAT::mdspan<int>(data.data(), {2, 3, 4});
    ASSERT_EQ(const_span.data(), data.data());
}

TEST(test_multidimension_span, get_set_item) {
    auto data = std::vector<int>(24);
    auto span = TAT::mdspan<int>(data.data(), {2, 3, 4});
    span.at({0, 0, 0}) = 2333;
    ASSERT_EQ(data[0], 2333);
    data[13] = 666;
    ASSERT_EQ(span.at({1, 0, 1}), 666);
}

TEST(test_multidimension_span, get_item_const) {
    auto data = std::vector<int>(24);
    const auto span = TAT::mdspan<int>(data.data(), {2, 3, 4});
    data[13] = 666;
    ASSERT_EQ(span.at({1, 0, 1}), 666);
}

TEST(test_multidimension_span, iterate) {
    auto data = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
    const auto span = TAT::mdspan<int>(data.data(), {2, 2, 2});
    auto v = 0;
    for (auto i = span.begin(); i != span.end(); ++i) {
        ASSERT_EQ(*i, v++);
    }
}

TEST(test_multidimension_span, iterate_and_arrow_operator) {
    struct V {
        int v;
        V(int v) : v(v) { }
    };
    auto data = std::vector<V>{0, 1, 2, 3, 4, 5, 6, 7};
    auto span = TAT::mdspan<V>(data.data(), {2, 2, 2});
    auto v = 0;
    for (auto i = span.begin(); i != span.end(); ++i) {
        ASSERT_EQ(i->v, v++);
    }
}

TEST(test_multidimension_span, iterate_on_empty) {
    const auto const_span = TAT::mdspan<int>(nullptr, {2, 0, 2});
    for (auto i = const_span.begin(); i != const_span.end(); ++i) {
        FAIL();
    }
    auto span = TAT::mdspan<int>(nullptr, {2, 0, 2});
    for (auto i = span.begin(); i != span.end(); ++i) {
        FAIL();
    }
}

TEST(test_multidimension_span, transpose) {
    auto data = std::vector<int>(24);
    auto span = TAT::mdspan<int>(data.data(), {2, 3, 4});
    auto transposed_span = span.transpose({2, 1, 0});
    ASSERT_EQ(&span.at({0, 1, 2}), &transposed_span.at({2, 1, 0}));
}

TEST(test_multidimension_span, transpose_const) {
    auto data = std::vector<int>(24);
    const auto span = TAT::mdspan<int>(data.data(), {2, 3, 4});
    auto transposed_span = span.transpose({2, 1, 0});
    ASSERT_EQ(&span.at({0, 1, 2}), &transposed_span.at({2, 1, 0}));
}

TEST(test_multidimension_span, transform_with_transpose) {
    auto s_data = std::vector<int>(24);
    auto s_span = TAT::mdspan<int>(s_data.data(), {2, 3, 4});
    s_span.at({1, 2, 3}) = 666;
    auto d_data = std::vector<int>(24);
    auto d_span = TAT::mdspan<int>(s_data.data(), {4, 3, 2});
    // To copy data, transpose them into same shape
    auto transposed_s_span = s_span.transpose({2, 1, 0});
    TAT::mdspan_transform(transposed_s_span, d_span, [](auto x) { return x; });
    ASSERT_EQ(d_span.at({3, 2, 1}), 666);
}

TEST(test_multidimension_span, transform_with_transpose_throw) {
    if constexpr (TAT::debug_mode) {
        EXPECT_ANY_THROW({
            auto s_data = std::vector<int>(24);
            auto s_span = TAT::mdspan<int>(s_data.data(), {2, 3, 4});
            auto d_data = std::vector<int>(24);
            auto d_span = TAT::mdspan<int>(s_data.data(), {4, 3, 2});
            // shape not same, should throw
            auto transposed_s_span = s_span.transpose({2, 0, 1});
            TAT::mdspan_transform(transposed_s_span, d_span, [](auto x) { return x; });
        });
    }
}

TEST(test_multidimension_span, transform_with_transpose_one_squash_line) {
    auto s_data = std::vector<int>(192);
    auto s_span = TAT::mdspan<int>(s_data.data(), {2, 1, 3, 4, 4, 2});
    s_span.at({1, 0, 2, 3, 2, 1}) = 777;
    auto d_data = std::vector<int>(192);
    auto d_span = TAT::mdspan<int>(s_data.data(), {4, 4, 3, 1, 2, 2});
    // To copy data, transpose them into same shape
    auto transposed_s_span = s_span.transpose({3, 4, 2, 1, 0, 5});
    TAT::mdspan_transform(transposed_s_span, d_span, [](auto x) { return x; });
    ASSERT_EQ(d_span.at({3, 2, 2, 0, 1, 1}), 777);
}

TEST(test_multidimension_span, transform_with_transpose_rank_0) {
    auto s_data = std::vector<int>(1);
    auto s_span = TAT::mdspan<int>(s_data.data(), {});
    s_span.at({}) = 888;
    auto d_data = std::vector<int>(1);
    auto d_span = TAT::mdspan<int>(s_data.data(), {});
    TAT::mdspan_transform(s_span, d_span, [](auto x) { return x; });
    ASSERT_EQ(d_span.at({}), 888);
}

TEST(test_multidimension_span, transform_with_transpose_size_0) {
    auto s_span = TAT::mdspan<int>(nullptr, {2, 3, 2, 0});
    auto d_span = TAT::mdspan<int>(nullptr, {2, 3, 2, 0});
    TAT::mdspan_transform(s_span, d_span, [](auto x) { return x; });
}

TEST(test_multidimension_span, matrix_transpose) {
    auto s_data = std::vector<int>{0, 1, 2, 3, 4, 5};
    auto d_data = std::vector<int>(6);
    TAT::matrix_transpose(2, 3, s_data.data(), d_data.data());
    ASSERT_THAT(d_data, testing::ElementsAre(0, 3, 1, 4, 2, 5));
}
