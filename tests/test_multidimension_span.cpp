#include <TAT/TAT.hpp>
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

TEST(test_multidimension_span, iterate_on_empty) {
   const auto span = TAT::mdspan<int>(nullptr, {2, 0, 2});
   for (auto i = span.begin(); i != span.end(); ++i) {
      FAIL();
   }
}
