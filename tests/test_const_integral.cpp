#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_const_integral, basic_usage_dynamic) {
   const auto value = TAT::to_const_integral_0_to_16(17);
   std::visit(
         [](const auto& v) {
            ASSERT_EQ(v.value(), 17);
            ASSERT_TRUE(v.is_dynamic);
         },
         value);
}

TEST(test_const_integral, basic_usage_static) {
   const auto value = TAT::to_const_integral_0_to_16(13);
   std::visit(
         [](const auto& v) {
            ASSERT_EQ(v.value(), 13);
            ASSERT_TRUE(v.is_static);
         },
         value);
}

TEST(test_const_integral, return_value) {
   const auto value = TAT::to_const_integral_0_to_16(13);
   auto v = std::visit(
         [](const auto& v) -> int {
            return v.value();
         },
         value);
   ASSERT_EQ(v, 13);
}
