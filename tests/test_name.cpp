#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_name, basic_usage) {
   auto n = TAT::DefaultName("left");
   ASSERT_EQ(static_cast<const std::string&>(n), "left");
   n = "right";
   ASSERT_EQ(static_cast<const std::string&>(n), "right");
}

TEST(test_name, ordered_map) {
   auto n = TAT::DefaultName("left");
   auto s = std::unordered_set<TAT::DefaultName>();
   s.insert(n);
   s.insert(n);
   ASSERT_EQ(s.size(), 1);
}

TEST(test_name, unordered_map) {
   auto n = TAT::DefaultName("left");
   auto s = std::unordered_set<TAT::DefaultName>();
   s.insert(n);
   s.insert(n);
   ASSERT_EQ(s.size(), 1);
}

TEST(test_name, unordered_pair_map) {
   auto n = TAT::DefaultName("left");
   auto s = std::unordered_set<std::pair<TAT::DefaultName, TAT::DefaultName>>();
   s.insert({n, n});
   s.insert({n, n});
   ASSERT_EQ(s.size(), 1);
}
