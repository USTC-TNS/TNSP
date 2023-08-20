#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_name, basic_usage) {
   auto n = TAT::DefaultName("left");
   ASSERT_STREQ(static_cast<const std::string&>(n).c_str(), "left");
   n = "right";
   ASSERT_STREQ(static_cast<const std::string&>(n).c_str(), "right");
}

TEST(test_name, hash_and_order) {
   auto n = TAT::DefaultName("left");
   auto n_set = std::unordered_set<TAT::DefaultName>();
   n_set.insert(n);
   n_set.insert(n);
   auto n_unordered_set = std::unordered_set<TAT::DefaultName>();
   n_unordered_set.insert(n);
   n_unordered_set.insert(n);
   auto n_set_unordered_set = std::unordered_set<std::pair<TAT::DefaultName, TAT::DefaultName>>();
   n_set_unordered_set.insert({n, n});
   n_set_unordered_set.insert({n, n});
}
