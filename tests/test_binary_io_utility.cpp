#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_binary_io_utility, write) {
   using namespace TAT;
   std::ostringstream out;
   int i = 2333;
   out < i;
   std::string out_str = out.str();
   ASSERT_EQ(*reinterpret_cast<int*>(out_str.data()), i);
}

TEST(test_binary_io_utility, read) {
   using namespace TAT;
   int v = 2333;
   std::istringstream in(reinterpret_cast<char*>(&v));
   int i;
   in > i;
   ASSERT_EQ(v, i);
}

TEST(test_binary_io_utility, output_string_for_name) {
   using namespace TAT;

   std::stringstream out;
   write_string_for_name(out, "hello");
   std::string out_str = out.str();
   ASSERT_EQ(5, *reinterpret_cast<Size*>(out_str.data()));
   ASSERT_STREQ("hello", out_str.data() + sizeof(Size));
}

TEST(test_binary_io_utility, output_fastname_for_name) {
   using namespace TAT;

   std::stringstream out;
   write_fastname_for_name(out, FastName("hello"));
   std::string out_str = out.str();
   ASSERT_EQ(5, *reinterpret_cast<Size*>(out_str.data()));
   ASSERT_STREQ("hello", out_str.data() + sizeof(Size));
}

TEST(test_binary_io_utility, input_string_for_name) {
   using namespace TAT;

   std::string prefix(sizeof(Size), ' ');
   std::string src = prefix + "abcdefg";
   *reinterpret_cast<Size*>(src.data()) = 4;

   std::stringstream in(src);
   std::string n;
   read_string_for_name(in, n);
   ASSERT_STREQ(n.c_str(), "abcd");
}

TEST(test_binary_io_utility, input_fastname_for_name) {
   using namespace TAT;

   std::string prefix(sizeof(Size), ' ');
   std::string src = prefix + "abcdefg";
   *reinterpret_cast<Size*>(src.data()) = 4;

   std::stringstream in(src);
   FastName n;
   read_fastname_for_name(in, n);
   ASSERT_STREQ(static_cast<const std::string&>(n).c_str(), "abcd");
}

TEST(test_binary_io_utility, io_vector_pod) {
   using namespace TAT;

   std::stringstream ss;
   ss < std::vector<int>{1, 2, 3};
   std::vector<int> a;
   ss > a;
   ASSERT_THAT(a, ElementsAre(1, 2, 3));
}

TEST(test_binary_io_utility, io_vector_name) {
   using namespace TAT;

   std::stringstream ss;
   ss < std::vector<std::string>{"1", "2", "3"};
   std::vector<std::string> a;
   ss > a;
   ASSERT_THAT(a, ElementsAre(std::string("1"), std::string("2"), std::string("3")));
}

TEST(test_binary_io_utility, io_vector_nest) {
   using namespace TAT;

   std::stringstream ss;
   ss < std::vector<std::vector<int>>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
   std::vector<std::vector<int>> a;
   ss > a;
   ASSERT_THAT(a, ElementsAre(ElementsAre(1, 2, 3), ElementsAre(2, 3, 4), ElementsAre(3, 4, 5)));
}
