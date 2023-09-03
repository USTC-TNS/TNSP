#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_special_stringstring, special_istream) {
   using namespace TAT;

   detail::basic_instringstream<char> in("1\n2\n3\n");
   int a, b, c;
   in >> a >> b >> c;
   EXPECT_EQ(a, 1);
   EXPECT_EQ(b, 2);
   EXPECT_EQ(c, 3);
}

TEST(test_special_stringstring, special_ostream) {
   using namespace TAT;

   detail::basic_outstringstream<char> out;
   int a = 1, b = 2, c = 3;
   out << a << b << c;
   auto string = std::move(out).str();
   EXPECT_EQ(string, "123");
   EXPECT_STREQ(std::move(out).str().c_str(), "");
}

TEST(test_special_stringstring, special_ostream_overflow) {
   // default size is 8
   using namespace TAT;

   detail::basic_outstringstream<char> out;
   int a = 123, b = 234, c = 345;
   out << a << b << c;
   auto string = std::move(out).str();
   EXPECT_EQ(string, "123234345");
   EXPECT_STREQ(std::move(out).str().c_str(), "");
}
