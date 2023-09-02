#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_text_io_edge, output_symmetry) {
   std::stringstream s1;
   s1 << TAT::Symmetry<>();
   ASSERT_STREQ(s1.str().c_str(), "");

   std::stringstream s2;
   s2 << TAT::Symmetry<TAT::bose<int>>(233);
   ASSERT_STREQ(s2.str().c_str(), "233");

   std::stringstream s3;
   s3 << TAT::Symmetry<int, int>(233, 666);
   ASSERT_STREQ(s3.str().c_str(), "(233,666)");
}

TEST(test_text_io_edge, input_symmetry) {
   std::stringstream s1("");
   TAT::Symmetry<> sym1;
   s1 >> sym1;

   std::stringstream s2("233");
   TAT::Symmetry<TAT::bose<int>> sym2;
   s2 >> sym2;
   ASSERT_EQ(std::get<0>(sym2), 233);

   std::stringstream s3("(233,666)");
   TAT::Symmetry<int, int> sym3;
   s3 >> sym3;
   ASSERT_EQ(std::get<0>(sym3), 233);
   ASSERT_EQ(std::get<1>(sym3), 666);
}
