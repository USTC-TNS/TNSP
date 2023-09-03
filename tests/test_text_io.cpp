#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_text_io, basic_usage_nosymmetry) {
   using namespace TAT;
   Tensor<> a;
   char b;
   auto input = "{names:[L.*&^eft,R--..ight],edges:[3,4],blocks:[0,1,2,3,4,5,6,7,8,9,10,11]}?";
   std::stringstream(input) >> a >> b;
   ASSERT_EQ(b, '?');
   std::stringstream out;
   out << a;
   auto expect = "{\x1B[32mnames\x1B[0m:[L.*&^eft,R--..ight],\x1B[32medges\x1B[0m:[3,4],\x1B[32mblocks\x1B[0m:[0,1,2,3,4,5,6,7,8,9,10,11]}";
   ASSERT_EQ(out.str(), expect);
}

TEST(test_text_io, basic_usage_u1symmetry) {
   using namespace TAT;
   Tensor<double, U1Symmetry> a;
   char b;
   auto input = "{names:[A,B,C,D],edges:[{-2:1,-1:1,0:1},{0:1,1:2},{0:2,1:2},{-2:2,-1:1,0:2}],blocks:{[-2,1,1,0]:[0,1,2,3,4,5,6,7],[-1,0,1,0]:[8,9,1"
                "0,11],[-1,1,0,0]:[12,13,14,15,16,17,18,19],[-1,1,1,-1]:[20,21,22,23],[0,0,0,0]:[24,25,26,27],[0,0,1,-1]:[28,29],[0,1,0,-1]:[30,31,3"
                "2,33],[0,1,1,-2]:[34,35,36,37,38,39,40,41]}}?";
   std::stringstream(input) >> a >> b;
   ASSERT_EQ(b, '?');
   std::stringstream out;
   out << a;
   auto expect = "{\x1B[32mnames\x1B[0m:[A,B,C,D],\x1B[32medges\x1B[0m:[{-2:1,-1:1,0:1},{0:1,1:2},{0:2,1:2},{-2:2,-1:1,0:2}],\x1B[32mblocks\x1B[0m:{"
                 "\x1B[33m[-2,1,1,0]\x1B[0m:[0,1,2,3,4,5,6,7],\x1B[33m[-1,0,1,0]\x1B[0m:[8,9,10,11],\x1B[33m[-1,1,0,0]\x1B[0m:[12,13,14,15,16,17,18,"
                 "19],\x1B[33m[-1,1,1,-1]\x1B[0m:[20,21,22,23],\x1B[33m[0,0,0,0]\x1B[0m:[24,25,26,27],\x1B[33m[0,0,1,-1]\x1B[0m:[28,29],\x1B[33m[0,1"
                 ",0,-1]\x1B[0m:[30,31,32,33],\x1B[33m[0,1,1,-2]\x1B[0m:[34,35,36,37,38,39,40,41]}}";
   ASSERT_EQ(out.str(), expect);
}

TEST(test_text_io, basic_usage_nosymmetry_shape) {
   using namespace TAT;
   Tensor<> a;
   char b;
   auto input = "{names:[L.*&^eft,R--..ight],edges:[3,4],blocks:[0,1,2,3,4,5,6,7,8,9,10,11]}?";
   std::stringstream(input) >> a >> b;
   ASSERT_EQ(b, '?');
   std::stringstream out;
   out << a.shape();
   auto expect = "{\x1B[32mnames\x1B[0m:[L.*&^eft,R--..ight],\x1B[32medges\x1B[0m:[3,4]}";
   ASSERT_EQ(out.str(), expect);
}

TEST(test_text_io, basic_usage_nosymmetry_show) {
   using namespace TAT;
   Tensor<> a;
   char b;
   auto input = "{names:[L.*&^eft,R--..ight],edges:[3,4],blocks:[0,1,2,3,4,5,6,7,8,9,10,11]}?";
   std::stringstream(input) >> a >> b;
   ASSERT_EQ(b, '?');
   auto expect = "{\x1B[32mnames\x1B[0m:[L.*&^eft,R--..ight],\x1B[32medges\x1B[0m:[3,4],\x1B[32mblocks\x1B[0m:[0,1,2,3,4,5,6,7,8,9,10,11]}";
   ASSERT_EQ(a.show(), expect);
}

TEST(test_text_io, basic_usage_nosymmetry_complex) {
   using namespace TAT;
   Tensor<std::complex<float>> a;
   char b;
   auto input = "{names:[L.*&^eft,R--..ight],edges:[3,4],blocks:[0,1,2,3,4,5,6,7,8,9,10,11]}?";
   std::stringstream(input) >> a >> b;
   ASSERT_EQ(b, '?');
   std::stringstream out;
   out << a;
   auto expect = "{\x1B[32mnames\x1B[0m:[L.*&^eft,R--..ight],\x1B[32medges\x1B[0m:[3,4],\x1B[32mblocks\x1B[0m:[0,1,2,3,4,5,6,7,8,9,10,11]}";
   ASSERT_EQ(out.str(), expect);
}

TEST(test_text_io, basic_usage_u1symmetry_complex) {
   using namespace TAT;
   Tensor<std::complex<double>, U1Symmetry> a;
   char b;
   auto input = "{names:[A,B,C,D],edges:[{-2:1,-1:1,0:1},{0:1,1:2},{0:2,1:2},{-2:2,-1:1,0:2}],blocks:{[-2,1,1,0]:[0,1,2,3,4,5,6,7],[-1,0,1,0]:[8,9,1"
                "0,11],[-1,1,0,0]:[12,13,14,15,16,17,18,19],[-1,1,1,-1]:[20,21,22,23],[0,0,0,0]:[24,25,26,27],[0,0,1,-1]:[28,29],[0,1,0,-1]:[30,31,3"
                "2,33],[0,1,1,-2]:[34,35,36,37,38,39,40,41]}}?";
   std::stringstream(input) >> a >> b;
   ASSERT_EQ(b, '?');
   std::stringstream out;
   out << a;
   auto expect = "{\x1B[32mnames\x1B[0m:[A,B,C,D],\x1B[32medges\x1B[0m:[{-2:1,-1:1,0:1},{0:1,1:2},{0:2,1:2},{-2:2,-1:1,0:2}],\x1B[32mblocks\x1B[0m:{"
                 "\x1B[33m[-2,1,1,0]\x1B[0m:[0,1,2,3,4,5,6,7],\x1B[33m[-1,0,1,0]\x1B[0m:[8,9,10,11],\x1B[33m[-1,1,0,0]\x1B[0m:[12,13,14,15,16,17,18,"
                 "19],\x1B[33m[-1,1,1,-1]\x1B[0m:[20,21,22,23],\x1B[33m[0,0,0,0]\x1B[0m:[24,25,26,27],\x1B[33m[0,0,1,-1]\x1B[0m:[28,29],\x1B[33m[0,1"
                 ",0,-1]\x1B[0m:[30,31,32,33],\x1B[33m[0,1,1,-2]\x1B[0m:[34,35,36,37,38,39,40,41]}}";
   ASSERT_EQ(out.str(), expect);
}

TEST(test_text_io, basic_usage_u1symmetry_empty_block) {
   using namespace TAT;
   Tensor<std::complex<double>, U1Symmetry> a;
   char b;
   auto input = "{names:[i,j],edges:[{0:0,1:1},{0:2,-1:3}],blocks:{[0,0]:[],[1,-1]:[0,1,2]}}?";
   std::stringstream(input) >> a >> b;
   ASSERT_EQ(b, '?');
   std::stringstream out;
   out << a;
   auto expect = "{\x1B[32mnames\x1B[0m:[i,j],\x1B[32medges\x1B[0m:[{0:0,1:1},{0:2,-1:3}],\x1B[32mblocks\x1B[0m:{\x1B[33m[0,0]\x1B[0m:[],\x1B[33m[1,"
                 "-1]\x1B[0m:[0,1,2]}}";
   ASSERT_EQ(out.str(), expect);
}

TEST(test_text_io, basic_usage_u1symmetry_no_block) {
   using namespace TAT;
   Tensor<std::complex<double>, U1Symmetry> a;
   char b;
   auto input = "{names:[i,j],edges:[{0:2},{-1:3}],blocks:{}}?";
   std::stringstream(input) >> a >> b;
   ASSERT_EQ(b, '?');
   std::stringstream out;
   out << a;
   auto expect = "{\x1B[32mnames\x1B[0m:[i,j],\x1B[32medges\x1B[0m:[{0:2},{-1:3}],\x1B[32mblocks\x1B[0m:{}}";
   ASSERT_EQ(out.str(), expect);
}
