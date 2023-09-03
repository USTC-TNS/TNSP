#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_text_io_utility, output_complex) {
   using namespace TAT;

   std::stringstream out;
   detail::print_complex(out, std::complex<float>(1, 2)) << "\n";
   detail::print_complex(out, std::complex<float>(1, -2)) << "\n";
   detail::print_complex(out, std::complex<float>(-1, 2)) << "\n";
   detail::print_complex(out, std::complex<float>(-1, -2)) << "\n";
   detail::print_complex(out, std::complex<float>(1, 0)) << "\n";
   detail::print_complex(out, std::complex<float>(0, 1)) << "\n";
   detail::print_complex(out, std::complex<float>(-1, 0)) << "\n";
   detail::print_complex(out, std::complex<float>(0, -1)) << "\n";
   detail::print_complex(out, std::complex<float>(0, 0)) << "\n";
   ASSERT_EQ(out.str(), "1+2i\n1-2i\n-1+2i\n-1-2i\n1\n1i\n-1\n-1i\n0\n");

   ASSERT_EQ(static_cast<std::stringstream&&>(detail::print_complex(std::stringstream(), std::complex<float>(0, 0))).str(), "0");
}

TEST(test_text_io_utility, input_complex) {
   using namespace TAT;

   std::stringstream in("1+2i\n1-2i\n-1+2i\n-1-2i\n1\n1i\n-1\n-1i\n0\n");
   std::complex<float> z[9];
   for (auto i = 0; i < 9; i++) {
      detail::scan_complex(in, z[i]);
   }
   ASSERT_FLOAT_EQ(z[0].real(), 1);
   ASSERT_FLOAT_EQ(z[0].imag(), 2);
   ASSERT_FLOAT_EQ(z[1].real(), 1);
   ASSERT_FLOAT_EQ(z[1].imag(), -2);
   ASSERT_FLOAT_EQ(z[2].real(), -1);
   ASSERT_FLOAT_EQ(z[2].imag(), 2);
   ASSERT_FLOAT_EQ(z[3].real(), -1);
   ASSERT_FLOAT_EQ(z[3].imag(), -2);
   ASSERT_FLOAT_EQ(z[4].real(), 1);
   ASSERT_FLOAT_EQ(z[4].imag(), 0);
   ASSERT_FLOAT_EQ(z[5].real(), 0);
   ASSERT_FLOAT_EQ(z[5].imag(), 1);
   ASSERT_FLOAT_EQ(z[6].real(), -1);
   ASSERT_FLOAT_EQ(z[6].imag(), 0);
   ASSERT_FLOAT_EQ(z[7].real(), 0);
   ASSERT_FLOAT_EQ(z[7].imag(), -1);
   ASSERT_FLOAT_EQ(z[8].real(), 0);
   ASSERT_FLOAT_EQ(z[8].imag(), 0);

   std::complex<float> w;
   detail::scan_complex(std::stringstream("1+2i"), w);
   ASSERT_FLOAT_EQ(w.real(), 1);
   ASSERT_FLOAT_EQ(w.imag(), 2);
}

TEST(test_text_io_utility, output_list) {
   using namespace TAT;

   std::stringstream out;
   std::vector<int> v = {1, 2, 3};
   detail::print_list(
         out,
         [offset = 0, l = v.data(), count = v.size()](std::ostream& out) mutable {
            if (offset == count) {
               return true;
            }
            out << l[offset++];
            return offset == count;
         },
         '[',
         ']');
   ASSERT_EQ(out.str(), "[1,2,3]");
}
TEST(test_text_io_utility, output_list_empty) {
   using namespace TAT;

   std::stringstream out;
   std::vector<int> v = {};
   detail::print_list(
         out,
         [offset = 0, l = v.data(), count = v.size()](std::ostream& out) mutable {
            if (offset == count) {
               return true;
            }
            out << l[offset++];
            return offset == count;
         },
         '[',
         ']');
   ASSERT_EQ(out.str(), "[]");
}

TEST(test_text_io_utility, input_list) {
   using namespace TAT;

   std::stringstream in("[1,2,3]");
   std::vector<int> v;
   detail::scan_list(
         in,
         [&v](std::istream& in) {
            int& i = v.emplace_back();
            in >> i;
         },
         '[',
         ']');
   ASSERT_THAT(v, ElementsAre(1, 2, 3));
}

TEST(test_text_io_utility, input_list_empty) {
   using namespace TAT;

   std::stringstream in("[]");
   std::vector<int> v;
   detail::scan_list(
         in,
         [&v](std::istream& in) {
            int& i = v.emplace_back();
            in >> i;
         },
         '[',
         ']');
   ASSERT_THAT(v, ElementsAre());
}

TEST(test_text_io_utility, output_string_for_name) {
   using namespace TAT;

   std::stringstream out;
   print_string_for_name(out, "hello") << " world";
   ASSERT_EQ(out.str(), "hello world");
}

TEST(test_text_io_utility, output_fastname_for_name) {
   using namespace TAT;

   std::stringstream out;
   print_fastname_for_name(out, FastName("hello")) << " world";
   ASSERT_EQ(out.str(), "hello world");
}

TEST(test_text_io_utility, input_string_for_name) {
   using namespace TAT;

   std::stringstream in("hello\nworld,\nohh]\n");

   std::string name1;
   scan_string_for_name(in, name1);
   ASSERT_EQ(name1, "hello");
   ASSERT_EQ(in.get(), '\n');

   std::string name2;
   scan_string_for_name(in, name2);
   ASSERT_EQ(name2, "world");
   ASSERT_EQ(in.get(), ',');
   ASSERT_EQ(in.get(), '\n');

   std::string name3;
   scan_string_for_name(in, name3);
   ASSERT_EQ(name3, "ohh");
   ASSERT_EQ(in.get(), ']');
   ASSERT_EQ(in.get(), '\n');
}

TEST(test_text_io_utility, input_fastname_for_name) {
   using namespace TAT;

   std::stringstream in("hello\nworld,\nohh]\n");

   FastName name1;
   scan_fastname_for_name(in, name1);
   ASSERT_EQ(static_cast<const std::string&>(name1), "hello");
   ASSERT_EQ(in.get(), '\n');

   FastName name2;
   scan_fastname_for_name(in, name2);
   ASSERT_EQ(static_cast<const std::string&>(name2), "world");
   ASSERT_EQ(in.get(), ',');
   ASSERT_EQ(in.get(), '\n');

   FastName name3;
   scan_fastname_for_name(in, name3);
   ASSERT_EQ(static_cast<const std::string&>(name3), "ohh");
   ASSERT_EQ(in.get(), ']');
   ASSERT_EQ(in.get(), '\n');
}

TEST(test_text_io_utility, io_vector_empty) {
   using namespace TAT;

   std::stringstream ss;
   ss << std::vector<int>{};
   std::vector<int> a;
   ss >> a;
   ASSERT_THAT(a, ElementsAre());
}

TEST(test_text_io_utility, io_vector_normal) {
   using namespace TAT;

   std::stringstream ss;
   ss << std::vector<int>{1, 2, 3};
   std::vector<int> a;
   ss >> a;
   ASSERT_THAT(a, ElementsAre(1, 2, 3));
}

TEST(test_text_io_utility, io_vector_name) {
   using namespace TAT;

   std::stringstream ss;
   ss << std::vector<std::string>{"1", "2", "3"};
   std::vector<std::string> a;
   ss >> a;
   ASSERT_THAT(a, ElementsAre(std::string("1"), std::string("2"), std::string("3")));
}

TEST(test_text_io_utility, io_vector_complex) {
   using namespace TAT;

   std::stringstream ss;
   ss << std::vector<std::complex<float>>{{1, 2}, {2, 3}, {3, 4}};
   std::vector<std::complex<float>> a;
   ss >> a;
   ASSERT_THAT(a, ElementsAre(std::complex<float>(1, 2), std::complex<float>(2, 3), std::complex<float>(3, 4)));
}
