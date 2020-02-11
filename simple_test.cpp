/**
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <fstream>
#include <iostream>
#include <sstream>

#include <TAT/TAT.hpp>

#define RUN_TEST(x)                 \
   do {                             \
      std::cout << "# " #x << "\n"; \
      x();                          \
      std::cout << "\n";            \
   } while (false)

void test_create_tensor() {
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}
                      .test()
             << "\n";
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {0, 3}}
             << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{}, {}}.set([]() { return 10; }) << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{}, {}}.set([]() { return 10; }).at({})
             << "\n";
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}
                      .test()
                      .at({{TAT::Right, 2}, {TAT::Left, 1}})
             << "\n";
}
void test_create_symmetry_tensor() {
   std::cout << TAT::Tensor<double, TAT::Z2Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                     {{{1, 3}, {0, 1}},
                                                      {{1, 1}, {0, 2}},
                                                      {{1, 2}, {0, 3}}}}
                      .zero()
             << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                     {{{-1, 3}, {0, 1}, {1, 2}},
                                                      {{-1, 1}, {0, 2}, {1, 3}},
                                                      {{-1, 2}, {0, 3}, {1, 1}}}}
                      .test(2)
             << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry> {
      {TAT::Left, TAT::Right, TAT::Up},
#ifdef _MSVC_LANG
            // 这似乎是MSVC的一个bug, 如果用下面的写法, Edge的析构函数将会被调用两次
            {std::map<TAT::U1Symmetry, TAT::Size>{},
#else
      {
         {},
#endif
             {{-1, 1}, {0, 2}, {1, 3}},
             {{-1, 2}, {0, 3}, {1, 1}}}
   }
   .zero() << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>{{}, {}}.set([]() { return 123; }) << "\n";
}

void test_create_fermi_symmetry_tensor() {
   std::cout << TAT::Tensor<double, TAT::FermiSymmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                        {{{0, 1}, {1, 2}},
                                                         {{-1, 1}, {-2, 3}, {0, 2}},
                                                         {{0, 3}, {1, 1}}}}
                      .test(2)
             << "\n";
   std::cout
         << TAT::Tensor<double, TAT::FermiU1Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                      {{{{0, 0}, 1}, {{1, 1}, 2}},
                                                       {{{-1, -1}, 1}, {{-2, 0}, 3}, {{0, 0}, 2}},
                                                       {{{0, 0}, 3}, {{1, 1}, 1}}}}
                  .test(2)
         << "\n";
   std::cout
         << TAT::Tensor<double, TAT::FermiU1Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                      {{{{0, 0}, 1}, {{1, 1}, 2}},
                                                       {{{-1, -1}, 1}, {{-2, 0}, 3}, {{0, 0}, 2}},
                                                       {{{0, 0}, 3}, {{1, 1}, 1}}}}
                  .test(2)
                  .block({{TAT::Left, {0, 0}}, {TAT::Up, {1, 1}}, {TAT::Right, {1, -1}}})
         << "\n";
   std::cout << TAT::Tensor<double, TAT::FermiU1Symmetry>{1234}.at({}) << "\n";
   std::cout
         << TAT::Tensor<double, TAT::FermiU1Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                      {{{{0, 0}, 1}, {{1, 1}, 2}},
                                                       {{{-1, -1}, 1}, {{-2, 0}, 3}, {{0, 0}, 2}},
                                                       {{{0, 0}, 3}, {{1, 1}, 1}}}}
                  .test(2)
                  .at({{TAT::Left, {{0, 0}, 0}}, {TAT::Up, {{0, 0}, 1}}, {TAT::Right, {{0, 0}, 1}}})
         << "\n";
}

void test_type_conversion() {
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>{123} << "\n";
   std::cout << double(TAT::Tensor<double, TAT::U1Symmetry>{123}) << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}
                      .test(2)
                      .to<double>()
             << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}
                      .test(2)
                      .to<std::complex<double>>()
             << "\n";
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}
                      .test(2)
                      .to<double>()
             << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                     {{{-1, 3}, {0, 1}, {1, 2}},
                                                      {{-1, 1}, {0, 2}, {1, 3}},
                                                      {{-1, 2}, {0, 3}, {1, 1}}}}
                      .test(2)
                      .to<std::complex<double>>()
             << "\n";
}

void test_norm() {
   auto t =
         TAT::Tensor<double, TAT::U1Symmetry>{
               {TAT::Left, TAT::Right, TAT::Up},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .test(2)
               .to<std::complex<double>>();
   std::cout << t.norm<-1>() << "\n";
   std::cout << t.norm<0>() << "\n";
   std::cout << t.norm<1>() << "\n";
   std::cout << t.norm<2>() << "\n";
}

void test_scalar() {
   auto t = TAT::Tensor<double, TAT::Z2Symmetry>{
         {TAT::Left, TAT::Right, TAT::Phy}, {{{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}}};
   t.test();
   std::cout << t + 1.0 << "\n";
   std::cout << 1.0 / t << "\n";

   auto a = TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}.test();
   auto b = TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}.test(0, 0.1);
   std::cout << a + b << "\n";
   std::cout << a - b << "\n";
   std::cout << a * b << "\n";
   std::cout << a / b << "\n";
   std::cout << a + b.edge_rename({{TAT::Left, TAT::Up}}) << "\n";
}

void test_io() {
   std::stringstream ss;
   auto a =
         TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right, TAT::Up}, {2, 3, 4}}.test();
   ss <= a;
   auto b = TAT::Tensor<double, TAT::NoSymmetry>();
   ss >= b;
   std::cout << a << "\n";
   std::cout << b << "\n";
   auto c =
         TAT::Tensor<double, TAT::U1Symmetry>{
               {TAT::Left, TAT::Right, TAT::Up},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .test(2);
   ss <= c;
   auto d = TAT::Tensor<double, TAT::U1Symmetry>();
   ss >= d;
   std::cout << c << "\n";
   std::cout << d << "\n";
   auto e = TAT::Tensor<std::complex<int>, TAT::NoSymmetry>{{TAT::Up, TAT::Left, TAT::Right},
                                                            {1, 2, 3}}
                  .set([]() {
                     static int i = 0;
                     static int arr[6] = {0x12345, 0x23456, 0x34567, 0x45678, 0x56789, 0x6789a};
                     return arr[i++];
                  });
   ss <= e;
   auto f = TAT::Tensor<std::complex<int>, TAT::NoSymmetry>();
   ss >= f;
   std::cout << e << "\n";
   std::cout << f << "\n";
   auto g =
         TAT::Tensor<std::complex<double>, TAT::U1Symmetry>{
               {TAT::Left, TAT::Right, TAT::Up},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .test(2);
   ss <= g;
   auto h = TAT::Tensor<std::complex<double>, TAT::U1Symmetry>();
   ss >= h;
   std::cout << g << "\n";
   std::cout << h << "\n";
   auto i =
         TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>{
               {TAT::Left, TAT::Right, TAT::Up},
               {{{-2, 3}, {0, 1}, {-1, 2}}, {{0, 2}, {1, 3}}, {{0, 3}, {1, 1}}}}
               .test(2);
   ss <= i;
   auto j = TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>();
   ss >= j;
   std::cout << i << "\n";
   std::cout << j << "\n";
}

void test_edge_rename() {
   auto t1 = TAT::Tensor<double, TAT::Z2Symmetry>{
         {TAT::Left, TAT::Right, TAT::Phy}, {{{0, 1}, {1, 2}}, {{0, 3}, {1, 4}}, {{0, 5}, {1, 6}}}};
   auto t2 = t1.edge_rename({{TAT::Left, TAT::Up}});
   t1.test();
   std::cout << t1 << "\n";
   std::cout << t2 << "\n";
}

void test_edge_operator() {
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {8, 8}}.test() << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {8, 8}}.test().edge_operator(
                      {{"A", "C"}},
                      {{"C", {{"D", 4}, {"E", 2}}}, {"B", {{"F", 2}, {"G", 4}}}},
                      {"D", "F"},
                      {{"I", {"D", "F"}}, {"J", {"G", "E"}}},
                      {"J", "I"})
             << "\n";
   std::cout << TAT::Tensor<>{{"A", "B", "C"}, {2, 3, 4}}.test().edge_operator(
                      {}, {}, {}, {}, {"B", "C", "A"})
             << '\n';
}
// TODO: 重新处理edge operator
#if 0
void test_transpose() {
   auto a = TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {2, 3}}.set([]() {
      static double i = -1;
      return i += 1;
   });
   std::cout << a << "\n";
   std::cout << a.transpose({TAT::Right, TAT::Left}) << "\n";
   auto b = TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right, TAT::Up}, {2, 3, 4}}.set(
         []() {
            static double i = -1;
            return i += 1;
         });
   std::cout << b << "\n";
   std::cout << b.transpose({TAT::Right, TAT::Up, TAT::Left}) << "\n";
   auto c =
         TAT::Tensor<std::complex<double>, TAT::U1Symmetry>{
               {TAT::Left, TAT::Right, TAT::Up},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .set([]() {
                  static double i = 0;
                  return i += 1;
               });
   std::cout << c << "\n";
   std::cout << c.transpose({TAT::Right, TAT::Up, TAT::Left}) << "\n";
   auto d =
         TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>{
               {TAT::Left, TAT::Right, TAT::Up},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .set([]() {
                  static double i = 0;
                  return i += 1;
               });
   std::cout << d << "\n";
   std::cout << d.transpose({TAT::Right, TAT::Up, TAT::Left}) << "\n";
   auto e = TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Down, TAT::Up, TAT::Left, TAT::Right},
                                                 {2, 3, 4, 5}}
                  .set([]() {
                     static double i = 0;
                     return i += 1;
                  });
   std::cout << e << "\n";
   std::cout << e.transpose({TAT::Left, TAT::Down, TAT::Right, TAT::Up}) << "\n";
   auto f = TAT::Tensor<double, TAT::NoSymmetry>{{"l1", "l2", "l3"}, {2, 3, 4}}.set([]() {
      static double i = -1;
      return i += 1;
   });
   std::cout << f << "\n";
   std::cout << f.transpose({"l1", "l2", "l3"}) << "\n";
   std::cout << f.transpose({"l1", "l3", "l2"}) << "\n";
   std::cout << f.transpose({"l2", "l1", "l3"}) << "\n";
   std::cout << f.transpose({"l2", "l3", "l1"}) << "\n";
   std::cout << f.transpose({"l3", "l1", "l2"}) << "\n";
   std::cout << f.transpose({"l3", "l2", "l1"}) << "\n";
}

/*
void test_mpi() {
   auto f = TAT::MPIFile("log");
   f.seek(TAT::mpi.rank*20);
   auto s = "Hello From " + std::to_string(TAT::mpi.rank) + "\n";
   f.write(s.data(), s.size());
}
*/

void test_merge_split() {
   const auto a = TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {2, 3}}.set([]() {
      static double i = -1;
      return i += 1;
   });
   auto b = a.merge_edge({{"Merged", {"Left", "Right"}}});
   auto c = a.merge_edge({{"Merged", {"Right", "Left"}}});
   auto d = c.split_edge({{"Merged", {{"1", 3}, {"2", 2}}}});
   std::cout << a << "\n";
   std::cout << b << "\n";
   std::cout << c << "\n";
   std::cout << d << "\n";
   auto e =
         TAT::Tensor<std::complex<double>, TAT::U1Symmetry>{
               {TAT::Left, TAT::Right, TAT::Up},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .set([]() {
                  static double i = 0;
                  return i += 1;
               });
   auto f = e.merge_edge({{"Merged", {TAT::Left, "Up"}}});
   auto g = f.split_edge(
         {{"Merged", {{"Left", {{-1, 3}, {0, 1}, {1, 2}}}, {"Up", {{-1, 2}, {0, 3}, {1, 1}}}}}});
   std::cout << e << "\n";
   std::cout << f << "\n";
   std::cout << g << "\n";
}

void test_edge_operator() {
   do {
      auto a = TAT::Tensor<double, TAT::U1Symmetry>{{TAT::Left, TAT::Right, TAT::Up, TAT::Down},
                                                    {{{-1, 3}, {0, 1}, {1, 2}},
                                                     {{-1, 1}, {0, 4}, {1, 2}},
                                                     {{-1, 2}, {0, 3}, {1, 1}},
                                                     {{-1, 1}, {0, 3}, {1, 2}}}}
                     .set([]() {
                        static double i = 0;
                        return i += 1;
                     });
      std::cout << "origin = \n" << a << "\n";
#if 0
   auto b = a.edge_operator(
         {{"Right", "Right1"}},
         {{"Down", {{"Down1", {{0, 1}, {1, 2}}}, {"Down2", {{-1, 1}, {0, 1}}}}}},
         {},
         {"Left", "Right1", "Up", "Down1", "Down2"});
#else
      auto b = a.edge_rename({{"Right", "Right1"}})
                     .split_edge(
                           {{"Down", {{"Down1", {{0, 1}, {1, 2}}}, {"Down2", {{-1, 1}, {0, 1}}}}}});
#endif
      std::cout << "splitted = \n" << b << "\n";
#if 0
   auto c = a.edge_operator(
         {{"Right", "Right1"}},
         {{"Down", {{"Down1", {{0, 1}, {1, 2}}}, {"Down2", {{-1, 1}, {0, 1}}}}}},
         {},
         {"Down1", "Right1", "Up", "Left", "Down2"});
#else
      auto c = b.transpose({"Down1", "Right1", "Up", "Left", "Down2"});
#endif
      std::cout << "transposed = \n" << c << "\n";
#if 0
   auto d = a.edge_operator(
         {{"Right", "Right1"}},
         {{"Down", {{"Down1", {{0, 1}, {1, 2}}}, {"Down2", {{-1, 1}, {0, 1}}}}}},
         {{"Left", {"Left", "Down2"}}},
         {"Down1", "Right1", "Up", "Left"});
#else
      auto d = c.merge_edge({{"Left", {"Left", "Down2"}}});
#endif
      std::cout << "merged = \n" << d << "\n";
   } while (false);
   do {
      auto a = TAT::Tensor<double, TAT::FermiSymmetry>{{TAT::Left, TAT::Right, TAT::Up, TAT::Down},
                                                       {{{-1, 3}, {0, 1}, {1, 2}},
                                                        {{-1, 1}, {0, 4}, {1, 2}},
                                                        {{-1, 2}, {0, 3}, {1, 1}},
                                                        {{-1, 1}, {0, 3}, {1, 2}}}}
                     .set([]() {
                        static double i = 0;
                        return i += 1;
                     });
      std::cout << "origin = \n" << a << "\n";
#if 0
   auto b = a.edge_operator(
         {{"Right", "Right1"}},
         {{"Down", {{"Down1", {{0, 1}, {1, 2}}}, {"Down2", {{-1, 1}, {0, 1}}}}}},
         {},
         {"Left", "Right1", "Up", "Down1", "Down2"});
#else
      auto b = a.edge_rename({{"Right", "Right1"}})
                     .split_edge(
                           {{"Down", {{"Down1", {{0, 1}, {1, 2}}}, {"Down2", {{-1, 1}, {0, 1}}}}}});
#endif
      std::cout << "splitted = \n" << b << "\n";
#if 0
   auto c = a.edge_operator(
         {{"Right", "Right1"}},
         {{"Down", {{"Down1", {{0, 1}, {1, 2}}}, {"Down2", {{-1, 1}, {0, 1}}}}}},
         {},
         {"Down1", "Right1", "Up", "Left", "Down2"});
#else
      auto c = b.transpose({"Down1", "Right1", "Up", "Left", "Down2"});
#endif
      std::cout << "transposed = \n" << c << "\n";
#if 0
   auto d = a.edge_operator(
         {{"Right", "Right1"}},
         {{"Down", {{"Down1", {{0, 1}, {1, 2}}}, {"Down2", {{-1, 1}, {0, 1}}}}}},
         {{"Left", {"Left", "Down2"}}},
         {"Down1", "Right1", "Up", "Left"});
#else
      auto d = c.merge_edge({{"Left", {"Left", "Down2"}}});
#endif
      std::cout << "merged = \n" << d << "\n";
   } while (false);
}
#endif

int main(const int argc, char** argv) {
   std::stringstream out;
   auto cout_buf = std::cout.rdbuf();
   if (argc != 1) {
      std::cout.rdbuf(out.rdbuf());
   }
   RUN_TEST(test_create_tensor);
   RUN_TEST(test_create_symmetry_tensor);
   RUN_TEST(test_create_fermi_symmetry_tensor);
   RUN_TEST(test_type_conversion);
   RUN_TEST(test_norm);
   RUN_TEST(test_scalar);
   RUN_TEST(test_io);
   RUN_TEST(test_edge_rename);
   RUN_TEST(test_edge_operator);
#if 0
   // RUN_TEST(test_mpi);
   RUN_TEST(test_transpose);
   RUN_TEST(test_merge_split);
#endif
   if (argc != 1) {
      std::cout.rdbuf(cout_buf);
      std::ifstream fout(argv[1]);
      std::string sout((std::istreambuf_iterator<char>(fout)), std::istreambuf_iterator<char>());
      return sout != out.str();
   }
   return 0;
}

int simple_test() {
   return main(0, 0);
}
