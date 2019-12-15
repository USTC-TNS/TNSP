#include <fstream>
#include <iostream>
#include <sstream>

#include "TAT.hpp"

#define RUN_TEST(x)                 \
   do {                             \
      std::cout << "# " #x << "\n"; \
      x();                          \
      std::cout << "\n";            \
   } while (false)

void test_create_nosymmetry_tensor() {
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}
                      .set([]() {
                         static int i = 0;
                         return i++;
                      })
             << "\n";
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {0, 3}}
             << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{}, {}}.set([]() { return 10; }) << "\n";
}

void test_create_z2symmetry_tensor() {
   std::cout << TAT::Tensor<double, TAT::Z2Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                     {{{1, 3}, {0, 1}},
                                                      {{1, 1}, {0, 2}},
                                                      {{1, 2}, {0, 3}}}}
                      .set([]() { return 0; })
             << "\n";
   std::cout << TAT::Tensor<double, TAT::Z2Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                     {{}, {{1, 1}, {0, 2}}, {{1, 2}, {0, 3}}}}
                      .set([]() { return 0; })
             << "\n";
   std::cout << TAT::Tensor<double, TAT::Z2Symmetry>{{}, {}}.set([]() { return 123; }) << "\n";
}

void test_create_u1symmetry_tensor() {
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                     {{{-1, 3}, {0, 1}, {1, 2}},
                                                      {{-1, 1}, {0, 2}, {1, 3}},
                                                      {{-1, 2}, {0, 3}, {1, 1}}}}
                      .set([]() {
                         static double i = 1;
                         return i += 1;
                      })
             << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                     {{},
                                                      {{-1, 1}, {0, 2}, {1, 3}},
                                                      {{-1, 2}, {0, 3}, {1, 1}}}}
                      .set([]() { return 0; })
             << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>{{}, {}}.set([]() { return 123; }) << "\n";
}

void test_type_conversion() {
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}
                      .set([]() {
                         static double i = 1;
                         return i += 1;
                      })
                      .to<double>()
             << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}
                      .set([]() {
                         static double i = 1;
                         return i += 1;
                      })
                      .to<std::complex<double>>()
             << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>{{TAT::Left, TAT::Right, TAT::Up},
                                                     {{{-1, 3}, {0, 1}, {1, 2}},
                                                      {{-1, 1}, {0, 2}, {1, 3}},
                                                      {{-1, 2}, {0, 3}, {1, 1}}}}
                      .set([]() {
                         static double i = 1;
                         return i += 1;
                      })
                      .to<std::complex<double>>()
             << "\n";
}

void test_norm() {
   auto t =
         TAT::Tensor<double, TAT::U1Symmetry>{
               {TAT::Left, TAT::Right, TAT::Up},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .set([]() {
                  static double i = 1;
                  return i += 1;
               })
               .to<std::complex<double>>();
   std::cout << t.norm<-1>() << "\n";
   std::cout << t.norm<0>() << "\n";
   std::cout << t.norm<1>() << "\n";
   std::cout << t.norm<2>() << "\n";
}

void test_edge_rename() {
   auto t1 = TAT::Tensor<double, TAT::Z2Symmetry>{
         {TAT::Left, TAT::Right, TAT::Phy}, {{{0, 1}, {1, 2}}, {{0, 3}, {1, 4}}, {{0, 5}, {1, 6}}}};
   auto t2 = t1.edge_rename({{TAT::Left, TAT::Up}});
   t1.set([]() {
      static int i = 0;
      return i++;
   });
   std::cout << t1 << "\n";
   std::cout << t2 << "\n";
}

void test_scalar() {
   auto t = TAT::Tensor<double, TAT::Z2Symmetry>{
         {TAT::Left, TAT::Right, TAT::Phy}, {{{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}}};
   t.set([]() {
      static int i = 0;
      return i++;
   });
   std::cout << t + 1.0 << "\n";
   std::cout << 1.0 / t << "\n";

   auto a = TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}.set([]() {
      static double i = -1;
      return i += 1;
   });
   auto b = TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right}, {3, 4}}.set([]() {
      static double i = -0.1;
      return i += 0.1;
   });
   std::cout << a + b << "\n";
   std::cout << a - b << "\n";
   std::cout << a * b << "\n";
   std::cout << a / b << "\n";
   std::cout << a + b.edge_rename({{TAT::Left, TAT::Up}}) << "\n";
}

void test_io() {
   std::stringstream ss;
   auto a = TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Left, TAT::Right, TAT::Up}, {2, 3, 4}}.set(
         []() {
            static double i = -1;
            return i += 1;
         });
   ss << a;
   auto b = TAT::Tensor<double, TAT::NoSymmetry>();
   ss >> b;
   std::cout << a << "\n";
   std::cout << b << "\n";
   auto c =
         TAT::Tensor<double, TAT::U1Symmetry>{
               {TAT::Left, TAT::Right, TAT::Up},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .set([]() {
                  static double i = 1;
                  return i += 1;
               });
   ss << c;
   auto d = TAT::Tensor<double, TAT::U1Symmetry>();
   ss >> d;
   std::cout << c << "\n";
   std::cout << d << "\n";
   auto e = TAT::Tensor<std::complex<int>, TAT::NoSymmetry>{{TAT::Up, TAT::Left, TAT::Right},
                                                            {1, 2, 3}}
                  .set([]() {
                     static int i = 0;
                     static int arr[6] = {0x12345, 0x23456, 0x34567, 0x45678, 0x56789, 0x6789a};
                     return arr[i++];
                  });
   ss << e;
   auto f = TAT::Tensor<std::complex<int>, TAT::NoSymmetry>();
   ss >> f;
   std::cout << e << "\n";
   std::cout << f << "\n";
   auto g =
         TAT::Tensor<std::complex<double>, TAT::U1Symmetry>{
               {TAT::Left, TAT::Right, TAT::Up},
               {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
               .set([]() {
                  static double i = 1;
                  return i += 1;
               });
   ss << g;
   auto h = TAT::Tensor<std::complex<double>, TAT::U1Symmetry>();
   ss >> h;
   std::cout << g << "\n";
   std::cout << h << "\n";
}

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
   auto d = TAT::Tensor<double, TAT::NoSymmetry>{{TAT::Down, TAT::Up, TAT::Left, TAT::Right},
                                                 {2, 3, 4, 5}}
                  .set([]() {
                     static double i = 0;
                     return i += 1;
                  });
   std::cout << d << "\n";
   std::cout << d.transpose({TAT::Left, TAT::Down, TAT::Right, TAT::Up}) << "\n";
}

int main(int argc, char** argv) {
   std::stringstream out;
   auto coutbuf = std::cout.rdbuf();
   if (argc != 1) {
      std::cout.rdbuf(out.rdbuf());
   }
   RUN_TEST(test_create_nosymmetry_tensor);
   RUN_TEST(test_create_z2symmetry_tensor);
   RUN_TEST(test_create_u1symmetry_tensor);
   RUN_TEST(test_type_conversion);
   RUN_TEST(test_norm);
   RUN_TEST(test_edge_rename);
   RUN_TEST(test_scalar);
   RUN_TEST(test_io);
   RUN_TEST(test_transpose);
   if (argc != 1) {
      std::cout.rdbuf(coutbuf);
      std::ifstream fout(argv[1]);
      std::string sout((std::istreambuf_iterator<char>(fout)), std::istreambuf_iterator<char>());
      return sout != out.str();
   }
   return 0;
}
