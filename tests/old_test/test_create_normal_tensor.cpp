#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_create_normal_tensor, basic_usage) {
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range() << "\n";
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {0, 3}} << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{}, {}}.set([]() {
      return 10;
   }) << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{}, {}}
                      .set([]() {
                         return 10;
                      })
                      .at()
             << "\n";
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range().at({{"Right", 2}, {"Left", 1}}) << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>(2333, {"i", "j"}) << "\n";
}
