#include <TAT/TAT.hpp>

using Tensor = typename TAT::Tensor<float, TAT::NoSymmetry>;

int main() {
   auto a = Tensor({"A", "B"}, {5, 10}).test();
   std::cout << a << '\n';
   auto [q, r] = a.qr('r', {"A"}, "newB", "newA");
   std::cout << q << '\n';
   std::cout << r << '\n';
}
