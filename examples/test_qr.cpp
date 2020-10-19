#include <TAT/TAT.hpp>

using Tensor = typename TAT::Tensor<float, TAT::NoSymmetry>;

int main() {
   auto a = Tensor({"A", "B"}, {5, 10}).test();
   std::cout << a << '\n';
   {
      auto [q, r] = a.qr('r', {"A"}, "newB", "newA");
      std::cout << q << '\n';
      std::cout << r << '\n';
   }
   {
      auto [q, r] = a.qr('r', {"B"}, "newB", "newA");
      std::cout << q << '\n';
      std::cout << r << '\n';
   }
   auto b = Tensor({"A", "B"}, {10, 5}).test();
   std::cout << b << '\n';
   {
      auto [q, r] = b.qr('r', {"A"}, "newB", "newA");
      std::cout << q << '\n';
      std::cout << r << '\n';
   }
   {
      auto [q, r] = b.qr('r', {"B"}, "newB", "newA");
      std::cout << q << '\n';
      std::cout << r << '\n';
   }
}
