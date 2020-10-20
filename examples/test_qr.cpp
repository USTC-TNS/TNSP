#include <TAT/TAT.hpp>

using Tensor = typename TAT::Tensor<float, TAT::NoSymmetry>;

int main() {
   auto a = Tensor({"A", "B"}, {5, 10}).test();
   std::cout << a << '\n';
   {
      auto [q, r] = a.qr('r', {"A"}, "newQ", "newR");
      std::cout << q << '\n'; // "B" "newQ"
      std::cout << r << '\n'; // "A" "newR"
      std::cout << (q.contract(r, {{"newQ", "newR"}}) - a).norm<-1>() << '\n';
   }
   {
      auto [q, r] = a.qr('r', {"B"}, "newQ", "newR");
      std::cout << q << '\n';
      std::cout << r << '\n';
      std::cout << (q.contract(r, {{"newQ", "newR"}}) - a).norm<-1>() << '\n';
   }
   auto b = Tensor({"A", "B"}, {10, 5}).test();
   std::cout << b << '\n';
   {
      auto [q, r] = b.qr('r', {"A"}, "newQ", "newR");
      std::cout << q << '\n';
      std::cout << r << '\n';
      std::cout << (q.contract(r, {{"newQ", "newR"}}) - b).norm<-1>() << '\n';
   }
   {
      auto [q, r] = b.qr('r', {"B"}, "newQ", "newR");
      std::cout << q << '\n';
      std::cout << r << '\n';
      std::cout << (q.contract(r, {{"newQ", "newR"}}) - b).norm<-1>() << '\n';
   }
}
