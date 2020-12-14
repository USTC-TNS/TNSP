#include <iostream>
#include <lazy.hpp>

int main() {
   auto a = lazy::Root(1);
   auto b = lazy::Root(2);
   std::cout << **a << "\n";
   std::cout << **b << "\n";
   auto c = lazy::Path([](int a, int b) { return a + b; }, a, b);
   auto d = lazy::Node([](int c, int a) { return c * a; }, c, a);
   std::cout << **d << "\n";
   *a = 233;
   std::cout << **d << "\n";
   auto snap = lazy::default_graph.dump();
   *b = 666;
   std::cout << **d << "\n";
   lazy::default_graph.load(snap);
   std::cout << **d << "\n";
}
