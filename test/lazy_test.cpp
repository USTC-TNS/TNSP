#include <iostream>
#include <lazy.hpp>

int main() {
   auto a = lazy::Root(1);
   auto b = lazy::Root(2);
   std::cout << a->get() << "\n";
   std::cout << b->get() << "\n";
   auto c = lazy::Path([](int a, int b) { return a + b; }, a, b);
   auto d = lazy::Node([](int c, int a) { return c * a; }, c, a);
   std::cout << d->get() << "\n";
   a->set(233);
   std::cout << d->get() << "\n";
   auto snap = lazy::default_graph.dump();
   b->set(666);
   std::cout << d->get() << "\n";
   lazy::default_graph.load(snap);
   std::cout << d->get() << "\n";
}
