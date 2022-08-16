#include <chrono>
#include <complex>
#include <iostream>
#include <random>

#include <TAT/TAT.hpp>

using Tensor = TAT::Tensor<std::complex<double>, TAT::U1Symmetry>;
using Edge = TAT::Edge<TAT::U1Symmetry>;

int main() {
   auto out22 = Edge({{-2, 10}, {-1, 20}, {0, 40}, {+1, 20}, {+2, 10}});
   auto in22 = out22.conjugated_edge();
   auto out04 = Edge({{0, 1}, {1, 6}, {2, 17}, {3, 24}, {4, 16}});
   auto in04 = out04.conjugated_edge();

   auto eng = std::default_random_engine(std::random_device()());
   auto dist = std::normal_distribution<double>(0, 1);
   auto setter = [&]() -> std::complex<double> {
      return {dist(eng), dist(eng)};
   };

   auto L = Tensor({"g", "d", "a"}, {in22, out04, out22}).set(setter);
   auto U = Tensor({"c", "b", "a"}, {out22, in04, in22}).set(setter);
   auto M = Tensor({"b", "e", "f", "d"}, {out04, out04, in04, in04}).set(setter);
   auto D = Tensor({"h", "f", "g"}, {in22, out04, out22}).set(setter);

   auto begin = std::chrono::system_clock::now();
   std::cout << "BEGIN\n";

   for (auto i = 0; i < 10; i++) {
      auto res = L.contract(U, {{"a", "a"}}).contract(M, {{"b", "b"}, {"d", "d"}}).contract(D, {{"f", "f"}, {"g", "g"}});
   }

   auto end = std::chrono::system_clock::now();
   std::cout << "END\n";
   std::chrono::duration<double> dur = end - begin;
   std::cout << dur.count() / 10 << "\n";

   return 0;
}
