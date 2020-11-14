#include <TAT/TAT.hpp>
#include <sstream>

int main() {
   using namespace TAT;
   Tensor<> a;
   Tensor<double, U1Symmetry> b;
   char c, d;
   auto ss = std::stringstream(
         "{names:[Left,Right],edges:[3,4],blocks:[0,1,2,3,4,5,6,7,8,9,10,11]}?{names:[A,B,C,D],edges:[{-2:1,-1:1,0:1},{0:1,1:2},{0:2,1:2},{-2:2,-1:1,"
         "0:2}],blocks:{[-2,1,1,0]:[0,1,2,3,4,5,6,7],[-1,0,1,0]:[8,9,10,11],[-1,1,0,0]:[12,13,14,15,16,17,18,19],[-1,1,1,-1]:[20,21,22,23],[0,0,0,0]:"
         "[24,25,26,27],[0,0,1,-1]:[28,29],[0,1,0,-1]:[30,31,32,33],[0,1,1,-2]:[34,35,36,37,38,39,40,41]}}*");
   ss >> a >> c >> b >> d;
   std::cout << a << std::endl;
   std::cout << b << std::endl;
   std::cout << c << d << std::endl;
}
