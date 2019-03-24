#define TAT_USE_CPU

// SVD
// #define TAT_USE_GESDD
// #define TAT_USE_GESVD
#define TAT_USE_GESVDX

// QR
#define TAT_USE_GEQRF
// #define TAT_USE_GEQP3
// GEQP3 not understand, maybe useful if R will drop

#include "TAT.hpp"

struct MPS {
  TAT_Legs;
  using TAT::Size;
  using Tensor=TAT::Tensor<TAT::Device::CPU, double>;

  Size D;
  Size L;
  double delta_t;
  std::vector<Tensor> lattice;

  MPS(Size _D, Size _L) : D(_D), L(_L) {
    lattice.push_back(Tensor({2,D},{Phy, Right}));
    for(TAT::Size i=1;i<L-1;i++){
      lattice.push_back(Tensor({2,D,D},{Phy, Left, Right}));
    }
    lattice.push_back(Tensor({2,D},{Phy, Left}));
  }
};

int main() {
  MPS mps(4, 4);
  return 0;
}
