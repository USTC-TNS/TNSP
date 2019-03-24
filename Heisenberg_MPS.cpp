#include <cstdlib>
#include <iostream>
#include <ctime>

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
  using Size=TAT::Size;
  using Tensor=TAT::Tensor<TAT::Device::CPU, double>;

  Size D;
  Size L;
  Tensor hamiltonian;
  Tensor identity;
  std::vector<Tensor> lattice;

  static double random() {
    return double(std::rand())/(RAND_MAX)*2-1;
  }

  MPS(Size _D, Size _L) : D(_D), L(_L), hamiltonian({}, {}), identity({}, {}) {
    {
      lattice.push_back(Tensor({2, D}, {Phy, Right}));
      for (TAT::Size i=1; i<L-1; i++) {
        lattice.push_back(Tensor({2, D, D}, {Phy, Left, Right}));
      }
      lattice.push_back(Tensor({2, D}, {Phy, Left}));
    } // lattice
    {
      double default_H[16] = {
        1/4., 0, 0, 0,
        0, -1/4., 2/4., 0,
        0, 2/4., -1/4., 0,
        0, 0,    0, 1/4.
      };
      hamiltonian = Tensor({2, 2, 2, 2}, {Phy1, Phy2, Phy3, Phy4});
      double* H = hamiltonian.node.data.get();
      for (int i=0; i<16; i++) {
        H[i] = default_H[16];
      }
    } // hamiltonian
    {
      double default_I[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      };
      identity = Tensor({2, 2, 2, 2}, {Phy1, Phy2, Phy3, Phy4});
      double* I = identity.node.data.get();
      for (int i=0; i<16; i++) {
        I[i] = default_I[16];
      }
    } // identity
  }

  void set_random_state(unsigned seed) {
    std::srand(seed);
    for (auto& i : lattice) {
      i.set_random(random);
    }
  }

};

std::ostream& operator<<(std::ostream& out, const MPS& mps) {
  out << "[L(" << mps.L << ") D(" << mps.D << ") lattice(" << std::endl;
  for (auto& i : mps.lattice) {
    out << "  " << i << std::endl;
  }
  out << ")]" << std::endl;
  return out;
}

int main() {
  MPS mps(4, 4);
  mps.set_random_state(42);
  std::cout << mps;
  return 0;
}
