
/* TAT
 * Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#include <args.hxx>

#define TAT_USE_CPU

// SVD
#if (!defined TAT_USE_GESDD && !defined TAT_USE_GESVD && !defined TAT_USE_GESVDX)
#define TAT_USE_GESVD
#endif

// QR
#if (!defined TAT_USE_GEQRF && !defined TAT_USE_GEQP3)
#define TAT_USE_GEQRF
#endif

#include "TAT.hpp"

struct PEPS {
  using Size=TAT::Size;
  using Tensor=TAT::Tensor<TAT::Device::CPU, double>;

  int L1;
  int L2;
  Size d;
  Size D;
  Tensor hamiltonian;
  Tensor identity;

  enum class Direction {Right, Down};
  std::map<std::tuple<int, int>, Tensor> lattice;
  std::map<std::tuple<int, int, Direction>, Tensor> env;

  PEPS (int _L1, int _L2, Size _d, Size _D, double* H_data) : L1(_L1), L2(_L2), d(_d), D(_D) {
    using namespace TAT::legs_name;
    {
      Size left, right, up, down;
      for (int i=0; i<L1; i++) {
        for (int j=0; j<L2; j++) {
          left = right = up = down = D;
          if (i==0) {
            up = 1;
          }
          if (i==L1-1) {
            down = 1;
          }
          if (j==0) {
            left = 1;
          }
          if (j==L2-1) {
            right = 1;
          }
          lattice[ {i, j}] = Tensor({d, left, right, up, down}, {Phy, Left, Right, Up, Down});
        }
      }
    }
    {
      for (int i=0; i<L1; i++) {
        for (int j=0; j<L2-1; j++) {
          env[ {i, j, Direction::Right}] = Tensor({D}, {Phy});
          env[ {i, j, Direction::Right}].set_constant(1);
          std::cout << env[ {i, j, Direction::Right}];
        }
      }
      for (int i=0; i<L1-1; i++) {
        for (int j=0; j<L2; j++) {
          env[ {i, j, Direction::Down}] = Tensor({D}, {Phy});
          env[ {i, j, Direction::Down}].set_constant(1);
          std::cout << env[ {i, j, Direction::Down}];
        }
      }
    }
    {
      hamiltonian = Tensor({d, d, d, d}, {Phy1, Phy2, Phy3, Phy4});
      std::memcpy(hamiltonian.get(), H_data, hamiltonian.size()*sizeof(double));
    }
    {
      identity = Tensor({d, d, d, d}, {Phy1, Phy2, Phy3, Phy4});
      identity.set_zero();
      for (Size i=0; i<d*d; i++) {
        identity.get()[i*d+i] = 1;
      }
    }
  }

  static double random() {
    return double(std::rand())/(RAND_MAX)*2-1;
  }

  void set_random_state(unsigned seed) {
    std::srand(seed);
    for (int i=0; i<L1; i++) {
      for (int j=0; j<L2; j++) {
        lattice[ {i, j}].set_random(random);
        std::cout << lattice[ {i, j}] << std::endl;
      }
    }
  }
};

int main() {
  double default_H[16] = {
    0.25, 0, 0, 0,
    0, -0.25, 0.5, 0,
    0, 0.5, -0.25, 0,
    0, 0, 0, 0.25
  };
  PEPS peps(2, 3, 2, 4, default_H);
  peps.set_random_state(42);
}
