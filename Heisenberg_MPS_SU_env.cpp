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

struct MPS {
  using Size=TAT::Size;
  using RawTensor=TAT::Tensor<TAT::Device::CPU, double>;
  using Tensor=TAT::Site<TAT::Device::CPU, double>;

  int L;
  Size D;
  RawTensor hamiltonian;
  RawTensor identity;
  std::vector<Tensor> lattice;

  static double random() {
    return double(std::rand())/(RAND_MAX)*2-1;
  }

  MPS(int _L, Size _D) : L(_L), D(_D) {
    using namespace TAT::legs_name;
    {
      lattice.push_back(Tensor().set(RawTensor({Phy, Left, Right}, {2, 1, D})));
      for (int i=1; i<L-1; i++) {
        lattice.push_back(Tensor().set(RawTensor({Phy, Left, Right}, {2, D, 1})));
      }
      lattice.push_back(Tensor().set(RawTensor({Phy, Left, Right}, {2, D, 1})));
    } // lattice
    {
      for (int i=0; i<L-1; i++) {
        lattice[i].link_env(Right, lattice[i+1], Left);
      }
      //std::cout << lattice[0](Right).site()(Right).site()(Right).site().get() << std::endl ;
    }
    {
      double default_H[16] = {
        1, 0, 0, 0,
        0, -1, 2, 0,
        0, 2, -1, 0,
        0, 0, 0, 1
      };
      hamiltonian = RawTensor( {Phy1, Phy2, Phy3, Phy4}, {2, 2, 2, 2});
      double* hamiltonian_data = hamiltonian.get();
      for (int i=0; i<16; i++) {
        hamiltonian_data[i] = default_H[i];
      }
      hamiltonian /= 4;
    } // hamiltonian
    {
      double default_I[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      };
      identity = RawTensor({Phy1, Phy2, Phy3, Phy4}, {2, 2, 2, 2});
      double* identity_data = identity.get();
      for (int i=0; i<16; i++) {
        identity_data[i] = default_I[i];
      }
    } // identity
  }

  void set_random_state(unsigned seed) {
    std::srand(seed);
    for (auto& i : lattice) {
      i.tensor().set_random(random);
      std::cout << i.tensor() << std::endl;
      std::cout << std::endl << &i << " ";
      for (auto& j : i.neighbor) {
        std::cout << &j.second.site() << " ";
        std::cout << j.second.env() << "  ";
      }
      std::cout << std::endl;
    }
  }
};

int main() {
  MPS mps(4, 4);
  mps.set_random_state(42);
}
