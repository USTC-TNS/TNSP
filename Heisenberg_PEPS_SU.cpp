
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
  using Legs=TAT::Legs
  using Tensor=TAT::Tensor<TAT::Device::CPU, double>;

  int L1;
  int L2;
  Size d;
  Size D;
  Tensor hamiltonian;
  Tensor identity;

  enum class Direction {Right, Down};
  std::map<std::tuple<int, int>, Tensor> lattice;
  std::map<std::tuple<int, int, TAT::Legs>, Tensor> env;

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
          env[ {i, j, Right}] = Tensor({D}, {Phy});
          env[ {i, j, Right}].set_constant(1);
          std::cout << env[ {i, j, Right}];
        }
      }
      for (int i=0; i<L1-1; i++) {
        for (int j=0; j<L2; j++) {
          env[ {i, j, Down}] = Tensor({D}, {Phy});
          env[ {i, j, Down}].set_constant(1);
          std::cout << env[ {i, j, Down}];
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

  void update(const Tensor& updater) {
    using namespace TAT::legs_name;
    for (int i=0; i<L1; i++) {
      for (int j=0; j<L2-1; j++) {
        Tensor t1 = cal_neighbor(i, j, Right);
        Tensor t2 = cal_neighbor(i, j+1, Phy);
        Tensor Big = t1.contract(t2, {Right}, {Left}, {{Up, Up1},{Down, Down1},{Phy,Phy1}}, {{Up, Up2}, {Down, Down2}, {Phy, Phy2}}).contract(updater, {Phy1, Phy2}, {Phy3, Phy4});
        auto SVD = Big.svd({Left, Up1, Down1, Phy1}, Right, Left, D);
        env[{i,j,Right}] = std::move(SVD.S);
        lattice[{i,j}] = std::move(SVD.U.legs_rename({{Up1, Up},{Down1, Down}}));
      }
    }
    for (int i=0; i<L1-1; i++) {
      for (int j=0; j<L2; j++) {
      }
    }
  }

  std::tuple<Tensor&, Legs> get_neighbor(int i, int j, Legs leg) {
    std::tuple<Tensor&, Legs> res;
    if(i!=0 && leg!=Up) res.push_back({env[{i-1,j,Down}], Up});
    if(j!=0 && leg!=Left) res.push_back({env[{i,j-1,Right}], Left});
    if(i!=L1-1 && leg!=Down) res.push_back({env[{i,j,Down}], Down});
    if(j!=L2-1 && leg!=Right) res.push_back({env[{i,j,Right}], Right});
    return std::move(res);
  }

  Tensor calc_neighbor(int i, int j, Legs leg) {
    auto n = get_neighbor(i, j, leg);
    Tensor res = lattice[{i,j}].multiple(std::get<0>(n[0]), std::get<1>(n[0]));
    for(int i=1;i<n.size();i++) {
      res = res.multiple(std::get<0>(n[i]), std::get<1>(n[i]));
    }
    return std::move(res);
  }

  void update(int n, int t, double delta_t) {
    pre();
    auto updater = identity - delta_t* hamiltonian;
    for (int i=0; i<n; i++) {
      update(updater);
      std::cout << i << "\r" << std::flush;
      if ((i+1)%t==0) {
        std::cout << std::setprecision(12) << energy() << std::endl;
      }
    }
  }

  void pre();

  double energy();
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
