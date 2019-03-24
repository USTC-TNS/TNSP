/* TAT
 * Copyright (C) 2019  Hao Zhang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <iomanip>

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
  using Size=TAT::Size;
  using Tensor=TAT::Tensor<TAT::Device::CPU, double>;

  int L;
  Size D;
  Tensor hamiltonian;
  Tensor identity;
  std::vector<Tensor> lattice;

  std::map<int, Tensor> left_contract;
  std::map<int, Tensor> right_contract;

  static double random() {
    return double(std::rand())/(RAND_MAX)*2-1;
  }

  MPS(int _L, Size _D) : L(_L), D(_D), hamiltonian({}, {}), identity({}, {}) {
    using namespace TAT::legs_name;
    {
      lattice.push_back(Tensor({2, 1, D}, {Phy, Left, Right}));
      for (int i=1; i<L-1; i++) {
        lattice.push_back(Tensor({2, D, D}, {Phy, Left, Right}));
      }
      lattice.push_back(Tensor({2, D, 1}, {Phy, Left, Right}));
    } // lattice
    {
      double default_H[16] = {
        1, 0, 0, 0,
        0, -1, 2, 0,
        0, 2, -1, 0,
        0, 0, 0, 1
      };
      hamiltonian = Tensor({2, 2, 2, 2}, {Phy1, Phy2, Phy3, Phy4});
      double* H = hamiltonian.node.data.get();
      for (int i=0; i<16; i++) {
        H[i] = default_H[i];
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
      identity = Tensor({2, 2, 2, 2}, {Phy1, Phy2, Phy3, Phy4});
      double* I = identity.node.data.get();
      for (int i=0; i<16; i++) {
        I[i] = default_I[i];
      }
    } // identity
  }

  void set_random_state(unsigned seed) {
    std::srand(seed);
    for (auto& i : lattice) {
      i.set_random(random);
    }
  }

  void update(const Tensor& updater) {
    using namespace TAT::legs_name;
    for (int i=0; i<L-1; i++) {
      Tensor big = Tensor::contract(lattice[i], lattice[i+1], {Right}, {Left}, {{Phy, Phy1}}, {{Phy, Phy2}});
      Tensor Big = Tensor::contract(big, updater, {Phy1, Phy2}, {Phy1, Phy2});
      auto svd = Big.svd({Left, Phy3}, Right, Left, D);
      lattice[i] = std::move(svd.U);
      lattice[i].legs_rename({{Phy3, Phy}});
      lattice[i+1] = svd.V.multiple(svd.S, Left);
      lattice[i+1].legs_rename({{Phy4, Phy}});
    }
    for (int i=L-1; i>0; i--) {
      auto big = Tensor::contract(lattice[i], lattice[i-1], {Left}, {Right}, {{Phy, Phy1}}, {{Phy, Phy2}});
      auto Big = Tensor::contract(big, updater, {Phy1, Phy2}, {Phy1, Phy2});
      auto svd = Big.svd({Right, Phy3}, Left, Right, D);
      lattice[i] = std::move(svd.U);
      lattice[i].legs_rename({{Phy3, Phy}});
      lattice[i-1] = svd.V.multiple(svd.S, Right);
      lattice[i-1].legs_rename({{Phy4, Phy}});
    }
    for (int i=0; i<L; i++) {
      lattice[i] /= lattice[i].norm<-1>();
    }
  }

  void pre() {
    using namespace TAT::legs_name;
    for (int i=L-1; i>1; i--) {
      auto qr = lattice[i].qr({Phy, Right}, Left, Right);
      lattice[i] = std::move(qr.Q);
      lattice[i-1] = Tensor::contract(lattice[i-1], qr.R, {Right}, {Left});
    }
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
    std::cout << "\n";
  }

  double energy_at_i_and_i_plus_1(int i) {
    using namespace TAT::legs_name;
    auto psi = Tensor::contract(lattice[i], lattice[i+1], {Right}, {Left}, {{Phy, Phy1}}, {{Phy, Phy2}});
    auto Hpsi = Tensor::contract(psi, hamiltonian, {Phy1, Phy2}, {Phy1, Phy2});
    auto psiHpsi = Tensor::contract(Hpsi, psi, {Phy3, Phy4}, {Phy1, Phy2}, {{Left, Left1}, {Right, Right1}}, {{Left, Left2}, {Right, Right2}});
    auto leftpsiHpsi = Tensor::contract(psiHpsi, left_contract[i-1], {Left1, Left2}, {Right1, Right2});
    auto res = Tensor::contract(leftpsiHpsi, right_contract[i+2], {Right1, Right2}, {Left1, Left2});
    return *res.get();
  }

  void prepare_aux() {
    using namespace TAT::legs_name;
    left_contract[-1] = Tensor({1, 1}, {Right1, Right2});
    *left_contract[-1].get() = 1;
    right_contract[L] = Tensor({1, 1}, {Left1, Left2});
    *right_contract[L].get() = 1;
    for (int i=0; i<=L-1; i++) {
      left_contract[i] = Tensor::contract(left_contract[i-1], Tensor::contract(lattice[i], lattice[i], {Phy}, {Phy}, {{Left, Left1}, {Right, Right1}}, {{Left, Left2}, {Right, Right2}}), {Right1, Right2}, {Left1, Left2});
    }
    for (int i=L-1; i>=0; i--) {
      right_contract[i] = Tensor::contract(right_contract[i+1], Tensor::contract(lattice[i], lattice[i], {Phy}, {Phy}, {{Left, Left1}, {Right, Right1}}, {{Left, Left2}, {Right, Right2}}), {Left1, Left2}, {Right1, Right2});
    }
  }

  double energy() {
    prepare_aux();
    double total = 0;
    for (int i=0; i<L-1; i++) {
      total += energy_at_i_and_i_plus_1(i);
    }
    total /= *left_contract[L-1].get();
    return total/L;
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
  MPS mps(100, 10);
  mps.set_random_state(42);
  mps.update(100, 100, 0.01);
  return 0;
}
