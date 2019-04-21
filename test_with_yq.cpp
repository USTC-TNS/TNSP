/* TAT/test_with_yq.cpp
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

using namespace TAT::legs_name;
using Tensor = TAT::Tensor<TAT::Device::CPU, double>;

double get_random() {
  return double(std::rand())/(RAND_MAX)*2 - 1;
}

std::ostream& operator<<(std::ostream& os, const std::vector<Tensor>& value) {
  for (const auto& i : value) {
    os << i << std::endl;
  }
  return os;
}

int cut_by_svd(int L=4, unsigned long D=4, unsigned long d=4, unsigned long cut=3) {
  std::vector<Tensor> lattice;
  std::srand(0);
  for (int i=0; i<L; i++) {
    unsigned long l = D;
    unsigned long r = D;
    if (i==0) l=1;
    if (i==L-1) r=1;
    lattice.push_back(std::move(Tensor({Left, Right, Phy}, {l, r, d}).set_random(get_random)));
  }
  std::cout << "origin data is" << std::endl << lattice << std::endl;
  auto origin = lattice[0];
  origin.legs_rename({{Phy, TAT::Legs(std::to_string(0))}});
  for (int i=1; i<L-1; i++) {
    origin = origin.contract(lattice[i], {Right}, {Left}, {}, {{Phy, TAT::Legs(std::to_string(i))}});
  }
  origin = Tensor::contract(origin, lattice[L-1], {Right, Left}, {Left, Right}, {}, {{Phy, TAT::Legs(std::to_string(L-1))}});
  std::cout << "origin total vector is" << std::endl;
  std::cout << origin << std::endl << origin.norm<-1>() << std::endl << std::endl;
  auto bak = std::move(origin);
  for (int i=0; i<L-1; i++) {
    auto qr = lattice[i].qr({Left, Phy}, Right, Left);
    lattice[i] = std::move(qr.Q);
    lattice[i+1] = lattice[i+1].contract(qr.R, {Left}, {Right});
  }
  for (int i=L-1; i>0; i--) {
    auto svd = lattice[i].svd({Left}, Right, Left, cut);
    lattice[i] = std::move(svd.V);
    lattice[i-1] = lattice[i-1].contract(svd.U, {Right}, {Left}).multiple(svd.S, {Right});
  }
  std::cout << "cut data is" << std::endl << lattice << std::endl;
  origin = lattice[0];
  origin.legs_rename({{Phy, TAT::Legs(std::to_string(0))}});
  for (int i=1; i<L-1; i++) {
    origin = origin.contract(lattice[i], {Right}, {Left}, {}, {{Phy, TAT::Legs(std::to_string(i))}});
  }
  origin = Tensor::contract(origin, lattice[L-1], {Right, Left}, {Left, Right}, {}, {{Phy, TAT::Legs(std::to_string(L-1))}});
  std::cout << "total vector after cut is" << std::endl;
  std::cout << origin << std::endl << origin.norm<-1>() << std::endl << std::endl;
  origin = origin - bak;
  std::cout << "diff is" << std::endl;
  std::cout << origin << std::endl << origin.norm<-1>() << std::endl << std::endl;
  return 0;
}

int main() {
  cut_by_svd();
  return 0;
}
