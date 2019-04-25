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

#include <TAT.hpp>

#include <iomanip>

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

TAT::Legs get_leg(int i) {
  return TAT::Legs(std::to_string(i));
}

Tensor calc_total(const std::vector<Tensor>& lattice) {
  Tensor res;
  res = lattice[0].contract(lattice[1], {Right}, {Left}, {{Phy, get_leg(0)}}, {{Phy, get_leg(1)}});
  for (unsigned long i=2; i<lattice.size()-1; i++) {
    res = res.contract(lattice[i], {Right}, {Left}, {}, {{Phy, get_leg(i)}});
  }
  res = Tensor::contract(res, lattice[lattice.size()-1], {Right, Left}, {Left, Right}, {}, {{Phy, get_leg(lattice.size()-1)}});
  return res;
}

double dot(const Tensor& a, const Tensor& b) {
  auto res = Tensor::contract(a, b, a.legs, b.legs, {}, {});
  return *res.get();
}

void qr_to_right(Tensor& left, Tensor& right) {
  auto qr = left.qr({Left, Phy}, Right, Left);
  left = std::move(qr.Q);
  right = right.contract(qr.R, {Left}, {Right});
}

void qr_to_left(Tensor& left, Tensor& right) {
  auto qr = right.qr({Right, Phy}, Left, Right);
  right = std::move(qr.Q);
  left = left.contract(qr.R, {Right}, {Left});
}

void svd_to_left(Tensor& left, Tensor& right, unsigned long cut) {
  auto svd = right.svd({Left}, Right, Left, cut);
  right = std::move(svd.V);
  left = left.contract(svd.U, {Right}, {Left}).multiple(svd.S, {Right});
}

void svd_to_right(Tensor& left, Tensor& right, unsigned long cut) {
  auto svd = left.svd({Right}, Left, Right, cut);
  left = std::move(svd.V);
  right = right.contract(svd.U, {Left}, {Right}).multiple(svd.S, {Left});
}

int cut_by_svd(std::vector<bool> control, int L=4, unsigned long D=4, unsigned long d=4, unsigned long cut=2) {
  // init
  std::vector<Tensor> lattice;
  std::srand(0);
  for (int i=0; i<L; i++) {
    unsigned long l = D;
    unsigned long r = D;
    if (i==0) l=1;
    if (i==L-1) r=1;
    lattice.push_back(std::move(Tensor({Left, Phy, Right}, {l, d, r}).set_random(get_random)));
  }
  // print
  std::cout << "origin data is" << std::endl << lattice << std::endl;
  auto origin = calc_total(lattice);
  std::cout << "origin total vector is" << std::endl;
  std::cout << origin << std::endl << origin.norm<-1>() << std::endl << std::endl;
  // update
  for (unsigned long this_cut=D-1; this_cut>=cut; this_cut--) {
    if (control[D-this_cut-1]) {
      std::clog << "R";
      for (int i=L-1; i>0; i--) {
        qr_to_left(lattice[i-1], lattice[i]);
      }
      for (int i=0; i<L-1; i++) {
        svd_to_right(lattice[i], lattice[i+1], this_cut);
      }
    } else {
      std::clog << "L";
      for (int i=0; i<L-1; i++) {
        qr_to_right(lattice[i], lattice[i+1]);
      }
      for (int i=L-1; i>0; i--) {
        svd_to_left(lattice[i-1], lattice[i], this_cut);
      }
    }
  }
  std::clog << std::endl;
  // print
  std::cout << "cut data is" << std::endl << lattice << std::endl;
  auto after_cut = calc_total(lattice);
  std::cout << "total vector after cut is" << std::endl;
  std::cout << after_cut << std::endl << after_cut.norm<-1>() << std::endl << std::endl;
  auto diff = after_cut - origin;
  std::cout << "diff is" << std::endl;
  std::cout << diff << std::endl << diff.norm<-1>() << std::endl << std::endl;
  auto overlap = dot(after_cut, origin);
  overlap = overlap*overlap;
  overlap = overlap / (dot(after_cut, after_cut) * dot(origin, origin));
  std::cout << "overlay is " << std::setprecision(16) << overlap << std::endl;
  return 0;
}

int cut_by_qr(int L=4, unsigned long D=4, unsigned long d=4, unsigned long cut=2) {
  // init
  std::vector<Tensor> lattice;
  std::srand(0);
  for (int i=0; i<L; i++) {
    unsigned long l = D;
    unsigned long r = D;
    if (i==0) l=1;
    if (i==L-1) r=1;
    lattice.push_back(std::move(Tensor({Left, Phy, Right}, {l, d, r}).set_random(get_random)));
  }
  // print
  std::cout << "origin data is" << std::endl << lattice << std::endl;
  auto origin = calc_total(lattice);
  std::cout << "origin total vector is" << std::endl;
  std::cout << origin << std::endl << origin.norm<-1>() << std::endl << std::endl;
  // update
  std::vector<Tensor> new_lattice;
  // init
  for (int i=0; i<L; i++) {
    unsigned long l = cut;
    unsigned long r = cut;
    if (i==0) l=1;
    if (i==L-1) r=1;
    new_lattice.push_back(std::move(Tensor({Left, Phy, Right}, {l, d, r}).set_random(get_random)));
  }
  // qr right
  for (int i=L-1; i>0; i--) {
    qr_to_left(new_lattice[i-1], new_lattice[i]);
  }
  // calc right_total
  std::map<int, Tensor> left_total, right_total;
  right_total[L-1] = Tensor(1);
  left_total[0] = Tensor(1);
  for (int i=L-2; i>=0; i--) {
    right_total[i] = right_total[i+1]
    .contract(new_lattice[i+1], {Left1}, {Right}, {}, {{Left, Left1}})
    .contract(lattice[i+1], {Left2, Phy}, {Right, Phy}, {}, {{Left, Left2}});
  }
  for (int i=0; i<L; i++) {
    new_lattice[i] = lattice[i]
    .contract(left_total[i], {Left}, {Right2}, {}, {{Right1, Left}})
    .contract(right_total[i], {Right}, {Left2}, {}, {{Left1, Right}})
    .qr({Left, Phy}, Right, Left)
    .Q;
    left_total[i+1] = left_total[i]
    .contract(new_lattice[i], {Right1}, {Left}, {}, {{Right, Right1}})
    .contract(lattice[i], {Right2, Phy}, {Left, Phy}, {}, {{Right, Right2}});
  }
  // print
  std::cout << "cut data is" << std::endl << new_lattice << std::endl;
  auto after_cut = calc_total(new_lattice);
  std::cout << "total vector after cut is" << std::endl;
  std::cout << after_cut << std::endl << after_cut.norm<-1>() << std::endl << std::endl;
  auto diff = after_cut - origin;
  std::cout << "diff is" << std::endl;
  std::cout << diff << std::endl << diff.norm<-1>() << std::endl << std::endl;
  auto overlap = dot(after_cut, origin);
  overlap = overlap*overlap;
  overlap = overlap / (dot(after_cut, after_cut) * dot(origin, origin));
  std::cout << "overlay is " << std::setprecision(16) << overlap << std::endl;
  return 0;
  return 0;
}
int main() {
  // bool control[2] = {true, true}; // true mean right, and false mean left
  cut_by_svd({0, 0});
  cut_by_svd({0, 1});
  cut_by_svd({1, 0});
  cut_by_svd({1, 1});
  //cut_by_qr();
  return 0;
}
