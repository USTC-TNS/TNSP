/* TAT/Heisenberg_MPS_SU.cpp
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

#include <TAT.hpp>

struct MPS {
  using Size=TAT::Size;
  using Tensor=TAT::Tensor<TAT::Device::CPU, double>;
  using Site=TAT::Site<TAT::Device::CPU, double>;

  int L;
  Size D;
  Tensor hamiltonian;
  Tensor identity;
  std::vector<Site> lattice;

  std::vector<Site> psipsiUp;
  std::vector<Site> psipsiDown;
  std::vector<Site> psipsiLeft;
  std::vector<Site> psipsiRight;

  std::map<int, Tensor> left_contract;
  std::map<int, Tensor> right_contract;

  static double random() {
    return double(std::rand())/(RAND_MAX)*2-1;
  }

  void set_random_state(unsigned seed) {
    std::srand(seed);
    for (int i=0; i<L; i++) {
      lattice[i].set_random(random);
    }
  }

  MPS(int _L, Size _D) : L(_L), D(_D), hamiltonian({}, {}), identity({}, {}) {
    using namespace TAT::legs_name;
    {
      lattice.push_back(Site::make_site({Phy, Left, Right}, {2, 1, D}));
      for (int i=1; i<L-1; i++) {
        lattice.push_back(Site::make_site({Phy, Left, Right}, {2, D, D}));
      }
      lattice.push_back(Site::make_site({Phy, Left, Right}, {2, D, 1}));
    } // lattice
    {
      for (int i=0; i<L-1; i++) {
        Site::link(lattice[i], Right, lattice[i+1], Left);
      }
    } // link
    {
      double default_H[16] = {
        1, 0, 0, 0,
        0, -1, 2, 0,
        0, 2, -1, 0,
        0, 0, 0, 1
      };
      hamiltonian = Tensor({Phy1, Phy2, Phy3, Phy4}, {2, 2, 2, 2});
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
      identity = Tensor({Phy1, Phy2, Phy3, Phy4}, {2, 2, 2, 2});
      double* identity_data = identity.get();
      for (int i=0; i<16; i++) {
        identity_data[i] = default_I[i];
      }
    } // identity
  }

  void update(const Tensor& updater) {
    using namespace TAT::legs_name;
    for (int i=0; i<L-1; i++) {
      lattice[i].update_to(lattice[i+1], Right, Left, D, updater, {});
    }
    for (int i=L-1; i>0; i--) {
      lattice[i].update_to(lattice[i-1], Left, Right, D, updater, {});
    }
    for (int i=0; i<L; i++) {
      lattice[i].normalize<-1>();
    }
  }

  void pre() {
    using namespace TAT::legs_name;
    for (int i=L-1; i>1; i--) {
      lattice[i].qr_to(lattice[i-1], Left, Right);
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
  }

  double energy_at_i_and_i_plus_1(int i) {
    using namespace TAT::legs_name;
    auto psi = Tensor::contract(lattice[i].tensor(), lattice[i+1].tensor(), {Right}, {Left}, {{Phy, Phy1}}, {{Phy, Phy2}});
    auto Hpsi = Tensor::contract(psi, hamiltonian, {Phy1, Phy2}, {Phy1, Phy2}, {}, {});
    auto psiHpsi = Tensor::contract(Hpsi, psi, {Phy3, Phy4}, {Phy1, Phy2}, {{Left, Left1}, {Right, Right1}}, {{Left, Left2}, {Right, Right2}});
    auto leftpsiHpsi = Tensor::contract(psiHpsi, left_contract[i-1], {Left1, Left2}, {Right1, Right2}, {}, {});
    auto res = Tensor::contract(leftpsiHpsi, right_contract[i+2], {Right1, Right2}, {Left1, Left2}, {}, {});
    return *res.get();
  }

  void prepare_aux() {
    using namespace TAT::legs_name;
    left_contract[-1] = Tensor({Right1, Right2}, {1, 1});
    *left_contract[-1].get() = 1;
    right_contract[L] = Tensor({Left1, Left2}, {1, 1});
    *right_contract[L].get() = 1;
    for (int i=0; i<=L-1; i++) {
      left_contract[i] = Tensor::contract(left_contract[i-1], Tensor::contract(lattice[i].tensor(), lattice[i].tensor(), {Phy}, {Phy}, {{Left, Left1}, {Right, Right1}}, {{Left, Left2}, {Right, Right2}}), {Right1, Right2}, {Left1, Left2}, {}, {});
    }
    for (int i=L-1; i>=0; i--) {
      right_contract[i] = Tensor::contract(right_contract[i+1], Tensor::contract(lattice[i].tensor(), lattice[i].tensor(), {Phy}, {Phy}, {{Left, Left1}, {Right, Right1}}, {{Left, Left2}, {Right, Right2}}), {Left1, Left2}, {Right1, Right2}, {}, {});
    }

    /*
    for (int i=0; i<L; i++) {
      psipsiUp[i] = lattice[i];
      psipsiDown[i] = lattice[i];
      Site::link(psipsiUp[i], Phy, psipsiDown[i], Phy);
    }
    for (int i=0; i<L-1; i++) {
      Site::link(psipsiUp[i], Right, psipsiUp[i+1], Left);
      Site::link(psipsiDown[i], Right, psipsiDown[i+1], Left);
    }
    */
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

  friend std::ostream& operator<<(std::ostream& out, const MPS& mps) {
    out << "{" << rang::fgB::cyan << "\"L\": " << mps.L << rang::fg::reset << ", " << rang::fgB::cyan << "\"D\": " << mps.D << rang::fg::reset << ", \"lattice\": [";
    bool flag = false;
    for (auto& i : mps.lattice) {
      if (flag) {
        out << ", ";
      }
      out << i;
      flag = true;
    }
    out << "]}";
    return out;
  }
};


void Heisenberg_MPS(int L, unsigned long D, unsigned seed, int step, int print_step, double delta_t) {
  MPS mps(L, D);
  mps.set_random_state(seed);
  mps.update(step, print_step, delta_t);
  std::cout << mps << std::endl;
}

int main(int argc, char** argv) {
  args::ArgumentParser parser(
    "Heisenberg_MPS_SU " TAT_VERSION " (compiled " __DATE__ " " __TIME__ ")\n"
    "Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>\n"
    "This program comes with ABSOLUTELY NO WARRANTY. "
    "This is free software, and you are welcome to redistribute it "
    "under the terms and conditions of the GNU General Public License. "
    "See http://www.gnu.org/copyleft/gpl.html for details.",
    "Simple Update in MPS of Heisenberg Model.");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::Flag version(parser, "version", "Display the version", {'v', "version"});
  args::ValueFlag<int> length(parser, "L", "system size [default: 100]", {'L', "length"}, 100);
  args::ValueFlag<unsigned long> dimension(parser, "D", "bond dimension [default: 12]", {'D', "dimension"}, 12);
  args::ValueFlag<unsigned> random_seed(parser, "S", "random seed [default: 42]", {'S', "random_seed"}, 42);
  args::ValueFlag<int> step_num(parser, "N", "total step to run [default: 100]", {'N', "step_num"}, 100);
  args::ValueFlag<int> print_inteval(parser, "T", "print energy every T step [default: 100]", {'T', "print_inteval"}, 100);
  args::ValueFlag<double> step_size(parser, "I", "step size when update [default: 0.01]", {'I', "step_size"}, 0.01);
  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Help& h) {
    std::cout << parser;
    return 0;
  } catch (const args::ParseError& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  } catch (const args::ValidationError& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }
  if (version) {
    std::cout << "Heisenberg_MPS " TAT_VERSION << std::endl;
    return 0;
  }
  Heisenberg_MPS(args::get(length), args::get(dimension), args::get(random_seed), args::get(step_num), args::get(print_inteval), args::get(step_size));
  return 0;
}
