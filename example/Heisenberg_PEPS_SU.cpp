/* example/Heisenberg_PEPS_SU.cpp
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

#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include <args.hxx>

#define TAT_DEFAULT
#include <TAT.hpp>

using Node = TAT::LazyNode<double>;
using namespace TAT::legs_name;

class PEPS {
   public:
      int L1;
      int L2;
      TAT::Size D;
      TAT::Size d;

      Node hamiltonian;
      Node identity;

      std::map<std::tuple<int, int>, Node> lattice;
      std::map<std::tuple<int, int, TAT::Legs>, Node> env;
      Node energy;

      PEPS(int L1, int L2, TAT::Size D, const std::vector<double>& hamiltonian_vector) :
            L1(L1), L2(L2), D(D), d(sqrt(sqrt(hamiltonian_vector.size()))) {
            hamiltonian = Node({Phy1, Phy2, Phy3, Phy4}, {2, 2, 2, 2}).set([&hamiltonian_vector]() {
                  static int pos = 0;
                  return hamiltonian_vector[pos++];
            });
            identity = Node({Phy1, Phy2, Phy3, Phy4}, {2, 2, 2, 2}).set([this]() {
                  static int pos = 0;
                  return (pos++) % (d * d + 1) == 0;
            });
            generate_lattice();
            calc_energy();
            // TAT::lazy::eager = true;
      }

      void generate_lattice() {
            auto legs_generator = [this](int i, int j) {
                  auto res = std::vector<TAT::Legs>{Phy};
                  if (i != 0) {
                        res.push_back(Up);
                  }
                  if (i != L1 - 1) {
                        res.push_back(Down);
                  }
                  if (j != 0) {
                        res.push_back(Left);
                  }
                  if (j != L2 - 1) {
                        res.push_back(Right);
                  }
                  return res;
            };
            auto dims_generator = [this](int i, int j) {
                  auto res = std::vector<TAT::Size>{2};
                  if (i != 0) {
                        res.push_back(D);
                  }
                  if (i != L1 - 1) {
                        res.push_back(D);
                  }
                  if (j != 0) {
                        res.push_back(D);
                  }
                  if (j != L2 - 1) {
                        res.push_back(D);
                  }
                  return res;
            };
            for (int i = 0; i < L1; i++) {
                  for (int j = 0; j < L2; j++) {
                        lattice[{i, j}] = Node(legs_generator(i, j), dims_generator(i, j));
                  }
            }
            for (int i = 0; i < L1; i++) {
                  env[{i, -1, Right}] = Node({Phy}, {1}).set([]() { return 1; });
                  env[{i, L2 - 1, Right}] = Node({Phy}, {1}).set([]() { return 1; });
            }
            for (int i = 0; i < L1; i++) {
                  for (int j = 0; j < L2 - 1; j++) {
                        env[{i, j, Right}] = Node({Phy}, {D}).set([]() { return 1; });
                  }
            }
            for (int j = 0; j < L2; j++) {
                  env[{-1, j, Down}] = Node({Phy}, {1}).set([]() { return 1; });
                  env[{L1 - 1, j, Down}] = Node({Phy}, {1}).set([]() { return 1; });
            }
            for (int i = 0; i < L1 - 1; i++) {
                  for (int j = 0; j < L2; j++) {
                        env[{i, j, Down}] = Node({Phy}, {D}).set([]() { return 1; });
                  }
            }
      }

      void calc_energy() {
            auto down_leg = [](int i) {
                  return TAT::Legs(std::string("Leg_For_Calc_Energy_Down_") + std::to_string(i));
            };
            auto phy_leg = [](int i, int j) {
                  return TAT::Legs(
                        std::string("Leg_For_Calc_Energy_Phy_") + std::to_string(i) + "_" + std::to_string(j));
            };

            auto total_leg = std::vector<TAT::Legs>();
            auto total_size = std::vector<TAT::Size>();
            for (int i = 0; i < L1; i++) {
                  for (int j = 0; j < L2; j++) {
                        total_leg.push_back(phy_leg(i, j));
                        total_size.push_back(d);
                  }
            }

            // auto psi = lattice[{0, 0}];
            auto psi = lattice[{0, 0}]
                             .multiple(env[{0, 0, Down}], Down)
                             .multiple(env[{0, 0, Right}], Right)
                             .legs_rename({{Down, down_leg(0)}, {Phy, phy_leg(0, 0)}});
            for (int i = 0; i < L1; i++) {
                  for (int j = 0; j < L2; j++) {
                        if (i == 0 && j == 0) {
                              continue;
                        }
                        psi = Node::contract(
                              psi,
                              lattice[{i, j}].multiple(env[{i, j, Down}], Down).multiple(env[{i, j, Right}], Right),
                              {Right, down_leg(j)},
                              {Left, Up},
                              {},
                              {{Down, down_leg(j)}, {Phy, phy_leg(i, j)}});
                  }
            }
            psi = psi.transpose(total_leg);
            auto second = L2 == 1 ? phy_leg(1, 0) : phy_leg(0, 1);
            auto H_psi = Node::contract(
                               psi,
                               hamiltonian,
                               {phy_leg(0, 0), second},
                               {Phy1, Phy2},
                               {},
                               {{Phy3, phy_leg(0, 0)}, {Phy4, second}})
                               .transpose(total_leg);
            for (int i = 0; i < L1; i++) {
                  for (int j = 0; j < L2 - 1; j++) {
                        if (i == 0 && j == 0) {
                              continue;
                        }
                        H_psi += Node::contract(
                                       psi,
                                       hamiltonian,
                                       {phy_leg(i, j), phy_leg(i, j + 1)},
                                       {Phy1, Phy2},
                                       {},
                                       {{Phy3, phy_leg(i, j)}, {Phy4, phy_leg(i, j + 1)}})
                                       .transpose(total_leg);
                  }
            }
            for (int i = 0; i < L1 - 1; i++) {
                  for (int j = 0; j < L2; j++) {
                        if (i == 0 && j == 0 && L2 == 1) {
                              continue;
                        }
                        H_psi += Node::contract(
                                       psi,
                                       hamiltonian,
                                       {phy_leg(i, j), phy_leg(i + 1, j)},
                                       {Phy1, Phy2},
                                       {},
                                       {{Phy3, phy_leg(i, j)}, {Phy4, phy_leg(i + 1, j)}})
                                       .transpose(total_leg);
                  }
            }
            energy = Node::contract(psi, H_psi, total_leg, total_leg) / Node::contract(psi, psi, total_leg, total_leg) /
                     L1 / L2;
      }

      void set_random_state(std::function<double()> setter) {
            for (int i = 0; i < L1; i++) {
                  for (int j = 0; j < L2; j++) {
                        lattice[{i, j}].set(setter);
                  }
            }
      }

      void update(int total_step, int log_interval, double delta_t) {
            auto updater = identity - delta_t * hamiltonian;
            for (int i = 1; i <= total_step; i++) {
                  update_once(updater);
                  // std::cout << *this << std::endl;
                  std::cout << i << "\r" << std::flush;
                  if (log_interval == 0 || i % log_interval == 0) {
                        std::cout << std::setprecision(12) << energy.value().at({}) << std::endl;
                  }
            }
      }

      void update_once(Node updater) {
            auto do_svd_right =
                  TAT::graph::Dim2SVDGraph<double, 3, 3>(Right, Left, {Left, Up, Down}, {Right, Up, Down}, D);
            auto do_svd_down =
                  TAT::graph::Dim2SVDGraph<double, 3, 3>(Down, Up, {Up, Left, Right}, {Down, Left, Right}, D);
            for (int i = 0; i < L1; i++) {
                  for (int j = 0; j < L2 - 1; j++) {
                        do_svd_right(
                              updater,
                              lattice[{i, j}],
                              lattice[{i, j + 1}],
                              env[{i, j, Right}],
                              {env[{i, j - 1, Right}], env[{i - 1, j, Down}], env[{i, j, Down}]},
                              {env[{i, j + 1, Right}], env[{i - 1, j + 1, Down}], env[{i, j + 1, Down}]});
                  }
            }
            for (int i = 0; i < L1 - 1; i++) {
                  for (int j = 0; j < L2; j++) {
                        do_svd_down(
                              updater,
                              lattice[{i, j}],
                              lattice[{i + 1, j}],
                              env[{i, j, Down}],
                              {env[{i - 1, j, Down}], env[{i, j - 1, Right}], env[{i, j, Right}]},
                              {env[{i + 1, j, Down}], env[{i + 1, j - 1, Right}], env[{i + 1, j, Right}]});
                  }
            }
      }

      friend std::ostream& operator<<(std::ostream& out, const PEPS& peps) {
            out << "{" << rang::fgB::cyan << "\"L1\": " << peps.L1 << rang::fg::reset << ", " << rang::fgB::cyan
                << "\"L2\": " << peps.L2 << rang::fg::reset << ", " << rang::fgB::cyan << "\"D\": " << peps.D
                << rang::fg::reset << ", " << rang::fgB::cyan << "\"d\": " << peps.d << rang::fg::reset
                << ", lattice\": [";
            bool flag = false;
            for (int i = 0; i < peps.L1; i++) {
                  for (int j = 0; j < peps.L2; j++) {
                        if (flag) {
                              out << ", ";
                        }
                        out << peps.lattice.at({i, j});
                        flag = true;
                  }
            }
            out << "]}";
            return out;
      }
};

int main(int argc, char** argv) {
      std::ios::sync_with_stdio(false);
      args::ArgumentParser parser(
            "Heisenberg_PEPS_SU " TAT_VERSION " (compiled " __DATE__ " " __TIME__
            ")\n"
            "Copyright (C) 2019  Hao Zhang<zh970205@mail.ustc.edu.cn>\n"
            "This program comes with ABSOLUTELY NO WARRANTY. "
            "This is free software, and you are welcome to redistribute it "
            "under the terms and conditions of the GNU General Public License. "
            "See http://www.gnu.org/copyleft/gpl.html for details.",
            "Simple Update in MPS of Heisenberg Model.");
      args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
      args::Flag version(parser, "version", "Display the version", {'v', "version"});
      args::ValueFlag<int> length1(parser, "L1", "system size [default: 3]", {'A', "length1"}, 3);
      args::ValueFlag<int> length2(parser, "L2", "system size [default: 3]", {'B', "length2"}, 3);
      args::ValueFlag<unsigned long> dimension(parser, "D", "bond dimension [default: 4]", {'D', "dimension"}, 4);
      args::ValueFlag<unsigned> random_seed(parser, "S", "random seed [default: 0]", {'S', "random_seed"}, 0);
      args::ValueFlag<int> step_num(parser, "N", "total step to run [default: 100]", {'N', "step_num"}, 100);
      args::ValueFlag<int> print_interval(
            parser, "T", "print energy every T step [default: 100]", {'T', "print_interval"}, 100);
      args::ValueFlag<double> step_size(parser, "I", "step size when update [default: 0.1]", {'I', "step_size"}, 0.1);
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

      auto peps =
            PEPS(args::get(length1),
                 args::get(length2),
                 args::get(dimension),
                 {1 / 4., 0, 0, 0, 0, -1 / 4., 2 / 4., 0, 0, 2 / 4., -1 / 4., 0, 0, 0, 0, 1 / 4.}); // L = 3, D = 4, d=2
      auto engine = std::default_random_engine(args::get(random_seed));
      auto dist = std::uniform_real_distribution<double>(-1, 1);
      auto generator = [&dist, &engine]() { return dist(engine); };
      peps.set_random_state(generator);
      peps.update(args::get(step_num), args::get(print_interval), args::get(step_size));
      std::cout << peps << std::endl;

      return 0;
}
