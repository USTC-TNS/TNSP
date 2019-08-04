/* example/Heisenberg_MPS_SU.cpp
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

class MPS {
   public:
      int L;
      TAT::Size D;
      TAT::Size d;
      Node hamiltonian;
      Node identity;
      std::vector<Node> lattice;
      Node energy;

      MPS(int L, TAT::Size D, const std::vector<double>& hamiltonian_vector) :
            L(L), D(D), d(sqrt(sqrt(hamiltonian_vector.size()))) {
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
      }

      void generate_lattice() {
            lattice.push_back(Node({Phy, Left, Right}, {2, 1, D}));
            for (int i = 1; i < L - 1; i++) {
                  lattice.push_back(Node({Phy, Left, Right}, {2, D, D}));
            }
            lattice.push_back(Node({Phy, Left, Right}, {2, D, 1}));
      }

      void calc_energy() {
            auto left_contract = std::map<int, Node>();
            auto right_contract = std::map<int, Node>();
            left_contract[-1] = Node({Right1, Right2}, {1, 1}).set([]() { return 1; });
            right_contract[L] = Node({Left1, Left2}, {1, 1}).set([]() { return 1; });
            for (int i = 0; i <= L - 1; i++) {
                  left_contract[i] = Node::contract(
                        left_contract[i - 1],
                        Node::contract(
                              lattice[i],
                              lattice[i],
                              {Phy},
                              {Phy},
                              {{Left, Left1}, {Right, Right1}},
                              {{Left, Left2}, {Right, Right2}}),
                        {Right1, Right2},
                        {Left1, Left2},
                        {},
                        {});
            }
            for (int i = L - 1; i >= 0; i--) {
                  right_contract[i] = Node::contract(
                        right_contract[i + 1],
                        Node::contract(
                              lattice[i],
                              lattice[i],
                              {Phy},
                              {Phy},
                              {{Left, Left1}, {Right, Right1}},
                              {{Left, Left2}, {Right, Right2}}),
                        {Left1, Left2},
                        {Right1, Right2},
                        {},
                        {});
            }
            energy = Node(0);
            for (int i = 0; i < L - 1; i++) {
                  auto psi = Node::contract(lattice[i], lattice[i + 1], {Right}, {Left}, {{Phy, Phy1}}, {{Phy, Phy2}});
                  auto Hpsi = Node::contract(psi, hamiltonian, {Phy1, Phy2}, {Phy1, Phy2}, {}, {});
                  auto psiHpsi = Node::contract(
                        Hpsi,
                        psi,
                        {Phy3, Phy4},
                        {Phy1, Phy2},
                        {{Left, Left1}, {Right, Right1}},
                        {{Left, Left2}, {Right, Right2}});
                  auto leftpsiHpsi =
                        Node::contract(psiHpsi, left_contract[i - 1], {Left1, Left2}, {Right1, Right2}, {}, {});
                  auto res =
                        Node::contract(leftpsiHpsi, right_contract[i + 2], {Right1, Right2}, {Left1, Left2}, {}, {});
                  energy = energy + res;
            }
            energy = energy / left_contract[L - 1] / L;
      }

      void set_random_state(std::function<double()> setter) {
            for (auto& i : lattice) {
                  i.set(setter);
            }
      }

      void update(int total_step, int log_interval, double delta_t) {
            prepare();
            auto updater = identity - delta_t * hamiltonian;
            for (int i = 1; i <= total_step; i++) {
                  update_once(updater);
                  // std::cout << *this << std::endl;
                  std::cout << i << "\r" << std::flush;
                  if (log_interval == 0 || i % log_interval == 0) {
                        // std::cout << rang::fg::red << " Current Lattice is " << rang::fg::reset << *this <<
                        // "\n";
                        std::cout << std::setprecision(12) << energy.value().at({{Right1, 0}, {Right2, 0}})
                                  << std::endl;
                  }
            }
      }

      // qr to left
      void prepare() {
            auto qr_graph = TAT::graph::QRCanonicalizationGraph<double>(Left, Right);
            for (int i = L - 1; i > 1; i--) {
                  qr_graph(lattice[i], lattice[i - 1]);
            }
      }

      void update_once(Node updater) {
            auto to_left = TAT::graph::Dim1SVDGraph<double>(updater, D, true);
            auto to_right = TAT::graph::Dim1SVDGraph<double>(updater, D, false);
            for (int i = 0; i < L - 1; i++) {
                  to_right(lattice[i], lattice[i + 1]);
            }
            for (int i = L - 2; i >= 0; i--) {
                  to_left(lattice[i], lattice[i + 1]);
            }
      }

      friend std::ostream& operator<<(std::ostream& out, const MPS& mps) {
            out << "{" << rang::fgB::cyan << "\"L\": " << mps.L << rang::fg::reset << ", " << rang::fgB::cyan
                << "\"D\": " << mps.D << rang::fg::reset << ", " << rang::fgB::cyan << "\"d\": " << mps.d
                << rang::fg::reset << ", lattice\": [";
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

int main(int argc, char** argv) {
      std::ios::sync_with_stdio(false);
      args::ArgumentParser parser(
            "Heisenberg_MPS_SU " TAT_VERSION " (compiled " __DATE__ " " __TIME__
            ")\n"
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
      args::ValueFlag<int> print_interval(
            parser, "T", "print energy every T step [default: 100]", {'T', "print_interval"}, 100);
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

      auto mps =
            MPS(args::get(length),
                args::get(dimension),
                {1 / 4., 0, 0, 0, 0, -1 / 4., 2 / 4., 0, 0, 2 / 4., -1 / 4., 0, 0, 0, 0, 1 / 4.}); // L = 3, D = 2, d=2
      auto engine = std::default_random_engine(args::get(random_seed));
      auto dist = std::uniform_real_distribution<double>(-1, 1);
      auto generator = [&dist, &engine]() { return dist(engine); };
      mps.set_random_state(generator);
      mps.update(args::get(step_num), args::get(print_interval), args::get(step_size));
      std::cout << mps << std::endl;

      return 0;
}
