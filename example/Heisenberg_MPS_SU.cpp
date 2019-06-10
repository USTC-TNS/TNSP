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
#include <random>

#define TAT_DEFAULT
#include <TAT.hpp>

using Node = TAT::LazyNode<double>;
using namespace TAT::legs_name;
using TAT::Size;

class QRCanonicalizationGraph {
   public:
      Node to_split;
      Node to_absorb;
      Node splited;
      Node absorbed;
      QRCanonicalizationGraph(TAT::Legs split_leg, TAT::Legs absorb_leg) {
            auto qr = to_split.rq({split_leg}, absorb_leg, split_leg);
            splited = qr.Q;
            absorbed = Node::contract(qr.R, to_absorb, {split_leg}, {absorb_leg});
      }
};

class SVDGraph {
   public:
      Node old_A;
      Node old_B;
      Node new_A;
      Node new_B;
      SVDGraph(Node H, int cut, bool left = true) {
            auto big = Node::contract(old_A, old_B, {Right}, {Left}, {{Phy, Phy1}}, {{Phy, Phy2}});
            auto Big = Node::contract(big, H, {Phy1, Phy2}, {Phy3, Phy4});
            Big = Big / Big.norm<-1>();
            auto svd = Big.svd({Phy1, Left}, Right, Left, cut);
            new_A = svd.U.legs_rename({{Phy1, Phy}});
            new_B = svd.V.legs_rename({{Phy2, Phy}});
            if (left) {
                  new_A = new_A.multiple(svd.S, Right);
            } else {
                  new_B = new_B.multiple(svd.S, Left);
            }
      }
};

class MPS {
   public:
      int L;
      Size D;
      Size d;
      Node hamiltonian;
      Node identity;
      std::vector<Node> lattice;

      MPS(int L, Size D, const std::vector<double>& hamiltonian_vector) :
            L(L), D(D), d(sqrt(sqrt(hamiltonian_vector.size()))) {
            lattice.push_back(Node({Phy, Left, Right}, {2, 1, D}));
            for (int i = 1; i < L - 1; i++) {
                  lattice.push_back(Node({Phy, Left, Right}, {2, D, D}));
            }
            lattice.push_back(Node({Phy, Left, Right}, {2, D, 1}));
            hamiltonian = Node({Phy1, Phy2, Phy3, Phy4}, {2, 2, 2, 2}).set([&hamiltonian_vector]() {
                  static int pos = 0;
                  return hamiltonian_vector[pos++];
            });
            identity = Node({Phy1, Phy2, Phy3, Phy4}, {2, 2, 2, 2}).set([this]() {
                  static int pos = 0;
                  return (pos++) % (d + 1) == 0;
            });
      }

      void set_random_state(std::function<double()> setter) {
            for (auto& i : lattice) {
                  i.set(setter);
            }
      }

      void update(int total_step, int log_interval, double delta_t) {
            prepare();
            std::cout << *this << std::endl;
            auto updater = identity - delta_t * hamiltonian;
            for (int i = 1; i <= total_step; i++) {
                  update_once(updater);
                  std::cout << *this << std::endl;
                  // std::cout << i << "\r" << std::flush;
                  // if (i % log_interval == 0) {
                  //      std::cout << std::setprecision(12) << energy() << std::endl;
                  //}
            }
      }

      // qr to left
      void prepare() {
            auto qr_graph = QRCanonicalizationGraph(Left, Right);
            for (int i = L - 1; i > 1; i--) {
                  qr_graph.to_split.set_value(lattice[i].pop());
                  qr_graph.to_absorb.set_value(lattice[i - 1].pop());
                  lattice[i].set_value(qr_graph.splited.pop());
                  lattice[i - 1].set_value(qr_graph.absorbed.pop());
            }
      }

      void update_once(Node updater) {
            auto to_left = SVDGraph(updater, D, true);
            auto to_right = SVDGraph(updater, D, false);
            for (int i = 0; i < L - 1; i++) {
                  to_right.old_A.set_value(lattice[i].pop());
                  to_right.old_B.set_value(lattice[i + 1].pop());
                  lattice[i].set_value(to_right.new_A.pop());
                  lattice[i + 1].set_value(to_right.new_B.pop());
            }
            for (int i = L - 2; i >= 0; i--) {
                  to_left.old_A.set_value(lattice[i].pop());
                  to_left.old_B.set_value(lattice[i + 1].pop());
                  lattice[i].set_value(to_left.new_A.pop());
                  lattice[i + 1].set_value(to_left.new_B.pop());
            }
      }

      double energy() {
            return 0;
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

int main() {
      auto seed = 42;
      auto mps = MPS(3, 2, {1, 0, 0, 0, 0, -1, 2, 0, 0, 2, -1, 0, 0, 0, 0, 1}); // L = 3, D = 2, d=2
      auto engine = std::default_random_engine(seed);
      auto dist = std::uniform_real_distribution<double>(-1, 1);
      auto generator = [&dist, &engine]() { return dist(engine); };
      mps.set_random_state(generator);
      std::cout << mps << std::endl;
      mps.update(2, 0, 0.1);
      std::cout << mps << std::endl;
}
