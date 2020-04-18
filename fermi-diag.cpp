/**
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

#include <TAT/TAT.hpp>

int main() {
   auto number = TAT::Tensor<double, TAT::FermiSymmetry>({"in", "out"}, {{-1}, {1}}, true).test(1, 0) +
                 TAT::Tensor<double, TAT::FermiSymmetry>({"in", "out"}, {{-1, 0}, {0, 1}}, true).test(0, 0);
   auto more_and_less = TAT::Tensor<double, TAT::FermiSymmetry>({"control", "more", "less"}, {{-1}, {1}, {0}}, true).test(1, 0) +
                        TAT::Tensor<double, TAT::FermiSymmetry>({"control", "more", "less"}, {{-1, 0}, {0, 1}, {0, 0}}, true).test(0, 0);
   auto identity = TAT::Tensor<double, TAT::FermiSymmetry>({"in", "out"}, {{-1, 0}, {0, 1}}, true).test(1, 0);
   auto more_1 = identity.edge_rename({{"out", "out2"}, {"in", "in2"}}).contract(more_and_less, {}).edge_rename({{"more", "out1"}, {"less", "in1"}});
   auto more_2 = identity.edge_rename({{"out", "out1"}, {"in", "in1"}}).contract(more_and_less, {}).edge_rename({{"more", "out2"}, {"less", "in2"}});
   auto less_1 = identity.edge_rename({{"out", "out2"}, {"in", "in2"}})
                       .contract(more_and_less.conjugate(), {})
                       .edge_rename({{"more", "in1"}, {"less", "out1"}});
   auto less_2 = identity.edge_rename({{"out", "out1"}, {"in", "in1"}})
                       .contract(more_and_less.conjugate(), {})
                       .edge_rename({{"more", "in2"}, {"less", "out2"}});
   auto h_12 = less_1.contract(more_2, {{"control", "control"}, {"out1", "in1"}, {"out2", "in2"}}).transpose({"out1", "out2", "in1", "in2"});
   auto h_21 = less_2.contract(more_1, {{"control", "control"}, {"out1", "in1"}, {"out2", "in2"}}).transpose({"out1", "out2", "in1", "in2"});
   auto h_t = h_21 + h_12;

   int L = 6;
   auto get_hamiltonian_t = [&](int i) {
      // i and i+1
      auto this_hamiltonian = TAT::Tensor<double, TAT::FermiSymmetry>(1);
      for (int current = 0; current < i; current++) {
         this_hamiltonian = this_hamiltonian.contract(identity, {}).merge_edge({{"In", {"in", "In"}}, {"Out", {"Out", "out"}}});
      }
      this_hamiltonian = this_hamiltonian.contract(h_t, {}).merge_edge({{"In", {"in2", "in1", "In"}}, {"Out", {"Out", "out1", "out2"}}});
      for (int current = i + 2; current < L; current++) {
         this_hamiltonian = this_hamiltonian.contract(identity, {}).merge_edge({{"In", {"in", "In"}}, {"Out", {"Out", "out"}}});
      }
      return this_hamiltonian;
   };
   auto get_hamiltonian_U = [&](int i) {
      // i and i+1
      auto this_hamiltonian = TAT::Tensor<double, TAT::FermiSymmetry>(1);
      for (int current = 0; current < i; current++) {
         this_hamiltonian = this_hamiltonian.contract(identity, {}).merge_edge({{"In", {"in", "In"}}, {"Out", {"Out", "out"}}});
      }
      this_hamiltonian = this_hamiltonian.contract(number, {}).merge_edge({{"In", {"in", "In"}}, {"Out", {"Out", "out"}}});
      // TODO: 为什么这里的merge不应该加符号？ 难道哈密顿量就是这样的规则么？
      // 可能事因为收缩方向反了的原因， 但是为什么收缩方向反了呢， 这可能是一个bug
      // 可用L=2, i.e. n_1 * n_2这个哈密顿量做研究
      this_hamiltonian = this_hamiltonian.contract(number, {}).merge_edge({{"In", {"in", "In"}}, {"Out", {"Out", "out"}}});
      for (int current = i + 2; current < L; current++) {
         this_hamiltonian = this_hamiltonian.contract(identity, {}).merge_edge({{"In", {"in", "In"}}, {"Out", {"Out", "out"}}});
      }
      return this_hamiltonian;
   };
   auto T = get_hamiltonian_t(0);
   for (auto i = 1; i < L - 1; i++) {
      T = T + get_hamiltonian_t(i);
   }
   auto V = get_hamiltonian_U(0);
   for (auto i = 1; i < L - 1; i++) {
      V = V + get_hamiltonian_U(i);
   }
   std::cout << 8 * V - T << "\n";
   return 0;
}
