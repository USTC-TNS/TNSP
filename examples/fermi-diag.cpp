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

#include <fstream>
#include <iostream>

#include <TAT/TAT.hpp>

int main(int argc, char** argv) {
   auto number = TAT::Tensor<double, TAT::FermiSymmetry>({"in", "out"}, {{-1}, {1}}, true).test(1, 0) +
                 TAT::Tensor<double, TAT::FermiSymmetry>({"in", "out"}, {{-1, 0}, {0, 1}}, true).test(0, 0);

   auto more_and_less = TAT::Tensor<double, TAT::FermiSymmetry>({"control", "more", "less"}, {{-1}, {1}, {0}}, true).test(1, 0);
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
         this_hamiltonian = this_hamiltonian.contract(identity, {}).merge_edge({{"In", {"In", "in"}}, {"Out", {"out", "Out"}}});
         // TODO 有时候需要逆向merge, 现在只能通过两个conjugate来完成
         // 当block中只含一个元素时, 这实际上和反向merge不加符号时一样的
         // 反向不反向相差一个merge的符号, 而现在反向加个符号就相当于, 不反向不加符号
      }
      this_hamiltonian = this_hamiltonian.contract(h_t, {}).merge_edge({{"In", {"In", "in1", "in2"}}, {"Out", {"out2", "out1", "Out"}}});
      for (int current = i + 2; current < L; current++) {
         this_hamiltonian = this_hamiltonian.contract(identity, {}).merge_edge({{"In", {"In", "in"}}, {"Out", {"out", "Out"}}});
      }
      return this_hamiltonian;
   };
   auto get_hamiltonian_U = [&](int i) {
      // i and i+1
      auto this_hamiltonian = TAT::Tensor<double, TAT::FermiSymmetry>(1);
      for (int current = 0; current < i; current++) {
         this_hamiltonian = this_hamiltonian.contract(identity, {}).merge_edge({{"In", {"In", "in"}}, {"Out", {"out", "Out"}}});
      }
      this_hamiltonian = this_hamiltonian.contract(number, {}).merge_edge({{"In", {"In", "in"}}, {"Out", {"out", "Out"}}});
      this_hamiltonian = this_hamiltonian.contract(number, {}).merge_edge({{"In", {"In", "in"}}, {"Out", {"out", "Out"}}});
      for (int current = i + 2; current < L; current++) {
         this_hamiltonian = this_hamiltonian.contract(identity, {}).merge_edge({{"In", {"In", "in"}}, {"Out", {"out", "Out"}}});
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

   std::stringstream out;
   auto cout_buf = std::cout.rdbuf();
   if (argc != 1) {
      std::cout.rdbuf(out.rdbuf());
   }
   std::cout << (8 * V - T).transpose({"In", "Out"}).transform([](float i) { return i == -0 ? 0 : i; }) << "\n";
   if (argc != 1) {
      std::cout.rdbuf(cout_buf);
      std::ifstream fout(argv[1]);
      std::string sout((std::istreambuf_iterator<char>(fout)), std::istreambuf_iterator<char>());
      return sout != out.str();
   }
   return 0;
}