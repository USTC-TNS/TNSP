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

#include <functional>
#include <random>

#include <TAT/TAT.hpp>

using Tensor = TAT::Tensor<float, TAT::NoSymmetry>;

auto get_name(std::string alphabet) {
   return [=](int i) { return alphabet + std::to_string(i); };
}
auto left = get_name("left");
auto right = get_name("right");
auto up = get_name("up");
auto down = get_name("down");

struct PBC {
   int L;
   unsigned int D;
   std::map<int, std::map<int, Tensor>> lattice;
   std::map<std::set<std::tuple<int, int>>, std::map<TAT::NoSymmetry, std::vector<float>>> environment;

   PBC(int L, unsigned int D, std::function<float()> generator) : L(L), D(D) {
      for (int i = 0; i < L; i++) {
         for (int j = 0; j < L; j++) {
            lattice[i][j] = Tensor({left(i), right(i), up(j), down(j)}, {D, D, D, D}).set(generator);
         }
      }
   }

   float get_exact() {
      auto result = Tensor(1);
      for (int i = 0; i < L; i++) {
         for (int j = 0; j < L; j++) {
            auto contract_names = std::set<std::tuple<TAT::Name, TAT::Name>>{{right(i), left(i)}, {down(j), up(j)}};
            if (j == L - 1) {
               contract_names.insert({left(i), right(i)});
            }
            if (i == L - 1) {
               contract_names.insert({up(j), down(j)});
            }
            result = result.contract(lattice[i][j], contract_names);
         }
      }
      std::cout << result << "\n";
      return 0;
   }
};

int main() {
   std::mt19937 engine(0);
   std::uniform_real_distribution<double> dis(-1, 1);
   auto generator = [&]() { return dis(engine); };
   auto pbc = PBC(3, 2, generator);
   pbc.get_exact();
}
