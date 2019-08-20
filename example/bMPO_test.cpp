/* example/bMPO_test.cpp
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

#include <random>

#define TAT_DEFAULT
#include <TAT.hpp>

#include "Heisenberg_PEPS_GO.dir/bMPO.hpp"

int main() {
      using namespace TAT::legs_name;

      std::ios_base::sync_with_stdio(false);

      auto engine = std::default_random_engine(0);
      auto dist = std::uniform_real_distribution<double>(-1, 1);
      auto generator = [&dist, &engine]() { return dist(engine); };

      int L = 4;
      int scan = 2;
      TAT::Size D = 4;
      TAT::Size Dc = 6;

      auto bMPO = bounded_matrix_product_operator<TAT::Node>({L, scan, Left, Right, Up, Down, true});
      auto former = std::vector<TAT::Node<double>>(L);
      auto current = std::vector<TAT::Node<double>>(L);
      auto initial = std::vector<TAT::Node<double>>(L);
      for (int i = 0; i < L; i++) {
            if (i == 0) {
                  former[i] = TAT::Node({Right, Down}, {D, D}).set(generator);
                  current[i] = TAT::Node({Right, Up, Down}, {D, D, D}).set(generator);
                  initial[i] = TAT::Node({Right, Down}, {Dc, D}).set(generator);
            } else if (i != L - 1) {
                  former[i] = TAT::Node({Left, Right, Down}, {D, D, D}).set(generator);
                  current[i] = TAT::Node({Left, Right, Up, Down}, {D, D, D, D}).set(generator);
                  initial[i] = TAT::Node({Left, Right, Down}, {Dc, Dc, D}).set(generator);
            } else {
                  former[i] = TAT::Node({Left, Down}, {D, D}).set(generator);
                  current[i] = TAT::Node({Left, Up, Down}, {D, D, D}).set(generator);
                  initial[i] = TAT::Node({Left, Down}, {Dc, D}).set(generator);
            }
      }
      auto res = bMPO(former, current, initial);
      for (int i = 0; i < L; i++) {
            std::cout << res[i] << std::endl;
      }
      return 0;
}