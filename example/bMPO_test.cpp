/**
 * \file example/bMPO_test.cpp
 *
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

#include "PEPS_GO.dir/bMPO.hpp"

int main(int argc, char** argv) {
      using namespace TAT::legs_name;

      std::ios_base::sync_with_stdio(false);

      auto engine = std::default_random_engine(0);
      auto dist = std::uniform_real_distribution<double>(-1, 1);
      auto generator = [&dist, &engine]() { return dist(engine); };

      int L = 4;
      int scan = 2;
      if (argc > 1) {
            std::istringstream(argv[1]) >> scan;
            std::cout << "set scan time = " << scan << "\n";
      }
      TAT::Size D = 4;
      TAT::Size Dc = 6;
      if (argc > 2) {
            std::istringstream(argv[2]) >> Dc;
            std::cout << "set Dc = " << Dc << "\n";
      }

      auto bMPO = bounded_matrix_product_operator<0>({L, scan, Left, Right, Up, Down, true});
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

      // check origin
      auto dbllay = std::vector<TAT::Node<double>>(L);
      for (int i = 0; i < L; i++) {
            dbllay[i] = TAT::Node<double>::contract(
                  former[i].legs_rename({{Left, Left1}, {Right, Right1}}),
                  current[i].legs_rename({{Left, Left2}, {Right, Right2}}),
                  {Down},
                  {Up});
      }
      auto origin = TAT::Node<double>(1);
      for (int i = 0; i < L; i++) {
            origin = TAT::Node<double>::contract(
                  origin,
                  dbllay[i].legs_rename({{Down, TAT::Legs("Down_in_bMPO_test_" + std::to_string(i))}}),
                  {Right1, Right2},
                  {Left1, Left2});
      }
      std::cout << origin << "\n";
      // check result
      auto result = TAT::Node<double>(1);
      for (int i = 0; i < L; i++) {
            result = TAT::Node<double>::contract(
                  result,
                  res[i].legs_rename({{Down, TAT::Legs("Down_in_bMPO_test_" + std::to_string(i))}}),
                  {Right},
                  {Left});
      }
      std::cout << result << "\n";
      std::cout << (origin - result).norm<-1>() / origin.norm<-1>() << "\n";
      return 0;
}