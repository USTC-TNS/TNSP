/**
 * Copyright (C) 2019-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include "run_test.hpp"

void run_test() {
#define edge_seg(...) {__VA_ARGS__}
   std::cout << TAT::Tensor<
                      double,
                      TAT::Z2Symmetry>{{"Left", "Right", "Up"}, {edge_seg({1, 3}, {0, 1}), edge_seg({1, 1}, {0, 2}), edge_seg({1, 2}, {0, 3})}}
                      .zero()
             << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>(
                      {"Left", "Right", "Up"},
                      {edge_seg({-1, 3}, {0, 1}, {1, 2}), edge_seg({-1, 1}, {0, 2}, {1, 3}), edge_seg({-1, 2}, {0, 3}, {1, 1})})
                      .range(2)
             << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>(
                      {"Left", "Right", "Up"},
                      {std::vector<TAT::U1Symmetry>(), edge_seg({-1, 1}, {0, 2}, {1, 3}), edge_seg({-1, 2}, {0, 3}, {1, 1})})
                      .zero()
             << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>{{}, {}}.set([]() {
      return 123;
   }) << "\n";
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>(2333, {"i", "j"}, {-2, +2}) << "\n";
}