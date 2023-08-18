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
   auto t = TAT::Tensor<double, TAT::Z2Symmetry>{{"Left", "Right", "Phy"}, {{{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}, {{0, 2}, {1, 2}}}};
   t.range();
   std::cout << t + 1.0 << "\n";
   std::cout << 1.0 / t << "\n";

   auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range();
   auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.range(0, 0.1);
   std::cout << a + b << "\n";
   std::cout << a - b << "\n";
   std::cout << a * b << "\n";
   std::cout << a / b << "\n";
   std::cout << a + b.transpose({"Right", "Left"}) << "\n";

   auto c = TAT::Tensor<double, TAT::U1Symmetry>{{"L", "R"}, {{0, 1}, {-1, 0}}}.range(10);
   auto d = TAT::Tensor<double, TAT::U1Symmetry>{{"L", "R"}, {{0, 1}, {-1, 0}}}.range(100);
   std::cout << c << "\n";
   std::cout << d << "\n";
   std::cout << (c += d) << "\n";
}
