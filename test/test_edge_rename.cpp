/**
 * Copyright (C) 2019-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
   auto t1 = TAT::Tensor<double, TAT::Z2Symmetry>{{"Left", "Right", "Phy"}, {{{0, 1}, {1, 2}}, {{0, 3}, {1, 4}}, {{0, 5}, {1, 6}}}};
   auto t2 = t1.edge_rename({{"Left", "Up"}});
   t1.test();
   std::cout << t1 << "\n";
   // std::cout << t2 << "\n";
}
