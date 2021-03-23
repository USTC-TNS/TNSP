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

using namespace TAT;

void run_test() {
   auto a = Tensor<>({"A", "B", "C"}, {2, 3, 5}).range();
   auto b = Tensor<>({"A", "B", "D"}, {2, 3, 7}).range();
   auto c = Tensor<>::contract(a, b, {{"B", "B"}});
   auto a0 = a.shrink({{"A", 0}});
   auto a1 = a.shrink({{"A", 1}});
   auto b0 = b.shrink({{"A", 0}});
   auto b1 = b.shrink({{"A", 1}});
   auto c0 = c.shrink({{"A", 0}});
   auto c1 = c.shrink({{"A", 1}});
   std::cout << a0.contract(b0, {{"B", "B"}}) - c0 << "\n";
   std::cout << a1.contract(b1, {{"B", "B"}}) - c1 << "\n";
}
