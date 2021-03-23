/**
 * Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

using Tensor = typename TAT::Tensor<float, TAT::NoSymmetry>;

void run_test() {
   auto a = Tensor({"A", "B"}, {5, 10}).range();
   std::cout << a << '\n';
   {
      auto [q, r] = a.qr('r', {"A"}, "newQ", "newR");
      std::cout << q << '\n'; // "B" "newQ"
      std::cout << r << '\n'; // "A" "newR"
      std::cout << (q.contract(r, {{"newQ", "newR"}}) - a).norm<-1>() << '\n';
   }
   {
      auto [q, r] = a.qr('r', {"B"}, "newQ", "newR");
      std::cout << q << '\n';
      std::cout << r << '\n';
      std::cout << (q.contract(r, {{"newQ", "newR"}}) - a).norm<-1>() << '\n';
   }
   auto b = Tensor({"A", "B"}, {10, 5}).range();
   std::cout << b << '\n';
   {
      auto [q, r] = b.qr('r', {"A"}, "newQ", "newR");
      std::cout << q << '\n';
      std::cout << r << '\n';
      std::cout << (q.contract(r, {{"newQ", "newR"}}) - b).norm<-1>() << '\n';
   }
   {
      auto [q, r] = b.qr('r', {"B"}, "newQ", "newR");
      std::cout << q << '\n';
      std::cout << r << '\n';
      std::cout << (q.contract(r, {{"newQ", "newR"}}) - b).norm<-1>() << '\n';
   }
}
