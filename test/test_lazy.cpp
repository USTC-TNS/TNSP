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

#include <iostream>
#include <lazy.hpp>

#include "run_test.hpp"

void run_test() {
   {
      auto v = 1;
      auto a = lazy::Root(std::cref(v));
      auto b = lazy::Root(2);
      std::cout << a() << "\n";
      std::cout << b() << "\n";
      auto c = lazy::Node(
            [](int a, int b) {
               return a + b;
            },
            a,
            b);
      auto d = lazy::Node(
            [](int c, int a) {
               return c * a;
            },
            c,
            a);
      std::cout << d() << "\n";
      a.reset(233);
      std::cout << d() << "\n";
      b.reset(666);
      std::cout << d() << "\n";
   }
   {
      auto a = lazy::Root(1);
      auto b = lazy::Root(10);
      auto c = lazy::Node(
            [](auto i, auto j) {
               return i + j;
            },
            a,
            b);
      std::cout << c() << "\n";
      b.reset(100);
      std::cout << c() << "\n";
      auto copy = lazy::Copy();
      auto aa = copy(a);
      auto bb = copy(b);
      auto cc = copy(c);
      aa.reset(1000);
      std::cout << cc() << "\n";
      std::cout << c() << "\n";
   }
}
