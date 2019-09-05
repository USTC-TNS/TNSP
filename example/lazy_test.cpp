/**
 * \file example/lazy_test.cpp
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

#define TAT_DEFAULT
#include <lazy_TAT.hpp>

#define REPORT_TYPE(x) std::clog << lazy::type_name<decltype(x)>() << "\n"
int main() {
      using namespace TAT;
      auto a = LazyNode<double>(2);
      auto b = LazyNode<double>(3);
      auto c = a * b;
      auto d = c * 2;
      auto e = d + 2.3;
      auto f = e + a;
      std::cout << a << "\n" << b << "\n" << c << "\n" << d << "\n" << e << "\n" << f << "\n\n";
      a == 4;
      std::cout << a << "\n" << b << "\n" << c << "\n" << d << "\n" << e << "\n" << f << "\n\n";
      b == 10;
      std::cout << a << "\n" << b << "\n" << c << "\n" << d << "\n" << e << "\n" << f << "\n\n";

      auto g = LazyNode<double>();
      auto h = LazyNode<double, 2>();
      auto i = LazyNode<double, 2>(1);
      auto j = h + i;
      h == g;
      auto k = j.pop();
      h == k;
      auto l = j.pop();
      g == 10;
      std::cout << l << std::endl;
      i.value() == g; // operatpr>>
      i.value().value();
      g == 20;
      std::cout << l << std::endl;

      auto m = LazyNode<double>({legs_name::Left, legs_name::Right}, {2, 3});
      m.value().set([]() {
            static int i = 0;
            return i++;
      });
      m.fresh();

      auto n = m.legs_rename({{legs_name::Right, legs_name::Left1}});
      std::cout << m << "\n";
      std::cout << n << "\n";
      return 0;
}
