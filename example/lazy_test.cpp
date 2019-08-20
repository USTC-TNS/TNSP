/* example/lazy_test.cpp
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
#include <TAT.hpp>

TAT::Lazy<float> to_float(TAT::Lazy<int> a) {
      TAT::Lazy<float> res;
      res.set_func([=](int a) { return float(a); }, a);
      return res;
}
TAT::LazyNode<TAT::Node, float> to_float(TAT::LazyNode<TAT::Node, int> a) {
      return a.to<float>();
}

template<class Base>
using HighNode = TAT::LazyNode<TAT::Node, Base>;

using Lazy1 = TAT::LazyNode<TAT::Node, double>;
using Lazy2 = TAT::LazyNode<HighNode, double>;

int main() {
      // auto a = TAT::Lazy<int>(2);
      // auto b = TAT::Lazy<int>(3);
      auto a = TAT::LazyNode<TAT::Node, int>(2);
      auto b = TAT::LazyNode<TAT::Node, int>(3);
      auto c = a * b;
      auto d = c * 2;
      auto e = to_float(d) + 2.3;
      auto f = e + a;
      // std::cout << a << " " << e << std::endl;
      std::cout << a << "\n" << b << "\n" << c << "\n" << d << "\n" << e << "\n" << f << "\n\n";
      a.set_value(TAT::Node<int>(4));
      // a.set_value(4);
      std::cout << a << "\n" << b << "\n" << c << "\n" << d << "\n" << e << "\n" << f << "\n\n";
      b.set_value(TAT::Node<int>(10));
      // b.set_value(10);
      std::cout << a << "\n" << b << "\n" << c << "\n" << d << "\n" << e << "\n" << f << "\n\n";

      auto g = TAT::LazyNode<TAT::Node, double>();
      auto h = TAT::LazyNode<HighNode, double>();
      auto i = TAT::LazyNode<HighNode, double>(1);
      auto j = h + i;
      h.set_point_value(&g);
      auto k = j.pop();
      h.set_point_value(&k);
      auto l = j.pop();
      g.set_value(TAT::Node<double>(10));
      std::cout << l << std::endl;
      g.set_value(TAT::Node<double>(20));
      std::cout << l << std::endl;

      return 0;
}