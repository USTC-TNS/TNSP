/* example/test_new_tensor.cpp
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

#define TAT_USE_CPU

// SVD
#if (!defined TAT_USE_GESDD && !defined TAT_USE_GESVD && !defined TAT_USE_GESVDX)
#define TAT_USE_GESVD
#endif

// QR
#if (!defined TAT_USE_GEQRF && !defined TAT_USE_GEQP3)
#define TAT_USE_GEQRF
#endif

#include <TAT.hpp>

using namespace TAT::legs_name;
using Node=TAT::Node<TAT::Device::CPU, double>;

int main() {
  auto a = Node::make({Up, Down}, {2, 3})->set_test();
  std::cout << a->value() << std::endl;
  auto b = a->transpose({Down, Up});
  std::cout << b->value() << std::endl;
  srand(0);
  a->set_random(rand);
  std::cout << a->value() << std::endl;
  std::cout << b->value() << std::endl;
  //b->legs_rename({{Up, Right}});
  //b->normalize<-1>();
  //std::cout << b->value() << std::endl;
  //auto c = TAT::Lensor<TAT::Device::CPU, int>::make()->set(b->value().to<int>());
  //std::cout << c->value() << std::endl;
  return 0;
} // main
