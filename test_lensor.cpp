/* TAT/test_lensor.cpp
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

#include "TAT.hpp"

using namespace TAT::legs_name;
using Lensor=TAT::Lensor<TAT::Device::CPU, double>;
using Tensor=TAT::Tensor<TAT::Device::CPU, double>;

int main() {
  auto a = Lensor::make_lensor(Tensor({Up, Down}, {2, 3}));
  a->set_test();
  std::cout << a->value() << std::endl;
  auto b = a->transpose({Down, Up});
  std::cout << b->value() << std::endl;
  a->set_lensor(Tensor({Up, Down}, {4, 5}));
  a->set_test();
  std::cout << b->value() << std::endl;
  return 0;
} // main
