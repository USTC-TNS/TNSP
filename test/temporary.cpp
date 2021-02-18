/**
 * Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <TAT/TAT.hpp>

template<typename T>
void print(const T& a) {
   for (const auto& i : a) {
      std::cout << i << ' ';
   }
   std::cout << '\n';
}

int main() {
   TAT::Tensor<double, TAT::FermiSymmetry> b({"i", "j"}, {{{0, 2}, {1, 2}}, {{-1, 2}, {0, 2}}});
   b.test(233);
   std::cout << b << "\n";
   print(b.core->storage);
}
