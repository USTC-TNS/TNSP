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
   std::cout << TAT::Tensor<double, TAT::U1Symmetry>{123} << "\n";
   std::cout << double(TAT::Tensor<double, TAT::U1Symmetry>{123}) << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.test(2).to<double>() << "\n";
   std::cout << TAT::Tensor<double, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.test(2).to<std::complex<double>>() << "\n";
   std::cout << TAT::Tensor<std::complex<double>, TAT::NoSymmetry>{{"Left", "Right"}, {3, 4}}.test(2).to<double>() << "\n";
   std::cout << TAT::Tensor<
                      double,
                      TAT::U1Symmetry>{{"Left", "Right", "Up"}, {{{-1, 3}, {0, 1}, {1, 2}}, {{-1, 1}, {0, 2}, {1, 3}}, {{-1, 2}, {0, 3}, {1, 1}}}}
                      .test(2)
                      .to<std::complex<double>>()
             << "\n";
}
