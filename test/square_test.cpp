/**
 * Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

#include <square/square.hpp>

int main() {
   {
      auto lattice = square::ExactLattice<double>(4, 4, 2);
      lattice.set_all_horizontal_bond(square::Common<double>::SS());
      lattice.set_all_vertical_bond(square::Common<double>::SS());
      lattice.update(1000);
      std::cout << lattice.observe({{1, 1}, {1, 2}}, *square::Common<double>::SS(), true) << "\n";
      std::cout << lattice.observe_energy() << "\n";
   }
   {
      auto lattice = square::SimpleUpdateLattice<double>(4, 4, 4, 2);
      lattice.set_all_horizontal_bond(square::Common<double>::SS());
      lattice.set_all_vertical_bond(square::Common<double>::SS());
      lattice.update(1000, 0.1, 0);
      lattice.initialize_auxilaries(8);
   }
}
